from typing import List, Tuple

import numpy as np

from .utils import get_spectral_coords

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


class StandardMultisliceSolver:
    """
    Standard Split-Step Beam Method (SSBM) Solver for 3D Multislice.
    Processes a raster scan by propagating a probe through a large volume.
    """

    def __init__(
        self,
        calc_shape: Tuple[int, int, int],
        dx: float,
        dz: float,
        wavelength: float,
        n_mean: complex,
        gpu_id: int = 0,
    ):
        """
        Initialize the solver with physics constants and precompute the propagator.

        Args:
            calc_shape: (ny, nx, nz_steps) - Note: ny (rows) first, nx (cols) second.
            dx: Pixel size in x/y (assumes square pixels).
            dz: Step size in z.
            wavelength: Illumination wavelength.
            n_mean: Average refractive index (background).
            gpu_id: GPU device ID to use.
        """
        self.ny, self.nx, self.nz_steps = calc_shape
        self.dx = dx
        self.dz = dz
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        self.n_mean = n_mean

        self.complex_t = np.complex128
        self.real_t = np.float64

        self.use_gpu = HAS_GPU and gpu_id is not None
        self.xp = cp if self.use_gpu else np
        self.gpu_id = gpu_id

        # Precompute the Vacuum Propagator
        with self._device_context():
            # Spectral coordinates
            kx = get_spectral_coords(self.nx, self.dx, "FFT")
            ky = get_spectral_coords(self.ny, self.dx, "FFT")

            kx = self.xp.asarray(kx, dtype=self.real_t)[self.xp.newaxis, :]  # (1, Nx)
            ky = self.xp.asarray(ky, dtype=self.real_t)[:, self.xp.newaxis]  # (Ny, 1)

            # Transverse wavevector squared
            K_sq = kx**2 + ky**2

            # === SHIFTED HELMHOLTZ PROPAGATOR ===
            # Absorbs diffraction and mean phase to minimize parallel error
            # k_z = sqrt(k0^2 - k_trans^2)
            inside = self.real_t(self.k0) ** 2 - K_sq
            sqrt_term = self.xp.sqrt(self.xp.clip(inside, 0.0, None))

            # lambda_vac: Phase accumulation of vacuum per dz, relative to k0
            lambda_vac = 1j * (sqrt_term - self.real_t(self.k0))

            # phi_mean: Phase accumulation of the mean material per dz
            phi_mean = 1j * self.real_t(self.k0) * (self.n_mean - 1.0)

            # Linear propagator L-tilde = exp((lambda_vac + phi_mean) * dz)
            self.P = self.xp.exp((lambda_vac + phi_mean) * self.real_t(self.dz)).astype(
                self.complex_t
            )

    def _device_context(self):
        """Context manager for selecting the correct GPU device."""
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def precompute_probe(self, psi: np.ndarray):
        """
        Loads the probe wavefield onto the compute device.
        psi shape should be (ny, nx).
        """
        with self._device_context():
            self.psi_0 = self.xp.asarray(psi, dtype=self.complex_t)

    def run_scan(
        self,
        large_map: np.ndarray,
        positions: List[Tuple[int, int]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Executes the multislice propagation for a list of scan positions.

        Args:
            large_map: 3D Refractive Index Map (Global_Ny, Global_Nx, Nz).
            positions: List of (row, col) centers for the scan.
            batch_size: Number of positions to process in parallel.

        Returns:
            Diffracted wavefields at the exit surface. Shape: (N_pos, Ny, Nx).
        """
        if not hasattr(self, "psi_0"):
            raise AttributeError(
                "Probe not found. Call precompute_probe() before run_scan()."
            )

        num_pos = len(positions)
        results = np.zeros((num_pos, self.ny, self.nx), dtype=self.complex_t)

        # Window half-sizes (assuming positions are centers)
        half_ny = self.ny // 2
        half_nx = self.nx // 2

        with self._device_context():
            # Move the entire map to GPU once if it fits (Fastest)
            # If large_map is too large for VRAM, this line needs to be inside the batch loop
            # with slicing happening on CPU. Assuming it fits for "Update this to work".
            large_map_device = self.xp.asarray(large_map, dtype=self.complex_t)

            # Pre-calculate window grids for vectorized gathering
            # Shapes: y_grid (Ny, 1), x_grid (1, Nx)
            y_window = self.xp.arange(-half_ny, half_ny)[:, self.xp.newaxis]
            x_window = self.xp.arange(-half_nx, half_nx)[self.xp.newaxis, :]

            for b in range(0, num_pos, batch_size):
                b_end = min(b + batch_size, num_pos)
                curr_batch_size = b_end - b

                # 1. Prepare Batch Indices
                batch_positions = positions[b:b_end]

                # Convert list of tuples to array: (Batch, 2) -> col 0 is y(row), col 1 is x(col)
                pos_arr = self.xp.asarray(batch_positions)

                # Broadcast positions to create gather maps
                # centers_y: (Batch, 1, 1)
                centers_y = pos_arr[:, 0].reshape(curr_batch_size, 1, 1)
                centers_x = pos_arr[:, 1].reshape(curr_batch_size, 1, 1)

                # Gather indices: (Batch, Ny, Nx)
                gather_y = centers_y + y_window
                gather_x = centers_x + x_window

                # 2. Initialize Wavefronts
                # Tile the probe: (Batch, Ny, Nx)
                psi = self.xp.tile(self.psi_0, (curr_batch_size, 1, 1))

                # 3. Propagate through Z
                for z in range(self.nz_steps):
                    # --- A. Gather Refractive Index for Batch at depth z ---
                    # Using advanced indexing to pull (Batch, Ny, Nx) from (Global_Ny, Global_Nx, Nz)
                    # We access slice z directly from large_map
                    delta_n_batch = (
                        large_map_device[gather_y, gather_x, z] - self.n_mean
                    )

                    # --- B. Symmetrized Split-Step (Kick-Drift-Kick) ---

                    # 1. Entrance Refraction (Half-Kick)
                    # Apply exp(i * k0 * delta_n * dz / 2)
                    phase_kick = self.xp.exp(0.5j * self.k0 * delta_n_batch * self.dz)
                    psi *= phase_kick

                    # 2. Diffraction (Drift)
                    # FFT -> Apply Propagator P -> IFFT
                    psi_k = self.xp.fft.fft2(psi, axes=(1, 2))
                    psi_k *= self.P
                    psi = self.xp.fft.ifft2(psi_k, axes=(1, 2))

                    # 3. Exit Refraction (Half-Kick)
                    # Apply same phase kick again
                    psi *= phase_kick

                # 4. Store Results
                # Transfer batch back to CPU results array
                results[b:b_end] = self.xp.asnumpy(psi)

                # Cleanup GPU memory for this batch
                del psi, delta_n_batch, phase_kick, psi_k
                if self.use_gpu:
                    self.xp.get_default_memory_pool().free_all_blocks()

        return results
