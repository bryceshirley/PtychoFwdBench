import logging
from typing import Optional

import numpy as np

# Adjust this import to match your project structure
from ptycho_fwd_bench.dim_2.solvers.utils import get_spectral_coords

from .base import OpticalWaveSolver3D

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as spla

    HAS_GPU = True
except ImportError:
    cp = None
    import scipy.sparse.linalg as spla

    HAS_GPU = False

logger = logging.getLogger(__name__)


class ParallelMultisliceSolverASM3D(OpticalWaveSolver3D):
    """
    Parallel Multislice ASM Solver for 3D volumes.
    Uses CuPy for GPU acceleration with zero-allocation hot loops.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        dz: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        store_beam: bool = False,
        n_iter: int = 1,
        solver_type: str = "Richardson",
        use_gpu: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.n_iter = n_iter
        self.solver_type = solver_type.upper()

        # GPU Setup
        self.use_gpu = use_gpu and HAS_GPU
        self.gpu_id = gpu_id
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            with cp.cuda.Device(self.gpu_id):
                self.n_map = cp.asarray(self.n_map)
        else:
            if use_gpu and not HAS_GPU:
                logger.warning("CuPy not found. Falling back to CPU (NumPy).")
            else:
                logger.info("ParallelMultisliceSolverASM3D using CPU (NumPy).")

        self.ny, self.nx, self.nz_steps = self.n_map.shape
        self.n_mean = self.xp.mean(self.n_map)

        # Build operators in correct device context
        with self._device_context():
            self._setup_operators()

    def _device_context(self):
        """Context manager to ensure operations happen on the correct GPU."""
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def _get_3d_phase_shift(self) -> np.ndarray:
        """Calculates the 2D transverse phase shift and expands to 3D."""
        # 1. Get 1D spectral coordinates on CPU
        ky_cpu = get_spectral_coords(self.ny, self.dx, "FFT")
        kx_cpu = get_spectral_coords(self.nx, self.dx, "FFT")

        # 2. Move to GPU
        ky = self.xp.asarray(ky_cpu)
        kx = self.xp.asarray(kx_cpu)

        # 3. Create 2D Meshgrid for K^2
        KY, KX = self.xp.meshgrid(ky, kx, indexing="ij")
        K_sq = KX**2 + KY**2

        # 4. Transverse Propagator Phase (Vacuum + Mean phase)
        inside = self.k0**2 - K_sq
        sqrt_term = self.xp.sqrt(self.xp.clip(inside, 0.0, None))

        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # 5. Calculate phase shift and expand to (Ny, Nx, 1) for Z-broadcasting
        phase_shift = (lambda_vac + phi_mean) * self.dz
        return phase_shift[:, :, self.xp.newaxis]

    def _setup_operators(self):
        """Precomputes 3D kernels and static allocations."""
        phase_shift = self._get_3d_phase_shift()

        # Precompute the massive 3D exponential grids (Axis 2 is Z)
        z_idx = self.xp.arange(self.nz_steps)[self.xp.newaxis, self.xp.newaxis, :]

        # Forward and Inverse Propagation Kernels
        self._kernel_inv = self.xp.exp(-phase_shift * z_idx)
        self._kernel_fwd = self.xp.exp(phase_shift * (z_idx + 1))

        # Object transmission setup
        phase = 1j * self.k0 * (self.n_map - self.n_mean) * (self.dz / 2)
        self._half_obj = self.xp.exp(phase)

        # Object Scattering Correction E1
        inv_half_obj = self.xp.exp(-phase)
        self._E1_coeff = self.xp.zeros_like(self._half_obj)
        self._E1_coeff[:, :, 1:] = (
            inv_half_obj[:, :, 1:] * inv_half_obj[:, :, :-1]
        ) - 1.0

        # Pre-allocate zero-allocation buffer for Hot Loop
        self._E1_buffer = self.xp.zeros(
            (self.ny, self.nx, self.nz_steps), dtype=self.xp.complex128
        )

    def _apply_E1(self, u_sol: np.ndarray) -> np.ndarray:
        """Apply E1 error via in-place memory buffer (Nilpotent)."""
        self._E1_buffer[:, :, 0] = 0.0

        # In-place shifted multiplication to prevent memory allocation
        self.xp.multiply(
            u_sol[:, :, :-1], self._E1_coeff[:, :, 1:], out=self._E1_buffer[:, :, 1:]
        )
        return self._E1_buffer

    def _apply_M_inv(self, u_sol: np.ndarray) -> np.ndarray:
        """Applies M^-1 using strictly in-place GPU operations."""
        u_sol *= self._half_obj

        # 2D batched FFT along Y and X
        self.xp.fft.fft2(u_sol, axes=(0, 1), out=u_sol)

        u_sol = self._solve_bidiag(u_sol)

        # 2D batched IFFT along Y and X
        self.xp.fft.ifft2(u_sol, axes=(0, 1), out=u_sol)

        u_sol *= self._half_obj
        return u_sol

    def _solve_bidiag(self, u_sol):
        """Optimized recursive solver along the Z-axis (axis=2)."""
        u_sol *= self._kernel_inv

        # Cumulative sum along Z
        self.xp.cumsum(u_sol, axis=2, out=u_sol)

        u_sol *= self._kernel_fwd
        return u_sol

    def _apply_M_inv_source(self, psi_0: np.ndarray) -> np.ndarray:
        """Optimized M^-1 application for the initial source."""
        # Inject source into front face
        psi_in = self._half_obj[:, :, 0] * psi_0

        # 2D FFT and expand to (Ny, Nx, 1)
        psi_k = self.xp.fft.fft2(psi_in, axes=(0, 1))[:, :, self.xp.newaxis]

        # Apply 3D kernel
        u_sol = psi_k * self._kernel_fwd

        # 2D IFFT
        self.xp.fft.ifft2(u_sol, axes=(0, 1), out=u_sol)

        u_sol *= self._half_obj
        return u_sol

    def _solve_pass(
        self, psi_0: np.ndarray, u_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Runs the 3D solver pass."""
        b = self._apply_M_inv_source(psi_0)

        # Must copy 'b' because in-place subtraction alters u_sol
        u_sol = b.copy() if u_guess is None else u_guess.copy()

        if self.n_iter > 0:
            if self.solver_type == "RICHARDSON":
                for _ in range(self.n_iter):
                    correction = self._apply_M_inv(self._apply_E1(u_sol))
                    # In-place subtraction prevents memory bloat
                    self.xp.subtract(b, correction, out=u_sol)

            elif self.solver_type == "GMRES":

                def matvec(v):
                    v_reshaped = v.reshape(self.ny, self.nx, self.nz_steps)
                    return (
                        v_reshaped + self._apply_M_inv(self._apply_E1(v_reshaped))
                    ).ravel()

                total_size = self.ny * self.nx * self.nz_steps
                A = spla.LinearOperator(
                    (total_size, total_size),
                    matvec=matvec,
                    dtype=self.xp.complex128,
                )

                u_sol_flat, _ = spla.gmres(
                    A,
                    b.ravel(),
                    x0=u_sol.ravel(),
                    rtol=1e-10,
                    maxiter=self.n_iter,
                )
                u_sol = u_sol_flat.reshape(self.ny, self.nx, self.nz_steps)
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")

        return u_sol

    def _propagate_and_store(
        self, psi_0: np.ndarray, u_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Wrapper to run the solver and manage host/device memory transfers."""
        with self._device_context():
            u_sol = self._solve_pass(psi_0, u_guess=u_guess)

            if self.store_beam:
                self.beam_history = cp.asnumpy(u_sol) if self.use_gpu else u_sol

            self.psi_final = (
                cp.asnumpy(u_sol[:, :, -1]) if self.use_gpu else u_sol[:, :, -1]
            )

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
    ):
        """Runs the 3D solver."""
        psi_0 = self.initialize_wavefront(psi_init)

        with self._device_context():
            psi_0_device = self.xp.asarray(psi_0)

        self._propagate_and_store(psi_0_device)

        return self
