import logging
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

# Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpx_linalg

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

# Updated import based on your requirements
from ptycho_fwd_bench.dim_2.utils.utils import get_spectral_coords
from ptycho_fwd_bench.dim_3.generators_3d import get_2d_airy_probe


class ParallelMultisliceSolverBatched_3D:
    """
    Batched Parallel-in-Time Solver for 3D Ptychography.
    Solves for B probes simultaneously over a 3D volume (y, x, z).
    Supports GPU acceleration via CuPy.
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        wavelength: float,
        dz: float,
        nx: int,  # Window Width
        ny: int,  # Window Height
        nz_steps: int,  # Depth Steps
        probe_dia: float,
        probe_focus: float = 0,
        alpha: float = 1e-6,
        solver_type: str = "richardson",
        n_iter: int = 1,
        use_gpu: bool = True,
        **kwargs,
    ):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.wavelength = wavelength
        self.nx = nx
        self.ny = ny
        self.nz_steps = nz_steps
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.solver_type = solver_type
        self.n_iter = n_iter

        # Physics constants
        self.k0 = 2 * np.pi / wavelength
        self.k0sq = self.k0**2
        self.alpha = float(alpha)

        # GPU / Backend Setup
        if use_gpu and HAS_GPU:
            self.use_gpu = True
            self.xp = cp
            logging.info("ParallelMultisliceSolver: Using GPU (CuPy).")
        else:
            self.use_gpu = False
            self.xp = np
            if use_gpu and not HAS_GPU:
                logging.warning("CuPy not found. Falling back to CPU (NumPy).")
            else:
                logging.info("ParallelMultisliceSolver: Using CPU (NumPy).")

        logging.info("Initializing 3D spectral physics engine...")

    def setup_solver(self, n_map: np.ndarray):
        """
        Sets up the global refractive index map.
        n_map: 3D array (Global_Ny, Global_Nx, Nz)
        """
        # Ensure input is on the correct device
        if self.use_gpu and not isinstance(n_map, self.xp.ndarray):
            n_map = self.xp.asarray(n_map)

        if n_map.ndim != 3:
            raise ValueError(f"n_map must be 3D (Ny, Nx, Nz). Got shape {n_map.shape}")

        self.n_mean = self.xp.mean(n_map)
        self.delta_n_global = n_map - self.n_mean

        # Kernel is computed based on window size (ny, nx)
        # Shape: (1, Ny, Nx, Nz)
        self.K_4d = self._get_4d_kernel()

    def _get_4d_kernel(self) -> np.ndarray:
        """
        Computes the spectral propagator K for 3D propagation.
        Shape: (Ny, Nx, Nz) broadcastable to (Batch, Ny, Nx, Nz).
        """
        # 1. Get Spectral Coordinates (on CPU first to use utils, then move)
        kx_cpu = get_spectral_coords(self.nx, self.dx, "FFT")
        ky_cpu = get_spectral_coords(self.ny, self.dy, "FFT")

        kx = self.xp.asarray(kx_cpu)[self.xp.newaxis, :]  # (1, Nx)
        ky = self.xp.asarray(ky_cpu)[:, self.xp.newaxis]  # (Ny, 1)

        # 2. Transverse Propagator P (Vacuum)
        # k_trans^2 = kx^2 + ky^2
        k_trans_sq = kx**2 + ky**2

        inside = self.k0sq - k_trans_sq
        sqrt_term = self.xp.sqrt(self.xp.clip(inside, 0.0, None))

        # lambda_vac: vacuum phase advance per dz
        lambda_vac = 1j * (sqrt_term - self.k0)

        # phi_mean: mean potential phase advance per dz
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # P_shifted = exp(Lambda_bar * dz)
        # Shape: (Ny, Nx, 1)
        self._P_shifted = self.xp.exp((lambda_vac + phi_mean) * self.dz)[
            :, :, self.xp.newaxis
        ]

        # 3. Longitudinal Shifts (Twist)
        L = self.nz_steps
        kz = self.xp.arange(L)

        # lam_alpha = alpha^(1/L) * exp(-i 2pi k/L)
        # Shape: (1, 1, Nz)
        lam_alpha = (self.alpha ** (1 / L) * self.xp.exp(-2j * self.xp.pi * kz / L))[
            self.xp.newaxis, self.xp.newaxis, :
        ]

        # 4. Kernel: P / (1 - lambda * P)
        # Broadcasting: (Ny, Nx, 1) and (1, 1, Nz) -> (Ny, Nx, Nz)
        denom = 1.0 - (lam_alpha * self._P_shifted)

        # Expand dims to (1, Ny, Nx, Nz) for batch broadcasting
        return (self._P_shifted / (denom + 1e-15))[self.xp.newaxis, ...]

    def _setup_batch_operators(self, scan_indices: np.ndarray):
        """
        Prepares the local object slices and twist operators for the batch.
        scan_indices: (Batch, 2) integers indicating [y_start, x_start].
        """
        if self.use_gpu and not isinstance(scan_indices, self.xp.ndarray):
            scan_indices = self.xp.asarray(scan_indices)

        L = self.nz_steps
        Ny, Nx = self.ny, self.nx

        # 1. Gather Local Object (Raster Crop)
        batch_y = scan_indices[:, 0][:, self.xp.newaxis, self.xp.newaxis]
        batch_x = scan_indices[:, 1][:, self.xp.newaxis, self.xp.newaxis]

        y_grid = self.xp.arange(Ny)[self.xp.newaxis, :, self.xp.newaxis]
        x_grid = self.xp.arange(Nx)[self.xp.newaxis, self.xp.newaxis, :]

        # Global Indices: (Batch, Ny, Nx)
        gather_y = batch_y + y_grid
        gather_x = batch_x + x_grid

        # Fetch local delta_n: (Batch, Ny, Nx, Nz)
        local_delta_n = self.delta_n_global[gather_y, gather_x, :]

        # 2. Physics Terms (Phase Shift)
        N_phase = (1j * self.k0 * local_delta_n * self.dz / 2).astype(np.complex128)

        # 3. Twist Terms (Gamma)
        z_idx = self.xp.arange(L)
        # Shapes: (1, 1, 1, Nz)
        gamma = (self.alpha ** (-z_idx / L))[
            self.xp.newaxis, self.xp.newaxis, self.xp.newaxis, :
        ]
        gamma_inv = (self.alpha ** (z_idx / L))[
            self.xp.newaxis, self.xp.newaxis, self.xp.newaxis, :
        ]

        # 4. Operator Construction
        self._half_obj = self.xp.exp(N_phase)
        self._inv_half_obj = self.xp.exp(-N_phase)

        # Pre/Post Multipliers for M^-1
        self._pre_mul = self._half_obj * gamma_inv
        self._post_mul = self._half_obj * gamma

        # Error Terms (Using exact form: InvObj * shift(InvObj) - I)
        shifted_inv = self.xp.roll(self._inv_half_obj, shift=1, axis=3)
        self._E1_coeff = (self._inv_half_obj * shifted_inv) - 1.0

        # E2: Corner (Wrap-around)
        self._E2_coeff = (
            self.alpha * self._inv_half_obj[..., 0] * self._inv_half_obj[..., -1]
        )

    def _apply_M_inv(self, rhs_batch: np.ndarray) -> np.ndarray:
        """Forward Preconditioner: (Untwist+Refract) -> IFFT2 -> K -> FFT2 -> (Refract+Twist)"""
        v = rhs_batch * self._pre_mul
        v_k = self.xp.fft.fft2(v, axes=(1, 2))
        v_k *= self.K_4d
        v = self.xp.fft.ifft2(v_k, axes=(1, 2))
        return v * self._post_mul

    def _apply_M_inv_adjoint(self, rhs_batch: np.ndarray) -> np.ndarray:
        """Adjoint of M^-1."""
        v = rhs_batch * self.xp.conj(self._post_mul)
        v_k = self.xp.fft.fft2(v, axes=(1, 2))
        v_k *= self.xp.conj(self.K_4d)
        v = self.xp.fft.ifft2(v_k, axes=(1, 2))
        return v * self.xp.conj(self._pre_mul)

    def _apply_A(self, u_batch: np.ndarray) -> np.ndarray:
        """Applies A = I + M^-1 * E"""
        u_prev = self.xp.roll(u_batch, 1, axis=3)
        err = self._E1_coeff * u_prev
        err[..., 0] = 0.0
        err[..., 0] += self._E2_coeff * u_batch[..., -1]
        return u_batch + self._apply_M_inv(err)

    def _apply_A_adjoint(self, v_batch: np.ndarray) -> np.ndarray:
        """Computes A^H v"""
        v_next = self.xp.roll(v_batch, -1, axis=3)
        E1_conj_shifted = self.xp.roll(self.xp.conj(self._E1_coeff), -1, axis=3)
        err = E1_conj_shifted * v_next
        err[..., -1] = 0.0
        err[..., -1] += self.xp.conj(self._E2_coeff) * v_batch[..., 0]
        return v_batch + self._apply_M_inv_adjoint(err)

    def compute_gradient_object(
        self, u_sol: np.ndarray, v_sol: np.ndarray, scan_indices: np.ndarray
    ):
        """Assembles complex gradient components via scatter-add."""
        local_overlap = u_sol * self.xp.conj(v_sol)

        if self.use_gpu and not isinstance(scan_indices, self.xp.ndarray):
            scan_indices = self.xp.asarray(scan_indices)

        Ny, Nx = self.ny, self.nx
        batch_y = scan_indices[:, 0][:, self.xp.newaxis, self.xp.newaxis]
        batch_x = scan_indices[:, 1][:, self.xp.newaxis, self.xp.newaxis]

        y_grid = self.xp.arange(Ny)[self.xp.newaxis, :, self.xp.newaxis]
        x_grid = self.xp.arange(Nx)[self.xp.newaxis, self.xp.newaxis, :]

        gather_y = batch_y + y_grid
        gather_x = batch_x + x_grid

        global_grad_overlap = self.xp.zeros_like(
            self.delta_n_global, dtype=np.complex128
        )

        # Scatter add
        if self.use_gpu:
            self.xp.add.at(
                global_grad_overlap, (gather_y, gather_x, slice(None)), local_overlap
            )
        else:
            np.add.at(
                global_grad_overlap, (gather_y, gather_x, slice(None)), local_overlap
            )

        return global_grad_overlap

    def compute_gradient_probe(self, v_sol: np.ndarray) -> np.ndarray:
        """Computes the probe gradient by integrating the adjoint field at the source plane."""
        # v_sol is (Batch, Ny, Nx, Nz)
        # Probe is at z=0
        return self.xp.mean(v_sol[..., 0], axis=0)

    def run_ptycho_batch(
        self,
        psi_init_batch: np.ndarray,
        scan_indices: np.ndarray,
        n_map: np.ndarray,
        mode: str = "forward",
    ) -> np.ndarray:
        """
        Main runner.
        psi_init_batch: (Batch, Ny, Nx)
        scan_indices: (Batch, 2) -> [[y, x], ...]
        """
        # Ensure inputs are on correct device
        if self.use_gpu:
            if not isinstance(psi_init_batch, self.xp.ndarray):
                psi_init_batch = self.xp.asarray(psi_init_batch)
            if not isinstance(scan_indices, self.xp.ndarray):
                scan_indices = self.xp.asarray(scan_indices)

        self.setup_solver(n_map)
        B = scan_indices.shape[0]
        self._setup_batch_operators(scan_indices)

        # Initialize Volume S: (Batch, Ny, Nx, Nz)
        S = self.xp.zeros((B, self.ny, self.nx, self.nz_steps), dtype=np.complex128)

        if mode == "adjoint":
            S[..., -1] = psi_init_batch
        else:
            S[..., 0] = psi_init_batch

        # Initial Guess
        b = self._apply_M_inv_adjoint(S) if mode == "adjoint" else self._apply_M_inv(S)
        u_sol = b.copy()

        # Iterative Solver
        if self.n_iter > 0:
            if self.solver_type == "richardson":
                for _ in range(self.n_iter):
                    op_u = (
                        self._apply_A_adjoint(u_sol)
                        if mode == "adjoint"
                        else self._apply_A(u_sol)
                    )
                    res = b - op_u
                    u_sol += res

            elif self.solver_type == "gmres":

                def matvec(u_flat):
                    u_reshaped = u_flat.reshape(B, self.ny, self.nx, self.nz_steps)
                    res = (
                        self._apply_A_adjoint(u_reshaped)
                        if mode == "adjoint"
                        else self._apply_A(u_reshaped)
                    )
                    return res.flatten()

                # Select appropriate LinearOperator and GMRES
                if self.use_gpu:
                    LO = cpx_linalg.LinearOperator
                    solver = cpx_linalg.gmres
                else:
                    LO = LinearOperator
                    solver = gmres

                size = B * self.ny * self.nx * self.nz_steps
                A_op = LO(
                    (size, size),
                    matvec=matvec,
                    dtype=np.complex128,
                )

                u_flat, _ = solver(
                    A_op, b.flatten(), x0=b.flatten(), maxiter=self.n_iter
                )
                u_sol = u_flat.reshape(B, self.ny, self.nx, self.nz_steps)

        return u_sol

    def initialize_wavefront(self, psi_init: Optional[np.ndarray]) -> np.ndarray:
        """
        Generates or converts the initial wavefront.
        Uses `get_2d_airy_probe` from `generators_3d` if no init is provided.
        """
        if psi_init is not None:
            arr = psi_init.astype(complex)
            if self.use_gpu:
                return self.xp.asarray(arr)
            return arr

        # Use the imported generator
        # Note: The generator uses NumPy (CPU) for Bessel functions.
        # We generate on CPU then move to GPU if needed.
        psi_cpu = get_2d_airy_probe(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            diameter=self.probe_dia,
            focus=self.probe_focus,
            wavelength=self.wavelength,
        )

        if self.use_gpu:
            return self.xp.asarray(psi_cpu)

        return psi_cpu
