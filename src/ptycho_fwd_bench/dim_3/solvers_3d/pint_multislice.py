import logging
from typing import Optional

import numpy as np

from ptycho_fwd_bench.dim_2.solvers.utils import get_spectral_coords

from .base import OpticalWaveSolver3D

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

logger = logging.getLogger(__name__)


class ParallelMultisliceSolver3D(OpticalWaveSolver3D):
    """
    Parallel Multislice Solver with Woodbury Boundary Correction for 3D volumes.
    Uses CuPy for GPU acceleration if available.
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
        alpha: float = 1e-6,
        n_iter: int = 1,
        woodbury: bool = True,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)

        # GPU Setup
        self.use_gpu = use_gpu and HAS_GPU
        self.gpu_id = gpu_id
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            # Move map to GPU immediately
            with cp.cuda.Device(self.gpu_id):
                self.n_map = cp.asarray(self.n_map)
        else:
            if use_gpu and not HAS_GPU:
                logger.warning("CuPy not found. Falling back to CPU (NumPy).")
            else:
                logger.info("ParallelMultisliceSolver3D using CPU (NumPy).")

        # Extract 3D shape safely
        self.ny, self.nx, self.nz_steps = self.n_map.shape

        self.n_map[:, :, 0] = 1.0  # Ensure first slice is free space
        self.n_map[:, :, -1] = 1.0  # Ensure last slice is free space
        self.alpha = float(alpha)
        self.n_iter = n_iter
        self.woodbury = woodbury

        # Physics setup
        self.n_mean = self.xp.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean

        # Initial operator setup
        with self._device_context():
            self._setup_operators()

    def _device_context(self):
        """Context manager to ensure operations happen on the correct GPU."""
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def _setup_operators(self):
        """Sets up physics operators. Called in __init__ and during grid resizing."""
        L = self.nz_steps
        z_idx = self.xp.arange(L)

        # 1. Physics Terms (Refraction) - Now 3D
        N_phase = (1j * self.k0 * self.delta_n * self.dz / 2).astype(np.complex128)
        self._half_obj = self.xp.exp(N_phase)
        self._inv_half_obj = self.xp.exp(-N_phase)

        # 2. Parallel Twist (Gamma) - Broadcast to (1, 1, Nz)
        self._gamma = (self.alpha ** (-z_idx / L))[self.xp.newaxis, self.xp.newaxis, :]
        self._gamma_inv = (self.alpha ** (z_idx / L))[
            self.xp.newaxis, self.xp.newaxis, :
        ]

        # 3. Pre/Post Multipliers
        self._pre_mul = self._half_obj * self._gamma_inv
        self._post_mul = self._half_obj * self._gamma

        # 4. Error Terms
        # E1: Object Scattering (Sub-diagonal) - Shift along Z (axis 2)
        shifted_inv = self.xp.roll(self._inv_half_obj, shift=1, axis=2)
        self._E1_coeff = (self._inv_half_obj * shifted_inv) - 1.0

        # E2: Boundary Artifact (Corner)
        self._E2_coeff = self.alpha

        # 5. Compute Spectral Kernel
        self._K_3d = self._get_3d_kernel()

        if self.woodbury:
            k_idx = self.xp.arange(self.nz_steps)[self.xp.newaxis, self.xp.newaxis, :]
            last_slice_phase = self.xp.exp(-2j * self.xp.pi * k_idx / self.nz_steps)
            # Sum across Z (axis 2) to get 2D effective boundary operator
            H_eff = self.xp.sum(self._K_3d * last_slice_phase, axis=2)
            self._woodbury_kernel = 1.0 / ((1.0 / self.alpha) + H_eff)

    def _get_perp_kernel(self) -> np.ndarray:
        # get_spectral_coords returns CPU numpy arrays, so we must move them
        kx_cpu = get_spectral_coords(self.nx, self.dx, "FFT")
        ky_cpu = get_spectral_coords(self.ny, self.dx, "FFT")  # Assuming dy = dx

        kx = self.xp.asarray(kx_cpu)
        ky = self.xp.asarray(ky_cpu)

        KX, KY = self.xp.meshgrid(kx, ky, indexing="ij")
        K_sq = KX**2 + KY**2

        # Transverse Propagator P (Vacuum + Mean phase)
        inside = self.k0**2 - K_sq
        sqrt_term = self.xp.sqrt(self.xp.clip(inside, 0.0, None))

        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # P shape: (Ny, Nx) -> expand to (Ny, Nx, 1)
        self.prop_kernel_perp = self.xp.exp((lambda_vac + phi_mean) * self.dz)[
            :, :, self.xp.newaxis
        ]

    def _get_3d_kernel(self):
        """
        Computes K = P / (1 - lambda * P) for 3D propagation.
        """
        if not hasattr(self, "prop_kernel_perp"):
            self._get_perp_kernel()

        # Longitudinal Eigenvalues (decaying)
        L = self.nz_steps
        kz = self.xp.arange(L)

        # lam_alpha contains the shift operator eigenvalues
        # exp(-2j * pi * kz / L) is standard FFT definition
        lam_alpha = (self.alpha ** (1 / L) * self.xp.exp(-2j * self.xp.pi * kz / L))[
            self.xp.newaxis, self.xp.newaxis, :
        ]

        # Kernel calculation
        denom = 1.0 - (lam_alpha * self.prop_kernel_perp)

        # Numerical safeguard: avoid division by zero
        return self.prop_kernel_perp / (denom + 1e-15)

    def _apply_M_inv(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Applies M^-1 using standard 3D FFT.
        Sequence: (Refract+Twist) -> FFT3 -> Kernel -> IFFT3 -> (Untwist+Refract)
        """
        # 1. Combined Real-Space Multiply (Entry)
        v = rhs_vector * self._pre_mul

        # 2. Standard 3D FFT (Space Y, Space X, Time Z)
        v_k = self.xp.fft.fftn(v, axes=(0, 1, 2))

        # 3. Kernel Multiply (Spectral)
        v_k *= self._K_3d

        # 4. Standard 3D IFFT
        v = self.xp.fft.ifftn(v_k, axes=(0, 1, 2))

        # 5. Combined Real-Space Multiply (Exit)
        return v * self._post_mul

    def _apply_M_inv_source(self, psi_0: np.ndarray) -> np.ndarray:
        """Optimized M^-1 application for the initial source."""
        # 1. Compute 2D FFT along Y, X
        v_k_xy = self.xp.fft.fft2(psi_0, axes=(0, 1))

        # 2. Add empty axis for broadcasting: (Ny, Nx) -> (Ny, Nx, 1)
        v_k_xy = v_k_xy[:, :, self.xp.newaxis]

        # 3. Apply Kernel with standard multiplication
        # Broadcasts (Ny, Nx, 1) * (Ny, Nx, Nz) -> (Ny, Nx, Nz)
        v_k_3d = v_k_xy * self._K_3d

        # 4. Inverse 3D FFT
        v = self.xp.fft.ifftn(v_k_3d, axes=(0, 1, 2))

        # 5. Apply Post-multiplier
        return v * self._post_mul

    def _apply_E1(self, u: np.ndarray) -> np.ndarray:
        """Apply E1 error."""
        s_obj = self.xp.empty_like(u)
        s_obj[:, :, 1:] = u[:, :, :-1] * self._E1_coeff[:, :, 1:]
        s_obj[:, :, 0] = 0.0
        return s_obj

    def _compute_woodbury_correction(self, u_last: np.ndarray) -> np.ndarray:
        """Computes boundary correction s = U W^-1 V^T u."""
        u_hat = self.xp.fft.fft2(u_last, axes=(0, 1))
        u_hat *= self._woodbury_kernel
        return self.xp.fft.ifft2(u_hat, axes=(0, 1))

    def _solve_pass(
        self, psi_0: np.ndarray, u_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Runs the solver pass.
        """
        # 1. One-Shot Solution (Initial Guess u_0). Eliminates E_2.
        b = self._apply_M_inv_source(psi_0)
        u_sol = b.copy()

        if self.woodbury and u_guess is None:
            s_bound_source = self._compute_woodbury_correction(b[:, :, -1])

            u_sol = b - self._apply_M_inv_source(s_bound_source)
        elif u_guess is not None:
            u_sol = u_guess
        else:
            u_sol = b.copy()

        # 2. Richardson Update
        if self.n_iter > 0:
            for _ in range(self.n_iter):
                # Calculate Physical Scattering Error (E1)
                s_obj_field = self._apply_M_inv(self._apply_E1(u_sol))

                if self.woodbury:
                    # Compute correction source at z=0
                    s_bound_source = self._compute_woodbury_correction(
                        b[:, :, -1] - s_obj_field[:, :, -1]
                    )

                    # Propagate this source to get the correction field
                    s_bound_field = self._apply_M_inv_source(s_bound_source)

                    s_obj_field += s_bound_field

                # C. Update Solution
                u_sol = b - s_obj_field

        return u_sol

    def _propagate_and_store(
        self, psi_0: np.ndarray, u_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Wrapper to run the solver and store history if enabled."""
        with self._device_context():
            u_sol = self._solve_pass(psi_0, u_guess=u_guess)

            # Transfer back to CPU if requested, or keep on GPU
            if self.store_beam:
                self.beam_history = cp.asnumpy(u_sol) if self.use_gpu else u_sol

            self.psi_final = (
                cp.asnumpy(u_sol[:, :, -1]) if self.use_gpu else u_sol[:, :, -1]
            )

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
    ):
        """
        Runs the solver.
        """
        psi_0 = self.initialize_wavefront(psi_init)

        # Move probe to GPU
        with self._device_context():
            psi_0_device = self.xp.asarray(psi_0)

        self._propagate_and_store(psi_0_device)

        return self
