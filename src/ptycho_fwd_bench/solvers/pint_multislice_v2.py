import logging
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from .base import OpticalWaveSolver
from .utils import get_spectral_coords


class ParallelMultisliceSolver_v2(OpticalWaveSolver):
    """
    Symmetric Parallel-in-Time Multislice Solver.

    Corrections applied:
    1. Gamma is defined as alpha^(-z/L) (Growth) to cancel eigenvalue damping.
    2. E1 Error sign flipped to (O^-1 O^-1 - I).
    3. Kernel uses P / (1 - lambda P) form for stability.
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
        n_iter: int = 5,
        solver_type: str = "richardson",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.alpha = float(alpha)
        self.n_iter = n_iter
        self.solver_type = solver_type.lower()

        # Physics setup
        self.n_mean = np.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean

        # Initial operator setup
        self._setup_operators()

    def _setup_operators(self):
        """
        Sets up the diagonal and off-diagonal operators.
        Can be called again if dz or n_map changes (e.g. for sub-grid solvers).
        """
        L = self.nz_steps
        z_idx = np.arange(L)

        # 1. Physics Terms (Refraction)
        N_phase = (1j * self.k0 * self.delta_n * self.dz / 2).astype(np.complex128)
        self._half_obj = np.exp(N_phase)
        self._inv_half_obj = np.exp(-N_phase)

        # 2. Parallel Twist (Gamma)

        # Exit Twist (Gamma): Growth term alpha^(-z/L)
        self._gamma = (self.alpha ** (-z_idx / L))[np.newaxis, :]

        # Entry Twist (Gamma^-1): Decay term alpha^(z/L)
        self._gamma_inv = (self.alpha ** (z_idx / L))[np.newaxis, :]

        # 3. Pre/Post Multipliers
        # M^-1 = (O_1/2 * G) * [Spectral] * (G^-1 * O_1/2)
        # Pre-multiply applies G^-1 (Entry Twist)
        self._pre_mul = self._half_obj * self._gamma_inv

        # Post-multiply applies G (Exit Twist)
        self._post_mul = self._half_obj * self._gamma

        # 4. Error Term E1 (Sub-diagonal)
        # E1 = Exact - Approx = (-I) - (-InvObj * InvObj) = InvObj*InvObj - I
        shifted_inv = np.roll(self._inv_half_obj, shift=1, axis=1)
        self._E1_coeff = (self._inv_half_obj * shifted_inv) - 1.0

        # 5. Error Term E2 (Corner)
        # E2 = Corner term alpha * InvObj[0] * InvObj[L]
        self._E2_coeff = (
            self.alpha * self._inv_half_obj[:, 0] * self._inv_half_obj[:, -1]
        )

        # 6. Compute Spectral Kernel
        self._K_3d = self._get_3d_kernel()

    def _get_3d_kernel(self) -> np.ndarray:
        """
        Computes K = P / (1 - lambda * P)
        This matches the forward recurrence u_n = P * u_{n-1}
        """
        kx = get_spectral_coords(self.nx, self.dx, "FFT")

        # Transverse Propagator P
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))

        # lambda_vac: vacuum phase advance per dz
        lambda_vac = 1j * (sqrt_term - self.k0)

        # phi_mean: mean potential phase advance per dz
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # P_shifted corresponds to exp(Lambda_bar) in your derivation
        self._P_shifted = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]

        # Longitudinal Eigenvalues (decaying)
        L = self.nz_steps
        kz = np.arange(L)

        # lam_alpha contains the shift operator eigenvalues
        # exp(-2j * pi * kz / L) is standard FFT definition
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # Kernel calculation
        denom = 1.0 - (lam_alpha * self._P_shifted)

        # Numerical safeguard: avoid division by zero
        return self._P_shifted / (denom + 1e-15)

    def _apply_M_inv(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Applies M^-1 using standard 2D FFT.
        Sequence: (Refract+Twist) -> FFT2 -> Kernel -> IFFT2 -> (Untwist+Refract)
        """
        # 1. Combined Real-Space Multiply (Entry)
        v = rhs_vector * self._pre_mul

        # 2. Standard 2D FFT (Space X, Time Z)
        v_k = np.fft.fft2(v, axes=(0, 1))

        # 3. Kernel Multiply (Spectral)
        v_k *= self._K_3d

        # 4. Standard 2D IFFT
        v = np.fft.ifft2(v_k, axes=(0, 1))

        # 5. Combined Real-Space Multiply (Exit)
        return v * self._post_mul

    def _apply_A(self, u: np.ndarray) -> np.ndarray:
        """Applies A = I + M^-1 * E"""
        # E1 acts on u_{j-1}
        u_prev = np.roll(u, 1, axis=1)
        err = self._E1_coeff * u_prev
        err[:, 0] = 0.0  # E1 is strictly sub-diagonal

        # E2 acts on u_{L-1} and adds to slice 0
        err[:, 0] += self._E2_coeff * u[:, -1]

        # Apply M^-1 to error
        correction = self._apply_M_inv(err)
        return u + correction

    def _solve_single_pass(self, psi_0: np.ndarray) -> np.ndarray:
        """Helper to run one solve pass with current dz and operators."""

        # Source S at slice 0 (P applied spectrally later if needed, but here simple injection)
        S = np.zeros((self.nx, self.nz_steps), dtype=np.complex128)
        S[:, 0] = psi_0

        # Initial Guess
        b = self._apply_M_inv(S)
        u_sol = b.copy()

        # Iterative Solver
        if self.n_iter > 0:
            if self.solver_type == "richardson":
                for _ in range(self.n_iter):
                    res = b - self._apply_A(u_sol)
                    u_sol += res

            elif self.solver_type in ["gmres", "bicgstab"]:

                def matvec(u_flat):
                    return self._apply_A(
                        u_flat.reshape(self.nx, self.nz_steps)
                    ).flatten()

                A_op = LinearOperator(
                    (self.nx * self.nz_steps, self.nx * self.nz_steps),
                    matvec=matvec,
                    dtype=np.complex128,
                )

                solver = gmres if self.solver_type == "gmres" else bicgstab
                u_flat, _ = solver(
                    A_op, b.flatten(), x0=b.flatten(), maxiter=self.n_iter
                )
                u_sol = u_flat.reshape(self.nx, self.nz_steps)

        return u_sol

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
        richardson_extrapolation: bool = False,
    ) -> "ParallelMultisliceSolver_v2":
        """
        Runs the solver.

        If richardson_extrapolation is True:
          1. Solves on coarse grid (dz).
          2. Solves on fine grid (dz/2).
          3. Combines solutions: Psi_extrap = (4 * Psi_fine - Psi_coarse) / 3
        """
        psi_0 = self.initialize_wavefront(psi_init)

        if not richardson_extrapolation:
            # Standard single pass
            logging.info("Running Standard Solver...")
            u_sol = self._solve_single_pass(psi_0)
            self.psi_final = u_sol[:, -1]
            if self.store_beam:
                self.beam_history = u_sol
        else:
            logging.info("Running Richardson Extrapolation...")

            # --- Pass 1: Coarse Grid (Current dz) ---
            # Save original state
            orig_dz = self.dz
            orig_nz = self.nz_steps
            orig_delta_n = self.delta_n.copy()

            logging.info(f"Pass 1/2: Coarse grid (dz={orig_dz:.3e}, Nz={orig_nz})")
            u_coarse = self._solve_single_pass(psi_0)

            # Extract result at the end plane
            psi_coarse_end = u_coarse[:, -1]

            # --- Pass 2: Fine Grid (dz/2) ---
            # Update parameters for fine grid
            self.dz = orig_dz / 2.0
            self.nz_steps = orig_nz * 2

            # Interpolate delta_n to double the z-resolution
            # Simple nearest neighbor or linear interpolation along axis 1
            # Here using repeat for "nearest neighbor" equivalent which preserves steps
            self.delta_n = np.repeat(orig_delta_n, 2, axis=1)

            logging.info(f"Pass 2/2: Fine grid (dz={self.dz:.3e}, Nz={self.nz_steps})")

            # Recompute operators for new dz
            self._setup_operators()

            u_fine = self._solve_single_pass(psi_0)

            # Extract result at the end plane (which is now index -1 of the doubled array)
            psi_fine_end = u_fine[:, -1]

            # --- Richardson Combination ---
            # Formula for 2nd order methods: (4 * fine - coarse) / 3
            self.psi_final = (4.0 * psi_fine_end - psi_coarse_end) / 3.0

            if self.store_beam:
                # We can't easily combine the full history due to shape mismatch,
                # so we store the fine grid history as it's more accurate.
                self.beam_history = u_fine

            # --- Restore Original State ---
            self.dz = orig_dz
            self.nz_steps = orig_nz
            self.delta_n = orig_delta_n
            self._setup_operators()  # Restore operators to original state

        return self
