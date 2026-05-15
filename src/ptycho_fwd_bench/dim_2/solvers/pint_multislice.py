from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla

from .base import OpticalWaveSolver
from .utils import get_prop_kernel_perp, get_spectral_coords


class ParallelMultisliceSolver(OpticalWaveSolver):
    """
    Parallel Multislice Solver with Woodbury Boundary Correction.
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
        solver_type: str = "Richardson",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.alpha = float(alpha)
        self.n_iter = n_iter
        self.woodbury = woodbury
        self.solver_type = solver_type.upper()

        # Physics setup
        self.n_mean = np.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean

        self._setup_operators()

    def _setup_operators(self):
        """Sets up physics operators. Called in __init__ and during grid resizing."""
        L = self.nz_steps
        z_idx = np.arange(L)

        # 1. Physics Terms (Refraction)
        N_phase = (1j * self.k0 * self.delta_n * self.dz / 2).astype(np.complex128)
        self._half_obj = np.exp(N_phase)
        self._inv_half_obj = np.exp(-N_phase)

        # 2. Parallel Twist (Gamma)
        self._gamma = (self.alpha ** (-z_idx / L))[np.newaxis, :]
        self._gamma_inv = (self.alpha ** (z_idx / L))[np.newaxis, :]

        # 3. Pre/Post Multipliers
        self._pre_mul = self._half_obj * self._gamma_inv
        self._post_mul = self._half_obj * self._gamma

        # 4. Error Terms
        # E1: Object Scattering (Sub-diagonal)
        shifted_inv = np.roll(self._inv_half_obj, shift=1, axis=1)
        self._E1_coeff = (self._inv_half_obj * shifted_inv) - 1.0

        # E2: Boundary Artifact (Corner)
        self._E2_coeff = self.alpha

        # 5. Compute Spectral Kernel
        self._k_2d = self._get_2d_kernel()

        if self.woodbury:
            k_idx = np.arange(self.nz_steps)[np.newaxis, :]
            last_slice_phase = np.exp(-2j * np.pi * k_idx / self.nz_steps)
            H_eff = np.sum(self._k_2d * last_slice_phase, axis=1)
            self._woodbury_kernel = 1.0 / ((1.0 / self.alpha) + H_eff)

    def _get_2d_kernel(self) -> np.ndarray:
        """
        Computes K = P / (1 - lambda * P)
        """
        kx = get_spectral_coords(self.nx, self.dx, "FFT")

        # Transverse Propagator P
        self.prop_kernel_perp = get_prop_kernel_perp(self.k0, kx, self.dz, self.n_mean)[
            :, np.newaxis
        ]  # (Nx,) -> (Nx, 1) for broadcasting

        # Longitudinal Eigenvalues (decaying)
        L = self.nz_steps
        kz = np.arange(L)

        # lam_alpha contains the shift operator eigenvalues
        # exp(-2j * pi * kz / L) is standard FFT definition
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # Kernel calculation
        denom = 1.0 - (lam_alpha * self.prop_kernel_perp)

        # Numerical safeguard: avoid division by zero
        return self.prop_kernel_perp / (denom + 1e-15)

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
        v_k *= self._k_2d

        # 4. Standard 2D IFFT
        v = np.fft.ifft2(v_k, axes=(0, 1))

        # 5. Combined Real-Space Multiply (Exit)
        return v * self._post_mul

    def _apply_M_inv_source(self, psi_0: np.ndarray) -> np.ndarray:
        """Optimized M^-1 application for the initial source."""
        # 1. Compute 1D FFT along X
        v_k_x = np.fft.fft(psi_0, axis=0)

        # 2. Add empty axis for broadcasting: (Nx,) -> (Nx, 1)
        v_k_x = v_k_x[:, np.newaxis]

        # 3. Apply Kernel with standard multiplication
        # This automatically broadcasts (Nx, 1) * (Nx, Nz) -> (Nx, Nz)
        v_k_2d = v_k_x * self._k_2d

        # 4. Inverse 2D FFT
        v = np.fft.ifft2(v_k_2d, axes=(0, 1))

        # 5. Apply Post-multiplier
        return v * self._post_mul

    def _apply_E1(self, u: np.ndarray) -> np.ndarray:
        """Apply E1 error."""
        s_obj = np.empty_like(u)
        s_obj[:, 1:] = u[:, :-1] * self._E1_coeff[:, 1:]
        s_obj[:, 0] = 0.0
        return s_obj

    def _compute_woodbury_correction(self, u_last: np.ndarray) -> np.ndarray:
        """Computes boundary correction s = U W^-1 V^T u."""
        u_hat = np.fft.fft(u_last, axis=0)
        u_hat *= self._woodbury_kernel
        return np.fft.ifft(u_hat, axis=0)

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
            s_bound_source = self._compute_woodbury_correction(b[:, -1])

            u_sol = b - self._apply_M_inv_source(s_bound_source)
        elif u_guess is not None:
            u_sol = u_guess
        else:
            u_sol = b.copy()

        # 2. Richardson Update
        if self.n_iter > 0:
            if self.solver_type == "RICHARDSON":
                for _ in range(self.n_iter):
                    # Calculate Physical Scattering Error (E1)
                    s_obj_field = self._apply_M_inv(self._apply_E1(u_sol))

                    if self.woodbury:
                        # Compute correction source at z=0
                        s_bound_source = self._compute_woodbury_correction(
                            b[:, -1] - s_obj_field[:, -1]
                        )

                        # Propagate this source to get the correction field
                        s_bound_field = self._apply_M_inv_source(s_bound_source)

                        s_obj_field += s_bound_field

                    # C. Update Solution
                    u_sol = b - s_obj_field
            elif self.solver_type == "GMRES":

                def matvec(v):
                    v_reshaped = v.reshape(self.nx, self.nz_steps)

                    # Compute M^-1 * E1(v)
                    s1 = self._apply_M_inv(self._apply_E1(v_reshaped))
                    L_v = s1.copy()

                    if self.woodbury:
                        s_bound_source = self._compute_woodbury_correction(-s1[:, -1])
                        s_bound_field = self._apply_M_inv_source(s_bound_source)
                        L_v += s_bound_field

                    # Return Operator application: (I + L)v
                    return (v_reshaped + L_v).flatten()

                A = spla.LinearOperator(
                    (self.nx * self.nz_steps, self.nx * self.nz_steps),
                    matvec=matvec,
                    dtype=np.complex128,
                )

                # Solve using the correct effective RHS
                u_sol_flat, _ = spla.gmres(
                    A,
                    u_sol.flatten(),
                    x0=u_sol.flatten(),
                    rtol=1e-10,
                    maxiter=self.n_iter,
                )
                u_sol = u_sol_flat.reshape(self.nx, self.nz_steps)
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")

        return u_sol

    def _propagate_and_store(
        self, psi_0: np.ndarray, u_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Wrapper to run the solver and store history if enabled."""
        u_sol = self._solve_pass(psi_0, u_guess=u_guess)
        if self.store_beam:
            self.beam_history = u_sol

        self.psi_final = u_sol[:, -1]

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
    ):
        """
        Runs the solver.
        """
        # Initial operator setup

        psi_0 = self.initialize_wavefront(psi_init)

        self._propagate_and_store(psi_0)

        return self
