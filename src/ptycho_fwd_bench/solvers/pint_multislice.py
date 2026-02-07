import logging
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from .base import OpticalWaveSolver
from .utils import get_spectral_coords


class ParallelMultisliceSolver(OpticalWaveSolver):
    """
    Parallel 'One-Shot' Multislice Solver (2D Version: x, z).

    Solves the preconditioned system A * u = b, where:
      A = I + M^-1 * E  (The Preconditioned Operator)
      b = M^-1 * s      (The Preconditioned Source)
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        store_beam: bool = False,
        alpha: float = 1e-6,
        n_iter: int = 3,
        solver_type: str = "richardson",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.alpha = float(alpha)

        # Initialize Physics
        logging.info("Initializing solver...")
        self.n_mean = np.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean
        self.K_3d = self._get_3d_kernel()
        self.n_iter = n_iter
        self.solver_type = solver_type

    # =========================================================================
    #  1. Operator Setup (Simplified 2D Implementation)
    # =========================================================================

    def setup_operators(self):
        """
        Pre-computes phase terms.
        """
        L = self.nz_steps
        z_idx = np.arange(L)

        # A. Physics Terms (Refraction)
        # N_phase shape: (Nx, L)
        N_phase = 1j * self.k0 * self.delta_n * self.dz / 2
        inv_N = np.exp(-N_phase)

        # B. Parallel Solver Terms (The Twist)
        # Gamma acts on z-index (columns). Broadcast to (1, L)
        gamma = (self.alpha ** (z_idx / L))[np.newaxis, :]
        gamma_inv = (self.alpha ** (-z_idx / L))[np.newaxis, :]

        # C. Merged Operators for M^-1
        #   Pre:  Apply Refraction + Twist
        #   Post: Untwist + Apply Refraction
        self._pre_mul = inv_N * gamma
        self._post_mul = inv_N * gamma_inv

        # D. Error Operators (For calculating E)
        #   These use standard physics definitions (no twist)
        self._fwd_N = np.exp(N_phase)
        self._Error_Diag = 1.0 - np.exp(2 * N_phase)

    def _get_3d_kernel(self) -> np.ndarray:
        """Computes K for 2D plane (x, z)."""
        # 1. Transverse Propagator (x direction)
        kx = get_spectral_coords(self.nx, self.dx, "FFT")  # (Nx,)

        # Transverse Propagator P
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # _P_shifted shape: (Nx, 1) for broadcasting against Z
        self._P_shifted = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]

        # 2. Longitudinal Shift (z direction)
        L = self.nz_steps
        kz = np.arange(L)

        # lam_alpha
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # 3. Construct 3D Kernel (Nx, L)
        # K = 1 / (1 - lam_alpha * _P_shifted)
        denom = 1.0 - (lam_alpha * self._P_shifted)
        return 1.0 / (denom + 1e-15)

    # =========================================================================
    #  2. Core Physics Modules
    # =========================================================================

    def _apply_M_inv(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Applies M^-1 using standard 2D FFT.
        Sequence: (Refract+Twist) -> FFT2 -> Kernel -> IFFT2 -> (Untwist+Refract)
        """
        # 1. Combined Real-Space Multiply
        v = rhs_vector * self._pre_mul  # Depends on position in ptycho scan

        # 2. Standard 2D FFT (Space X, Time Z)
        v_k = np.fft.fft2(v, axes=(0, 1))

        # 3. Kernel Multiply (Spectral)
        v_k *= self.K_3d  # Does not depend on position in ptycho scan

        # 4. Standard 2D IFFT
        v = np.fft.ifft2(v_k, axes=(0, 1))

        # 5. Combined Real-Space Multiply
        return v * self._post_mul  # Depends on position in ptycho scan

    def _apply_Error(self, u_vec: np.ndarray) -> np.ndarray:
        """
        Applies the Error Operator (E1 + E2).
        """
        # 1. Diagonal Error (E1)
        err = self._Error_Diag * u_vec

        # 2. Corner Error (E2)
        # Connects last slice L-1 to first slice 0
        u_last = u_vec[:, -1]

        # A. Apply Exit Phase (slice L-1)
        val = u_last * self._fwd_N[:, -1]

        # B. Apply Diffraction L (1D FFT over x)
        val_k = np.fft.fft(val, axis=0)
        val_k *= self._P_shifted.flatten()  # Does not depend on position in ptycho scan
        val = np.fft.ifft(val_k, axis=0)

        # C. Apply Entry Phase (slice 0)
        val *= self._fwd_N[:, 0]

        # Add to first slice of error vector, scaled by alpha
        err[:, 0] += val * self.alpha
        return err

    def _apply_A(self, u_vec: np.ndarray) -> np.ndarray:
        """
        Applies the full Linear Operator A = I + M^-1 * E.
        """
        # e = E * u
        error_term = self._apply_Error(u_vec)
        # c = M^-1 * e
        correction = self._apply_M_inv(error_term)
        # result = u + c
        return u_vec + correction

    def _apply_M_inv_adjoint(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Applies (M^-1)^H.
        Forward: (Refract+Twist) -> FFT2 -> Kernel -> IFFT2 -> (Untwist+Refract)
        Adjoint: (Untwist+Refract)^H -> FFT2 -> (Kernel)^H -> IFFT2 -> (Refract+Twist)^H
        """
        # 1. Adjoint of Post-Multiply (Complex Conjugate)
        v = rhs_vector * np.conj(self._post_mul)

        # 2. 2D FFT
        v_k = np.fft.fft2(v, axes=(0, 1))

        # 3. Adjoint of Kernel (Complex Conjugate)
        v_k *= np.conj(self.K_3d)

        # 4. 2D IFFT
        v = np.fft.ifft2(v_k, axes=(0, 1))

        # 5. Adjoint of Pre-Multiply
        return v * np.conj(self._pre_mul)

    def _apply_Error_adjoint(self, v_vec: np.ndarray) -> np.ndarray:
        """
        Applies E^H.
        Diagonal terms are easy. Corner term E2 moves from slice 0 to slice L-1.
        """
        # 1. Adjoint of Diagonal Error (E1)
        err = np.conj(self._Error_Diag) * v_vec

        # 2. Adjoint of Corner Error (E2^H)
        # Forward: u[:, -1] -> slice 0
        # Adjoint: v[:, 0]  -> slice L-1
        v_first = v_vec[:, 0]

        # A. Adjoint of Entry Phase (slice 0)
        val = v_first * np.conj(self._fwd_N[:, 0])

        # B. Adjoint of Diffraction (Reverse _P_shifted)
        val_k = np.fft.fft(val, axis=0)
        val_k *= np.conj(self._P_shifted.flatten())
        val = np.fft.ifft(val_k, axis=0)

        # C. Adjoint of Exit Phase (slice L-1)
        val *= np.conj(self._fwd_N[:, -1])

        # Add back to the LAST slice, scaled by alpha (which is real)
        err[:, -1] += val * self.alpha
        return err

    def _apply_A_adjoint(self, v_vec: np.ndarray) -> np.ndarray:
        """
        Applies A^H = I + E^H * (M^-1)^H
        """
        # 1. Apply (M^-1)^H first
        m_inv_adj = self._apply_M_inv_adjoint(v_vec)
        # 2. Apply E^H
        error_adj = self._apply_Error_adjoint(m_inv_adj)
        # Result = I + E^H @ M^-H
        return v_vec + error_adj

    def _compute_b(self, S: np.ndarray, mode: str = "forward") -> np.ndarray:
        """Computes b = M^-1 * s."""
        if mode == "adjoint":
            return self._apply_M_inv_adjoint(S)
        if mode == "forward":
            return self._apply_M_inv(S)

    # =========================================================================
    #  Solvers
    # =========================================================================

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
        mode: str = "forward",
        tol: float = 1e-5,
        output_video: Optional[str] = None,
    ) -> "ParallelMultisliceSolver":
        # 1. Setup Physics Operators
        if not hasattr(self, "_pre_mul"):
            self.setup_operators()

        # 2. Setup Source S
        psi_0 = self.initialize_wavefront(psi_init)
        L = self.nz_steps
        S = np.zeros((self.nx, L), dtype=np.complex128)

        # Compute exact P = O * P_prop * O * psi_0
        # A. First Half-Refract
        temp = psi_0 * self._fwd_N[:, 0]
        # B. Propagate (requires FFT)
        temp_k = np.fft.fft(temp)
        temp_k *= self._P_shifted.flatten()
        temp = np.fft.ifft(temp_k)
        # C. Second Half-Refract
        P_exact = temp * self._fwd_N[:, 0]
        S[:, 0] = (
            P_exact  # Source is the exact first slice after probe entry (including refraction)
        )

        # 3. Compute b (Initial Guess / RHS)
        logging.info("Computing Preconditioned Source b = M^-1 s...")
        b = self._compute_b(S, mode=mode)

        # 4. Define Linear Operator A (for SciPy solvers)
        N_tot = self.nx * L

        def matvec_A(u_flat):
            u_vol = u_flat.reshape((self.nx, L))
            if mode == "adjoint":
                return self._apply_A_adjoint(u_vol).flatten()
            else:
                return self._apply_A(u_vol).flatten()

        A_op = LinearOperator((N_tot, N_tot), matvec=matvec_A, dtype=np.complex128)
        # We start with u0 = b.
        # This is because b is the solution to M*u=s, i.e., the approximation.
        u_sol = b.copy()

        if output_video:
            # --- Frames & Initialization ---
            frames_data: List[Tuple[np.ndarray, str]] = []
            frames_data.append((np.abs(u_sol), "Iter 0: Initial Guess (b)"))

        if self.n_iter == 0:
            logging.info(
                "n_iter=0: Skipping solver iterations. Output will be initial guess b."
            )
            # --- Finalize ---
            self.beam_history = u_sol
            self.psi_final = self.beam_history[:, -1]

            return self

        stype = self.solver_type.lower()

        # --- Solver Logic ---
        if stype == "richardson":
            # Update: u_{k+1} = u_k + (b - A * u_k)
            logging.info("Starting Richardson (Preconditioned Form)...")

            for it in range(self.n_iter):
                # Apply A
                A_u = (
                    self._apply_A(u_sol)
                    if mode == "forward"
                    else self._apply_A_adjoint(u_sol)
                )

                # Residual: r = b - A*u
                residual = b - A_u

                # Update
                u_sol = u_sol + residual

                # Convergence Check
                res_norm = np.linalg.norm(residual)

                if output_video:
                    frames_data.append((np.abs(u_sol), f"Iter {it + 1}: Richardson"))

                if res_norm < tol:
                    logging.info(f"Richardson converged at iter {it + 1}")
                    break

        elif stype == "anderson":
            # Solve u = u + (b - A*u)  => Find fixed point of G(u) = u + r
            history_depth = 5
            logging.info(f"Starting Anderson (m={history_depth})...")

            X = []  # History of u
            F = []  # History of residuals (G(u) - u) = (b - A*u)

            for it in range(self.n_iter):
                # 1. Evaluate Residual
                A_u = self._apply_A(u_sol)
                residual = b - A_u  # This is f(u)

                # Check Convergence
                if np.linalg.norm(residual) < tol:
                    logging.info(f"Anderson converged at iter {it}")
                    break

                # 2. Update History
                u_flat = u_sol.flatten()
                f_flat = residual.flatten()
                X.append(u_flat)
                F.append(f_flat)

                if len(X) > history_depth:
                    X.pop(0)
                    F.pop(0)

                # 3. Anderson Mixing
                m_k = len(X)
                if m_k == 1:
                    u_sol = u_sol + residual
                else:
                    # Minimize || f_k - dF * gamma ||
                    F_mat = np.column_stack(F)
                    dF = F_mat[:, :-1] - F_mat[:, -1:]

                    if dF.shape[1] > 0:
                        gamma, _, _, _ = np.linalg.lstsq(dF, f_flat, rcond=None)

                        X_mat = np.column_stack(X)
                        dX = X_mat[:, :-1] - X_mat[:, -1:]

                        u_mix = u_flat - (dX @ gamma)
                        f_mix = f_flat - (dF @ gamma)

                        # New Guess = Mixed U + Mixed Residual
                        u_next = u_mix + f_mix
                        u_sol = u_next.reshape((self.nx, L))
                    else:
                        u_sol = u_sol + residual

                if output_video:
                    frames_data.append((np.abs(u_sol), f"Iter {it + 1}: Anderson"))

        elif stype in ["gmres", "bicgstab"]:
            # Solve A * u = b directly
            # Note: We do NOT pass M to scipy here, because A_op IS the preconditioned system.
            logging.info(f"Starting {stype.upper()} on A*u = b...")

            iter_count = 0

            def callback(xk):
                nonlocal iter_count
                iter_count += 1
                if output_video:
                    img = np.abs(xk.reshape((self.nx, L)))
                    frames_data.append((img, f"Iter {iter_count}: {stype.upper()}"))

            solver_func = gmres if stype == "gmres" else bicgstab
            # For GMRES, we need callback_type='x' to get the vector for video
            kwargs = {"callback_type": "x"} if stype == "gmres" else {}

            u_flat, exit_code = solver_func(
                A_op,
                b.flatten(),
                x0=b.flatten(),  # Initialize with b (result of One-Shot)
                atol=tol,
                maxiter=self.n_iter,
                callback=callback,
                **kwargs,
            )

            if exit_code == 0:
                logging.info(f"{stype.upper()} converged in {iter_count} iterations.")
            else:
                logging.warning(f"{stype.upper()} did not fully converge.")

            u_sol = u_flat.reshape((self.nx, L))

        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        # --- Finalize ---
        self.beam_history = u_sol
        self.psi_final = self.beam_history[:, -1]

        if output_video:
            self.save_animation(frames_data, output_video)

        return self

    def save_animation(self, frames: List[Tuple[np.ndarray, str]], filename: str):
        if not frames:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        img_data, title = frames[0]
        im = ax.imshow(
            img_data, cmap="magma", aspect="auto", origin="lower", animated=True
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        im.set_clim(0, np.max([np.max(f[0]) for f in frames]))

        def update(i):
            im.set_array(frames[i][0])
            ax.set_title(frames[i][1])
            return (im,)

        ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
        try:
            ani.save(filename, writer="pillow", fps=2)
        except Exception as e:
            logging.error(str(e))
        plt.close(fig)
