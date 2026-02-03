import logging
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from .multislice import MultisliceSolver
from .utils import get_spectral_coords


class ParallelMultisliceSolver(MultisliceSolver):
    """
    Parallel 'One-Shot' Multislice Solver using Twisted 3D FFT.

    Solves the preconditioned system A * u = b, where:
      A = I + M^-1 * E  (The Preconditioned Operator)
      b = M^-1 * s      (The Preconditioned Source)
    """

    def __init__(self, *args, alpha: float = 1e-6, **kwargs):
        kwargs["store_beam"] = True
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.n_iter = 30

        if self.transform_type != "FFT":
            raise ValueError(
                "ParallelMultisliceSolver only supports FFT transform type."
            )

        logging.info("Initializing solver...")
        self.K_3d = self._get_3d_kernel()

    # =========================================================================
    #  Core Physics Modules (Modular A and b)
    # =========================================================================

    def _setup_operators(self):
        """Pre-computes phase terms needed for A and M^-1."""
        # N_phase is the half-step phase kick: exp(i * delta_n * dz / 2)
        N_phase = 1j * self.k0 * self.delta_n * self.dz / 2
        self._inv_N = np.exp(-N_phase)
        self._fwd_N = np.exp(N_phase)
        self._Error_Diag = 1.0 - np.exp(2 * N_phase)

    def _apply_M_inv(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Applies the Approximate Parallel Solver (M^-1).
        Operation: Refract -> Twisted 3D FFT -> Kernel -> Inverse Twisted FFT -> Refract
        """
        v = rhs_vector * self._inv_N

        #
        v_k = self._twisted_fft(v, inverse=False)
        v_k *= self.K_3d
        v = self._twisted_fft(v_k, inverse=True)

        return v * self._inv_N

    def _apply_Error(self, u_vec: np.ndarray) -> np.ndarray:
        """
        Applies the Error Operator (E1 + E2).
        E1: Diagonal phase errors.
        E2: Corner/Wrap-around errors.
        """
        # 1. Diagonal Error (E1)
        err = self._Error_Diag * u_vec

        # 2. Corner Error (E2)
        u_last = u_vec[:, -1]
        val = u_last * self._fwd_N[:, -1]  # Exit Phase (last slice)
        val_k = np.fft.fft(val) * self.L_step.flatten()  # Diffract
        val = np.fft.ifft(val_k) * self._fwd_N[:, 0]  # Entry Phase (first slice)

        err[:, 0] += val * self.alpha
        return err

    def _apply_A(self, u_vec: np.ndarray) -> np.ndarray:
        """
        Applies the full Linear Operator A = I + M^-1 * E.
        """
        # 1. Calculate Error: e = E * u
        error_term = self._apply_Error(u_vec)

        # 2. Precondition Error: c = M^-1 * e
        correction = self._apply_M_inv(error_term)

        # 3. Apply Identity: u + c
        return u_vec + correction

    def _compute_b(self, S: np.ndarray) -> np.ndarray:
        """
        Computes the RHS vector b = M^-1 * s.
        This effectively acts as the 'One-Shot' predictor.
        """
        return self._apply_M_inv(S)

    # =========================================================================
    #  Solvers
    # =========================================================================

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
        solver_type: str = "gmres",  # "richardson", "gmres", "bicgstab", "anderson"
        tol: float = 1e-5,
        output_video: Optional[str] = "convergence.mp4",
        history_depth: int = 5,
    ) -> "ParallelMultisliceSolver":
        psi_0 = self.initialize_wavefront(psi_init)
        L = self.nz_steps

        # 1. Setup Source S
        S = np.zeros((self.nx, L), dtype=np.complex128)
        S[:, 0] = psi_0

        # 2. Setup Physics Operators
        self._setup_operators()

        # 3. Compute b (Initial Guess / RHS)
        logging.info("Computing Preconditioned Source b = M^-1 s...")
        b = self._compute_b(S)

        # 4. Define Linear Operator A (for SciPy solvers)
        N_tot = self.nx * L

        def matvec_A(u_flat):
            u_vol = u_flat.reshape((self.nx, L))
            return self._apply_A(u_vol).flatten()

        A_op = LinearOperator((N_tot, N_tot), matvec=matvec_A, dtype=np.complex128)

        # --- Frames & Initialization ---
        frames_data: List[Tuple[np.ndarray, str]] = []

        # Ideally, we start with u0 = b.
        # This is because b is the solution to M*u=s, i.e., the approximation.
        u_sol = b.copy()
        frames_data.append((np.abs(u_sol), "Iter 0: Initial Guess (b)"))

        stype = solver_type.lower()

        # --- Solver Logic ---

        if stype == "richardson":
            # Update: u_{k+1} = u_k + (b - A * u_k)
            logging.info("Starting Richardson (Preconditioned Form)...")

            for it in range(self.n_iter):
                # Apply A
                A_u = self._apply_A(u_sol)

                # Residual: r = b - A*u
                residual = b - A_u

                # Update
                u_sol = u_sol + residual

                # Convergence Check
                res_norm = np.linalg.norm(residual)

                frames_data.append((np.abs(u_sol), f"Iter {it + 1}: Richardson"))

                if res_norm < tol:
                    logging.info(f"Richardson converged at iter {it + 1}")
                    break

        elif stype == "anderson":
            # Solve u = u + (b - A*u)  => Find fixed point of G(u) = u + r
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
            raise ValueError(f"Unknown solver type: {solver_type}")

        # --- Finalize ---
        self.beam_history = u_sol
        self.psi_final = self.beam_history[:, -1]

        if output_video:
            self.save_animation(frames_data, output_video)

        return self

    def _twisted_fft(self, u: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Implements the Twisted 3D FFT."""
        L = self.nz_steps
        if not inverse:
            u_x = np.fft.fft(u, axis=0)
        else:
            u_x = np.fft.ifft(u, axis=0)

        z_indices = np.arange(L)
        if not inverse:
            gamma = self.alpha ** (z_indices / L)
            u_twisted = u_x * gamma[np.newaxis, :]
            return np.fft.fft(u_twisted, axis=1)
        else:
            u_z = np.fft.ifft(u_x, axis=1)
            gamma_inv = self.alpha ** (-z_indices / L)
            return u_z * gamma_inv[np.newaxis, :]

    def _get_3d_kernel(self) -> np.ndarray:
        """Computes K and stores L_step."""
        self.n_mean = np.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean
        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)
        self.L_step = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]
        L = self.nz_steps
        kz = np.arange(L)
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]
        denom = 1.0 - (lam_alpha * self.L_step)
        return 1.0 / denom

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
