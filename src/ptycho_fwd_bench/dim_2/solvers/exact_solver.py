import logging
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from .base import OpticalWaveSolver


class ExactParallelSolver(OpticalWaveSolver):
    """
    Exact Parallel-in-Time Multislice Solver (Distorted Born / Integral Equation Method).
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
        n_iter: int = 20,
        solver_type: str = "gmres",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.alpha = float(alpha)

        # Initialize Physics
        logging.info("Initializing Exact Parallel Solver...")
        self.n_mean = np.mean(self.n_map)

        # The perturbation is the difference from the mean
        self.delta_n = self.n_map - self.n_mean

        # Precompute the Exact Green's Function Kernel
        self.Greens_Kernel = self._get_exact_greens_kernel()

        self.n_iter = n_iter
        self.solver_type = solver_type

    # =========================================================================
    #  1. Physics Kernels (Exact Spectral Definitions)
    # =========================================================================

    def _get_exact_greens_kernel(self) -> np.ndarray:
        """
        Computes the exact Twisted Green's Function in (kx, kz) space.

        G_spectral = 1 / ( i*kz_alpha - (L(kx) + N_mean) )
        """
        L = self.nz_steps

        # --- 1. Transverse Operator L(kx) (Exact Non-Paraxial Envelope) ---
        # L = i(sqrt(k^2 - kx^2) - k)
        # Use fftfreq to match the exact discrete grid modes
        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi

        inside_sqrt = self.k0sq - kx**2
        # Evanescent wave protection
        sqrt_term = np.sqrt(np.clip(inside_sqrt, 0.0, None))

        # Exact non-paraxial diffraction kernel (envelope form)
        L_perp = 1j * (sqrt_term - self.k0)

        # --- 2. Mean Refraction Operator (Constant Shift) ---
        # N_mean = i * k * (n_mean - 1)
        N_bar = 1j * self.k0 * (self.n_mean - 1.0)

        # --- 3. Longitudinal Operator i*kz (Twisted) ---
        # CRITICAL: Use fftfreq here because the Green's function works on the
        # periodic spectrum of the "twisted" signal.
        # Order: [0, 1, ..., L/2, -L/2, ..., -1]
        m = np.fft.fftfreq(L, d=1.0) * L

        D = L * self.dz
        # Twisted wavenumbers: kz = 2*pi*m/D + i*ln(alpha)/D
        kz_alpha = (2 * np.pi * m / D) + (1j * np.log(self.alpha) / D)

        # Broadcast to (Nx, L)
        # denominator = i*kz - (L + N_bar)
        denominator = (1j * kz_alpha)[np.newaxis, :] - (L_perp[:, np.newaxis] + N_bar)

        return 1.0 / denominator

    # =========================================================================
    #  2. Core Operators (Matrix-Free)
    # =========================================================================

    def _apply_Greens_Function(self, source_vol: np.ndarray) -> np.ndarray:
        """
        Applies G * source.
        Step: FFT3D -> Filter -> IFFT3D.
        """
        # 1. Forward 3D FFT (Space X, Time Z)
        source_k = np.fft.fft2(source_vol, axes=(0, 1))

        # 2. Apply Spectral Filter
        scattered_k = source_k * self.Greens_Kernel

        # 3. Inverse 3D FFT
        scattered_vol = np.fft.ifft2(scattered_k, axes=(0, 1))

        return scattered_vol

    def _apply_linear_operator(self, psi_current: np.ndarray) -> np.ndarray:
        """
        The Linear Operator 'A' for the system A * psi = b using 4th-order Simpson's rule.

        Equation: (I - G * V_eff) * psi
        """
        L = self.nz_steps
        dz = self.dz

        # 1. Volumetric Scattering Source using Simpson's Rule (4th-order)
        interaction_coeff = 1j * self.k0 * self.delta_n  # shape (Nx, Nz)

        # Simpson's Rule weights
        weights = np.ones(L)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        weights *= dz / 3.0  # Scale for Simpson integration

        # Apply Simpson's rule along z for each pixel (x)
        # psi_current: shape (Nx, L)
        # Multiply interaction_coeff * psi_current slice-wise
        source_vol = interaction_coeff * psi_current  # shape (Nx, L)

        # Weighted sum along z using Simpson's rule
        source_vol = source_vol * weights[np.newaxis, :]  # broadcast weights along x

        # 2. Propagate Sources (Apply G)
        scattered_wave = self._apply_Greens_Function(source_vol)

        # 3. Return (I - Scattered)
        return psi_current - scattered_wave

    def _compute_incident_wave(self, psi_0: np.ndarray) -> np.ndarray:
        """
        Computes psi_inc (b vector).
        This propagates the initial condition through the MEAN potential exactly.
        """
        L = self.nz_steps

        # CRITICAL FIX: The incident wave is a physical object in real space.
        # We must use monotonic coordinates z = [0, dz, 2dz, ...].
        # Do NOT use fftfreq here (which jumps to negative z).
        z_dist = np.arange(L) * self.dz

        # Fourier Transform Initial Condition
        psi_0_k = np.fft.fft(psi_0)

        # Compute Envelope Propagator in k-space
        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
        inside_sqrt = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside_sqrt, 0.0, None))

        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        total_exponent = lambda_vac + phi_mean  # Shape (Nx,)

        # Propagate to all slices: exp( exponent * z )
        propagator = np.exp(total_exponent[:, np.newaxis] * z_dist[np.newaxis, :])

        psi_inc_k = psi_0_k[:, np.newaxis] * propagator

        return np.fft.ifft(psi_inc_k, axis=0)

    # =========================================================================
    #  3. Solvers
    # =========================================================================

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        output_video: Optional[str] = None,
    ) -> "ExactParallelSolver":
        # 1. Initialize Source
        psi_0 = self.initialize_wavefront(psi_init)

        # 2. Compute RHS (b = psi_inc)
        b_vol = self._compute_incident_wave(psi_0)

        # Video Data
        frames_data: List[Tuple[np.ndarray, str]] = []
        if output_video:
            frames_data.append((np.abs(b_vol), "Iter 0: Incident Wave (Init)"))

        b_flat = b_vol.flatten()

        # 3. Setup Linear Operator A
        L = self.nz_steps
        N_tot = self.nx * L

        def matvec_A(u_flat):
            u_vol = u_flat.reshape((self.nx, L))
            return self._apply_linear_operator(u_vol).flatten()

        A_op = LinearOperator((N_tot, N_tot), matvec=matvec_A, dtype=np.complex128)

        # 4. Solve
        logging.info(f"Starting {self.solver_type.upper()} on Integral Equation...")

        iter_count = 0

        def callback(xk):
            nonlocal iter_count
            iter_count += 1

        # Solver Selection
        if self.solver_type.lower() == "gmres":
            u_flat, info = gmres(
                A_op,
                b_flat,
                x0=b_flat,
                atol=tol,
                maxiter=self.n_iter,
                callback=callback,
            )
        elif self.solver_type.lower() == "bicgstab":
            u_flat, info = bicgstab(
                A_op, b_flat, x0=b_flat, atol=tol, maxiter=self.n_iter
            )
        else:
            # Fallback to Richardson (Born Series)
            logging.info("Using Born Series Iteration (Richardson)...")
            u_flat = b_flat.copy()
            for _i in range(self.n_iter):
                res = b_flat - matvec_A(u_flat)
                if np.linalg.norm(res) < tol:
                    break
                u_flat += res
            info = 0

        if info == 0:
            logging.info("Solver converged successfully.")
        else:
            logging.warning(f"Solver did not converge (info={info}).")

        # 5. Store Results
        self.beam_history = u_flat.reshape((self.nx, L))
        self.psi_final = self.beam_history[:, -1]

        if output_video:
            frames_data.append(
                (np.abs(self.beam_history), f"Final Solution ({self.solver_type})")
            )
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
