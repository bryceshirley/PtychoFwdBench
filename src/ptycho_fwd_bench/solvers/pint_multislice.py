import logging
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .multislice import MultisliceSolver
from .utils import get_spectral_coords


class ParallelMultisliceSolver(MultisliceSolver):
    """
    Parallel 'One-Shot' Multislice Solver using Twisted 3D FFT.

    Replaces sequential time-stepping with a 3D spectral filter.
    Includes adaptive Richardson correction with Corner Error handling.
    """

    def __init__(self, *args, alpha: float = 1e-6, **kwargs):
        # Force store_beam=True because parallel solve computes all slices at once
        kwargs["store_beam"] = True
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.n_iter = 20  # Fixed number of Richardson iterations

        if self.transform_type != "FFT":
            raise ValueError(
                "ParallelMultisliceSolver only supports FFT transform type."
            )

        # --- Pre-processing ---
        logging.info("Initializing solver...")
        # We compute the kernel and store the single-step propagator for corrections
        self.K_3d = self._get_3d_kernel()

    def _twisted_fft(self, u: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Implements the Twisted 3D FFT.
        Order: Transform X -> Twist Z -> Transform Z
        """
        L = self.nz_steps

        # 1. Transverse Transform (X)
        if not inverse:
            u_x = np.fft.fft(u, axis=0)
        else:
            u_x = np.fft.ifft(u, axis=0)

        # 2. Twist & Longitudinal Transform (Z)
        z_indices = np.arange(L)

        if not inverse:
            # Forward: Twist then FFT
            gamma = self.alpha ** (z_indices / L)
            u_twisted = u_x * gamma[np.newaxis, :]  # Broadcast to (Nx, L)
            return np.fft.fft(u_twisted, axis=1)

        else:
            # Inverse: IFFT then Untwist
            u_z = np.fft.ifft(u_x, axis=1)
            gamma_inv = self.alpha ** (-z_indices / L)
            return u_z * gamma_inv[np.newaxis, :]

    def _get_3d_kernel(self) -> np.ndarray:
        """
        Computes the static 3D Dispersion Kernel K.
        Also stores self.L_step (2D propagator) for error calculation.
        """
        # --- Pre-processing ---
        self.n_mean = np.mean(self.n_map)
        self.delta_n = self.n_map - self.n_mean

        # 1. Transverse Propagator (Shifted)
        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))

        # Phase shift from Mean Potential: k0 * (n_mean - 1) * dz
        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (self.n_mean - 1.0)

        # Store single-step diffraction operator (used in Corner Error)
        self.L_step = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]

        # 2. Longitudinal Eigenvalues (Shift Operator)
        L = self.nz_steps
        kz = np.arange(L)
        # lambda_alpha term representing the cyclic shift in Fourier space
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # 3. Construct 3D Kernel
        # K = 1 / (1 - lam_alpha * L_kx)
        denom = 1.0 - (lam_alpha * self.L_step)

        # Avoid division by zero if alpha is too small/unstable
        return 1.0 / denom

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
        output_video: str = "convergence.mp4",
    ) -> "ParallelMultisliceSolver":
        psi_0 = self.initialize_wavefront(psi_init)
        L = self.nz_steps

        # --- Step 1: Prepare Source Vector ---
        S = np.zeros((self.nx, L), dtype=np.complex128)
        S[:, 0] = psi_0

        # --- Step 2: Define Operators ---
        N_phase = 1j * self.k0 * self.delta_n * self.dz / 2
        inv_N = np.exp(-N_phase)
        fwd_N = np.exp(N_phase)
        Error_Diag = 1.0 - np.exp(2 * N_phase)

        def apply_parallel_propagator(rhs_vector):
            v = rhs_vector * inv_N
            v_k = self._twisted_fft(v, inverse=False)
            v_k *= self.K_3d
            v = self._twisted_fft(v_k, inverse=True)
            return v * inv_N

        def compute_corner_error(u_current):
            u_last = u_current[:, -1]
            val = u_last * fwd_N[:, -1]
            val_k = np.fft.fft(val)
            val_k *= self.L_step.flatten()
            val = np.fft.ifft(val_k)
            val *= fwd_N[:, 0]
            val *= self.alpha
            return val

        # --- Capture Frames for Video ---
        # List to store tuples of (Wave Field, Title)
        frames_data: List[Tuple[np.ndarray, str]] = []

        # --- Step 3: Richardson Iteration ---
        # Initial Predictor
        u_base = apply_parallel_propagator(S)
        u_sol = u_base.copy()

        # Save Frame 0
        frames_data.append((np.abs(u_base), "Iter 0: Initial Predictor"))

        # Correction Loop
        for it in range(self.n_iter):
            err_vec = Error_Diag * u_sol

            corner_correction = compute_corner_error(u_sol)
            err_vec[:, 0] += corner_correction

            correction = apply_parallel_propagator(err_vec)
            u_sol = u_base - correction

            # Save Frame k
            frames_data.append((np.abs(u_sol), f"Iter {it + 1}: Correction"))

        # --- Finalize ---
        self.beam_history = u_sol
        self.psi_final = self.beam_history[:, -1]

        # Generate the video
        self.save_animation(frames_data, output_video)

        return self

    def save_animation(self, frames: List[Tuple[np.ndarray, str]], filename: str):
        """
        Compiles the collected frames into an MP4 or GIF animation.
        """
        if not frames:
            logging.warning("No frames to animate.")
            return

        # Setup Figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Initial Plot
        img_data, title = frames[0]
        im = ax.imshow(
            img_data, cmap="magma", aspect="auto", origin="lower", animated=True
        )
        ax.set_xlabel("Propagation Depth (z)")
        ax.set_ylabel("Transverse Position (x)")
        title_text = ax.set_title(title)

        # Colorbar
        fig.colorbar(im, ax=ax, label="Wave Amplitude")

        # Fixed scaling based on the maximum of the *final* solution for stability
        # or use the max of the current frame if you want dynamic scaling.
        # Here we use global max to stop flickering.
        global_max = np.max([np.max(f[0]) for f in frames])
        im.set_clim(0, global_max)

        def update(frame_idx):
            data, title = frames[frame_idx]
            im.set_array(data)
            title_text.set_text(title)
            return im, title_text

        # Create Animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=500,  # 500ms per frame
            blit=True,
        )

        # Save
        logging.info(f"Saving animation to {filename}...")
        try:
            # Try saving as mp4 (requires ffmpeg)
            if filename.endswith(".mp4"):
                ani.save(filename, writer="ffmpeg", fps=2)
            # Fallback to gif if mp4 fails or is requested
            else:
                ani.save(filename, writer="pillow", fps=2)
        except Exception as e:
            logging.error(
                f"Could not save video: {e}. Try installing ffmpeg or saving as .gif."
            )

        plt.close(fig)
