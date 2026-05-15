import logging

import numpy as np

from ptycho_fwd_bench.dim_2.solvers.utils import get_spectral_coords

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


class ParallelSTEMSolver:
    def __init__(self, calc_shape, dx, dz, wavelength, n_mean, alpha=1e-10, gpu_id=0):
        self.nx, self.ny, self.nz_steps = calc_shape
        self.dx, self.dz, self.wavelength = dx, dz, wavelength
        self.k0 = 2 * np.pi / wavelength
        self.n_mean = n_mean

        # High Precision for numerical stability with small alpha
        self.complex_t = np.complex128
        self.real_t = np.float64

        self.use_gpu = HAS_GPU and gpu_id is not None
        self.xp = cp if self.use_gpu else np
        self.gpu_id = gpu_id

        self.alpha = self.real_t(alpha)

        logger.debug(
            f"Initializing ParallelSTEMSolver: Grid={calc_shape}, GPU={self.use_gpu} (ID={gpu_id})"
        )

        self._init_static_kernels()
        self._u0_static = None  # Precomputed mean-field predictor

    def _device_context(self):
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def _init_static_kernels(self):
        """Initializes shifted propagator and 3D dispersion kernel."""
        logger.debug("Building static 3D dispersion kernels...")
        with self._device_context():
            kx = get_spectral_coords(self.nx, self.dx, "FFT")
            ky = get_spectral_coords(self.ny, self.dx, "FFT")

            kx = self.xp.asarray(kx, dtype=self.real_t)
            ky = self.xp.asarray(ky, dtype=self.real_t)

            KX, KY = self.xp.meshgrid(kx, ky, indexing="ij")
            K_sq = KX**2 + KY**2

            # Shifted Linear Step (Diffraction + Mean Phase)
            inside = self.real_t(self.k0) ** 2 - K_sq
            sqrt_term = self.xp.sqrt(self.xp.clip(inside, 0.0, None))
            lambda_vac = 1j * (sqrt_term - self.real_t(self.k0))
            phi_mean = 1j * self.real_t(self.k0) * (self.n_mean - 1.0)
            H = self.xp.exp((lambda_vac + phi_mean) * self.real_t(self.dz))

            # 3D Dispersion Kernel Construction
            L = self.nz_steps
            kz = self.xp.arange(L, dtype=self.real_t)
            lam_alpha = self.alpha ** (1 / L) * self.xp.exp(-2j * np.pi * kz / L)
            denom = 1.0 - lam_alpha[None, None, :] * H[:, :, None]
            self.K_3d = (1.0 / denom).astype(self.complex_t)[None, ...]

            kernel_mb = self.K_3d.nbytes / (1024**2)
            logger.debug(f"3D Kernel generated. Memory footprint: {kernel_mb:.2f} MB")

    def _twisted_fft_3d(self, u, inverse=False):
        """Implementation of the Twisted 3D FFT"""
        L = self.nz_steps
        fft = self.xp.fft
        z = self.xp.arange(L, dtype=self.real_t)
        shape = (1, 1, 1, L)

        if not inverse:
            u = fft.fft2(u, axes=(1, 2))
            gamma = (self.alpha ** (z / L)).reshape(shape)
            return fft.fft(u * gamma, axis=3)
        else:
            u = fft.ifft(u, axis=3)
            gamma_inv = (self.alpha ** (-z / L)).reshape(shape)
            return fft.ifft2(u * gamma_inv, axes=(1, 2))

    def precompute_probe(self, probe_init):
        """Computes the static u0 predictor (mean-field solve)."""
        logger.info("Precomputing mean-field predictor (u0) for the probe...")
        with self._device_context():
            psi = self.xp.asarray(probe_init, dtype=self.complex_t)

            # Source vector S with only the first slice occupied
            S = self.xp.zeros(
                (1, self.nx, self.ny, self.nz_steps), dtype=self.complex_t
            )
            S[0, :, :, 0] = psi

            v = self._twisted_fft_3d(S)
            v *= self.K_3d  # Propagate through mean potential
            self._u0_static = self._twisted_fft_3d(v, inverse=True)
            logger.debug("Mean-field predictor successfully precomputed and cached.")

    def run_scan(self, large_n_map, positions, batch_size=1, n_iter=3):
        """Main batch solver using Richardson Iteration."""
        if self._u0_static is None:
            logger.error("Attempted to run scan without precomputing the probe.")
            raise ValueError("Must call precompute_probe before run_scan.")

        num_pos = len(positions)
        logger.info(
            f"Starting parallel scan: {num_pos} positions, batch_size={batch_size}, n_iter={n_iter}"
        )

        if self.use_gpu:
            logger.debug("Clearing GPU memory pool before scan block...")
            self.xp.get_default_memory_pool().free_all_blocks()

        results = np.zeros((num_pos, self.nx, self.ny), dtype=self.complex_t)
        hx, hy = self.nx // 2, self.ny // 2

        with self._device_context():
            logger.debug("Transferring full potential map to target device...")
            n_map = self.xp.asarray(large_n_map, dtype=self.complex_t)

            for b0 in range(0, num_pos, batch_size):
                b1 = min(b0 + batch_size, num_pos)
                B = b1 - b0

                delta_n = self.xp.empty(
                    (B, self.nx, self.ny, self.nz_steps), dtype=self.complex_t
                )
                for i in range(B):
                    r, c = positions[b0 + i]
                    delta_n[i] = (
                        n_map[r - hx : r + hx, c - hy : c + hy, :] - self.n_mean
                    )

                # Refraction Operators (Perturbation Only)
                D = self.xp.exp(0.5j * self.k0 * delta_n * self.dz)
                D_inv = self.xp.conj(D)
                Error_Diag = 1.0 - (D * D)  # Diagonal Error E

                # Initialize u with the precomputed predictor
                # Apply symmetric split: u = D_inv * Solve * D_inv * S
                # Since S is just at z=0, the predictor u_base already handles D_inv internally
                u = self.xp.tile(self._u0_static, (B, 1, 1, 1)) * D

                # Richardson Correction Loop
                for _ in range(n_iter - 1):
                    rhs = Error_Diag * u  # Local Phase Error

                    v_err = self._twisted_fft_3d(rhs * D_inv)
                    v_err *= self.K_3d
                    corr = self._twisted_fft_3d(v_err, inverse=True)

                    u = u - (corr * D)  # Refocusing update

                results[b0:b1] = self.xp.asnumpy(u[:, :, :, -1])

        logger.info("Scan complete.")
        return results
