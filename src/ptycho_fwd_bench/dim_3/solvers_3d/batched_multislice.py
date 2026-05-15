import logging

import numpy as np

from ptycho_fwd_bench.dim_2.solvers.utils import get_spectral_coords

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

logger = logging.getLogger(__name__)


class StandardMultisliceSolver:
    def __init__(self, calc_shape, dx, dz, wavelength, n_mean, gpu_id=0):
        self.nx, self.ny, self.nz_steps = calc_shape
        self.dx, self.dz, self.wavelength = dx, dz, wavelength
        self.k0 = 2 * np.pi / wavelength
        self.n_mean = n_mean

        self.complex_t = np.complex128
        self.real_t = np.float64

        self.use_gpu = HAS_GPU and gpu_id is not None
        self.xp = cp if self.use_gpu else np
        self.gpu_id = gpu_id

        logger.debug(
            f"Initializing StandardMultisliceSolver: Grid={calc_shape}, GPU={self.use_gpu} (ID={gpu_id})"
        )

        with self._device_context():
            logger.debug("Building standard 2D shifted Helmholtz propagator...")
            kx = get_spectral_coords(self.nx, self.dx, "FFT")
            ky = get_spectral_coords(self.ny, self.dx, "FFT")

            kx = self.xp.asarray(kx, dtype=self.real_t)
            ky = self.xp.asarray(ky, dtype=self.real_t)

            KX, KY = self.xp.meshgrid(kx, ky, indexing="ij")
            K_sq = KX**2 + KY**2
            del KX, KY

            # === SHIFTED HELMHOLTZ PROPAGATOR  ===
            # Absorbs diffraction and mean phase to minimize parallel error
            inside = self.real_t(self.k0) ** 2 - K_sq
            sqrt_term = self.xp.sqrt(self.xp.clip(inside, 0.0, None))

            lambda_vac = 1j * (sqrt_term - self.real_t(self.k0))
            phi_mean = 1j * self.real_t(self.k0) * (self.n_mean - 1.0)

            # Linear step L-tilde
            self.P = self.xp.exp((lambda_vac + phi_mean) * self.real_t(self.dz)).astype(
                self.complex_t
            )
            logger.debug(
                f"Propagator generated. Memory footprint: {self.P.nbytes / (1024**2):.2f} MB"
            )

    def _device_context(self):
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def precompute_probe(self, psi):
        """Pre-sets the probe on the correct device memory."""
        logger.info("Precomputing probe for Standard Multislice...")
        with self._device_context():
            self.psi_0 = self.xp.asarray(psi, dtype=self.complex_t)
            logger.debug("Probe successfully loaded to device memory.")

    def run_scan(self, large_map, positions, batch_size=16):
        if not hasattr(self, "psi_0"):
            logger.error("Attempted to run scan without precomputing the probe.")
            raise AttributeError(
                "Probe not found. Call precompute_probe() before run_scan()."
            )

        num_pos = len(positions)
        logger.info(
            f"Starting standard sequential scan: {num_pos} positions, batch_size={batch_size}"
        )

        results = np.zeros((num_pos, self.nx, self.ny), dtype=self.complex_t)
        half_nx, half_ny = self.nx // 2, self.ny // 2

        with self._device_context():
            logger.debug("Transferring full potential map to target device...")
            large_map_gpu = self.xp.asarray(large_map, dtype=self.complex_t)

            for b in range(0, num_pos, batch_size):
                b_end = min(b + batch_size, num_pos)
                curr_batch = b_end - b

                psi = self.xp.tile(self.psi_0, (curr_batch, 1, 1))

                # --- Pre-fetch the local 3D maps for the entire batch ---
                delta_n_batch = self.xp.empty(
                    (curr_batch, self.nx, self.ny, self.nz_steps), dtype=self.complex_t
                )
                for i in range(curr_batch):
                    r, c = positions[b + i]
                    delta_n_batch[i] = (
                        large_map_gpu[
                            r - half_nx : r + half_nx, c - half_ny : c + half_ny, :
                        ]
                        - self.n_mean
                    )

                # Pre-calculate the refraction phase shifts for the whole batch
                ref_operator = self.xp.exp(0.5j * self.k0 * delta_n_batch * self.dz)

                for z in range(self.nz_steps):
                    # 1. ENTRANCE REFRACTION
                    psi *= ref_operator[:, :, :, z]

                    # 2. DIFFRACTION + MEAN PHASE
                    psi_k = self.xp.fft.fft2(psi, axes=(1, 2))
                    psi_k *= self.P
                    psi = self.xp.fft.ifft2(psi_k, axes=(1, 2))

                    # 3. EXIT REFRACTION
                    psi *= ref_operator[:, :, :, z]

                results[b:b_end] = self.xp.asnumpy(psi)

                del psi, delta_n_batch, ref_operator
                if self.use_gpu:
                    self.xp.get_default_memory_pool().free_all_blocks()

        logger.info("Scan complete.")
        return results
