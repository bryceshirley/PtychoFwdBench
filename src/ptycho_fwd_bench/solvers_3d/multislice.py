import numpy as np

from ptycho_fwd_bench.utils import get_spectral_coords

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


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

        with self._device_context():
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

    def _device_context(self):
        if self.use_gpu:
            return cp.cuda.Device(self.gpu_id)
        return np.errstate(all="ignore")

    def precompute_probe(self, psi):
        """Pre-sets the probe on the correct device memory."""
        with self._device_context():
            self.psi_0 = self.xp.asarray(psi, dtype=self.complex_t)

    def run_scan(self, large_map, positions, batch_size=16):
        if not hasattr(self, "psi_0"):
            raise AttributeError(
                "Probe not found. Call precompute_probe() before run_scan()."
            )

        num_pos = len(positions)
        results = np.zeros((num_pos, self.nx, self.ny), dtype=self.complex_t)
        half_nx, half_ny = self.nx // 2, self.ny // 2

        with self._device_context():
            large_map_gpu = self.xp.asarray(large_map, dtype=self.complex_t)

            for b in range(0, num_pos, batch_size):
                b_end = min(b + batch_size, num_pos)
                curr_batch = b_end - b

                psi = self.xp.tile(self.psi_0, (curr_batch, 1, 1))

                for z in range(self.nz_steps):
                    # 1. ENTRANCE REFRACTION (N_1/2)
                    # Only local variations relative to mean
                    for i in range(curr_batch):
                        r, c = positions[b + i]
                        delta_n = (
                            large_map_gpu[
                                r - half_nx : r + half_nx, c - half_ny : c + half_ny, z
                            ]
                            - self.n_mean
                        )
                        psi[i] *= self.xp.exp(0.5j * self.k0 * delta_n * self.dz)

                    # 2. DIFFRACTION + MEAN PHASE (L-tilde)
                    psi_k = self.xp.fft.fft2(psi, axes=(1, 2))
                    psi_k *= self.P
                    psi = self.xp.fft.ifft2(psi_k, axes=(1, 2))

                    # 3. EXIT REFRACTION (N_1/2)
                    for i in range(curr_batch):
                        r, c = positions[b + i]
                        delta_n = (
                            large_map_gpu[
                                r - half_nx : r + half_nx, c - half_ny : c + half_ny, z
                            ]
                            - self.n_mean
                        )
                        psi[i] *= self.xp.exp(0.5j * self.k0 * delta_n * self.dz)

                results[b:b_end] = self.xp.asnumpy(psi)

                del psi
                if self.use_gpu:
                    self.xp.get_default_memory_pool().free_all_blocks()

        return results
