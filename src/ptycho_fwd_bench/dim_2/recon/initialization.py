import numpy as np


def initialize_probe(mode: str, nx_p: int, seed: int) -> np.ndarray:
    if mode == "ones":
        return np.ones(nx_p, dtype=np.complex128)
    elif mode == "zeros":
        return np.zeros(nx_p, dtype=np.complex128)
    elif mode == "gaussian":
        x = np.arange(nx_p)
        center = nx_p // 2
        width = nx_p // 6
        return (np.exp(-((x - center) ** 2) / (2 * width**2))).astype(np.complex128)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return ((rng.random(nx_p) + 1j * rng.random(nx_p)) * 0.1 + 0.5).astype(
            np.complex128
        )
    elif mode == "cauchy":
        x = np.arange(nx_p)
        center = nx_p // 2
        width = nx_p // 10
        return (1.0 / (1.0 + ((x - center) / width) ** 2)).astype(np.complex128)
    else:
        raise ValueError(f"Unknown probe init mode: {mode}")


def initialize_object(mode: str, n_map_shape: tuple, seed: int) -> np.ndarray:
    if mode == "free-space":
        return np.full(n_map_shape, 1.0 + 0j, dtype=np.complex128)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        n_real = 1.0 + (rng.random(n_map_shape) * 1e-4)
        n_imag = rng.random(n_map_shape) * 1e-5
        return (n_real + 1j * n_imag).astype(np.complex128)
    else:
        raise ValueError(f"Unknown object init mode: {mode}")
