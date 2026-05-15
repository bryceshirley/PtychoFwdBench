import numpy as np
from scipy.ndimage import zoom
from scipy.special import j1, jn_zeros

# =============================================================================
# 1. 3D PHANTOM GENERATOR
# =============================================================================


def generate_3d_blob_phantom(
    nx,
    ny,
    nz,
    n_background=1.0 + 0j,
    delta_n=-1.5e-3,
    beta_n=2.0e-4,
    n_blobs=100,
    seed=42,
):
    np.random.seed(seed)
    n_map = np.full((nx, ny, nz), n_background, dtype=np.complex128)

    # Feature sizes optimized for x-ray resolution
    r_min, r_max = nx // 40, nx // 15

    for _ in range(n_blobs):
        # Random center in 3D
        cx, cy, cz = (
            np.random.randint(0, nx),
            np.random.randint(0, ny),
            np.random.randint(0, nz),
        )
        radius = np.random.randint(r_min, r_max + 1)

        # Determine local bounds
        x0, x1 = max(0, cx - radius), min(nx, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(ny, cy + radius + 1)
        z0, z1 = max(0, cz - radius), min(nz, cz + radius + 1)

        ix, iy, iz = np.ogrid[x0:x1, y0:y1, z0:z1]
        dist_sq = (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
        mask = dist_sq <= radius**2

        # Soft-edged features reduce numerical artifacts at slice boundaries
        taper = np.cos(np.pi * np.sqrt(dist_sq[mask]) / (2 * radius))
        n_map[x0:x1, y0:y1, z0:z1][mask] += (delta_n + 1j * beta_n) * taper

    return n_map


# =============================================================================
# 2. 2D AIRY PROBE GENERATOR
# =============================================================================


def get_2d_airy_probe(nx, ny, dx, diameter, focus, wavelength):
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    XX, YY = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(XX**2 + YY**2)
    k0 = 2 * np.pi / wavelength
    first_zero = jn_zeros(1, 1)[0]
    scale_factor = first_zero / (diameter / 2.0)
    arg = r * scale_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = 2.0 * j1(arg) / arg
    amplitude[arg == 0] = 1.0
    if abs(focus) < 1e-12:
        phase = 1.0
    else:
        sign = -1.0 if focus > 0 else 1.0
        dist = abs(focus)
        delta_path = np.sqrt(dist**2 + r**2) - dist
        phase = np.exp(1j * sign * k0 * delta_path)
    return (amplitude * phase).astype(np.complex128)


def interpolate_to_coarse_3d(n_map_fine: np.ndarray, n_steps_coarse: int) -> np.ndarray:
    """
    Downsamples the 3D refractive index map strictly in the propagation direction (Z).

    Used to create coarse-stepped approximations for benchmarking.

    Args:
        n_map_fine (np.ndarray): Original high-resolution 3D map of shape (Ny, Nx, Nz).
        n_steps_coarse (int): Target number of steps in the propagation dimension.

    Returns:
        np.ndarray: The downsampled 3D map.
    """
    ny, nx, nz_fine = n_map_fine.shape
    zoom_factor = n_steps_coarse / nz_fine

    # Zoom factors: 1.0 for Y, 1.0 for X, zoom_factor for Z
    return zoom(n_map_fine, (1.0, 1.0, zoom_factor), order=1)
