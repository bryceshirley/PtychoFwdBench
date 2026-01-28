import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import j1, jn_zeros

# =============================================================================
# 1. PROBE GENERATOR
# =============================================================================


def get_probe_field(
    coord: np.ndarray, center: float, diameter: float, focus: float, wavelength: float
) -> np.ndarray:
    """
    Generates a 1D probe field (Airy Disk Amplitude + Spherical Curvature).

    Args:
        coord (np.ndarray): 1D array of spatial coordinates.
        center (float): Center position of the probe.
        diameter (float): Diameter of the aperture.
        focus (float): Focal distance.
            Positive for convergent/focusing, negative for divergent/defocusing.
        wavelength (float): Wavelength of the probe.
    """
    k0 = 2 * np.pi / wavelength
    r = np.abs(coord - center)

    first_zero = jn_zeros(1, 1)[0]

    # --- 1. Amplitude: Airy Disk ---
    # When r == radius, we want the Bessel argument to be the First Zero.
    scale_factor = first_zero / (diameter / 2.0)
    x = r * scale_factor

    # Handle x=0 singularity safely
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = 2.0 * j1(x) / x
    amplitude[x == 0] = 1.0

    # --- 2. Phase: Spherical Wavefront ---
    k0 = 2 * np.pi / wavelength
    if abs(focus) < 1e-12:
        phase = 1.0
    else:
        # Exact spherical curvature
        sign = 1.0 if focus < 0 else -1.0
        dist = abs(focus)
        delta_path = np.sqrt(dist**2 + r**2) - dist
        phase = np.exp(1j * sign * k0 * delta_path)

    return amplitude * phase


# =============================================================================
# 2. PHANTOM GENERATORS
# =============================================================================


def generate_empty_phantom(nz: int, nr: int, n_background: float = 1.0) -> np.ndarray:
    """
    Generates a homogeneous refractive index map.

    Args:
        nz (int): Number of pixels in the transverse direction (rows).
        nr (int): Number of pixels in the propagation direction (columns).
        n_background (float): The refractive index value.

    Returns:
        np.ndarray: 2D complex array initialized to the background value.
    """
    return np.ones((nz, nr), dtype=np.complex128) * n_background


def generate_blob_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.0,
    delta_n: float = 0.01,
    beta_n: float = 0.01,
    n_blobs: int = 100,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Generates a phantom with soft, cosine-tapered circular blobs.
    Respects physical sizing if 'blob_r_range_um' and 'dx' are provided.

    Parameters:
        nz (int): Transverse grid size.
        nr (int): Propagation grid size.
        n_background (float): Background refractive index.
        delta_n (float): Refractive index contrast for blobs.
        beta_n (float): Absorption contrast for blobs.
        n_blobs (int): Number of blobs to generate.
        seed (int): Random seed for reproducibility.
    """
    n_map = generate_empty_phantom(nz, nr, n_background)
    z_idx, x_idx = np.meshgrid(np.arange(nr), np.arange(nz))

    np.random.seed(seed)

    # Determine Blob Size Limits (Pixels)
    if "blob_r_range_um" in kwargs and "dx" in kwargs:
        # Physics-aware sizing
        r_min_um, r_max_um = kwargs["blob_r_range_um"]
        dx = kwargs["dx"]
        r_min = int(r_min_um / dx)
        r_max = int(r_max_um / dx)
    else:
        # Legacy pixel-relative sizing
        r_min = max(1, nz // 30)
        r_max = max(2, nz // 12)

    pad = nz // 8

    for _ in range(n_blobs):
        cx = np.random.randint(pad, nz - pad)
        cz = np.random.randint(0, nr)
        radius = np.random.randint(r_min, r_max + 1)

        dist = np.sqrt((x_idx - cx) ** 2 + (z_idx - cz) ** 2)
        mask = dist <= radius

        # Cosine Taper for soft edges
        taper = np.cos(np.pi * dist[mask] / (2 * radius))
        n_map[mask] += (delta_n + 1j * beta_n) * taper

    return n_map


def generate_gravel_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.0,
    delta_n: float = 0.2,
    beta_n: float = 0.02,
    n_blobs: int = 200,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Generates a phantom with hard-edged, high-contrast scatterers.

    Parameters:
        nz (int): Number of pixels in the transverse direction (rows).
        nr (int): Number of pixels in the propagation direction (columns).
        n_background (float): The refractive index value.
        delta_n (float): Refractive index contrast of the blobs.
        beta_n (float): Absorptive part of the refractive index of the blobs.
        n_blobs (int): Number of blobs to generate.
        seed (int): Random seed for reproducibility.
    """
    n_map = generate_empty_phantom(nz, nr, n_background)
    z_idx, x_idx = np.meshgrid(np.arange(nr), np.arange(nz))

    np.random.seed(seed)

    # Determine Particle Size Limits
    if "avg_grain_size_um" in kwargs and "dx" in kwargs:
        avg_r_um = kwargs["avg_grain_size_um"] / 2.0
        dx = kwargs["dx"]
        avg_r_px = int(avg_r_um / dx)
        r_min = max(1, int(avg_r_px * 0.5))
        r_max = max(2, int(avg_r_px * 1.5))
    else:
        r_min = max(1, nz // 100)
        r_max = max(2, nz // 40)

    pad = nz // 8

    for _ in range(n_blobs):
        cx = np.random.randint(pad, nz - pad)
        cz = np.random.randint(0, nr)
        radius = np.random.randint(r_min, r_max + 1)

        dist = np.sqrt((x_idx - cx) ** 2 + (z_idx - cz) ** 2)
        mask = dist <= radius

        # Randomized contrast per particle
        dn = delta_n * (0.8 + 0.4 * np.random.rand())
        bn = beta_n * (0.8 + 0.4 * np.random.rand())
        n_map[mask] += dn + 1j * bn

    return n_map


def generate_waveguide_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.0,
    delta_n: float = 0.1,
    beta_n: float = 0.0,
    width_um: float = 10.0,
    dx: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Generates a simple straight waveguide channel centered in the grid.

    Args:
        nz (int): Transverse grid size.
        nr (int): Propagation grid size.
        n_background (float): Cladding refractive index.
        delta_n (float): Core refractive index difference (n_core = n_bg + delta_n).
        beta_n (float): Core absorption.
        width_um (float): Width of the waveguide core in microns.
        dx (float): Pixel size in microns.
        **kwargs: Additional arguments passed by the runner are ignored.

    Returns:
        np.ndarray: The generated refractive index map.
    """
    n_map = generate_empty_phantom(nz, nr, n_background)
    width_pixels = int(width_um / dx)
    center_x = nz // 2
    start = center_x - width_pixels // 2
    end = center_x + width_pixels // 2
    n_map[start:end, :] += delta_n + 1j * beta_n
    return n_map


def generate_branching_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.33,
    delta_n: float = 0.03,
    initial_thickness: float = 5.0,
    branching_prob: float = 0.008,
    split_angle: float = 0.25,
    repulsion: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    Generates a 'root-like' or 'neuronal' branching network.

    Uses a walker algorithm where tips grow, branch probabilistically,
    and are repelled from the center axis to fill the volume.

    Args:
        nz (int): Transverse grid size.
        nr (int): Propagation grid size.
        n_background (float): Base refractive index.
        delta_n (float): Refractive index change of the branches.
        initial_thickness (float): Starting thickness of the root trunks (pixels).
        branching_prob (float): Probability per step of a branch splitting.
        split_angle (float): Angle (radians) between split branches.
        repulsion (float): Strength of the force pushing branches outward.
        **kwargs: Additional arguments (like 'dx') passed by the runner are ignored.

    Returns:
        np.ndarray: The generated refractive index map.
    """
    # Explicitly matches logic of original code
    n_map = np.zeros((nz, nr), dtype=float)
    center_x = nz / 2.0

    active_tips = []
    n_trunks = 3
    for i in range(n_trunks):
        angle_offset = (i - 1) * 0.05
        active_tips.append(
            {
                "x": center_x,
                "z": 0,
                "angle": angle_offset,
                "thick": initial_thickness,
                "generation": 0,
            }
        )

    step_size = 1.0

    while len(active_tips) > 0:
        new_tips = []
        for tip in active_tips:
            tip["z"] += step_size
            tip["x"] += np.sin(tip["angle"]) * step_size

            if 0 <= tip["x"] < nz and 0 <= tip["z"] < nr:
                ix, iz = int(tip["x"]), int(tip["z"])
                r = int(round(tip["thick"] / 2.0))
                x_start = max(0, ix - r)
                x_end = min(nz, ix + r + 1)
                n_map[x_start:x_end, iz] = 1.0

            dist_from_center = tip["x"] - center_x
            push_force = (
                np.sign(dist_from_center) * repulsion * (abs(dist_from_center) / nz)
            )
            tip["angle"] += push_force

            if (
                (tip["z"] > nr * 0.1)
                and (np.random.rand() < branching_prob)
                and (tip["z"] < nr * 0.9)
                and len(new_tips) < 50
            ):
                tip["angle"] -= split_angle / 2.0
                tip["thick"] = max(1.5, tip["thick"] * 0.7)
                tip["generation"] += 1

                new_branch = tip.copy()
                new_branch["angle"] += split_angle
                new_tips.append(new_branch)

            if tip["z"] < nr - 1 and 0 <= tip["x"] < nz and tip["thick"] >= 1.0:
                new_tips.append(tip)

        active_tips = new_tips

    n_map_smooth = gaussian_filter(n_map, sigma=initial_thickness / 2.5)

    if n_map_smooth.max() > 0:
        n_map_smooth = n_map_smooth / n_map_smooth.max() * delta_n

    # Add background and absorption
    n_map_final = n_background + n_map_smooth + 1j * (n_map_smooth * 0.05)
    return n_map_final


def generate_fiber_bundle_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.33,
    delta_n: float = 0.02,
    fiber_rad_um: float = 2.0,
    density: float = 0.3,
    waviness_um: float = 1.0,
    dx: float = 1.0,
    seed: int = 101,
    **kwargs,
) -> np.ndarray:
    """
    Generates a dense bundle of wavy fibers (e.g., muscle or nerve bundle).

    Fibers propagate in Z with a sinusoidal perturbation in X.

    Args:
        nz (int): Transverse grid size.
        nr (int): Propagation grid size.
        n_background (float): Base refractive index.
        delta_n (float): Refractive index change of fibers.
        fiber_rad_um (float): Radius of individual fibers.
        density (float): Fill density (approximate).
        waviness_um (float): Amplitude of the sinusoidal waviness.
        dx (float): Pixel size in microns.
        seed (int): Random seed.
        **kwargs: Additional arguments passed by the runner are ignored.

    Returns:
        np.ndarray: The generated refractive index map.
    """
    n_map = np.ones((nz, nr), dtype=np.complex128) * n_background
    z_coords_um = np.linspace(0, nr * dx, nr)
    physical_width_um = nz * dx
    n_fibers = int((physical_width_um * density) / (2 * fiber_rad_um))

    np.random.seed(seed)
    pad = nz // 10

    for _ in range(n_fibers):
        x_center_0 = np.random.randint(pad, nz - pad)
        period_um = np.random.uniform(50.0, 150.0)
        phase = np.random.uniform(0, 2 * np.pi)

        path_x_um = (x_center_0 * dx) + waviness_um * np.sin(
            2 * np.pi * z_coords_um / period_um + phase
        )
        path_x_idx = path_x_um / dx

        x_grid = np.arange(nz).reshape(-1, 1)
        dist = np.abs(x_grid - path_x_idx.reshape(1, -1))
        rad_pix = fiber_rad_um / dx

        dn = delta_n * (0.9 + 0.2 * np.random.rand())
        mask = dist <= rad_pix
        n_map[mask] += dn + 1j * (dn * 0.01)

    return n_map


def interpolate_to_coarse(n_map_fine: np.ndarray, n_steps_coarse: int) -> np.ndarray:
    """
    Downsamples the refractive index map in the propagation direction.

    Used to create coarse-stepped approximations for benchmarking.

    Args:
        n_map_fine (np.ndarray): Original high-resolution map.
        n_steps_coarse (int): Target number of steps in the propagation dimension.

    Returns:
        np.ndarray: The downsampled map.
    """
    nx, nz_fine = n_map_fine.shape
    zoom_factor = n_steps_coarse / nz_fine
    return zoom(n_map_fine, (1, zoom_factor), order=1)
