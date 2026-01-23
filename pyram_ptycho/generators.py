import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import j1
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
# 0. CONSTANTS & CONFIGURATION
# =============================================================================

# First root of Bessel function J1 (approx 3.8317)
# Used for Airy disk radius calculation
BESSEL_ROOT_J1 = 3.8317
EPSILON = 1e-12

# =============================================================================
# 1. UTILITIES & HELPERS
# =============================================================================


def get_random_generator(seed: Optional[int] = None) -> np.random.Generator:
    """
    Returns a numpy random generator.

    Using a local generator instead of the global `np.random` ensures
    reproducibility and thread-safety during parallel unit testing.
    """
    return np.random.default_rng(seed)


def get_grid_coords(nz: int, nr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates meshgrid coordinates for the simulation window.

    Args:
        nz (int): Number of pixels in the transverse (vertical) direction.
        nr (int): Number of pixels in the propagation (horizontal) direction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (z_idx, x_idx) coordinate grids.
        Note: z_idx corresponds to columns (propagation), x_idx to rows (transverse).
    """
    z_idx, x_idx = np.meshgrid(np.arange(nr), np.arange(nz))
    return z_idx, x_idx


def generate_empty_phantom(nz: int, nr: int, n_background: float = 1.0) -> np.ndarray:
    """
    Creates a base homogeneous refractive index map.

    Args:
        nz (int): Transverse pixels.
        nr (int): Propagation pixels.
        n_background (float): Base refractive index (real part).

    Returns:
        np.ndarray: Complex grid initialized to (n_background + 0j).
    """
    if nz <= 0 or nr <= 0:
        raise ValueError(f"Grid dimensions must be positive, got nz={nz}, nr={nr}")

    return np.ones((nz, nr), dtype=np.complex128) * n_background


def interpolate_to_coarse(n_map_fine: np.ndarray, n_steps_coarse: int) -> np.ndarray:
    """
    Downsamples the refractive index map along the propagation axis.

    Used to generate the low-resolution inputs for the Padé solver.

    Args:
        n_map_fine (np.ndarray): High-resolution input map (nz, nr_fine).
        n_steps_coarse (int): Target number of steps in propagation direction.

    Returns:
        np.ndarray: Resampled map with shape (nz, n_steps_coarse).
    """
    if n_steps_coarse <= 0:
        raise ValueError("Target coarse steps must be > 0")

    _, nz_fine = n_map_fine.shape
    zoom_factor = n_steps_coarse / nz_fine

    # Order 1 (bilinear) preserves features well enough for this physics without ringing artifacts
    return zoom(n_map_fine, (1, zoom_factor), order=1)


# =============================================================================
# 2. PROBE GENERATORS
# =============================================================================


def get_probe_field(
    coord: np.ndarray, center: float, diameter: float, focus: float, wavelength: float
) -> np.ndarray:
    """
    Calculates a 1D probe field (Airy disk-like) with optional quadratic curvature.

    The amplitude is modeled as 2*J1(v)/v (Airy pattern), and phase is quadratic
    (Fresnel approximation).

    Args:
        coord (np.ndarray): 1D array of transverse coordinates (e.g., microns).
        center (float): Center position of the beam on the coord axis.
        diameter (float): Aperture diameter defining the Airy disk size.
        focus (float): Focal distance.
                       - Positive: Converging beam.
                       - Negative: Diverging beam.
                       - Zero/Near-zero: Plane wave (collimated).
        wavelength (float): Radiation wavelength in same units as coord.

    Returns:
        np.ndarray: Complex 1D field array.
    """
    if diameter <= 0:
        raise ValueError("Probe diameter must be positive.")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")

    k0 = 2 * np.pi / wavelength
    delta_x = coord - center
    r = np.abs(delta_x)
    radius = diameter / 2.0

    # Safe division for Bessel function
    r_safe = np.maximum(r, EPSILON)
    v = BESSEL_ROOT_J1 * r_safe / radius
    amplitude = 2.0 * j1(v) / v

    # Fix singularity at r=0 (limit of 2*J1(v)/v as v->0 is 1)
    amplitude[r < EPSILON] = 1.0

    if abs(focus) < EPSILON:
        phase = 1.0
    else:
        # Fresnel quadratic phase approximation: phi = k * x^2 / (2R)
        # Note: R = -focus convention implies focus>0 is converging (concave phase)
        R = -focus
        phase = np.exp(1j * k0 * delta_x**2 / (2.0 * R))

    return amplitude * phase


# =============================================================================
# 3. SAMPLE GENERATORS
# =============================================================================


def generate_blob_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.0,
    delta_n: float = 0.01,
    beta_n: float = 0.01,
    n_blobs: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a phantom with soft, cosine-tapered blobs.

    Useful for testing weak scattering in continuous media.

    Args:
        nz, nr (int): Grid dimensions.
        n_background (float): Base refractive index.
        delta_n (float): Peak real refractive index change per blob.
        beta_n (float): Peak imaginary (absorption) index change.
        n_blobs (int): Number of blobs to scatter.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Complex refractive index map.
    """
    rng = get_random_generator(seed)
    n_map = generate_empty_phantom(nz, nr, n_background)
    z_idx, x_idx = get_grid_coords(nz, nr)

    pad = nz // 8

    for _ in range(n_blobs):
        cx = rng.integers(pad, nz - pad)
        cz = rng.integers(0, nr)
        radius = rng.integers(nz // 30, nz // 12)

        # Boolean mask for the circular region (optimization: check square dist first)
        dist_sq = (x_idx - cx) ** 2 + (z_idx - cz) ** 2
        mask = dist_sq <= radius**2

        # Calculate soft taper only inside the blob
        dist = np.sqrt(dist_sq[mask])
        taper = np.cos(np.pi * dist / (2 * radius))
        n_map[mask] += (delta_n + 1j * beta_n) * taper

    return n_map


def generate_gravel_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.0,
    delta_n: float = 0.2,
    beta_n: float = 0.02,
    n_blobs: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a phantom with hard-edged, high-contrast scatterers ("gravel").

    Useful for testing stability with high-contrast, discontinuous media.

    Args:
        nz, nr (int): Grid dimensions.
        delta_n (float): Base real index contrast (randomized per particle).
        beta_n (float): Base absorption contrast (randomized per particle).
        n_blobs (int): Number of particles.

    Returns:
        np.ndarray: Complex refractive index map.
    """
    rng = get_random_generator(seed)
    n_map = generate_empty_phantom(nz, nr, n_background)
    z_idx, x_idx = get_grid_coords(nz, nr)

    pad = nz // 8

    for _ in range(n_blobs):
        cx = rng.integers(pad, nz - pad)
        cz = rng.integers(0, nr)
        radius = rng.integers(nz // 100, nz // 40)  # Smaller, sharper features

        mask = (x_idx - cx) ** 2 + (z_idx - cz) ** 2 <= radius**2

        # Randomize contrast to simulate varied material composition
        dn = delta_n * (0.8 + 0.4 * rng.random())
        bn = beta_n * (0.8 + 0.4 * rng.random())
        n_map[mask] += dn + 1j * bn

    return n_map


def generate_waveguide_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.0,
    delta_n: float = 0.1,
    beta_n: float = 0.001,
    width_um: float = 10.0,
    dx: float = 1.0,
) -> np.ndarray:
    """
    Generates a single horizontal waveguide core.

    Deterministic function (no random seed needed).

    Args:
        nz, nr (int): Grid dimensions.
        width_um (float): Core width in microns.
        dx (float): Pixel size in microns.
    """
    n_map = generate_empty_phantom(nz, nr, n_background)

    width_pixels = int(width_um / dx)
    if width_pixels == 0:
        width_pixels = 1  # Ensure at least 1 pixel width

    center_x = nz // 2
    start = max(0, center_x - width_pixels // 2)
    end = min(nz, center_x + width_pixels // 2)

    n_map[start:end, :] += delta_n + 1j * beta_n
    return n_map


def generate_grin_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.5,
    delta_n: float = 0.05,
    radius_um: float = 40.0,
    dx: float = 1.0,
) -> np.ndarray:
    """
    Generates a Parabolic Gradient Index (GRIN) Rod/Lens.

    Profile: n(r) = n0 - delta_n * (r/R)^2
    """
    n_map = generate_empty_phantom(nz, nr, n_background)
    center_x = nz // 2
    R_pixels = radius_um / dx

    # 1D radial profile computation across the transverse aperture
    x_idx = np.arange(nz)
    r_pixels = np.abs(x_idx - center_x)

    # Vectorized mask and profile application
    mask = r_pixels <= R_pixels
    profile = n_background - delta_n * (r_pixels[mask] / R_pixels) ** 2

    # Apply profile to all Z columns (broadcasting)
    # n_map[mask, :] (N_mask, nr) = profile[:, None] (N_mask, 1)
    n_map[mask, :] = profile[:, None]

    return n_map


def generate_branching_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.33,
    delta_n: float = 0.03,
    beta_n: float = 0.001,
    initial_thickness: float = 5.0,
    branching_prob: float = 0.008,
    split_angle: float = 0.25,
    repulsion: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a 'root-like' spreading network simulation.

    Uses a random walker agent model with repulsive forces and branching logic.

    Args:
        nz, nr (int): Grid dimensions.
        initial_thickness (float): Thickness of root fibers in pixels.
        branching_prob (float): Probability per step of a fiber splitting.
        split_angle (float): Angle (radians) between new branches.
        repulsion (float): Strength of force pushing fibers away from center axis.
        seed (int, optional): Random seed.

    Returns:
        np.ndarray: Complex refractive index map.
    """
    rng = get_random_generator(seed)
    # Temporary high-precision float map for density accumulation
    n_map_density = np.zeros((nz, nr), dtype=float)
    center_x = nz / 2.0

    # --- Initialization ---
    # Start with 3 main trunks clustered near the center
    active_tips: List[Dict[str, Any]] = []
    n_trunks = 3
    for i in range(n_trunks):
        angle_offset = (i - 1) * 0.05
        active_tips.append(
            {
                "x": center_x,
                "z": 0.0,
                "angle": angle_offset,
                "thick": initial_thickness,
                "generation": 0,
            }
        )

    step_size = 1.0

    # --- Simulation Loop ---
    while len(active_tips) > 0:
        new_tips = []
        for tip in active_tips:
            # 1. Update Position
            tip["z"] += step_size
            tip["x"] += np.sin(tip["angle"]) * step_size

            # 2. Draw Tip
            # Check bounds before drawing
            if 0 <= tip["x"] < nz and 0 <= tip["z"] < nr:
                ix, iz = int(tip["x"]), int(tip["z"])
                r = int(round(tip["thick"] / 2.0))

                x_start = max(0, ix - r)
                x_end = min(nz, ix + r + 1)

                # Draw "density" (1.0) into the map
                n_map_density[x_start:x_end, iz] = 1.0

            # 3. Dynamics (Repulsion)
            # Push fibers away from the center axis to ensure volume filling
            dist_from_center = tip["x"] - center_x
            push_force = (
                np.sign(dist_from_center) * repulsion * (abs(dist_from_center) / nz)
            )
            tip["angle"] += push_force

            # 4. Branching Logic
            # Only branch in the middle 80% of sample to avoid edge artifacts
            # Limit total active tips to 50 to prevent exponential explosion
            if (nr * 0.1 < tip["z"] < nr * 0.9) and (len(new_tips) < 50):
                if rng.random() < branching_prob:
                    # Branch A (Left + Thinning)
                    tip["angle"] -= split_angle / 2.0
                    tip["thick"] = max(1.5, tip["thick"] * 0.7)
                    tip["generation"] += 1

                    # Branch B (Right + New)
                    new_branch = tip.copy()
                    new_branch["angle"] += split_angle
                    new_tips.append(new_branch)

            # 5. Survival Check
            # Kill fibers that hit the wall or get too thin
            if tip["z"] < nr - 1 and 0 <= tip["x"] < nz and tip["thick"] >= 1.0:
                new_tips.append(tip)

        active_tips = new_tips

    # --- Post-Processing ---
    # Smooth the binary paths into realistic, continuous tubes
    n_map_smooth = gaussian_filter(n_map_density, sigma=initial_thickness / 2.5)

    # Normalize density to max at 1.0
    if n_map_smooth.max() > EPSILON:
        n_map_smooth /= n_map_smooth.max()

    # Apply Physical Properties (Refractive Index + Absorption)
    n_map = generate_empty_phantom(nz, nr, n_background)
    n_map += n_map_smooth * (delta_n + 1j * beta_n)

    return n_map


def generate_fiber_bundle_phantom(
    nz: int,
    nr: int,
    n_background: float = 1.33,
    delta_n: float = 0.02,
    beta_n: float = 0.01,
    fiber_rad_um: float = 2.0,
    density: float = 0.3,
    waviness_um: float = 1.0,
    dx: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulates a dense bundle of wavy fibers (e.g. muscle, nerve bundle).

    Args:
        nz, nr (int): Grid dimensions.
        fiber_rad_um (float): Radius of each fiber in microns.
        density (float): Approximate area density of fibers (0 < density < 1).
        waviness_um (float): Amplitude of sine wave perturbation.
        dx (float): Pixel size in microns.
    """
    rng = get_random_generator(seed)
    n_map = generate_empty_phantom(nz, nr, n_background)

    z_coords_um = np.linspace(0, nr * dx, nr)
    physical_width_um = nz * dx

    # Calculate number of fibers based on density fill factor
    n_fibers = int((physical_width_um * density) / (2 * fiber_rad_um))
    if n_fibers < 1:
        n_fibers = 1

    pad = nz // 10
    rad_pix = fiber_rad_um / dx

    # Pre-calculate grids to avoid re-allocating inside loop
    x_grid = np.arange(nz).reshape(-1, 1)

    for _ in range(n_fibers):
        x_center_0 = rng.integers(pad, nz - pad)
        period_um = rng.uniform(50.0, 150.0)
        phase = rng.uniform(0, 2 * np.pi)

        # Calculate fiber path across all Z
        path_x_um = (x_center_0 * dx) + waviness_um * np.sin(
            2 * np.pi * z_coords_um / period_um + phase
        )
        path_x_idx = path_x_um / dx

        # Vectorized distance calculation
        # (nz, nr) = (nz, 1) - (1, nr)
        dist = np.abs(x_grid - path_x_idx.reshape(1, -1))

        # Random contrast variation per fiber
        var_factor = 0.9 + 0.2 * rng.random()
        dn = delta_n * var_factor

        mask = dist <= rad_pix
        n_map[mask] += dn + 1j * (dn * beta_n)

    return n_map
