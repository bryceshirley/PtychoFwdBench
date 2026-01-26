import pytest
import numpy as np
from ptycho_fwd_bench.generators import (
    get_probe_field,
    generate_empty_phantom,
    generate_blob_phantom,
    generate_gravel_phantom,
    generate_waveguide_phantom,
    generate_branching_phantom,
    generate_fiber_bundle_phantom,
    interpolate_to_coarse,
)

# =============================================================================
# 1. PROBE GENERATOR TESTS
# =============================================================================


def test_get_probe_field_shape_and_type():
    """Ensure probe field matches grid size and returns complex array."""
    n_points = 128
    dx = 0.1
    x_grid = np.arange(n_points) * dx
    center = (n_points * dx) / 2.0

    psi = get_probe_field(
        coord=x_grid, center=center, diameter=2.0, focus=-100.0, wavelength=0.000124
    )

    assert psi.shape == (n_points,)
    assert np.iscomplexobj(psi)
    assert not np.any(np.isnan(psi))


def test_probe_collimated():
    """Test focus=0 producing a flat phase (collimated beam)."""
    n_points = 64
    x_grid = np.arange(n_points)

    # Focus = 0.0 -> Collimated
    psi = get_probe_field(x_grid, 32, 10, 0.0, 0.5)

    # Phase should be uniform (1.0 + 0j) where amplitude > 0
    center_val = psi[32]
    assert np.isclose(np.angle(center_val), 0.0)
    assert np.abs(center_val) > 0


def test_probe_curvature_direction():
    """Test that convergent and divergent probes have opposite phase curvatures."""
    x_grid = np.arange(64)
    # Divergent (Source behind) -> Focus negative
    psi_div = get_probe_field(x_grid, 32, 10, -100.0, 0.5)
    # Convergent (Focus in front) -> Focus positive
    psi_conv = get_probe_field(x_grid, 32, 10, 100.0, 0.5)

    # Unwrap phase to check curvature
    phase_div = np.unwrap(np.angle(psi_div))
    phase_conv = np.unwrap(np.angle(psi_conv))

    # Check curvature sign by looking at second derivative or simply center vs edge difference
    # For spherical waves, curvature signs should be opposite
    # (Checking indices near center where amplitude is high)
    curve_div = phase_div[32] - phase_div[25]
    curve_conv = phase_conv[32] - phase_conv[25]

    # They should have opposite signs (approximate check)
    assert np.sign(curve_div) != np.sign(curve_conv)


# =============================================================================
# 2. PHANTOM GENERATOR TESTS
# =============================================================================


@pytest.fixture
def grid_params():
    return {"nz": 100, "nr": 50, "dx": 0.1}


def test_generate_empty_phantom(grid_params):
    bg = 1.33
    n_map = generate_empty_phantom(
        grid_params["nz"], grid_params["nr"], n_background=bg
    )

    assert n_map.shape == (grid_params["nz"], grid_params["nr"])
    assert np.allclose(n_map, bg)
    assert np.iscomplexobj(n_map)


def test_generate_blob_phantom_defaults(grid_params):
    """Test basic blob generation with defaults."""
    n_map = generate_blob_phantom(
        grid_params["nz"], grid_params["nr"], n_background=1.0, n_blobs=10
    )
    assert n_map.shape == (grid_params["nz"], grid_params["nr"])
    # Should have values other than background
    assert not np.allclose(n_map, 1.0)


def test_generate_blob_phantom_physics_sizing(grid_params):
    """Test that providing dx and physical ranges works without crashing."""
    n_map = generate_blob_phantom(
        grid_params["nz"],
        grid_params["nr"],
        n_background=1.0,
        dx=grid_params["dx"],
        blob_r_range_um=[0.5, 1.0],  # 5 to 10 pixels
        n_blobs=5,
    )
    assert n_map.shape == (grid_params["nz"], grid_params["nr"])


def test_generate_gravel_phantom_reproducibility(grid_params):
    """Test that seed produces identical results."""
    kwargs = {
        "nz": grid_params["nz"],
        "nr": grid_params["nr"],
        "n_blobs": 20,
        "seed": 42,
    }
    map1 = generate_gravel_phantom(**kwargs)
    map2 = generate_gravel_phantom(**kwargs)

    assert np.array_equal(map1, map2)


def test_generate_waveguide_phantom(grid_params):
    """Check that the waveguide core has higher index than cladding."""
    bg = 1.0
    delta = 0.1
    width_um = 2.0  # 20 pixels with dx=0.1

    n_map = generate_waveguide_phantom(
        grid_params["nz"],
        grid_params["nr"],
        n_background=bg,
        delta_n=delta,
        width_um=width_um,
        dx=grid_params["dx"],
    )

    center = grid_params["nz"] // 2
    edge = 0

    # Core check (real part)
    assert np.isclose(n_map[center, 0].real, bg + delta)
    # Cladding check
    assert np.isclose(n_map[edge, 0].real, bg)


def test_generate_branching_phantom_structure(grid_params):
    """Test basic structure generation."""
    n_map = generate_branching_phantom(
        grid_params["nz"], grid_params["nr"], initial_thickness=2.0
    )
    assert n_map.shape == (grid_params["nz"], grid_params["nr"])
    assert np.iscomplexobj(n_map)


def test_generate_fiber_bundle_phantom(grid_params):
    """Test fiber bundle generation."""
    n_map = generate_fiber_bundle_phantom(
        grid_params["nz"],
        grid_params["nr"],
        n_background=1.0,
        delta_n=0.1,
        fiber_rad_um=1.0,  # 10 pixels
        dx=grid_params["dx"],
    )
    assert n_map.shape == (grid_params["nz"], grid_params["nr"])


# =============================================================================
# 3. UTILITY TESTS
# =============================================================================


def test_interpolate_to_coarse():
    """Test downsampling interpolation."""
    # Create a 100x100 map
    nz, nr_fine = 100, 100
    n_map_fine = np.ones((nz, nr_fine))

    # Downsample Z to 10 steps
    nr_coarse = 10
    n_map_coarse = interpolate_to_coarse(n_map_fine, nr_coarse)

    assert n_map_coarse.shape == (nz, nr_coarse)
    # Values should be preserved for a constant map
    assert np.allclose(n_map_coarse, 1.0)
