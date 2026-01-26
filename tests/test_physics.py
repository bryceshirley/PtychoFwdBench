import warnings
import pytest
import numpy as np
from typing import Dict, Any
from ptycho_fwd_bench.physics import (
    parse_simulation_parameters,
    validate_sampling_conditions,
)

# =============================================================================
# 1. PARSING TESTS
# =============================================================================


@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """Returns a standard valid configuration dictionary."""
    return {
        "physics": {"wavelength_um": 1.0, "probe_dia_um": 2.0, "probe_focus_um": 10.0},
        "grid": {"n_physical": 100, "n_padding": 25, "physical_width_um": 10.0},
        "sample": {
            "type": "TEST_SAMPLE",
            "thickness_um": 5.0,
            "params": {"param_a": 1},
            "ground_truth": {"solver": "PADE"},
        },
    }


def test_parse_simulation_parameters_values(valid_config):
    """Test that values are extracted and derived correctly."""
    params = parse_simulation_parameters(valid_config)

    # Check Direct Conversions (um -> m)
    # Round values to avoid floating point precision issues
    assert round(params["wavelength"], 12) == 1.0e-6
    assert round(params["probe_dia"], 12) == 2.0e-6
    assert round(params["sample_thickness"], 12) == 5.0e-6

    # Check Grid Derivations
    # dx = width / n_physical = 10um / 100 = 0.1um
    expected_dx = 0.1e-6
    assert np.isclose(params["dx"], expected_dx)

    # n_total = n_physical + 2 * padding = 100 + 50 = 150
    assert params["n_total"] == 150

    # total_width = n_total * dx = 150 * 0.1um = 15um
    assert np.isclose(params["total_width"], 15.0e-6)


def test_parse_simulation_parameters_structure(valid_config):
    """Test that optional dictionaries are retrieved correctly."""
    params = parse_simulation_parameters(valid_config)

    assert params["sample_type"] == "TEST_SAMPLE"
    assert params["sample_params"] == {"param_a": 1}
    assert params["ground_truth_cfg"] == {"solver": "PADE"}


def test_parse_defaults_if_missing_optional(valid_config):
    """Test behavior when optional keys (params, ground_truth) are missing."""
    del valid_config["sample"]["params"]
    del valid_config["sample"]["ground_truth"]

    params = parse_simulation_parameters(valid_config)

    assert params["sample_params"] == {}
    assert params["ground_truth_cfg"] == {}


# =============================================================================
# 2. VALIDATION / PHYSICS TESTS
# =============================================================================
def test_validate_sampling_safe():
    """Test a 'safe' condition (No warnings expected)."""
    params = {
        "wavelength": 1e-6,
        "dx": 0.6e-6,
        "probe_dia": 10e-6,
        "probe_focus": 500e-6,
    }

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        validate_sampling_conditions(params)
        # assert no warnings were caught
        assert len(record) == 0, (
            f"Expected 0 warnings, got: {[str(r.message) for r in record]}"
        )


def test_validate_sampling_aliased():
    """Test a known aliasing condition (Warning expected)."""
    params = {
        "wavelength": 1e-6,
        "dx": 2.0e-6,
        "probe_dia": 10e-6,
        "probe_focus": 10e-6,
    }

    with pytest.warns(UserWarning, match="CRITICAL WARNING"):
        validate_sampling_conditions(params)


def test_validate_sampling_collimated():
    """Test collimated beam (No warnings expected)."""
    params = {"wavelength": 1e-6, "dx": 1.0e-6, "probe_dia": 10e-6, "probe_focus": 0.0}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        validate_sampling_conditions(params)
        assert len(record) == 0


def test_validate_super_resolution_grid():
    """Test case where grid is finer than half-wavelength (Invalid arcsin)."""
    params = {"wavelength": 1.0, "dx": 0.1, "probe_dia": 1.0, "probe_focus": 10.0}

    with np.errstate(invalid="ignore"):  # Ignore numpy's own invalid value warning
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_sampling_conditions(params)
            assert len(record) == 0
