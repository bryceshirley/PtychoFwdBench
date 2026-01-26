import pytest
import numpy as np
from ptycho_fwd_bench.ptycho_solvers import (
    create_solver,
    MultisliceSolver,
    PtychoPadeSolver,
)

# =============================================================================
# 1. FIXTURES & MOCKS
# =============================================================================


@pytest.fixture
def basic_setup():
    """Provides standard grid and physics parameters for testing."""
    nz, nr = 64, 10
    n_map = np.ones((nz, nr), dtype=np.complex128)  # Vacuum

    sim_params = {
        "dx": 0.1,  # 100nm pixel
        "wavelength": 0.0001,  # 1 Angstrom
        "probe_dia": 1.0,
        "probe_focus": 0.0,  # Collimated
    }

    dz = 0.1  # 100nm step

    return n_map, sim_params, dz


@pytest.fixture
def psi_init(basic_setup):
    """Provides a simple initial probe."""
    n_map, _, _ = basic_setup
    psi = np.zeros(n_map.shape[0], dtype=np.complex128)
    psi[n_map.shape[0] // 2] = 1.0
    return psi


# =============================================================================
# 2. FACTORY TESTS (create_solver)
# =============================================================================


def test_create_solver_pade(basic_setup):
    n_map, sim_params, dz = basic_setup

    solver = create_solver(
        solver_type="PADE",
        solver_params={"pade_order": 4},
        n_map=n_map,
        sim_params=sim_params,
        dz=dz,
    )

    assert isinstance(solver, PtychoPadeSolver)
    assert solver.pade_order == 4


def test_create_solver_multislice_aliases(basic_setup):
    n_map, sim_params, dz = basic_setup

    # Test 'MULTISLICE'
    s1 = create_solver("MULTISLICE", {}, n_map, sim_params, dz)
    assert isinstance(s1, MultisliceSolver)

    # Test 'MS' alias
    s2 = create_solver("MS", {}, n_map, sim_params, dz)
    assert isinstance(s2, MultisliceSolver)


def test_create_solver_unknown_raises(basic_setup):
    n_map, sim_params, dz = basic_setup

    with pytest.raises(ValueError, match="Unknown solver type"):
        create_solver("MAGIC_SOLVER", {}, n_map, sim_params, dz)


def test_create_solver_spectral_pade_raises_not_implemented(basic_setup):
    n_map, sim_params, dz = basic_setup
    solver = create_solver("SPECTRAL_PADE", {}, n_map, sim_params, dz)

    with pytest.raises(NotImplementedError):
        solver.run()


# =============================================================================
# 3. MULTISLICE SOLVER TESTS
# =============================================================================


def test_multislice_initialization(basic_setup):
    n_map, sim_params, dz = basic_setup

    solver = MultisliceSolver(
        n_map,
        sim_params["dx"],
        sim_params["wavelength"],
        sim_params["probe_dia"],
        sim_params["probe_focus"],
        transform_type="FFT",
        symmetric=True,
        dz=dz,
    )

    assert solver.transform_type == "FFT"
    assert solver.symmetric is True
    # Check derived prop
    assert solver.k0 == 2 * np.pi / sim_params["wavelength"]


def test_multislice_run_execution(basic_setup, psi_init):
    """Test that it runs and returns an exit wave of correct shape."""
    n_map, sim_params, dz = basic_setup

    solver = MultisliceSolver(
        n_map,
        sim_params["dx"],
        sim_params["wavelength"],
        sim_params["probe_dia"],
        sim_params["probe_focus"],
        dz=dz,
    )

    # Run
    solver.run(psi_init=psi_init)

    # Check Output
    exit_wave = solver.get_exit_wave()
    assert exit_wave.shape == psi_init.shape
    assert not np.any(np.isnan(exit_wave))

    # Check cropping
    crop_size = 10
    cropped = solver.get_exit_wave(n_crop=crop_size)
    assert cropped.shape == (crop_size,)


def test_multislice_beam_storage(basic_setup, psi_init):
    """Test that beam history is saved when requested."""
    n_map, sim_params, dz = basic_setup

    solver = MultisliceSolver(
        n_map,
        sim_params["dx"],
        sim_params["wavelength"],
        sim_params["probe_dia"],
        sim_params["probe_focus"],
        dz=dz,
        store_beam=True,
    )

    solver.run(psi_init=psi_init)

    beam = solver.get_beam_field()
    assert beam is not None
    # Shape should be (N_x, N_steps)
    assert beam.shape == (n_map.shape[0], n_map.shape[1])


def test_multislice_dst_kernel_cache(basic_setup):
    """Ensure kernel caching works (calling _get_propagation_kernel twice)."""
    n_map, sim_params, dz = basic_setup
    solver = MultisliceSolver(
        n_map,
        sim_params["dx"],
        sim_params["wavelength"],
        sim_params["probe_dia"],
        sim_params["probe_focus"],
        transform_type="DST",
    )

    k1 = solver._get_propagation_kernel(dz)
    k2 = solver._get_propagation_kernel(dz)

    # Should be exact same object in memory
    assert k1 is k2
    assert len(solver._kernel_cache) == 1


# =============================================================================
# 4. PADE SOLVER TESTS (Mocking PyRAM)
# =============================================================================


def test_pade_solver_initialization(basic_setup):
    n_map, sim_params, dz = basic_setup

    solver = PtychoPadeSolver(
        n_map=n_map,
        dx=sim_params["dx"],
        wavelength=sim_params["wavelength"],
        probe_dia=sim_params["probe_dia"],
        probe_focus=sim_params["probe_focus"],
        pade_order=2,
        dz=dz,
    )

    # FIX: Check stored attribute
    assert solver.pade_order == 2
    assert solver.nz == 64


def test_pade_solver_run_and_inject(basic_setup, psi_init):
    n_map, sim_params, dz = basic_setup

    solver = PtychoPadeSolver(
        n_map=n_map,
        dx=sim_params["dx"],
        wavelength=sim_params["wavelength"],
        probe_dia=sim_params["probe_dia"],
        probe_focus=sim_params["probe_focus"],
        pade_order=2,
        dz=dz,
    )

    solver.run(psi_init=psi_init)

    exit_wave = solver.get_exit_wave()

    # PyRAM internal grid is 66 (with boundary), get_exit_wave crops it back to 64.
    assert exit_wave.shape[0] == 64
    assert not np.any(np.isnan(exit_wave))


def test_pade_solver_padding_logic(basic_setup, psi_init):
    """Ensure psi_init is padded before being passed to PyRAM."""
    n_map, sim_params, dz = basic_setup

    solver = PtychoPadeSolver(
        n_map=n_map,
        dx=sim_params["dx"],
        wavelength=sim_params["wavelength"],
        probe_dia=sim_params["probe_dia"],
        probe_focus=sim_params["probe_focus"],
        pade_order=2,
        dz=dz,
    )

    # Spy on the internal attribute before run
    solver.run(psi_init=psi_init)

    # _external_psi_init should be padded by (1,1)
    assert len(solver._external_psi_init) == len(psi_init) + 2
    assert solver._external_psi_init[0] == 0  # Dirchlet BC
    assert solver._external_psi_init[-1] == 0
