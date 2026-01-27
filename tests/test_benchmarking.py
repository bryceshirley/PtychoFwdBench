import pytest
import numpy as np
from unittest import mock
from unittest.mock import MagicMock, patch

import ptycho_fwd_bench.benchmarking as benchmarking

# =============================================================================
# FIXTURES (Standard Setup Data)
# =============================================================================


@pytest.fixture
def mock_sim_params():
    """Standard simulation parameters dictionary."""
    return {
        "sample_type": "BLOBS",
        "n_total": 100,
        "n_pad": 10,
        "dx": 1e-6,  # 1 micron
        "wavelength": 0.5e-6,
        "total_width": 100e-6,
        "probe_dia": 10e-6,
        "probe_focus": 0,
        "sample_thickness": 10e-6,
        "physical_width": 100e-6,
        "sample_params": {"blob_r_range_um": [1, 2]},
        "ground_truth_cfg": {
            "n_prop_fine": 100,
            "solver_type": "PADE",
            "solver_params": {"order": 2},
        },
    }


@pytest.fixture
def mock_config(mock_sim_params):
    """Standard full config dictionary."""
    return {
        "experiment": {"name": "TEST_EXP", "description": "Test"},
        "sample": {
            "type": "BLOBS",
            "ground_truth": {
                "n_prop_fine": 100,
                "solver_type": "PADE",
                "solver_params": {"order": 2},
            },
        },
        "solvers": [{"name": "Solver A", "type": "TEST_SOLVER", "solver_params": {}}],
        "benchmark": {"step_counts": [10, 20], "save_beam_idx": 0},
    }


# =============================================================================
# TESTS
# =============================================================================


class TestSimulationInputs:
    # This ensures the code looks up our mock when it checks the dictionary.
    @patch.dict("ptycho_fwd_bench.benchmarking.GENERATOR_MAP", clear=False)
    @patch("ptycho_fwd_bench.benchmarking.get_probe_field")
    def test_generate_simulation_inputs_success(self, mock_probe, mock_sim_params):
        """Test that inputs are generated using correct parameters."""

        # 1. Setup the Mock Generator inside the patched dict
        mock_gen_func = MagicMock()
        # Update the dict to point "BLOBS" to our mock
        benchmarking.GENERATOR_MAP["BLOBS"] = mock_gen_func

        # 2. Setup Return Values
        expected_n_map = np.zeros((100, 100))  # The phantom
        expected_psi = np.ones(100)  # The probe

        mock_gen_func.return_value = expected_n_map
        mock_probe.return_value = expected_psi

        # 3. Execute
        n_map, psi_0 = benchmarking.generate_simulation_inputs(mock_sim_params)

        # 4. Assertions
        assert n_map is expected_n_map
        assert psi_0 is expected_psi

        # Verify the generator was called with microns (1e-6 m -> 1.0 um)
        mock_gen_func.assert_called_once()
        call_kwargs = mock_gen_func.call_args.kwargs
        assert call_kwargs["dx"] == pytest.approx(1.0)


class TestComputeGroundTruth:
    @patch("ptycho_fwd_bench.benchmarking.create_solver")
    def test_compute_ground_truth_flow(self, mock_create, mock_sim_params):
        """Test that the solver is initialized, run, and results extracted."""

        # Setup Mock Solver
        mock_solver_instance = MagicMock()
        mock_create.return_value = mock_solver_instance

        # Mock Solver Returns
        # FIX: Return the FULL size (100) so the code can crop 10 pixels off each side
        full_wave = np.ones(100)
        mock_solver_instance.get_exit_wave.return_value = full_wave
        mock_solver_instance.get_beam_field.return_value = np.zeros(
            (100, 5)
        )  # <--- Changed from 80 to 100

        # Inputs
        n_map = np.zeros((100, 100))
        psi_0 = np.ones(100)

        # Execute
        psi_gt, beam_gt = benchmarking.compute_ground_truth(
            n_map, psi_0, mock_sim_params
        )

        assert mock_create.call_count == 1
        args, kwargs = mock_create.call_args

        # args[0]: type, args[1]: params, args[2]: n_map, args[3]: sim_params, args[4]: dz
        assert args[0] == "PADE"
        assert args[1] == {"order": 2}
        assert args[2] is n_map

        assert kwargs["save_beam"] is True

        # Verify run was called
        mock_solver_instance.run.assert_called_with(psi_init=psi_0)
        assert args[4] == pytest.approx(0.1e-6)

        # Verify Crop Logic
        # (100 total - 10 pad_left - 10 pad_right = 80)
        assert psi_gt.shape == (80,)
        assert beam_gt.shape == (80, 5)


class TestBenchmarkLoop:
    @patch("ptycho_fwd_bench.benchmarking.plotters")
    @patch("ptycho_fwd_bench.benchmarking.create_solver")
    @patch("ptycho_fwd_bench.benchmarking.interpolate_to_coarse")
    def test_run_benchmark_loop(
        self, mock_interp, mock_create, mock_plotters, mock_config, mock_sim_params
    ):
        """Test the main loop: iteration over steps, running solvers, and plotting."""

        # Setup Data
        n_map_fine = np.zeros((100, 100))
        psi_0 = np.ones(100)
        # psi_gt must match cropped size (80)
        psi_gt = np.ones(80)
        beam_gt = None
        out_dir = "/tmp/test"

        # Mock Solver Behavior
        mock_solver = MagicMock()
        mock_create.return_value = mock_solver
        # Return a wave that matches psi_gt so error is 0
        mock_solver.get_exit_wave.return_value = np.ones(100)  # Pre-crop size

        # Mock Interpolation
        mock_interp.return_value = n_map_fine

        # Execute
        benchmarking.run_benchmark_loop(
            mock_config, mock_sim_params, n_map_fine, psi_0, psi_gt, out_dir, beam_gt
        )

        # Assertions
        # 1. Check solvers were created
        # Logic: 2 step counts * 1 solver = 2 calls
        assert mock_create.call_count == 2

        # 2. Check Plotting calls
        mock_plotters.plot_fine_vs_coarse.assert_called_once()
        mock_plotters.plot_exit_wave_comparison.assert_called_once()
        mock_plotters.plot_convergence_metrics.assert_called_once()


class TestFullBenchmarkOrchestrator:
    @patch(
        "builtins.open",
        new_callable=mock.mock_open,
        read_data="experiment:\n  name: TEST",
    )
    @patch("ptycho_fwd_bench.benchmarking.yaml.safe_load")
    @patch("ptycho_fwd_bench.benchmarking.setup_output_directory")
    @patch("ptycho_fwd_bench.benchmarking.setup_logging")
    @patch("ptycho_fwd_bench.benchmarking.parse_simulation_parameters")
    @patch("ptycho_fwd_bench.benchmarking.validate_sampling_conditions")
    @patch("ptycho_fwd_bench.benchmarking.generate_simulation_inputs")
    @patch("ptycho_fwd_bench.benchmarking.compute_ground_truth")
    @patch("ptycho_fwd_bench.benchmarking.run_benchmark_loop")
    @patch(
        "ptycho_fwd_bench.benchmarking.save_ground_truth"
    )  # Optional, mocked just in case
    def test_run_full_benchmark_compute_mode(
        self,
        mock_save,
        mock_loop,
        mock_compute,
        mock_gen,
        mock_valid,
        mock_parse,
        mock_log,
        mock_dir,
        mock_yaml,
        mock_file,
        mock_config,
        mock_sim_params,
    ):
        """Test the full flow where Ground Truth is COMPUTED."""

        # Setup
        mock_yaml.return_value = mock_config
        mock_parse.return_value = mock_sim_params
        mock_dir.return_value = "/tmp/out"

        mock_gen.return_value = ("n_map", "psi_0")
        mock_compute.return_value = ("psi_gt", "beam_gt")

        # Execute
        benchmarking.run_full_benchmark("dummy_config.yml")

        # Assertions
        mock_gen.assert_called_once()  # Should generate inputs
        mock_compute.assert_called_once()  # Should compute GT
        mock_loop.assert_called_once()  # Should run loop
        mock_save.assert_not_called()  # Config didn't have save_path set in mock_config fixture

    @patch("builtins.open", new_callable=mock.mock_open, read_data="experiment: TEST")
    @patch("ptycho_fwd_bench.benchmarking.yaml.safe_load")
    @patch("ptycho_fwd_bench.benchmarking.setup_output_directory")
    @patch("ptycho_fwd_bench.benchmarking.setup_logging")
    @patch("ptycho_fwd_bench.benchmarking.parse_simulation_parameters")
    @patch("ptycho_fwd_bench.benchmarking.validate_sampling_conditions")
    @patch("ptycho_fwd_bench.benchmarking.load_ground_truth")  # <-- Key Mock
    @patch("ptycho_fwd_bench.benchmarking.run_benchmark_loop")
    @patch("ptycho_fwd_bench.benchmarking.compute_ground_truth")  # Should NOT be called
    def test_run_full_benchmark_load_mode(
        self,
        mock_compute,
        mock_loop,
        mock_load,
        mock_valid,
        mock_parse,
        mock_log,
        mock_dir,
        mock_yaml,
        mock_file,
        mock_config,
        mock_sim_params,
    ):
        """Test the full flow where Ground Truth is LOADED."""

        # Modify config to have load_file
        mock_config["sample"]["ground_truth"]["load_file"] = "existing_gt.npz"
        mock_yaml.return_value = mock_config
        mock_parse.return_value = mock_sim_params

        # Mock Load Return
        # Ensure n_map shape matches sim_params['n_total'] (100) to avoid warning
        n_map_loaded = np.zeros((100, 100))
        mock_load.return_value = (n_map_loaded, "psi_0", "psi_gt", "beam_gt")

        # Execute
        benchmarking.run_full_benchmark("dummy_config.yml")

        # Assertions
        mock_load.assert_called_with("existing_gt.npz")
        mock_compute.assert_not_called()  # Crucial check
        mock_loop.assert_called_once()
