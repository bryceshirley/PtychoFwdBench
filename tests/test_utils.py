import logging
import os
import subprocess
from unittest import mock
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from ptycho_fwd_bench.utils import (
    get_git_revision_hash,
    load_ground_truth,
    parse_config,
    save_ground_truth,
    setup_logging,
    setup_output_directory,
)


@patch("ptycho_fwd_bench.utils.setup_logging")
@patch("ptycho_fwd_bench.utils.setup_output_directory")
@patch("ptycho_fwd_bench.utils.yaml.safe_load")
def test_parse_config_logic(mock_yaml, mock_dir, mock_log):
    """Test that parse_config correctly orchestrates setup."""

    # Setup
    fake_config_path = "config.yaml"
    fake_cfg = {"experiment": {"name": "TestExp"}}
    mock_yaml.return_value = fake_cfg
    mock_dir.return_value = "outputs/TestExp_Date"

    # We use mock_open to avoid actual file IO
    with patch("builtins.open", mock_open(read_data="data")):
        cfg, out_dir = parse_config(fake_config_path)

    # Assertions
    assert cfg == fake_cfg
    assert out_dir == "outputs/TestExp_Date"
    mock_dir.assert_called_once_with(fake_config_path, "TestExp")
    mock_log.assert_called_once_with("outputs/TestExp_Date")


# =============================================================================
# 1. GIT HASH TESTS
# =============================================================================


@mock.patch("subprocess.check_output")
def test_get_git_revision_hash_success(mock_subprocess):
    """Test successful retrieval of git hash."""
    # Mock return value as bytes (subprocess standard output)
    mock_subprocess.return_value = b"abcdef123456\n"

    revision = get_git_revision_hash()
    assert revision == "abcdef123456"


@mock.patch("subprocess.check_output")
def test_get_git_revision_hash_failure(mock_subprocess):
    """Test behavior when git command fails (CalledProcessError)."""
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "git")

    revision = get_git_revision_hash()
    assert revision == "Unknown"


@mock.patch("subprocess.check_output")
def test_get_git_revision_hash_no_git(mock_subprocess):
    """Test behavior when git is not installed (OSError)."""
    mock_subprocess.side_effect = OSError("No such file or directory")

    revision = get_git_revision_hash()
    assert revision == "Unknown"


# =============================================================================
# 2. DIRECTORY SETUP TESTS
# =============================================================================


def test_setup_output_directory(tmp_path):
    """
    Test creation of output directory and file copying.
    Uses pytest's tmp_path fixture to avoid creating real folders.
    """
    # Create a dummy config file
    dummy_config = tmp_path / "dummy_config.yaml"
    dummy_config.write_text("experiment: test")

    with mock.patch("ptycho_fwd_bench.utils.datetime") as mock_date:
        # Freeze time for consistent folder naming
        mock_date.now.return_value.strftime.return_value = "20230101_120000"

        # We need to ensure results are written to tmp_path, not the real 'results' dir
        # We patch 'results' in the join call implicitly by just checking the returned path
        # But setup_output_directory hardcodes "results".

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Run the function
            out_dir = setup_output_directory(str(dummy_config), "TEST_EXP")

            # Verify directory exists
            assert os.path.exists(out_dir)
            assert "20230101_120000_TEST_EXP" in out_dir

            # Verify config snapshot exists
            snapshot_path = os.path.join(out_dir, "config_snapshot.yaml")
            assert os.path.exists(snapshot_path)

            # Verify git hash file exists
            hash_path = os.path.join(out_dir, "commit_hash.txt")
            assert os.path.exists(hash_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# 3. LOGGING SETUP TESTS
# =============================================================================


def test_setup_logging(tmp_path):
    """Test that logging is initialized and writes to file."""
    # Create a temp dir for logs
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Setup logging
    setup_logging(str(log_dir))

    # Log a test message
    test_msg = "Test Log Message"
    logging.info(test_msg)

    # Force flush handlers to ensure write to disk
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.flush()
        if isinstance(handler, logging.FileHandler):
            handler.close()

    # Read file content
    log_file = log_dir / "benchmark.log"
    assert log_file.exists()

    content = log_file.read_text()
    assert "Log initialized" in content
    assert test_msg in content


# =============================================================================
# 4. DATA I/O TESTS (save_ground_truth / load_ground_truth)
# =============================================================================


def test_save_ground_truth_full(tmp_path):
    """Test saving all ground truth arrays including beam history."""
    # 1. Setup dummy data
    file_path = tmp_path / "test_gt.npz"
    n_map_fine = np.ones((10, 10))
    psi_0 = np.zeros((10,), dtype=complex)
    psi_gt = np.random.rand(10) + 1j * np.random.rand(10)
    beam_gt = np.ones((10, 5))

    # 2. Execute Save
    save_ground_truth(str(file_path), n_map_fine, psi_0, psi_gt, beam_gt)

    # 3. Verify file exists
    assert file_path.exists()

    # 4. Verify content manually
    with np.load(file_path) as data:
        np.testing.assert_array_equal(data["n_map_fine"], n_map_fine)
        np.testing.assert_array_equal(data["psi_0"], psi_0)
        np.testing.assert_array_equal(data["psi_gt"], psi_gt)
        np.testing.assert_array_equal(data["beam_gt"], beam_gt)


def test_save_ground_truth_no_beam(tmp_path):
    """Test saving ground truth when beam_gt is None."""
    file_path = tmp_path / "test_gt_no_beam.npz"
    n_map_fine = np.ones((5, 5))
    psi_0 = np.zeros((5,))
    psi_gt = np.zeros((5,))

    # Execute Save with None for beam_gt
    save_ground_truth(str(file_path), n_map_fine, psi_0, psi_gt, beam_gt=None)

    # Verify that beam_gt is saved as an empty array (implementation detail)
    with np.load(file_path) as data:
        assert "beam_gt" in data
        assert data["beam_gt"].size == 0


def test_load_ground_truth_success(tmp_path):
    """Test loading a valid ground truth file."""
    # 1. Setup: Create a real .npz file
    file_path = tmp_path / "load_test.npz"
    n_map_in = np.ones((8, 8))
    psi_0_in = np.ones((8,))
    psi_gt_in = np.random.rand(8)
    # Simulate a file that HAS beam data
    beam_gt_in = np.random.rand(8, 2)

    np.savez_compressed(
        file_path,
        n_map_fine=n_map_in,
        psi_0=psi_0_in,
        psi_gt=psi_gt_in,
        beam_gt=beam_gt_in,
    )

    # 2. Execute Load
    n_map, psi0, gt, beam = load_ground_truth(str(file_path))

    # 3. Assertions
    np.testing.assert_array_equal(n_map, n_map_in)
    np.testing.assert_array_equal(psi0, psi_0_in)
    np.testing.assert_array_equal(gt, psi_gt_in)
    np.testing.assert_array_equal(beam, beam_gt_in)


def test_load_ground_truth_handles_missing_beam(tmp_path):
    """Test loading a file that has empty beam data returns None."""
    file_path = tmp_path / "load_empty_beam.npz"

    # Save with empty beam array
    np.savez_compressed(
        file_path,
        n_map_fine=np.zeros(1),
        psi_0=np.zeros(1),
        psi_gt=np.zeros(1),
        beam_gt=np.array([]),  # Empty
    )

    # Execute Load
    _, _, _, beam = load_ground_truth(str(file_path))

    # Assert beam is None (as per your logic: if arr.size > 0: beam_gt = arr)
    assert beam is None


def test_load_ground_truth_file_not_found():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_ground_truth("non_existent_file.npz")
