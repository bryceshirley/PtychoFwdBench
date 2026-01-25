import os
import logging
import subprocess
from unittest import mock
from pyram_ptycho.utils import (
    get_git_revision_hash,
    setup_output_directory,
    setup_logging,
)

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

    with mock.patch("pyram_ptycho.utils.datetime") as mock_date:
        # Freeze time for consistent folder naming
        mock_date.now.return_value.strftime.return_value = "20230101_120000"

        # We need to ensure results are written to tmp_path, not the real 'results' dir
        # We patch 'results' in the join call implicitly by just checking the returned path
        # But setup_output_directory hardcodes "results".
        # To make this clean without modifying code, we run it and clean up,
        # OR we temporarily chdir to tmp_path. Let's chdir.

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
        # Close file handlers so we can read the file safely on Windows/strict OS
        if isinstance(handler, logging.FileHandler):
            handler.close()

    # Read file content
    log_file = log_dir / "benchmark.log"
    assert log_file.exists()

    content = log_file.read_text()
    assert "Log initialized" in content
    assert test_msg in content
