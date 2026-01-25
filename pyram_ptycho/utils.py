import logging
import os
import shutil
from datetime import datetime
import subprocess


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "Unknown"


def setup_output_directory(config_path: str, experiment_name: str) -> str:
    """Creates timestamped output directory and saves simulation metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"{timestamp}_{experiment_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Archive config and version info
    shutil.copy(config_path, os.path.join(out_dir, "config_snapshot.yaml"))
    with open(os.path.join(out_dir, "commit_hash.txt"), "w") as f:
        f.write(get_git_revision_hash())

    return out_dir


def setup_logging(out_dir: str):
    """Initializes logging to both file and console."""
    log_file = os.path.join(out_dir, "benchmark.log")

    # Clear existing handlers to prevent duplication
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logging.info(f"Log initialized: {log_file}")
