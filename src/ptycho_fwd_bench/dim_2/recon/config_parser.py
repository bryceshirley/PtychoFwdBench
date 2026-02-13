import os
import shutil
from datetime import datetime

import yaml


def load_config(path="recon_config.yml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/recon/{cfg['experiment']['name']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    dest_path = os.path.join(output_dir, "config_run.yml")
    shutil.copy2(path, dest_path)

    # Calculate derived parameters
    width_m = cfg["sample"]["width_um"] * 1e-6
    thick_m = cfg["sample"]["thickness_um"] * 1e-6

    cfg["grid"]["dx"] = width_m / cfg["grid"]["global_nx"]
    cfg["grid"]["dz"] = thick_m / cfg["forward_model"]["coarse_nz"]
    cfg["grid"]["window_nx_padded"] = (
        cfg["grid"]["window_nx"] + 2 * cfg["grid"]["window_padding_nx"]
    )

    return cfg, output_dir


def build_sim_params(config):
    """Maps the UI config to the internal generator params."""
    sample_cfg = config["sample"]
    probe_cfg = config["probe"]
    grid_cfg = config["grid"]
    gt_cfg = config["ground_truth"]
    fwd_cfg = config["forward_model"]

    return {
        "sample_type": sample_cfg["type"],
        "dx": grid_cfg["dx"],
        "n_total": grid_cfg["global_nx"],
        "sample_params": sample_cfg["params"],
        "ground_truth_cfg": {
            "n_prop_fine": gt_cfg["fine_nz"],
            "solver_type": fwd_cfg["solver_name"],
            "solver_params": fwd_cfg["params"],
        },
        "total_width": sample_cfg["width_um"] * 1e-6,
        "probe_dia": probe_cfg["diameter_um"] * 1e-6,
        "probe_focus": probe_cfg["focus_um"] * 1e-6,
        "wavelength": probe_cfg["wavelength_um"] * 1e-6,
        "sample_thickness": sample_cfg["thickness_um"] * 1e-6,
        "n_pad": grid_cfg["window_padding_nx"],
    }
