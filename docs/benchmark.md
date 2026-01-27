# Benchmark Guide

This guide details how to configure and run benchmarks for comparing the **PyRAM (Pade)** and **Multislice** optical wave solvers. The system relies on YAML configuration files to define the physical experiment, sample geometry, solver list, and execution parameters.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Reference](#configuration-reference)
3. [Outputs](#outputs)

---

## Quick Start

The benchmark script provides a Command Line Interface (CLI). Execute a benchmark by supplying the path to your YAML configuration file.

**Option 1: Run with uv (Recommended)** uv automatically handles the environment

```bash
# Or run using uv
uv run python run_benchmarks.py configs/blobs_default.yml
```

**Option 2: Run directly with Python** Manually activate the virtual environment first.

```bash
# Run with standard python
source .venv/bin/activate # Activate virtual environment Linux/Mac
# .venv\Scripts\activate  # Activate virtual environment Windows
python run_benchmarks.py configs/blobs_default.yml
```

---

## Configuration Reference

The YAML file is structured into six primary sections. Below is a detailed breakdown of each parameter.

### I. Experiment Metadata

Defines metadata for logging and file naming conventions.

```yaml
experiment:
  name: "BLOBS"          # Prefix for output filenames (e.g., BLOBS_1_setup.png)
  description: "Standard random soft blobs."

```

### II. Physics

Sets the fundamental optical properties for the simulation.
**Important:** All units are in **Microns ()**.

```yaml
physics:
  wavelength_um: 0.635   # Radiation wavelength (e.g., 635 nm red light)

  # Probe Geometry (Airy Disk with Spherical Phase)
  probe_dia_um: 10.0     # Aperture diameter (FWHM of intensity)
  probe_focus_um: -10.0  # Focal length.
                         # Negative = Diverging (Source behind aperture)
                         # Positive = Converging
                         # 0.0      = Collimated (Plane wave)

```

### III. Grid

Defines the simulation resolution and computational window.

```yaml
grid:
  n_physical: 1024       # Pixels in the physical computation window width
  n_padding: 256         # Pixels added to *each side* for boundary absorption
  physical_width_um: 150.0 # Physical width corresponding to 'n_physical'

```

> **Critical:** The pixel size  is derived as `physical_width_um / n_physical`. You must ensure  is small enough to resolve your probe's divergence angle (Nyquist limit).

### IV. Sample Definition

Configures the phantom generator and the "Ground Truth" reference simulation.

```yaml
sample:
  type: "BLOBS"          # Must correspond to a key in GENERATOR_MAP
  thickness_um: 40.0     # Total Z-depth of the sample

  # Ground Truth Settings
  # Runs first to generate the "Correct Answer" (psi_gt)
  ground_truth:
    n_prop_fine: 2000    # Step count for reference (fine step size)
    solver_type: "PADE"  # Algorithm to trust as truth (PADE handles high angles well)
    solver_params:
      pade_order: 8      # Expansion order for maximum accuracy

  # Generator Parameters (Passed to the specific generator function)
  params:
    delta_n: 0.05        # Refractive index contrast (Real part)
    beta_n: 0.005        # Absorption (Imaginary part)
    n_blobs: 300         # Generator-specific param
    blob_r_range_um: [0.05, 0.2] # Generator-specific param

```

### V. Solvers List

A list of solvers to benchmark against the Ground Truth. Multiple configurations of the same solver type are permitted.

```yaml
solvers:
  - name: "Pade (Order 8)"      # Display name for plots
    type: "PADE"                # Solver Class: "PADE" or "MULTISLICE"
    solver_params:
      pade_order: 8             # [PADE] Expansion order (higher = wider angle)

  - name: "Multislice (DST)"
    type: "MULTISLICE"
    solver_params:
      transform_type: "DST"     # [MS] "DST" (Sine Transform) or "FFT"
      symmetric: false          # [MS] Use symmetric split-step or not

```

### VI. Benchmark Execution

Controls the simulation loop and data saving preferences.

```yaml
benchmark:
  # List of step counts (N) to iterate through.
  # Solver runs with dz = thickness / N.
  step_counts: [8, 16, 32, 64]

  # Which run index to save for the full beam propagation plot?
  # 0 saves the first run (N=8), -1 saves the last run (N=64).
  save_beam_idx: 0

```

---

## Outputs

Results are automatically saved to a directory in `results/`, named with the timestamp and experiment name (e.g., `results/20260125_140000_BLOBS`).

* **`benchmark.log`**: Detailed logs containing execution times, grid stats, and error metrics.
* **`*_1_setup_exit.png`**: Visual comparison of refractive index maps and final exit wave intensities.
* **`*_2_beams.png`**: Full 2D beam propagation history (Z-X plane).
* **`*_3_metrics.png`**: Convergence analysis plot (Relative Error vs. Step Size).
* **`*_4_fine_coarse.png`**: Side-by-side comparison of the Ground Truth refractive index and the coarsest used.
* **`config_snapshot.yaml`**: An exact copy of the configuration used for reproducibility.

---

## Troubleshooting

### Zig-Zag Convergence (Phase Wrapping)

**Symptom:** The solver error oscillates (improves, worsens, improves) or fails to converge monotonically.

**Cause:** The phase shift per step exceeds  radians. This usually happens when simulating X-rays with "Optical" refractive indices.

**Fix:** Ensure `delta_n` is set to physical X-ray scales rather than optical scales.

### Aliasing Warning

**Symptom:** Log shows `CRITICAL WARNING: Probe divergence exceeds grid limit`.

**Cause:** Your grid resolution is too coarse to support the high angles in your divergent probe.

**Fix:**

1. Increase `n_physical` (e.g., 1024  2048).
2. Decrease the Field of View (`physical_width_um`).
3. Reduce the probe divergence (increase `probe_focus_um`).
