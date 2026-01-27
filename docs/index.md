# PtychoFwdBench Documentation

Welcome to **PtychoFwdBench**, a benchmark suite for X-ray Ptychography propagation algorithms. This project focuses specifically on simulating "thick" volumetric samples where multiple scattering effects are significant.

The suite provides a interface to compare accuracy, computational cost, and memory usage across different propagation physics models.

## User Guide

### 1. Getting Started
* **[Installation](installation.md)**
  Set up the environment using `uv` and install dependencies.

### 2. Configuration & Usage
* **[Running Benchmarks](benchmark.md)**
  Learn how to use the Command Line Interface (CLI), structure your YAML configuration files, and interpret the results.
* **[Data Generators](generators.md)**
  Instructions on creating synthetic phantoms and refractive index maps (blobs, wave guides, etc.) for testing.

### 3. Solvers
* **[Solvers](solvers.md)**
  Details on the interface and parameter definitions for the implemented propagation engines:
  * **Multislice** (Fourier Split-Step)
  * **Finite Difference Pade** (PyRAM adaptation)
  * **Spectral Pade** (SSSP adaptation)

---

## Quick Example

Once installed, a benchmark can be run with a single command:

```bash
# Run using the uv runner
uv run run_benchmarks.py configs/blobs_default.yml
```
