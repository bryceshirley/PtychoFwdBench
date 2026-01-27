# PtychoFwdBench

A benchmark suite for X-ray Ptychography propagation algorithms, focusing on "thick" volumetric samples where multiple scattering effects are significant.

## Documentation
For details on installation, benchmarking instructions and methodology, please see [docs/index.md](docs/index.md).

## Solvers

This suite currently benchmarks two primary forward-model propagation techniques:

* **Finite Difference Pade (`FiniteDifferencePadeSolver`):**
    A wide-angle parabolic equation solver that uses finite-difference approximations for the transverse Laplacian. This implementation is directly adapted from the **PyRAM** library.

* **Multislice (`MultisliceSolver`):**
    The standard Fourier Split-Step method used in electron microscopy and X-ray ptychography. It utilizes FFTs (or Discrete Sine Transforms) to alternate between diffraction and refraction.

* **Spectral Pade (`SpectralPadeSolver`):** An implementation that combines the wide-angle accuracy of Pade approximants with the exact derivative calculation and speed of spectral methods. This implementation is directly adapted from the **SSSP** solver.

## Acknowledgements

The core Finite Difference solver (`FiniteDifferencePadeSolver`) is uses an adapted **PyRAM** (Python Range-dependent Acoustic Model) by **Marcus Donnelly**. We gratefully acknowledge his work in making this solver available.

* **Original PyRAM Repository:** [https://github.com/marcuskd/pyram](https://github.com/marcuskd/pyram)

The core Spectral solver (`SpectralPadeSolver`) is uses an adapted **SSSP** (Spectral Split-Step Padé) by **Daniel Walsken**. We gratefully acknowledge his work in making this solver available.

* **Original SSSP Repository:** [https://git.uni-wuppertal.de/walsken/sssp](https://git.uni-wuppertal.de/walsken/sssp)
