# PtychoFwdBench

A benchmark suite for X-ray Ptychography propagation algorithms, focusing on "thick" volumetric samples where multiple scattering effects are significant.

## Documentation
For detailed benchmarking instructions and methodology, please see [docs/benchmark.md](docs/benchmark.md).

## Solvers

This suite currently benchmarks two primary forward-model propagation techniques:

* **Finite Difference Padé (`PtychoPadeSolver`):**
    A wide-angle parabolic equation solver that uses finite-difference approximations for the transverse Laplacian. This implementation is directly adapted from the **PyRAM** library.

* **Multislice (`MultisliceSolver`):**
    The standard Fourier Split-Step method used in electron microscopy and X-ray ptychography. It utilizes FFTs (or Discrete Sine Transforms) to alternate between diffraction and refraction.

### Roadmap
* **Spectral Padé (`SpectralPadeSolver`):** An upcoming implementation that combines the wide-angle accuracy of Padé approximants with the exact derivative calculation and speed of spectral methods.

## Acknowledgements

The core Finite Difference solver (`PtychoPadeSolver`) is uses an adapted **PyRAM** (Python Range-dependent Acoustic Model) by **Marcus Donnelly**. We gratefully acknowledge his work in making this robust solver available.

* **Original PyRAM Repository:** [https://github.com/marcuskd/pyram](https://github.com/marcuskd/pyram)
