# Sample Generators Reference

This page details the available `sample: type` options for the simulation. These generators create the **complex refractive index map** used to test the solvers:

$$
n(x, z) = n_{\text{background}} + (\delta_n + i \beta_n)
$$

Where:
* $\delta_n$ (Real part): Governs **Phase Shift** (Refraction).
* $\beta_n$ (Imaginary part): Governs **Absorption** (Attenuation).

You select a generator in your YAML config using the `type` field:

```yaml
sample:
  type: "BLOBS"  # Selects the generator
  params:        # Parameters specific to that generator
    delta_n: 1.0e-5  # Phase contrast
    beta_n:  1.0e-7  # Absorption contrast
    n_blobs: 50
```

---

## 1. BLOBS
**Key:** `BLOBS`

Generates random, overlapping Gaussian-softened spheres. This is the standard "biological tissue" phantom, useful for testing bulk scattering and phase accumulation without sharp edges.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `delta_n` | float | Refractive index contrast (Real part, $\delta_n$). | `0.01` |
| `beta_n` | float | Absorption (Imaginary part, $\beta_n$). | `0.01` |
| `n_blobs` | int | Number of random blobs to generate. | `100` |
| `blob_r_range_um` | [min, max] | Range of blob radii in microns ($r_{min}, r_{max}$). | `N/A` |
| `n_background` | float | Base refractive index ($n_{\text{bg}}$). | `1.0` |

> **Note:** Providing `blob_r_range_um` and `dx` allows physics-aware sizing. If omitted, sizes are calculated relative to grid pixels.

---

## 2. GRAVEL
**Key:** `GRAVEL`

Generates dense, non-overlapping sharp-edged polygons (Voronoi-like tessellation).
**Best for:** Testing diffraction from sharp edges and high-frequency spatial features. Simulates aggregate materials like powder or soil.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `delta_n` | float | Index contrast of the gravel pieces ($\delta_n$). | `0.2` |
| `beta_n` | float | Absorption contrast ($\beta_n$). | `0.02` |
| `avg_grain_size_um`| float | Approximate diameter of each grain ($2r$). | `N/A` |
| `n_blobs` | int | Number of particles/grains. | `200` |

> **Note:** Providing `avg_grain_size_um` and `dx` allows physics-aware sizing. If omitted, sizes are calculated relative to grid pixels.

---

## 3. WAVEGUIDE
**Key:** `WAVEGUIDE`

Generates a single straight channel (step-index waveguide) running along the $z$-axis.
**Best for:** Testing numerical stability, energy leakage, and verifying that the solver preserves confinement over long distances.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `delta_n` | float | Index contrast between core and cladding ($\delta_{\text{core}} - \delta_{\text{clad}}$). | `0.1` |
| `beta_n` | float | Core absorption ($\beta_{\text{core}}$). | `0.0` |
| `width_um` | float | Width of the waveguide core in microns ($w$). | `10.0` |

---

## 4. BRANCHING
**Key:** `BRANCHING`

Generates a fractal-like branching structure (similar to blood vessels or dendrites).
**Best for:** Testing multiscale resolution. The structure contains both large trunks and fine capillary tips in the same field of view.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `delta_n` | float | Refractive index contrast ($\delta_n$). | `0.03` |
| `initial_thickness`| float | Starting thickness of trunks (pixels). | `5.0` |
| `branching_prob` | float | Probability per step of a branch splitting ($P_{\text{branch}}$). | `0.008` |
| `split_angle` | float | Angle (radians) between split branches ($\theta$). | `0.25` |

---

## 5. FIBERS
**Key:** `FIBERS`

Generates a bundle of parallel cylinders (circles in cross-section) aligned perpendicular to the propagation direction ($z$).
**Best for:** Simulating fiber reinforced composites or grids. Creates strong periodic diffraction patterns.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `delta_n` | float | Refractive index contrast ($\delta_n$). | `0.02` |
| `fiber_rad_um` | float | Radius of each fiber ($r$). | `2.0` |
| `density` | float | Fill density (approximate, $\rho$). | `0.3` |
| `waviness_um` | float | Amplitude of sinusoidal waviness in position ($A$). | `1.0` |

---

## Physics Note: X-Ray vs. Optical Scales

The scale of your `delta_n` (Phase) and `beta_n` (Absorption) must match your wavelength $\lambda$ to avoid numerical artifacts like phase wrapping.

### 1. Visible Light ($\lambda \approx 400-700 \text{ nm}$)
Materials are often transparent or strongly absorbing.
* **$\delta_n$ (Real):** $\approx 0.01 - 0.5$ (e.g., Glass vs Air).
* **$\beta_n$ (Imaginary):** $\approx 0.0$ (Transparent) to $>1.0$ (Metals).

### 2. Hard X-Rays ($\lambda \approx 0.1 \text{ nm} / 10 \text{ keV}$)
Refractive indices are very close to $1.0$. The imaginary part ($\beta$) is typically much smaller than the real part ($\delta$).
* **$\delta_n$ (Real):** $\approx 10^{-5} - 10^{-6}$.
* **$\beta_n$ (Imaginary):** $\approx 10^{-7} - 10^{-9}$.

> **Critical Warning:** If you run an X-ray simulation ($\lambda \approx 10^{-4} \mu\text{m}$) using "Optical" parameters ($\delta_n \approx 0.1$), the phase shift per pixel will exceed $\pi$ radians, causing **Phase Wrapping** and erroneous solver convergence. Always use $10^{-5}$ scale for X-rays.
