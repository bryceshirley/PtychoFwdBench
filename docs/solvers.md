# Solvers Configuration

This benchmark suite supports three primary propagation engines. Solvers are defined in the `solvers` list within your YAML configuration file.

Each entry in the list requires:
1.  **`name`**: A unique display name for the plot legends.
2.  **`type`**: The identifier string for the solver class.
3.  **`solver_params`**: A dictionary of arguments specific to that solver.

---

## 1. Multislice Solver
**Type ID:** `MULTISLICE`

The standard Fourier Split-Step method used in electron microscopy and X-ray ptychography. It alternates between a phase screen (refraction) and free-space propagation (diffraction).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `transform_type` | `str` | `"DST"` | The spectral transform method. Options: `"DST"` (Discrete Sine Transform, enforces Dirichlet BCs) or `"FFT"` (Periodic BCs). |
| `symmetric` | `bool` | `False` | If `True`, performs a half-step propagation before and after the phase screen (Strang splitting). If `False`, uses simple first-order splitting. |
| `store_beam` | `bool` | `True` | If `True`, stores the full complex wavefield at every slice. |

---

## 2. Finite Difference Pade
**Type ID:** `PADE`

A wide-angle parabolic equation solver adapted from **PyRAM**. It uses finite-difference approximations for the transverse Laplacian with Pade approximants for the propagation operator.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pade_order` | `int` | *Required* | The order of the Pade approximation (number of terms). Higher orders (e.g., 4, 8) handle wider angles but are slower. |
| `beam_store_resolution` | `tuple` | `None` | A tuple `(ndr, ndz)` to downsample and store the beam history. If `None`, history is not stored. |

---

## 3. Spectral Pade
**Type ID:** `SPECTRAL_PADE`

A solver combining wide-angle Pade accuracy with the speed and exact laplacian operator of spectral methods. It solves the implicit Pade step using iterative methods (BiCGSTAB or GMRES).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pade_order` | `int` | *Required* | Order of the Pade approximation. |
| `solver_type` | `str` | `"bicgstab"` | The iterative solver to use. Options: `"bicgstab"`, `"gmres"`. |
| `preconditioner` | `str` | `"split_step"` | Accelerates convergence. Options: `"split_step"`, `"shifted_mean"`, `"additive"`. |
| `max_iter` | `int` | `4` | Maximum iterations for the Richardson/Iterative solver per step. |
| `transform_type` | `str` | `"DST"` | Options: `"DST"` (Dirichlet) or `"FFT"` (Periodic). |
| `mode` | `str` | `"spectral"` | Coordinate mode: `"spectral"`, `"fd2"`, `"fd4"`, or `"pseudo"`. |
| `envelope` | `bool` | `False` | Whether to include the slowly varying envelope in Pade coefficients. |
| `store_beam` | `bool` | `True` | If `True`, stores the full complex wavefield at every step. |

---

## Configuration Example

Below is a valid YAML configuration testing all three solvers with various settings.

```yaml
solvers:
  # --- Finite Difference Pade ---
  - name: "PyRAM Pade (Order 8)"
    type: "PADE"
    solver_params:
      pade_order: 8

  # --- Multislice ---
  - name: "Multislice (DST Symmetric)"
    type: "MULTISLICE"
    solver_params:
      transform_type: "DST"
      symmetric: true

  - name: "Multislice (FFT Periodic)"
    type: "MULTISLICE"
    solver_params:
      transform_type: "FFT"
      symmetric: false

  # --- Spectral Pade ---
  - name: "Spectral Pade (Order 4, Split-Step Precon)"
    type: "SPECTRAL_PADE"
    solver_params:
      pade_order: 4
      transform_type: "FFT"
      solver_type: "bicgstab"
      preconditioner: "split_step"
      max_iter: 10
```
