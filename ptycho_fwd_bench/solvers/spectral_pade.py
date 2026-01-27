import numpy as np
from .base import OpticalWaveSolver
from .utils import apply_spectral_kernel, get_spectral_coords
from ptycho_fwd_bench.sssp.pade import pade_coefficients
from scipy.sparse.linalg import bicgstab, LinearOperator

import logging

logger = logging.getLogger(__name__)


class SpectralPadeSolver(OpticalWaveSolver):
    """
    Spectral Pade Solver implementing High-Order Spectral Split-Step Pade.

    Parameters
    ----------
    n_map : np.ndarray
        Complex refractive index map.
    dz : float
        Step size in propagation direction.
    dx : float
        Spatial grid spacing.
    wavelength : float
        Wavelength of the wave.
    probe_dia : float, optional
        Diameter of the probe beam.
    probe_focus : float, optional
        Focus position of the probe beam.
    pade_order : int, optional
        Order of the Pade approximation.
    transform_type : str, optional
        Type of spectral transform ("DST" or "FFT").
    max_iter : int, optional
        Maximum iterations for Richardson solver.
    store_beam : bool, optional
        Whether to store beam history.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dz: float,
        dx: float,
        wavelength: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        pade_order: int = 4,
        transform_type: str = "DST",
        max_iter: int = 2,
        store_beam: bool = False,
        envelope: bool = False,
        mode: str = "spectral",
        preconditioner: str = "split_step",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.pade_order = pade_order
        self.max_iter = max_iter
        self.transform_type = transform_type
        self.preconditioner = preconditioner

        # Pade Coeffs
        hk0 = self.dz * self.k0
        self.b_coeffs_raw, self.d_coeffs = pade_coefficients(
            hk0, self.pade_order, envelope=envelope
        )

        # Spectral Operator Lambda = -kx^2
        kx = get_spectral_coords(self.nx, self.dx, self.transform_type, mode=mode)
        self.Lambda = -(kx**2)

        # Initialize stats container
        self._solver_stats = {"iters": [], "residuals": []}

    def _apply_diffraction(self, psi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return apply_spectral_kernel(psi, kernel, self.transform_type)

    def run(self, psi_init: np.ndarray = None) -> "SpectralPadeSolver":
        """
        Propagates the wavefront through the medium using Spectral Pade method.
        Parameters:
            psi_init: Initial wavefront (np.ndarray)
        Returns:
            self: Updated solver with final wavefront
        """
        # Reset stats
        self._solver_stats = {"iters": [], "residuals": []}

        # Base class handles creation or validation
        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Propagation loop
        for i in range(self.nz_steps - 1):
            # 1. Initialize Summation: d0 * psi
            psi_next = self.d_coeffs[0] * psi

            # 2. Define Refractive Term N = n^2 - 1
            N_vals = (self.n_map[:, i] ** 2) - 1.0

            # 3. Add partial fraction terms sum(d_j * w_j)
            for j in range(self.pade_order):
                b_j = self.b_coeffs_raw[j]
                d_j = self.d_coeffs[j + 1]
                w_j = self._solve_pade_term(psi, b_j, N_vals)
                psi_next += d_j * w_j

            psi = psi_next
            if self.store_beam:
                self.beam_history[:, i + 1] = psi

        self.psi_final = psi

        # Log aggregated statistics
        if self._solver_stats["iters"]:
            avg_iter = np.mean(self._solver_stats["iters"])
            avg_resid = np.mean(self._solver_stats["residuals"])
            logger.info(
                f"Spectral Pade Run [{self.preconditioner}]: "
                f"Avg Iters: {avg_iter:.2f} | Avg Rel. Residual: {avg_resid:.2e}"
            )

        return self

    def _solve_pade_term(
        self, psi: np.ndarray, b_j: complex, N_vals: np.ndarray
    ) -> np.ndarray:
        # 1. Setup Operators
        b_diff = b_j / self.k0sq

        # Precompute Preconditioner terms
        M_L_kernel = 1.0 + b_diff * self.Lambda
        M_N_vals = 1.0 + b_j * N_vals

        # Define the Linear Operator A(x) = (1 + bL + bN)x
        def matvec(x_vec):
            # Reshape vector to grid for operation
            x_grid = x_vec.reshape(self.nx)

            # Apply L term: b_diff * FT^-1[ k^2 * FT[x] ]
            term_L = b_diff * self._apply_diffraction(x_grid, self.Lambda)

            # Apply N term: b_j * N * x
            term_N = b_j * N_vals * x_grid

            # A = I + L term + N term
            return (x_grid + term_L + term_N).ravel()

        # 2. Select Preconditioner Logic
        if self.preconditioner == "split_step":
            # M^-1 ~ (1+bN)^-1 (1+bL)^-1
            # Inverse of L (Spectral)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_L_kern = 1.0 / M_L_kernel
                inv_L_kern[np.isclose(M_L_kernel, 0)] = 0.0

            # Inverse of N (Spatial)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_N_vals = 1.0 / M_N_vals
                inv_N_vals[np.isclose(M_N_vals, 0)] = 0.0

            def matvec_M_inv(x_vec):
                x_grid = x_vec.reshape(self.nx)
                # Apply L^-1 then N^-1
                temp = self._apply_diffraction(x_grid, inv_L_kern)
                out = temp * inv_N_vals
                return out.ravel()

        elif self.preconditioner == "shifted_mean":
            # M^-1 ~ (1 + bL + b*mean(N))^-1
            # Effectively a pure spectral filter with shifted wavenumber
            N_mean = np.mean(N_vals)

            # The spectral kernel becomes (1 + bL_k + b*N_mean)
            shifted_kernel = 1.0 + b_diff * self.Lambda + b_j * N_mean

            with np.errstate(divide="ignore", invalid="ignore"):
                inv_shifted_kernel = 1.0 / shifted_kernel
                inv_shifted_kernel[np.isclose(shifted_kernel, 0)] = 0.0

            def matvec_M_inv(x_vec):
                x_grid = x_vec.reshape(self.nx)
                out = self._apply_diffraction(x_grid, inv_shifted_kernel)
                return out.ravel()

        elif self.preconditioner == "additive":
            # M^-1 ~ (1+bN)^-1 + (1+bL)^-1 - I

            # Inverse L
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_L_kern = 1.0 / M_L_kernel
                inv_L_kern[np.isclose(M_L_kernel, 0)] = 0.0

            # Inverse N
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_N_vals = 1.0 / M_N_vals
                inv_N_vals[np.isclose(M_N_vals, 0)] = 0.0

            def matvec_M_inv(x_vec):
                x_grid = x_vec.reshape(self.nx)

                # Term 1: Spatial part
                term1 = x_grid * inv_N_vals

                # Term 2: Diffraction part
                term2 = self._apply_diffraction(x_grid, inv_L_kern)

                # Additive combination
                out = term1 + term2 - x_grid
                return out.ravel()

        else:
            raise ValueError(f"Unknown preconditioner: {self.preconditioner}")

        # 2. Construct Scipy LinearOperators
        n_size = self.nx
        A_op = LinearOperator((n_size, n_size), matvec=matvec, dtype=complex)
        M_op = LinearOperator((n_size, n_size), matvec=matvec_M_inv, dtype=complex)

        # 3. Initial Guess (Split-Step Solution)
        # Providing this x0 drastically reduces the required iterations
        b_vec = psi.ravel()
        x0 = M_op.matvec(b_vec)

        # 4. Callback for logging
        iter_count = 0

        def callback(xk):
            nonlocal iter_count
            iter_count += 1

        # 5. Run BiCGStab
        # Note: 'tol' is deprecated in Scipy 1.12+, use 'rtol'
        w_flat, info = bicgstab(
            A_op,
            b_vec,
            x0=x0,
            M=M_op,
            rtol=1e-5,
            maxiter=self.max_iter,
            callback=callback,
        )

        # 6. Calculate Residual and Store Stats
        final_residual_vec = b_vec - A_op.matvec(w_flat)
        final_rel_resid = np.linalg.norm(final_residual_vec) / np.linalg.norm(b_vec)

        self._solver_stats["iters"].append(iter_count)
        self._solver_stats["residuals"].append(final_rel_resid)

        return w_flat.reshape(self.nx)
