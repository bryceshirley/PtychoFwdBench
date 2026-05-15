import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from ptycho_fwd_bench.dim_2.solvers.sssp.pade import pade_coefficients

from .base import OpticalWaveSolver
from .utils import apply_spectral_kernel, get_spectral_coords

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
    n_iter : int, optional
        Maximum iterations for Richardson solver.
    store_beam : bool, optional
        Whether to store beam history.
    envelope : bool, optional
        Whether to include envelope in Pade coefficients.
    mode : str, optional
        Mode for spectral coordinates ('spectral', 'fd2', 'fd4', 'pseudo').
    preconditioner : str, optional
        Type of preconditioner to use ('split_step', 'shifted_mean', 'additive').
    solver_type : str, optional
        Type of iterative solver ('bicgstab', 'gmres').
    solver_stats : bool, false
        Optional logging of inner iterative solve iteration count and residual.
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
        transform_type: str = "FFT",
        n_iter: int = 2,
        store_beam: bool = False,
        envelope: bool = False,
        mode: str = "spectral",
        preconditioner: str = "split_step",
        solver_type: str = "bicgstab",
        solver_stats: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.pade_order = pade_order
        self.n_iter = n_iter
        self.transform_type = transform_type
        self.preconditioner = preconditioner
        self.solver_type = solver_type
        self.solve_stats = solver_stats

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
        if self._solver_stats and self._solver_stats["iters"]:
            avg_iter = np.mean(self._solver_stats["iters"])
            avg_resid = np.mean(self._solver_stats["residuals"])
            logger.info(
                f"Spectral Pade [{self.solver_type}|{self.preconditioner}]: "
                f"Avg Iters: {avg_iter:.2f} | Avg Rel. Residual: {avg_resid:.2e}"
            )

        return self

    def _get_preconditioner_op(
        self, b_j: complex, b_diff: complex, N_vals: np.ndarray
    ) -> LinearOperator:
        """Constructs the LinearOperator for M^-1 based on self.preconditioner.
        Parameters:
            b_j: Pade coefficient for the j-th term.
            b_diff: Scaled diffraction coefficient.
            N_vals: Refractive term values (np.ndarray)
        Returns:
            LinearOperator representing M^-1 Preconditioner.
        """

        M_L_kernel = 1.0 + b_diff * self.Lambda
        M_N_vals = 1.0 + b_j * N_vals
        n_size = self.nx

        if self.preconditioner == "split_step":
            # M^-1 ~ (1+bN)^-1 (1+bL)^-1
            # A = (I+bL)(I+bN) - b^2 L N
            # M^-1 A = I + b^2 (1+bN)^-1 L (1+bL)^-1 N
            # L = F^-1 P F, (1+bL) = F^-1 (1+bP) F, so (1+bL)^-1 = F^-1 (1+bP)^-1 F
            # M^-1 A = I + b^2 (1 + bN)^-1 F^-1 (P/(1+bP)) F N, where P/(1+bP) is the modified spectral kernel for the correction term
            # Inverse L (Spectral)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_L_kern = 1.0 / M_L_kernel
                inv_L_kern[np.isclose(M_L_kernel, 0)] = 0.0

            # Inverse N (Spatial)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_N_vals = 1.0 / M_N_vals
                inv_N_vals[np.isclose(M_N_vals, 0)] = 0.0

            def matvec_M_inv(x_vec):
                x_grid = x_vec.reshape(n_size)
                # Apply L^-1 then N^-1
                temp = self._apply_diffraction(x_grid, inv_L_kern)
                out = temp * inv_N_vals
                return out.ravel()

        elif self.preconditioner == "shifted_mean":
            # M^-1 ~ (1 + bL + b*mean(N))^-1 = F^-1 (1 + bP + b*mean(N))^-1 F
            # A = I + bL + b*mean(N) + b perb_N
            # M^-1 A = I + b M^-1 perb_N, where perb_N = N - mean(N)
            #        = I + b F^-1 (1 + bP + b*mean(N))^-1 F perb_N
            N_mean = np.mean(N_vals)
            shifted_kernel = 1.0 + b_diff * self.Lambda + b_j * N_mean

            with np.errstate(divide="ignore", invalid="ignore"):
                inv_shifted_kernel = 1.0 / shifted_kernel
                inv_shifted_kernel[np.isclose(shifted_kernel, 0)] = 0.0

            def matvec_M_inv(x_vec):
                x_grid = x_vec.reshape(n_size)
                out = self._apply_diffraction(x_grid, inv_shifted_kernel)
                return out.ravel()

        else:
            raise ValueError(f"Unknown preconditioner: {self.preconditioner}")

        return LinearOperator((n_size, n_size), matvec=matvec_M_inv, dtype=complex)

    def _apply_diffraction(self, psi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply the spectral diffraction operator using the given kernel.
        L = F^-1 [kernel * F[psi]]
        Parameters:
            psi: Input wavefront (np.ndarray)
            kernel: Spectral kernel to apply (np.ndarray)
        Returns:
            Resulting wavefront after applying diffraction (np.ndarray)
        """
        return apply_spectral_kernel(psi, kernel, self.transform_type)

    def _get_direct_op(self, b_j, b_diff, N_vals):
        """
        Construct the Linear Operator for the full direct step
        A = I + b_diff L + b_j N
        Parameters:
            b_j: Pade coefficient for the j-th term.
            b_diff: Scaled diffraction coefficient.
            N_vals: Refractive term values (np.ndarray)
        Returns:
            LinearOperator representing the direct operator A.
        """
        n_size = self.nx

        def matvec_A(x_vec):
            x_grid = x_vec.reshape(n_size)
            # A = I + bL + bN = (I+bL)(I+bN) - b^2 L N
            term_L = b_diff * self._apply_diffraction(x_grid, self.Lambda)
            term_N = b_j * N_vals * x_grid
            return (x_grid + term_L + term_N).ravel()

        return LinearOperator((n_size, n_size), matvec=matvec_A, dtype=complex)

    def _solve_pade_term(
        self, psi: np.ndarray, b_j: complex, N_vals: np.ndarray
    ) -> np.ndarray:
        """
        Solve for the j-th Pade term w_j using an iterative solver.
        Parameters:
            psi: Current wavefront (np.ndarray)
            b_j: Pade coefficient for the j-th term.
            N_vals: Refractive term values (np.ndarray)
        Returns:
            w_j: Solution for the j-th Pade term (np.ndarray)
        """
        # 1. Get Preconditioner Operator and the Full Direct Operator
        b_diff = b_j / self.k0sq
        A_op = self._get_direct_op(b_j, b_diff, N_vals)
        M_op = self._get_preconditioner_op(b_j, b_diff, N_vals)

        # 2. Initial Guess (Preconditioned b)
        b_vec = psi.ravel()
        x0 = M_op.matvec(b_vec)

        # 3. Iteration Callback
        iter_count = 0

        def callback(xk):
            nonlocal iter_count
            iter_count += 1

        # 4. Run Solver
        rtol = 1e-8  # Tight tolerance for inner solve
        if self.solver_type == "bicgstab":
            w_flat, info = bicgstab(
                A_op,
                b_vec,
                x0=x0,
                M=M_op,
                rtol=rtol,
                maxiter=self.n_iter,
                callback=callback,
            )
        elif self.solver_type == "gmres":
            w_flat, info = gmres(
                A_op,
                b_vec,
                x0=x0,
                M=M_op,
                rtol=rtol,
                maxiter=self.n_iter,
                callback=callback,
            )
        elif self.solver_type == "richardson":
            # Simple Richardson iteration:
            # x_{k+1} = x_k + M^-1 (b - A x_k)
            #         = x_k + M^-1 b - M^-1 A x_k
            #          = x_k + x_0 - M^-1 A x_k
            #          = x_0 + (I - (I + M^-1 V)) x_k, where A = M + V
            #          = x_0 + M^-1 V x_k
            # V depends on the preconditioner choice.
            V = N_vals - np.mean(N_vals)
            w_flat = x0.copy()
            for _ in range(self.n_iter):
                w_flat = x0 + M_op.matvec(V * w_flat)  # Update step
                iter_count += 1
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        # 5. Stats
        # Calculate residual manually to verify
        if self._solver_stats:
            final_residual_vec = b_vec - A_op.matvec(w_flat)
            final_rel_resid = np.linalg.norm(final_residual_vec) / np.linalg.norm(b_vec)

            self._solver_stats["iters"].append(iter_count)
            self._solver_stats["residuals"].append(final_rel_resid)

        return w_flat.reshape(self.nx)
