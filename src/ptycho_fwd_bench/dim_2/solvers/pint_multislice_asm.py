from typing import Optional

from .base import OpticalWaveSolver
from .utils import get_prop_phase_shift

# Use cupy if available, otherwise fallback to numpy
try:
    import cupy as xp
    import cupyx.scipy.sparse.linalg as spla
except ImportError:
    import numpy as xp
    import scipy.sparse.linalg as spla


class ParallelMultisliceSolverASM(OpticalWaveSolver):
    """
    Parallel Multislice Solver.
    """

    def __init__(
        self,
        n_map: xp.ndarray,
        dx: float,
        wavelength: float,
        dz: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        store_beam: bool = False,
        n_iter: int = 1,
        solver_type: str = "Richardson",
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.n_iter = n_iter
        self.solver_type = solver_type.upper()
        n_mean = xp.mean(self.n_map)

        # 1. Transverse Propagator Phase (Nx,1)
        # Fetch the raw phase shift, no exponential applied yet.
        phase_shift = get_prop_phase_shift(
            self.k0, self.nx, self.dx, self.dz, n_mean=n_mean
        )[:, xp.newaxis]

        # 2. Precompute the massive 2D exponential grids for max speed
        z_idx = xp.arange(self.nz_steps)[xp.newaxis, :]

        # Storing the kernels in memory as requested
        self._kernel_inv = xp.exp(-phase_shift * z_idx)
        self._kernel_fwd = xp.exp(phase_shift * (z_idx + 1))

        # 3. Object transmission setup
        phase = 1j * self.k0 * (self.n_map - n_mean) * (self.dz / 2)
        self._half_obj = xp.exp(phase)

        # 4. Object Scattering Correction E1
        # E1_j = (O_j)^{-1} (O_{j-1})^{-1} - 1
        inv_half_obj = xp.exp(-phase)
        self._E1_coeff = xp.zeros_like(self._half_obj)
        self._E1_coeff[:, 1:] = (inv_half_obj[:, 1:] * inv_half_obj[:, :-1]) - 1.0

        self._E1_buffer = xp.zeros((self.nx, self.nz_steps), dtype=xp.complex128)

    def _apply_E1(self, u_sol: xp.ndarray) -> xp.ndarray:
        """
        Apply E1 error.
        u = [u_0, u_1, u_2, ..., u_Nz]
        E1 is nilpotent:
        E1(u) = [0, u_0 * coeff_1, u_1 * coeff_2, ..., u_{Nz-1} * coeff_{Nz}]
        """
        # 1. Reset the first column (in case M_inv filled it previously)
        self._E1_buffer[:, 0] = 0.0

        # 2. Apply the shifted multiplication into the pre-allocated buffer
        self._E1_buffer[:, 1:] = u_sol[:, :-1] * self._E1_coeff[:, 1:]

        return self._E1_buffer

    def _apply_M_inv(self, u_sol: xp.ndarray) -> xp.ndarray:
        """
        Applies M^-1 using gpu optimized operations.
         This is the core operator application for the correction step.
        """
        # 1. Combined Real-Space Multiply (Entry)
        u_sol *= self._half_obj

        # 2. 1D batched FFT along x for each z-slice (in-place)
        xp.fft.fft(u_sol, axis=0, out=u_sol)

        # 3. Solve Bidiagonal Block Sparse Eigenvalues
        u_sol = self._solve_bidiag(u_sol)

        # 4. 1D batched iFFT along x for each z-slice (in-place)
        xp.fft.ifft(u_sol, axis=0, out=u_sol)

        # 5. Combined Real-Space Multiply (Exit)
        u_sol *= self._half_obj
        return u_sol

    def _solve_bidiag(self, u_sol):
        """
        [ V^{-1}       ][ w1]   [ u1]
        [-I V^{-1}     ][ w2]   [ u2]
        [  -I V^{-1}   ][ w3] = [ u3]
        [       ...    ][...]   [...]
        V is diagonal, so V^{-1} is just 1/V.
        V = exp(i * kz * dz)
        w_j = V(u_j + w_{j-1})
        w_j = sum_{k=1}^{j} V^{j-k+1} u_k = V^{j+1}sum_{k=1}^{j} V^{-k} u_k

        V^{-k} = exp(- i * kz * dz * k)
        """
        u_sol *= self._kernel_inv

        # Cumulative sum along propagation axis
        xp.cumsum(u_sol, axis=1, out=u_sol)

        # Multiply by V^{z} using the stored forward kernel
        u_sol *= self._kernel_fwd
        return u_sol

    def _apply_M_inv_source(self, psi_0: xp.ndarray) -> xp.ndarray:
        """Optimized M^-1 application for the initial source."""
        # 1. Compute 1D FFT along X
        psi_k = xp.fft.fft(self._half_obj[:, 0] * psi_0, axis=0)[:, xp.newaxis]

        # 2. Apply stored 2D kernel (Broadcasts Nx,1 * Nx,Nz -> Nx,Nz)
        u_sol = psi_k * self._kernel_fwd

        # 3. Inverse batched 1D FFT along X across all z-slices (in-place)
        xp.fft.ifft(u_sol, axis=0, out=u_sol)

        # 5. Apply Post-multiplier
        u_sol *= self._half_obj
        return u_sol

    def _solve_pass(
        self, psi_0: xp.ndarray, u_guess: Optional[xp.ndarray] = None
    ) -> xp.ndarray:
        """
        Runs the solver pass.
        """
        # 1. One-Shot Solution (Initial Guess u_0). Eliminates E_2.
        b = self._apply_M_inv_source(psi_0)
        u_sol = b if u_guess is None else u_guess

        # 2. Richardson Update
        if self.n_iter > 0:
            if self.solver_type == "RICHARDSON":
                for _ in range(self.n_iter):
                    # Apply Solution Correction
                    u_sol = b - self._apply_M_inv(self._apply_E1(u_sol))
            elif self.solver_type == "GMRES":

                def matvec(v):
                    v_reshaped = v.reshape(self.nx, self.nz_steps)
                    # Return Operator application: (I +  M^-1 * E1)v
                    return (
                        v_reshaped + self._apply_M_inv(self._apply_E1(v_reshaped))
                    ).ravel()

                A = spla.LinearOperator(
                    (self.nx * self.nz_steps, self.nx * self.nz_steps),
                    matvec=matvec,
                    dtype=xp.complex128,
                )

                # Solve using the correct effective RHS
                u_sol_flat, _ = spla.gmres(
                    A,
                    b.ravel(),
                    x0=u_sol.ravel(),
                    rtol=1e-10,
                    maxiter=self.n_iter,
                )
                u_sol = u_sol_flat.reshape(self.nx, self.nz_steps)
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")

        return u_sol

    def _propagate_and_store(
        self, psi_0: xp.ndarray, u_guess: Optional[xp.ndarray] = None
    ) -> xp.ndarray:
        """Wrapper to run the solver and store history if enabled."""
        u_sol = self._solve_pass(psi_0, u_guess=u_guess)
        if self.store_beam:
            self.beam_history = u_sol

        self.psi_final = u_sol[:, -1]

    def run(
        self,
        psi_init: Optional[xp.ndarray] = None,
    ):
        """
        Runs the solver.
        """
        # Initial operator setup

        psi_0 = self.initialize_wavefront(psi_init)

        self._propagate_and_store(psi_0)

        return self
