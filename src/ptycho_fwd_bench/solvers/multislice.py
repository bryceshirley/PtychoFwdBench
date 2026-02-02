from typing import Optional

import numpy as np

from .base import OpticalWaveSolver
from .utils import apply_spectral_kernel, get_spectral_coords


class MultisliceSolver(OpticalWaveSolver):
    """
    Multislice beam propagation solver using spectral methods.

    Parameters
    ----------
    n_map : np.ndarray
        Complex refractive index map of shape (nx, nz_steps).
    dx : float
        Spatial sampling interval in x (um).
    dz : float
        Slice thickness in z (um).
    wavelength : float
        Wavelength of the optical wave (um).
    probe_dia : float
        Diameter of the probe beam (um).
    probe_focus : float
        Focal distance of the probe beam (um).
    symmetric : bool, optional
        If True, uses symmetric propagation (half-step before and after phase). Defaults to True.
    transform_type : str, optional
        Type of spectral transform to use ("DST" or "FFT"). Defaults to "DST".
    store_beam : bool, optional
        If True, stores the wavefield at each slice. Defaults to False.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        dz: float,
        symmetric: bool = True,
        transform_type: str = "FFT",
        store_beam: bool = False,
    ):
        super().__init__(n_map, dx, wavelength, dz, probe_dia, probe_focus, store_beam)
        self.symmetric = symmetric
        self.transform_type = transform_type
        self._kernel_cache = {}

    def _get_propagation_kernel(self, dz: float) -> np.ndarray:
        key = (self.transform_type, dz)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)

        # Standard vacuum propagator: exp(i * sqrt(k0^2 - kx^2) * z)
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        Lambda = 1j * (sqrt_term - self.k0)
        H = np.exp(Lambda * dz).astype(np.complex128)

        self._kernel_cache[key] = H
        return H

    def run(self, psi_init: Optional[np.ndarray] = None) -> "MultisliceSolver":
        # Base class handles creation or validation
        psi = self.initialize_wavefront(psi_init)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)
            self.beam_history[:, 0] = psi

        # Propagation Loop
        for i in range(self.nz_steps):
            step_dist = (self.dz / 2.0) if self.symmetric else self.dz

            # Half-step
            H_half = self._get_propagation_kernel(step_dist)
            psi = apply_spectral_kernel(psi, H_half, self.transform_type)

            # Phase
            n_slice = self.n_map[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)

            # Half-step
            if self.symmetric:
                psi = apply_spectral_kernel(psi, H_half, self.transform_type)

            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi
        return self


class ParallelMultisliceSolver(MultisliceSolver):
    """
    Parallel 'One-Shot' Multislice Solver using Twisted 3D FFT.

    replaces sequential time-stepping with a 3D spectral filter.
    Includes adaptive Richardson correction for robustness.
    """

    def __init__(self, *args, alpha: float = 1e-6, **kwargs):
        # Force store_beam=True because parallel solve computes all slices at once
        kwargs["store_beam"] = True
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.n_iter = 3  # Fixed number of Richardson iterations for correction

    def _twisted_fft(self, u: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Implements the Twisted 3D FFT (Eq 12 in PDF).
        Order: Transform X -> Twist Z -> Transform Z
        """
        L = self.nz_steps

        # 1. Transverse Transform (X)
        # We respect the DST/FFT choice for the spatial dimension
        if self.transform_type == "FFT":
            if not inverse:
                u_x = np.fft.fft(u, axis=0)
            else:
                u_x = np.fft.ifft(u, axis=0)
        else:
            raise NotImplementedError(
                f"Parallel solver not optimized for {self.transform_type}"
            )

        # 2. Twist & Longitudinal Transform (Z)
        # Gamma_alpha = alpha^(i/L)
        # Note: We must use FFT for Z dimension (Circulant property)

        # Create Twist Vector (1, L)
        z_indices = np.arange(L)
        # Forward: Twist then FFT
        if not inverse:
            gamma = self.alpha ** (z_indices / L)
            u_twisted = u_x * gamma[np.newaxis, :]  # Broadcast to (Nx, L)
            return np.fft.fft(u_twisted, axis=1)

        # Inverse: IFFT then Untwist
        else:
            u_z = np.fft.ifft(u_x, axis=1)  # Here u_x is actually the input in k-space
            gamma_inv = self.alpha ** (-z_indices / L)
            return u_z * gamma_inv[np.newaxis, :]

    def _get_3d_kernel(self) -> np.ndarray:
        """
        Computes the static 3D Dispersion Kernel K (Eq 13 in PDF).
        Includes the Mean Potential Shift for stability.
        """
        # 1. Transverse Propagator (Shifted)
        # L_kx includes vacuum diffraction + mean potential phase
        kx = get_spectral_coords(self.nx, self.dx, self.transform_type)
        inside = self.k0sq - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))

        # We calculate the MEAN refractive index to shift the propagator
        n_mean = np.mean(self.n_map)

        # Phase shift from Mean Potential: k0 * (n_mean - 1) * dz
        # Vacuum Propagator: i * (sqrt - k0) * dz
        # Combined exponent: i * (sqrt - k0 + k0*(n_mean - 1)) * dz
        #                  = i * (sqrt - k0 + k0*n_mean - k0) * dz ... Wait
        # Let's stick to: L_vac * exp(i * phi_mean)

        lambda_vac = 1j * (sqrt_term - self.k0)
        phi_mean = 1j * self.k0 * (n_mean - 1.0)

        # L_kx shape: (Nx, 1)
        L_kx = np.exp((lambda_vac + phi_mean) * self.dz)[:, np.newaxis]

        # 2. Longitudinal Eigenvalues (Shift Operator)
        # lambda_alpha(kz) = alpha^(1/L) * exp(-2pi * i * kz / L)
        L = self.nz_steps
        kz = np.arange(L)
        lam_alpha = (self.alpha ** (1 / L) * np.exp(-2j * np.pi * kz / L))[
            np.newaxis, :
        ]

        # 3. Construct 3D Kernel
        # K = 1 / (1 - lam_alpha * L_kx)
        denom = 1.0 - (lam_alpha * L_kx)
        return 1.0 / denom

    def run(self, psi_init: Optional[np.ndarray] = None) -> "ParallelMultisliceSolver":
        psi_0 = self.initialize_wavefront(psi_init)
        L = self.nz_steps

        # --- Pre-processing: Decomposition ---
        # Decompose N(x,y) = N_mean + Delta_N
        n_mean = np.mean(self.n_map)
        delta_n = self.n_map - n_mean

        # --- Step 1: Prepare Source Vector 's' ---
        # The RHS of the system M*u = s.
        # s = [P_0 * u_0, 0, ... 0]
        # We propagate u_0 through the FIRST slice manually to get the source term
        # Note: We must use the SHIFTED propagator for consistency

        # Calculate single-slice propagator P_0 = N_half * L_shifted * N_half
        # But for the source term 's', it's simpler to treat u_input as entering slice 1.
        # s has only the first element non-zero.

        # 1. Apply N_half (Entrance Refraction of Slice 0)
        psi_in = psi_0 * np.exp(1j * self.k0 * delta_n[:, 0] * self.dz / 2)

        # 2. Apply L_shifted (Diffraction + Mean Phase)
        # We reuse _get_propagation_kernel but need to add mean phase manually
        # or just compute it locally.
        H_vac = self._get_propagation_kernel(self.dz)
        H_mean = np.exp(1j * self.k0 * (n_mean - 1.0) * self.dz)
        psi_in = apply_spectral_kernel(psi_in, H_vac * H_mean, self.transform_type)

        # 3. Apply N_half (Exit Refraction of Slice 0)
        psi_in *= np.exp(1j * self.k0 * delta_n[:, 0] * self.dz / 2)

        # Construct Source Vector S (Nx, L) - mostly zeros
        S = np.zeros((self.nx, L), dtype=np.complex128)
        S[:, 0] = psi_in

        # --- Step 2: Parallel Solve Loop (Richardson Iteration) ---
        # u_k+1 = u_k + M_approx^-1 * (s - M_exact * u_k)

        # Initial Guess u_0 = M_approx^-1 * S
        # Pre-compute Kernel
        K_3d = self._get_3d_kernel()

        # Define the Parallel Propagator Function (M_approx^-1)
        def apply_parallel_propagator(rhs_vector):
            # 1. Refract (Entrance) - Part of M definition in
            # M = diag(N) * Solve * diag(N)
            # So M^-1 = diag(N^-1) * Solve^-1 * diag(N^-1)
            # But the PDF Eq 11 defines M*u. We want u = M^-1 s.
            # The decomposition was M_approx = D_N * K_L * D_N
            # So u = D_N^-1 * (K_L^-1 * (D_N^-1 * s))

            # Note: D_N contains exp(i * delta_n * dz / 2)
            # So D_N^-1 contains exp(-i ...)

            N_phase = 1j * self.k0 * delta_n * self.dz / 2
            inv_N = np.exp(-N_phase)

            v = rhs_vector * inv_N  # Entrance

            # 2. Twisted 3D Solve
            v_k = self._twisted_fft(v, inverse=False)
            v_k *= K_3d  # Apply Dispersion Kernel
            v = self._twisted_fft(v_k, inverse=True)

            # 3. Refract (Exit)
            u_out = v * inv_N
            return u_out

        # --- 3. Run Approximate Richardson Iteration ---
        # "One-Shot" Predictor (u_0)
        u_base = apply_parallel_propagator(S)
        u_sol = u_base.copy()

        if self.n_iter > 1:
            # Error Matrix E = I - N^2
            full_N_phase = 1j * self.k0 * delta_n * self.dz
            Error_Diag = 1.0 - np.exp(full_N_phase)

            # Fixed Point Iteration: u_{k+1} = u_base - M^-1 * E * u_k
            # This correctly anchors the series to the source term u_base
            for _k in range(self.n_iter - 1):
                # 1. Calculate Error term: E * u_current
                err_term = Error_Diag * u_sol

                # 2. Propagate Error: delta = M^-1 * (E * u)
                correction = apply_parallel_propagator(err_term)

                # 3. Update: u_new = u_base + correction
                # Note: M^-1 * (s - E*u) = M^-1*s - M^-1*E*u = u_base - correction
                # Wait, s = (M+E)u => Mu = s - Eu => u = M^-1 s - M^-1 Eu
                u_sol = u_base - correction

        self.beam_history = u_sol
        self.psi_final = self.beam_history[:, -1]

        return self
