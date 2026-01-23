import numpy as np
from scipy.fft import dst, idst
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from pyram.PyRAM import PyRAM
from pyram_ptycho.generators import get_probe_field

# =============================================================================
# 1. Abstract Base Class (The Interface)
# =============================================================================


class OpticalWaveSolver(ABC):
    """
    Abstract Base Class defining the interface for X-ray propagation solvers.

    This ensures that different solver implementations (e.g., Multislice, Pade)
    expose a consistent API for benchmarking and integration.
    """

    @abstractmethod
    def run(self) -> "OpticalWaveSolver":
        """
        Executes the simulation pipeline.

        Returns:
            OpticalWaveSolver: Returns self to allow method chaining.
        """
        pass

    @abstractmethod
    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        """
        Retrieves the 1D complex field at the end of the sample (Exit Surface Wave).

        Args:
            n_crop (int, optional): Number of pixels to crop to (centered).
                                    Useful for removing padding.

        Returns:
            np.ndarray: 1D complex array of the wavefront.
        """
        pass

    @abstractmethod
    def get_beam_field(self) -> Optional[np.ndarray]:
        """
        Retrieves the full 2D beam propagation history.

        Returns:
            np.ndarray: 2D complex array (Transverse, Propagation) if storage was enabled.
            None: If beam storage was disabled.
        """
        pass


# =============================================================================
# 2. Pade Solver (PyRAM Wrapper)
# =============================================================================


class PtychoPadeSolver(PyRAM, OpticalWaveSolver):
    """
    A concrete implementation of an OpticalWaveSolver using the PyRAM library.

    This class acts as an adapter, mapping X-ray optical parameters (refractive index,
    wavelength) to the acoustic parabolic equation parameters (sound speed, frequency)
    expected by the PyRAM backend.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        c0: float = 1500.0,
        pade_order: int = 8,
        beam_store_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Initializes the Pade Solver.

        Args:
            n_map (np.ndarray): 2D Complex Refractive Index Map (nz_transverse, nr_propagation).
            dx (float): Pixel size in meters (assumed isotropic).
            wavelength (float): Radiation wavelength in meters.
            probe_dia (float): Diameter of the probe aperture in meters.
            probe_focus (float): Focal distance of the probe in meters.
            c0 (float): Reference velocity constant for the acoustic analogy. Defaults to 1500.0.
            pade_order (int): Order of the Pade approximation (higher is more accurate but slower).
            beam_store_resolution (tuple, optional): (ndr, ndz) decimation factors for storing the beam.
                                                     If None, defaults to (1, 1).
        """
        # --- 1. Store Optical Physics Parameters ---
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.wavelength = wavelength

        # --- 2. Geometry Setup ---
        # Convention: Z is propagation direction, X is transverse direction.
        nz, nr_steps = n_map.shape
        transverse_width = nz * dx
        propagation_dist = nr_steps * dx

        # --- 3. Parameter Translation (Optics to Acoustics) ---
        # PyRAM solves the acoustic parabolic equation. We map optical parameters:
        # Frequency = Reference Velocity / Wavelength
        freq = c0 / wavelength

        # Generate Coordinate Grids
        # z_ss corresponds to the transverse depth in acoustics
        z_ss = np.linspace(0, transverse_width, nz)
        # rp_ss corresponds to the propagation range in acoustics
        rp_ss = np.linspace(0, propagation_dist, nr_steps)

        # Map Refractive Index (n) to Sound Speed (cw)
        # Relationship: v = c0 / n
        cw = c0 / n_map

        # --- 4. Boundary Layers (PyRAM Requirements) ---
        # Define bottom boundary properties to satisfy PyRAM structure.
        # These are effectively dummy values for transmission mode.
        z_sb = np.array([0.0, 2.0 * dx])
        rp_sb = np.array([0.0])
        cb = np.array([[c0], [c0]])  # Bottom velocity
        rhob = np.array([[1.0], [1.0]])  # Bottom density
        attn_sb = np.array([[0.0], [0.0]])  # Bottom attenuation

        # Define Computational Domain Bounds: [[r_min, z_min], [r_max, z_max]]
        rbzb = np.array([[0, transverse_width], [propagation_dist, transverse_width]])

        # --- 5. Output Storage Settings ---
        if beam_store_resolution:
            ndr, ndz = beam_store_resolution
        else:
            # Default to full resolution (1, 1)
            ndr, ndz = 1, 1

        # --- 6. Initialize Parent PyRAM ---
        super().__init__(
            freq=freq,  # Frequency (Hz)
            zs=transverse_width / 2.0,  # Source depth (Transverse center)
            zr=transverse_width / 2.0,  # Receiver depth (Transverse center)
            z_ss=z_ss,  # Transverse grid points array
            rp_ss=rp_ss,  # Propagation grid points array
            cw=cw,  # Sound speed map (derived from Refractive Index)
            attn=attn_sb,  # Attenuation map (optional)
            z_sb=z_sb,  # Bottom boundary depth grid
            rp_sb=rp_sb,  # Bottom boundary range grid
            cb=cb,  # Bottom boundary sound speed
            rhob=rhob,  # Bottom boundary density
            rbzb=rbzb,  # Domain boundaries
            rmax=propagation_dist,  # Total propagation distance
            dr=propagation_dist / nr_steps,  # Step size in Propagation (Optics Z)
            dz=dx,  # Step size in Transverse (Optics X)
            c0=c0,  # Reference sound speed
            np=pade_order,  # Pade coefficients order (Accuracy)
            lyrw=0,  # Absorbing layer width (0 = Dirichlet Wall)
            ndr=ndr,  # Output decimation factor (Propagation)
            ndz=ndz,  # Output decimation factor (Transverse)
        )

    def selfs(self):
        """
        PyRAM Hook: Generates the initial condition (Source Field) at r=0.

        This method is called automatically by the parent PyRAM.run() method.
        It generates the optical probe field and applies Dirichlet boundary conditions.
        """
        # PyRAM internal grid usually includes boundary points.
        # self.nz and self._dz are properties of the parent PyRAM class.
        x_grid = np.arange(self.nz + 2) * self._dz

        # Generate the optical probe
        field = get_probe_field(
            x_grid,
            self._zs,  # Source center defined in init
            self.probe_dia,
            self.probe_focus,
            self.wavelength,
        )

        self.u[:] = field

        # Enforce Hard-wall boundaries (Dirichlet BCs) at the edges
        self.u[0] = 0.0
        self.u[-1] = 0.0

    def run(self) -> "PtychoPadeSolver":
        """
        Runs the PyRAM simulation.
        """
        super().run()
        return self

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        """
        Retrieves the 1D complex field at the end of the simulation.

        Removes the boundary points used by the Finite Difference grid.
        """
        # Slice [1:-1] to remove boundary points (u[0] and u[-1])
        wave = self.u[1:-1].copy()

        if n_crop is not None:
            if len(wave) >= n_crop:
                return wave[:n_crop]
            else:
                raise ValueError(
                    f"Output grid size ({len(wave)}) is smaller than requested crop ({n_crop})."
                )
        return wave

    def get_beam_field(self) -> Optional[np.ndarray]:
        """
        Extracts the stored 2D beam field if available.

        Returns:
            np.ndarray: Shape (Transverse, Propagation).
        """
        # PyRAM stores 'out_cp' or 'CP Grid' in shape (Depth, Range).
        # We return it directly as (Transverse, Propagation).

        if hasattr(self, "out_cp") and self.out_cp is not None:
            return self.out_cp
        if "CP Grid" in self.res:
            return self.res["CP Grid"]
        return None


# =============================================================================
# 3. Multislice Solver (Native Implementation)
# =============================================================================


class MultisliceSolver(OpticalWaveSolver):
    """
    Standard Split-Step Fourier Method (Multislice) solver.

    Propagates the wavefield by alternating between the spectral domain (Diffraction)
    and the spatial domain (Refraction).
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        symmetric: bool = True,
        transform_type: str = "DST",
        store_beam: bool = False,
    ):
        """
        Initializes the Multislice Solver.

        Args:
            n_map (np.ndarray): 2D Complex Refractive Index Map (nz, nr_steps).
            dx (float): Pixel size in meters.
            wavelength (float): Radiation wavelength in meters.
            probe_dia (float): Probe diameter in meters.
            probe_focus (float): Probe focal distance in meters.
            symmetric (bool): If True, uses the Symmetrized Split-Step (Prop/2 -> Phase -> Prop/2).
            transform_type (str): "FFT" for Periodic BCs, "DST" for Dirichlet (Hard Wall) BCs.
            store_beam (bool): If True, records the wavefront at every step.
        """
        self.n_map = n_map
        self.nx, self.nz_steps = n_map.shape
        self.dx = dx
        self.k0 = 2 * np.pi / wavelength
        self.symmetric = symmetric
        self.transform_type = transform_type
        self.store_beam = store_beam

        # Probe Parameters
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.wavelength = wavelength
        self.total_width = self.nx * dx

        # Internal Cache & Results Storage
        self._kernel_cache = {}
        self.psi_final = None
        self.beam_history = None

    def _get_propagation_kernel(self, dz: float) -> np.ndarray:
        """
        Computes the free-space propagation kernel H = exp(i * kz * dz).

        Results are cached to improve performance during iteration.

        Args:
            dz (float): Propagation step size.

        Returns:
            np.ndarray: Complex propagator in Fourier/Spectral domain.
        """
        key = (self.transform_type, dz)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        L = self.nx * self.dx

        # 1. Generate Wavenumbers (kx)
        if self.transform_type == "FFT":
            fx = np.fft.fftfreq(self.nx, d=self.dx)
            kx = 2 * np.pi * fx
        elif self.transform_type == "DST":
            # Type 2 DST modes: kx = (pi * (n + 1)) / L
            modes = np.arange(self.nx) + 1
            kx = np.pi * modes / L
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

        # 2. Exact Helmholtz propagator setup
        # kz = sqrt(k0^2 - kx^2)
        inside = self.k0**2 - kx**2

        # Clip negative values to 0 to handle evanescent waves (non-propagating)
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))

        # Carrier removal: exp(i * (kz - k0) * dz)
        # This transforms the operator to the frame of reference moving at k0
        Lambda = 1j * (sqrt_term - self.k0)
        H = np.exp(Lambda * dz).astype(np.complex128)

        self._kernel_cache[key] = H
        return H

    def _apply_diffraction(self, psi: np.ndarray, dz: float) -> np.ndarray:
        """
        Applies the diffraction operator in the spectral domain.

        Args:
            psi (np.ndarray): Current wavefront in spatial domain.
            dz (float): Distance to propagate.

        Returns:
            np.ndarray: Propagated wavefront in spatial domain.
        """
        H = self._get_propagation_kernel(dz)

        if self.transform_type == "FFT":
            # Standard Periodic Propagation
            return np.fft.ifft(np.fft.fft(psi) * H)

        elif self.transform_type == "DST":
            # Unitary DST-II (Discrete Sine Transform)
            # Decompose into Real and Imaginary parts for the transform
            psi_real = dst(np.real(psi), type=2, norm="ortho")
            psi_imag = dst(np.imag(psi), type=2, norm="ortho")

            # Apply Propagator
            psi_spectral = (psi_real + 1j * psi_imag) * H

            # Inverse Transform
            out_real = idst(np.real(psi_spectral), type=2, norm="ortho")
            out_imag = idst(np.imag(psi_spectral), type=2, norm="ortho")

            return out_real + 1j * out_imag

        return psi

    def run(self, sample_thick: Optional[float] = None) -> "MultisliceSolver":
        """
        Generates the probe and propagates it through the sample map.

        Args:
            sample_thick (float, optional): Total physical thickness of the sample.
                                            If None, derived from n_map steps * dx.
        """
        # 1. Generate Probe (Initial Condition)
        # Center the probe in the simulation window
        center_x = self.total_width / 2.0

        # Generate coordinates aligned with pixel centers
        x_coords = (np.arange(self.nx) + 0.5) * self.dx

        psi = get_probe_field(
            x_coords, center_x, self.probe_dia, self.probe_focus, self.wavelength
        )
        psi = psi.astype(complex)

        # 2. Determine Step Size (dz)
        if sample_thick is None:
            # Assume isotropic pixels if not specified
            dz = self.dx
        else:
            # Distribute thickness over the available slices
            dz = sample_thick / self.nz_steps

        # 3. Initialize Beam Storage
        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)

        # 4. Main Propagation Loop
        for i in range(self.nz_steps):
            # Split the diffraction step if symmetric (Lie-Trotter / Strang Splitting)
            step_dist = (dz / 2.0) if self.symmetric else dz

            # Step A: Diffraction (1/2)
            psi = self._apply_diffraction(psi, step_dist)

            # Step B: Refraction (Phase mask)
            # phase = k0 * delta_n * dz
            n_slice = self.n_map[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * dz)

            # Step C: Diffraction (2/2)
            if self.symmetric:
                psi = self._apply_diffraction(psi, step_dist)

            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi
        return self

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        """
        Retrieves the final wavefront.
        """
        if self.psi_final is None:
            raise RuntimeError("Run method must be called before getting exit wave.")

        wave = self.psi_final.copy()

        # Multislice generally maintains the grid size, so simple bounds check
        if n_crop is not None:
            if len(wave) < n_crop:
                raise ValueError("Grid is smaller than crop size.")
            # Center crop logic could be added here if grids differ,
            # but usually they match in this implementation.

        return wave

    def get_beam_field(self) -> Optional[np.ndarray]:
        """
        Retrieves the beam history.
        """
        if self.store_beam and self.beam_history is not None:
            return self.beam_history
        return None
