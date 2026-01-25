import numpy as np
from scipy.fft import dst, idst
from typing import Any, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .pyram.PyRAM import PyRAM
from pyram_ptycho.generators import get_probe_field

# --------------------------------------------
# Solver Factory
# --------------------------------------------


def create_solver(
    solver_type: str,
    solver_params: Dict[str, Any],
    n_map: np.ndarray,
    sim_params: Dict[str, Any],
    dz: float,
    save_beam: bool = False,
):
    """
    Factory function to instantiate solvers based on type string.
    """
    s_type = solver_type.upper()

    # Common Physics Args
    common_args = {
        "n_map": n_map,
        "dx": sim_params["dx"],
        "wavelength": sim_params["wavelength"],
        "probe_dia": sim_params["probe_dia"],
        "probe_focus": sim_params["probe_focus"],
        "dz": dz,
    }

    if s_type == "PADE":
        # Handle PyRAM specific args
        store_res = (1, 1) if save_beam else None
        return PtychoPadeSolver(
            **common_args,
            pade_order=solver_params.get("pade_order", 8),
            beam_store_resolution=store_res,
        )
    if s_type == "SPECTRAL_PADE":
        return SpectralPadeSolver(
            **common_args,
            pade_order=solver_params.get("pade_order", 8),
            store_beam=save_beam,
        )

    elif s_type in ["MULTISLICE", "MS"]:
        return MultisliceSolver(
            **common_args,
            symmetric=solver_params.get("symmetric", False),
            transform_type=solver_params.get("transform_type", "DST"),
            store_beam=save_beam,
        )

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


# --------------------------------------------
# Solver Base Classes
# --------------------------------------------


class OpticalWaveSolver(ABC):
    """
    Abstract Base Class for optical wave propagation solvers.
    """

    @abstractmethod
    def run(self, psi_init: Optional[np.ndarray] = None) -> "OpticalWaveSolver":
        pass

    @abstractmethod
    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        pass

    @abstractmethod
    def get_beam_field(self) -> Optional[np.ndarray]:
        pass


# --------------------------------------------
# Spectral Pade Solver (not implemented yet)
# --------------------------------------------


class SpectralPadeSolver(OpticalWaveSolver):
    """
    Spectral Pade Solver (Placeholder).
    """

    def __init__(self, **kwargs):
        pass

    def run(self, psi_init: Optional[np.ndarray] = None) -> "SpectralPadeSolver":
        raise NotImplementedError("SpectralPadeSolver is not implemented yet.")

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError("SpectralPadeSolver is not implemented yet.")

    def get_beam_field(self) -> Optional[np.ndarray]:
        raise NotImplementedError("SpectralPadeSolver is not implemented yet.")


# --------------------------------------------
# PyRAM Pade Solver Wrapper
# --------------------------------------------


class PtychoPadeSolver(PyRAM, OpticalWaveSolver):
    """
    PyRAM Pade solver wrapper.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        probe_dia: float,
        probe_focus: float,
        pade_order: int = 8,
        beam_store_resolution: Optional[Tuple[int, int]] = None,
        dz: Optional[float] = None,
    ):
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.wavelength = wavelength
        self.dx = dx
        self.pade_order = pade_order

        # Arbitrary reference sound speed (m/s)
        c0 = 1.0

        # Grid Dimensions
        nz, nr_steps = n_map.shape
        self.nz = nz
        self._dz = dx  # Transverse step for PyRAM internal use

        # Logic: If we downsample Z, dz increases.
        self.prop_step = dz if dz is not None else dx

        # Grid Extents
        # Physical width of N points spaced by dx is (N-1)*dx
        transverse_width = (nz - 1) * dx
        propagation_dist = nr_steps * self.prop_step

        # Physics & Grids
        freq = c0 / wavelength
        z_ss = np.linspace(0, transverse_width, nz)
        rp_ss = np.linspace(0, propagation_dist, nr_steps)
        cw = c0 / n_map  # Sound speed map

        if beam_store_resolution:
            ndr, ndz = beam_store_resolution
        else:
            ndr, ndz = 1, 1

        super().__init__(
            # --- PHYSICAL CONSTANTS ---
            freq=freq,  # Frequency (Hz). Derived as c0 / wavelength.
            c0=c0,  # Reference sound speed (m/s). Serves as the base velocity for the simulation.
            # --- GEOMETRY & SOURCE ---
            zs=transverse_width
            / 2.0,  # Source Depth (m). In optics, this places the probe at the transverse center of the grid.
            zr=transverse_width
            / 2.0,  # Receiver Depth (m). Center of the detector plane.
            # --- MEDIA PROPERTIES (Optics -> Acoustics Mapping) ---
            z_ss=z_ss,  # Transverse grid coordinates (Vertical axis in acoustics, X-axis in optics).
            rp_ss=rp_ss,  # Propagation grid coordinates (Range axis in acoustics, Z-axis in optics).
            cw=cw,  # Sound Speed Map (m/s). Calculated as c0 / n_map (inverse refractive index).
            # --- BOUNDARY CONDITIONS (Dummy values for Optics) ---
            # PyRAM requires seabed definitions.
            z_sb=np.array([0.0, 2.0 * dx]),  # Seabed depth profile (Dummy).
            rp_sb=np.array([0.0]),  # Seabed range profile (Dummy).
            cb=np.array([[c0], [c0]]),  # Seabed sound speed (Dummy, set to c0).
            rhob=np.array([[1.0], [1.0]]),  # Seabed density (Dummy, set to 1.0).
            attn=np.array(
                [[0.0], [0.0]]
            ),  # Attenuation (Dummy, set to 0.0 for transparent boundary).
            rbzb=np.array(  # Domain bounds. Defines the computation box [0, width] x [0, length].
                [[0, transverse_width], [propagation_dist, transverse_width]]
            ),
            # --- SOLVER GRID SETTINGS ---
            rmax=propagation_dist,  # Maximum calculation range (Total sample thickness).
            dr=self.prop_step,  # Range step size (Longitudinal step \Delta z).
            dz=dx,  # Depth step size (Transverse pixel size \Delta x).
            # --- NUMERICAL PARAMETERS ---
            np=pade_order,  # Number of Pade terms. Higher = more accurate wide-angle propagation but slower.
            lyrw=0,  # Absorbing Layer Width. 0 = Hard-wall (Dirichlet) boundary conditions at edges.
            ndr=ndr,  # Output decimation factor for Range (save every Nth step).
            ndz=ndz,  # Output decimation factor for Depth (save every Nth pixel).
        )

        self._external_psi_init = None

    def selfs(self):
        """PyRAM Hook: Sets initial condition."""
        if self._external_psi_init is not None:
            self.u[:] = self._external_psi_init
        else:
            # Generate consistent internal probe
            x_grid = np.arange(self.nz + 2) * self.dx
            phys_center = (self.nz * self.dx) / 2.0
            self.u[:] = get_probe_field(
                x_grid, phys_center, self.probe_dia, self.probe_focus, self.wavelength
            )

        self.u[0] = 0.0
        self.u[-1] = 0.0

    def run(self, psi_init: Optional[np.ndarray] = None) -> "PtychoPadeSolver":
        if psi_init is not None:
            if len(psi_init) != self.nz:
                raise ValueError(
                    f"psi_init size {len(psi_init)} does not match solver grid {self.nz}"
                )

            self._raw_psi_init = psi_init.copy()
            # Pad with 1 pixel of zeros on each side for PyRAM Dirichlet BCs
            self._external_psi_init = np.pad(psi_init, (1, 1), mode="constant")

        self.res = super().run()

        # Inject initial condition at column 0
        if "CP Grid" in self.res:
            # Explicit slicing ensures safety if grid sizes drift slightly
            out_shape = self.res["CP Grid"].shape[0]
            if self._raw_psi_init is not None:
                # Truncate or pad input to match output exactly
                inj = self._raw_psi_init[:out_shape]
                self.res["CP Grid"][: len(inj), 0] = inj

        return self

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        wave = self.u[1:-1].copy()
        if n_crop is not None:
            start = (len(wave) - n_crop) // 2
            return wave[start : start + n_crop]
        return wave

    def get_beam_field(self) -> Optional[np.ndarray]:
        res = getattr(self, "res", {})
        if "CP Grid" in res:
            return res["CP Grid"]
        return getattr(self, "cpg", None)


# --------------------------------------------
# Multislice Solver
# --------------------------------------------


class MultisliceSolver(OpticalWaveSolver):
    """
    Standard Split-Step Fourier Solver (Multislice).
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
        dz: Optional[float] = None,
    ):
        self.n_map = n_map
        self.nx, self.nz_steps = n_map.shape
        self.dx = dx
        self.dz = dz if dz is not None else dx
        self.k0 = 2 * np.pi / wavelength
        self.symmetric = symmetric
        self.transform_type = transform_type
        self.store_beam = store_beam
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.wavelength = wavelength
        self.total_width = self.nx * dx
        self._kernel_cache = {}
        self.psi_final = None
        self.beam_history = None

    def _get_propagation_kernel(self, dz: float) -> np.ndarray:
        key = (self.transform_type, dz)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        if self.transform_type == "FFT":
            fx = np.fft.fftfreq(self.nx, d=self.dx)
            kx = 2 * np.pi * fx
        elif self.transform_type == "DST":
            modes = np.arange(self.nx) + 1
            kx = np.pi * modes / (self.nx * self.dx)

        inside = self.k0**2 - kx**2
        sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
        Lambda = 1j * (sqrt_term - self.k0)
        H = np.exp(Lambda * dz).astype(np.complex128)
        self._kernel_cache[key] = H
        return H

    def _apply_diffraction(self, psi: np.ndarray, dz: float) -> np.ndarray:
        H = self._get_propagation_kernel(dz)
        if self.transform_type == "FFT":
            return np.fft.ifft(np.fft.fft(psi) * H)
        elif self.transform_type == "DST":
            psi_real = dst(np.real(psi), type=2, norm="ortho")
            psi_imag = dst(np.imag(psi), type=2, norm="ortho")
            psi_spectral = (psi_real + 1j * psi_imag) * H
            out_real = idst(np.real(psi_spectral), type=2, norm="ortho")
            out_imag = idst(np.imag(psi_spectral), type=2, norm="ortho")
            return out_real + 1j * out_imag
        return psi

    def run(self, psi_init: Optional[np.ndarray] = None) -> "MultisliceSolver":
        if psi_init is not None:
            psi = psi_init.astype(complex)
        else:
            center_x = self.total_width / 2.0
            x_coords = np.arange(self.nx) * self.dx
            psi = get_probe_field(
                x_coords, center_x, self.probe_dia, self.probe_focus, self.wavelength
            )
            psi = psi.astype(complex)

        if self.store_beam:
            self.beam_history = np.zeros((self.nx, self.nz_steps), dtype=complex)

        for i in range(self.nz_steps):
            step_dist = (self.dz / 2.0) if self.symmetric else self.dz
            psi = self._apply_diffraction(psi, step_dist)
            n_slice = self.n_map[:, i]
            psi *= np.exp(1j * self.k0 * (n_slice - 1.0) * self.dz)
            if self.symmetric:
                psi = self._apply_diffraction(psi, step_dist)
            if self.store_beam:
                self.beam_history[:, i] = psi

        self.psi_final = psi
        return self

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        if n_crop is not None and self.psi_final is not None:
            start = (len(self.psi_final) - n_crop) // 2
            return self.psi_final[start : start + n_crop]
        return self.psi_final

    def get_beam_field(self) -> Optional[np.ndarray]:
        return self.beam_history
