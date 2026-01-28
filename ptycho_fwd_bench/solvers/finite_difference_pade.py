from typing import Optional, Tuple

import numpy as np

from ptycho_fwd_bench.pyram.PyRAM import PyRAM

from .base import OpticalWaveSolver


class FiniteDifferencePadeSolver(PyRAM, OpticalWaveSolver):
    """
    Ptycho solver using the Pade beam propagation method via PyRAM.

    Parameters
    ----------
    n_map : np.ndarray
        Complex refractive index map of shape (nz, nr_steps).
    dx : float
        Spatial sampling interval in x (um).
    wavelength : float
        Wavelength of the optical wave (um).
    probe_dia : float
        Diameter of the probe beam (um).
    probe_focus : float
        Focal distance of the probe beam (um).
    pade_order : int
        Order of the Pade approximation (number of terms).
    beam_store_resolution : Optional[Tuple[int, int]]
        Resolution for storing beam history as (ndr, ndz). If None, no storage.
    dz : Optional[float]
        Propagation step size in z (um). If None, defaults to dx.
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
        self.nx = nz  # For probe generation compatibility
        self.nz = nz
        self._dz = dx  # Transverse step for PyRAM internal use
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
            self.store_beam = True
        else:
            ndr, ndz = 1, 1
            self.store_beam = False

        PyRAM.__init__(
            self,
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

        # --- 2. Initialize OpticalWaveSolver (Parent 2) ---
        # Explicit call ensures self.nx, self.k0, self.total_width are set
        OpticalWaveSolver.__init__(
            self,
            n_map=n_map,
            dx=dx,
            wavelength=wavelength,
            dz=self.prop_step,
            probe_dia=probe_dia,
            probe_focus=probe_focus,
            store_beam=(beam_store_resolution is not None),
        )
        self._external_psi_init = None

    def selfs(self):
        """
        PyRAM Hook: Sets initial condition.
        This is called internally by PyRAM.run() -> PyRAM.setup().
        At this point, self.u has been allocated.
        """
        if self._external_psi_init is not None:
            self.u[:] = self._external_psi_init
        else:
            # Fallback: Generate probe if run() wasn't used to prep
            # This relies on OpticalWaveSolver attributes (total_width, etc.)
            psi = self.initialize_wavefront(None)
            self.u[:] = np.pad(psi, (1, 1), mode="constant")

        # Enforce Dirichlet BCs
        self.u[0] = 0.0
        self.u[-1] = 0.0

    def run(
        self, psi_init: Optional[np.ndarray] = None
    ) -> "FiniteDifferencePadeSolver":
        # 1. Prepare Initial Wavefront using Base Class Logic
        psi_standard = self.initialize_wavefront(psi_init)

        # 2. Store raw version for output injection later
        self._raw_psi_init = psi_standard.copy()

        # 3. Store padded version for PyRAM engine (N+2)
        # We do NOT set self.u here; selfs() will pick this up later.
        self._external_psi_init = np.pad(psi_standard, (1, 1), mode="constant")

        # 4. Run PyRAM
        # This calls setup() -> selfs() (sets u) -> propagation loop
        self.res = PyRAM.run(self)

        # 5. Inject clean initial condition into result grid (visual consistency)
        if "CP Grid" in self.res:
            out_shape = self.res["CP Grid"].shape[0]
            inj = self._raw_psi_init[:out_shape]
            self.res["CP Grid"][: len(inj), 0] = inj

        # 6. Extract Final Result (remove padding)
        self.psi_final = self.u[1:-1].copy()

        # 7. Map PyRAM results to OpticalWaveSolver expected history
        if self.store_beam and "CP Grid" in self.res:
            self.beam_history = self.res["CP Grid"]

        return self
