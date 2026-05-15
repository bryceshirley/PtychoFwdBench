import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

# Updated import for the 3D generator
from ptycho_fwd_bench.dim_3.generators_3d import get_2d_airy_probe


class OpticalWaveSolver3D(ABC):
    """
    Base class handling common optical parameters, state, and probe generation for 3D volumes.
    """

    def __init__(
        self,
        n_map: np.ndarray,
        dx: float,
        wavelength: float,
        dz: float,
        probe_dia: float = 0,
        probe_focus: float = 0,
        store_beam: bool = False,
    ):
        self.complex_t = np.complex128
        self.real_t = np.float64
        if np.any(np.abs(n_map) < 1e-9):
            logging.warning("Found zeros in n_map! Replacing with 1.0 (Vacuum).")
            n_map = np.where(np.abs(n_map) < 1e-9, 1.0, n_map)

        if n_map.ndim != 3:
            raise ValueError(f"n_map must be 3D (Ny, Nx, Nz). Got shape {n_map.shape}")

        self.n_map = n_map
        self.ny, self.nx, self.nz_steps = n_map.shape
        self.dx = dx
        self.dz = dz
        self.wavelength = wavelength
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.store_beam = store_beam

        # Derived constants
        self.k0 = 2 * np.pi / wavelength
        self.k0sq = self.k0**2

        # Assuming isotropic transverse pixels (dx = dy)
        self.total_width_x = self.nx * self.dx
        self.total_width_y = self.ny * self.dx

        # State
        self.psi_final: Optional[np.ndarray] = None
        self.beam_history: Optional[np.ndarray] = None

    @abstractmethod
    def run(self, psi_init: Optional[np.ndarray] = None) -> "OpticalWaveSolver3D":
        pass

    def initialize_wavefront(self, psi_init: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns the initial wavefront.
        If psi_init is None, generates a 2D probe field using class parameters.
        """
        if psi_init is not None:
            if psi_init.shape != (self.ny, self.nx):
                raise ValueError(
                    f"Input field size {psi_init.shape} does not match grid {(self.ny, self.nx)}"
                )
            return psi_init.astype(complex)

        # Generate default 2D Airy probe
        psi = get_2d_airy_probe(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            diameter=self.probe_dia,
            focus=self.probe_focus,
            wavelength=self.wavelength,
        )
        return psi.astype(complex)

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        """
        Returns the final exit wave, optionally center-cropped in 2D.
        """
        if self.psi_final is None:
            raise RuntimeError("Run the solver before requesting exit wave.")

        if n_crop is not None:
            start_y = (self.ny - n_crop) // 2
            start_x = (self.nx - n_crop) // 2
            return self.psi_final[
                start_y : start_y + n_crop, start_x : start_x + n_crop
            ]

        return self.psi_final

    def get_beam_field(self) -> Optional[np.ndarray]:
        return self.beam_history
