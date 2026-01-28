from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ptycho_fwd_bench.generators import get_probe_field


class OpticalWaveSolver(ABC):
    """
    Base class handling common optical parameters, state, and probe generation.
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
        self.n_map = n_map
        self.nx, self.nz_steps = n_map.shape
        self.dx = dx
        self.dz = dz
        self.wavelength = wavelength
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.store_beam = store_beam

        # Derived constants
        self.k0 = 2 * np.pi / wavelength
        self.k0sq = self.k0**2
        self.total_width = self.nx * self.dx

        # State
        self.psi_final: Optional[np.ndarray] = None
        self.beam_history: Optional[np.ndarray] = None

    @abstractmethod
    def run(self, psi_init: Optional[np.ndarray] = None) -> "OpticalWaveSolver":
        pass

    def initialize_wavefront(self, psi_init: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns the initial wavefront.
        If psi_init is None, generates a probe field using class parameters.
        """
        if psi_init is not None:
            if len(psi_init) != self.nx:
                raise ValueError(
                    f"Input field size {len(psi_init)} does not match grid {self.nx}"
                )
            return psi_init.astype(complex)

        # Generate default probe
        center_x = self.total_width / 2.0
        x_coords = np.arange(self.nx) * self.dx

        psi = get_probe_field(
            x_coords, center_x, self.probe_dia, self.probe_focus, self.wavelength
        )
        return psi.astype(complex)

    def get_exit_wave(self, n_crop: Optional[int] = None) -> np.ndarray:
        if self.psi_final is None:
            raise RuntimeError("Run the solver before requesting exit wave.")
        if n_crop is not None:
            start = (len(self.psi_final) - n_crop) // 2
            return self.psi_final[start : start + n_crop]
        return self.psi_final

    def get_beam_field(self) -> Optional[np.ndarray]:
        return self.beam_history
