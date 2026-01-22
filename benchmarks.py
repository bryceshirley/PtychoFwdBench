import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from time import process_time
from pyram.PyRAM import PyRAM

# =============================================================================
# 1. SOLVER CLASSES
# =============================================================================

class PtychoProbePyRAM(PyRAM):
    """
    Extended PyRAM class for Ptychography.
    """
    def __init__(self, probe_dia, probe_focus, wavelength, *args, **kwargs):
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.wavelength = wavelength
        super().__init__(*args, **kwargs)

    def selfs(self):
        """Injects Gaussian beam with defocus curvature."""
        # Transverse grid (x-axis in ptycho, z-axis in RAM)
        x_grid = np.arange(self.nz + 2) * self._dz
        delta_x = x_grid - self._zs
        
        # Gaussian Amplitude
        sigma = self.probe_dia / 2.0
        amplitude = np.exp(-0.5 * (delta_x / sigma)**2)
        
        # Quadratic Phase
        k0 = 2 * np.pi / self.wavelength
        if abs(self.probe_focus) < 1e-12:
            phase = 1.0
        else:
            R = -self.probe_focus 
            curvature = k0 / (2.0 * R)
            phase = np.exp(1j * curvature * delta_x**2)
        
        self.u[:] = amplitude * phase
        self.u[0] = 0.0 # Boundary condition

    def get_exit_wave(self, n_crop=None):
        """
        Returns the complex wavefield.
        Args:
            n_crop (int): If set, crops the output to this many pixels 
                          (removes PyRAM's absorbing layer padding).
        """
        # PyRAM stores boundary points at 0 and -1.
        wave = self.u[1:-1].copy()
        
        if n_crop is not None:
            # Crop to the physical domain size
            if len(wave) >= n_crop:
                return wave[:n_crop]
            else:
                raise ValueError(f"PyRAM grid ({len(wave)}) smaller than requested crop ({n_crop}).")
        return wave


class MultisliceSolver:
    def __init__(self, nx, dx, wavelength, symmetric=True):
        """
        Split-Step Fourier Solver.
        symmetric=True:  Diff(dz/2) -> Refr(dz) -> Diff(dz/2) [Strang]
        symmetric=False: Diff(dz) -> Refr(dz) [Lie-Trotter]
        """
        self.nx = nx
        self.dx = dx
        self.k0 = 2 * np.pi / wavelength
        self.symmetric = symmetric
        
        # Precompute Diffraction Operator (Wide-Angle / Exact)
        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        # Exact phase: sqrt(k0^2 - kx^2) - k0
        kz_sq = self.k0**2 - self.kx**2
        self.kz_envelope = np.sqrt(kz_sq.astype(complex)) - self.k0
        
    def propagate(self, psi_in, n_map, sample_thick):
        """
        psi_in: 1D complex array
        n_map: 2D array (nx, nz_steps) - The 'coarse' interpolated object
        sample_thick: float
        """
        nz_steps = n_map.shape[1]
        dz = sample_thick / nz_steps
        psi = psi_in.astype(complex)
        
        # Precompute operators for this dz
        if self.symmetric:
            diffract_op = np.exp(1j * (dz / 2.0) * self.kz_envelope)
        else:
            diffract_op = np.exp(1j * dz * self.kz_envelope)
        
        t0 = process_time()
        
        for i in range(nz_steps):
            # 1. Diffraction
            if self.symmetric:
                psi = np.fft.ifft(np.fft.fft(psi) * diffract_op)
            else:
                psi = np.fft.ifft(np.fft.fft(psi) * diffract_op)
            
            # 2. Refraction
            # n_map is (transverse, propagation)
            n_slice = n_map[:, i]
            refraction_phase = np.exp(1j * self.k0 * (n_slice - 1.0) * dz)
            psi *= refraction_phase
            
            # 3. Second Diffraction (only for Symmetric)
            if self.symmetric:
                psi = np.fft.ifft(np.fft.fft(psi) * diffract_op)
            
        t_elapsed = process_time() - t0
        return psi, t_elapsed


# =============================================================================
# 2. DATA GENERATION & UTILS
# =============================================================================

def generate_blob_phantom(nz, nr, n_background=1.0, delta_n=0.01):
    """Generates high-res refractive index map."""
    n_map = np.ones((nz, nr)) * n_background
    z_idx, x_idx = np.meshgrid(np.arange(nr), np.arange(nz))
    np.random.seed(42)
    
    for _ in range(25): 
        cx = np.random.randint(0, nz)
        cz = np.random.randint(0, nr)
        radius = np.random.randint(nz // 20, nz // 8)
        dist = np.sqrt((x_idx - cx)**2 + (z_idx - cz)**2)
        mask = dist <= radius
        taper = np.cos(np.pi * dist[mask] / (2 * radius))
        n_map[mask] += delta_n * taper
    return n_map

def interpolate_to_coarse(n_map_fine, n_steps_coarse):
    """
    Interpolates the fine object map to a coarser grid in the Z-direction.
    """
    nx, nz_fine = n_map_fine.shape
    zoom_factor = n_steps_coarse / nz_fine
    n_map_coarse = scipy.ndimage.zoom(n_map_fine, (1, zoom_factor), order=1) # Order 1 is faster/safer for phantoms
    return n_map_coarse

# =============================================================================
# 3. BENCHMARK SUITE
# =============================================================================

def run_benchmark():
    # --- Parameters ---
    um = 1e-6
    wavelength = 0.635 * um
    c0 = 1500.0
    freq = c0 / wavelength
    
    sample_width = 150.0 * um
    sample_thick = 40.0 * um
    
    # 1. Define Fine Mesh (Ground Truth Resolution)
    n_transverse = 1024 # Fixed transverse resolution
    n_prop_fine = 4000  # Very fine z-steps
    
    dx = sample_width / n_transverse
    
    # 2. Generate Fine Object
    print(f"Generating Fine Phantom ({n_transverse}x{n_prop_fine})...")
    n_map_fine = generate_blob_phantom(n_transverse, n_prop_fine, delta_n=0.05)
    
    # Setup PyRAM Environment arrays for Fine Mesh
    z_ss = np.linspace(0, sample_width, n_transverse)
    rp_ss_fine = np.linspace(0, sample_thick, n_prop_fine)
    cw_fine = c0 / n_map_fine
    
    # Common PyRAM Setup
    z_sb = np.array([0.0, 10.0*um])
    rp_sb = np.array([0.0])
    cb = np.array([[c0], [c0]])
    rhob = np.array([[1.0], [1.0]])
    attn = np.array([[0.0], [0.0]])
    rbzb = np.array([[0.0, sample_width], [sample_thick, sample_width]])
    
    # 3. Generate Ground Truth (High Order Padé on Fine Mesh)
    print("Computing Ground Truth (Padé Order 8, Fine Mesh)...")
    gt_solver = PtychoProbePyRAM(
        probe_dia=15.0 * um, probe_focus=-10.0 * um, wavelength=wavelength,
        freq=freq, zs=sample_width/2, zr=sample_width/2,
        z_ss=z_ss, rp_ss=rp_ss_fine, cw=cw_fine,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn, rbzb=rbzb,
        rmax=sample_thick, dr=sample_thick/n_prop_fine, dz=dx, c0=c0, np=8 
    )
    gt_solver.run()
    # FIX: Crop GT to physical grid size
    psi_gt = gt_solver.get_exit_wave(n_crop=n_transverse)
    
    # --- Define Experiments ---
    step_counts = [16, 32, 64, 128, 256, 512]  # Number of z-steps to test
    methods = {
        "Multislice (Std)": {"err": [], "time": [], "style": "bo--"},
        "Multislice (Sym)": {"err": [], "time": [], "style": "go-"},
        "Padé [1,1]":       {"err": [], "time": [], "style": "rs-"},
        "Padé [8,8]":       {"err": [], "time": [], "style": "ms-"}
    }
    
    # Initialize Probe for Multislice (Same parameters)
    x = np.arange(n_transverse) * dx - sample_width/2
    k0 = 2*np.pi/wavelength
    sigma = (15.0*um)/2
    psi_0 = np.exp(-0.5*(x/sigma)**2) * np.exp(1j * (k0/(2*10.0*um)) * x**2)

    # 4. Loop over Step Sizes
    for steps in step_counts:
        print(f"Benchmarking N={steps}...")
        dz_curr = sample_thick / steps
        
        # A. Interpolate Object for this resolution
        n_map_coarse = interpolate_to_coarse(n_map_fine, steps)
        
        # B. Run Multislice Standard
        ms_std = MultisliceSolver(n_transverse, dx, wavelength, symmetric=False)
        psi_ms1, t_ms1 = ms_std.propagate(psi_0, n_map_coarse, sample_thick)
        err_ms1 = np.linalg.norm(psi_ms1 - psi_gt) / np.linalg.norm(psi_gt)
        methods["Multislice (Std)"]["err"].append(err_ms1)
        methods["Multislice (Std)"]["time"].append(t_ms1)
        
        # C. Run Multislice Symmetric
        ms_sym = MultisliceSolver(n_transverse, dx, wavelength, symmetric=True)
        psi_ms2, t_ms2 = ms_sym.propagate(psi_0, n_map_coarse, sample_thick)
        err_ms2 = np.linalg.norm(psi_ms2 - psi_gt) / np.linalg.norm(psi_gt)
        methods["Multislice (Sym)"]["err"].append(err_ms2)
        methods["Multislice (Sym)"]["time"].append(t_ms2)
        
        # D. Run Padé [1,1] (Low Order)
        rp_ss_coarse = np.linspace(0, sample_thick, steps)
        cw_coarse = c0 / n_map_coarse
        
        t0 = process_time()
        pade_low = PtychoProbePyRAM(
            probe_dia=15.0*um, probe_focus=-10.0*um, wavelength=wavelength,
            freq=freq, zs=sample_width/2, zr=sample_width/2,
            z_ss=z_ss, rp_ss=rp_ss_coarse, cw=cw_coarse,
            z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn, rbzb=rbzb,
            rmax=sample_thick, dr=dz_curr, dz=dx, c0=c0, np=1
        )
        pade_low.run()
        t_pade1 = process_time() - t0
        # FIX: Crop output
        psi_pade1 = pade_low.get_exit_wave(n_crop=n_transverse)
        err_pade1 = np.linalg.norm(psi_pade1 - psi_gt) / np.linalg.norm(psi_gt)
        methods["Padé [1,1]"]["err"].append(err_pade1)
        methods["Padé [1,1]"]["time"].append(t_pade1)
        
        # E. Run Padé [8,8] (High Order)
        t0 = process_time()
        pade_high = PtychoProbePyRAM(
            probe_dia=15.0*um, probe_focus=-10.0*um, wavelength=wavelength,
            freq=freq, zs=sample_width/2, zr=sample_width/2,
            z_ss=z_ss, rp_ss=rp_ss_coarse, cw=cw_coarse,
            z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn, rbzb=rbzb,
            rmax=sample_thick, dr=dz_curr, dz=dx, c0=c0, np=8
        )
        pade_high.run()
        t_pade8 = process_time() - t0
        # FIX: Crop output
        psi_pade8 = pade_high.get_exit_wave(n_crop=n_transverse)
        err_pade8 = np.linalg.norm(psi_pade8 - psi_gt) / np.linalg.norm(psi_gt)
        methods["Padé [8,8]"]["err"].append(err_pade8)
        methods["Padé [8,8]"]["time"].append(t_pade8)

    # =========================================================================
    # 4. PLOTTING
    # =========================================================================
    
    dz_values = [sample_thick / n for n in step_counts]
    
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Error vs Step Size (dz)
    ax1 = plt.subplot(2, 2, 1)
    for name, data in methods.items():
        ax1.loglog(dz_values, data["err"], data["style"], label=name)
    ax1.set_xlabel(r"Step Size $\Delta z$ (m)")
    ax1.set_ylabel("Relative Error (L2 Norm)")
    ax1.set_title("Convergence Analysis")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.invert_xaxis() # Smaller steps to the right
    ax1.legend()
    
    # Plot 2: Error vs Computation Time
    ax2 = plt.subplot(2, 2, 2)
    for name, data in methods.items():
        ax2.loglog(data["time"], data["err"], data["style"], label=name)
    ax2.set_xlabel("Computation Time (s)")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Efficiency Analysis")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    
    # Plot 3: Object Phantom
    ax3 = plt.subplot(2, 2, 3)
    extent = [0, sample_thick/um, sample_width/um, 0]
    ax3.imshow(n_map_fine, extent=extent, aspect='auto', cmap='bone')
    ax3.set_title(f"Fine Phantom n(x,z) ({n_transverse}x{n_prop_fine})")
    ax3.set_xlabel("Z (um)")
    ax3.set_ylabel("X (um)")
    
    # Plot 4: Reconstruction Comparison (at lowest resolution N=50)
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(np.abs(psi_gt), 'k-', label="Ground Truth")
    ax4.plot(np.abs(psi_ms1), 'b--', label=f"Multislice (N={step_counts[-1]})")
    ax4.plot(np.abs(psi_pade8), 'm:', label=f"Padé [8,8] (N={step_counts[-1]})")
    ax4.set_title(f"Exit Wave Amplitude (at N={step_counts[-1]})")
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()