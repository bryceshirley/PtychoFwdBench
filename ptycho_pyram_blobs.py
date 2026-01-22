import numpy as np
import matplotlib.pyplot as plt
from pyram.PyRAM import PyRAM

# --- 1. Define Probe Solver (Same as before) ---
class PtychoProbePyRAM(PyRAM):
    def __init__(self, probe_dia, probe_focus, wavelength, *args, **kwargs):
        self.probe_dia = probe_dia
        self.probe_focus = probe_focus
        self.wavelength = wavelength
        super().__init__(*args, **kwargs)

    def selfs(self):
        # Transverse grid (x-axis in ptychography, z-axis in RAM)
        x_grid = np.arange(self.nz + 2) * self._dz
        delta_x = x_grid - self._zs
        
        # Gaussian Amplitude
        sigma = self.probe_dia / 2.0
        amplitude = np.exp(-0.5 * (delta_x / sigma)**2)
        
        # Quadratic Phase (Defocus)
        k0 = 2 * np.pi / self.wavelength
        if abs(self.probe_focus) < 1e-12:
            phase = 1.0
        else:
            R = -self.probe_focus 
            curvature = k0 / (2.0 * R)
            phase = np.exp(1j * curvature * delta_x**2)
        
        self.u[:] = amplitude * phase
        self.u[0] = 0.0 

# --- 2. Helper: Generate Blobs ---
def generate_blob_phantom(nz, nr, n_background=1.0, delta_n=0.01):
    """
    Generates a refractive index map n(x, z) with random circular blobs.
    """
    n_map = np.ones((nz, nr)) * n_background
    
    # Grid coordinates
    z_idx, x_idx = np.meshgrid(np.arange(nr), np.arange(nz))

    # Random seed for reproducibility
    np.random.seed(42)
    
    # Add random blobs
    num_blobs = 20
    for _ in range(num_blobs):
        # Random center and radius
        cx = np.random.randint(0, nz)
        cz = np.random.randint(0, nr)
        radius = np.random.randint(nz // 20, nz // 8)
        
        # Distance mask
        dist = np.sqrt((x_idx - cx)**2 + (z_idx - cz)**2)
        mask = dist <= radius
        
        # Add refractive index perturbation (smooth edges)
        # Using a simple cosine taper for smoothness
        taper = np.cos(np.pi * dist[mask] / (2 * radius))
        n_map[mask] += delta_n * taper

    return n_map

def simulate_blobs():
    # --- Parameters ---
    um = 1e-6
    wavelength = 0.635 * um
    c0 = 1500.0
    freq = c0 / wavelength
    
    sample_width = 150.0 * um
    sample_thick = 40.0 * um # Thicker sample to see scattering
    
    # Grid Resolution
    n_transverse = 1024
    n_prop_steps = 500  # Number of slices in Z direction
    
    dz_param = sample_width / n_transverse
    
    # --- 3. Create Variable Refractive Index Object ---
    print("Generating Phantom Object...")
    # Generate n(x, z)
    n_map = generate_blob_phantom(n_transverse, n_prop_steps, delta_n=0.05)
    
    # Convert to Sound Speed Map c(x, z)
    # c = c0 / n
    # Note: PyRAM expects cw to include the boundaries, so we might need padding 
    # but providing the exact grid usually works if sizes match z_ss.
    c_map = c0 / n_map
    
    # Define the coordinates for the map
    z_ss = np.linspace(0, sample_width, n_transverse)
    rp_ss = np.linspace(0, sample_thick, n_prop_steps)
    
    # Important: PyRAM expects cw to have shape (num_depths, num_ranges)
    # Our n_map is (nz, nr), which matches (num_depths, num_ranges) in PyRAM terminology.
    
    # --- 4. Setup Simulation ---
    # Transparent Boundaries
    z_sb = np.array([0.0, 10.0*um])
    rp_sb = np.array([0.0])
    cb = np.array([[c0], [c0]])
    rhob = np.array([[1.0], [1.0]])
    attn = np.array([[0.0], [0.0]])
    
    rbzb = np.array([[0.0, sample_width], [sample_thick, sample_width]])

    pyram = PtychoProbePyRAM(
        probe_dia=15.0 * um,
        probe_focus=-10.0 * um, # Diverging beam
        wavelength=wavelength,
        
        freq=freq,
        zs=sample_width/2.0,
        zr=sample_width/2.0,
        
        # --- Injecting the Variable Profile Here ---
        z_ss=z_ss, 
        rp_ss=rp_ss, 
        cw=c_map,  # <--- The "Complex Blobs" Velocity Map
        
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
        rbzb=rbzb,
        rmax=sample_thick,
        dr=sample_thick / n_prop_steps, # Step size matches object resolution
        dz=dz_param,
        c0=c0,
        np=8
    )

    print("Propagating through blobs...")
    results = pyram.run()
    
    # --- 5. Visualization ---
    tl_grid = results['TL Grid']
    ranges_um = results['Ranges'] / um
    transverse_um = results['Depths'] / um
    
    pressure_amp = 10**(-tl_grid / 20.0)
    R_grid, _ = np.meshgrid(results['Ranges'], results['Depths'])
    envelope_amp = pressure_amp * np.sqrt(R_grid + 1e-12)

    # Create figure with 2 subplots: Object and Field
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: The Refractive Index Object
    extent = [0, sample_thick/um, sample_width/um, 0] # [left, right, bottom, top]
    im1 = ax1.imshow(n_map, extent=extent, aspect='auto', cmap='bone')
    ax1.set_title("Refractive Index Phantom n(x, z)")
    ax1.set_xlabel("Propagation z (um)")
    ax1.set_ylabel("Transverse x (um)")
    plt.colorbar(im1, ax=ax1, label="Refractive Index")
    
    # Plot 2: The Wavefield
    # Note: PyRAM output grid might be slightly different size due to padding/steps
    # We use the extent from results to be accurate
    extent_wave = [ranges_um[0], ranges_um[-1], transverse_um[-1], transverse_um[0]]
    im2 = ax2.imshow(
        envelope_amp, 
        extent=extent_wave, 
        aspect='auto', 
        cmap='viridis',
        vmax=np.percentile(envelope_amp, 99.5)
    )
    ax2.set_title("Wavefield Magnitude |u|")
    ax2.set_xlabel("Propagation z (um)")
    ax2.set_ylabel("Transverse x (um)")
    plt.colorbar(im2, ax=ax2, label="Field Amplitude")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_blobs()