import numpy as np
import matplotlib.pyplot as plt
from time import process_time

# --- 1. Blob Generation (Exact Copy from your PyRAM code) ---
def generate_blob_phantom(nz, nr, n_background=1.0, delta_n=0.01):
    """
    Generates a refractive index map n(x, z) with random circular blobs.
    """
    n_map = np.ones((nz, nr)) * n_background
    
    # Grid coordinates
    z_idx, x_idx = np.meshgrid(np.arange(nr), np.arange(nz))

    # Random seed for reproducibility (Same seed as your PyRAM script)
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
        taper = np.cos(np.pi * dist[mask] / (2 * radius))
        n_map[mask] += delta_n * taper

    return n_map

# --- 2. Multislice Solver Class ---
class MultisliceSolver:
    def __init__(self, nx, dx, wavelength,symmetric=True):
        """
        Wide-Angle Split-Step Fourier Solver.
        If symmetric=True,
            Uses Symmetric Strang Splitting: Diff(dz/2) -> Refr(dz) -> Diff(dz/2).
        If symmetric=False,
            Uses Non-Symmetric Splitting: Diff(dz) -> Refr(dz).
        """
        self.nx = nx
        self.dx = dx
        self.k0 = 2 * np.pi / wavelength
        self.symmetric = symmetric
        
        # --- Precompute Diffraction Operator (Exact / Wide-Angle) ---
        # Frequencies kx
        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        
        # Exact Propagation Phase: k_z = sqrt(k0^2 - kx^2)
        # We subtract k0 to remove the carrier (envelope frame)
        # Handle evanescent waves (where argument < 0)
        kz_sq = self.k0**2 - self.kx**2
        self.kz_envelope = np.sqrt(kz_sq.astype(complex)) - self.k0
        
    def propagate(self, psi_in, n_map, sample_thick):
        """
        Propagates the field through the refractive index map.
        n_map shape: (nx, nz_steps)
        """
        nz_steps = n_map.shape[1]
        dz = sample_thick / nz_steps
        
        psi = psi_in.astype(complex)
        
        # Storage for visualization (x, z)
        field_storage = np.zeros((self.nx, nz_steps), dtype=complex)
        
        # Precompute the half-step diffraction operator for this specific dz
        if self.symmetric:
            diffract_half = np.exp(1j * (dz / 2.0) * self.kz_envelope)
        else:
            diffract_full = np.exp(1j * dz * self.kz_envelope)
        
        t0 = process_time()
        
        for i in range(nz_steps):
            # Store current field
            field_storage[:, i] = psi
            
            # 1. First Half-Step Diffraction (Spectral)
            if self.symmetric:
                psi = np.fft.ifft(np.fft.fft(psi) * diffract_half)
            else:
                psi = np.fft.ifft(np.fft.fft(psi) * diffract_full)
            
            # 2. Full-Step Refraction (Spatial)
            # n_map is (transverse, propagation)
            n_slice = n_map[:, i]
            
            # Phase shift R = exp(i * k0 * (n(x) - 1) * dz)
            refraction_phase = np.exp(1j * self.k0 * (n_slice - 1.0) * dz)
            psi *= refraction_phase
            
            # 3. Second Half-Step Diffraction (Spectral)
            if self.symmetric:
                psi = np.fft.ifft(np.fft.fft(psi) * diffract_half)
            
        t_elapsed = process_time() - t0
        return field_storage, t_elapsed

# --- 3. Simulation Setup (Matches PyRAM Config) ---
def simulate_blobs_multislice():
    # --- Parameters ---
    um = 1e-6
    wavelength = 0.635 * um
    
    sample_width = 150.0 * um
    sample_thick = 40.0 * um 
    
    # Grid Resolution
    n_transverse = 1024
    n_prop_steps = 500
    
    dx = sample_width / n_transverse
    
    # --- Create Variable Refractive Index Object ---
    print("Generating Phantom Object...")
    # Matches the PyRAM 'cw' input generation
    n_map = generate_blob_phantom(n_transverse, n_prop_steps, delta_n=0.05)
    
    # --- Initialize Probe (Gaussian with Quadratic Phase) ---
    probe_dia = 15.0 * um
    probe_focus = -10.0 * um
    zs = sample_width / 2.0  # Center of grid
    
    # Grid for probe generation
    x_grid = np.arange(n_transverse) * dx
    delta_x = x_grid - zs
    
    # Gaussian Amplitude
    sigma = probe_dia / 2.0
    amplitude = np.exp(-0.5 * (delta_x / sigma)**2)
    
    # Quadratic Phase (Defocus)
    k0 = 2 * np.pi / wavelength
    R = -probe_focus 
    curvature = k0 / (2.0 * R)
    phase = np.exp(1j * curvature * delta_x**2)
    
    psi_0 = amplitude * phase

    # --- Run Multislice ---
    print("Running Multislice Propagation...")
    solver = MultisliceSolver(n_transverse, dx, wavelength, symmetric=True)
    field_history, run_time = solver.propagate(psi_0, n_map, sample_thick)
    
    print(f"Propagation Complete. Time: {run_time:.4f} s")
    
    # --- Visualization (Matches PyRAM Plotting) ---
    # Multislice results are already in |u| (if we ignore 1/sqrt(r) spreading for 1D)
    # But for fair comparison with PyRAM's geometric spreading, 
    # we can treat this as a 1D slice of a 2D cylindrical wave if desired, 
    # or just plot |u| directly as multislice is typically Planar (2D -> 1D slice).
    # Since PyRAM output was converted to |u|, we compare |psi| directly.
    
    envelope_amp = np.abs(field_history)
    
    # Axis arrays for plotting
    z_axis = np.linspace(0, sample_thick/um, n_prop_steps)
    x_axis = np.linspace(0, sample_width/um, n_transverse)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Object
    extent = [z_axis[0], z_axis[-1], x_axis[-1], x_axis[0]]
    im1 = ax1.imshow(n_map, extent=extent, aspect='auto', cmap='bone')
    ax1.set_title("Refractive Index Phantom n(x, z)")
    ax1.set_xlabel("Propagation z (um)")
    ax1.set_ylabel("Transverse x (um)")
    plt.colorbar(im1, ax=ax1, label="Refractive Index")
    
    # Plot 2: Wavefield
    # Note: vmax matches the PyRAM percentile scaling for consistency
    im2 = ax2.imshow(
        envelope_amp, 
        extent=extent, 
        aspect='auto', 
        cmap='viridis',
        vmax=np.percentile(envelope_amp, 99.5)
    )
    ax2.set_title(f"Multislice Amplitude |u| ({n_prop_steps} steps)")
    ax2.set_xlabel("Propagation z (um)")
    ax2.set_ylabel("Transverse x (um)")
    plt.colorbar(im2, ax=ax2, label="Field Amplitude")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_blobs_multislice()