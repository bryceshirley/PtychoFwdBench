import numpy as np
import matplotlib.pyplot as plt
from pyram.PyRAM import PyRAM  # Ensure this matches your file structure

# --- 1. Define Custom Solver with Divergent Source ---
class DivergentBeamPyRAM(PyRAM):
    def __init__(self, beam_width, phase_curvature, *args, **kwargs):
        """
        Extended PyRAM class that supports a custom divergent Gaussian source.
        
        Args:
            beam_width (float): Gaussian sigma (width) in meters.
            phase_curvature (float): Strength of the quadratic phase. 
                                     Higher value = faster spreading.
        """
        self.beam_width = beam_width
        self.phase_curvature = phase_curvature
        super().__init__(*args, **kwargs)

    def selfs(self):
        """
        Overrides the standard 'selfs' method to inject a custom 
        Gaussian beam with Quadratic Phase.
        """
        # Create depth grid (z axis in math, y axis in array)
        # self.nz is number of depth points, self._dz is depth step
        z_grid = np.arange(self.nz + 2) * self._dz
        
        # Centered depth coordinates relative to source depth
        delta_z = z_grid - self._zs
        
        # 1. Gaussian Amplitude Profile
        # A = exp( - (z - zs)^2 / width^2 )
        amplitude = np.exp(-0.5 * (delta_z / self.beam_width)**2)
        
        # 2. Quadratic Phase (The "Spreading" Factor)
        # Phase = exp( -i * curvature * (z - zs)^2 )
        # This acts like a lens. 
        phase = np.exp(-1j * self.phase_curvature * delta_z**2)
        
        # 3. Set the initial field
        self.u[:] = amplitude * phase
        
        # Enforce boundary condition at surface (approximate)
        self.u[0] = 0.0

def simulate_divergent_gaussian():
    # --- 2. Simulation Parameters ---
    freq = 50.0        # Frequency (Hz)
    c0 = 1500.0        # Sound speed (m/s)
    max_range = 5000.0 # 5 km (Increased from 50m to allow spreading)
    max_depth = 4000.0
    source_depth = 2000.0
    
    # --- 3. Environment (Free Space / Isovelocity) ---
    z_ss = np.array([0.0, max_depth])
    cw = np.array([[c0], [c0]])  
    rp_ss = np.array([0.0])      

    z_sb = np.array([0.0, 100.0]) 
    rp_sb = np.array([0.0])
    cb = np.array([[c0], [c0]])   
    rhob = np.array([[1.0], [1.0]]) 
    
    # Zero attenuation for clear visualization
    attn = np.array([[0.0], [0.0]]) 
    
    rbzb = np.array([[0.0, max_depth], [max_range, max_depth]])

    # --- 4. Configure & Run Custom Model ---
    # Adjust 'phase_curvature' to control spread angle.
    # Try 1e-5 for weak spread, 5e-5 for strong spread.
    pyram = DivergentBeamPyRAM(
        beam_width=150.0,       # Initial width of the beam (m)
        phase_curvature=-8e-5,   # Quadratic phase factor (1/m^2)
        
        # Standard PyRAM args
        freq=freq,
        zs=source_depth,
        zr=source_depth,
        z_ss=z_ss, rp_ss=rp_ss, cw=cw,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
        rbzb=rbzb,
        rmax=max_range,
        dr=10.0,   
        dz=2.0,    
        c0=c0,
        np=8       
    )

    print("Propagating divergent beam...")
    results = pyram.run()
    
    # --- 5. Visualization ---
    tl_grid = results['TL Grid']
    ranges = results['Ranges']
    depths = results['Depths']

    # Convert dB loss back to Pressure Amplitude |p|
    pressure_amp = 10**(-tl_grid / 20.0)
    
    # Convert |p| to Envelope |u| to remove cylindrical geometric spreading 
    # (Makes the beam easier to see)
    R_grid, _ = np.meshgrid(ranges, depths)
    envelope_amp = pressure_amp * np.sqrt(R_grid)

    plt.figure(figsize=(10, 8))
    
    plt.imshow(
        envelope_amp, 
        extent=[ranges[0], ranges[-1], depths[-1], depths[0]],
        aspect='auto',
        cmap='viridis',
        origin='upper',
        # Clip the very bright source point to see the tail
        vmax=np.percentile(envelope_amp, 99.5) 
    )
    
    plt.colorbar(label='Wavefield Envelope |u|')
    plt.title('Divergent Gaussian Beam (Quadratic Initial Phase)')
    plt.xlabel('Propagation Distance z (m)')
    plt.ylabel('Transverse Position x (m)')
    
    # Zoom window to center
    plt.ylim(source_depth + 1500, source_depth - 1500)
    
    plt.show()

if __name__ == "__main__":
    simulate_divergent_gaussian()