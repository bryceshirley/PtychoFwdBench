import numpy as np
from scipy.fftpack import dst, idst, dct, idct


def get_spectral_coords(nx: int, dx: float, transform_type: str) -> np.ndarray:
    """Generates the correct k-vector coordinates for a given transform.

    Parameters
    ----------
    nx : int
        Number of spatial grid points.
    dx : float
        Spatial sampling interval (um).
    transform_type : str
        Type of spectral transform ("FFT", "DST", "DCT").
    Returns
    -------
    kx : np.ndarray
        Spectral coordinates array.
    """
    total_width = nx * dx

    if transform_type == "FFT":
        fx = np.fft.fftfreq(nx, d=dx)
        return 2 * np.pi * fx

    elif transform_type == "DST":
        # Standard discrete sine transform modes
        modes = np.arange(1, nx + 1)
        return np.pi * modes / total_width

    elif transform_type == "DCT":
        modes = np.arange(nx)
        return (np.pi * modes) / total_width

    raise ValueError(f"Unknown transform type: {transform_type}")


def apply_spectral_kernel(
    psi: np.ndarray,
    kernel: np.ndarray,
    transform_type: str,
) -> np.ndarray:
    """
    Applies a spectral kernel K to a field Psi:  FT^-1 [ K * FT [ Psi ] ]
    Handles the complexity of treating Real/Imaginary parts separately for DST/DCT.

    Parameters
    ----------
    psi : np.ndarray
        Input field in spatial domain.
    kernel : np.ndarray
        Spectral kernel to apply.
    transform_type : str
        Type of spectral transform ("FFT", "DST", "DCT").
    """
    if transform_type == "FFT":
        return np.fft.ifft(np.fft.fft(psi) * kernel)

    # Helper for Scipy transforms which work on real arrays
    def _transform(data, func, type_arg):
        return func(data, type=type_arg, norm="ortho")

    # Select transform functions
    if transform_type == "DST":
        fwd, inv = dst, idst
        t_type = 1
    elif transform_type == "DCT":
        fwd, inv = dct, idct
        t_type = 2
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    # 1. Forward Transform (Real and Imag separately)
    psi_real_k = _transform(np.real(psi), fwd, t_type)
    psi_imag_k = _transform(np.imag(psi), fwd, t_type)

    # 2. Apply Kernel (Complex multiplication in spectral domain)
    # (Re + jIm) * K
    psi_spectral = (psi_real_k + 1j * psi_imag_k) * kernel

    # 3. Inverse Transform
    out_real = _transform(np.real(psi_spectral), inv, t_type)
    out_imag = _transform(np.imag(psi_spectral), inv, t_type)

    return out_real + 1j * out_imag
