import numpy as np
from scipy.fftpack import dct, dst, idct, idst


def get_spectral_coords(
    nx: int, dx: float, transform_type: str, mode: str = "spectral"
) -> np.ndarray:
    """
    Generates the correct k-vector coordinates (or effective FD eigenvalues).

    Parameters
    ----------
    nx : int
        Number of spatial grid points.
    dx : float
        Spatial sampling interval (um).
    transform_type : str
        Type of spectral transform ("FFT", "DST", "DCT").
    mode : str, optional
        Determines the eigenvalues used for the Laplacian operator.
        'spectral': Standard unbounded k (k^2). Accurate but unstable for Pade.
        'fd2'     : 2nd Order FD eigenvalues. Robust, acts as low-pass filter.
        'fd4'     : 4th Order FD eigenvalues. Higher accuracy, still bounded.
        'pseudo'  : Explicit pseudo-spectral Sinc approximation (Same as fd2).

    Returns
    -------
    k_eff : np.ndarray
        Effective spectral coordinates such that Lambda = -(k_eff)^2.
    """

    # 1. Determine the Base Wavenumbers (k)
    # -------------------------------------
    # CRITICAL: For DST-I/DCT-I, the domain includes boundaries, so width is (N+1)dx
    if transform_type == "FFT":
        # FFT is periodic, width is N*dx
        k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)

    elif transform_type == "DST":
        # DST-I: Modes k_m = pi * m / ((N+1)*dx)
        total_width = (nx + 1) * dx
        modes = np.arange(1, nx + 1)
        k = np.pi * modes / total_width

    elif transform_type == "DCT":
        # DCT-I: Modes k_m = pi * m / ((N+1)*dx)
        total_width = (nx + 1) * dx
        modes = np.arange(nx)
        k = np.pi * modes / total_width

    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    # 2. Apply Mode (Eigenvalue Modification)
    # ---------------------------------------
    if mode == "spectral":
        # Standard: Lambda = -k^2
        return k

    elif mode == "fd2":
        # 2nd Order Central Difference
        # Lambda = (2 / dx^2) * (cos(k*dx) - 1)
        # We return sqrt(|Lambda|)
        lambda_fd = (2.0 / dx**2) * (np.cos(k * dx) - 1.0)
        return np.sqrt(np.abs(lambda_fd))

    elif mode == "fd4":
        # 4th Order Central Difference
        # Lambda = (1 / 12*dx^2) * (-cos(2kx) + 16cos(kx) - 15) * 2
        term1 = 16.0 * np.cos(k * dx)
        term2 = np.cos(2.0 * k * dx)
        # Factor of 2 comes from the symmetric cosine sum in the derivative stencil
        lambda_fd = (2.0 / (12.0 * dx**2)) * (term1 - term2 - 15.0)
        return np.sqrt(np.abs(lambda_fd))

    elif mode == "pseudo":
        # Pseudo-spectral Sinc approximation
        # Effectively identical to fd2 mathematically, but calculated explicitly
        # k_eff = | sin(k*dx/2) / (dx/2) |
        # This bounds the maximum k to 2/dx (Nyquist slope)
        k_eff = np.abs(np.sin(k * dx / 2.0)) / (dx / 2.0)
        return k_eff

    else:
        raise ValueError(f"Unknown mode: {mode}")


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
