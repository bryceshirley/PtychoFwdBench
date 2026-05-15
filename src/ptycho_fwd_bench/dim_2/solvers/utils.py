from scipy.fftpack import dct, dst, idct, idst

# Use cupy if available, otherwise fallback to numpy
try:
    import cupy as xp
except ImportError:
    import numpy as xp


def get_prop_phase_shift(
    k0: float,
    nx: int,
    dx: float,
    dz: float,
    n_mean: float = 1.0,
    transform_type: str = "FFT",
) -> xp.ndarray:
    """
    Computes the spectral propagation kernel for a given slice thickness dz.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber.
    nx: int
        Number of transverse points.
    dx : float
        Transverse pixel width.
    dz : float
        Thickness of the slice to propagate through.
    n_mean : float, optional
        Mean refractive index of the medium (default is 1.0).

    Returns
    -------
    xp.ndarray
        The spectral propagation kernel H(kx) = exp(i * sqrt(k0^2 - kx^2) * dz).
    """
    kx = get_spectral_coords(nx, dx, transform_type)

    # Transverse Propagator P
    inside = k0**2 - kx**2
    sqrt_term = xp.sqrt(xp.clip(inside, 0.0, None))
    lambda_vac = 1j * (sqrt_term - k0)
    phi_mean = 1j * k0 * (n_mean - 1.0)

    return (lambda_vac + phi_mean) * dz


def get_prop_kernel_perp(
    k0: float, kx: xp.ndarray, dz: float, n_mean: float = 1.0
) -> xp.ndarray:
    """
    Computes the spectral propagation kernel for a given slice thickness dz.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber.
    kx : xp.ndarray
        Spectral coordinates in the transverse direction.
    dz : float
        Thickness of the slice to propagate through.
    n_mean : float, optional
        Mean refractive index of the medium (default is 1.0).

    Returns
    -------
    xp.ndarray
        The spectral propagation kernel H(kx) = exp(i * sqrt(k0^2 - kx^2) * dz).
    """
    # Transverse Propagator P
    inside = k0**2 - kx**2
    sqrt_term = xp.sqrt(xp.clip(inside, 0.0, None))
    lambda_vac = 1j * (sqrt_term - k0)
    phi_mean = 1j * k0 * (n_mean - 1.0)

    return xp.exp((lambda_vac + phi_mean) * dz).astype(xp.complex128)


def get_spectral_coords(
    nx: int, dx: float, transform_type: str, mode: str = "spectral"
) -> xp.ndarray:
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
    k_eff : xp.ndarray
        Effective spectral coordinates such that Lambda = -(k_eff)^2.
    """

    # 1. Determine the Base Wavenumbers (k)
    # -------------------------------------
    # CRITICAL: For DST-I/DCT-I, the domain includes boundaries, so width is (N+1)dx
    if transform_type == "FFT":
        # FFT is periodic, width is N*dx
        k = 2 * xp.pi * xp.fft.fftfreq(nx, d=dx)

    elif transform_type == "DST":
        # DST-I: Modes k_m = pi * m / ((N+1)*dx)
        total_width = (nx + 1) * dx
        modes = xp.arange(1, nx + 1)
        k = xp.pi * modes / total_width

    elif transform_type == "DCT":
        # DCT-I: Modes k_m = pi * m / ((N+1)*dx)
        total_width = (nx + 1) * dx
        modes = xp.arange(nx)
        k = xp.pi * modes / total_width

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
        lambda_fd = (2.0 / dx**2) * (xp.cos(k * dx) - 1.0)
        return xp.sqrt(xp.abs(lambda_fd))

    elif mode == "fd4":
        # 4th Order Central Difference
        # Lambda = (1 / 12*dx^2) * (-cos(2kx) + 16cos(kx) - 15) * 2
        term1 = 16.0 * xp.cos(k * dx)
        term2 = xp.cos(2.0 * k * dx)
        # Factor of 2 comes from the symmetric cosine sum in the derivative stencil
        lambda_fd = (2.0 / (12.0 * dx**2)) * (term1 - term2 - 15.0)
        return xp.sqrt(xp.abs(lambda_fd))

    elif mode == "pseudo":
        # Pseudo-spectral Sinc approximation
        # Effectively identical to fd2 mathematically, but calculated explicitly
        # k_eff = | sin(k*dx/2) / (dx/2) |
        # This bounds the maximum k to 2/dx (Nyquist slope)
        k_eff = xp.abs(xp.sin(k * dx / 2.0)) / (dx / 2.0)
        return k_eff

    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_spectral_kernel(
    psi: xp.ndarray,
    kernel: xp.ndarray,
    transform_type: str,
) -> xp.ndarray:
    """
    Applies a spectral kernel K to a field Psi:  FT^-1 [ K * FT [ Psi ] ]
    Handles the complexity of treating Real/Imaginary parts separately for DST/DCT.

    Parameters
    ----------
    psi : xp.ndarray
        Ixput field in spatial domain.
    kernel : xp.ndarray
        Spectral kernel to apply.
    transform_type : str
        Type of spectral transform ("FFT", "DST", "DCT").
    """
    if transform_type == "FFT":
        return xp.fft.ifft(xp.fft.fft(psi) * kernel)

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
    psi_real_k = _transform(xp.real(psi), fwd, t_type)
    psi_imag_k = _transform(xp.imag(psi), fwd, t_type)

    # 2. Apply Kernel (Complex multiplication in spectral domain)
    # (Re + jIm) * K
    psi_spectral = (psi_real_k + 1j * psi_imag_k) * kernel

    # 3. Inverse Transform
    out_real = _transform(xp.real(psi_spectral), inv, t_type)
    out_imag = _transform(xp.imag(psi_spectral), inv, t_type)

    return out_real + 1j * out_imag
