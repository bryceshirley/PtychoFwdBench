import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import _ArrayLike
from scipy.fftpack import dst, idst
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import trange

from pkg.utils.pade import pade_coefficients


def greenes_starter(z: _ArrayLike, zs: float, krs: complex):
    """Generate Greenes source

    Args:
        z (_ArrayLike): mesh
        zs (float): source depth
        krs (complex): wave number

    Returns:
        np.ndarray: greens source on mesh
    """
    IC = (
        (1.4467 - 0.8402 * krs**2 * (z - zs) ** 2)
        * np.exp(-(krs**2) * (z - zs) ** 2 / 1.5256)
        / (2 * np.sqrt(np.pi))
    )
    return IC


def l2norm(u: _ArrayLike, dz: float):
    """Compute approximation of L2 norm using trapezoidal rule

    Args:
        u (_ArrayLike): discretized function to integrate
        dz (float): discretization constant

    Returns:
        float: L2norm
    """
    np.size(u)
    q = u * np.conj(u)
    _norm = 1 / 2 * (q[0] + q[1]) + np.sum(q[1:-1])
    _norm *= dz
    return np.sqrt(_norm)


def normalize(u: _ArrayLike, dz: float, norm: float = 1.0):
    """set u to desired L2 norm

    Args:
        u (_ArrayLike): function
        dz (float): discretization constant
        norm (float, optional): desired norm. Defaults to 1.0.

    Returns:
        np.ndarray: output Array
    """
    _norm = l2norm(u, dz)
    return u * (norm / _norm)


def logField(field: _ArrayLike):
    return 20 * np.log10(np.abs(field) * np.pi * 4)


def TL(field: _ArrayLike):
    p0_index = np.argmax(np.abs(field))
    p0 = np.abs(field.ravel()[p0_index])
    TL = 20 * np.log10(np.abs(field) / p0)
    return TL


def plot_TL(
    ax: plt.Axes,
    total_range: float,
    total_depth: float,
    TL: _ArrayLike,
    title: str | None = None,
    **kwargs,
):
    """Plot transmission loss / acoustic field as image

    Args:
        ax (plt.Axes): axis object in order to be used in a subplots figure
        total_range (float): total range of the field
        total_depth (float): total depth of the field
        TL (_ArrayLike): transmission loss
        title (str | None, optional): plot title. Defaults to None.

    Returns:
        _type_: axis object
    """
    logging.info("Creating TL plot")
    default_vmin = kwargs.get("default_vmin", False)
    if not default_vmin:
        vmax = kwargs.get("vmax", -25)
        vmin = kwargs.get("vmin", -60)
        img = ax.imshow(TL.T, vmin=vmin, vmax=vmax, cmap="jet", aspect="auto")
    else:
        img = ax.imshow(TL.T, cmap="jet", aspect="auto")

    r_labels = np.linspace(0, total_range, 11, dtype=np.int32)
    z_labels = np.linspace(0, total_depth, 5, dtype=np.int32)

    ax.set_xticks(
        np.linspace(0, TL.shape[0], r_labels.shape[0]),
        labels=[f"{l_idx / 1000:.1f}" for l_idx in r_labels],
    )
    ax.set_yticks(
        np.linspace(0, TL.shape[1], z_labels.shape[0]),
        labels=[f"{l_idx / 1000:.1f}" for l_idx in z_labels],
    )
    ax.set_xlabel("r [km]")
    ax.set_ylabel("z [km]")
    if title is not None:
        ax.set_title(title)

    return img


def invert_SSP_operator(u: _ArrayLike, H: float, b: float):
    """Invert the partial split step pade operator (1 + b*dy^2) w = u

    Args:
        u (_ArrayLike): Previous state, right hand side of the operator inversion
        H (float): Total depth in meters
        b (float): Pade coefficient
    Returns:
        np.ndarray: The solution vector
    """
    N = np.size(u)

    k = (np.arange(N) + 1) * np.pi / H
    fact = 1.0 - b * (k**2)

    u_hat = dst(u, norm="ortho", type=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        du_hat = u_hat / fact
    du_hat[np.isclose(fact, 0)] = 0
    Du = np.zeros_like(u, dtype=np.complex128)
    Du = idst(du_hat, norm="ortho", type=1)
    return Du


def constant_matrix(u: _ArrayLike, H: float, b: float):
    N = np.size(u) - 2
    dy = H / (N + 1)

    A_sub = np.ones(N - 1) * b / (dy**2)
    A_sup = np.ones(N - 1) * b / (dy**2)
    A_main = np.ones(N) * (1 - b / dy**2 * (2))

    A = diags([A_sup, A_main, A_sub], offsets=(-1, 0, 1), format="csr")
    y = np.zeros_like(u, dtype=np.complex128)
    y[1:-1] = spsolve(A, u[1:-1])

    return y


def discrete_matrix(u: _ArrayLike, delta_ksq: _ArrayLike, H: float, b: float):
    N = np.size(u)
    dy = H / (N - 1)

    A_sub = np.ones(N - 1) * b / (dy**2)
    A_sup = np.ones(N - 1) * b / (dy**2)
    A_main = np.ones(N) * (1 + b * (delta_ksq[1:-1] - 2 / dy**2))

    A = diags([A_sub, A_main, A_sup], (-1, 0, 1), format="csr")
    y = spsolve(A, u)

    return y


def correction_n(
    u: _ArrayLike, b_tilde: float, H: float, delta_ksq: _ArrayLike, n: int = 1
):
    """Solve w = delta_ksq^n / (1 + b_tilde*d_y^2)^(n+1) u

    Args:
        w (_ArrayLike): w uncorrected
        b_tilde (float): modified pade coefficient b_tilde = b/k0sq
        delta_ksq (_ArrayLike): (k0sq - ksq)
        n (int): maximum number of terms to expand the solution operator as

    Returns:
        _type_: v
    """
    w = u
    w = invert_SSP_operator(w, H, b_tilde)
    for _ in range(n + 1):
        w *= delta_ksq * b_tilde
        w = invert_SSP_operator(w, H, b_tilde)
    return w


def step_matrix(
    u0: _ArrayLike,
    ksq: _ArrayLike,
    dr: float,
    H: float,
    k0sq: float,
    k0: float,
    corrector: bool = False,
    **kwargs,
):
    """Step using FFT and a corrector for WAPE-FFT

    Args:
        u0 (_ArrayLike): previous step field
        ksq (_ArrayLike): wavenumber squared, discretized in y
        dr (float): step size
        dy (float): discretization constant in y
        k0sq (float): reference wave number squared
        k0 (float): reference wave number
        nP (int, optional): order of Pade expansion. Defaults to 4.

    Returns:
        np.ndarray: next step field
    """
    nP = kwargs.get("nP", 4)
    n = np.size(u0)
    # hard coded pade coefficients for dr = 10, k0= 100*2*pi/1500
    b, d = pade_coefficients(dr * k0, nP)
    b = b / k0sq
    p = np.size(d)
    w = np.zeros((p, n), dtype=np.complex128)
    w[0] = u0
    delta_ksq = ksq - k0sq
    if not corrector:
        for i in range(1, p):
            w[i, 1:-1] = constant_matrix(u0[1:-1], H, b[i - 1])
    else:
        for i in range(1, p):
            w[i, 1:-1] = discrete_matrix(u0[1:-1], delta_ksq, H, b[i - 1])
    u0 = np.exp(1j * dr * k0) * np.dot(d, w)
    return u0


def step_fft(
    u0: _ArrayLike,
    ksq: _ArrayLike,
    dr: float,
    H: float,
    k0sq: float,
    k0: float,
    **kwargs,
):
    """Step using FFT and a corrector for WAPE-FFT

    Args:
        u0 (_ArrayLike): previous step field
        ksq (_ArrayLike): wavenumber squared, discretized in y
        dr (float): step size
        dy (float): discretization constant in y
        k0sq (float): reference wave number squared
        k0 (float): reference wave number
        corrector (bool, optional): whether to correct for nonconstant c

    Returns:
        np.ndarray: next step field
    """
    corrector = kwargs.get("corrector", True)
    corrector_terms = kwargs.get("correction_order", 3)
    nP = kwargs.get("nP", 4)
    # print(f"using {corrector_terms} correction terms")
    n = np.size(u0)
    # hard coded pade coefficients for dr = 10, k0= 100*2*pi/1500
    hk0 = dr * k0
    b, d = pade_coefficients(hk0, nP)

    b = b / k0sq
    p = np.size(d)
    w = np.zeros((p, n), dtype=np.complex128)
    w[0] = u0
    delta_ksq = ksq - k0sq
    H / (n)
    if not corrector:
        for i in range(1, p):
            w[i] = invert_SSP_operator(u0, H, b[i - 1])
        u0 = np.exp(1j * dr * k0) * np.dot(d, w)
    else:
        # _norm = l2norm(u0, dy)
        for i in range(1, p):
            w[i] = invert_SSP_operator(u0, H, b[i - 1])
            for j in range(corrector_terms):
                w[i] += ((-1) ** (j + 1)) * correction_n(u0, b[i - 1], H, delta_ksq, j)

        u0 = np.exp(1j * dr * k0) * np.dot(d, w)
        # u0 = normalize(u0, dy, _norm)
    return u0


def normal_modes(f: float, zs: float, dr: float, R: float, H: float):
    phij = np.loadtxt(f"modes/phij_f_{int(f)}Hz.txt")[:, 1:]
    kj = np.loadtxt(f"modes/kj_f_{int(f)}Hz.txt")

    n_modes = np.size(kj)
    n = np.shape(phij)[0]
    z = np.linspace(0, H, n)
    izs = np.nonzero(z > zs)[0][0]
    n_steps = int(R / dr) + 1
    x = np.linspace(0, R, n_steps)

    PP = np.zeros((n, n_steps), dtype=np.complex128)
    for i in range(n_modes):
        PP += np.outer(phij[izs, i] * phij[:, i], np.exp(1j * kj[i] * x)) / np.sqrt(
            kj[i]
        )
    return PP


def eddy(z: _ArrayLike, x: _ArrayLike, y: float, **kwargs):
    """Computes the propagation of the acoustic field through a synoptic eddy

    Args:
        z (_ArrayLike): Array of discretization points in depth
        x (_ArrayLike): Array of discretization points in range
        y (float): position w.r.t. eddy

    Returns:
        _type_: _description_
    """

    save = kwargs.get("save", False)
    n_steps = x.size - 1
    dr = x[1] - x[0]
    dz = z[1] - z[0]
    logging.info("Computing propagation through synoptic eddy")
    logging.info(f"dr={dr}\ndz={dz}")
    rr, zz = np.meshgrid(x, z)

    # vertical source position
    zs = 1100
    # vertical source index
    np.nonzero(z > zs)[0][0]
    # profile parameters from "The Parameterization of the Sound Speed
    # Profile in the Sea of Japan and Its Perturbation Caused by a
    # Synoptic Eddy" by Mikhail Sorokin, Pavel Petrov, et. al.
    beta = 1.7125
    rx = 32000  # m
    ry = 18000  # m
    rz = 250  # m
    x0 = 50000
    y0 = 0
    z0 = 1100
    cm = 40  # m/s

    # Munk Profile paramters
    z1 = 1300
    B = z1
    epsilon = 0.00737
    f = 100  # Hz
    omega = 2 * np.pi * f

    # edddy location indices
    ix0 = np.nonzero(x > x0)[0][0]
    iz0 = np.nonzero(z > z0)[0][0]
    izb = np.nonzero(z > (z0 - rz))[0][0]
    ize = np.nonzero(z > (z0 + rz))[0][0]
    ixb = np.nonzero(x > (x0 - rx))[0][0]
    ixe = np.nonzero(x > (x0 + rx))[0][0]

    logging.info(f"{ix0=}, {iz0=}")

    eta = 2 * (z - z1) / B
    c0 = 1500 * (1 + epsilon * (eta + np.exp(-eta) - 1))

    delta_c = (
        -cm
        * np.exp(-((rr - x0) ** 2) / rx**2)
        * np.exp(-((y - y0) ** 2) / ry**2)
        * (zz - z0)
        / rz
        * np.exp(-beta * (zz - z0) ** 2 / rz)
    )
    ksq = omega**2 / (c0[:, None] + delta_c) ** 2
    k0sq = np.mean(ksq)
    k0 = np.sqrt(k0sq)
    # plt.imshow(delta_c)

    field = np.zeros_like(delta_c, dtype=np.complex128)
    field_ref = np.zeros_like(field, dtype=np.complex128)
    IC = np.loadtxt(f"modes/IC_{int(f)}Hz.txt")
    field[:, 0] = np.interp(z, np.linspace(0, z[-1], IC.size), IC)
    field_ref[:, 0] = np.interp(z, np.linspace(0, z[-1], IC.size), IC)

    logging.info("Propagating Munk profile eddy with DST-SSP")
    for i in trange(n_steps):
        field[:-1, i + 1] = step_fft(
            field[:-1, i],
            ksq[:-1, i],
            dr,
            z[-1],
            k0sq,
            k0,
            corrector=True,
            nP=6,
            correction_order=6,
        )

    logging.info("Propagating Munk profile eddy with FDM-SSP")
    for i in trange(n_steps):
        field_ref[:, i + 1] = step_matrix(
            field_ref[:, i], ksq[:, i], dr, z[-1], k0sq, k0, corrector=True, nP=6
        )

    ksq_munk = omega**2 / c0**2
    k0sq_munk = np.mean(ksq_munk)
    k0_munk = np.sqrt(k0sq_munk)
    logging.info("Propagating Munk profile reference solution")
    field_no_eddy = np.zeros_like(field)
    field_no_eddy_dst = np.zeros_like(field)
    field_no_eddy[:, 0] = np.interp(z, np.linspace(0, z[-1], IC.size), IC)
    field_no_eddy_dst[:, 0] = np.interp(z, np.linspace(0, z[-1], IC.size), IC)
    # field_no_eddy = propagate(
    #     field[:, 0], z[-1], k0sq_munk, ksq_munk, dr, dz, x[-1], matrix=True
    # ).T
    for i in trange(n_steps):
        field_no_eddy[:, i + 1] = step_matrix(
            field_no_eddy[:, i],
            ksq_munk,
            dr,
            z[-1],
            k0sq_munk,
            k0_munk,
            corrector=True,
            nP=6,
        )

    # produce slices at slice depth
    izsl = np.nonzero(z > 900)[0][0]
    s = field[izsl, :]
    s_ref = field_ref[izsl, :]
    s_no_eddy = field_no_eddy[izsl, :]

    fig, ax = plt.subplots(2, 2)
    img00 = plot_TL(
        ax[0, 0],
        x[-1],
        z[-1],
        logField(field_no_eddy.T),
        "Reference propagation Munk profile",
        vmin=-110,
        vmax=-60,
    )
    fig.colorbar(img00, ax=ax[0, 0])

    img01 = plot_TL(
        ax[0, 1],
        x[-1],
        z[-1],
        logField(field_ref.T),
        "Eddy propagated by FDM-SSP",
        vmin=-110,
        vmax=-60,
    )
    fig.colorbar(img01, ax=ax[0, 1])
    ax[0, 1].plot(ix0, iz0, marker="x", color="black", markersize=10)
    ax[0, 1].axhline(y=izb)
    ax[0, 1].axhline(y=ize)
    ax[0, 1].axvline(x=ixb)
    ax[0, 1].axvline(x=ixe)

    img10 = plot_TL(
        ax[1, 0],
        x[-1],
        z[-1],
        logField(field.T),
        "Eddy propagated by DST-SSP",
        vmin=-110,
        vmax=-60,
    )
    fig.colorbar(img10, ax=ax[1, 0])
    ax[1, 0].plot(ix0, iz0, marker="x", color="black", markersize=10)
    ax[1, 0].axhline(y=izb)
    ax[1, 0].axhline(y=ize)
    ax[1, 0].axvline(x=ixb)
    ax[1, 0].axvline(x=ixe)

    # ax[1,1].imshow((ksq_munk[:,None] - ksq), aspect="auto")
    il = np.nonzero(x > 30000)[0][0]
    ax[1, 1].plot(
        x[:il] / 1000, 20 * np.log10(np.abs(s_no_eddy[:il])), "-", label="Munk FDM-SSP"
    )
    ax[1, 1].plot(
        x[:il] / 1000,
        20 * np.log10(np.abs(s_ref[:il])),
        "--",
        label="Munk with eddy FDM-SSP",
    )
    ax[1, 1].plot(
        x[:il] / 1000,
        20 * np.log10(np.abs(s[:il])),
        "-.",
        label="Munk with eddy DST-SSP",
    )
    ax[1, 1].legend()
    if save:
        import matplot2tikz

        matplot2tikz.save("graphics/eddy.tex")

    plt.show()

    #    eddy_influence = logField(field_no_eddy - field_ref)
    #    dst_influence = logField(field_ref - field)
    #    logging.info(f"{np.min(eddy_influence)=}, {np.max(eddy_influence)=}")
    #    logging.info(f"{np.min(dst_influence)=}, {np.max(dst_influence)=}")
    #    fig, ax = plt.subplots(2, 1)
    #    ax[0] = plot_TL(ax[0], x[-1], z[-1], eddy_influence.T, vmin=-220, vmax=20)
    #    ax[0].plot(ix0, iz0, marker="x", color="black", markersize=10)
    #    ax[1] = plot_TL(ax[1], x[-1], z[-1], dst_influence.T, vmin=-220, vmax=20)
    #    ax[1].plot(ix0, iz0, marker="x", color="black", markersize=10)
    #
    #    if save:
    #        import matplot2tikz
    #        #matplot2tikz.save("graphics/eddy_comparison.tex")
    #
    #    plt.show()
    return field


def propagate(
    u0: _ArrayLike,
    H: float,
    k0sq: float,
    ksq: _ArrayLike,
    dr: float,
    dz: float,
    R: float,
    **kwargs,
):
    """Propagation using either matrix inversion method (classical) or DST operator inversion method.
    Both assume homogeneous Dirichlet BCs.

    Args:
        u0 (_ArrayLike): previous step value
        H (float): total depth
        k0sq (float): reference wave number squared
        ksq (_ArrayLike): wave number squared (k(z_n) = omega / c(z_n)), discretized in z direction
        dr (float): discretization constant in range r
        dz (float): discretization constant in depth z
        R (float): total range
        matrix (bool, optional): whether to use the matrix inversion instead of the DST algo. Defaults to False.
        correction_order (int, optional): The maximum order of corrector terms to include. Defaults to 3.

    Returns:
        np.ndarray: The propagated field.
    """
    matrix = kwargs.get("matrix", False)
    correction_order = kwargs.get("correction_order", 3)
    var_normalize = kwargs.get("normalize", False)
    nP = kwargs.get("nP", 4)
    n_steps = int(R / dr) + 1
    n = np.size(u0)
    k0 = np.sqrt(k0sq)
    u = np.zeros((n_steps, n), dtype=np.complex128)
    u[0] = u0
    if matrix:
        for i in range(1, n_steps):
            u[i, 1:-1] = step_matrix(
                u[i - 1, 1:-1], ksq[1:-1], dr, H - 2 * dz, k0sq, k0, corrector=True
            )
    if not matrix and not var_normalize:
        for i in range(1, n_steps):
            u[i, 1:-1] = step_fft(
                u[i - 1, 1:-1],
                ksq[1:-1],
                dr,
                H - 2 * dz,
                k0sq,
                k0,
                corrector=True,
                correction_order=correction_order,
                nP=nP,
            )
    if not matrix and var_normalize:
        for i in range(1, n_steps):
            norm = l2norm(u[i - 1], dz)
            u[i, 1:-1] = step_fft(
                u[i - 1, 1:-1],
                ksq[1:-1],
                dr,
                H - 2 * dz,
                k0sq,
                k0,
                corrector=True,
                correction_order=correction_order,
                nP=nP,
            )
            u[i] = normalize(u[i], dz, norm)
    return u


def normal_modes_starter_no_comparison(**kwargs):
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1100  # source depth

    R = 10000
    dr = 10

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    iz0 = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    # cz = np.ones_like(z) * c0
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    kz[iz0]

    k0sq = k0**2

    ksq = kz**2

    # IC = (
    #    (1.4467 - 0.8402 * krs**2 * (z - zs) ** 2)
    #    * np.exp(-(krs**2) * (z - zs) ** 2 / 1.5256)
    #    / (2 * np.sqrt(np.pi))
    # )
    IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")

    print("propagating using DST-SSP")
    u = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=False, correction_order=2)

    fig, ax = plt.subplots(1, 1)
    # ax[0].imshow(TL(u_m.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[0].set_title("a)")
    ax = plot_TL(ax, R, H, TL(u), "TL\nsolution by DST-SSP", vmax=-10, vmin=-50)
    # ax[1].imshow(TL(u.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[1].set_title("b)")
    fig.tight_layout()
    plt.show()


def normal_modes_starter(**kwargs):
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1100  # source depth

    R = 500_000
    dr = 100

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    izs = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    # cz = np.ones_like(z) * c0
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    kz[izs]

    k0sq = k0**2

    ksq = kz**2

    # IC = (
    #    (1.4467 - 0.8402 * krs**2 * (z - zs) ** 2)
    #    * np.exp(-(krs**2) * (z - zs) ** 2 / 1.5256)
    #    / (2 * np.sqrt(np.pi))
    # )
    IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")

    print("propagating using DST-SSP")
    # u1 = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=False, correction_order=2, normalize=True)
    # u2 = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=False, correction_order=3, normalize=True)
    # u3 = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=False, correction_order=4, normalize=True)
    # u4 = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=False, correction_order=5, normalize=True)
    # u5 = propagate( IC, H, k0sq, ksq, dr, dz, R, matrix=False, correction_order=6, normalize=True)
    np.load("data/q2_propagation.npy")
    u2 = np.load("data/q3_propagation.npy")
    u3 = np.load("data/q4_propagation.npy")
    u4 = np.load("data/q5_propagation.npy")
    u5 = np.load("data/q6_propagation.npy")
    print("propagating using classical SSP")
    u_m = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=True)
    print("Plotting")

    fig, ax = plt.subplots(2, 2)
    # ax[0].imshow(TL(u_m.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[0].set_title("a)")
    ax[0, 0] = plot_TL(ax[0, 0], R, H, TL(u_m), "FDM-SSP", default_vmin=True)
    ax[0, 1] = plot_TL(ax[0, 1], R, H, TL(u3), "DST-SSP, q=4", default_vmin=True)
    ax[1, 0] = plot_TL(ax[1, 0], R, H, TL(u4), "DST-SSP, q=5", default_vmin=True)
    ax[1, 1] = plot_TL(ax[1, 1], R, H, TL(u5), "DST-SSP, q=6", default_vmin=True)
    # ax[1].imshow(TL(u.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[1].set_title("b)")
    fig.tight_layout()

    save = kwargs.get("save", False)
    if save:
        import matplot2tikz

        matplot2tikz.save("order_comparison.tex")

    plt.show()

    # if save:
    #     np.save("data/reference_propagation.npy", u_m)
    #     np.save("data/q2_propagation.npy", u1)
    #     np.save("data/q3_propagation.npy", u2)
    #     np.save("data/q4_propagation.npy", u3)
    #     np.save("data/q5_propagation.npy", u4)
    #     np.save("data/q6_propagation.npy", u5)

    # s1 = u1[izs,:]
    s2 = np.abs(u2[izs, :])
    s3 = np.abs(u3[izs, :])
    s4 = np.abs(u4[izs, :])
    s5 = np.abs(u5[izs, :])
    s_ref = np.abs(u_m[izs, :])
    r = np.linspace(0, R / 1000, s2.size)

    plt.figure
    plt.plot(r, s_ref, "k-", label="reference")
    # plt.plot(r, s1, "b.", label="q=2")
    plt.plot(r, s2, "r,", label="q=3")
    plt.plot(r, s3, "g,", label="q=4")
    plt.plot(r, s4, "b,", label="q=5")
    plt.plot(r, s5, "c,", label="q=6")
    plt.legend()
    if save:
        import matplot2tikz

        matplot2tikz.save("single_plot_order.tex")
    plt.show()

    return u_m, u2, u3


def main():
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1100  # source depth

    R = 10000
    dr = 10

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    iz0 = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    # cz = np.ones_like(z) * c0
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    krs = kz[iz0]

    k0sq = k0**2

    ksq = kz**2

    IC = (
        (1.4467 - 0.8402 * krs**2 * (z - zs) ** 2)
        * np.exp(-(krs**2) * (z - zs) ** 2 / 1.5256)
        / (2 * np.sqrt(np.pi))
    )

    Aj = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=False)
    Aj_m = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=True)

    fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(TL(u_m.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[0].set_title("a)")
    ax[0] = plot_TL(ax[0], R, H, TL(Aj_m), "TL\nSolution by matrix inversion")
    ax[1] = plot_TL(ax[1], R, H, TL(Aj), "Solution by DST + correction")
    # ax[1].imshow(TL(u.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[1].set_title("b)")
    fig.tight_layout()

    import matplot2tikz

    matplot2tikz.save("isovelocity.tex")

    plt.show()


def test_step():
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1100  # source depth

    R = 20
    dr = 10

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    iz0 = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    # cz = np.ones_like(z) * c0
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    kz[iz0]

    k0sq = k0**2

    ksq = kz**2

    IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")

    twosteps_u = propagate(IC, H, k0sq, ksq, dr, dz, R)
    twosteps_u_m = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=True)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("DST")
    ax[0].plot(twosteps_u[0], label="u0")
    ax[0].plot(twosteps_u[1], label="u1")
    ax[0].plot(twosteps_u[2], label="u2")
    ax[0].legend()

    ax[0].set_title("SSP")
    ax[1].plot(twosteps_u_m[0], label="u0")
    ax[1].plot(twosteps_u_m[1], label="u1")
    ax[1].plot(twosteps_u_m[2], label="u2")
    ax[1].legend()

    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("direct comparison")
    ax.plot(twosteps_u[1], "b-", label="u1 DST")
    ax.plot(twosteps_u[2], "r-", label="u2 DST")
    ax.plot(twosteps_u_m[1], "b.-", label="u1 SSP")
    ax.plot(twosteps_u_m[2], "r.-", label="u2 SSP")
    plt.legend()

    plt.show()


def test_inversion():
    for n in range(4, 8):
        N = n**2
        L = np.pi
        x = np.linspace(0, L, N)
        f = np.ones_like(x)
        u = np.abs(invert_SSP_operator(f, L, 1))
        U = constant_matrix(f, L, 1)
        Uhat = dst(U)
        uhat = dst(u)

        print(np.max(U))
        print(np.max(u))
        print(np.max(U) / np.max(u))
        print(Uhat / uhat)
        fig, ax = plt.subplots()
        ax.plot(f, label="f")
        ax.plot(U, label="matrix u")
        ax.plot(u, label="dst u")
        plt.legend()
        plt.show()


def test_normal_modes():
    f = 100
    omega = 100 * np.pi * 2
    zs = 1100
    N = 8001
    H = 4000
    z = np.linspace(0, H, N)
    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile

    ksq = omega**2 / cz**2
    k0sq = np.mean(ksq)

    dz = z[1] - z[0]
    R = 50000
    dr = 10
    field = normal_modes(f, zs, dr, R, H)
    IC = field[:, 0]
    field_dst = propagate(IC, H, k0sq, ksq, dr, dz, R)

    #    np.save("field_dst.npy", field_dst)
    #    np.save("field_normal_modes.npy", field)
    fig, ax = plt.subplots(2, 1)
    _TL = 20 * np.log10(np.abs(field.T))
    ax[0] = plot_TL(ax[0], R, H, _TL, "Normal modes", vmin=-120, vmax=-60)
    _TL_dst = 20 * np.log10(np.abs(field_dst))
    ax[1] = plot_TL(ax[1], R, H, _TL_dst, "DST propagation", vmin=-120, vmax=-60)

    import matplot2tikz

    matplot2tikz.save("normal_modes_comparison.tex")
    plt.show()

    return field


def compare_all_single_graph(**kwargs):
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1100  # source depth

    R = 40000
    dr = 25

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    iz0 = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    # cz = np.ones_like(z) * c0
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    kz[iz0]

    k0sq = k0**2

    ksq = kz**2

    # IC = (
    #    (1.4467 - 0.8402 * krs**2 * (z - zs) ** 2)
    #    * np.exp(-(krs**2) * (z - zs) ** 2 / 1.5256)
    #    / (2 * np.sqrt(np.pi))
    # )
    IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")
    Ps = []
    orders = list(range(0, 4))

    print("propagating using DST-SSP")
    for corrector_order in orders:
        print(f"propagating order {corrector_order}")
        Ps.append(
            propagate(
                IC,
                H,
                k0sq,
                ksq,
                dr,
                dz,
                R,
                matrix=False,
                correction_order=corrector_order,
                normalize=True,
            )
        )

    # print("propagating fd")
    # P_FD = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=True)
    print("Plotting")
    fig, ax = plt.subplots(len(orders) // 2, 2)
    # ax[0] = plot_TL(ax[0], R, H, TL(P_FD), f"Finite_differences", vmax=-10, vmin=-50)
    for i, o in enumerate(orders):
        idx1 = i // 2
        idx2 = i % 2
        ax[idx1, idx2] = plot_TL(
            ax[idx1, idx2],
            R,
            H,
            TL(Ps[i]),
            f"corrector of order {o}",
            vmax=-10,
            vmin=-50,
        )
    # ax[1].imshow(TL(u.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[1].set_title("b)")
    fig.tight_layout()

    save = kwargs.get("save", False)
    if save:
        import matplot2tikz

        matplot2tikz.save("correction_order.tex")

    fig.savefig("comparison_orders.pdf")
    plt.show()
    return Ps


def test_correction_order(**kwargs):
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1300  # source depth

    R = 50000
    dr = 100

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    iz0 = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    # cz = np.ones_like(z) * c0
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    kz[iz0]

    k0sq = k0**2

    ksq = kz**2

    # IC = (
    #    (1.4467 - 0.8402 * krs**2 * (z - zs) ** 2)
    #    * np.exp(-(krs**2) * (z - zs) ** 2 / 1.5256)
    #    / (2 * np.sqrt(np.pi))
    # )
    IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")
    Ps = []
    orders = list(range(0, 4))

    print("propagating using DST-SSP")
    for corrector_order in orders:
        print(f"propagating order {corrector_order}")
        Ps.append(
            propagate(
                IC,
                H,
                k0sq,
                ksq,
                dr,
                dz,
                R,
                matrix=False,
                correction_order=corrector_order,
                normalize=True,
            )
        )

    p_normal_modes = normal_modes(100, 1100, dr, R, H).T
    n = l2norm(Ps[-1][0], dz)
    for i in range(1, p_normal_modes.shape[0]):
        p_normal_modes[i] = normalize(p_normal_modes[i], dz, n)

    izs = np.nonzero(z > zs)[0][0]
    s = []
    for i in range(len(orders)):
        s.append(Ps[i][:, izs])

    for o, p in zip(orders, Ps):
        np.save(f"order_{o}_100km.npy", p, allow_pickle=True)

    # print("propagating fd")
    # P_FD = propagate(IC, H, k0sq, ksq, dr, dz, R, matrix=True)
    print("Plotting")
    fig, ax = plt.subplots(len(orders) // 2 + 1, 2)
    # ax[0] = plot_TL(ax[0], R, H, TL(P_FD), f"Finite_differences", vmax=-10, vmin=-50)
    for i, o in enumerate(orders):
        idx1 = i // 2
        idx2 = i % 2
        img = plot_TL(
            ax[idx1, idx2],
            R,
            H,
            logField(Ps[i]),
            f"corrector of order {o}",
            vmin=-110,
            vmax=-60,
        )
        fig.colorbar(img, ax=ax[idx1, idx2])
    img = plot_TL(
        ax[-1, 0],
        R,
        H,
        logField(p_normal_modes),
        "reference normal modes",
        vmin=-110,
        vmax=-60,
    )
    fig.colorbar(img, ax=ax[-1, 0])
    r = np.linspace(0, R / 1000, s[0].size)
    for i in range(len(orders)):
        ax[-1, 1].plot(r, logField(s[i]), "-.", label=f"order {orders[i]}")
    ax[-1, 1].plot(
        r, logField(p_normal_modes[:, izs]), "-", label="normal modes reference"
    )
    ax[-1, 1].set_xlabel("r [km]")
    ax[-1, 1].set_ylabel("log(|u|)")
    ax[-1, 1].legend()
    # ax[1].imshow(TL(u.T), vmin=-75, vmax=-25, aspect="auto", cmap="jet")
    # ax[1].set_title("b)")
    fig.tight_layout()

    save = kwargs.get("save", False)
    if save:
        import matplot2tikz

        logging.info("saving")
        matplot2tikz.save("graphics/correction_order.tex")

    plt.show()
    return Ps


def compare_discretizations(**kwargs):
    f = 100
    omega = f * 2 * np.pi
    H = 4000  # total depth
    zs = 1300  # source depth

    R = 50000
    dr = 100

    Z = np.linspace(0, H, 8001)
    ns = list(range(7, 10))

    s = []
    dzs = []
    Ps = []
    Ps_dst = []
    for n in ns:
        N = 2**n
        z = np.linspace(0, H, N)
        dz = z[1] - z[0]
        dzs.append(dz)

        # index of source
        iz0 = np.nonzero(z >= zs)[0][0]

        z_tilde = 2 * (z - 1300) / 1300
        cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
        # cz = np.ones_like(z) * c0
        c0 = np.mean(cz)
        k0 = omega / c0

        kz = omega / cz
        kz[iz0]

        k0sq = k0**2

        ksq = kz**2

        IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")
        IC = np.interp(z, Z, IC)
        order = 4

        logging.info("propagating using DST-SSP")
        logging.info(f"propagating with {N=}")
        Ps_dst.append(
            propagate(
                IC,
                H,
                k0sq,
                ksq,
                dr,
                dz,
                R,
                matrix=False,
                correction_order=order,
                normalize=True,
            )
        )
        Ps.append(
            propagate(
                IC,
                H,
                k0sq,
                ksq,
                dr,
                dz,
                R,
                matrix=True,
                correction_order=order,
                normalize=True,
            )
        )

        izs = np.nonzero(z > zs)[0][0]
        s.append(Ps[-1][:, izs])

    p_normal_modes = normal_modes(100, 1100, dr, R, H).T
    n = l2norm(Ps[-1][0], dz)
    for i in range(1, p_normal_modes.shape[0]):
        p_normal_modes[i] = normalize(p_normal_modes[i], dz, n)

    fig, ax = plt.subplots(nrows=len(ns), ncols=2)
    for i in range(len(ns)):
        img = plot_TL(ax[i, 0], R, H, logField(Ps[i]), vmin=-110, vmax=-60)
        N = 2 ** ns[i]

        ax[i, 0].set_title(f"FDM {N=}")
        img = plot_TL(ax[i, 1], R, H, logField(Ps_dst[i]), vmin=-110, vmax=-60)
        ax[i, 1].set_title(f"DST {N=}")

    fig.colorbar(img, ax=ax, orientation="vertical")

    save = kwargs.get("save", False)
    if save:
        import matplot2tikz

        matplot2tikz.save("graphics/discretizations.tex")

    plt.show()


def compare_stepsizes(**kwargs):
    save = kwargs.get("save", False)
    f = 100
    omega = f * 2 * np.pi
    N = 8001
    H = 4000  # total depth
    zs = 1100  # source depth

    R = 100000
    dr = 20

    z = np.linspace(0, H, N)
    dz = z[1] - z[0]

    # index of source
    izs = np.nonzero(z >= zs)[0][0]

    z_tilde = 2 * (z - 1300) / 1300
    cz = 1500 * (1.0 + 0.00737 * (z_tilde - 1 + np.exp(-z_tilde)))  # Munk profile
    c0 = np.mean(cz)
    k0 = omega / c0

    kz = omega / cz
    kz[izs]

    k0sq = k0**2

    ksq = kz**2

    IC = np.loadtxt(f"modes/IC_{int(f):}Hz.txt")
    Ps = []
    rs = []

    stepsizes = [10, 50, 100, 200]

    R = 10000

    for dr in stepsizes:
        Ps.append(propagate(IC, H, k0sq, ksq, dr, dz, R, correction_order=2, nP=4))
        n_steps = int(R / dr) + 1
        r = np.linspace(0, R, n_steps)
        rs.append(r)

    # for higher order scheme include 500m as stepsize
    PP = []
    stepsizes.append(300)
    rs.append(np.linspace(0, R, int(R / stepsizes[-1]) + 1))
    for dr in stepsizes:
        PP.append(propagate(IC, H, k0sq, ksq, dr, dz, R, correction_order=4, nP=10))

    markers = ["-", ".", "+", "x", "d"]
    plt.figure()
    for i in range(len(Ps)):
        r = rs[i]
        P = Ps[i]
        col = "b"
        spec = markers[i] + col
        s = TL(P)[:, izs]
        dr = stepsizes[i]
        plt.plot(r / 1000, s, spec, label=f"dr={dr}, p=4, q=2")
    plt.ylabel("TL")
    plt.xlabel("r [km]")

    for i in range(len(rs)):
        P = PP[i]
        r = rs[i]
        s = TL(P)[:, izs]
        col = "r"
        spec = markers[i] + col
        dr = stepsizes[i]
        plt.plot(r / 1000, s, spec, label=f"dr={dr}, p=10, q=4")

    plt.title("Step Size comparison")
    plt.legend()

    if save:
        import matplot2tikz

        matplot2tikz.save("stepsize_comparison.tex")
    plt.show()
    return (
        stepsizes,
        rs,
        Ps,
        PP,
    )


logging.basicConfig(level=logging.INFO)
os.makedirs("graphics", exist_ok=True)  # savedir for the graphics generated

eddy(
    np.linspace(0, 4000, 2048), np.linspace(0, 100_000, 1000), 2000, save=True
)  # propagation through the eddy
test_correction_order(save=True)  # order of correction comparisons
compare_discretizations(save=True)  # discretization / aliasing comparisons
