import numpy as np
import math
from scipy.sparse import spdiags
from numpy._typing import _ArrayLike


def pade0fromT(cTaylor: _ArrayLike, xo: float, n: int, m: int):
    """
    PADE computes the Padè approximant of the function F at the point XO.

    Parameters:
    cTaylor : array-like
        Taylor coefficients of the approximated function
    xo : float
        Point around which the approximation is made
    n : int
        Order of the numerator
    m : int
        Order of the denominator

    Returns:
    p : array
        Coefficients of the numerator polynomial
    q : array
        Coefficients of the denominator polynomial

    FROM MATLAB:
    % PADE computes the Padè approximant of the function F at the point XO.

    [P,Q]=PADE(cTaylor,XO,N,M) returns two polynomial forms, P of order N and Q of order M,
    representing the denominator and the numerator of the rational form
    which approximates the function F around XO.
    cTaylor contains taylor coefficients of the approximated function
    N and M must be positive integer values.


    Reference paper: Baker, G. A. Jr. Essentials of Padé Approximants in
    Theoretical Physics. New York: Academic Press, pp. 27-38, 1975.
    see also
    http://mathworld.wolfram.com/PadeApproximant.html

    This routine has been programmed by Luigi Sanguigno, Phd.
    Affiliation: Italian Institute of Technology
    Modified and slightly corrected by Pavel Petrov PhD
    Affiliation: Il'ichev Pacific Oceanological Inst

    Slightly modified and translated to python by Daniel Walsken
    Affiliation: Bergische Universität Wuppertal - IMACM

    Calculate n+m Taylor coeffs at xo

    """

    # Calculate n+m Taylor coeffs at xo
    a = np.array(cTaylor).T

    # Calculation of Padè coefficients.
    top = np.vstack((np.eye(n + 1), np.zeros((m, n + 1))))
    bottom = spdiags(
        np.repeat(-np.concatenate((a[::-1], [0]))[None, ...], n + m + 1, axis=0).T,
        np.arange(-(n + m + 1), 1),
        n + m + 1,
        m,
    ).todense()
    pq = np.linalg.solve(np.hstack((top, bottom)), a)

    # Rewrite the output as evaluable polynomial forms.
    p = shiftpoly(pq[n::-1], xo)
    q = shiftpoly(np.concatenate((pq[-1:n:-1], [1])), xo)

    return p, q


def shiftpoly(p: _ArrayLike, xo: float):
    """
    Displaces the origin of -xo

    Parameters:
    p : array-like
        Coefficients of the polynomial
    xo : float
        Value to shift the polynomial

    Returns:
    ps : array
        Coefficients of the shifted polynomial
    """

    # Initialize values.
    ps = np.zeros_like(p)
    q = np.array([1])
    base = np.array([1, -xo])
    ps[-1] = p[-1]

    # Substitute the base polynomial in the original polynomial form.
    for n in range(1, len(p)):
        q = np.convolve(q, base)
        ps[-(n + 1) :] += p[-(n + 1)] * q

    return ps


def tay_expsqrt(hk0: float, N: int, envelope: bool = True):
    # derivatives for Taylor coeffs
    cTaylor = np.zeros(N, dtype=complex)
    cTaylor[0] = 1

    vp = np.zeros(2 * N, dtype=complex)
    vc = np.zeros(2 * N, dtype=complex)
    vp[1] = 1j * hk0 / 2

    for ii in range(2, N + 1):
        cTaylor[ii - 1] = np.sum(vp) / math.factorial(ii - 1)
        vc.fill(0)

        for jj in range(3, 2 * ii + 1):
            vc[jj - 1] = -vp[jj - 3] * (jj - 3) / 2 + vp[jj - 2] * 1j * hk0 / 2

        vp = vc.copy()

    if not envelope:
        cTaylor *= np.exp(1j * hk0)
    return cTaylor


def pade_expsqrt(hk0: float, n: int, m: int, envelope: bool = True):
    cTaylor = tay_expsqrt(hk0, n + m + 1, envelope)
    p, q = pade0fromT(cTaylor, 0, n, m)
    return p, q


def pade_coefficients(hk0: float, nP: int, envelope: bool = True):
    pP, qP = pade_expsqrt(hk0, nP, nP, envelope)

    cP = -1 / np.roots(pP)
    bP = -1 / np.roots(qP)

    aP = np.zeros((bP.size), dtype=np.complex128)

    for ii in range(nP):
        aP[ii] = (cP[ii] - bP[ii]) * np.prod(
            (cP[np.r_[0:ii, ii + 1 : nP]] - bP[ii])
            / (bP[np.r_[0:ii, ii + 1 : nP]] - bP[ii])
        )

    if nP == 1:
        aP[0] = cP[0] - bP[0]
    dP = -aP / bP
    dP = np.concatenate(([1 - sum(dP)], dP))
    return bP, dP


if __name__ == "__main__":
    b, d = pade_coefficients(4.0575, 8)
