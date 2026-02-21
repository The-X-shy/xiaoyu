"""Zernike polynomial generation and wavefront reconstruction.

Conventions:
- Uses Noll sequential indexing (1-based).
- Wavefront is defined on unit disk; outside values are zero.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert Noll index j (1-based) to radial degree n and azimuthal order m.

    Uses the standard Noll ordering convention.
    Reference: Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence."

    j=1:(0,0), j=2:(1,1), j=3:(1,-1), j=4:(2,0), j=5:(2,-2), j=6:(2,2),
    j=7:(3,-1), j=8:(3,1), j=9:(3,-3), j=10:(3,3), j=11:(4,0), ...
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")

    # Find n such that n(n+1)/2 < j <= (n+1)(n+2)/2
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1

    # Position within order n (0-based)
    i = j - n * (n + 1) // 2 - 1

    # Build the |m| sequence for order n (Noll convention).
    # Within each order n, |m| values go ascending: 0 (or 1), 2 (or 3), ..., n
    # Each |m| > 0 appears twice (for +m and -m), |m| = 0 appears once.
    m_abs_sequence = []
    for ma in range(n % 2, n + 1, 2):
        if ma == 0:
            m_abs_sequence.append(0)
        else:
            m_abs_sequence.append(ma)
            m_abs_sequence.append(ma)

    m_abs = m_abs_sequence[i]

    if m_abs == 0:
        return (n, 0)

    # Determine sign: even j -> positive m, odd j -> negative m
    if j % 2 == 0:
        return (n, m_abs)
    else:
        return (n, -m_abs)


def num_zernike_terms(max_order: int) -> int:
    """Number of Zernike terms up to and including radial order max_order.

    = (max_order + 1) * (max_order + 2) // 2
    """
    return (max_order + 1) * (max_order + 2) // 2


def _factorial(n: int) -> int:
    """Compute n! for non-negative integer n."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def zernike_radial(n: int, m_abs: int, r: np.ndarray) -> np.ndarray:
    """Compute radial polynomial R_n^|m|(r).

    Args:
        n: Radial degree.
        m_abs: Absolute value of azimuthal order.
        r: Radial coordinate array.

    Returns:
        R_n^|m| evaluated at r.
    """
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(r, dtype=float)

    result = np.zeros_like(r, dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        num = (-1) ** s * _factorial(n - s)
        den = (
            _factorial(s)
            * _factorial((n + m_abs) // 2 - s)
            * _factorial((n - m_abs) // 2 - s)
        )
        result = result + (num / den) * r ** (n - 2 * s)
    return result


def zernike_polynomial(j: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate Zernike polynomial Z_j at polar coordinates (r, theta).

    Uses Noll normalization convention.

    Args:
        j: Noll index (1-based).
        r: Radial coordinate array.
        theta: Angular coordinate array (radians).

    Returns:
        Z_j evaluated at (r, theta).
    """
    n, m = noll_to_nm(j)
    m_abs = abs(m)
    R = zernike_radial(n, m_abs, r)

    # Normalization factor
    if m == 0:
        norm = np.sqrt(float(n + 1))
    else:
        norm = np.sqrt(2.0 * (n + 1))

    if m >= 0:
        return norm * R * np.cos(m_abs * theta)
    else:
        return norm * R * np.sin(m_abs * theta)


def zernike_wavefront(
    coeffs: np.ndarray,
    grid_size: int,
    max_order: Optional[int] = None,
) -> np.ndarray:
    """Generate 2D wavefront from Zernike coefficients on unit disk grid.

    Args:
        coeffs: 1D array, coeffs[0] = coeff for Noll j=1 (piston), etc.
        grid_size: Output grid size (grid_size x grid_size).
        max_order: Max radial order (unused, kept for API compat).

    Returns:
        2D array (grid_size, grid_size). Zero outside unit circle.
    """
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = R <= 1.0

    W = np.zeros((grid_size, grid_size), dtype=float)
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-15:
            continue
        j = i + 1  # Noll index
        Z = zernike_polynomial(j, R, Theta)
        W += c * Z

    W[~mask] = 0.0
    return W


def zernike_gradient(
    coeffs: np.ndarray,
    grid_size: int,
    max_order: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute wavefront gradient (dW/dx, dW/dy) via finite differences.

    Args:
        coeffs: Zernike coefficients (Noll ordering).
        grid_size: Grid size.
        max_order: Max radial order (unused, kept for API compat).

    Returns:
        (dWdx, dWdy) tuple, each (grid_size, grid_size). Zero outside disk.
    """
    W = zernike_wavefront(coeffs, grid_size, max_order)
    dx = 2.0 / (grid_size - 1)
    dWdy, dWdx = np.gradient(W, dx, dx)

    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    mask = np.sqrt(X**2 + Y**2) <= 1.0
    dWdx[~mask] = 0.0
    dWdy[~mask] = 0.0
    return dWdx, dWdy
