"""Direct numerical evaluation of the Fraunhofer diffraction integral by composite
Simpson quadrature, used as an independent cross-check of the FFT propagator.

The Fraunhofer amplitude at a screen point (X, Y) is
    A_tilde(X, Y) = integral of aperture(x, y) * exp(-i 2pi/(lambda z) (x X + y Y)) dx dy,
evaluated with Simpson weights. The 2D routine sums over the full aperture grid for
each screen point, giving the O(M^2 N^2) cost (M screen points per axis, N aperture
samples per axis) used in the efficiency experiment. Use small sizes only.
"""

import numpy as np

from . import quadrature

__all__ = ["fraunhofer_simpson_1d", "fraunhofer_simpson_2d"]


def fraunhofer_simpson_1d(field, dx, wavelength, z, x_screen):
    """1D Fraunhofer amplitude at the given screen points via composite Simpson."""
    n = field.shape[-1]
    axis = (np.arange(n) - n // 2) * dx
    weighted = quadrature.simpson_weights(n - 1, dx) * field  # fold weights into aperture
    factor = 2 * np.pi / (wavelength * z)
    # kernel[i, m] = exp(-i factor x_m X_i); one matrix-vector product per screen point.
    kernel = np.exp(-1j * factor * np.outer(x_screen, axis))
    return kernel @ weighted


def fraunhofer_simpson_2d(field, dx, wavelength, z, x_screen, y_screen):
    """2D Fraunhofer amplitude on a screen grid via direct Simpson quadrature.

    Cost is O(M^2 N^2): for each of the M_y x M_x screen points the full N x N
    aperture sum is formed explicitly (no separability shortcut).
    """
    n = field.shape[-1]
    axis = (np.arange(n) - n // 2) * dx
    w = quadrature.simpson_weights(n - 1, dx)
    weighted = np.outer(w, w) * field  # 2D Simpson weights folded into the aperture
    factor = 2 * np.pi / (wavelength * z)

    # Precompute per-axis phase factors; combine them per screen point.
    phase_x = np.exp(-1j * factor * np.outer(x_screen, axis))  # (M_x, N)
    phase_y = np.exp(-1j * factor * np.outer(y_screen, axis))  # (M_y, N)

    out = np.empty((y_screen.size, x_screen.size), dtype=complex)
    for iy in range(y_screen.size):
        ky = phase_y[iy]
        for ix in range(x_screen.size):
            kernel = ky[:, None] * phase_x[ix][None, :]
            out[iy, ix] = np.sum(weighted * kernel)
    return out
