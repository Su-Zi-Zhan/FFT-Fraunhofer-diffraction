"""Single-FFT diffraction propagators (Fraunhofer and Fresnel) on the hand-written FFT.

The Fraunhofer field is the Fourier transform of the aperture; the Fresnel field is
the transform of the aperture times an aperture-plane quadratic phase (chirp). Both
are sampled on the same screen grid X = lambda*z*f_x, so they are directly comparable.
Centering uses ifftshift on the input and fftshift on the output, which removes the
(-1)^p phase that arises from sampling an origin-centered aperture (see reports/math).
The constant Fresnel prefactor exp(ikz)/(i*lambda*z) is dropped: only normalized
intensity is ever compared, and a global constant does not change shape or null positions.
"""

import numpy as np

from . import fft as F

__all__ = [
    "screen_axis",
    "fraunhofer_field_1d",
    "fraunhofer_field_2d",
    "fresnel_field_2d",
    "fraunhofer_pattern_1d",
    "fraunhofer_pattern_2d",
]


def screen_axis(n, dx, wavelength, z):
    """Centered screen-coordinate axis X = lambda*z*f; spacing is lambda*z/(n*dx)."""
    return wavelength * z * F.fftshift(F.fftfreq(n, dx))


def _centered_ft_1d(field):
    """Sampled FT of an origin-centered 1D field, returned in centered order."""
    return F.fftshift(F.fft(F.ifftshift(field)))


def _centered_ft_2d(field):
    """Sampled FT of an origin-centered 2D field, returned in centered order."""
    return F.fftshift(F.fft2(F.ifftshift(field)))


def fraunhofer_field_1d(field, dx):
    """1D Fraunhofer amplitude: aperture FT scaled by the dx Riemann factor."""
    return _centered_ft_1d(field) * dx


def fraunhofer_field_2d(field, dx):
    """2D Fraunhofer amplitude: aperture FT scaled by the dx^2 Riemann factor."""
    return _centered_ft_2d(field) * dx ** 2


def fresnel_field_2d(field, dx, wavelength, z):
    """2D Fresnel amplitude on the same screen grid as Fraunhofer.

    Multiplies the aperture by the quadratic phase exp(i k/(2z)(x^2+y^2)) that the
    Fraunhofer approximation drops; keeping it yields the near-field reference.
    """
    n = field.shape[-1]
    axis = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(axis, axis)
    k = 2 * np.pi / wavelength
    chirp = np.exp(1j * k / (2 * z) * (xx ** 2 + yy ** 2))
    return _centered_ft_2d(field * chirp) * dx ** 2


def fraunhofer_pattern_1d(field, dx, wavelength, z):
    """Convenience wrapper: return (screen axis, intensity) for a 1D aperture."""
    x_screen = screen_axis(field.shape[-1], dx, wavelength, z)
    return x_screen, np.abs(fraunhofer_field_1d(field, dx)) ** 2


def fraunhofer_pattern_2d(field, dx, wavelength, z):
    """Convenience wrapper: return (screen axis, intensity) for a 2D aperture."""
    x_screen = screen_axis(field.shape[-1], dx, wavelength, z)
    return x_screen, np.abs(fraunhofer_field_2d(field, dx)) ** 2
