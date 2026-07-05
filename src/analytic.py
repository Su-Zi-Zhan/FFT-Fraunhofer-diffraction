"""Closed-form Fraunhofer intensity patterns (peak-normalized) and first-null
locations, used as ground truth for experiment 1.

scipy.special supplies the Bessel function J1 for the Airy pattern; this is a
reference solution, not part of the FFT main algorithm.
"""

import numpy as np
from scipy import special

__all__ = [
    "AIRY_FIRST_ZERO",
    "RAYLEIGH_CONSTANT",
    "single_slit_intensity",
    "single_slit_first_null",
    "double_slit_intensity",
    "rectangle_intensity",
    "airy_intensity",
    "airy_first_null",
]

# First zero of J1 and the Rayleigh constant (= first zero / pi ~ 1.21966989).
AIRY_FIRST_ZERO = float(special.jn_zeros(1, 1)[0])
RAYLEIGH_CONSTANT = AIRY_FIRST_ZERO / np.pi


def single_slit_intensity(x_screen, width, wavelength, z):
    """Fraunhofer intensity of a slit of given width, peak-normalized to 1."""
    u = width * x_screen / (wavelength * z)
    return np.sinc(u) ** 2  # numpy sinc is sin(pi u)/(pi u)


def single_slit_first_null(width, wavelength, z):
    """Screen position of the first dark fringe: X = lambda*z / width."""
    return wavelength * z / width


def double_slit_intensity(x_screen, slit_width, separation, wavelength, z):
    """Double-slit intensity: single-slit envelope times two-beam interference."""
    envelope = np.sinc(slit_width * x_screen / (wavelength * z)) ** 2
    interference = np.cos(np.pi * separation * x_screen / (wavelength * z)) ** 2
    return envelope * interference


def rectangle_intensity(X, Y, width, height, wavelength, z):
    """Fraunhofer intensity of a rectangular aperture, peak-normalized to 1."""
    ux = width * X / (wavelength * z)
    uy = height * Y / (wavelength * z)
    return (np.sinc(ux) * np.sinc(uy)) ** 2


def airy_intensity(r_screen, diameter, wavelength, z):
    """Airy pattern [2 J1(v)/v]^2 with v = pi*D*r/(lambda*z), peak-normalized."""
    v = np.pi * diameter * np.asarray(r_screen, dtype=float) / (wavelength * z)
    out = np.ones_like(v)
    nz = v != 0  # limit at v=0 is 1
    out[nz] = (2 * special.j1(v[nz]) / v[nz]) ** 2
    return out


def airy_first_null(diameter, wavelength, z):
    """Radius of the first dark ring: r = 1.21967 * lambda*z / D."""
    return RAYLEIGH_CONSTANT * wavelength * z / diameter
