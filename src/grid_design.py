"""Sampling-grid design for FFT diffraction (Section 4 of the project plan).

Given the physical setup and screen targets, pick the minimal grid (N, dx, L) that
meets three constraints:
  1. screen range       X_max = lambda*z/(2 dx) >= X_max_target  =>  dx <= lambda*z/(2 X_max_target)
  2. screen resolution   dX   = lambda*z / L     <= dX_target     =>  L  >= lambda*z / dX_target
  3. aperture-edge resolution                                     =>  dx <= a / k_edge
N is rounded up to the next power of two for the radix-2 FFT.
"""

import numpy as np

from .grids import next_power_of_two

__all__ = ["screen_sampling", "design_grid"]


def screen_sampling(n, dx, wavelength, z):
    """Achieved (X_max, dX) on the screen for an (n, dx) aperture grid."""
    x_max = wavelength * z / (2 * dx)
    d_screen = wavelength * z / (n * dx)
    return x_max, d_screen


def design_grid(a, wavelength, z, x_max_target, dx_screen_target, k_edge=20):
    """Minimal (N, dx, L) meeting the three constraints; reports which bounds dx."""
    dx_nyquist = wavelength * z / (2 * x_max_target)   # screen-range bound on dx
    dx_edge = a / k_edge                               # aperture-edge bound on dx
    dx = min(dx_nyquist, dx_edge)
    binding = "nyquist" if dx_nyquist <= dx_edge else "edge"

    # Window must reach the resolution target; N covers it, rounded up to a power of two.
    l_min = wavelength * z / dx_screen_target
    n = next_power_of_two(np.ceil(l_min / dx))
    length = n * dx
    x_max, d_screen = screen_sampling(n, dx, wavelength, z)
    return {
        "N": n,
        "dx": dx,
        "L": length,
        "dx_nyquist": dx_nyquist,
        "dx_edge": dx_edge,
        "binding": binding,
        "x_max_achieved": x_max,
        "dx_screen_achieved": d_screen,
    }
