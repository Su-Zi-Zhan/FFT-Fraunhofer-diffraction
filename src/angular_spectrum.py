"""Angular Spectrum Method (ASM) for near-field propagation, built on the hand-written FFT.

A field is decomposed into plane waves, each propagated by its own phase, then
recombined:
    U(z) = IFFT{ FFT{U(0)} * H },   H = exp(i k z sqrt(1 - (lambda f_x)^2 - (lambda f_y)^2)).
Evanescent components (lambda^2 (f_x^2 + f_y^2) > 1) get a real exponential decay.
The transfer function is built on the natural-order frequency grid to match the FFT
output ordering; ifftshift/fftshift move the field between centered and natural order.
Output lives on the same grid as the input, so ASM is the near-field reference where
the Fraunhofer approximation breaks down.
"""

import numpy as np

from . import fft as F

__all__ = ["transfer_function", "propagate"]


def transfer_function(n, dx, wavelength, z):
    """ASM transfer function H on the natural-order (fftfreq) frequency grid."""
    f = F.fftfreq(n, dx)
    fx, fy = np.meshgrid(f, f)
    arg = 1.0 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2
    k = 2 * np.pi / wavelength

    h = np.empty(arg.shape, dtype=np.complex128)
    propagating = arg >= 0
    # Propagating waves: oscillatory phase. Evanescent waves: decay over distance z.
    h[propagating] = np.exp(1j * k * z * np.sqrt(arg[propagating]))
    h[~propagating] = np.exp(-k * z * np.sqrt(-arg[~propagating]))
    return h


def propagate(field, dx, wavelength, z):
    """Propagate a 2D field a distance z by the angular spectrum method."""
    h = transfer_function(field.shape[-1], dx, wavelength, z)
    spectrum = F.fft2(F.ifftshift(field))      # natural-order spectrum
    return F.fftshift(F.ifft2(spectrum * h))   # apply H, return to centered order
