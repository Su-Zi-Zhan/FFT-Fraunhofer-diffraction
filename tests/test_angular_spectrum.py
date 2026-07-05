"""Tests for the angular spectrum propagator: identity, energy, reversibility, beam physics."""

import numpy as np

from src import grids
from src import angular_spectrum as ASM


def _gaussian_beam_field(n, dx, w0):
    """Flat-phase Gaussian aperture exp(-r^2/w0^2): a Gaussian beam at its waist."""
    X, Y = grids.square_grid(n, dx)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(complex)


def test_zero_distance_is_identity():
    n, dx, lam = 128, 1e-6, 0.5e-6
    field = _gaussian_beam_field(n, dx, 1e-5)
    out = ASM.propagate(field, dx, lam, 0.0)
    assert np.allclose(out, field, atol=1e-12)


def test_energy_conserved_for_bandlimited_field():
    # A smooth Gaussian has negligible evanescent content, so |H| ~ 1 and Parseval holds.
    n, dx, lam = 512, 7.8e-7, 0.5e-6
    field = _gaussian_beam_field(n, dx, 5e-5)
    out = ASM.propagate(field, dx, lam, 5e-3)
    e0 = np.sum(np.abs(field) ** 2)
    ez = np.sum(np.abs(out) ** 2)
    assert np.isclose(ez, e0, rtol=1e-6)


def test_propagation_is_reversible():
    # Propagating +z then -z returns the original field (propagating components only).
    n, dx, lam = 512, 7.8e-7, 0.5e-6
    field = _gaussian_beam_field(n, dx, 5e-5)
    there = ASM.propagate(field, dx, lam, 5e-3)
    back = ASM.propagate(there, dx, lam, -5e-3)
    assert np.allclose(back, field, atol=1e-9)


def test_gaussian_beam_peak_decay_matches_paraxial():
    # At one Rayleigh range the waist grows by sqrt(2), so peak intensity halves.
    n, dx, lam = 512, 7.8e-7, 0.5e-6
    w0 = 5e-5
    z_rayleigh = np.pi * w0 ** 2 / lam
    field = _gaussian_beam_field(n, dx, w0)
    out = ASM.propagate(field, dx, lam, z_rayleigh)
    peak_ratio = np.abs(out[n // 2, n // 2]) ** 2 / np.abs(field[n // 2, n // 2]) ** 2
    assert np.isclose(peak_ratio, 0.5, rtol=0.03)
