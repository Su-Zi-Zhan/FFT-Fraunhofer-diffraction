"""Tests for the Fraunhofer/Fresnel FFT propagators against analytic limits."""

import numpy as np

from src import grids, apertures, analytic
from src import fft_diffraction as D


def _first_null_radius(x_screen, intensity):
    """Radius of the first intensity minimum to the right of the central peak."""
    center = np.argmax(intensity)
    half = intensity[center:]
    # First index where the profile stops decreasing = first local minimum.
    drop = np.diff(half) < 0
    turn = np.argmax(~drop[1:] & drop[:-1]) + 1
    return abs(x_screen[center + turn] - x_screen[center])


def test_screen_axis_spacing_and_symmetry():
    n, dx, lam, z = 256, 8e-6, 0.5e-6, 1.0
    x = D.screen_axis(n, dx, lam, z)
    assert np.isclose(x[1] - x[0], lam * z / (n * dx))      # ΔX = λz/(NΔx)
    assert np.isclose(x[n // 2], 0.0)                       # origin centered


def test_centered_delta_gives_flat_magnitude():
    # FT of a delta at the origin is a constant, so |U| must be uniform.
    n = 64
    field = np.zeros(n)
    field[n // 2] = 1.0
    u = D.fraunhofer_field_1d(field, dx=1.0)
    assert np.allclose(np.abs(u), np.abs(u[0]))


def test_single_slit_first_null_matches_analytic():
    n, lam, z, a = 2048, 0.5e-6, 1.0, 1e-3
    dx = 16 * a / n                                         # window L = 16a (zero padding)
    x = grids.centered_axis(n, dx)
    field = apertures.single_slit(x, a)
    x_screen, intensity = D.fraunhofer_pattern_1d(field, dx, lam, z)
    r_num = _first_null_radius(x_screen, intensity)
    r_ref = analytic.single_slit_first_null(a, lam, z)
    dX = lam * z / (n * dx)
    assert abs(r_num - r_ref) < 1.5 * dX                    # within one screen sample


def test_circle_first_ring_matches_airy():
    n, lam, z, diam = 1024, 0.5e-6, 1.0, 1e-3
    dx = 16 * diam / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.circle(X, Y, diam)
    x_screen, intensity = D.fraunhofer_pattern_2d(field, dx, lam, z)
    row = intensity[n // 2]                                 # central horizontal cut
    r_num = _first_null_radius(x_screen, row)
    r_ref = analytic.airy_first_null(diam, lam, z)
    assert np.isclose(r_num, r_ref, rtol=0.03)              # Rayleigh radius within 3%


def test_fresnel_approaches_fraunhofer_for_large_z():
    # As z grows (F -> 0) the aperture chirp -> 1, so Fresnel -> Fraunhofer.
    n, lam, a = 512, 0.5e-6, 5e-4
    dx = 16 * a / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.rectangle(X, Y, a, a)
    far = D.fraunhofer_field_2d(field, dx)
    near = D.fresnel_field_2d(field, dx, lam, z=200.0)      # F = a^2/(λz) ~ 2.5e-3
    i_far = np.abs(far) ** 2
    i_near = np.abs(near) ** 2
    rel = np.linalg.norm(i_near - i_far) / np.linalg.norm(i_far)
    assert rel < 0.02
