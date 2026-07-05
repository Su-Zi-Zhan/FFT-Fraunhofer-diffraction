"""Tests for composite Simpson quadrature and the direct Simpson diffraction solver."""

import numpy as np

from src import grids, apertures, quadrature
from src import simpson_diffraction as S
from src import fft_diffraction as D


def test_simpson_exact_for_cubic():
    # Simpson (both 1/3 and 3/8) integrates cubics exactly; test even and odd intervals.
    for n in (8, 9):
        x = np.linspace(0.0, 2.0, n + 1)
        f = 1 + 2 * x - 3 * x ** 2 + 4 * x ** 3
        exact = 2 + 4 - 8 + 16  # integral on [0,2] of the polynomial
        assert np.isclose(quadrature.composite_simpson(f, x[1] - x[0]), exact)


def test_simpson_complex_integrand():
    # integral_0^pi exp(i x) dx = (exp(i pi) - 1)/i = 2i.
    n = 256
    x = np.linspace(0.0, np.pi, n + 1)
    val = quadrature.composite_simpson(np.exp(1j * x), x[1] - x[0])
    assert np.isclose(val, 2j, atol=1e-6)


def test_simpson_fourth_order_convergence():
    # Smooth integrand on even-interval grids -> error ~ h^4 (log-log slope ~ 4).
    exact = np.e - 1.0
    hs, errs = [], []
    for n in (8, 16, 32, 64, 128):
        x = np.linspace(0.0, 1.0, n + 1)
        err = abs(quadrature.composite_simpson(np.exp(x), x[1] - x[0]) - exact)
        hs.append(1.0 / n)
        errs.append(err)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 3.7 < slope < 4.3


def test_simpson_1d_matches_fft_in_central_region():
    # Simpson and FFT both compute the same FT; away from the band edges they agree
    # to machine precision. (Near Nyquist, Simpson's alternating weights inject an
    # aliased ghost -- checked separately below.)
    n, lam, z = 256, 0.5e-6, 1.0
    waist = 2e-4
    dx = 16 * waist / n
    x = grids.centered_axis(n, dx)
    field = np.exp(-(x ** 2) / waist ** 2)
    x_screen = D.screen_axis(n, dx, lam, z)
    i_fft = np.abs(D.fraunhofer_field_1d(field, dx)) ** 2
    i_sim = np.abs(S.fraunhofer_simpson_1d(field, dx, lam, z, x_screen)) ** 2
    center = slice(n // 4, 3 * n // 4)
    rel = np.linalg.norm(i_sim[center] - i_fft[center]) / np.linalg.norm(i_fft[center])
    assert rel < 1e-9


def test_simpson_band_edge_ghost():
    # Simpson's weight comb aliases ~1/9 of the intensity to the band edges; the FFT
    # (uniform weights) does not. This is the documented Simpson-vs-FFT distinction.
    n, lam, z = 256, 0.5e-6, 1.0
    waist = 2e-4
    dx = 16 * waist / n
    x = grids.centered_axis(n, dx)
    field = np.exp(-(x ** 2) / waist ** 2)
    x_screen = D.screen_axis(n, dx, lam, z)
    i_fft = np.abs(D.fraunhofer_field_1d(field, dx)) ** 2
    i_sim = np.abs(S.fraunhofer_simpson_1d(field, dx, lam, z, x_screen)) ** 2
    edge = np.abs(np.arange(n) - n // 2) > 3 * n // 8
    assert np.isclose(i_sim[edge].sum() / i_sim.sum(), 0.1, atol=1e-3)
    assert i_fft[edge].sum() / i_fft.sum() < 1e-12


def test_simpson_2d_matches_fft_in_central_region():
    n, lam, z = 64, 0.5e-6, 1.0
    waist = 2e-4
    dx = 12 * waist / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.gaussian(X, Y, waist)
    axis = D.screen_axis(n, dx, lam, z)
    i_fft = np.abs(D.fraunhofer_field_2d(field, dx)) ** 2
    i_sim = np.abs(S.fraunhofer_simpson_2d(field, dx, lam, z, axis, axis)) ** 2
    c = slice(n // 4, 3 * n // 4)
    rel = np.linalg.norm(i_sim[c, c] - i_fft[c, c]) / np.linalg.norm(i_fft[c, c])
    assert rel < 1e-9
