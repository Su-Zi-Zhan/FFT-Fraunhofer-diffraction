"""Tests for grids, aperture masks, and analytic Fraunhofer patterns."""

import numpy as np
import pytest

from src import grids, apertures, analytic


# ---- grids ----

@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (5, 8), (8, 8), (9, 16), (1000, 1024)])
def test_next_power_of_two(n, expected):
    assert grids.next_power_of_two(n) == expected


def test_centered_axis_spacing_and_origin():
    axis = grids.centered_axis(8, 0.5)
    assert axis.size == 8
    assert np.isclose(axis[4], 0.0)            # origin at index n//2
    assert np.allclose(np.diff(axis), 0.5)     # uniform spacing


# ---- apertures ----

def test_single_slit_width():
    x = np.linspace(-1, 1, 2001)
    mask = apertures.single_slit(x, 0.4)
    open_len = mask.sum() * (x[1] - x[0])
    assert np.isclose(open_len, 0.4, atol=2e-3)


def test_double_slit_is_symmetric_with_two_openings():
    # Even point count keeps slit edges strictly between samples (grid stays symmetric).
    x = np.linspace(-1, 1, 2000)
    mask = apertures.double_slit(x, 0.1, 0.6)
    assert np.allclose(mask, mask[::-1])       # even symmetry
    # Two disjoint openings -> two rising edges.
    rising = np.sum(np.diff(mask) > 0)
    assert rising == 2


def test_circle_area_matches_pi_r_squared():
    X, Y = grids.square_grid(1024, 1.0 / 1024)
    diameter = 0.5
    area = apertures.circle(X, Y, diameter).sum() * (1.0 / 1024) ** 2
    assert np.isclose(area, np.pi * (diameter / 2) ** 2, rtol=2e-3)


def test_rectangle_area():
    X, Y = grids.square_grid(1024, 1.0 / 1024)
    area = apertures.rectangle(X, Y, 0.4, 0.2).sum() * (1.0 / 1024) ** 2
    assert np.isclose(area, 0.4 * 0.2, rtol=2e-3)


def test_gaussian_peak_and_decay():
    X, Y = grids.square_grid(64, 0.1)
    g = apertures.gaussian(X, Y, 1.0)
    assert np.isclose(g.max(), 1.0)            # peak at center
    assert g[0, 0] < g[32, 32]                 # decays outward


def test_grating_has_expected_slit_count():
    X, Y = grids.square_grid(2048, 4.0 / 2048)
    g = apertures.grating(X, Y, period=0.2, duty_cycle=0.5, n_slits=5, height=2.0)
    row = g[1024]                              # central row through all slits
    rising = np.sum(np.diff(row) > 0)
    assert rising == 5


def test_annulus_blocks_center_and_struts():
    X, Y = grids.square_grid(512, 2.0 / 512)
    pupil = apertures.annulus_with_spider(X, Y, 1.0, 0.3, 4, 0.02)
    assert pupil[256, 256] == 0.0              # central obstruction
    assert pupil.max() == 1.0                  # open annulus exists


# ---- analytic patterns ----

def test_rayleigh_constant_value():
    assert np.isclose(analytic.RAYLEIGH_CONSTANT, 1.21966989, atol=1e-6)


def test_single_slit_peak_and_first_null():
    lam, z, a = 0.5e-6, 1.0, 1e-3
    assert np.isclose(analytic.single_slit_intensity(0.0, a, lam, z), 1.0)
    x1 = analytic.single_slit_first_null(a, lam, z)
    assert np.isclose(analytic.single_slit_intensity(x1, a, lam, z), 0.0, atol=1e-12)


def test_double_slit_envelope_and_fringes():
    lam, z, a, d = 0.5e-6, 1.0, 1e-3, 4e-3
    assert np.isclose(analytic.double_slit_intensity(0.0, a, d, lam, z), 1.0)
    # First interference null where cos^2 = 0: X = lambda*z/(2d).
    x_null = lam * z / (2 * d)
    assert np.isclose(analytic.double_slit_intensity(x_null, a, d, lam, z), 0.0, atol=1e-12)


def test_airy_peak_and_first_ring():
    lam, z, D = 0.5e-6, 1.0, 1e-3
    assert np.isclose(analytic.airy_intensity(0.0, D, lam, z), 1.0)
    r1 = analytic.airy_first_null(D, lam, z)
    assert np.isclose(analytic.airy_intensity(r1, D, lam, z), 0.0, atol=1e-10)
