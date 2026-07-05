"""Tests for grid design (Section 4) and the error/profile metrics."""

import numpy as np

from src import grids, analytic, metrics
from src import grid_design as GD


# ---- grid design ----

def test_design_satisfies_all_constraints():
    a, lam, z = 1e-3, 0.5e-6, 1.0
    x_max_t, dx_t = 5e-3, 2e-5
    d = GD.design_grid(a, lam, z, x_max_t, dx_t, k_edge=20)
    assert grids.next_power_of_two(d["N"]) == d["N"]        # N is a power of two
    assert d["dx"] <= d["dx_nyquist"] + 1e-18               # screen-range bound met
    assert d["dx"] <= d["dx_edge"] + 1e-18                  # aperture-edge bound met
    assert d["x_max_achieved"] >= x_max_t                   # range target reached
    assert d["dx_screen_achieved"] <= dx_t * (1 + 1e-9)     # resolution target reached


def test_tighter_resolution_needs_larger_N():
    a, lam, z, x_max_t = 1e-3, 0.5e-6, 1.0, 5e-3
    coarse = GD.design_grid(a, lam, z, x_max_t, 4e-5)["N"]
    fine = GD.design_grid(a, lam, z, x_max_t, 1e-5)["N"]
    assert fine > coarse


def test_binding_constraint_reported():
    lam, z, x_max_t, dx_t = 0.5e-6, 1.0, 5e-3, 2e-5
    # Very small aperture -> edge resolution forces a finer dx than Nyquist.
    edge_bound = GD.design_grid(1e-5, lam, z, x_max_t, dx_t)
    assert edge_bound["binding"] == "edge"


# ---- metrics ----

def test_relative_l2_zero_for_identical():
    x = np.linspace(0, 1, 50)
    assert metrics.relative_l2(x, x) == 0.0


def test_relative_l2_known_value():
    ref = np.array([3.0, 4.0])           # norm 5
    num = np.array([3.0, 0.0])           # difference norm 4
    assert np.isclose(metrics.relative_l2(num, ref), 0.8)


def test_first_local_minimum_on_cosine():
    x = np.linspace(0, 2 * np.pi, 1001)
    idx = metrics.first_local_minimum_index(np.cos(x))
    assert np.isclose(x[idx], np.pi, atol=1e-2)             # cos minimum at pi


def test_refine_minimum_subsample_accuracy():
    x = np.linspace(-2, 2, 41)
    y = (x - 0.137) ** 2                                    # vertex off the grid
    idx = int(np.argmin(y))
    assert np.isclose(metrics.refine_minimum(x, y, idx), 0.137, atol=1e-3)


def test_first_zero_radius_recovers_airy_ring():
    n, lam, z, diam = 512, 0.5e-6, 1.0, 1e-3
    dx = 16 * diam / n
    X, Y = grids.square_grid(n, dx)
    r = np.sqrt(X ** 2 + Y ** 2)
    # Build the analytic Airy pattern on the screen grid, then recover its first ring.
    x_axis = grids.centered_axis(n, dx)
    image = analytic.airy_intensity(r, diam, lam, z)
    r_num = metrics.first_zero_radius(image, x_axis)
    r_ref = analytic.airy_first_null(diam, lam, z)
    assert np.isclose(r_num, r_ref, rtol=0.02)


def test_fwhm_of_gaussian():
    # FWHM of exp(-x^2/(2 sigma^2)) is 2*sqrt(2 ln2)*sigma.
    x = np.linspace(-10, 10, 4001)
    sigma = 1.5
    y = np.exp(-(x ** 2) / (2 * sigma ** 2))
    assert np.isclose(metrics.fwhm(x, y), 2 * np.sqrt(2 * np.log(2)) * sigma, rtol=1e-3)
