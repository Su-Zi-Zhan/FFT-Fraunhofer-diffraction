"""Experiment 2: convergence order of the FFT diffraction solver.

The window L is fixed and the spacing Delta x = L/N is refined for N = 2^k. The
relative L2 error against the analytic pattern is measured over a fixed screen region.
A step aperture (rectangle, not band-limited) is expected to converge algebraically
(~1/N); a Gaussian aperture (band-limited) is expected to converge spectrally. Showing
both on one log-log plot contrasts algebraic and spectral convergence.
"""

import numpy as np

import _common as C
from _common import plt
from src import grids, apertures, analytic, metrics
from src import fft_diffraction as D

LAM = C.WAVELENGTH
Z = C.DISTANCE
L = 4e-3                     # fixed window
A_RECT = L / 8               # square aperture side (edges land on grid nodes)
W_GAUSS = L / 64             # narrow waist so the spectral cliff sits inside the N range
X_CMP = 3 * (LAM * Z / A_RECT)   # fixed comparison half-width (a few rect lobes)
EXPONENTS = range(6, 13)     # N = 64 .. 4096


def _region_mask(axis):
    """Boolean mask selecting the fixed comparison square |X|,|Y| <= X_CMP."""
    Xs, Ys = np.meshgrid(axis, axis)
    return (np.abs(Xs) <= X_CMP) & (np.abs(Ys) <= X_CMP)


def _rect_error(n):
    """Relative L2 error of the FFT rectangle pattern vs analytic at grid size n."""
    dx = L / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.rectangle(X, Y, A_RECT, A_RECT)
    axis, image = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    Xs, Ys = np.meshgrid(axis, axis)
    reference = analytic.rectangle_intensity(Xs, Ys, A_RECT, A_RECT, LAM, Z)
    mask = _region_mask(axis)
    return metrics.relative_l2(metrics.normalize_peak(image[mask]),
                               metrics.normalize_peak(reference[mask]))


def _gauss_error(n):
    """Relative L2 error of the FFT Gaussian pattern vs its analytic FT at grid size n."""
    dx = L / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.gaussian(X, Y, W_GAUSS)
    axis, image = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    Xs, Ys = np.meshgrid(axis, axis)
    # FT of a Gaussian is a Gaussian; intensity ~ exp(-2 (pi w/(lambda z))^2 (X^2+Y^2)).
    reference = np.exp(-2 * (np.pi * W_GAUSS / (LAM * Z)) ** 2 * (Xs ** 2 + Ys ** 2))
    mask = _region_mask(axis)
    return metrics.relative_l2(metrics.normalize_peak(image[mask]),
                               metrics.normalize_peak(reference[mask]))


def convergence_curve(error_fn):
    """Evaluate an error function over all grid sizes N = 2^k."""
    sizes = np.array([2 ** k for k in EXPONENTS])
    errors = np.array([error_fn(n) for n in sizes])
    return sizes, errors


def _fit_slope(sizes, errors):
    """Log-log slope of error vs N (the algebraic convergence order)."""
    return np.polyfit(np.log(sizes), np.log(errors), 1)[0]


def plot_convergence(sizes, rect_err, gauss_err, rect_slope, fname):
    """Log-log error-vs-N plot contrasting algebraic and spectral convergence."""
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.loglog(sizes, rect_err, "o-", color="C3", label=f"rectangle (step), slope={rect_slope:.2f}")
    ax.loglog(sizes, np.maximum(gauss_err, 1e-16), "s-", color="C0", label="Gaussian (band-limited)")
    # 1/N guide line anchored to the first rectangle point.
    guide = rect_err[0] * sizes[0] / sizes
    ax.loglog(sizes, guide, "k--", lw=1, alpha=0.6, label="reference slope -1 (1/N)")
    ax.axhline(1e-15, color="gray", ls=":", lw=1, label="machine precision")
    ax.set(xlabel="N", ylabel="relative L2 error", title="Convergence: algebraic vs spectral")
    ax.legend()
    return C.save_figure(fig, fname)


def main():
    sizes, rect_err = convergence_curve(_rect_error)
    _, gauss_err = convergence_curve(_gauss_error)
    rect_slope = _fit_slope(sizes, rect_err)
    plot_convergence(sizes, rect_err, gauss_err, rect_slope, "experiment2_convergence.pdf")

    print(f"\n{'N':>6}{'rect L2':>14}{'gauss L2':>14}")
    for n, re, ge in zip(sizes, rect_err, gauss_err):
        print(f"{n:>6}{re:>14.3e}{ge:>14.3e}")
    print(f"\nrectangle fitted slope = {rect_slope:.3f}  (expected -1, algebraic)")
    print(f"Gaussian floor at N>=256 = {gauss_err[2:].min():.2e}  (machine precision, spectral)")


if __name__ == "__main__":
    main()
