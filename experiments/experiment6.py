"""Experiment 6 (headline): numerically recover the Rayleigh constant 1.21967.

The FFT of a circular aperture is the Airy pattern. We locate its first dark ring on
a central cut (parabolic sub-sample refinement) and back out the dimensionless constant
c = r1 * D / (lambda z), to be compared with the analytic value J1-first-zero / pi =
1.2196699. We show c converges to this value as the grid refines and is independent of
D/lambda (scale invariance). The central cut is used rather than azimuthal averaging,
which carries a ~0.3% annular-binning bias.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, FixedFormatter

import _common as C
from _common import plt
from src import grids, apertures, analytic, metrics
from src import fft_diffraction as D

LAM = C.WAVELENGTH
Z = C.DISTANCE
EXACT = analytic.RAYLEIGH_CONSTANT


def recover_constant(n, dx, diameter):
    """Recover c = r1*D/(lambda z) from the first dark ring of the FFT Airy pattern."""
    X, Y = grids.square_grid(n, dx)
    field = apertures.circle(X, Y, diameter)
    x_screen, intensity = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    cut = intensity[n // 2]                              # central horizontal cut
    idx = metrics.first_local_minimum_index(cut)         # first minimum right of center
    r1 = abs(metrics.refine_minimum(x_screen, cut, idx))
    return r1 * diameter / (LAM * Z)


def convergence(diameter, sizes):
    """Recovered constant at each grid size N (aperture sampled at D/128)."""
    return np.array([recover_constant(n, diameter / 128, diameter) for n in sizes])


def scale_invariance(n, diameters):
    """Recovered constant for several D/lambda at fixed grid size."""
    return np.array([recover_constant(n, d / 128, d) for d in diameters])


def plot_headline(diameter, sizes, c_values, fname):
    """Airy pattern with the recovered first ring, plus the convergence of c to 1.21967."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: illustrative Airy pattern with the recovered first dark ring drawn on top.
    n_img = 1024
    dx = diameter / 128
    X, Y = grids.square_grid(n_img, dx)
    x_screen, image = D.fraunhofer_pattern_2d(apertures.circle(X, Y, diameter), dx, LAM, Z)
    r1_exact = analytic.airy_first_null(diameter, LAM, Z)
    c_img = recover_constant(n_img, dx, diameter)
    extent = np.array([x_screen[0], x_screen[-1], x_screen[0], x_screen[-1]]) / r1_exact
    ax1.imshow(np.log10(metrics.normalize_peak(image) + 1e-6), extent=extent,
               origin="lower", vmin=-6, vmax=0)
    ax1.add_patch(mpatches.Circle((0, 0), c_img / EXACT, fill=False, color="cyan", ls="--", lw=1.2))
    ax1.set(xlabel="X / r1", ylabel="Y / r1", title="Airy pattern + recovered first ring",
            xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))

    # Right: c vs N approaching the exact Rayleigh constant (N axis in actual values).
    ax2.plot(sizes, c_values, "o-", color="C3", label="recovered c")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_locator(FixedLocator(sizes))
    ax2.xaxis.set_major_formatter(FixedFormatter([str(s) for s in sizes]))
    ax2.xaxis.set_minor_locator(FixedLocator([]))
    ax2.axhline(EXACT, color="k", ls="--", lw=1, label=f"exact = {EXACT:.5f}")
    ax2.set(xlabel="N (grid samples per side)", ylabel="c = r1 D / (lambda z)",
            title="Convergence to the Rayleigh constant")
    ax2.annotate(f"N={sizes[-1]}: c={c_values[-1]:.5f}\nerr={abs(c_values[-1]-EXACT)/EXACT:.1e}",
                 (sizes[-1], c_values[-1]), textcoords="offset points", xytext=(-110, 20))
    ax2.legend()
    return C.save_figure(fig, fname)


def main():
    sizes = np.array([512, 1024, 2048, 4096, 8192])
    diameter = 0.5e-3
    c_values = convergence(diameter, sizes)
    plot_headline(diameter, sizes, c_values, "experiment6_rayleigh.pdf")

    print(f"exact Rayleigh constant = {EXACT:.7f}\n")
    print(f"{'N':>6}{'c':>12}{'rel error':>14}")
    for n, c in zip(sizes, c_values):
        print(f"{n:>6}{c:>12.6f}{abs(c - EXACT) / EXACT:>14.2e}")

    diameters = np.array([0.25e-3, 0.5e-3, 1e-3, 2e-3])
    c_scale = scale_invariance(4096, diameters)
    print(f"\nscale invariance at N=4096 (c should be constant):")
    for d, c in zip(diameters, c_scale):
        print(f"  D/lambda={d / LAM:>6.0f}  c={c:.6f}")

    best = c_values[-1]
    print(f"\nHEADLINE: recovered c = {best:.5f} vs exact {EXACT:.5f} "
          f"(relative error {abs(best - EXACT) / EXACT:.2e})")


if __name__ == "__main__":
    main()
