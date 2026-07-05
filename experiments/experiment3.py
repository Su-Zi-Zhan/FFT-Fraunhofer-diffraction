"""Experiment 3: reverse-verification of the Section 4 grid design.

The two sampling constraints are verified separately to show they are tight and
independent. For each, the design is scaled by 0.5x / 1x / 2x:
  * E1 (aliasing): Gaussian aperture, fix the window L, vary dx (Nyquist range).
    0.5x undersamples and aliases; 2x gives no gain over the design.
  * E2 (resolution): double slit, fix dx, vary L (screen sampling dX).
    0.5x cannot resolve the fringes; 2x gives no gain over the design.
Each error is the relative L2 against the analytic pattern over a fixed region.
"""

import numpy as np

import _common as C
from _common import plt
from src import grids, apertures, analytic, metrics
from src import fft_diffraction as D
from src import grid_design as GD

LAM = C.WAVELENGTH
Z = C.DISTANCE
FACTORS = [0.5, 1.0, 2.0]

# E1 scenario: Gaussian aperture, design set by the screen-range target.
W_GAUSS = 2e-4
E1_DESIGN = GD.design_grid(a=2 * W_GAUSS, wavelength=LAM, z=Z,
                           x_max_target=2.4e-3, dx_screen_target=2e-5, k_edge=3)
E1_CMP = 1.5e-3

# E2 scenario: double slit, design set by the screen-resolution (fringe) target.
A_SLIT, D_SLIT = 1e-4, 5e-4
FRINGE = LAM * Z / D_SLIT
E2_DESIGN = GD.design_grid(a=A_SLIT, wavelength=LAM, z=Z,
                           x_max_target=6e-3, dx_screen_target=FRINGE / 2, k_edge=200)
E2_CMP = 2.5e-3


def _gaussian_error(n, dx):
    """Relative L2 of the FFT Gaussian pattern vs its analytic FT over |X|<=E1_CMP."""
    X, Y = grids.square_grid(n, dx)
    field = apertures.gaussian(X, Y, W_GAUSS)
    axis, image = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    Xs, Ys = np.meshgrid(axis, axis)
    reference = np.exp(-2 * (np.pi * W_GAUSS / (LAM * Z)) ** 2 * (Xs ** 2 + Ys ** 2))
    mask = (np.abs(Xs) <= E1_CMP) & (np.abs(Ys) <= E1_CMP)
    return axis, image, metrics.relative_l2(metrics.normalize_peak(image[mask]),
                                            metrics.normalize_peak(reference[mask]))


def _double_slit_error(n, dx):
    """Relative L2 of the FFT double-slit pattern vs analytic over |X|<=E2_CMP."""
    x = grids.centered_axis(n, dx)
    field = apertures.double_slit(x, A_SLIT, D_SLIT)
    axis, intensity = D.fraunhofer_pattern_1d(field, dx, LAM, Z)
    reference = analytic.double_slit_intensity(axis, A_SLIT, D_SLIT, LAM, Z)
    mask = np.abs(axis) <= E2_CMP
    return axis, intensity, metrics.relative_l2(metrics.normalize_peak(intensity[mask]),
                                                metrics.normalize_peak(reference[mask]))


def e1_sweep():
    """E1: fix L = L*, vary dx = L*/N for N = factor * N*."""
    errors, cuts = [], []
    for f in FACTORS:
        n = int(E1_DESIGN["N"] * f)
        dx = E1_DESIGN["L"] / n
        axis, image, err = _gaussian_error(n, dx)
        errors.append(err)
        cuts.append((axis, image[n // 2], LAM * Z / (2 * dx)))  # cut + achieved X_max
    return errors, cuts


def e2_sweep():
    """E2: fix dx = dx*, vary L = N*dx* for N = factor * N*."""
    errors, cuts = [], []
    for f in FACTORS:
        n = int(E2_DESIGN["N"] * f)
        dx = E2_DESIGN["dx"]
        axis, intensity, err = _double_slit_error(n, dx)
        errors.append(err)
        cuts.append((axis, intensity, LAM * Z / (n * dx)))      # cut + achieved dX
    return errors, cuts


def plot_tightness(e1_errors, e2_errors, fname):
    """Two panels: relative L2 vs design scale factor for E1 and E2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, errs, title in [(ax1, e1_errors, "E1 aliasing (vary dx, fix L)"),
                            (ax2, e2_errors, "E2 resolution (vary L, fix dx)")]:
        ax.semilogy(FACTORS, errs, "o-", color="C3", ms=9)
        ax.axvline(1.0, color="gray", ls="--", lw=1)
        ax.annotate("design 1x", (1.0, errs[1]), textcoords="offset points",
                    xytext=(8, 8), color="gray")
        ax.set(xlabel="scale factor of N", ylabel="relative L2 error", title=title,
               xticks=FACTORS)
    return C.save_figure(fig, fname)


def plot_patterns(e1_cuts, e2_cuts, fname):
    """Representative cuts showing aliasing (E1) and fringe under-resolution (E2)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for (axis, cut, xmax), f in zip(e1_cuts, FACTORS):
        m = np.abs(axis) <= E1_CMP
        ax1.semilogy(axis[m] * 1e3, metrics.normalize_peak(cut[m]) + 1e-16,
                     label=f"{f}x (Xmax={xmax*1e3:.1f} mm)")
    ax1.set(xlabel="X (mm)", ylabel="normalized intensity", title="E1: Gaussian cut (log)",
            ylim=(1e-12, 2))
    ax1.legend(fontsize=8)
    for (axis, cut, dX), f in zip(e2_cuts, FACTORS):
        m = np.abs(axis) <= E2_CMP
        ax2.plot(axis[m] * 1e3, metrics.normalize_peak(cut[m]), label=f"{f}x (dX={dX*1e6:.0f} um)")
    ax2.set(xlabel="X (mm)", ylabel="normalized intensity", title="E2: double-slit fringes")
    ax2.legend(fontsize=8)
    return C.save_figure(fig, fname)


def main():
    e1_errors, e1_cuts = e1_sweep()
    e2_errors, e2_cuts = e2_sweep()
    plot_tightness(e1_errors, e2_errors, "experiment3_tightness.pdf")
    plot_patterns(e1_cuts, e2_cuts, "experiment3_patterns.pdf")

    print(f"E1 design: N*={E1_DESIGN['N']}, dx*={E1_DESIGN['dx']:.3e}, L*={E1_DESIGN['L']:.3e}")
    print(f"E2 design: N*={E2_DESIGN['N']}, dx*={E2_DESIGN['dx']:.3e}, L*={E2_DESIGN['L']:.3e}")
    print(f"\n{'factor':>8}{'E1 (alias) L2':>16}{'E2 (resol) L2':>16}")
    for f, e1, e2 in zip(FACTORS, e1_errors, e2_errors):
        print(f"{f:>8}{e1:>16.3e}{e2:>16.3e}")


if __name__ == "__main__":
    main()
