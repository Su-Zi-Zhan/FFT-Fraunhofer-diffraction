"""Experiment 7 (showcase): diffraction patterns of complex apertures.

Four demonstrations of the FFT solver on richer apertures:
  1. N-slit grating, varying the slit count -> principal maxima sharpen, N-2
     subsidiary maxima appear (grating resolving power).
  2. N-slit grating, varying the duty cycle -> missing orders where the single-slit
     envelope zero lands on a grating order.
  3. Letter-shaped aperture -> its 2D Fraunhofer pattern.
  4. Telescope pupil (annulus + spider) -> PSF with diffraction spikes.
"""

import numpy as np

import _common as C
from _common import plt
from src import grids, apertures, metrics
from src import fft_diffraction as D

LAM = C.WAVELENGTH
Z = C.DISTANCE


def _grating_cut(n_slits, duty, period, n=2048, dx=2e-6):
    """Central-row far-field intensity of an N-slit grating, with the order axis."""
    X, Y = grids.square_grid(n, dx)
    field = apertures.grating(X, Y, period, duty, n_slits, height=0.6 * n * dx)
    x_screen, image = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    order_axis = x_screen / (LAM * Z / period)   # X in units of order spacing
    return order_axis, metrics.normalize_peak(image[n // 2])


def plot_grating_slit_count(fname):
    """Far-field vs slit count: more slits -> sharper orders and subsidiary maxima."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for n_slits in (2, 5, 20):
        orders, cut = _grating_cut(n_slits, duty=0.5, period=40e-6)
        m = np.abs(orders) <= 2.6
        ax.semilogy(orders[m], cut[m] + 1e-7, lw=1.2, label=f"{n_slits} slits")
    ax.set(xlabel="diffraction order  X / (lambda z / d)", ylabel="normalized intensity",
           title="Grating: principal maxima sharpen with slit count", ylim=(1e-5, 2))
    ax.legend()
    return C.save_figure(fig, fname)


def plot_grating_duty(fname):
    """Far-field vs duty cycle: envelope zeros suppress orders at multiples of 1/duty."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, duty in zip(axes, (0.5, 0.25)):
        orders, cut = _grating_cut(8, duty, period=40e-6)
        m = np.abs(orders) <= 6.5
        ax.semilogy(orders[m], cut[m] + 1e-7, color="C0", lw=1.0)
        missing = int(round(1 / duty))
        for k in range(missing, 7, missing):  # missing orders at multiples of 1/duty
            for s in (-1, 1):
                ax.axvline(s * k, color="C3", ls=":", lw=1)
        ax.set(xlabel="diffraction order", ylabel="normalized intensity",
               title=f"duty = {duty}: orders +/-{missing}, +/-{2*missing}... missing", ylim=(1e-5, 2))
    return C.save_figure(fig, fname)


def plot_letter(fname):
    """Letter-shaped aperture and its 2D Fraunhofer pattern."""
    n = 512
    mask = apertures.letter_mask(n, "F", fill_fraction=0.7)
    _, image = D.fraunhofer_pattern_2d(mask, dx=1e-5, wavelength=LAM, z=Z)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    ax1.imshow(mask, origin="lower", cmap="gray")
    ax1.set(title="aperture: letter F", xticks=[], yticks=[])
    crop = slice(n // 2 - 64, n // 2 + 64)  # zoom into the bright central region
    ax2.imshow(np.log10(metrics.normalize_peak(image)[crop, crop] + 1e-6), origin="lower", vmin=-5, vmax=0)
    ax2.set(title="Fraunhofer pattern (log)", xticks=[], yticks=[])
    return C.save_figure(fig, fname)


def plot_telescope(fname):
    """Annular pupil with a 4-vane spider and its PSF (diffraction spikes)."""
    n, dx = 1024, 8e-6
    X, Y = grids.square_grid(n, dx)
    pupil = apertures.annulus_with_spider(X, Y, outer_diameter=1e-3, inner_diameter=0.3e-3,
                                          n_struts=4, strut_width=2e-5)
    _, image = D.fraunhofer_pattern_2d(pupil, dx, LAM, Z)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    crop_pup = slice(n // 2 - 90, n // 2 + 90)
    ax1.imshow(pupil[crop_pup, crop_pup], origin="lower", cmap="gray")
    ax1.set(title="pupil: annulus + 4-vane spider", xticks=[], yticks=[])
    crop = slice(n // 2 - 80, n // 2 + 80)
    ax2.imshow(np.log10(metrics.normalize_peak(image)[crop, crop] + 1e-6), origin="lower", vmin=-5, vmax=0)
    ax2.set(title="PSF (log): Airy core + diffraction spikes", xticks=[], yticks=[])
    return C.save_figure(fig, fname)


def main():
    plot_grating_slit_count("experiment7_grating_slitcount.pdf")
    plot_grating_duty("experiment7_grating_duty.pdf")
    plot_letter("experiment7_letter.pdf")
    plot_telescope("experiment7_telescope.pdf")
    print("experiment 7 figures written to figures/")


if __name__ == "__main__":
    main()
