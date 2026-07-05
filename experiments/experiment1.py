"""Experiment 1: validate the FFT diffraction solver against analytic patterns.

Four apertures (1D single slit, 1D double slit, 2D rectangle, 2D circle) are
propagated with the FFT Fraunhofer solver and compared to their closed-form
intensities. For each we report main-lobe FWHM error, first-null position error,
and the peak-normalized relative L2 and Linf errors, plus a comparison figure.
"""

import numpy as np

import _common as C
from _common import plt
from src import grids, apertures, analytic, metrics
from src import fft_diffraction as D

LAM = C.WAVELENGTH
Z = C.DISTANCE


def _cut_metrics(x, numerical, reference, null_ref, half_window):
    """Shape metrics on a 1D cut over |x| <= half_window (both peak-normalized)."""
    mask = np.abs(x) <= half_window
    xm, num, ref = x[mask], numerical[mask], reference[mask]
    num_n, ref_n = metrics.normalize_peak(num), metrics.normalize_peak(ref)
    null_idx = metrics.first_local_minimum_index(num)
    null_num = metrics.refine_minimum(xm, num, null_idx)
    return {
        "fwhm_err": abs(metrics.fwhm(xm, num) - metrics.fwhm(xm, ref)) / metrics.fwhm(xm, ref),
        "null_num": null_num,
        "null_ref": null_ref,
        "null_err": abs(null_num - null_ref) / null_ref,
        "l2": metrics.relative_l2(num_n, ref_n),
        "linf": metrics.relative_linf(num_n, ref_n),
    }


def _plot_1d(name, fname, x, num, ref, half_window, x1):
    """Linear + log comparison of numerical vs analytic intensity over the window."""
    mask = np.abs(x) <= half_window
    xm = x[mask] / x1  # axis in units of the first-null position
    num_n = metrics.normalize_peak(num[mask])
    ref_n = metrics.normalize_peak(ref[mask])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xm, ref_n, "k-", lw=1.5, label="analytic")
    ax1.plot(xm, num_n, "r--", lw=1.2, label="FFT")
    ax1.set(xlabel="X / X_1", ylabel="normalized intensity", title=f"{name} (linear)")
    ax1.legend()
    ax2.semilogy(xm, ref_n + 1e-12, "k-", lw=1.5, label="analytic")
    ax2.semilogy(xm, num_n + 1e-12, "r--", lw=1.2, label="FFT")
    ax2.set(xlabel="X / X_1", ylabel="normalized intensity", title=f"{name} (log)", ylim=(1e-6, 2))
    ax2.legend()
    return C.save_figure(fig, fname)


def _plot_2d(name, fname, axis, image_num, image_ref, x, num_cut, ref_cut, half_window, x1):
    """2D log-intensity map plus a central-cut comparison."""
    extent = [axis[0] / x1, axis[-1] / x1, axis[0] / x1, axis[-1] / x1]
    span = half_window / x1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    disp = metrics.normalize_peak(image_num)
    ax1.imshow(np.log10(disp + 1e-6), extent=extent, origin="lower", vmin=-6, vmax=0)
    ax1.set(xlabel="X / X_1", ylabel="Y / X_1", title=f"{name}: FFT log-intensity",
            xlim=(-span, span), ylim=(-span, span))
    mask = np.abs(x) <= half_window
    xm = x[mask] / x1
    ax2.plot(xm, metrics.normalize_peak(ref_cut[mask]), "k-", lw=1.5, label="analytic")
    ax2.plot(xm, metrics.normalize_peak(num_cut[mask]), "r--", lw=1.2, label="FFT")
    ax2.set(xlabel="X / X_1", ylabel="normalized intensity", title=f"{name}: central cut")
    ax2.legend()
    return C.save_figure(fig, fname)


def run_single_slit():
    """1D single slit -> sinc^2."""
    n, a = 2048, 2e-4
    dx = 16 * a / n
    x = grids.centered_axis(n, dx)
    field = apertures.single_slit(x, a)
    x_screen, intensity = D.fraunhofer_pattern_1d(field, dx, LAM, Z)
    reference = analytic.single_slit_intensity(x_screen, a, LAM, Z)
    x1 = analytic.single_slit_first_null(a, LAM, Z)
    m = _cut_metrics(x_screen, intensity, reference, x1, 10 * x1)
    _plot_1d("single slit", "experiment1_single_slit.pdf", x_screen, intensity, reference, 10 * x1, x1)
    return "single slit", m


def run_double_slit():
    """1D double slit -> sinc^2 * cos^2."""
    n, a, d = 8192, 1e-4, 5e-4
    dx = a / 100  # ~100 samples per slit so edge quantization stays ~1%
    x = grids.centered_axis(n, dx)
    field = apertures.double_slit(x, a, d)
    x_screen, intensity = D.fraunhofer_pattern_1d(field, dx, LAM, Z)
    reference = analytic.double_slit_intensity(x_screen, a, d, LAM, Z)
    x1 = LAM * Z / (2 * d)  # first interference null
    m = _cut_metrics(x_screen, intensity, reference, x1, 6 * (LAM * Z / a))
    _plot_1d("double slit", "experiment1_double_slit.pdf", x_screen, intensity, reference,
             6 * (LAM * Z / a), LAM * Z / a)
    return "double slit", m


def run_rectangle():
    """2D rectangular aperture -> sinc^2 * sinc^2."""
    n, a, b = 2048, 2e-4, 1e-4
    dx = 32 * a / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.rectangle(X, Y, a, b)
    axis, image = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    Xs, Ys = np.meshgrid(axis, axis)
    image_ref = analytic.rectangle_intensity(Xs, Ys, a, b, LAM, Z)
    x1 = LAM * Z / a  # first null along x
    cut_num, cut_ref = image[n // 2], image_ref[n // 2]
    m = _cut_metrics(axis, cut_num, cut_ref, x1, 8 * x1)
    _plot_2d("rectangle", "experiment1_rectangle.pdf", axis, image, image_ref,
             axis, cut_num, cut_ref, 8 * x1, x1)
    return "rectangle", m


def run_circle():
    """2D circular aperture -> Airy pattern."""
    n, diam = 2048, 2e-4
    dx = 32 * diam / n
    X, Y = grids.square_grid(n, dx)
    field = apertures.circle(X, Y, diam)
    axis, image = D.fraunhofer_pattern_2d(field, dx, LAM, Z)
    Xs, Ys = np.meshgrid(axis, axis)
    image_ref = analytic.airy_intensity(np.sqrt(Xs ** 2 + Ys ** 2), diam, LAM, Z)
    x1 = analytic.airy_first_null(diam, LAM, Z)
    cut_num, cut_ref = image[n // 2], image_ref[n // 2]
    m = _cut_metrics(axis, cut_num, cut_ref, x1, 6 * x1)
    _plot_2d("circle (Airy)", "experiment1_circle.pdf", axis, image, image_ref,
             axis, cut_num, cut_ref, 6 * x1, x1)
    return "circle", m


def main():
    runs = [run_single_slit(), run_double_slit(), run_rectangle(), run_circle()]
    print(f"\n{'aperture':<14}{'FWHM err':>12}{'null err':>12}{'rel L2':>12}{'rel Linf':>12}")
    for name, m in runs:
        print(f"{name:<14}{m['fwhm_err']:>12.2e}{m['null_err']:>12.2e}{m['l2']:>12.2e}{m['linf']:>12.2e}")


if __name__ == "__main__":
    main()
