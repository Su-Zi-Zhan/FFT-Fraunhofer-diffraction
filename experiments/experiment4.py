"""Experiment 4: the Fraunhofer approximation's failure boundary (model error E3).

The Fresnel number F = a^2/(lambda z) is swept by varying z. The Fraunhofer pattern
is compared to the single-FFT Fresnel pattern on the same screen grid; the two differ
by exactly the aperture quadratic phase that Fraunhofer drops, so this relative L2
error isolates E3. The error should be tiny for F << 1 and approach 1 for F >> 1, with
the transition near F = 1. The angular spectrum method is used as an independent
near-field cross-check that the Fresnel reference is trustworthy.
"""

import numpy as np

import _common as C
from _common import plt
from src import grids, apertures, metrics
from src import fft_diffraction as D
from src import angular_spectrum as ASM

LAM = C.WAVELENGTH
A_RADIUS = 2e-4              # aperture radius; Fresnel number F = a^2/(lambda z)
N = 512
DX = A_RADIUS / 40          # fine enough to sample the chirp up to F = 10
CENTER = slice(N // 4, 3 * N // 4)


def _aperture():
    """Circular aperture of radius A_RADIUS on the standard grid."""
    X, Y = grids.square_grid(N, DX)
    return apertures.circle(X, Y, 2 * A_RADIUS)


def distance_for_fresnel_number(f):
    """Propagation distance giving Fresnel number f."""
    return A_RADIUS ** 2 / (LAM * f)


def e3_error(field, z):
    """Relative L2 between Fraunhofer and Fresnel intensities over the central region."""
    i_fraun = np.abs(D.fraunhofer_field_2d(field, DX)) ** 2
    i_fresnel = np.abs(D.fresnel_field_2d(field, DX, LAM, z)) ** 2
    i_fraun /= i_fraun.max()
    i_fresnel /= i_fresnel.max()
    return metrics.relative_l2(i_fraun[CENTER, CENTER], i_fresnel[CENTER, CENTER])


def e3_sweep(fresnel_numbers):
    """Evaluate E3 across a range of Fresnel numbers."""
    field = _aperture()
    return np.array([e3_error(field, distance_for_fresnel_number(f)) for f in fresnel_numbers])


def asm_crosscheck(field, f):
    """Near-field check: relative L2 of ASM vs Fresnel-FFT central profiles.

    At F ~ a few the screen grid and the aperture grid nearly coincide, so the two
    propagators can be compared by interpolating the ASM cut onto the screen axis.
    """
    z = distance_for_fresnel_number(f)
    screen = D.screen_axis(N, DX, LAM, z)
    aperture_axis = grids.centered_axis(N, DX)
    i_fresnel = metrics.normalize_peak(np.abs(D.fresnel_field_2d(field, DX, LAM, z)[N // 2]) ** 2)
    i_asm = metrics.normalize_peak(np.abs(ASM.propagate(field, DX, LAM, z)[N // 2]) ** 2)
    span = 0.8 * min(screen[-1], aperture_axis[-1])
    mask = np.abs(screen) <= span
    i_asm_on_screen = np.interp(screen[mask], aperture_axis, i_asm)
    return metrics.relative_l2(i_fresnel[mask], i_asm_on_screen)


def plot_e3_curve(fresnel_numbers, errors, fname):
    """Log-log E3-vs-F curve with the F=1 boundary and a slope-2 small-F guide."""
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.loglog(fresnel_numbers, errors, "o-", color="C3", label="E3 = ||Fraunhofer - Fresnel||")
    guide = errors[0] * (fresnel_numbers / fresnel_numbers[0]) ** 2
    ax.loglog(fresnel_numbers, guide, "k--", lw=1, alpha=0.6, label="slope 2 (E3 ~ F^2)")
    ax.axvline(1.0, color="C0", ls=":", lw=1.5, label="F = 1 (boundary)")
    ax.set(xlabel="Fresnel number F = a^2/(lambda z)", ylabel="relative L2 error (E3)",
           title="Fraunhofer approximation failure vs Fresnel number")
    ax.legend()
    return C.save_figure(fig, fname)


def plot_pattern_comparison(field, fname):
    """Central cuts: Fraunhofer vs Fresnel at a far-field and a near-field F."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, f in zip(axes, [0.01, 3.0]):
        z = distance_for_fresnel_number(f)
        screen = D.screen_axis(N, DX, LAM, z)
        i_fraun = metrics.normalize_peak(np.abs(D.fraunhofer_field_2d(field, DX)[N // 2]) ** 2)
        i_fresnel = metrics.normalize_peak(np.abs(D.fresnel_field_2d(field, DX, LAM, z)[N // 2]) ** 2)
        m = np.abs(screen) <= 0.5 * screen[-1]
        ax.plot(screen[m] / screen[-1], i_fresnel[m], "k-", lw=1.5, label="Fresnel (reference)")
        ax.plot(screen[m] / screen[-1], i_fraun[m], "r--", lw=1.2, label="Fraunhofer")
        ax.set(xlabel="X / X_max", ylabel="normalized intensity", title=f"F = {f}")
        ax.legend()
    return C.save_figure(fig, fname)


def main():
    field = _aperture()
    fresnel_numbers = np.logspace(-3, 1, 13)
    errors = e3_sweep(fresnel_numbers)
    plot_e3_curve(fresnel_numbers, errors, "experiment4_e3_vs_fresnel.pdf")
    plot_pattern_comparison(field, "experiment4_patterns.pdf")

    small = fresnel_numbers < 0.1
    slope = np.polyfit(np.log(fresnel_numbers[small]), np.log(errors[small]), 1)[0]
    print(f"{'F':>10}{'E3':>14}")
    for f, e in zip(fresnel_numbers, errors):
        print(f"{f:>10.3e}{e:>14.3e}")
    print(f"\nsmall-F slope = {slope:.2f} (E3 ~ F^2)")
    print(f"E3 at F=1 (interp) ~ {np.interp(1.0, fresnel_numbers, errors):.3f}")
    print(f"ASM vs Fresnel-FFT cross-check at F=3: rel L2 = {asm_crosscheck(field, 3.0):.3e}")


if __name__ == "__main__":
    main()
