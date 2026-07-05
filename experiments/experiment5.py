"""Experiment 5: runtime efficiency of the FFT vs direct Simpson integration.

Both methods compute the full N x N Fraunhofer pattern of a Gaussian aperture (where
both reach machine precision, so it is a same-accuracy comparison). The FFT costs
O(N^2 log N); direct Simpson with M = N screen points costs O(N^2 M^2) = O(N^4).
Runtime is measured against N (log-log) and slopes are fitted; the FFT memory
footprint (N^2 complex128 array) is plotted alongside.
"""

import time

import numpy as np

import _common as C
from _common import plt
from src import grids, apertures, metrics
from src import fft_diffraction as D
from src import simpson_diffraction as S

LAM = C.WAVELENGTH
Z = C.DISTANCE
WAIST = 2e-4


def _gaussian(n, dx):
    X, Y = grids.square_grid(n, dx)
    return apertures.gaussian(X, Y, WAIST)


def _best_time(call, repeats):
    """Minimum wall-clock time of `call` over a few repeats (less noisy)."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - t0)
    return best


def time_fft(n):
    """Time the full-pattern FFT Fraunhofer transform at grid size n."""
    field = _gaussian(n, 12 * WAIST / n)
    repeats = 3 if n <= 1024 else 1
    return _best_time(lambda: D.fraunhofer_field_2d(field, 12 * WAIST / n), repeats)


def time_simpson(n):
    """Time direct Simpson integration onto an n x n screen (O(N^4))."""
    dx = 12 * WAIST / n
    field = _gaussian(n, dx)
    axis = D.screen_axis(n, dx, LAM, Z)
    return _best_time(lambda: S.fraunhofer_simpson_2d(field, dx, LAM, Z, axis, axis), 1)


def accuracy_at(n):
    """Central-region relative L2 between FFT and Simpson at grid size n."""
    dx = 12 * WAIST / n
    field = _gaussian(n, dx)
    axis = D.screen_axis(n, dx, LAM, Z)
    i_fft = np.abs(D.fraunhofer_field_2d(field, dx)) ** 2
    i_sim = np.abs(S.fraunhofer_simpson_2d(field, dx, LAM, Z, axis, axis)) ** 2
    c = slice(n // 4, 3 * n // 4)
    return metrics.relative_l2(i_sim[c, c], i_fft[c, c])


def _slope(sizes, times, tail):
    """Log-log slope of runtime vs N over the asymptotic tail (largest `tail` points)."""
    return np.polyfit(np.log(sizes[-tail:]), np.log(times[-tail:]), 1)[0]


def plot_efficiency(fft_n, fft_t, simp_n, simp_t, fname):
    """Log-log runtime vs N with fitted asymptotic slopes, plus the FFT memory footprint."""
    fft_slope = _slope(fft_n, fft_t, tail=4)
    simp_slope = _slope(simp_n, simp_t, tail=4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(simp_n, simp_t, "s-", color="C3", label=f"Simpson direct, slope={simp_slope:.2f}")
    ax1.loglog(fft_n, fft_t, "o-", color="C0", label=f"FFT, slope={fft_slope:.2f}")
    ax1.set(xlabel="N", ylabel="runtime (s)", title="Runtime: FFT vs direct Simpson")
    ax1.legend()

    mem_mb = (fft_n.astype(float) ** 2 * 16) / 1024 ** 2  # one complex128 array
    ax2.loglog(fft_n, mem_mb, "o-", color="C0")
    ax2.set(xlabel="N", ylabel="array memory (MB)", title="FFT memory footprint (N^2 complex128)")
    return C.save_figure(fig, fname), fft_slope, simp_slope


def main():
    fft_sizes = np.array([2 ** k for k in range(6, 13)])
    simp_sizes = np.array([16, 32, 64, 128, 256])
    fft_times = np.array([time_fft(n) for n in fft_sizes])
    simp_times = np.array([time_simpson(n) for n in simp_sizes])

    _, fft_slope, simp_slope = plot_efficiency(fft_sizes, fft_times, simp_sizes, simp_times,
                                               "experiment5_efficiency.pdf")

    print(f"{'N':>6}{'FFT (s)':>14}{'Simpson (s)':>16}")
    simp_map = dict(zip(simp_sizes.tolist(), simp_times.tolist()))
    for n, ft in zip(fft_sizes, fft_times):
        st = f"{simp_map[n]:.3e}" if n in simp_map else "-"
        print(f"{n:>6}{ft:>14.3e}{st:>16}")
    print(f"\nFFT slope = {fft_slope:.2f} (O(N^2 log N)), Simpson slope = {simp_slope:.2f} (O(N^4))")
    for n in (32, 64, 128):
        print(f"FFT/Simpson agreement at N={n}: rel L2 = {accuracy_at(n):.2e}")


if __name__ == "__main__":
    main()
