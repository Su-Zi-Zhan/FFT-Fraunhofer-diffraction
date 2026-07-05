"""Hand-written composite Simpson quadrature on uniform grids.

The 1/3 rule needs an even number of intervals; to support any n >= 2 intervals
(including the N = 2^k grids used by the FFT, which give an odd interval count),
an odd interval count is handled by a Simpson 3/8 panel on the last three intervals
plus the 1/3 rule on the rest. Both rules are exact for cubics, so the combination
keeps the O(h^4) order. scipy.integrate is never used.
"""

import numpy as np

__all__ = ["simpson_weights", "composite_simpson"]


def simpson_weights(n_intervals, h):
    """Composite Simpson quadrature weights (including the step h) for n+1 nodes."""
    n = n_intervals
    if n < 2:
        raise ValueError("composite Simpson needs at least 2 intervals")
    w = np.zeros(n + 1)

    if n % 2 == 0:
        # Pure 1/3 rule: endpoints 1, interior alternating 4 and 2.
        w[0] = w[n] = h / 3
        w[1:n:2] = 4 * h / 3
        w[2:n:2] = 2 * h / 3
        return w

    # Odd interval count: 1/3 rule on [0, n-3], 3/8 rule on the last 3 intervals.
    m = n - 3
    if m >= 2:
        w[0] += h / 3
        w[m] += h / 3
        w[1:m:2] += 4 * h / 3
        w[2:m:2] += 2 * h / 3
    w[n - 3] += 3 * h / 8
    w[n - 2] += 9 * h / 8
    w[n - 1] += 9 * h / 8
    w[n] += 3 * h / 8
    return w


def composite_simpson(values, h):
    """Integrate uniformly sampled values (real or complex) by composite Simpson."""
    values = np.asarray(values)
    w = simpson_weights(values.shape[-1] - 1, h)
    return np.dot(w, values)
