"""Error metrics and radial-profile tools for comparing diffraction patterns.

Includes the unified relative L2/Linf errors, peak normalization, an azimuthal
radial profile, and a sub-sample first-minimum locator used to find the first dark
ring of the Airy pattern (the Rayleigh experiment).
"""

import numpy as np

__all__ = [
    "relative_l2",
    "relative_linf",
    "normalize_peak",
    "fwhm",
    "radial_profile",
    "first_local_minimum_index",
    "refine_minimum",
    "first_zero_radius",
]


def relative_l2(numerical, reference):
    """Relative L2 error ||numerical - reference|| / ||reference||."""
    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return np.linalg.norm(numerical - reference) / np.linalg.norm(reference)


def relative_linf(numerical, reference):
    """Relative Linf error max|numerical - reference| / max|reference|."""
    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return np.max(np.abs(numerical - reference)) / np.max(np.abs(reference))


def normalize_peak(array):
    """Scale an array so its maximum is 1 (for comparing pattern shapes)."""
    array = np.asarray(array, dtype=float)
    return array / array.max()


def fwhm(x, y):
    """Full width at half maximum of a single-peaked profile, by linear interpolation."""
    peak = int(np.argmax(y))
    half = y[peak] / 2.0
    # Walk left/right from the peak to the first samples below half maximum.
    li = peak
    while li > 0 and y[li] > half:
        li -= 1
    ri = peak
    while ri < len(y) - 1 and y[ri] > half:
        ri += 1
    x_left = np.interp(half, [y[li], y[li + 1]], [x[li], x[li + 1]])
    x_right = np.interp(half, [y[ri], y[ri - 1]], [x[ri], x[ri - 1]])
    return abs(x_right - x_left)


def radial_profile(image, x_axis):
    """Azimuthally averaged radial profile about the peak pixel.

    Returns (radii, mean_intensity); radii are in the physical units of x_axis.
    """
    cy, cx = np.unravel_index(np.argmax(image), image.shape)
    yy, xx = np.indices(image.shape)
    r_pixels = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    # Mean intensity per integer-radius bin.
    total = np.bincount(r_pixels.ravel(), weights=image.ravel())
    count = np.bincount(r_pixels.ravel())
    profile = total / np.maximum(count, 1)
    spacing = x_axis[1] - x_axis[0]
    radii = np.arange(profile.size) * spacing
    return radii, profile


def first_local_minimum_index(values, start=None):
    """Index of the first local minimum at or after `start` (default: the global max)."""
    if start is None:
        start = int(np.argmax(values))
    for i in range(start + 1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] < values[i + 1]:
            return i
    raise ValueError("no local minimum found")


def refine_minimum(x, y, idx):
    """Sub-sample abscissa of a minimum near index idx via a 3-point parabolic fit."""
    if idx <= 0 or idx >= len(x) - 1:
        return x[idx]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    curvature = y0 - 2 * y1 + y2
    if curvature == 0:
        return x[idx]
    offset = 0.5 * (y0 - y2) / curvature  # vertex offset in sample units
    return x[idx] + offset * (x[idx] - x[idx - 1])


def first_zero_radius(image, x_axis):
    """Physical radius of the first dark ring (first radial minimum) of a 2D pattern."""
    radii, profile = radial_profile(image, x_axis)
    idx = first_local_minimum_index(profile)
    return refine_minimum(radii, profile, idx)
