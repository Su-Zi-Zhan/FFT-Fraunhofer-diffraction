"""Spatial sampling grids for the aperture plane."""

import numpy as np

__all__ = ["next_power_of_two", "centered_axis", "square_grid"]


def next_power_of_two(n):
    """Smallest power of two >= n (integer-exact, no float rounding issues)."""
    n = int(np.ceil(n))
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def centered_axis(n, delta):
    """Length-n coordinate axis with spacing delta, origin at index n//2.

    Sample positions are (m - n//2) * delta for m = 0..n-1.
    """
    return (np.arange(n) - n // 2) * delta


def square_grid(n, delta):
    """Return (X, Y) meshgrid for an n x n aperture-plane grid of spacing delta."""
    axis = centered_axis(n, delta)
    return np.meshgrid(axis, axis)
