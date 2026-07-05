"""Radix-2 Cooley-Tukey FFT (1D/2D), inverse, and helpers.

Conventions:
* Transform length must be a power of two (all experiments use N = 2^k).
* Forward sign is exp(-2i*pi*k*n/N), matching numpy.fft.
* Inverse transforms carry the 1/N normalization.
"""

import numpy as np

__all__ = [
    "is_power_of_two",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftshift",
    "ifftshift",
    "fftfreq",
]


def is_power_of_two(n):
    """Return True if n is a positive power of two."""
    return n >= 1 and (n & (n - 1)) == 0


def _bit_reversal_permutation(n):
    """Index permutation that bit-reverses 0..n-1 (n a power of two)."""
    bits = n.bit_length() - 1
    indices = np.arange(n, dtype=np.int64)
    reversed_idx = np.zeros(n, dtype=np.int64)
    # Move bit b of each index to mirror position (bits-1-b).
    for b in range(bits):
        reversed_idx |= ((indices >> b) & 1) << (bits - 1 - b)
    return reversed_idx


def _fft_along_last_axis(data, inverse):
    """Radix-2 butterfly along the last axis; leading axes are batched.

    Returns the unnormalized transform (caller applies 1/N for the inverse).
    """
    n = data.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"FFT length must be a power of two, got {n}")

    # Reorder inputs so the iterative butterfly can merge adjacent blocks.
    x = data[..., _bit_reversal_permutation(n)].astype(np.complex128, copy=True)
    sign = 1.0 if inverse else -1.0
    batch_shape = x.shape[:-1]

    block = 2
    while block <= n:
        half = block // 2
        twiddle = np.exp(sign * 2j * np.pi * np.arange(half) / block)
        # Reshape into n/block blocks of length `block`, butterfly all at once.
        x = x.reshape(*batch_shape, n // block, block)
        lower = x[..., :half]
        upper = x[..., half:] * twiddle  # twiddle broadcasts over the half
        combined = np.empty_like(x)
        combined[..., :half] = lower + upper
        combined[..., half:] = lower - upper
        x = combined.reshape(*batch_shape, n)
        block *= 2

    return x


def fft(a):
    """1D FFT along the last axis."""
    return _fft_along_last_axis(np.asarray(a, dtype=np.complex128), inverse=False)


def ifft(a):
    """1D inverse FFT (with 1/N) along the last axis."""
    data = np.asarray(a, dtype=np.complex128)
    return _fft_along_last_axis(data, inverse=True) / data.shape[-1]


def fft2(a):
    """2D FFT via separable 1D transforms: rows (axis -1) then columns (axis -2)."""
    data = np.asarray(a, dtype=np.complex128)
    out = _fft_along_last_axis(data, inverse=False)
    out = _fft_along_last_axis(out.swapaxes(-1, -2), inverse=False).swapaxes(-1, -2)
    return out


def ifft2(a):
    """2D inverse FFT (with 1/(M*N))."""
    data = np.asarray(a, dtype=np.complex128)
    rows, cols = data.shape[-2], data.shape[-1]
    out = _fft_along_last_axis(data, inverse=True)
    out = _fft_along_last_axis(out.swapaxes(-1, -2), inverse=True).swapaxes(-1, -2)
    return out / (rows * cols)


def _normalize_axes(a, axes):
    """Coerce the axes argument to a tuple; None means all axes."""
    if axes is None:
        return tuple(range(a.ndim))
    if np.isscalar(axes):
        return (int(axes),)
    return tuple(int(ax) for ax in axes)


def fftshift(a, axes=None):
    """Circularly shift the zero-frequency component to the array center."""
    arr = np.asarray(a)
    use_axes = _normalize_axes(arr, axes)
    shifts = [arr.shape[ax] // 2 for ax in use_axes]
    return np.roll(arr, shifts, axis=use_axes)


def ifftshift(a, axes=None):
    """Inverse of fftshift."""
    arr = np.asarray(a)
    use_axes = _normalize_axes(arr, axes)
    shifts = [-(arr.shape[ax] // 2) for ax in use_axes]
    return np.roll(arr, shifts, axis=use_axes)


def fftfreq(n, d=1.0):
    """DFT sample frequencies (cycles per unit), matching numpy.fft.fftfreq."""
    val = 1.0 / (n * d)
    results = np.empty(n, dtype=np.int64)
    split = (n - 1) // 2 + 1
    results[:split] = np.arange(0, split)         # nonnegative frequencies
    results[split:] = np.arange(-(n // 2), 0)     # negative frequencies
    return results * val
