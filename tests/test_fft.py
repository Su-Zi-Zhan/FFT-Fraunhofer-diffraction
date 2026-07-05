"""Tests for the hand-written FFT against a direct DFT oracle and FFT identities."""

import numpy as np
import pytest

from src import fft as F


def direct_dft(x):
    """O(N^2) DFT straight from the definition, used as ground truth."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[-1]
    k = np.arange(n).reshape(n, 1)
    kernel = np.exp(-2j * np.pi * k * np.arange(n) / n)
    return x @ kernel.T


@pytest.mark.parametrize("n", [2, 4, 8, 16, 64, 256])
def test_fft_matches_direct_dft(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert np.allclose(F.fft(x), direct_dft(x), atol=1e-10)


def test_fft_matches_numpy_reference():
    # numpy.fft is allowed only here, as an independent oracle.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512) + 1j * rng.standard_normal(512)
    assert np.allclose(F.fft(x), np.fft.fft(x), atol=1e-10)


def test_ifft_inverts_fft():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    assert np.allclose(F.ifft(F.fft(x)), x, atol=1e-12)


def test_fft_is_linear():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    y = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    a, b = 1.7, -0.4j
    assert np.allclose(F.fft(a * x + b * y), a * F.fft(x) + b * F.fft(y), atol=1e-10)


def test_parseval_theorem():
    # sum|x|^2 = (1/N) sum|X|^2 for the unnormalized forward transform.
    rng = np.random.default_rng(3)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    lhs = np.sum(np.abs(x) ** 2)
    rhs = np.sum(np.abs(F.fft(x)) ** 2) / x.size
    assert np.isclose(lhs, rhs, rtol=1e-10)


def test_shift_theorem():
    # A circular shift in time multiplies the spectrum by a linear phase.
    rng = np.random.default_rng(4)
    n = 128
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    s = 5
    shifted = np.roll(x, s)
    k = np.arange(n)
    expected = F.fft(x) * np.exp(-2j * np.pi * k * s / n)
    assert np.allclose(F.fft(shifted), expected, atol=1e-10)


def test_non_power_of_two_raises():
    with pytest.raises(ValueError):
        F.fft(np.ones(6))


def direct_dft2(x):
    """2D DFT by applying the 1D oracle along both axes."""
    return direct_dft(direct_dft(x).T).T


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_fft2_matches_direct_dft2(n):
    rng = np.random.default_rng(100 + n)
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    assert np.allclose(F.fft2(x), direct_dft2(x), atol=1e-10)


def test_ifft2_inverts_fft2():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((32, 64)) + 1j * rng.standard_normal((32, 64))
    assert np.allclose(F.ifft2(F.fft2(x)), x, atol=1e-12)


def test_fftshift_roundtrip_and_center():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((8, 16))
    assert np.allclose(F.ifftshift(F.fftshift(x)), x)
    # For even length the DC sample lands exactly at index N//2.
    v = np.zeros(8)
    v[0] = 1.0
    assert F.fftshift(v)[4] == 1.0


def test_fftfreq_matches_numpy():
    for n in (8, 9, 16, 32):
        assert np.allclose(F.fftfreq(n, d=0.5), np.fft.fftfreq(n, d=0.5))
