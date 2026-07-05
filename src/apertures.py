"""Aperture transmittance masks (1D and 2D), used as inputs to the diffraction solvers.

Each function returns the amplitude transmittance on a given coordinate grid:
1.0 inside the opening and 0.0 outside (binary masks), or a smooth profile for
the Gaussian aperture. These are the input fields only; they call no FFT logic.
"""

import numpy as np

__all__ = [
    "single_slit",
    "double_slit",
    "rectangle",
    "circle",
    "gaussian",
    "grating",
    "letter_mask",
    "annulus_with_spider",
]


def single_slit(x, width):
    """1D slit: unit transmittance for |x| <= width/2."""
    return (np.abs(x) <= width / 2).astype(float)


def double_slit(x, slit_width, separation):
    """1D double slit: two slits of slit_width centered at +/- separation/2."""
    left = np.abs(x + separation / 2) <= slit_width / 2
    right = np.abs(x - separation / 2) <= slit_width / 2
    return (left | right).astype(float)


def rectangle(X, Y, width, height):
    """2D rectangular aperture of size width x height."""
    return ((np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2)).astype(float)


def circle(X, Y, diameter):
    """2D circular aperture of given diameter."""
    return (X ** 2 + Y ** 2 <= (diameter / 2) ** 2).astype(float)


def gaussian(X, Y, waist):
    """2D Gaussian aperture exp(-(X^2+Y^2)/waist^2): smooth and band-limited."""
    return np.exp(-(X ** 2 + Y ** 2) / waist ** 2)


def grating(X, Y, period, duty_cycle, n_slits, height):
    """N-slit amplitude grating periodic in x, extended in y to +/- height/2.

    Openings have width duty_cycle*period and are centered symmetrically about x=0.
    """
    slit_width = duty_cycle * period
    centers = (np.arange(n_slits) - (n_slits - 1) / 2) * period
    in_x = np.zeros(X.shape, dtype=bool)
    # Each iteration adds one slit; centers are distinct so this builds N openings.
    for c in centers:
        in_x |= np.abs(X - c) <= slit_width / 2
    in_y = np.abs(Y) <= height / 2
    return (in_x & in_y).astype(float)


def _load_font(pixel_size, font_path):
    """Load a TrueType font at the requested pixel size, falling back gracefully."""
    from PIL import ImageFont

    candidates = [font_path] if font_path else []
    candidates += [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]
    for path in candidates:
        if path:
            try:
                return ImageFont.truetype(path, pixel_size)
            except OSError:
                continue
    return ImageFont.load_default()


def letter_mask(n, text, fill_fraction=0.7, font_path=None):
    """Rasterize `text` to an n x n binary aperture (1 inside the glyph).

    PIL only renders the glyph bitmap; it is not part of the numerical algorithm.
    """
    from PIL import Image, ImageDraw

    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    font = _load_font(int(n * fill_fraction), font_path)

    # Center the glyph box inside the grid.
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((n - w) / 2 - bbox[0], (n - h) / 2 - bbox[1])
    draw.text(pos, text, fill=255, font=font)

    # Flip vertically so the glyph is upright in y-up plotting coordinates.
    return (np.flipud(np.array(img)) > 127).astype(float)


def annulus_with_spider(X, Y, outer_diameter, inner_diameter, n_struts, strut_width):
    """Telescope pupil: annulus (central obstruction) blocked by n_struts vanes.

    Each vane is a half-line from the center to the rim at an equally spaced angle.
    """
    r2 = X ** 2 + Y ** 2
    pupil = (r2 <= (outer_diameter / 2) ** 2) & (r2 >= (inner_diameter / 2) ** 2)

    vanes = np.zeros(X.shape, dtype=bool)
    # Block a thin radial bar at each strut angle (perp distance <= strut_width/2).
    for k in range(n_struts):
        angle = 2 * np.pi * k / n_struts
        along = X * np.cos(angle) + Y * np.sin(angle)
        perp = -X * np.sin(angle) + Y * np.cos(angle)
        vanes |= (np.abs(perp) <= strut_width / 2) & (along >= 0)

    return (pupil & ~vanes).astype(float)
