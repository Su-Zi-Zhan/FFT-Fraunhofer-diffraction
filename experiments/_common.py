"""Shared setup for experiment scripts: import path, matplotlib config, figure saving.

Importing this module makes `import src...` work when running an experiment directly,
configures a headless ASCII-only plotting style (GBK-console safe), and exposes a
single figure-saving helper that writes vector PDFs into figures/.
"""

import os
import sys

import matplotlib

# Make the project root importable when running `python experiments/experimentX.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

matplotlib.use("Agg")  # file output only, no GUI
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

FIGURES_DIR = os.path.join(_ROOT, "figures")

# Default optical parameters shared across experiments (SI units).
WAVELENGTH = 0.5e-6  # 500 nm
DISTANCE = 1.0       # 1 m propagation to the screen

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "image.cmap": "inferno",
})


def save_figure(fig, filename):
    """Save a figure as a vector PDF under figures/ and print its path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
    return path
