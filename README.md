# FFT-Fraunhofer-Diffraction

FFT-Based Fraunhofer Diffraction: Sampling Design and Rayleigh Recovery

A numerical computation project for the course *Numerical Approximation II*, exploring FFT-based Fraunhofer diffraction simulation with rigorous sampling analysis and Rayleigh resolution recovery.

## Overview

This project implements a full pipeline for simulating Fraunhofer diffraction using the Fast Fourier Transform (FFT). It focuses on:

- **Physical modeling**: From Helmholtz equation → angular spectrum → Fresnel → Fraunhofer approximation
- **Sampling theory**: Designing grid parameters $(N, \Delta x, L)$ based on 2D sampling theorem
- **Error analysis**: Quantifying aliasing, truncation, and Fraunhofer approximation errors
- **Rayleigh criterion**: Numerically recovering the $1.22\lambda/D$ resolution limit from computed Airy patterns

## Repository Structure

```
├── src/                  # Core algorithm implementations
│   ├── fft.py           # 2D FFT diffraction solver
│   ├── apertures.py     # Aperture definitions (circle, rectangle, slit, etc.)
│   ├── analytic.py      # Analytic diffraction patterns for verification
│   ├── angular_spectrum.py  # Angular Spectrum Method (ASM)
│   ├── simpson_diffraction.py  # Simpson's rule based integration
│   ├── grids.py         # Grid utilities
│   ├── grid_design.py   # Sampling-aware grid design
│   ├── quadrature.py    # Numerical quadrature helpers
│   └── metrics.py       # Error metrics and analysis
├── experiments/         # Numerical experiments (7 experiments)
├── tests/               # Pytest unit tests
├── figures/             # Generated figures (PDF vector format)
├── reports/             # Report (LaTeX/PDF) and slides
└── .gitignore
```

## Experiments

| # | Experiment | Focus |
|---|-----------|-------|
| 1 | Basic diffraction patterns | Circle, rectangle, single/double slit |
| 2 | Convergence analysis | Grid resolution vs. accuracy |
| 3 | Sampling tightness | Validating grid design rules |
| 4 | Fraunhofer vs. Fresnel | Approximation regime comparison |
| 5 | Computational efficiency | FFT vs. direct integration |
| 6 | Rayleigh resolution | Recovering $1.22\lambda/D$ |
| 7 | Grating & imaging | Duty cycle, slit count, telescope simulation |

## Requirements

- Python 3.12+
- NumPy, SciPy
- Matplotlib
- pytest (for testing)

## Usage

```bash
# Run all experiments
python experiments/run_all.py

# Run a single experiment
python experiments/experiment1.py

# Run tests
pytest tests/
```

## Author

**Ruijie Li** (24300720157) — *Numerical Approximation II, Fall 2025*
