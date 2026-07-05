"""One-click reproduction: run every experiment and regenerate all figures in figures/.

Usage:
    python experiments/run_all.py
"""

import time

import _common  # noqa: F401  sets the import path and matplotlib backend

import experiment1
import experiment2
import experiment3
import experiment4
import experiment5
import experiment6
import experiment7

EXPERIMENTS = [
    ("experiment1 (analytic validation)", experiment1),
    ("experiment2 (convergence order)", experiment2),
    ("experiment3 (grid-design tightness)", experiment3),
    ("experiment4 (Fraunhofer failure / E3)", experiment4),
    ("experiment5 (efficiency)", experiment5),
    ("experiment6 (Rayleigh headline)", experiment6),
    ("experiment7 (complex apertures)", experiment7),
]


def main():
    """Run each experiment's main() in order, timing each."""
    total = time.perf_counter()
    for name, module in EXPERIMENTS:
        print(f"\n===== {name} =====")
        start = time.perf_counter()
        module.main()
        print(f"  done in {time.perf_counter() - start:.1f}s")
    print(f"\nAll experiments finished in {time.perf_counter() - total:.1f}s")


if __name__ == "__main__":
    main()
