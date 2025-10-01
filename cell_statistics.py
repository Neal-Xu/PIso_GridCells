#!/usr/bin/env python3
"""
Entry point that adds a --grid-threshold flag without touching the representation module.

Why this file exists:
- representation.main() already parses --rho/--size/--run and returns (ratemaps, gs, file_appendix).
- We want to override grid_score_threshold only for this invocation, while keeping other callers unchanged.
- This wrapper parses --grid-threshold itself, strips it from sys.argv, then delegates the rest to representation.main().
"""

import sys
import argparse

# Import the public API we need from the representation module.
# Assumptions based on your original snippet:
#   - representation.main() -> (ratemaps, gs, file_appendix)
#   - representation exposes analyze_grid_cells(ratemaps, gs, grid_score_threshold, file_appendix)
from representation import main as rep_main, analyze_grid_cells


def parse_own_args(argv):
    """
    Parse ONLY the flag introduced by this entry-point, leaving all the
    original flags (like --rho/--size/--run) untouched for representation.main().

    We use add_help=False so -h/--help remains available to representation.main().
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--grid-threshold",
        type=float,
        default=0.7,   # keep backward compatibility for callers that don't pass it
        help="Threshold for grid score (default: 0.7)."
    )
    # parse_known_args returns the parsed args and the rest (unconsumed) arguments
    own_args, rest = parser.parse_known_args(argv[1:])
    return own_args, rest


def main():
    # 1) Parse our own flag and keep the rest for representation.main()
    own_args, rest_argv = parse_own_args(sys.argv)

    # Optional: light validation (comment out if you don't need it)
    # if not (0.0 <= own_args.grid_threshold <= 1.0):
    #     raise ValueError("--grid-threshold should be in [0, 1].")

    # 2) Temporarily replace sys.argv so representation.main() only sees its own flags
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + rest_argv
        ratemaps, gs, file_appendix = rep_main()
    finally:
        # Always restore sys.argv to be polite
        sys.argv = saved_argv

    # 3) Run the downstream analysis with the requested threshold
    analyze_grid_cells(
        ratemaps,
        gs,
        grid_score_threshold=own_args.grid_threshold,
        file_appendix=file_appendix
    )


if __name__ == "__main__":
    main()
