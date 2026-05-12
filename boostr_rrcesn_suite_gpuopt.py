#!/usr/bin/env python3
"""Command-line wrapper preserving the original one-file command interface.

The implementation now lives in the modular ``boostr_rrcesn`` package.
All previous commands continue to work, for example:

    python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --device cuda
"""
from boostr_rrcesn.cli import main

if __name__ == "__main__":
    main()
