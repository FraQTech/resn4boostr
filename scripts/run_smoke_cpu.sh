#!/usr/bin/env bash
set -euo pipefail
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --max-rows 200000 --device auto
