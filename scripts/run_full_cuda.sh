#!/usr/bin/env bash
set -euo pipefail
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --device cuda --chunk-size 1000000 --accum-batch 16384 --n-reservoir 1200 --n-ensemble 3 --temporal-scales 1 5 15 --ae-batch 256 --ae-hidden 256 --ae-max-train-windows 20000 --attribution-topk 10
