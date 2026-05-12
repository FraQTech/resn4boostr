@echo off
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite main --max-rows 20000 --device auto --n-reservoir 200 --n-ensemble 1 --temporal-scales 1 --no-attribution --experiment-name quick_sanity
