# Reproducibility notes

## Primary command

```bash
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --device cuda --chunk-size 1000000 --accum-batch 16384 --n-reservoir 1200 --n-ensemble 3 --temporal-scales 1 5 15 --ae-batch 256 --ae-hidden 256 --ae-max-train-windows 20000 --attribution-topk 10
```

This command runs the main RRC-ESN, baselines, ablations, proxy validation, runtime accounting, top-event attribution, and synthetic injection evaluation.

## Strict threshold sensitivity command

```bash
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite main --device cuda --chunk-size 1000000 --accum-batch 16384 --n-reservoir 1200 --n-ensemble 3 --temporal-scales 1 5 15 --lags 1 3 5 7 --weights 0.5 0.3 0.15 0.05 --quantile 0.999 --target-events-per-hour 1 --smooth-s 1.0 --min-dwell-s 1.0 --merge-gap-s 0.5 --hysteresis 0.1 --attribution-topk 10 --experiment-name threshold_sensitivity_strict
```

## Smoke command

```bash
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --max-rows 200000 --device auto
```

## Expected headline values from the paper run

Small differences may occur across hardware, PyTorch versions, BLAS backends, and CUDA/TF32 behavior, but the final paper run produced approximately:

- Rows: 1,295,713
- Raw numeric channels: 45
- Sampling rate: about 14.925 Hz
- Main RRC-ESN detected events: 770
- Main RRC-ESN event rate: 31.93 events/hour
- Main RRC-ESN alarm duty: 11.72%
- Event-level proxy precision vs I:IB proxy: 0.965
- Event-level proxy recall vs I:IB proxy: 0.0129
- Sample AP vs proxy: 0.461
- Top event: B:VIMIN/B:VIMAX transient near sample 253000, about 16867 seconds
- GPU scoring latency: about 0.553 ms/sample
- Main limitation: severe late-run nonstationarity in the test segment

## Output files to check

After a run, inspect the latest directory under `runs/` and look for:

```text
summary.csv
runtime.csv
proxy_validation_metrics.json
proxy_alignment.csv
feature_catalog.csv
channel_summary.csv
*_timeseries.png
*_hist_trim.png
*_hist_log.png
```

The script writes many intermediate artifacts because the PRAB analysis was assembled from these run outputs.
