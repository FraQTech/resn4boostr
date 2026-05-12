
#!/usr/bin/env python3
"""
BOOSTR RRC-ESN Research Suite (GPU-Optimized)
=============================================

Standalone end-to-end script for BOOSTR Partial Release CSV:
- Physics-augmented feature engineering for BOOSTR / GMPS / beam channels
- GPU/CPU reconstructive multi-lag ESN with streaming ridge regression
- GPU-optimized ESN training: CUDA-resident inputs, circular lag buffer, batched ridge accumulators, TF32
- Correct score exclusion semantics:
    conditioning features (e.g. reg_ / cycle_) can shape the reservoir
    while being excluded from prediction/scoring
- Ensemble over seeds / temporal scales
- Thresholding with validation calibration to target events/hour
- Event extraction, feature attribution, runtime accounting
- Baselines: PCA-Q, Isolation Forest, LSTM reconstruction baseline
- Ablations: no physics, no score exclusion, no ensemble, threshold MAD
- Integrated proxy/expert validation, synthetic injection evaluation
- Visualizations and CSV/JSON artifacts
- tqdm progress bars everywhere meaningful

Requirements:
    pip install numpy pandas matplotlib tqdm scipy scikit-learn torch

Typical usage:
    python boostr_rrcesn_suite.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    uniform_filter1d = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    PCA = None
    RobustScaler = None
    IsolationForest = None

import torch
import torch.nn as nn

# CUDA performance knobs. TF32 speeds up fp32 GEMM/matvecs on Ampere/Ada/Hopper GPUs.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ============================================================
# constants / utility
# ============================================================

DEFAULT_BOOSTER_FS = 15.0
MAX_PLOT_POINTS = 150_000
EPS = 1e-12

PHYSICS_CONFIG = {
    "booster_frequency_hz": DEFAULT_BOOSTER_FS,
    "accelerator_regimes": {
        "injection": {"threshold_factor": 0.90},
        "acceleration": {"threshold_factor": 1.00},
        "extraction": {"threshold_factor": 0.92},
    },
    "proxy_preferred_cols": ["I:IB", "I:MXIB", "I:MDAT40", "I:IBEAM", "I:IBCT"],
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_time_seconds(series: pd.Series) -> Optional[np.ndarray]:
    try:
        ts = pd.to_datetime(series, errors="coerce", utc=True)
        if ts.isna().all():
            return None
        t0 = ts.iloc[0]
        return (ts - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    except Exception:
        return None


def estimate_fs_hz(tsec: Optional[np.ndarray], fallback: float = DEFAULT_BOOSTER_FS) -> float:
    if tsec is None or len(tsec) < 3:
        return float(fallback)
    dt = np.diff(tsec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float(fallback)
    return float(1.0 / np.median(dt))


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def moving_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.astype(np.float32, copy=False)
    x = np.asarray(x, dtype=np.float64)
    if HAS_SCIPY:
        return uniform_filter1d(x, size=w, mode="nearest").astype(np.float32)
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, k, mode="same").astype(np.float32)


def moving_std(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return np.zeros_like(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float64)
    if HAS_SCIPY:
        m = uniform_filter1d(x, size=w, mode="nearest")
        m2 = uniform_filter1d(x * x, size=w, mode="nearest")
        v = np.maximum(m2 - m * m, 0.0)
        return np.sqrt(v).astype(np.float32)
    k = np.ones(w, dtype=np.float64) / float(w)
    m = np.convolve(x, k, mode="same")
    m2 = np.convolve(x * x, k, mode="same")
    v = np.maximum(m2 - m * m, 0.0)
    return np.sqrt(v).astype(np.float32)


def moving_mean_absdiff(x: np.ndarray, w: int) -> np.ndarray:
    d = np.abs(np.diff(np.asarray(x, dtype=np.float64), prepend=float(x[0]) if len(x) else 0.0))
    if w <= 1:
        return d.astype(np.float32)
    if HAS_SCIPY:
        return uniform_filter1d(d, size=w, mode="nearest").astype(np.float32)
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(d, k, mode="same").astype(np.float32)


def trim_for_plot(x: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    if len(x) <= MAX_PLOT_POINTS:
        return x, [np.asarray(y) for y in ys]
    step = int(math.ceil(len(x) / MAX_PLOT_POINTS))
    return x[::step], [np.asarray(y)[::step] for y in ys]


def infer_subsystem(col: str) -> str:
    c = col.lower()
    if c.startswith("i:") or "beam" in c or "current" in c or "ib" in c:
        return "beam_monitoring"
    if "gmps" in c or "vim" in c or "vinh" in c or "viphas" in c or "magnet" in c or "vi" in c:
        return "gmps_or_magnet"
    if "rf" in c or "phase" in c or "frq" in c or "linfrq" in c or "iph" in c:
        return "rf_or_phase"
    if "acm" in c or "imax" in c or "imin" in c:
        return "control_or_limits"
    if c.startswith("b:") or c.startswith("b_"):
        return "booster_control"
    return "other"


def pick_proxy_col(columns: List[str]) -> Optional[str]:
    for c in PHYSICS_CONFIG["proxy_preferred_cols"]:
        if c in columns:
            return c
    i_cols = [c for c in columns if c.startswith("I:")]
    return i_cols[0] if i_cols else None


def count_events(binary: np.ndarray) -> int:
    b = np.asarray(binary).astype(bool)
    if b.size == 0:
        return 0
    return int(np.sum((~b[:-1]) & (b[1:])) + (1 if b[0] else 0))


def event_durations(binary: np.ndarray, fs: float) -> List[float]:
    b = np.asarray(binary).astype(bool)
    out: List[float] = []
    i = 0
    n = len(b)
    while i < n:
        if b[i]:
            j = i + 1
            while j < n and b[j]:
                j += 1
            out.append((j - i) / fs)
            i = j
        else:
            i += 1
    return out

# ============================================================
# configs
# ============================================================

@dataclass
class PreprocessConfig:
    time_col: str = "time"
    enable_physics_features: bool = True
    booster_cycle_harmonics: int = 5
    gmps_feature_k: int = 8
    beam_feature_k: int = 6
    rolling_window_s: float = 0.2
    add_phase_cross_terms: bool = True
    iqr_floor: float = 1e-2
    treat_binary_as_unscaled: bool = True
    pca_retain: float = 1.0  # 1.0 disables PCA


@dataclass
class ESNConfig:
    n_reservoir: int = 1200
    spectral_radius: float = 0.98
    input_scaling: float = 0.30
    leak_rate: float = 0.15
    sparsity: float = 0.02
    ridge_alpha: float = 1e-4
    washout: int = 150
    lags: Tuple[int, ...] = (1, 3, 5, 7)
    weights: Tuple[float, ...] = (0.5, 0.3, 0.15, 0.05)
    include_input_in_state: bool = False
    accum_batch_size: int = 4096  # state/readout normal-equation flush size for GEMM acceleration


@dataclass
class ScoreConfig:
    mode: str = "mse"       # mse|huber|capped
    huber_delta: float = 1.0
    residual_cap: float = 8.0


@dataclass
class ThresholdConfig:
    method: str = "quantile"   # mad|quantile
    mad_factor: float = 6.0
    quantile: float = 0.998
    calibrate_on: str = "val"  # none|val|train
    target_events_per_hour: float = 5.0
    smooth_s: float = 1.0
    min_dwell_s: float = 1.0
    merge_gap_s: float = 0.5
    hysteresis: float = 0.10
    regime_aware: bool = True
    transition_sensitivity: float = 1.0


@dataclass
class InjectConfig:
    enabled: bool = False
    segment_max_samples: int = 250_000
    warmup_samples: int = 5000
    n_injections: int = 10
    duration_s: float = 2.0
    spike_scale: float = 6.0
    drift_scale: float = 2.5
    dropout_prob: float = 0.5
    random_channels: int = 5
    seed: int = 123


@dataclass
class Event:
    start_idx: int
    end_idx: int  # exclusive
    peak_idx: int
    peak_score: float
    duration_s: float
    start_time_s: Optional[float]
    end_time_s: Optional[float]
    peak_time_s: Optional[float]


@dataclass
class MethodResult:
    name: str
    scores: np.ndarray
    threshold: np.ndarray
    alarms: np.ndarray
    events: List[Event]
    summary: Dict[str, float]
    top_event: Optional[Event]
    extra: Dict[str, Any]


# Shared tensor helper used by ESN and runtime modules.
def as_device_tensor(U: Any, device: torch.device) -> torch.Tensor:
    """Convert numpy/tensor input to one fp32 tensor on the target device, once."""
    if torch.is_tensor(U):
        return U.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(U, dtype=torch.float32).to(device=device, non_blocking=True)

