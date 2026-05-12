from .common import *

# ============================================================
# thresholding / events
# ============================================================

def compute_regime_masks(phase: Optional[np.ndarray], T: int) -> Dict[str, np.ndarray]:
    if phase is None or len(phase) != T:
        return {"all": np.ones((T,), dtype=bool)}
    inj = phase < (np.pi / 4.0)
    ext = phase >= (7.0 * np.pi / 4.0)
    acc = ~(inj | ext)
    return {"injection": inj, "acceleration": acc, "extraction": ext}


def transition_mask(scores: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
    s = np.nan_to_num(scores, nan=np.nanmedian(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 0.0)
    ds = np.abs(np.diff(s, prepend=s[0]))
    w = max(5, int(win_s * fs))
    var = moving_mean(ds, w)
    thr = np.quantile(var, 0.90)
    return var > thr


def smooth_scores(scores: np.ndarray, fs: float, smooth_s: float) -> np.ndarray:
    base = np.nanmedian(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 0.0
    return moving_mean(np.nan_to_num(scores, nan=base).astype(np.float32), max(1, int(smooth_s * fs)))


def apply_hysteresis(score: np.ndarray, th_hi: np.ndarray, hysteresis: float) -> np.ndarray:
    th_lo = (1.0 - float(hysteresis)) * th_hi
    out = np.zeros_like(score, dtype=bool)
    state = False
    for i in range(len(score)):
        if not np.isfinite(score[i]):
            state = False
            out[i] = False
            continue
        if (not state) and score[i] > th_hi[i]:
            state = True
        elif state and score[i] < th_lo[i]:
            state = False
        out[i] = state
    return out


def enforce_min_dwell(binary: np.ndarray, dwell_samples: int) -> np.ndarray:
    b = np.asarray(binary).astype(bool).copy()
    i = 0
    n = len(b)
    while i < n:
        if b[i]:
            j = i + 1
            while j < n and b[j]:
                j += 1
            if (j - i) < dwell_samples:
                b[i:j] = False
            i = j
        else:
            i += 1
    return b


def merge_close_events(binary: np.ndarray, gap_samples: int) -> np.ndarray:
    if gap_samples <= 0:
        return np.asarray(binary).astype(bool)
    b = np.asarray(binary).astype(bool).copy()
    n = len(b)
    i = 0
    while i < n:
        if b[i]:
            j = i + 1
            while j < n and b[j]:
                j += 1
            k = j
            while k < n and (not b[k]) and (k - j) < gap_samples:
                k += 1
            if k < n and b[k] and (k - j) < gap_samples:
                b[j:k] = True
            i = k
        else:
            i += 1
    return b


def calibrate_quantile_on_slice(scores: np.ndarray, phase: Optional[np.ndarray], train_end: int, val_end: int, fs: float, cfg: ThresholdConfig) -> float:
    val_slice = slice(train_end, val_end)
    s_val = scores[val_slice]
    s_tr = scores[:train_end]
    s_tr = s_tr[np.isfinite(s_tr)]
    if s_val.size == 0 or s_tr.size == 0:
        return float(cfg.quantile)

    masks = compute_regime_masks(phase, len(scores))
    dwell = max(3, int(cfg.min_dwell_s * fs))
    gap = max(0, int(cfg.merge_gap_s * fs))
    grid = np.unique(np.clip(np.concatenate([
        np.linspace(0.990, 0.999, 10),
        np.linspace(0.999, 0.9999, 8),
        np.linspace(0.9999, 0.99999, 5),
    ]), 0.90, 0.99999))

    best_q = float(cfg.quantile)
    best_err = float("inf")
    for q in grid:
        base = float(np.quantile(s_tr, q))
        th = np.full_like(scores, base, dtype=np.float32)
        if cfg.regime_aware and "all" not in masks:
            for name, m in masks.items():
                fac = float(PHYSICS_CONFIG["accelerator_regimes"].get(name, {"threshold_factor": 1.0})["threshold_factor"])
                th[m] *= fac
        if cfg.transition_sensitivity < 1.0:
            tm = transition_mask(scores, fs)
            th[tm] *= float(cfg.transition_sensitivity)
        det = apply_hysteresis(scores, th, cfg.hysteresis)
        det = enforce_min_dwell(det, dwell)
        det = merge_close_events(det, gap)
        det_val = det[val_slice]
        hours = (len(det_val) / fs) / 3600.0
        rate = count_events(det_val) / hours if hours > 0 else np.inf
        err = abs(rate - float(cfg.target_events_per_hour))
        if err < best_err:
            best_err = err
            best_q = float(q)
    return best_q


def detect_anomalies(scores_raw: np.ndarray, phase: Optional[np.ndarray], train_end: int, val_end: int, fs: float, cfg: ThresholdConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    scores = smooth_scores(scores_raw.astype(np.float32), fs, cfg.smooth_s)
    masks = compute_regime_masks(phase, len(scores))
    chosen_q = float(cfg.quantile)
    if cfg.method == "quantile" and cfg.calibrate_on == "val":
        chosen_q = calibrate_quantile_on_slice(scores, phase, train_end, val_end, fs, cfg)

    th = np.zeros_like(scores, dtype=np.float32)
    if cfg.regime_aware and "all" not in masks:
        for name, m in masks.items():
            s_tr = scores[:train_end][m[:train_end]]
            s_tr = s_tr[np.isfinite(s_tr)]
            if s_tr.size == 0:
                base = np.nanmedian(scores[:train_end]) if np.any(np.isfinite(scores[:train_end])) else 0.0
            else:
                if cfg.method == "mad":
                    base = float(np.median(s_tr) + cfg.mad_factor * (robust_mad(s_tr) + EPS))
                else:
                    base = float(np.quantile(s_tr, chosen_q))
            fac = float(PHYSICS_CONFIG["accelerator_regimes"].get(name, {"threshold_factor": 1.0})["threshold_factor"])
            th[m] = base * fac
    else:
        s_tr = scores[:train_end]
        s_tr = s_tr[np.isfinite(s_tr)]
        if s_tr.size == 0:
            base = np.nanmedian(scores) if np.any(np.isfinite(scores)) else 0.0
        else:
            base = float(np.median(s_tr) + cfg.mad_factor * (robust_mad(s_tr) + EPS)) if cfg.method == "mad" else float(np.quantile(s_tr, chosen_q))
        th[:] = base

    if cfg.transition_sensitivity < 1.0:
        tm = transition_mask(scores, fs)
        th[tm] *= float(cfg.transition_sensitivity)

    alarms = apply_hysteresis(scores, th, cfg.hysteresis)
    alarms = enforce_min_dwell(alarms, max(3, int(cfg.min_dwell_s * fs)))
    alarms = merge_close_events(alarms, max(0, int(cfg.merge_gap_s * fs)))

    finite = np.isfinite(scores)
    events = count_events(alarms[finite])
    hours = (np.sum(finite) / fs) / 3600.0
    event_rate = events / hours if hours > 0 else float("nan")
    duty = float(np.mean(alarms[finite])) if np.any(finite) else float("nan")
    durs = event_durations(alarms[finite], fs)

    return alarms, th, {
        "events": float(events),
        "events_per_hour": float(event_rate),
        "alarm_duty_cycle": float(duty),
        "mean_event_duration_s": float(np.mean(durs) if durs else 0.0),
        "median_event_duration_s": float(np.median(durs) if durs else 0.0),
        "threshold_method": cfg.method,
        "chosen_quantile": float(chosen_q),
        "smooth_s": float(cfg.smooth_s),
        "min_dwell_s": float(cfg.min_dwell_s),
        "merge_gap_s": float(cfg.merge_gap_s),
        "hysteresis": float(cfg.hysteresis),
        "regime_aware": bool(cfg.regime_aware),
        "transition_sensitivity": float(cfg.transition_sensitivity),
    }


def extract_events(alarms: np.ndarray, scores: np.ndarray, fs: float, time_seconds: Optional[np.ndarray]) -> List[Event]:
    b = np.asarray(alarms).astype(bool)
    evs: List[Event] = []
    i = 0
    n = len(b)
    while i < n:
        if b[i]:
            j = i + 1
            while j < n and b[j]:
                j += 1
            seg = scores[i:j]
            rel = int(np.nanargmax(seg)) if np.any(np.isfinite(seg)) else 0
            peak = i + rel
            if time_seconds is not None and len(time_seconds) == n:
                st = float(time_seconds[i]); en = float(time_seconds[j - 1]); pk = float(time_seconds[peak])
            else:
                st = en = pk = None
            evs.append(Event(
                start_idx=int(i),
                end_idx=int(j),
                peak_idx=int(peak),
                peak_score=float(scores[peak]) if np.isfinite(scores[peak]) else float("nan"),
                duration_s=float((j - i) / fs),
                start_time_s=st,
                end_time_s=en,
                peak_time_s=pk,
            ))
            i = j
        else:
            i += 1
    return evs


def alarms_jaccard(a: np.ndarray, b: Optional[np.ndarray]) -> float:
    if b is None:
        return float("nan")
    aa = np.asarray(a).astype(bool)
    bb = np.asarray(b).astype(bool)
    union = np.sum(aa | bb)
    return float(np.sum(aa & bb) / union) if union > 0 else 0.0


def event_overlaps_any(e: Event, others: List[Event]) -> bool:
    for o in others:
        if not (e.end_idx <= o.start_idx or e.start_idx >= o.end_idx):
            return True
    return False


