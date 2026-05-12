from .common import *
from .events import *

# ============================================================
# validation helpers
# ============================================================

def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if y_true.size == 0:
        return float("nan")
    P = int(y_true.sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / float(P)
    pos_idx = np.where(y == 1)[0]
    if pos_idx.size == 0:
        return float("nan")
    ap = 0.0
    prev_rec = 0.0
    for k in pos_idx:
        r = rec[k]
        ap += prec[k] * (r - prev_rec)
        prev_rec = r
    return float(ap)


def build_events_from_binary(mask: np.ndarray, score: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, float]]:
    m = np.asarray(mask).astype(bool)
    events = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i + 1
            while j < len(m) and m[j]:
                j += 1
            start, end = i, j - 1
            if score is not None and np.any(np.isfinite(score[start:j])):
                rel = int(np.nanargmax(score[start:j]))
                peak = start + rel
                pk = float(score[peak])
            else:
                peak = (start + end) // 2
                pk = float("nan")
            events.append((start, end, peak, pk))
            i = j
        else:
            i += 1
    return events


def event_pr_curve(det_events: List[Tuple[int, int, int, float]], gt_events: List[Tuple[int, int, int, float]]) -> Tuple[np.ndarray, np.ndarray, float]:
    if len(gt_events) == 0:
        return np.array([]), np.array([]), float("nan")
    scores = np.array([(-1e30 if not np.isfinite(e[3]) else e[3]) for e in det_events], dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    det_sorted = [det_events[i] for i in order]
    matched_gt = np.zeros((len(gt_events),), dtype=bool)
    tp = 0
    prec_list = []
    rec_list = []
    for k, de in enumerate(det_sorted, start=1):
        hit = False
        for gi, ge in enumerate(gt_events):
            if matched_gt[gi]:
                continue
            if overlap((de[0], de[1]), (ge[0], ge[1])):
                matched_gt[gi] = True
                hit = True
                break
        if hit:
            tp += 1
        prec_list.append(tp / k)
        rec_list.append(tp / len(gt_events))
    precision = np.array(prec_list)
    recall = np.array(rec_list)
    auc = 0.0
    prev_r = 0.0
    prev_p = 1.0
    for r, p in zip(recall, precision):
        auc += (r - prev_r) * (p + prev_p) / 2.0
        prev_r, prev_p = r, p
    return recall, precision, float(auc)


def compute_proxy_mask_from_intensity(
    x: np.ndarray,
    fs: float,
    baseline_s: float,
    ratio_thr: float,
    abs_quantile: float,
    min_dwell_s: float,
    merge_gap_s: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros((len(x),), dtype=bool)
    x = np.nan_to_num(x, nan=np.nanmedian(x[finite]), posinf=np.nanmax(x[finite]), neginf=0.0)
    win = max(3, int(baseline_s * fs))
    base = pd.Series(x).rolling(win, center=True, min_periods=max(3, win // 10)).median().to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(base)):
        base = pd.Series(base).bfill().ffill().to_numpy(dtype=np.float64)
    base = np.maximum(base, 1e-6)
    ratio = x / base
    qthr = float(np.quantile(x, abs_quantile))
    raw = (ratio < float(ratio_thr)) | (x < qthr)
    m = enforce_min_dwell(raw, max(1, int(min_dwell_s * fs)))
    m = merge_close_events(m, max(0, int(merge_gap_s * fs)))
    return m.astype(bool)


def validate_against_proxy(
    df: pd.DataFrame,
    score: np.ndarray,
    alarms: np.ndarray,
    fs: float,
    out_dir: Path,
    proxy_col: Optional[str] = None,
    baseline_s: float = 30.0,
    ratio_thr: float = 0.85,
    abs_quantile: float = 0.005,
    min_dwell_s: float = 0.5,
    merge_gap_s: float = 0.25,
) -> Dict[str, Any]:
    cols = list(df.columns)
    proxy_col = proxy_col or pick_proxy_col(cols)
    if proxy_col is None or proxy_col not in df.columns:
        return {"proxy_validation_available": False}

    x = pd.to_numeric(df[proxy_col], errors="coerce").to_numpy(dtype=np.float64)
    proxy_mask = compute_proxy_mask_from_intensity(
        x=x, fs=fs, baseline_s=baseline_s, ratio_thr=ratio_thr,
        abs_quantile=abs_quantile, min_dwell_s=min_dwell_s, merge_gap_s=merge_gap_s,
    )
    det_events = build_events_from_binary(alarms, score)
    gt_events = build_events_from_binary(proxy_mask, -np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else 0.0))

    overlap_count = 0
    delays_start = []
    delays_peak = []
    gt_hit = np.zeros((len(gt_events),), dtype=bool)
    rows = []
    for i, de in enumerate(det_events):
        hit_idx = None
        for gi, ge in enumerate(gt_events):
            if overlap((de[0], de[1]), (ge[0], ge[1])):
                hit_idx = gi
                break
        if hit_idx is not None:
            overlap_count += 1
            gt_hit[hit_idx] = True
            delays_start.append((de[0] - gt_events[hit_idx][0]) / fs)
            delays_peak.append((de[2] - gt_events[hit_idx][2]) / fs)
        rows.append({
            "det_i": i, "det_start_idx": de[0], "det_end_idx": de[1], "det_peak_idx": de[2],
            "det_peak_score": de[3], "hit": int(hit_idx is not None),
            "matched_gt_i": -1 if hit_idx is None else int(hit_idx),
            "delay_start_s": np.nan if hit_idx is None else (de[0] - gt_events[hit_idx][0]) / fs,
            "delay_peak_s": np.nan if hit_idx is None else (de[2] - gt_events[hit_idx][2]) / fs,
        })

    det_precision = overlap_count / max(len(det_events), 1)
    gt_recall = int(gt_hit.sum()) / max(len(gt_events), 1)
    sample_ap = average_precision_binary(proxy_mask.astype(np.int32), score) if len(score) == len(proxy_mask) else float("nan")
    rec, prec, event_auc = event_pr_curve(det_events, gt_events)

    pd.DataFrame(rows).to_csv(out_dir / "proxy_det_event_matches.csv", index=False)
    pd.DataFrame({
        "proxy_mask": proxy_mask.astype(np.int32),
        "score": score.astype(np.float32),
        "alarm": alarms.astype(np.int32),
    }).to_csv(out_dir / "proxy_alignment.csv", index=False)
    pd.DataFrame(gt_events, columns=["start_idx", "end_idx", "peak_idx", "peak_score"]).to_csv(out_dir / "proxy_events.csv", index=False)
    if len(rec):
        pd.DataFrame({"recall": rec, "precision": prec}).to_csv(out_dir / "proxy_event_pr_curve.csv", index=False)

    metrics = {
        "proxy_validation_available": True,
        "proxy_col": proxy_col,
        "proxy_events": int(len(gt_events)),
        "detected_events": int(len(det_events)),
        "event_precision_vs_proxy": float(det_precision),
        "event_recall_vs_proxy": float(gt_recall),
        "event_pr_auc_vs_proxy": float(event_auc),
        "sample_ap_vs_proxy": float(sample_ap),
        "mean_delay_start_s": float(np.mean(delays_start) if delays_start else float("nan")),
        "median_delay_start_s": float(np.median(delays_start) if delays_start else float("nan")),
        "mean_delay_peak_s": float(np.mean(delays_peak) if delays_peak else float("nan")),
        "median_delay_peak_s": float(np.median(delays_peak) if delays_peak else float("nan")),
    }
    with open(out_dir / "proxy_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

