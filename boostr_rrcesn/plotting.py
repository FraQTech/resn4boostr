from .common import *

# ============================================================
# plots
# ============================================================

def plot_score_series(out_dir: Path, xaxis: np.ndarray, scores: np.ndarray, th: np.ndarray, alarms: np.ndarray, title: str, uncert: Optional[np.ndarray] = None) -> None:
    xaxis, [scores, th, alarms, uncert2] = trim_for_plot(xaxis, scores, th, alarms.astype(np.float32), (uncert if uncert is not None else np.zeros_like(scores)))
    plt.figure(figsize=(14, 4))
    plt.plot(xaxis, scores, linewidth=1.0, label="score")
    plt.plot(xaxis, th, linewidth=1.0, label="threshold")
    if uncert is not None and np.any(np.isfinite(uncert2)):
        u = np.nan_to_num(uncert2, nan=0.0)
        plt.fill_between(xaxis, scores - u, scores + u, alpha=0.15, label="uncertainty")
    idx = np.where(alarms > 0.5)[0]
    if len(idx):
        plt.scatter(xaxis[idx], scores[idx], s=8, marker="x", label="alarm", alpha=0.6)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_').lower()}_timeseries.png", dpi=180)
    plt.close()


def plot_score_series_log(out_dir: Path, xaxis: np.ndarray, scores: np.ndarray, th: np.ndarray, alarms: np.ndarray, title: str) -> None:
    eps = 1e-12
    xaxis, [scores, th, alarms] = trim_for_plot(xaxis, scores, th, alarms.astype(np.float32))
    plt.figure(figsize=(14, 4))
    plt.plot(xaxis, np.log10(np.maximum(scores, 0) + eps), linewidth=1.0, label="log10(score+eps)")
    plt.plot(xaxis, np.log10(np.maximum(th, 0) + eps), linewidth=1.0, label="log10(th+eps)")
    idx = np.where(alarms > 0.5)[0]
    if len(idx):
        plt.scatter(xaxis[idx], np.log10(np.maximum(scores[idx], 0) + eps), s=8, marker="x", label="alarm", alpha=0.6)
    plt.title(f"{title} (log)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_').lower()}_timeseries_log.png", dpi=180)
    plt.close()


def plot_hist_trim(out_dir: Path, scores: np.ndarray, title: str, trim_q: float = 0.999) -> None:
    s = scores[np.isfinite(scores)]
    if s.size == 0:
        return
    hi = np.quantile(s, trim_q)
    plt.figure(figsize=(8, 4))
    plt.hist(s[s <= hi], bins=80)
    plt.title(f"{title} (trim {trim_q:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_').lower()}_hist_trim.png", dpi=180)
    plt.close()


def plot_hist_log(out_dir: Path, scores: np.ndarray, title: str) -> None:
    s = scores[np.isfinite(scores)]
    if s.size == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(np.log10(np.maximum(s, 0) + 1e-12), bins=80)
    plt.title(f"{title} (log10)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_').lower()}_hist_log.png", dpi=180)
    plt.close()


def plot_delay_hist(out_dir: Path, delays: np.ndarray, title: str, fname: str) -> None:
    d = delays[np.isfinite(delays)]
    if d.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(d, bins=40)
    plt.title(title)
    plt.xlabel("delay (s)")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=180)
    plt.close()


def plot_split_shift_heatmap(out_dir: Path, channel_summary: pd.DataFrame, top_k: int = 25) -> None:
    if channel_summary.empty:
        return
    df = channel_summary.head(top_k).copy()
    vals = np.vstack([
        df["train_mean"].to_numpy(dtype=np.float64),
        df["val_mean"].to_numpy(dtype=np.float64),
        df["test_mean"].to_numpy(dtype=np.float64),
    ])
    plt.figure(figsize=(12, max(6, 0.35 * len(df))))
    plt.imshow(vals, aspect="auto")
    plt.yticks([0, 1, 2], ["train", "val", "test"])
    plt.xticks(np.arange(len(df)), df["feature"], rotation=90, fontsize=7)
    plt.colorbar(label="mean value")
    plt.title("Top shifted channels: split-wise means")
    plt.tight_layout()
    plt.savefig(out_dir / "split_shift_heatmap.png", dpi=180)
    plt.close()


def plot_top_event_raw_and_score(
    out_dir: Path,
    df: pd.DataFrame,
    time_seconds: Optional[np.ndarray],
    event: Optional[Event],
    scores: np.ndarray,
    threshold: np.ndarray,
    feature_catalog: pd.DataFrame,
    top_attrib: Optional[List[str]] = None,
    fallback_raw_cols: Optional[List[str]] = None,
) -> None:
    if event is None:
        return
    if fallback_raw_cols is None:
        fallback_raw_cols = [c for c in df.columns if c.startswith("I:")][:3]
        if not fallback_raw_cols:
            fallback_raw_cols = [c for c in df.select_dtypes(include=[np.number]).columns[:3]]

    cols = []
    for c in (top_attrib or []):
        if c in df.columns and c not in cols:
            cols.append(c)
        base = c.split("_")[0]
        if base in df.columns and base not in cols:
            cols.append(base)
    for c in fallback_raw_cols:
        if c in df.columns and c not in cols:
            cols.append(c)
    cols = cols[:3]
    if not cols:
        return

    pad = 300
    a = max(0, event.start_idx - pad)
    b = min(len(df), event.end_idx + pad)
    if time_seconds is not None and len(time_seconds) == len(df):
        x = time_seconds[a:b]
        xlabel = "seconds"
    else:
        x = np.arange(a, b)
        xlabel = "index"

    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(14, 3 * (len(cols) + 1)), sharex=True)
    if len(cols) == 0:
        axes = [axes]
    for k, c in enumerate(cols):
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)[a:b]
        axes[k].plot(x, y, linewidth=1.0)
        axes[k].axvspan(x[max(0, event.start_idx - a)], x[min(len(x)-1, event.end_idx - a - 1)], alpha=0.15)
        axes[k].set_title(c)
    axes[-1].plot(x, scores[a:b], label="score", linewidth=1.0)
    axes[-1].plot(x, threshold[a:b], label="threshold", linewidth=1.0)
    axes[-1].axvspan(x[max(0, event.start_idx - a)], x[min(len(x)-1, event.end_idx - a - 1)], alpha=0.15)
    axes[-1].legend(loc="upper right")
    axes[-1].set_title("Anomaly score")
    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_dir / "top_event_raw_and_score.png", dpi=180)
    plt.close()

