from .common import *
from .preprocessing import *
from .events import *
from .models import *
from .plotting import *
from .summaries import *

# ============================================================
# runtime / methods
# ============================================================

def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_info_string(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        gb = props.total_memory / (1024 ** 3)
        return f"CUDA:{idx} {props.name} ({gb:.1f} GB)"
    except Exception:
        return "CUDA"


def as_device_tensor(U: Any, device: torch.device) -> torch.Tensor:
    """Convert numpy/tensor input to one fp32 tensor on the target device, once."""
    if torch.is_tensor(U):
        return U.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(U, dtype=torch.float32).to(device=device, non_blocking=True)


def run_method_rrcesn(
    name: str,
    df: pd.DataFrame,
    time_seconds: Optional[np.ndarray],
    fs: float,
    train_end: int,
    val_end: int,
    pcfg: PreprocessConfig,
    exclude_prefixes: Sequence[str],
    esn_cfg: ESNConfig,
    score_cfg: ScoreConfig,
    thr_cfg: ThresholdConfig,
    device: torch.device,
    seed: int,
    chunk_size: int,
    out_dir: Path,
    n_ensemble: int,
    temporal_scales: Sequence[int],
    agg: str,
    attribution_topk: int,
    gpu_resident_input: bool = True,
    store_recon: bool = False,
) -> Tuple[MethodResult, Dict[str, Any], pd.DataFrame, PhysicsFeatureEngineer, np.ndarray, np.ndarray]:
    method_dir = out_dir / name
    ensure_dir(method_dir)

    t0 = time.perf_counter()
    fe = PhysicsFeatureEngineer(pcfg)
    X_train_full, feat_names, phase_train, raw_numeric_cols = fe.fit(df.iloc[:train_end], fs=fs)
    X_full, phase_all = fe.transform(df, fs=fs)
    preprocess_time = time.perf_counter() - t0

    score_mask, score_feat_names = select_score_features(feat_names, exclude_prefixes)
    out_idx = np.where(score_mask)[0].astype(np.int64)

    # Keep the full feature matrix on the GPU for ESN runs when possible.
    # This avoids repeated host->device copies inside fit/score loops.
    use_gpu_resident = bool(gpu_resident_input and device.type == "cuda")
    X_full_esn = as_device_tensor(X_full, device) if use_gpu_resident else X_full

    t1 = time.perf_counter()
    score, uncert = run_rrcesn_ensemble(
        U_full=X_full_esn,
        out_idx=out_idx,
        train_end=train_end,
        cfg=esn_cfg,
        scfg=score_cfg,
        n_ensemble=n_ensemble,
        temporal_scales=temporal_scales,
        agg=agg,
        device=device,
        seed=seed,
        chunk_size=chunk_size,
    )
    rrcesn_train_score_total = time.perf_counter() - t1

    alarms, th, summary = detect_anomalies(score, phase_all, train_end=train_end, val_end=val_end, fs=fs, cfg=thr_cfg)
    events = extract_events(alarms, score, fs=fs, time_seconds=time_seconds)

    # Optional representative model for feature attribution. This is expensive,
    # so set --attribution-topk 0 to skip it for speed.
    train_time_rep = 0.0
    rep_score_time = 0.0
    feat_err = None
    recon_rep = None
    if int(attribution_topk) > 0 and events:
        t2 = time.perf_counter()
        rep = RRCESN(cfg=esn_cfg, n_inputs=X_full.shape[1], out_idx=out_idx, device=device, seed=seed)
        rep.fit_streaming(X_full_esn[:train_end] if torch.is_tensor(X_full_esn) else X_full[:train_end], chunk_size=chunk_size)
        train_time_rep = time.perf_counter() - t2

        peaks = [e.peak_idx for e in sorted(events, key=lambda e: e.peak_score, reverse=True)[: int(attribution_topk)]]
        t3 = time.perf_counter()
        _, _, feat_err, recon_rep = rep.score_streaming(
            X_full_esn, scfg=ScoreConfig(mode="mse"), chunk_size=chunk_size,
            store_feature_errors_for_indices=peaks, store_recon=store_recon,
        )
        rep_score_time = time.perf_counter() - t3

    # persist artifacts
    score_df = pd.DataFrame({
        "score": score.astype(np.float32),
        "threshold": th.astype(np.float32),
        "alarm": alarms.astype(np.int32),
        "uncertainty": uncert.astype(np.float32),
    })
    score_df.to_csv(method_dir / f"{name}_scores.csv", index=False)
    pd.DataFrame([asdict(e) for e in events]).to_csv(method_dir / f"events_{name}.csv", index=False)

    if feat_err:
        payload = {}
        for idx, vec in feat_err.items():
            order = np.argsort(vec)[::-1][:15]
            payload[str(idx)] = [(score_feat_names[i], float(vec[i])) for i in order]
        with open(method_dir / "feature_attribution_topk.json", "w") as f:
            json.dump(payload, f, indent=2)
    else:
        payload = {}

    summary = dict(summary)
    summary["D_input_total"] = float(X_full.shape[1])
    summary["D_scored"] = float(out_idx.size)
    summary["physics_enabled"] = bool(pcfg.enable_physics_features)
    summary["score_exclude_prefixes"] = list(exclude_prefixes)
    summary["fs_hz"] = float(fs)

    runtime = {
        "method": name,
        "device": str(device),
        "n_rows": int(len(df)),
        "preprocess_time_s": float(preprocess_time),
        "ensemble_train_plus_score_time_s": float(rrcesn_train_score_total),
        "rep_train_time_s": float(train_time_rep),
        "rep_score_time_s": float(rep_score_time),
        "score_ms_per_sample_rep": float(1000.0 * rep_score_time / max(1, len(df))),
        "train_ms_per_sample_rep": float(1000.0 * train_time_rep / max(1, train_end)),
        "fits_15hz_budget_rep": bool((1000.0 * rep_score_time / max(1, len(df))) < (1000.0 / DEFAULT_BOOSTER_FS)),
    }

    feature_catalog = fe.feature_catalog(score_mask=score_mask)
    feature_catalog.to_csv(method_dir / "feature_catalog.csv", index=False)

    xaxis = time_seconds if time_seconds is not None else np.arange(len(score), dtype=np.float64)
    plot_score_series(method_dir, xaxis, score, th, alarms, title=name, uncert=uncert)
    plot_score_series_log(method_dir, xaxis, score, th, alarms, title=name)
    plot_hist_trim(method_dir, score, title=f"{name}_score_hist", trim_q=0.999)
    plot_hist_log(method_dir, score, title=f"{name}_score_hist")

    # top event diagnostic plot
    top_attrib_names = payload.get(str(events[0].peak_idx), []) if events and payload else []
    top_attrib_names = [k for k, v in top_attrib_names][:3] if top_attrib_names else []
    plot_top_event_raw_and_score(method_dir, df, time_seconds, (max(events, key=lambda e: e.peak_score) if events else None), score, th, feature_catalog, top_attrib=top_attrib_names)

    res = summarize_method(name, score, alarms, th, events, summary)
    return res, runtime, feature_catalog, fe, X_full, phase_all


def run_single_baseline(
    name: str,
    score: np.ndarray,
    phase_all: Optional[np.ndarray],
    time_seconds: Optional[np.ndarray],
    fs: float,
    train_end: int,
    val_end: int,
    thr_cfg: ThresholdConfig,
    method_dir: Path,
) -> MethodResult:
    alarms, th, summ = detect_anomalies(score, phase_all, train_end, val_end, fs, thr_cfg)
    evs = extract_events(alarms, score, fs, time_seconds)
    pd.DataFrame({"score": score, "threshold": th, "alarm": alarms.astype(np.int32)}).to_csv(method_dir / f"{name}_scores.csv", index=False)
    pd.DataFrame([asdict(e) for e in evs]).to_csv(method_dir / f"events_{name}.csv", index=False)
    xaxis = time_seconds if time_seconds is not None else np.arange(len(score), dtype=np.float64)
    plot_score_series(method_dir, xaxis, score, th, alarms, title=name)
    plot_hist_trim(method_dir, score, title=f"{name}_score_hist", trim_q=0.999)
    return summarize_method(name, score, alarms, th, evs, summ)
