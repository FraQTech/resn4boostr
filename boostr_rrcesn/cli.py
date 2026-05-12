from .common import *
from .preprocessing import *
from .events import *
from .models import *
from .baselines import *
from .validation import *
from .injection import *
from .plotting import *
from .summaries import *
from .runtime import *

# ============================================================
# main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--time-col", type=str, default="time")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--experiment-name", type=str, default="boostr_rrcesn_suite")
    p.add_argument("--out-dir", type=str, default="runs")

    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)

    p.add_argument("--suite", type=str, default="main", choices=["main", "baselines", "ablations", "full"])
    p.add_argument("--run-baselines", action="store_true")
    p.add_argument("--run-injection-eval", action="store_true")

    # preprocessing
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--harmonics", type=int, default=5)
    p.add_argument("--gmps-k", type=int, default=8)
    p.add_argument("--beam-k", type=int, default=6)
    p.add_argument("--rolling-window-s", type=float, default=0.2)
    p.add_argument("--iqr-floor", type=float, default=1e-2)
    p.add_argument("--score-exclude-prefix", nargs="*", default=["reg_", "cycle_"])
    p.add_argument("--pca-retain", type=float, default=1.0)

    # ESN
    p.add_argument("--n-reservoir", type=int, default=1200)
    p.add_argument("--spectral-radius", type=float, default=0.98)
    p.add_argument("--input-scaling", type=float, default=0.3)
    p.add_argument("--leak-rate", type=float, default=0.15)
    p.add_argument("--sparsity", type=float, default=0.02)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--washout", type=int, default=150)
    p.add_argument("--include-input-in-state", action="store_true")
    p.add_argument("--accum-batch", type=int, default=4096, help="Ridge accumulator GEMM batch size; larger uses more GPU memory but reduces outer-product overhead.")
    p.add_argument("--lags", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.3, 0.15, 0.05])

    # scoring
    p.add_argument("--score-mode", type=str, default="mse", choices=["mse", "huber", "capped"])
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--residual-cap", type=float, default=8.0)

    # ensemble
    p.add_argument("--n-ensemble", type=int, default=3)
    p.add_argument("--temporal-scales", type=int, nargs="+", default=[1, 5, 15])
    p.add_argument("--agg", type=str, default="median", choices=["median", "mean"])

    # thresholding
    p.add_argument("--threshold-method", type=str, default="quantile", choices=["mad", "quantile"])
    p.add_argument("--mad-factor", type=float, default=6.0)
    p.add_argument("--quantile", type=float, default=0.998)
    p.add_argument("--calibrate-on", type=str, default="val", choices=["none", "val", "train"])
    p.add_argument("--target-events-per-hour", type=float, default=5.0)
    p.add_argument("--smooth-s", type=float, default=1.0)
    p.add_argument("--min-dwell-s", type=float, default=1.0)
    p.add_argument("--merge-gap-s", type=float, default=0.5)
    p.add_argument("--hysteresis", type=float, default=0.10)
    p.add_argument("--no-regime-aware", action="store_true")
    p.add_argument("--transition-sensitivity", type=float, default=1.0)

    # runtime / GPU performance
    p.add_argument("--chunk-size", type=int, default=50000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-resident-input", action="store_true", default=True, help="Keep ESN input tensors resident on CUDA when available.")
    p.add_argument("--cpu-stream-input", action="store_false", dest="gpu_resident_input", help="Disable persistent CUDA input tensors and stream chunks instead.")
    p.add_argument("--store-recon", action="store_true", help="Store full reconstruction samples; off by default to save memory/time.")

    # attribution
    p.add_argument("--attribution-topk", type=int, default=30)
    p.add_argument("--no-attribution", action="store_true", help="Skip representative attribution model for faster full runs.")

    # baselines
    p.add_argument("--baseline-train-subsample", type=int, default=200000)
    p.add_argument("--pcaq-retain", type=float, default=0.98)
    p.add_argument("--iforest-estimators", type=int, default=300)

    # LSTM
    p.add_argument("--ae-win", type=int, default=128)
    p.add_argument("--ae-stride", type=int, default=10)
    p.add_argument("--ae-hidden", type=int, default=128)
    p.add_argument("--ae-layers", type=int, default=2)
    p.add_argument("--ae-dropout", type=float, default=0.0)
    p.add_argument("--ae-epochs", type=int, default=3)
    p.add_argument("--ae-batch", type=int, default=64)
    p.add_argument("--ae-max-train-windows", type=int, default=6000)
    p.add_argument("--ae-lr", type=float, default=3e-4)

    # optional operational logs
    p.add_argument("--ops-log", type=str, default=None)
    p.add_argument("--ops-start-col", type=str, default="start_time")
    p.add_argument("--ops-end-col", type=str, default="end_time")

    # optional expert labels (index-based)
    p.add_argument("--expert-labels", type=str, default=None)
    p.add_argument("--expert-start-col", type=str, default="start_idx")
    p.add_argument("--expert-end-col", type=str, default="end_idx")
    p.add_argument("--expert-label-col", type=str, default="label")

    # proxy validation
    p.add_argument("--proxy-col", type=str, default=None)
    p.add_argument("--proxy-baseline-s", type=float, default=30.0)
    p.add_argument("--proxy-ratio-thr", type=float, default=0.85)
    p.add_argument("--proxy-abs-quantile", type=float, default=0.005)
    p.add_argument("--proxy-min-dwell-s", type=float, default=0.5)
    p.add_argument("--proxy-merge-gap-s", type=float, default=0.25)

    # synthetic injection
    p.add_argument("--inject-segment-max-samples", type=int, default=250000)
    p.add_argument("--inject-warmup-samples", type=int, default=5000)
    p.add_argument("--inject-n", type=int, default=10)
    p.add_argument("--inject-duration-s", type=float, default=2.0)
    p.add_argument("--inject-random-channels", type=int, default=5)

    return p.parse_args()


def maybe_load_ops_log(path: Optional[str], start_col: str, end_col: str) -> List[Tuple[float, float]]:
    if not path:
        return []
    df = pd.read_csv(path)
    if start_col not in df.columns or end_col not in df.columns:
        return []
    s = pd.to_datetime(df[start_col], errors="coerce", utc=True)
    e = pd.to_datetime(df[end_col], errors="coerce", utc=True)
    if s.isna().all() or e.isna().all():
        return []
    t0 = s.iloc[0]
    starts = (s - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    ends = (e - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    out = []
    for a, b in zip(starts, ends):
        if np.isfinite(a) and np.isfinite(b) and b >= a:
            out.append((float(a), float(b)))
    return out


def ops_overlap(events: List[Event], ops: List[Tuple[float, float]]) -> Dict[str, float]:
    if not ops:
        return {}
    hits = 0
    for ev in events:
        if ev.start_time_s is None or ev.end_time_s is None:
            continue
        for a, b in ops:
            if not (ev.end_time_s < a or ev.start_time_s > b):
                hits += 1
                break
    return {"ops_hit_fraction": float(hits / max(1, len(events)))}


def load_expert_events(path: Optional[str], start_col: str, end_col: str, label_col: str) -> List[Tuple[int, int, int, float]]:
    if not path:
        return []
    df = pd.read_csv(path)
    for c in [start_col, end_col, label_col]:
        if c not in df.columns:
            return []
    df = df[df[label_col].astype(int) == 1].copy()
    if df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        s = int(max(0, pd.to_numeric(r[start_col])))
        e = int(max(s, pd.to_numeric(r[end_col])))
        p = (s + e) // 2
        out.append((s, e, p, float("nan")))
    return out


def validate_against_expert(events: List[Event], expert_events: List[Tuple[int, int, int, float]], fs: float, out_dir: Path) -> Dict[str, Any]:
    if not expert_events:
        return {"expert_validation_available": False}
    det = [(e.start_idx, e.end_idx - 1, e.peak_idx, e.peak_score) for e in events]
    overlap_count = 0
    gt_hit = np.zeros((len(expert_events),), dtype=bool)
    delays_start = []
    delays_peak = []
    rows = []
    for i, de in enumerate(det):
        hit_idx = None
        for gi, ge in enumerate(expert_events):
            if overlap((de[0], de[1]), (ge[0], ge[1])):
                hit_idx = gi
                break
        if hit_idx is not None:
            overlap_count += 1
            gt_hit[hit_idx] = True
            delays_start.append((de[0] - expert_events[hit_idx][0]) / fs)
            delays_peak.append((de[2] - expert_events[hit_idx][2]) / fs)
        rows.append({
            "det_i": i, "hit": int(hit_idx is not None), "matched_gt_i": -1 if hit_idx is None else int(hit_idx),
            "delay_start_s": np.nan if hit_idx is None else (de[0] - expert_events[hit_idx][0]) / fs,
            "delay_peak_s": np.nan if hit_idx is None else (de[2] - expert_events[hit_idx][2]) / fs,
        })
    pd.DataFrame(rows).to_csv(out_dir / "expert_det_event_matches.csv", index=False)
    rec, prec, event_auc = event_pr_curve(det, expert_events)
    if len(rec):
        pd.DataFrame({"recall": rec, "precision": prec}).to_csv(out_dir / "expert_event_pr_curve.csv", index=False)
    metrics = {
        "expert_validation_available": True,
        "expert_events": int(len(expert_events)),
        "detected_events": int(len(det)),
        "event_precision_vs_expert": float(overlap_count / max(1, len(det))),
        "event_recall_vs_expert": float(int(gt_hit.sum()) / max(1, len(expert_events))),
        "event_pr_auc_vs_expert": float(event_auc),
        "mean_delay_start_s": float(np.mean(delays_start) if delays_start else float("nan")),
        "median_delay_start_s": float(np.median(delays_start) if delays_start else float("nan")),
        "mean_delay_peak_s": float(np.mean(delays_peak) if delays_peak else float("nan")),
        "median_delay_peak_s": float(np.median(delays_peak) if delays_peak else float("nan")),
    }
    with open(out_dir / "expert_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    if getattr(args, "no_attribution", False):
        args.attribution_topk = 0
    set_seed(args.seed)
    device = choose_device(args.device)

    run_dir = Path(args.out_dir) / f"{args.experiment_name}_{now_ts()}"
    ensure_dir(run_dir)
    print(f"[out] {run_dir.resolve()}")
    print(f"[device] {device}  {cuda_info_string(device)}")

    # load data
    df = pd.read_csv(args.data, low_memory=False, nrows=args.max_rows)
    if args.time_col not in df.columns:
        print(f"[warn] time column '{args.time_col}' not found; falling back to index axis for plotting.")
    tsec = parse_time_seconds(df[args.time_col]) if args.time_col in df.columns else None
    fs = estimate_fs_hz(tsec, fallback=DEFAULT_BOOSTER_FS)
    print(f"[data] rows={len(df):,} cols={len(df.columns)} fs≈{fs:.4f}Hz")

    T = len(df)
    train_end = int(T * args.train_frac)
    val_end = int(T * (args.train_frac + args.val_frac))
    train_end = min(max(train_end, 1), T)
    val_end = min(max(val_end, train_end + 1), T)
    print(f"[split] train={train_end:,} val={val_end-train_end:,} test={T-val_end:,}")

    # save channel summary / drift
    channel_summary = build_channel_summary(df, train_end, val_end)
    channel_summary.to_csv(run_dir / "channel_summary.csv", index=False)
    plot_split_shift_heatmap(run_dir, channel_summary, top_k=25)

    pcfg = PreprocessConfig(
        time_col=args.time_col,
        enable_physics_features=not args.no_physics,
        booster_cycle_harmonics=args.harmonics,
        gmps_feature_k=args.gmps_k,
        beam_feature_k=args.beam_k,
        rolling_window_s=args.rolling_window_s,
        iqr_floor=args.iqr_floor,
        pca_retain=args.pca_retain,
    )
    esn_cfg = ESNConfig(
        n_reservoir=args.n_reservoir,
        spectral_radius=args.spectral_radius,
        input_scaling=args.input_scaling,
        leak_rate=args.leak_rate,
        sparsity=args.sparsity,
        ridge_alpha=args.ridge_alpha,
        washout=args.washout,
        lags=tuple(args.lags),
        weights=tuple(args.weights),
        include_input_in_state=args.include_input_in_state,
        accum_batch_size=args.accum_batch,
    )
    score_cfg = ScoreConfig(mode=args.score_mode, huber_delta=args.huber_delta, residual_cap=args.residual_cap)
    thr_cfg = ThresholdConfig(
        method=args.threshold_method,
        mad_factor=args.mad_factor,
        quantile=args.quantile,
        calibrate_on=args.calibrate_on,
        target_events_per_hour=args.target_events_per_hour,
        smooth_s=args.smooth_s,
        min_dwell_s=args.min_dwell_s,
        merge_gap_s=args.merge_gap_s,
        hysteresis=args.hysteresis,
        regime_aware=not args.no_regime_aware,
        transition_sensitivity=args.transition_sensitivity,
    )

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "preprocess": asdict(pcfg),
            "esn": asdict(esn_cfg),
            "score": asdict(score_cfg),
            "threshold": asdict(thr_cfg),
            "args": vars(args),
            "device": str(device),
            "fs_hz": fs,
        }, f, indent=2)

    results: List[MethodResult] = []
    runtimes: List[Dict[str, Any]] = []

    # main method
    main_res, main_runtime, feature_catalog, fe_main, X_full_main, phase_all_main = run_method_rrcesn(
        name="rrcesn_main",
        df=df,
        time_seconds=tsec,
        fs=fs,
        train_end=train_end,
        val_end=val_end,
        pcfg=pcfg,
        exclude_prefixes=args.score_exclude_prefix,
        esn_cfg=esn_cfg,
        score_cfg=score_cfg,
        thr_cfg=thr_cfg,
        device=device,
        seed=args.seed,
        chunk_size=args.chunk_size,
        out_dir=run_dir,
        n_ensemble=args.n_ensemble,
        temporal_scales=args.temporal_scales,
        agg=args.agg,
        attribution_topk=args.attribution_topk,
        gpu_resident_input=args.gpu_resident_input,
        store_recon=args.store_recon,
    )
    results.append(main_res)
    runtimes.append(main_runtime)

    # integrated validation
    proxy_metrics = validate_against_proxy(
        df=df,
        score=main_res.scores,
        alarms=main_res.alarms,
        fs=fs,
        out_dir=run_dir / "rrcesn_main",
        proxy_col=args.proxy_col,
        baseline_s=args.proxy_baseline_s,
        ratio_thr=args.proxy_ratio_thr,
        abs_quantile=args.proxy_abs_quantile,
        min_dwell_s=args.proxy_min_dwell_s,
        merge_gap_s=args.proxy_merge_gap_s,
    )
    main_res.extra.update(proxy_metrics)

    expert_events = load_expert_events(args.expert_labels, args.expert_start_col, args.expert_end_col, args.expert_label_col)
    expert_metrics = validate_against_expert(main_res.events, expert_events, fs, run_dir / "rrcesn_main")
    main_res.extra.update(expert_metrics)

    ops = maybe_load_ops_log(args.ops_log, args.ops_start_col, args.ops_end_col)
    main_res.extra.update(ops_overlap(main_res.events, ops))

    # baselines
    need_baselines = args.run_baselines or args.suite in ("baselines", "full")
    if need_baselines:
        X_train_sub = X_full_main[np.linspace(0, train_end - 1, num=min(args.baseline_train_subsample, train_end), dtype=int)] if train_end > 0 else X_full_main[:0]
        # PCA-Q
        if HAS_SKLEARN:
            pcaq_dir = run_dir / "pcaq"
            ensure_dir(pcaq_dir)
            t0 = time.perf_counter()
            pcaq = PCABaseline(retain=args.pcaq_retain, random_state=args.seed)
            pcaq.fit(X_train_sub)
            train_t = time.perf_counter() - t0
            t1 = time.perf_counter()
            s = pcaq.score(X_full_main)
            score_t = time.perf_counter() - t1
            res = run_single_baseline("pcaq", s, phase_all_main, tsec, fs, train_end, val_end, thr_cfg, pcaq_dir)
            res.extra.update(validate_against_proxy(df, res.scores, res.alarms, fs, pcaq_dir, proxy_col=args.proxy_col,
                                                   baseline_s=args.proxy_baseline_s, ratio_thr=args.proxy_ratio_thr,
                                                   abs_quantile=args.proxy_abs_quantile, min_dwell_s=args.proxy_min_dwell_s,
                                                   merge_gap_s=args.proxy_merge_gap_s))
            res.extra.update(ops_overlap(res.events, ops))
            results.append(res)
            runtimes.append({
                "method": "pcaq",
                "device": "cpu",
                "n_rows": int(len(df)),
                "preprocess_time_s": 0.0,
                "train_time_s": float(train_t),
                "score_time_s": float(score_t),
                "score_ms_per_sample": float(1000.0 * score_t / max(1, len(df))),
                "fits_15hz_budget": bool((1000.0 * score_t / max(1, len(df))) < (1000.0 / DEFAULT_BOOSTER_FS)),
            })

            # Isolation Forest
            if_dir = run_dir / "iforest"
            ensure_dir(if_dir)
            t0 = time.perf_counter()
            ifm = IFBaseline(n_estimators=args.iforest_estimators, random_state=args.seed)
            ifm.fit(X_train_sub)
            train_t = time.perf_counter() - t0
            t1 = time.perf_counter()
            s2 = ifm.score(X_full_main)
            score_t = time.perf_counter() - t1
            res2 = run_single_baseline("iforest", s2, phase_all_main, tsec, fs, train_end, val_end, thr_cfg, if_dir)
            res2.extra.update(validate_against_proxy(df, res2.scores, res2.alarms, fs, if_dir, proxy_col=args.proxy_col,
                                                    baseline_s=args.proxy_baseline_s, ratio_thr=args.proxy_ratio_thr,
                                                    abs_quantile=args.proxy_abs_quantile, min_dwell_s=args.proxy_min_dwell_s,
                                                    merge_gap_s=args.proxy_merge_gap_s))
            res2.extra.update(ops_overlap(res2.events, ops))
            results.append(res2)
            runtimes.append({
                "method": "iforest",
                "device": "cpu",
                "n_rows": int(len(df)),
                "preprocess_time_s": 0.0,
                "train_time_s": float(train_t),
                "score_time_s": float(score_t),
                "score_ms_per_sample": float(1000.0 * score_t / max(1, len(df))),
                "fits_15hz_budget": bool((1000.0 * score_t / max(1, len(df))) < (1000.0 / DEFAULT_BOOSTER_FS)),
            })

        # LSTM baseline
        lstm_dir = run_dir / "lstm_recon"
        ensure_dir(lstm_dir)
        t0 = time.perf_counter()
        model = train_lstm_baseline(
            X_train=X_full_main[:train_end],
            device=device,
            win=args.ae_win,
            hidden=args.ae_hidden,
            layers=args.ae_layers,
            dropout=args.ae_dropout,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch,
            max_train_windows=args.ae_max_train_windows,
            lr=args.ae_lr,
            seed=args.seed,
        )
        train_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        s3 = score_lstm_baseline(model, X_full_main, device=device, win=args.ae_win, stride=args.ae_stride, batch_size=args.ae_batch)
        score_t = time.perf_counter() - t1
        res3 = run_single_baseline("lstm_recon", s3, phase_all_main, tsec, fs, train_end, val_end, thr_cfg, lstm_dir)
        res3.extra.update(validate_against_proxy(df, res3.scores, res3.alarms, fs, lstm_dir, proxy_col=args.proxy_col,
                                                baseline_s=args.proxy_baseline_s, ratio_thr=args.proxy_ratio_thr,
                                                abs_quantile=args.proxy_abs_quantile, min_dwell_s=args.proxy_min_dwell_s,
                                                merge_gap_s=args.proxy_merge_gap_s))
        res3.extra.update(ops_overlap(res3.events, ops))
        results.append(res3)
        runtimes.append({
            "method": "lstm_recon",
            "device": str(device),
            "n_rows": int(len(df)),
            "preprocess_time_s": 0.0,
            "train_time_s": float(train_t),
            "score_time_s": float(score_t),
            "score_ms_per_sample": float(1000.0 * score_t / max(1, len(df))),
            "fits_15hz_budget": bool((1000.0 * score_t / max(1, len(df))) < (1000.0 / DEFAULT_BOOSTER_FS)),
        })

    # ablations
    if args.suite in ("ablations", "full"):
        # no physics
        ab_pcfg = PreprocessConfig(**asdict(pcfg))
        ab_pcfg.enable_physics_features = False
        res, rt, _, _, _, _ = run_method_rrcesn(
            name="ablation_no_physics",
            df=df, time_seconds=tsec, fs=fs, train_end=train_end, val_end=val_end,
            pcfg=ab_pcfg, exclude_prefixes=[], esn_cfg=esn_cfg, score_cfg=score_cfg, thr_cfg=thr_cfg,
            device=device, seed=args.seed, chunk_size=args.chunk_size, out_dir=run_dir,
            n_ensemble=args.n_ensemble, temporal_scales=args.temporal_scales, agg=args.agg,
            attribution_topk=min(10, args.attribution_topk),
        )
        results.append(res); runtimes.append(rt)

        # no score exclusion
        res, rt, _, _, _, _ = run_method_rrcesn(
            name="ablation_no_score_exclusion",
            df=df, time_seconds=tsec, fs=fs, train_end=train_end, val_end=val_end,
            pcfg=pcfg, exclude_prefixes=[], esn_cfg=esn_cfg, score_cfg=score_cfg, thr_cfg=thr_cfg,
            device=device, seed=args.seed, chunk_size=args.chunk_size, out_dir=run_dir,
            n_ensemble=args.n_ensemble, temporal_scales=args.temporal_scales, agg=args.agg,
            attribution_topk=min(10, args.attribution_topk),
        )
        results.append(res); runtimes.append(rt)

        # no ensemble
        res, rt, _, _, _, _ = run_method_rrcesn(
            name="ablation_no_ensemble",
            df=df, time_seconds=tsec, fs=fs, train_end=train_end, val_end=val_end,
            pcfg=pcfg, exclude_prefixes=args.score_exclude_prefix, esn_cfg=esn_cfg, score_cfg=score_cfg, thr_cfg=thr_cfg,
            device=device, seed=args.seed, chunk_size=args.chunk_size, out_dir=run_dir,
            n_ensemble=1, temporal_scales=[1], agg=args.agg,
            attribution_topk=min(10, args.attribution_topk),
        )
        results.append(res); runtimes.append(rt)

        # MAD threshold variant
        thr_mad = ThresholdConfig(**asdict(thr_cfg))
        thr_mad.method = "mad"
        res, rt, _, _, _, _ = run_method_rrcesn(
            name="ablation_threshold_mad",
            df=df, time_seconds=tsec, fs=fs, train_end=train_end, val_end=val_end,
            pcfg=pcfg, exclude_prefixes=args.score_exclude_prefix, esn_cfg=esn_cfg, score_cfg=score_cfg, thr_cfg=thr_mad,
            device=device, seed=args.seed, chunk_size=args.chunk_size, out_dir=run_dir,
            n_ensemble=args.n_ensemble, temporal_scales=args.temporal_scales, agg=args.agg,
            attribution_topk=min(10, args.attribution_topk),
        )
        results.append(res); runtimes.append(rt)

        # single-lag variant
        single_cfg = ESNConfig(**asdict(esn_cfg))
        single_cfg.lags = (int(args.lags[0]),)
        single_cfg.weights = (1.0,)
        res, rt, _, _, _, _ = run_method_rrcesn(
            name="ablation_single_lag",
            df=df, time_seconds=tsec, fs=fs, train_end=train_end, val_end=val_end,
            pcfg=pcfg, exclude_prefixes=args.score_exclude_prefix, esn_cfg=single_cfg, score_cfg=score_cfg, thr_cfg=thr_cfg,
            device=device, seed=args.seed, chunk_size=args.chunk_size, out_dir=run_dir,
            n_ensemble=args.n_ensemble, temporal_scales=args.temporal_scales, agg=args.agg,
            attribution_topk=min(10, args.attribution_topk),
        )
        results.append(res); runtimes.append(rt)

    # synthetic injection eval
    if args.run_injection_eval or args.suite == "full":
        inj_dir = run_dir / "synthetic_injection"
        ensure_dir(inj_dir)
        icfg = InjectConfig(
            enabled=True,
            segment_max_samples=args.inject_segment_max_samples,
            warmup_samples=args.inject_warmup_samples,
            n_injections=args.inject_n,
            duration_s=args.inject_duration_s,
            random_channels=args.inject_random_channels,
            seed=args.seed + 1000,
        )
        test_start = val_end
        seg_start = max(0, test_start - icfg.warmup_samples)
        seg_end = min(T, seg_start + icfg.warmup_samples + icfg.segment_max_samples)
        X_seg = X_full_main[seg_start:seg_end]
        rng = np.random.RandomState(icfg.seed)
        X_inj, injections = inject_synthetic(X_seg, fs, icfg, rng)

        rep = RRCESN(cfg=esn_cfg, n_inputs=X_full_main.shape[1], out_idx=np.where(select_score_features(fe_main.feature_names_, args.score_exclude_prefix)[0])[0], device=device, seed=args.seed)
        rep.fit_streaming(X_full_main[:train_end], chunk_size=args.chunk_size)
        s_clean, _, _, _ = rep.score_streaming(X_seg, score_cfg, chunk_size=args.chunk_size)
        s_inj, _, _, _ = rep.score_streaming(X_inj, score_cfg, chunk_size=args.chunk_size)

        seg_train_end = min(icfg.warmup_samples, len(s_clean) // 2)
        seg_val_end = min(seg_train_end + max(1000, seg_train_end // 2), len(s_clean))
        alarms_clean, th_clean, summ_clean = detect_anomalies(s_clean, phase=None, train_end=seg_train_end, val_end=seg_val_end, fs=fs, cfg=thr_cfg)
        alarms_inj, th_inj, summ_inj = detect_anomalies(s_inj, phase=None, train_end=seg_train_end, val_end=seg_val_end, fs=fs, cfg=thr_cfg)

        shifted_inj = [(s - icfg.warmup_samples, e - icfg.warmup_samples, typ, ch) for (s, e, typ, ch) in injections if s >= icfg.warmup_samples]
        injm = injection_metrics(alarms_inj[icfg.warmup_samples:], shifted_inj, fs)
        with open(inj_dir / "synthetic_injection_report.json", "w") as f:
            json.dump({"injections": injections, "metrics": injm}, f, indent=2)

        xaxis = np.arange(len(s_inj), dtype=np.float64) / fs
        plot_score_series(inj_dir, xaxis, s_inj, th_inj, alarms_inj, title="synthetic_injection")
        print("[inject]", json.dumps(injm, indent=2))

    # seed sensitivity mini-study on main
    seed_rows = []
    for extra_seed in [args.seed, args.seed + 1, args.seed + 2]:
        tmp_cfg = ESNConfig(**asdict(esn_cfg))
        model = RRCESN(tmp_cfg, n_inputs=X_full_main.shape[1], out_idx=np.where(select_score_features(fe_main.feature_names_, args.score_exclude_prefix)[0])[0], device=device, seed=extra_seed)
        t0 = time.perf_counter()
        model.fit_streaming(X_full_main[:train_end], chunk_size=args.chunk_size)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        s, _, _, _ = model.score_streaming(X_full_main, score_cfg, chunk_size=args.chunk_size)
        sc_t = time.perf_counter() - t1
        alarms, _, summ = detect_anomalies(s, phase_all_main, train_end, val_end, fs, thr_cfg)
        seed_rows.append({
            "seed": extra_seed,
            "fit_time_s": fit_t,
            "score_time_s": sc_t,
            "events_per_hour": summ["events_per_hour"],
            "alarm_duty_cycle": summ["alarm_duty_cycle"],
        })
    pd.DataFrame(seed_rows).to_csv(run_dir / "seed_sensitivity.csv", index=False)

    # write summary tables
    summary_df = build_summary_table(results, main_name="rrcesn_main")
    summary_df.to_csv(run_dir / "summary_table.csv", index=False)
    runtime_df = pd.DataFrame(runtimes)
    runtime_df.to_csv(run_dir / "runtime_summary.csv", index=False)

    # human-readable top line
    print("\n=== summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== runtime ===")
    print(runtime_df.to_string(index=False))

    # save compact report
    report = {
        "run_dir": str(run_dir.resolve()),
        "main_method": "rrcesn_main",
        "summary_table_path": str((run_dir / "summary_table.csv").resolve()),
        "runtime_summary_path": str((run_dir / "runtime_summary.csv").resolve()),
        "main_proxy_metrics": proxy_metrics,
        "main_expert_metrics": expert_metrics,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[done] wrote all artifacts to: {run_dir.resolve()}")

