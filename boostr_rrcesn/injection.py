from .common import *

# ============================================================
# synthetic injection
# ============================================================

def inject_synthetic(X: np.ndarray, fs: float, cfg: InjectConfig, rng: np.random.RandomState) -> Tuple[np.ndarray, List[Tuple[int, int, str, List[int]]]]:
    n, d = X.shape
    X2 = X.copy()
    injections: List[Tuple[int, int, str, List[int]]] = []
    dur = max(1, int(cfg.duration_s * fs))
    for _ in range(int(cfg.n_injections)):
        start = rng.randint(0, max(1, n - dur))
        end = min(n, start + dur)
        typ = rng.choice(["spike", "drift", "dropout"])
        chans = rng.choice(np.arange(d), size=min(cfg.random_channels, d), replace=False).tolist()
        if typ == "spike":
            X2[start:end, chans] += cfg.spike_scale * np.std(X2[:, chans], axis=0, keepdims=True).astype(np.float32)
        elif typ == "drift":
            slope = (cfg.drift_scale * np.std(X2[:, chans], axis=0, keepdims=True).astype(np.float32)) / float(end - start + 1)
            ramp = np.arange(0, end - start, dtype=np.float32)[:, None]
            X2[start:end, chans] += ramp * slope
        else:
            if rng.rand() < cfg.dropout_prob:
                X2[start:end, chans] = 0.0
        injections.append((start, end, typ, chans))
    return X2, injections


def injection_metrics(alarms: np.ndarray, injections: List[Tuple[int, int, str, List[int]]], fs: float) -> Dict[str, float]:
    hits = 0
    delays = []
    for s, e, typ, ch in injections:
        seg = alarms[s:e]
        if np.any(seg):
            hits += 1
            first = int(np.argmax(seg))
            delays.append(first / fs)
    return {
        "inject_hit_rate": float(hits / max(1, len(injections))),
        "inject_mean_delay_s": float(np.mean(delays) if delays else float("nan")),
        "inject_median_delay_s": float(np.median(delays) if delays else float("nan")),
    }

