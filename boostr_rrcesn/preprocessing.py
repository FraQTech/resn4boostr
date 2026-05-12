from .common import *

# ============================================================
# data / preprocessing
# ============================================================

class PhysicsFeatureEngineer:
    """
    Adds physics-augmented features, fits robust scaling on train,
    optionally keeps conditioning features excluded from score.
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.feature_names_: List[str] = []
        self._median: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._binary_mask: Optional[np.ndarray] = None
        self._pca: Any = None
        self._orig_numeric_cols: Optional[List[str]] = None
        self._feature_catalog: Optional[pd.DataFrame] = None

    def _cycle_phase(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        out = df.copy()
        if self.cfg.time_col not in out.columns:
            return out, None
        secs = parse_time_seconds(out[self.cfg.time_col])
        if secs is None:
            return out, None
        period = 1.0 / float(PHYSICS_CONFIG["booster_frequency_hz"])
        phase = (secs % period) / period * 2.0 * np.pi
        out["cycle_phase"] = phase.astype(np.float32)
        out["cycle_sin"] = np.sin(phase).astype(np.float32)
        out["cycle_cos"] = np.cos(phase).astype(np.float32)
        for k in range(2, int(self.cfg.booster_cycle_harmonics) + 1):
            out[f"cycle_h{k}"] = np.sin(k * phase).astype(np.float32)
            out[f"cycle_c{k}"] = np.cos(k * phase).astype(np.float32)

        out["reg_injection"] = (phase < (np.pi / 4.0)).astype(np.float32)
        out["reg_extraction"] = (phase >= (7.0 * np.pi / 4.0)).astype(np.float32)
        out["reg_acceleration"] = (1.0 - out["reg_injection"] - out["reg_extraction"]).clip(0, 1).astype(np.float32)
        return out, phase.astype(np.float32)

    def _gmps_features(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        out = df.copy()
        cols = [c for c in out.columns if ("gmps" in c.lower() or "vi" in c.lower()) and pd.api.types.is_numeric_dtype(out[c])]
        cols = cols[: int(self.cfg.gmps_feature_k)]
        if not cols:
            return out
        w = max(3, int(self.cfg.rolling_window_s * fs))
        for c in cols:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            x = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else 0.0).astype(np.float32)
            out[f"{c}_reg_std"] = moving_std(x, w)
            out[f"{c}_reg_absdiff_ma"] = moving_mean_absdiff(x, w)
            mu = moving_mean(x, max(5, 2 * w))
            out[f"{c}_setpoint_dev"] = np.abs(x - mu).astype(np.float32)
            out[f"{c}_trend"] = np.gradient(moving_mean(x, max(5, w))).astype(np.float32)
            if self.cfg.add_phase_cross_terms and "cycle_sin" in out.columns:
                out[f"{c}_x_sin"] = (x * out["cycle_sin"].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
                out[f"{c}_x_cos"] = (x * out["cycle_cos"].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
        return out

    def _beam_features(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        out = df.copy()
        cols = [c for c in out.columns if c.startswith("I:") and pd.api.types.is_numeric_dtype(out[c])]
        cols = cols[: int(self.cfg.beam_feature_k)]
        if not cols:
            return out
        w = max(5, int(self.cfg.rolling_window_s * fs))
        for c in cols:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            x = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else 0.0).astype(np.float32)
            env = np.sqrt(np.maximum(moving_mean((x.astype(np.float64) ** 2), w), 0.0)).astype(np.float32)
            out[f"{c}_envelope"] = env
            out[f"{c}_stability"] = moving_std(x, max(5, 2 * w))
            dx = np.diff(x, prepend=x[0])
            thr = -0.1 * (np.std(x) + 1e-8)
            out[f"{c}_drop_flag"] = (dx < thr).astype(np.float32)
            out[f"{c}_drop_mag"] = np.maximum(-dx, 0.0).astype(np.float32)
        return out

    def _numeric_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include=[np.number]).copy()

    def _robust_fit_scale(self, X: np.ndarray) -> np.ndarray:
        med = np.median(X, axis=0).astype(np.float32)
        q25 = np.percentile(X, 25, axis=0).astype(np.float32)
        q75 = np.percentile(X, 75, axis=0).astype(np.float32)
        iqr = np.maximum(q75 - q25, float(self.cfg.iqr_floor)).astype(np.float32)

        is_bin = np.zeros((X.shape[1],), dtype=bool)
        if self.cfg.treat_binary_as_unscaled:
            for j in range(X.shape[1]):
                col = X[: min(len(X), 200000), j]
                u = np.unique(col[np.isfinite(col)])
                if u.size <= 3 and np.all((u >= -1e-6) & (u <= 1 + 1e-6)):
                    is_bin[j] = True
            iqr[is_bin] = 1.0
            med[is_bin] = 0.0

        self._median = med
        self._scale = iqr
        self._binary_mask = is_bin
        Xs = ((X - med) / iqr).astype(np.float32, copy=False)
        return Xs

    def _robust_transform(self, X: np.ndarray) -> np.ndarray:
        assert self._median is not None and self._scale is not None
        return ((X - self._median) / self._scale).astype(np.float32, copy=False)

    def _build_feature_catalog(self, raw_cols: List[str], all_cols: List[str], score_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
        rows = []
        raw_set = set(raw_cols)
        for i, c in enumerate(all_cols):
            origin = "derived"
            if c in raw_set:
                origin = "raw"
            role = "conditioning_and_score"
            if score_mask is not None and not bool(score_mask[i]):
                role = "conditioning_only"
            rows.append({
                "feature": c,
                "origin": origin,
                "subsystem": infer_subsystem(c),
                "is_cycle_or_regime": int(c.startswith("cycle_") or c.startswith("reg_")),
                "role": role,
            })
        return pd.DataFrame(rows)

    def fit(self, df_train: pd.DataFrame, fs: float) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], List[str]]:
        tmp = df_train.copy()
        phase = None
        if self.cfg.enable_physics_features:
            tmp, phase = self._cycle_phase(tmp)
            tmp = self._gmps_features(tmp, fs=fs)
            tmp = self._beam_features(tmp, fs=fs)

        num_df = self._numeric_df(tmp)
        raw_numeric_cols = list(df_train.select_dtypes(include=[np.number]).columns)
        self._orig_numeric_cols = list(num_df.columns)
        X = num_df.to_numpy(dtype=np.float32, copy=True)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
        self.feature_names_ = list(num_df.columns)
        Xs = self._robust_fit_scale(X)

        if self.cfg.pca_retain < 1.0 and HAS_SKLEARN:
            self._pca = PCA(n_components=self.cfg.pca_retain, svd_solver="full", random_state=42)
            Xp = self._pca.fit_transform(Xs).astype(np.float32)
            feat_names = [f"pca_{i:03d}" for i in range(Xp.shape[1])]
            self.feature_names_ = feat_names
            self._feature_catalog = pd.DataFrame({
                "feature": feat_names,
                "origin": "pca",
                "subsystem": "mixed",
                "is_cycle_or_regime": 0,
                "role": "conditioning_and_score",
            })
            return Xp, feat_names, phase, raw_numeric_cols

        self._feature_catalog = self._build_feature_catalog(raw_numeric_cols, self.feature_names_)
        return Xs, self.feature_names_, phase, raw_numeric_cols

    def transform(self, df: pd.DataFrame, fs: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        tmp = df.copy()
        phase = None
        if self.cfg.enable_physics_features:
            tmp, phase = self._cycle_phase(tmp)
            tmp = self._gmps_features(tmp, fs=fs)
            tmp = self._beam_features(tmp, fs=fs)

        if self._pca is None:
            for c in self.feature_names_:
                if c not in tmp.columns:
                    tmp[c] = 0.0
            X = tmp[self.feature_names_].to_numpy(dtype=np.float32, copy=True)
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
            return self._robust_transform(X), phase

        assert self._orig_numeric_cols is not None
        for c in self._orig_numeric_cols:
            if c not in tmp.columns:
                tmp[c] = 0.0
        X = tmp[self._orig_numeric_cols].to_numpy(dtype=np.float32, copy=True)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
        Xs = self._robust_transform(X)
        Xp = self._pca.transform(Xs).astype(np.float32)
        return Xp, phase

    def feature_catalog(self, score_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
        if self._feature_catalog is None:
            return pd.DataFrame()
        df = self._feature_catalog.copy()
        if score_mask is not None and len(score_mask) == len(df):
            df["role"] = np.where(score_mask, "conditioning_and_score", "conditioning_only")
        return df


def select_score_features(feature_names: List[str], exclude_prefixes: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    excl = tuple(exclude_prefixes or [])
    mask = np.ones((len(feature_names),), dtype=bool)
    for i, n in enumerate(feature_names):
        for p in excl:
            if n.startswith(p):
                mask[i] = False
                break
    score_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    return mask, score_names


def build_channel_summary(df: pd.DataFrame, train_end: int, val_end: int) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    rows = []
    for c in num.columns:
        x = pd.to_numeric(num[c], errors="coerce").to_numpy(dtype=np.float64)
        def stats(y: np.ndarray, name: str) -> Dict[str, float]:
            y = y[np.isfinite(y)]
            if y.size == 0:
                return {f"{name}_mean": np.nan, f"{name}_std": np.nan, f"{name}_q01": np.nan, f"{name}_q50": np.nan, f"{name}_q99": np.nan}
            return {
                f"{name}_mean": float(np.mean(y)),
                f"{name}_std": float(np.std(y)),
                f"{name}_q01": float(np.quantile(y, 0.01)),
                f"{name}_q50": float(np.quantile(y, 0.50)),
                f"{name}_q99": float(np.quantile(y, 0.99)),
            }
        row = {"feature": c, "subsystem": infer_subsystem(c)}
        row.update(stats(x[:train_end], "train"))
        row.update(stats(x[train_end:val_end], "val"))
        row.update(stats(x[val_end:], "test"))
        row["train_test_mean_shift_std_units"] = (
            (row["test_mean"] - row["train_mean"]) / (row["train_std"] + 1e-12)
            if np.isfinite(row["train_mean"]) and np.isfinite(row["test_mean"]) else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(by="train_test_mean_shift_std_units", key=lambda s: np.abs(s), ascending=False)

