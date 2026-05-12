from .common import *

# ============================================================
# RRC-ESN
# ============================================================

class RRCESN:
    """
    Full input dimensions condition the reservoir.
    Only out_idx dimensions are reconstructed and scored.
    """

    def __init__(self, cfg: ESNConfig, n_inputs: int, out_idx: np.ndarray, device: torch.device, seed: int):
        self.cfg = cfg
        self.n_inputs = int(n_inputs)
        self.out_idx = np.asarray(out_idx, dtype=np.int64)
        self.n_outputs = int(self.out_idx.size)
        if self.n_outputs <= 0:
            raise ValueError("out_idx is empty; nothing to score.")
        self.out_idx_t = torch.tensor(self.out_idx, device=device, dtype=torch.long)
        self.device = device
        self.seed = int(seed)
        self.N = int(cfg.n_reservoir)
        self.max_lag = int(max(cfg.lags))
        self.W_out: Dict[int, torch.Tensor] = {}
        self._init_weights()

        w = np.asarray(cfg.weights, dtype=np.float64)
        if w.size < len(cfg.lags):
            w = np.pad(w, (0, len(cfg.lags) - w.size), constant_values=float(w[-1] if w.size else 1.0))
        w = w[:len(cfg.lags)]
        w = w / (w.sum() + EPS)
        self.weights_np = w.astype(np.float32)

    def _init_weights(self) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)

        W_in = (torch.rand((self.N, self.n_inputs), generator=g) * 2.0 - 1.0) * float(self.cfg.input_scaling)
        W_res = (torch.rand((self.N, self.N), generator=g) * 2.0 - 1.0)
        mask = (torch.rand((self.N, self.N), generator=g) < float(self.cfg.sparsity)).to(W_res.dtype)
        W_res = W_res * mask

        # spectral radius scaling with power iteration
        v = torch.randn((self.N,), generator=g)
        for _ in range(30):
            v = torch.mv(W_res, v)
            n = torch.linalg.norm(v)
            if float(n) == 0.0:
                break
            v = v / n
        est = torch.linalg.norm(torch.mv(W_res, v))
        if float(est) > 1e-8:
            W_res = W_res * (float(self.cfg.spectral_radius) / float(est))

        self.W_in = W_in.to(self.device, dtype=torch.float32)
        self.W_res = W_res.to(self.device, dtype=torch.float32)
        self.bias = (torch.rand((self.N,), generator=g) * 2.0 - 1.0).to(self.device, dtype=torch.float32) * 0.01

    @property
    def zdim(self) -> int:
        return self.N + (self.n_inputs if self.cfg.include_input_in_state else 0) + 1

    @torch.no_grad()
    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pre = torch.mv(self.W_in, u) + torch.mv(self.W_res, x) + self.bias
        return (1.0 - float(self.cfg.leak_rate)) * x + float(self.cfg.leak_rate) * torch.tanh(pre)

    def _z(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.include_input_in_state:
            assert u is not None
            return torch.cat([x, u, torch.ones((1,), device=self.device, dtype=torch.float32)], dim=0)
        return torch.cat([x, torch.ones((1,), device=self.device, dtype=torch.float32)], dim=0)

    @torch.no_grad()
    def fit_streaming(self, U: Any, chunk_size: int = 50000, accum_batch_size: Optional[int] = None) -> None:
        """
        Streaming ESN training with GPU-resident input and batched normal-equation accumulation.

        The recurrence remains sequential in time, but the expensive ridge accumulators
        A += z z^T and B += z y^T are flushed as GEMMs over a buffer of states. This
        is substantially better for large GPUs than one outer product per sample.
        """
        U_t = as_device_tensor(U, self.device)
        accum_batch_size = int(accum_batch_size or getattr(self.cfg, "accum_batch_size", 4096))
        T, D = U_t.shape
        Z = self.zdim
        A = torch.zeros((Z, Z), device=self.device, dtype=torch.float32)
        B = {lag: torch.zeros((Z, self.n_outputs), device=self.device, dtype=torch.float32) for lag in self.cfg.lags}

        x = torch.zeros((self.N,), device=self.device, dtype=torch.float32)
        buf_len = self.max_lag + 1
        buf = torch.zeros((buf_len, D), device=self.device, dtype=torch.float32)
        ptr = 0

        z_buf: List[torch.Tensor] = []
        y_buf: Dict[int, List[torch.Tensor]] = {lag: [] for lag in self.cfg.lags}

        def flush_accum() -> None:
            if not z_buf:
                return
            Zb = torch.stack(z_buf, dim=0)  # (B, Z)
            A.add_(Zb.T @ Zb)
            for lag in self.cfg.lags:
                Yb = torch.stack(y_buf[lag], dim=0)  # (B, n_outputs)
                B[lag].add_(Zb.T @ Yb)
            z_buf.clear()
            for lag in self.cfg.lags:
                y_buf[lag].clear()

        pbar = tqdm(total=T, desc="Train (streaming ridge)", leave=True)
        for a in range(0, T, chunk_size):
            b = min(T, a + chunk_size)
            Uc = U_t[a:b]
            for i in range(Uc.shape[0]):
                t = a + i
                u = Uc[i]
                x = self.step(x, u)
                buf[ptr] = u
                ptr = (ptr + 1) % buf_len
                if t < max(self.cfg.washout, self.max_lag):
                    pbar.update(1)
                    continue
                z = self._z(x, u)
                z_buf.append(z)
                for lag in self.cfg.lags:
                    lag_ptr = (ptr - lag) % buf_len
                    y = buf[lag_ptr].index_select(0, self.out_idx_t)
                    y_buf[lag].append(y)
                if len(z_buf) >= int(accum_batch_size):
                    flush_accum()
                pbar.update(1)
        pbar.close()
        flush_accum()

        lam = float(self.cfg.ridge_alpha)
        I = torch.eye(Z, device=self.device, dtype=torch.float32)
        M = A + lam * I
        for lag in self.cfg.lags:
            self.W_out[lag] = torch.linalg.solve(M, B[lag])

    def _reduce_error(self, diff: torch.Tensor, scfg: ScoreConfig) -> torch.Tensor:
        if scfg.mode == "huber":
            d = float(scfg.huber_delta)
            absd = torch.abs(diff)
            quad = torch.minimum(absd, torch.tensor(d, device=diff.device))
            lin = absd - quad
            loss = 0.5 * quad * quad + d * lin
            return torch.mean(loss)
        if scfg.mode == "capped":
            cap = float(scfg.residual_cap)
            diff2 = diff * diff
            return torch.mean(torch.minimum(diff2, torch.tensor(cap * cap, device=diff.device)))
        return torch.mean(diff * diff)

    @torch.no_grad()
    def score_streaming(
        self,
        U: Any,
        scfg: ScoreConfig,
        chunk_size: int = 50000,
        store_feature_errors_for_indices: Optional[Sequence[int]] = None,
        store_recon: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, np.ndarray]], Optional[np.ndarray]]:
        if not self.W_out:
            raise RuntimeError("Model not trained.")
        U_t = as_device_tensor(U, self.device)
        T, D = U_t.shape
        scores = np.full((T,), np.nan, dtype=np.float32)
        uncert = np.full((T,), np.nan, dtype=np.float32)
        idx_set = set(store_feature_errors_for_indices or [])
        feat_err_at_idx: Optional[Dict[int, np.ndarray]] = {} if idx_set else None
        recon_sample = np.full((T, self.n_outputs), np.nan, dtype=np.float32) if store_recon else None

        x = torch.zeros((self.N,), device=self.device, dtype=torch.float32)
        buf_len = self.max_lag + 1
        buf = torch.zeros((buf_len, D), device=self.device, dtype=torch.float32)
        ptr = 0
        w = torch.tensor(self.weights_np, device=self.device, dtype=torch.float32)

        pbar = tqdm(total=T, desc="Score", leave=True)
        for a in range(0, T, chunk_size):
            b = min(T, a + chunk_size)
            Uc = U_t[a:b]
            for i in range(Uc.shape[0]):
                t = a + i
                u = Uc[i]
                x = self.step(x, u)
                buf[ptr] = u
                ptr = (ptr + 1) % buf_len
                if t < max(self.cfg.washout, self.max_lag):
                    pbar.update(1)
                    continue
                z = self._z(x, u)
                lag_losses = []
                per_feat = torch.zeros((self.n_outputs,), device=self.device, dtype=torch.float32) if (feat_err_at_idx is not None and t in idx_set) else None
                recon_accum = torch.zeros((self.n_outputs,), device=self.device, dtype=torch.float32) if store_recon else None

                for li, lag in enumerate(self.cfg.lags):
                    lag_ptr = (ptr - lag) % buf_len
                    y = buf[lag_ptr].index_select(0, self.out_idx_t)
                    y_hat = torch.mv(self.W_out[lag].T, z)
                    diff = y_hat - y
                    if per_feat is not None:
                        per_feat.add_(w[li] * (diff * diff))
                    if recon_accum is not None:
                        recon_accum.add_(w[li] * y_hat)
                    lag_losses.append(self._reduce_error(diff, scfg))

                mses = torch.stack(lag_losses)
                mu = torch.mean(mses)
                sd = torch.std(mses) if mses.numel() > 1 else torch.tensor(0.0, device=self.device)
                score = torch.sum(w[:mses.numel()] * mses)
                scores[t] = float(score)
                uncert[t] = float(sd / (mu + EPS)) if mses.numel() > 1 else 0.0
                if recon_sample is not None and recon_accum is not None:
                    recon_sample[t] = recon_accum.detach().cpu().numpy().astype(np.float32)

                if per_feat is not None and feat_err_at_idx is not None:
                    feat_err_at_idx[t] = per_feat.detach().cpu().numpy().astype(np.float32)
                pbar.update(1)
        pbar.close()
        return scores, uncert, feat_err_at_idx, recon_sample


def upsample_to_full(x: np.ndarray, full_len: int, scale: int) -> np.ndarray:
    if scale <= 1 or len(x) == full_len:
        return x.astype(np.float32, copy=False)
    t_ds = np.arange(0, full_len, scale, dtype=np.float64)
    if t_ds.size != len(x):
        t_ds = np.linspace(0, full_len - 1, num=len(x), dtype=np.float64)
    t_full = np.arange(full_len, dtype=np.float64)
    return np.interp(t_full, t_ds, x.astype(np.float64)).astype(np.float32)


def run_rrcesn_ensemble(
    U_full: np.ndarray,
    out_idx: np.ndarray,
    train_end: int,
    cfg: ESNConfig,
    scfg: ScoreConfig,
    n_ensemble: int,
    temporal_scales: Sequence[int],
    agg: str,
    device: torch.device,
    seed: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    member_scores: List[np.ndarray] = []
    member_unc: List[np.ndarray] = []
    scales = list(temporal_scales) if (n_ensemble > 1 and len(temporal_scales) > 0) else [1]

    for mi in range(int(n_ensemble)):
        scale = int(scales[mi % len(scales)])
        jitterN = (mi % 3 - 1) * 50
        jitterR = (mi % 5 - 2) * 0.01
        mcfg = ESNConfig(**asdict(cfg))
        mcfg.n_reservoir = max(200, int(mcfg.n_reservoir + jitterN))
        mcfg.spectral_radius = float(np.clip(mcfg.spectral_radius + jitterR, 0.7, 1.2))
        mseed = int(seed + mi * 7)

        if scale > 1:
            U_ds = U_full[::scale]
            train_end_ds = train_end // scale
        else:
            U_ds = U_full
            train_end_ds = train_end

        print(f"[rrcesn] member {mi+1}/{n_ensemble}  scale={scale}  seed={mseed}  N={mcfg.n_reservoir}  rho={mcfg.spectral_radius:.3f}")
        m = RRCESN(cfg=mcfg, n_inputs=U_ds.shape[1], out_idx=out_idx, device=device, seed=mseed)
        m.fit_streaming(U_ds[:train_end_ds], chunk_size=chunk_size)
        s_ds, u_ds, _, _ = m.score_streaming(U_ds, scfg=scfg, chunk_size=chunk_size)
        member_scores.append(upsample_to_full(s_ds, full_len=U_full.shape[0], scale=scale))
        member_unc.append(upsample_to_full(u_ds, full_len=U_full.shape[0], scale=scale))

    S = np.stack(member_scores, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        score = np.nanmedian(S, axis=0) if agg == "median" else np.nanmean(S, axis=0)
        if len(member_scores) > 1:
            meanS = np.nanmean(S, axis=0)
            stdS = np.nanstd(S, axis=0)
            unc = stdS / (meanS + EPS)
        else:
            unc = member_unc[0] if member_unc else np.zeros_like(score)
    return score.astype(np.float32), unc.astype(np.float32)

