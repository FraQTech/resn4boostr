from .common import *

# ============================================================
# baselines
# ============================================================

class PCABaseline:
    def __init__(self, retain: float = 0.98, random_state: int = 42):
        if not HAS_SKLEARN:
            raise RuntimeError("PCA baseline requires scikit-learn.")
        self.pca = PCA(n_components=retain, svd_solver="full", random_state=random_state)

    def fit(self, X: np.ndarray) -> None:
        self.pca.fit(X)

    def score(self, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        out = np.empty((len(X),), dtype=np.float32)
        for a in tqdm(range(0, len(X), chunk), desc="PCA-Q score", leave=False):
            b = min(len(X), a + chunk)
            Xr = self.pca.inverse_transform(self.pca.transform(X[a:b]))
            resid = X[a:b] - Xr
            out[a:b] = np.sum(resid * resid, axis=1).astype(np.float32)
        return out


class IFBaseline:
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        if not HAS_SKLEARN:
            raise RuntimeError("Isolation Forest baseline requires scikit-learn.")
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def score(self, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        out = np.empty((len(X),), dtype=np.float32)
        for a in tqdm(range(0, len(X), chunk), desc="iForest score", leave=False):
            b = min(len(X), a + chunk)
            out[a:b] = (-self.model.decision_function(X[a:b])).astype(np.float32)
        return out


class LSTMRecon(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.proj(h)


def sample_window_starts(n: int, win: int, count: int, rng: np.random.RandomState) -> np.ndarray:
    max_start = max(0, n - win)
    if max_start == 0:
        return np.zeros((count,), dtype=int)
    return rng.randint(0, max_start, size=count)


def train_lstm_baseline(
    X_train: np.ndarray,
    device: torch.device,
    win: int,
    hidden: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    max_train_windows: int,
    lr: float,
    seed: int,
) -> LSTMRecon:
    rng = np.random.RandomState(seed)
    model = LSTMRecon(d_in=X_train.shape[1], hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    starts = sample_window_starts(len(X_train), win, max_train_windows, rng)

    model.train()
    for ep in range(1, epochs + 1):
        rng.shuffle(starts)
        pbar = tqdm(range(0, len(starts), batch_size), desc=f"LSTM train ep{ep}/{epochs}", leave=False)
        for i in pbar:
            batch_starts = starts[i:i + batch_size]
            xb = np.stack([X_train[s:s + win] for s in batch_starts], axis=0)
            x = torch.from_numpy(xb).to(device, dtype=torch.float32, non_blocking=True)
            yhat = model(x)
            loss = loss_fn(yhat, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        pbar.close()
    return model


@torch.no_grad()
def score_lstm_baseline(model: LSTMRecon, X: np.ndarray, device: torch.device, win: int, stride: int, batch_size: int) -> np.ndarray:
    model.eval()
    n, _ = X.shape
    if n < win:
        x = torch.from_numpy(X[None, :, :]).to(device, dtype=torch.float32)
        y = model(x)
        err = torch.mean((y - x) ** 2, dim=(1, 2)).cpu().numpy()
        return np.full((n,), float(err[0]), dtype=np.float32)

    centers = np.arange(win // 2, n - (win - win // 2), stride, dtype=int)
    s_vals = np.full((centers.size,), np.nan, dtype=np.float32)
    for a in tqdm(range(0, len(centers), batch_size), desc="LSTM score", leave=False):
        b = min(len(centers), a + batch_size)
        c_batch = centers[a:b]
        starts = c_batch - win // 2
        xb = np.stack([X[s:s + win] for s in starts], axis=0)
        x = torch.from_numpy(xb).to(device, dtype=torch.float32, non_blocking=True)
        y = model(x)
        mse = torch.mean((y - x) ** 2, dim=(1, 2)).detach().cpu().numpy().astype(np.float32)
        s_vals[a:b] = mse

    full = np.full((n,), np.nan, dtype=np.float32)
    full[centers] = s_vals
    finite = np.isfinite(full)
    if not np.any(finite):
        return np.zeros((n,), dtype=np.float32)
    idx = np.arange(n)
    return np.interp(idx, idx[finite], full[finite]).astype(np.float32)

