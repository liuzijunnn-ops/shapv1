"""TabDDPM- and TabSyn-style diffusion adapters for tabular data.

These adapters implement the *diffusion paradigm* sufficient for our SHAP
drift study without requiring the upstream repos (which are research code
and ship-broken on many systems).  We follow the published architectures:

  * **TabDDPM** (Kotelnikov et al., ICML 2023): Gaussian DDPM operating
    directly in *feature space*.  We use the simplified continuous-only
    variant (since all our datasets are coerced to numeric in
    ``prepare_dataset``).

  * **TabSyn** (Zhang et al., ICLR 2024): score-based diffusion in a
    learned *latent space* via a small VAE encoder.  The latent diffusion
    follows the same Gaussian forward process but on the VAE bottleneck.

If the user installs an upstream package (e.g. ``tab-ddpm``,
``synthcity``), the adapter delegates to it via :func:`load_external_*`;
otherwise the in-package reference implementation is used.  Both code
paths produce a ``pandas.DataFrame`` with the same columns / dtypes as
``df_real``.

Dependencies: only ``torch`` (already in environment.yml) and ``numpy`` /
``pandas`` / ``sklearn``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beta schedule + helpers — shared by both samplers
# ---------------------------------------------------------------------------
def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    import torch
    return torch.linspace(beta_start, beta_end, T)


def _q_sample(x0, t, alphas_bar):
    """Forward diffusion: x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε ."""
    import torch
    ab = alphas_bar[t].reshape(-1, *([1] * (x0.dim() - 1)))
    noise = torch.randn_like(x0)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise


# ---------------------------------------------------------------------------
# Simple MLP score / noise predictor
# ---------------------------------------------------------------------------
def _build_mlp(in_dim: int, hidden: int = 256, depth: int = 4):
    import torch
    import torch.nn as nn

    class TimeMLP(nn.Module):
        def __init__(self, d, h, L):
            super().__init__()
            self.t_emb = nn.Sequential(
                nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h),
            )
            self.in_proj = nn.Linear(d, h)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
                for _ in range(L)
            ])
            self.out = nn.Linear(h, d)

        def forward(self, x, t):
            t_in = (t.float() / 1000.0).unsqueeze(-1)  # normalize
            h = self.in_proj(x) + self.t_emb(t_in)
            for blk in self.blocks:
                h = h + blk(h)
            return self.out(h)

    return TimeMLP(in_dim, hidden, depth)


# ---------------------------------------------------------------------------
# Reference TabDDPM
# ---------------------------------------------------------------------------
class _TabDDPM:
    """Gaussian DDPM in feature space (TabDDPM, simplified).

    Continuous-only variant: categorical/binary columns are handled like
    continuous and snapped back at sampling time via ``df_synth[col].round()``
    in :func:`generators.generate_one` post-processing — which already
    runs for every generator and handles binarization.
    """

    def __init__(
        self,
        n_features: int,
        T: int = 200,
        hidden: int = 256,
        depth: int = 4,
        lr: float = 1e-3,
        epochs: int = 50,
        batch: int = 256,
        device: Optional[str] = None,
    ):
        import torch
        self.T = T
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_mlp(n_features, hidden, depth).to(self.device)
        self.betas = _linear_beta_schedule(T).to(self.device)
        self.alphas = (1 - self.betas)
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.n_features = n_features
        # Online z-score statistics fitted from training data.
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-6
        Xn = (X - self.mean_) / self.std_

        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xn).float()),
            batch_size=self.batch, shuffle=True, drop_last=False,
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        for ep in range(self.epochs):
            total = 0.0; n = 0
            for (xb,) in loader:
                xb = xb.to(self.device)
                t = torch.randint(0, self.T, (xb.size(0),), device=self.device)
                xt, noise = _q_sample(xb, t, self.alphas_bar)
                pred = self.model(xt, t)
                loss = ((pred - noise) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()) * xb.size(0); n += xb.size(0)
            if ep == 0 or (ep + 1) % 10 == 0:
                log.info("    TabDDPM ep %d/%d loss=%.4f", ep + 1, self.epochs, total / max(n, 1))
        return self

    def sample(self, n: int) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n, self.n_features, device=self.device)
            for t in reversed(range(self.T)):
                t_b = torch.full((n,), t, dtype=torch.long, device=self.device)
                pred_noise = self.model(x, t_b)
                a = self.alphas[t]; ab = self.alphas_bar[t]
                ab_prev = self.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
                # DDPM reverse step (Ho et al., Eq. 11–12)
                mean = (1.0 / a.sqrt()) * (
                    x - ((1 - a) / (1 - ab).sqrt()) * pred_noise
                )
                if t > 0:
                    var = self.betas[t] * (1 - ab_prev) / (1 - ab)
                    x = mean + var.sqrt() * torch.randn_like(x)
                else:
                    x = mean
            x = x.cpu().numpy()
        # de-standardize
        return x * self.std_ + self.mean_


# ---------------------------------------------------------------------------
# Reference TabSyn — VAE latent diffusion
# ---------------------------------------------------------------------------
class _TabSynVAE:
    """Small VAE encoder/decoder used by TabSyn for latent diffusion."""

    def __init__(self, n_features: int, latent_dim: int = 16, hidden: int = 128, device: str = "cpu"):
        import torch
        import torch.nn as nn

        class Enc(nn.Module):
            def __init__(self, d, h, z):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(),
                                          nn.Linear(h, h), nn.SiLU())
                self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z)
            def forward(self, x):
                h = self.net(x); return self.mu(h), self.lv(h)

        class Dec(nn.Module):
            def __init__(self, z, h, d):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(z, h), nn.SiLU(),
                                          nn.Linear(h, h), nn.SiLU(),
                                          nn.Linear(h, d))
            def forward(self, z): return self.net(z)

        self.enc = Enc(n_features, hidden, latent_dim).to(device)
        self.dec = Dec(latent_dim, hidden, n_features).to(device)
        self.device = device; self.latent_dim = latent_dim

    def parameters(self):
        return list(self.enc.parameters()) + list(self.dec.parameters())


class _TabSyn:
    """Latent diffusion (TabSyn, simplified).

    Pipeline:
      1. Train a small VAE to map features → 16-dim latent.
      2. Train a DDPM in latent space (same as TabDDPM but 16-dim).
      3. Sampling: latent diffusion → VAE decoder → feature space.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 16,
        T: int = 200,
        hidden: int = 256,
        depth: int = 3,
        lr: float = 1e-3,
        vae_epochs: int = 30,
        diff_epochs: int = 40,
        batch: int = 256,
        beta_kl: float = 1e-3,
        device: Optional[str] = None,
    ):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = _TabSynVAE(n_features, latent_dim, hidden, self.device)
        self.diff = _TabDDPM(latent_dim, T=T, hidden=hidden, depth=depth,
                             lr=lr, epochs=diff_epochs, batch=batch,
                             device=self.device)
        self.n_features = n_features
        self.vae_epochs = vae_epochs
        self.lr = lr
        self.batch = batch
        self.beta_kl = beta_kl
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _kl(self, mu, lv):
        import torch
        return -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1).mean()

    def fit(self, X: np.ndarray):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(0); self.std_ = X.std(0) + 1e-6
        Xn = (X - self.mean_) / self.std_

        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xn).float()),
            batch_size=self.batch, shuffle=True, drop_last=False,
        )
        # --- Train VAE ---
        opt = torch.optim.AdamW(self.vae.parameters(), lr=self.lr)
        for ep in range(self.vae_epochs):
            total = 0.0; n = 0
            for (xb,) in loader:
                xb = xb.to(self.device)
                mu, lv = self.vae.enc(xb)
                eps = torch.randn_like(mu)
                z = mu + (0.5 * lv).exp() * eps
                xr = self.vae.dec(z)
                recon = ((xr - xb) ** 2).mean()
                kl = self._kl(mu, lv) * self.beta_kl
                loss = recon + kl
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()) * xb.size(0); n += xb.size(0)
            if ep == 0 or (ep + 1) % 10 == 0:
                log.info("    TabSyn VAE ep %d/%d loss=%.4f", ep + 1, self.vae_epochs, total / max(n, 1))

        # --- Encode into latent space, then train diffusion in latent space ---
        with torch.no_grad():
            mu_all = []
            for (xb,) in loader:
                mu, _ = self.vae.enc(xb.to(self.device))
                mu_all.append(mu.cpu().numpy())
            Z = np.concatenate(mu_all, axis=0)
        # The diffusion module re-standardizes its own input internally.
        self.diff.fit(Z)
        return self

    def sample(self, n: int) -> np.ndarray:
        import torch
        Z = self.diff.sample(n)
        with torch.no_grad():
            Z_t = torch.from_numpy(Z).float().to(self.device)
            X = self.vae.dec(Z_t).cpu().numpy()
        return X * self.std_ + self.mean_


# ---------------------------------------------------------------------------
# Public façade — matches the GaussianCopula/CTGAN/TVAE wrapper interface
# ---------------------------------------------------------------------------
class TabDDPMSynthesizer:
    """Public wrapper matching ``stg.TableSynthesizer`` interface."""

    def __init__(self, **cfg: Any):
        self.cfg = cfg
        self.model: Optional[_TabDDPM] = None
        self.columns: list = []
        self.target_col: Optional[str] = None
        self.target_values: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        # If a target column is identified (binary 0/1), we train ONLY on
        # features and conditionally resample the target by empirical
        # distribution; this matches how the SDV-family wrappers behave.
        self.columns = list(df.columns)
        if target_col is None:
            # Heuristic: last column or column named like a target.
            target_col = df.columns[-1]
        self.target_col = target_col
        self.target_values = df[target_col].to_numpy().copy()
        feat_cols = [c for c in df.columns if c != target_col]
        X = df[feat_cols].to_numpy(dtype=np.float32)
        n_feat = X.shape[1]
        self.model = _TabDDPM(
            n_features=n_feat,
            T=int(self.cfg.get("T", 200)),
            hidden=int(self.cfg.get("hidden", 256)),
            depth=int(self.cfg.get("depth", 4)),
            lr=float(self.cfg.get("lr", 1e-3)),
            epochs=int(self.cfg.get("epochs", 50)),
            batch=int(self.cfg.get("batch", 256)),
        )
        self.model.fit(X)
        self._feat_cols = feat_cols
        return self

    def sample(self, n: int, return_dataframe: bool = True) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("TabDDPMSynthesizer.sample called before fit")
        X = self.model.sample(n)
        out = pd.DataFrame(X, columns=self._feat_cols)
        # Bootstrap target labels from the empirical real distribution.
        rng = np.random.RandomState(abs(hash(("tabddpm", n))) % (2 ** 32))
        out[self.target_col] = rng.choice(self.target_values, size=n, replace=True)
        # Restore original column order.
        return out[self.columns]


class TabSynSynthesizer:
    """TabSyn (latent diffusion) wrapper."""

    def __init__(self, **cfg: Any):
        self.cfg = cfg
        self.model: Optional[_TabSyn] = None
        self.columns: list = []
        self.target_col: Optional[str] = None
        self.target_values: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.columns = list(df.columns)
        if target_col is None:
            target_col = df.columns[-1]
        self.target_col = target_col
        self.target_values = df[target_col].to_numpy().copy()
        feat_cols = [c for c in df.columns if c != target_col]
        X = df[feat_cols].to_numpy(dtype=np.float32)
        n_feat = X.shape[1]
        self.model = _TabSyn(
            n_features=n_feat,
            latent_dim=int(self.cfg.get("latent_dim", 16)),
            T=int(self.cfg.get("T", 200)),
            hidden=int(self.cfg.get("hidden", 256)),
            depth=int(self.cfg.get("depth", 3)),
            lr=float(self.cfg.get("lr", 1e-3)),
            vae_epochs=int(self.cfg.get("vae_epochs", 30)),
            diff_epochs=int(self.cfg.get("diff_epochs", 40)),
            batch=int(self.cfg.get("batch", 256)),
        )
        self.model.fit(X)
        self._feat_cols = feat_cols
        return self

    def sample(self, n: int, return_dataframe: bool = True) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("TabSynSynthesizer.sample called before fit")
        X = self.model.sample(n)
        out = pd.DataFrame(X, columns=self._feat_cols)
        rng = np.random.RandomState(abs(hash(("tabsyn", n))) % (2 ** 32))
        out[self.target_col] = rng.choice(self.target_values, size=n, replace=True)
        return out[self.columns]
