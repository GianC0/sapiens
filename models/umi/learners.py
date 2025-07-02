"""
Module definitions for the UMI.
This module contains implementations for:
1. Supervised InfoNCE loss
2. Stock-level factor learning
3. Market-level factor learning
4. Forecasting model

"""


import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

###############################################################################
# Utility losses & metrics                                                    #
###############################################################################


class SupervisedInfoNCE(nn.Module):
    """Supervised InfoNCE (user‑provided)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        sim = features @ features.t() / self.temperature
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.size(0), device=mask.device))
        neg_mask = 1.0 - mask
        pos_add = neg_mask * -1000.0
        neg_add = pos_mask * -1000.0
        per_ex = (sim * pos_mask + pos_add).logsumexp(-1) - (sim * neg_mask + neg_add).logsumexp(-1)
        return -per_ex.mean()


def rank_ic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pr = pred.argsort(dim=1).argsort(dim=1).float()
    tr = target.argsort(dim=1).argsort(dim=1).float()
    pr = pr - pr.mean(1, keepdim=True)
    tr = tr - tr.mean(1, keepdim=True)
    ic = (pr * tr).mean(1) / (pr.std(1) * tr.std(1) + 1e-6)
    return ic.mean()

###############################################################################
# Stock‑level factor (Eqs. 8‑14)                                              #
###############################################################################

class StockLevelFactorLearning(nn.Module):
    def __init__(self, num_stocks: int, lambda_ic: float = 0.1):
        super().__init__()
        self.I = num_stocks
        self.attn_weights = nn.Parameter(torch.zeros(num_stocks, num_stocks))
        self.beta = nn.Parameter(torch.randn(num_stocks, num_stocks) * 0.01)
        self._rho = nn.Parameter(torch.zeros(1))
        self.lambda_ic = lambda_ic

    @property
    def rho(self):
        return torch.tanh(self._rho)  # ensures ρ ∈ (‑1,1)

    def forward(self, prices_seq: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # prices_seq : (B, L+1, I)
        B, L1, I = prices_seq.shape
        assert I == self.I
        beta = self.beta * (~torch.eye(I, device=prices_seq.device).bool()).float()
        attn = self.attn_weights.masked_fill(torch.eye(I, device=prices_seq.device).bool(), -1e9)
        attn = F.softmax(attn, 1)
        virt = prices_seq @ beta.t()            # (B,L+1,I)
        u_seq = virt @ attn.t()                 # (B,L+1,I)

        # ─── Masking out delisted/removed stocks ─────────────────────────────────────
        if active_mask is not None:
            m = active_mask.unsqueeze(1).unsqueeze(1)     # (B,1,1,I)
            prices_seq = prices_seq * m
            virt       = virt       * m
            u_seq      = u_seq      * m


        # ─── Losses ─────────────────────────────────────────────────────────────────
        loss_beta = F.mse_loss(prices_seq, virt.detach())
        loss_station = F.mse_loss(u_seq[:, 1:], self.rho * u_seq[:, :-1])
        loss = loss_beta + self.lambda_ic * loss_station
        return u_seq, loss, {"loss_beta": loss_beta.detach(), "loss_station": loss_station.detach()}

###############################################################################
# Market‑level factor (Eqs. 16‑24)                                            #
###############################################################################

class MarketLevelFactorLearning(nn.Module):
    """
    • Dynamic stock representation r_t  (Eq. 16-17)
    • Market vector m_t                 (Eq. 18-19)
    • Sub-market contrastive loss       (Eq. 20-21)
    • Synchronism predictor             (Eq. 22-24) – labels built with Eq. 15
    """

    def __init__(
        self,
        num_stocks: int,
        feature_dim: int,
        window_L: int = 5,
        lambda_sync: float = 1.0,
        temperature: float = 0.07,
        sync_threshold: float = 0.6,      # θ in Eq. 15
    ):
        super().__init__()
        self.I = num_stocks
        self.F = feature_dim
        self.out_dim = 2 * feature_dim      # because of concatenation
        self.L = window_L
        self.sync_threshold = sync_threshold            # threshold for events synchronism 

        # shared map W_s : ℝ^F → ℝ^F
        self.W_s = nn.Linear(feature_dim, feature_dim, bias=True)

        # Eq. 18   W_ι : ℝ^I → ℝ^{2F}
        self.W_iota = nn.Linear(num_stocks, self.out_dim, bias=False)
        self.w_eta  = nn.Parameter(torch.randn(self.out_dim))

        # objectives
        self.info_nce  = SupervisedInfoNCE(temperature)
        self.sync_cls  = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, 3),
        )
        self.lambda_sync = lambda_sync

    @staticmethod
    def _weighted_mean(eta: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Eq. 19   η-weighted market vector."""
        return torch.einsum("bi,bid->bd", eta, r) / (eta.sum(1, keepdim=True) + 1e-6)

    def _compute_m_t(
        self,
        e_window: torch.Tensor,   # (B,Lτ,I,F)
        stockID_b: torch.Tensor,    # (B,I,I) one-hot matrix
        active_mask: Optional[torch.Tensor] = None,  # (B,I) mask for active stocks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns (r_t , m_t) for a given window ending at τ.
            r_t : (B,I,2F)   — dynamic stock embedding
            m_t : (B,2F)     — market vector
            """
            B, Lτ, I, f = e_window.shape
            

            if active_mask is not None:
                # zero out dead columns before any linear or attention op
                m = active_mask.unsqueeze(1).unsqueeze(-1)        # (B,1,I,1)
                e_window = e_window * m
            e_t   = e_window[:, -1]                     # (B,I,F)
            k_win = self.W_s(e_window)                  # (B,Lτ,I,F)
            q_t   = self.W_s(e_t)                       # (B,I,F)

            scores = torch.einsum("bif,blif->bli", q_t, k_win) / math.sqrt(f)
            ATT    = scores.softmax(1)                  # (B,Lτ,I)

            r_hist = torch.einsum("bli,blif->bif", ATT, e_window)  # (B,I,F)
            r_t    = torch.cat([e_t, r_hist], -1)                  # (B,I,2F)

            eta_in = self.W_iota(stockID_b) + r_t                    # (B,I,2F)
            eta    = F.relu(torch.einsum("bid,d->bi", eta_in, self.w_eta))

            m_t    = self._weighted_mean(eta, r_t)                 # (B,2F)
            return r_t, m_t, eta

    def forward(
        self,
        e_seq: torch.Tensor,      # (B,T,I,F)
        stockID: torch.Tensor,    # (I,I) identity matrix
        active_mask: Optional[torch.Tensor] = None,  # (B,I) mask for active stocks
        close_idx: int = 3,       # OHLCV index from 0 to
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns
            r_t : (B,I,2F)  – current dynamic stock embeddings
            m_t : (B,2F)    – current market vector
            loss_total
            log-dict
        """
        B, T, I, f = e_seq.shape
        assert I == self.I and f == self.F

        # expand stockID to batch
        stockID_b = stockID.unsqueeze(0).expand(B, -1, -1)      # (B,I,I)

        # ---------------- current period (t = T-1) ----------------
        L_cur  = min(self.L, T)
        r_t, m_t, _ = self._compute_m_t(e_seq[:, -L_cur:], stockID_b, active_mask)  # (B,I,2F) , (B,2F)

        # ---------------- contrastive InfoNCE (Eq. 20-21) ---------
        m1_list, m2_list, period_ids = [], [], []
        for τ in range(T):
            # split S1 / S2  (same for the whole batch, random per τ)
            idx = torch.randperm(I, device=e_seq.device)
            S1, S2 = idx[: I // 2], idx[I // 2 :]

            Lτ = min(self.L, τ + 1)
            r_τ, m_τ, eta_τ= self._compute_m_t(e_seq[:, τ - Lτ + 1 : τ + 1], stockID_b, active_mask)  # (B,I,2F), (B,2F), (B,I,2F)
            m1 = self._weighted_mean(eta_τ[:, S1], r_τ[:, S1])  # (B,2F)
            m2 = self._weighted_mean(eta_τ[:, S2], r_τ[:, S2])  # (B,2F)
            m1_list.append(m1) ; m2_list.append(m2)
            period_ids.append(torch.full((B,), τ, device=e_seq.device, dtype=torch.long))

        feats   = torch.cat(m1_list + m2_list, 0)                         # (2B*T,2F)
        labels  = torch.cat(period_ids * 2,      0)                       # (2B*T)
        loss_contrast = self.info_nce(feats, labels)

        # ---------------- synchronism labels (Eq. 15) ----------------
        close = e_seq[..., close_idx]                      # (B,T,I)
        if active_mask is not None:
            close = close * active_mask.unsqueeze(1)
        ret   = torch.log(close[:, 1:] / (close[:, :-1] + 1e-8))  # (B,T-1,I)
        pos_ratio = (ret > 0).float().mean(-1)             # (B,T-1)
        neg_ratio = (ret < 0).float().mean(-1)             # (B,T-1)
        sync_lbls = torch.where(
            pos_ratio >= self.sync_threshold, 0,
            torch.where(neg_ratio >= self.sync_threshold, 1, 2)
        )                                                  # (B,T-1)

        # compute m_{t-1} for every t
        m_prev = []
        for t in range(1, T):
            Lτ = min(self.L, t)
            _, m_prev_t, _ = self._compute_m_t(e_seq[:, t - Lτ : t], stockID_b)
            m_prev.append(m_prev_t)
        m_prev = torch.stack(m_prev, 1)                    # (B,T-1,2F)

        logits_sync = self.sync_cls(m_prev)                # (B,T-1,3)
        loss_sync   = F.cross_entropy(
            logits_sync.reshape(-1, 3),
            sync_lbls.reshape(-1),
        )

        loss_total = loss_contrast + self.lambda_sync * loss_sync
        return r_t, m_t, loss_total, {
            "loss_contrast": loss_contrast.detach(),
            "loss_sync":     loss_sync.detach(),
        }


###############################################################################
# Forecasting model                                                           #
###############################################################################


class ForecastingLearning(nn.Module):
    """
    Inputs
    -------
    e_seq   : (B, L, I, F)    raw features
    u_seq   : (B, L, I)       irrationality factor from stock module
    r_t     : (B, I, 2F)      dynamic stock embeddings from market module
    m_t     : (B, 2F)         market vector from market module
    stockID : (I, I)          one-hot identity matrix

    Output
    ------
    pred    : (B, I)          next-period stock return
    """

    def __init__(
        self,
        num_stocks: int,
        feature_dim: int,
        u_dim: int,
        W_iota: nn.Linear,                # shared projector from market block
        pred_len: int = 1,
        lambda_rankic: float = 0.1,
    ):
        super().__init__()
        self.I        = num_stocks
        self.F        = feature_dim
        self.D_enc    = feature_dim + u_dim        # encoder hidden size
        self.D_market = 2 * feature_dim            # r_t and m_t dim

        head_dim_target = 16          # ≈ sweet-spot per-head size
        max_heads        = 8          # don’t explode memory on small GPUs
        nhead = max(1, min(max_heads, self.D_enc // head_dim_target))
        while self.D_enc % nhead != 0 and nhead > 1:   # make it divide cleanly
            nhead -= 1
        self.nheads = nhead

        # ───── Transformer encoder over g_t = [e_t ∥ u_t]  (Eq. 25) ───── #
        self.pos_enc = PositionalEncoding(self.D_enc)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.D_enc, nhead=self.nheads, batch_first=True), num_layers=2, norm=nn.LayerNorm(self.D_enc))

        # ───── Stock-relation parameters  (Eq. 26) ───── #
        self.W_iota  = W_iota                       # shared weights
        self.W_gamma = nn.Linear(self.D_market, self.D_market, bias=False)

        # ───── Prediction head  (Eq. 28) ───── #
        mlp_in = 2 * self.D_enc + self.D_market     # c_t ∥ d_t ∥ m_t
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        self.pred_len        = pred_len
        self.lambda_rankic   = lambda_rankic

    # ------------------------------------------------------------------ #

    def forward(
        self,
        e_seq: torch.Tensor,      # (B,L,I,F)
        u_seq: torch.Tensor,      # (B,L,I)
        r_t:   torch.Tensor,      # (B,I,2F)
        m_t:   torch.Tensor,      # (B,2F)
        stockID: torch.Tensor,    # (I,I)
        target: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,  # (B,I) mask for active stocks
    ):
        B, L, I, _ = e_seq.shape
        assert I == self.I, "stock dimension mismatch"

        # ── 1. Encode g_t = [e_t ∥ u_t] ─────────────────────────────── #
        g = torch.cat([e_seq, u_seq.unsqueeze(-1)], -1)         # (B,L,I,D_enc)
        g_flat = g.reshape(B * I, L, -1)
        g_flat  = self.pos_enc(g_flat)                          # add positions
        enc_flat = self.encoder(g_flat)                         # (B·I,L,D_enc)
        enc = enc_flat.view(B, I, L, -1)                        # (B,I,L,D_enc)
        c_t   = enc[:, :, -1]                                   # (B,I,D_enc)
        c_prev = enc[:, :, -2] if L > 1 else c_t                # safety fallback

        # ── 2. Relation weights γ_{ij}  (Eq. 26) ───────────────────── #
        stockID_b = stockID.unsqueeze(0).expand(B, -1, -1)      # (B,I,I)
        h = self.W_gamma(r_t + self.W_iota(stockID_b))          # (B,I,2F)
        rel_scores = torch.einsum("bid,bjd->bij", h, h) / math.sqrt(self.D_market)
        gamma_ij   = rel_scores.softmax(dim=2)                  # softmax over j

        # ── 3. Dependency vector d_t   (Eq. 27) ────────────────────── #
        d_t = torch.einsum("bij,bjd->bid", gamma_ij, c_prev)    # (B,I,D_enc)

        # ── 4. Final prediction   (Eq. 28) ─────────────────────────── #
        m_expand = m_t.unsqueeze(1).expand(-1, I, -1)           # (B,I,2F)
        final_in = torch.cat([c_t, d_t, m_expand], -1)          # (B,I,mlp_in)
        out = self.mlp(final_in).squeeze(-1)                    # (B,I)

        # ── 5. Loss (MSE + λ·RankIC) ──────────────────────────────── #
        loss      = torch.tensor(0.0, device=e_seq.device)
        rank_loss = torch.tensor(0.0, device=e_seq.device)
        if target is not None:
            target = target.squeeze(-1)  # (B,I)
            if active_mask is not None:
                mask = active_mask.float()                         # (B,I)
                out = out * mask
                tgt_masked = target * mask
            mse   = F.mse_loss(out, target)
            rank_loss = 1 - rank_ic(out, target)
            loss = mse + self.lambda_rankic * rank_loss

        return out, loss, {"rank_loss": rank_loss.detach()}



################################################################################
# Helper class Positional Encoding for Transformer models
################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)  -> broadcastable
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)