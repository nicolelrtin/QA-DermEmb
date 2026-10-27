"""Patch linear models with an aggregator, trained as multiple-instance learning on image labels (Track P).

patch score s_k = w . x_k + b (P3: w_k per grid position), patch prob p_k = sigmoid(s_k).
Aggregators: mean / max / topk (mean of top-k p_k) / noisyor (1 - prod(1 - p_k)) /
attention (P(img) = sigmoid(sum_k a_k s_k), a = softmax(v . tanh(U x_k))).
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class PatchLinear(torch.nn.Module):
    def __init__(self, n_feat: int, n_pos: int, mcfg):
        super().__init__()
        self.agg, self.topk = mcfg.aggregator, mcfg.topk
        self.w = torch.nn.Parameter(torch.zeros(n_pos if mcfg.position_specific else 1, n_feat))
        self.b = torch.nn.Parameter(torch.zeros(1))
        if self.agg == "attention":
            self.U = torch.nn.Linear(n_feat, 16)
            self.v = torch.nn.Linear(16, 1, bias=False)

    def forward(self, x):  # x [N, K, Q] -> (image prob [N], patch scores [N, K], attention [N, K] or None)
        s = (x * self.w).sum(-1) + self.b
        if self.agg == "attention":
            a = torch.softmax(self.v(torch.tanh(self.U(x))).squeeze(-1), dim=1)
            return torch.sigmoid((a * s).sum(1)), s, a
        p = torch.sigmoid(s)
        if self.agg == "mean":
            P = p.mean(1)
        elif self.agg == "max":
            P = p.max(1).values
        elif self.agg == "topk":
            P = p.topk(min(self.topk, p.shape[1]), dim=1).values.mean(1)
        elif self.agg == "noisyor":
            P = 1 - torch.exp(torch.log1p(-p.clamp(max=1 - 1e-6)).sum(1))
        else:
            raise ValueError(f"Unknown aggregator {self.agg!r}")
        return P, s, None


def _auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


class MILModel:
    def __init__(self, mcfg, seed: int):
        self.mcfg, self.seed = mcfg, seed

    def _t(self, X):
        return torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)

    def fit(self, X, y, groups=None, X_val=None, y_val=None):
        m = self.mcfg
        torch.manual_seed(self.seed)
        N, K, Q = X.shape
        self.mu, self.sd = X.reshape(-1, Q).mean(0), X.reshape(-1, Q).std(0) + 1e-6
        self.net = PatchLinear(Q, K, m)
        opt = torch.optim.Adam(self.net.parameters(), lr=m.lr, weight_decay=m.weight_decay)
        Xt, yt = self._t(X), torch.tensor(y, dtype=torch.float32)
        wpos = (1 - y.mean()) / max(y.mean(), 1e-6)  # balanced class weights
        sw = torch.where(yt == 1, torch.tensor(float(wpos)), torch.tensor(1.0))
        has_val = X_val is not None and len(X_val) > 0
        Xv = self._t(X_val) if has_val else None
        best, best_state, wait, self.curves = float("inf"), None, 0, []
        for epoch in range(m.epochs):
            self.net.train()
            P, _, _ = self.net(Xt)
            loss = torch.nn.functional.binary_cross_entropy(P.clamp(1e-6, 1 - 1e-6), yt, weight=sw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            row = {"epoch": epoch, "train_loss": float(loss), "train_auroc": _auroc(y, P.detach().numpy())}
            if has_val:
                with torch.no_grad():
                    Pv = self.net(Xv)[0]
                    vl = float(torch.nn.functional.binary_cross_entropy(Pv.clamp(1e-6, 1 - 1e-6), torch.tensor(y_val, dtype=torch.float32)))
                row.update(val_loss=vl, val_auroc=_auroc(y_val, Pv.numpy()))
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    wait += 1
            self.curves.append(row)
            if has_val and wait >= m.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        w = self.net.w.detach().numpy()
        self.coef = w.mean(0)  # P3: averaged over positions; per-position weights in coef_by_position
        self.coef_by_position = w if m.position_specific else None
        self.params = {"aggregator": m.aggregator, "epochs_run": len(self.curves)}
        return self

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        self.net.eval()
        return self.net(self._t(X))[0].numpy()

    @torch.no_grad()
    def patch_outputs(self, X):
        self.net.eval()
        _, s, a = self.net(self._t(X))
        return s.numpy(), (a.numpy() if a is not None else None)
