"""Win-probability model (numpy + scipy only, no scikit-learn).

One ridge-regularised logistic regression per innings, on hand-built features (hinge/spline terms
in the required rate, runs needed, wickets and balls left, plus a few interactions):
  innings 1: balls left, wickets lost, runs scored, current run rate, format
  innings 2: balls left, wickets lost, runs needed, required rate, format

Out-of-fold, by match: matches are split into N_FOLDS groups and every match is scored by a model
that never saw it, so a player's own match cannot leak into the win probabilities used to judge him.
(In testing this beat a gradient-boosted model on held-out log-loss, and it is easy to explain.)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from . import config as C


def _hinge(x, knots):
    return np.column_stack([np.maximum(x - k, 0) for k in knots])


def make_features(inn, runs, wkts, legal, target, t20i):
    runs, wkts, legal, t20i = (np.asarray(x, dtype=float) for x in (runs, wkts, legal, t20i))
    bl = C.BALLS_PER_INNINGS - legal
    base = [bl / 120, (bl / 120) ** 2, wkts, wkts ** 2 / 10, t20i, wkts * bl / 120]
    if inn == 1:
        crr = np.clip(runs / np.maximum(legal, 1) * 6, 0, 15)
        cols = base + [crr, runs / 100, crr * bl / 120, runs * wkts / 500]
        return np.column_stack(cols + [_hinge(crr, [4, 6, 8, 10, 12]), _hinge(bl, [20, 40, 60, 80, 100]),
                                       _hinge(wkts, [2, 4, 6, 8])])
    need = np.clip(np.asarray(target, dtype=float) - runs, -50, 300)
    rrr = np.clip(need / np.maximum(bl, 1) * 6, -5, 40)
    cols = base + [rrr, np.log1p(np.maximum(rrr, 0)), need / 100, np.sqrt(np.maximum(need, 0)) / 10,
                   need * wkts / 1000, rrr * wkts / 10, rrr * bl / 120]
    H = _hinge(rrr, [2, 4, 6, 8, 10, 12, 14, 16, 20, 25])
    Hn = _hinge(need, [10, 30, 50, 80, 110, 140])
    return np.column_stack(cols + [H, Hn, H[:, [3, 5, 7]] * wkts[:, None] / 5])


def _fit_logit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    n = len(y)

    def loss(w):
        p = expit(Z @ w)
        ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + C.WP_L2 * np.sum(w[1:] ** 2) / n
        g = Z.T @ (p - y) / n
        g[1:] += 2 * C.WP_L2 * w[1:] / n
        return ll, g

    w = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=C.WP_MAXITER)).x
    return mu, sd, w


def _predict_logit(model, X):
    mu, sd, w = model
    return expit(np.c_[np.ones(len(X)), (X - mu) / sd] @ w)


def _auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def _logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class WinProb:
    def __init__(self):
        self.models = {}          # (innings, fold) -> fitted model
        self.fold_of = {}         # match_id -> fold
        self.metrics = None

    def fit(self, s):
        rng = np.random.RandomState(C.RANDOM_SEED)
        ids = np.array(sorted(s.match_id.unique()))
        rng.shuffle(ids)
        self.fold_of = {m: i % C.N_FOLDS for i, m in enumerate(ids)}
        rows = []
        for inn in (1, 2):
            d = s[s.innings == inn]
            f = d.match_id.map(self.fold_of).values
            X = make_features(inn, d.runs_before, d.wkts_before, d.legal_before, d.target, d.t20i)
            y = d.bat_wins.values.astype(float)
            oof = np.zeros(len(d))
            for k in range(C.N_FOLDS):
                te = f == k
                self.models[(inn, k)] = _fit_logit(X[~te], y[~te])
                oof[te] = _predict_logit(self.models[(inn, k)], X[te])
            rows.append(dict(innings=inn, rows=len(d), matches=d.match_id.nunique(), auc=_auc(y, oof),
                             logloss=_logloss(y, oof), baseline_logloss=_logloss(y, np.full(len(y), y.mean()))))
        self.metrics = pd.DataFrame(rows)
        return self

    def predict(self, inn, runs, wkts, legal, target, t20i, match_id):
        """Win probability for the BATTING team in the given states (arrays, one entry per state)."""
        runs, wkts, legal = np.asarray(runs), np.asarray(wkts), np.asarray(legal)
        fold = pd.Series(match_id).map(self.fold_of).values
        X = make_features(inn, runs, wkts, legal, target, t20i)
        p = np.zeros(len(X))
        for k in range(C.N_FOLDS):
            m = fold == k
            if m.any():
                p[m] = _predict_logit(self.models[(inn, k)], X[m])
        if inn == 2:
            need = np.asarray(target, dtype=float) - runs
            bl = C.BALLS_PER_INNINGS - legal
            p = np.where(need <= 0, 1.0, np.where((wkts >= 10) | (bl <= 0), 0.0, p))
        return p