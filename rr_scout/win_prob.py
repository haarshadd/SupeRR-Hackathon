"""Win-probability model.

Two gradient-boosted models, one per innings, trained on every delivery state:
  innings 1: balls left, wickets lost, runs scored, current run rate, format
  innings 2: balls left, wickets lost, runs needed, required rate, format

Predictions are OUT-OF-FOLD: matches are split into N_FOLDS groups, and each match is scored by a
model that never saw it. That keeps a player's own match from leaking into the win probabilities
that are later used to judge him.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from . import config as C


def make_features(inn, runs, wkts, legal, target, t20i):
    runs, wkts, legal, t20i = (np.asarray(x, dtype=float) for x in (runs, wkts, legal, t20i))
    balls_left = C.BALLS_PER_INNINGS - legal
    if inn == 1:
        crr = runs / np.maximum(legal, 1) * 6
        return np.c_[balls_left, wkts, runs, crr, t20i]
    need = np.asarray(target, dtype=float) - runs
    rrr = need / np.maximum(balls_left, 1) * 6
    return np.c_[balls_left, wkts, need, rrr, t20i]


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
        fold = s.match_id.map(self.fold_of).values
        rows = []
        for inn in (1, 2):
            d = s[s.innings == inn]
            f = fold[(s.innings == inn).values]
            X = make_features(inn, d.runs_before, d.wkts_before, d.legal_before, d.target, d.t20i)
            y = d.bat_wins.values
            oof = np.zeros(len(d))
            for k in range(C.N_FOLDS):
                te = f == k
                mdl = HistGradientBoostingClassifier(random_state=C.RANDOM_SEED, **C.WP_PARAMS).fit(X[~te], y[~te])
                self.models[(inn, k)] = mdl
                oof[te] = mdl.predict_proba(X[te])[:, 1]
            rows.append(dict(innings=inn, rows=len(d), matches=d.match_id.nunique(),
                             auc=roc_auc_score(y, oof), logloss=log_loss(y, oof),
                             baseline_logloss=log_loss(y, np.full(len(y), y.mean()))))
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
                p[m] = self.models[(inn, k)].predict_proba(X[m])[:, 1]
        balls_left = C.BALLS_PER_INNINGS - legal
        if inn == 2:
            need = np.asarray(target, dtype=float) - runs
            p = np.where(need <= 0, 1.0, np.where((wkts >= 10) | (balls_left <= 0), 0.0, p))
        return p