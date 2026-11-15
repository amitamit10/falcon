"""Probability calibration and the historical confidence filter.

Raw model scores are not probabilities. We map them (isotonic regression, fitted on the EARLIER part of the out-of-sample history) to the
frequency that actually happened, check the mapping on the LATER part (a true hold-out), and only call a prediction 'high confidence' if its
bucket reached the required hit rate in that hold-out. If no bucket does, nothing is flagged: we never show confidence the data cannot back."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .forward import brier, ece

EDGES = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def reliability_table(p, y, edges=EDGES) -> pd.DataFrame:
    """For each probability bucket: how many predictions, the mean predicted probability, the actual hit rate and its 95% interval."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum()); l, h = wilson(k, n)
        rows.append({"bucket": f"{lo:.0%}-{hi:.0%}", "n": n, "mean_predicted": round(float(p[m].mean()), 4),
                     "actual_hit_rate": round(k / n, 4), "ci95_low": round(l, 4), "ci95_high": round(h, 4)})
    return pd.DataFrame(rows)


@dataclass
class Calibration:
    x: list = field(default_factory=list)          # isotonic mapping knots (raw score -> calibrated probability)
    y: list = field(default_factory=list)
    t_up: float | None = None                      # = min_hit when the hold-out confirmed 'up' calls at that level, else None
    t_down: float | None = None                    # = 1 - min_hit when the hold-out confirmed 'down' calls at that level, else None
    min_hit: float = 0.55
    stats: dict = field(default_factory=dict)
    table: list = field(default_factory=list)
    fitted_through: str = ""

    def apply(self, raw) -> np.ndarray:
        return np.interp(np.asarray(raw, float), self.x, self.y) if self.x else np.asarray(raw, float)

    def flag(self, p_cal) -> np.ndarray:
        """'up' / 'down' / '' per prediction, from the validated thresholds only."""
        p = np.asarray(p_cal, float)
        out = np.full(len(p), "", dtype=object)
        if self.t_up is not None:
            out[p >= self.t_up] = "up"
        if self.t_down is not None:
            out[p <= self.t_down] = "down"
        return out

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> "Calibration":
        return Calibration(**json.loads(s))


def _validate(p: np.ndarray, y: np.ndarray, side: str, min_hit: float, min_n: int, tol: float = 0.02):
    """A prediction is flagged only if its OWN calibrated probability reaches the target (>= min_hit for 'up', <= 1-min_hit for 'down'), and the
    hold-out confirms that bucket really hit about that rate (within `tol`) on at least min_n cases. Returns (threshold or None, n, hit rate)."""
    t = min_hit if side == "up" else 1.0 - min_hit
    m = p >= t if side == "up" else p <= t
    n = int(m.sum())
    if n == 0:
        return None, 0, float("nan")
    hit = float(y[m].mean()) if side == "up" else float(1.0 - y[m].mean())
    return (t if (n >= min_n and hit >= min_hit - tol) else None), n, hit


def fit_calibration(oos: pd.DataFrame, min_hit: float = 0.55, min_n: int = 300, holdout_frac: float = 0.4) -> Calibration:
    """oos: columns date, p (raw), y (0/1). Fit on the earlier part, validate on the later part by date."""
    d = oos.dropna(subset=["p", "y"]).sort_values("date")
    cut = d["date"].quantile(1 - holdout_frac)
    fit, hold = d[d["date"] <= cut], d[d["date"] > cut]
    if len(fit) < 1000 or len(hold) < 500:
        raise ValueError("not enough out-of-sample history to calibrate")
    xs, ys = _fit_mapping(fit["p"].to_numpy(float), fit["y"].to_numpy(float))
    cal = Calibration(x=xs.tolist(), y=ys.tolist(), min_hit=min_hit, fitted_through=str(d["date"].max().date()))
    ph = cal.apply(hold["p"]); yh = hold["y"].to_numpy(float)
    cal.t_up, n_up, hit_up = _validate(ph, yh, "up", min_hit, min_n)
    cal.t_down, n_dn, hit_dn = _validate(ph, yh, "down", min_hit, min_n)
    cal.table = reliability_table(ph, yh).to_dict("records")
    cal.stats = {"n_fit": int(len(fit)), "n_holdout": int(len(hold)), "base_rate_holdout": round(float(yh.mean()), 4),
                 "brier_raw": round(brier(hold["p"], yh), 5), "brier_calibrated": round(brier(ph, yh), 5),
                 "ece_raw": round(ece(hold["p"], yh), 4), "ece_calibrated": round(ece(ph, yh), 4),
                 "max_calibrated_p": round(float(ph.max()), 3), "min_calibrated_p": round(float(ph.min()), 3),
                 "holdout_up_n": n_up, "holdout_up_hit": None if n_up == 0 else round(hit_up, 4),
                 "holdout_down_n": n_dn, "holdout_down_hit": None if n_dn == 0 else round(hit_dn, 4)}
    return cal


def _fit_mapping(p: np.ndarray, y: np.ndarray, bins: int = 20, min_bin: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Monotone mapping raw score -> hit rate, fitted on WELL-POPULATED quantile buckets (isotonic over bucket means, weighted by size).
    Plain isotonic regression overfits the sparse extremes (a few lucky points claim 98%). Here no output can exceed what a whole bucket achieved,
    and scores beyond the outermost bucket are clipped to its level, never extrapolated."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ws = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_bin:
            xs.append(float(p[m].mean())); ys.append(float(y[m].mean())); ws.append(int(m.sum()))
    if len(xs) < 3:                      # scores carry no usable spread: the honest mapping is the base rate for everything
        return np.array([float(p.mean())]), np.array([float(y.mean())])
    iso = IsotonicRegression(y_min=0.02, y_max=0.98, increasing=True).fit(xs, ys, sample_weight=ws)
    xs_arr = np.array(xs)
    return xs_arr, iso.predict(xs_arr)


def save(cal: Calibration, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(cal.to_json())


def load(path: str | Path, max_age_days: int = 35) -> Calibration | None:
    p = Path(path)
    if not p.exists():
        return None
    age = (pd.Timestamp.today().normalize() - pd.Timestamp(p.stat().st_mtime, unit="s").normalize()).days
    return Calibration.from_json(p.read_text()) if age <= max_age_days else None
