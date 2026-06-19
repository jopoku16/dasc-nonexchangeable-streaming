from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

from run_first_simulation import (
    adaptive_conformal,
    dasc,
    residuals_from_lag,
    rolling_conformal,
    simulate_stream,
    spectral_feature,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def lag_features(y, max_lag=6):
    rows = []
    targets = []
    times = []
    for t in range(max_lag, len(y)):
        rows.append([y[t - lag] for lag in range(1, max_lag + 1)])
        targets.append(y[t])
        times.append(t)
    return np.asarray(rows), np.asarray(targets), np.asarray(times)


def interval_score(y, lo, hi, alpha):
    width = hi - lo
    return width + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)


def make_model(method, seed):
    cv = BlockBootstrap(
        n_resamplings=30,
        length=80,
        overlapping=True,
        random_state=seed,
    )
    return TimeSeriesRegressor(
        estimator=LinearRegression(),
        method=method,
        cv=cv,
        agg_function="mean",
        random_state=seed,
    )


def run_seed(seed, alpha=0.1):
    data = simulate_stream(seed=seed)
    y = data["y"].to_numpy()
    X, target, times = lag_features(y)
    train_mask = times < 700
    test_mask = times >= 700
    X_train, y_train = X[train_mask], target[train_mask]
    X_test, y_test, t_test = X[test_mask], target[test_mask], times[test_mask]

    rows = []
    confidence_level = 1 - alpha
    configs = [("MAPIE_EnbPI", "enbpi")]
    for label, method in configs:
        model = make_model(method, seed)
        model.fit(X_train, y_train)

        for x_i, y_i, t_i in zip(X_test, y_test, t_test):
            x_i = x_i.reshape(1, -1)
            y_pred, intervals = model.predict(
                x_i,
                confidence_level=confidence_level,
                ensemble=False,
                allow_infinite_bounds=True,
            )
            lo = float(intervals[0, 0, 0])
            hi = float(intervals[0, 1, 0])
            miss = int(y_i < lo or y_i > hi)
            rows.append(
                {
                    "seed": seed,
                    "t": int(t_i),
                    "method": label,
                    "miss": miss,
                    "coverage": 1 - miss,
                    "width": hi - lo,
                    "interval_score": interval_score(float(y_i), lo, hi, alpha),
                    "prediction": float(y_pred[0]),
                    "lo": lo,
                    "hi": hi,
                }
            )
            model.update(
                x_i,
                np.asarray([y_i]),
                confidence_level=confidence_level,
                ensemble=False,
            )
    return pd.DataFrame(rows)


def run_agaci_seed(seed, alpha=0.1):
    data = simulate_stream(seed=seed)
    y = data["y"].to_numpy()
    residuals, _ = residuals_from_lag(y)
    gammas = np.asarray([0.0025, 0.005, 0.01, 0.02, 0.04])
    expert_alpha = np.full(len(gammas), alpha, dtype=float)
    expert_weights = np.full(len(gammas), 1 / len(gammas), dtype=float)
    eta = 0.02
    rows = []

    for t in range(700, len(y)):
        alpha_mix = float(np.clip(np.sum(expert_weights * expert_alpha), 0.01, 0.35))
        lo, hi, q, ncal = adaptive_conformal(y, residuals, t, alpha_mix, window=180)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(
            {
                "seed": seed,
                "t": t,
                "method": "AgACI_style",
                "miss": miss,
                "coverage": 1 - miss,
                "width": hi - lo,
                "interval_score": interval_score(float(y[t]), lo, hi, alpha),
                "prediction": float(y[t - 1]),
                "lo": float(lo),
                "hi": float(hi),
            }
        )

        losses = []
        misses = []
        for j, gamma in enumerate(gammas):
            lo_j, hi_j, _, _ = rolling_conformal(y, residuals, t, expert_alpha[j], window=180)
            miss_j = int(y[t] < lo_j or y[t] > hi_j)
            losses.append(interval_score(float(y[t]), lo_j, hi_j, alpha))
            misses.append(miss_j)
            expert_alpha[j] = float(np.clip(expert_alpha[j] + gamma * (alpha - miss_j), 0.01, 0.35))
        losses = np.asarray(losses)
        losses = losses - np.nanmin(losses)
        expert_weights = expert_weights * np.exp(-eta * losses)
        if not np.isfinite(expert_weights).all() or expert_weights.sum() == 0:
            expert_weights = np.full(len(gammas), 1 / len(gammas), dtype=float)
        else:
            expert_weights = expert_weights / expert_weights.sum()
    return pd.DataFrame(rows)


def run_dasc_seed(seed, alpha=0.1):
    data = simulate_stream(seed=seed)
    y = data["y"].to_numpy()
    residuals, _ = residuals_from_lag(y)
    features = np.vstack([spectral_feature(y, i) for i in range(len(y))])
    alpha_dasc = alpha
    rows = []
    for t in range(700, len(y)):
        lo, hi, q, neff, drift, m_t, alpha_eff = dasc(
            y,
            residuals,
            features,
            t,
            alpha_dasc,
            window=360,
            h=0.55,
            drift_lambda=0.45,
            m_min=80,
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(
            {
                "seed": seed,
                "t": t,
                "method": "DASC_same_window",
                "miss": miss,
                "coverage": 1 - miss,
                "width": hi - lo,
                "interval_score": interval_score(float(y[t]), lo, hi, alpha),
                "prediction": float(y[t - 1]),
                "lo": float(lo),
                "hi": float(hi),
            }
        )
        alpha_dasc = float(np.clip(alpha_dasc + 0.015 * (alpha - miss), 0.01, 0.35))
    return pd.DataFrame(rows)


def summarize(results):
    return (
        results.groupby("method")
        .agg(
            empirical_miscoverage=("miss", "mean"),
            empirical_coverage=("coverage", "mean"),
            avg_width=("width", "mean"),
            interval_score=("interval_score", "mean"),
        )
        .reset_index()
    )


def main():
    warnings.filterwarnings("ignore")
    frames = []
    for seed in range(10):
        frames.append(run_seed(seed))
        frames.append(run_agaci_seed(seed))
        frames.append(run_dasc_seed(seed))
    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    results.to_csv(OUT / "external_mapie_results.csv", index=False)
    summary.to_csv(OUT / "external_mapie_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
