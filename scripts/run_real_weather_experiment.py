from pathlib import Path

import numpy as np
import pandas as pd

from run_first_simulation import (
    dasc,
    exponential_weighted_conformal,
    pid_update,
    spectral_feature,
    spectral_only,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real" / "open_meteo_dallas"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_weather(max_hours=24000):
    path = DATA_DIR / "open_meteo_dallas_hourly_temperature.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_weather_data.py first.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"temperature_2m": "y"})
    return df.iloc[:max_hours].copy()


def seasonal_residuals(y, lag=24):
    pred = np.full(len(y), np.nan)
    pred[lag:] = y[:-lag]
    residuals = np.abs(y - pred)
    return residuals, pred


def rolling_interval(y, residuals, pred, t, alpha, window=24 * 45):
    idx = np.arange(max(24, t - window), t)
    idx = idx[np.isfinite(residuals[idx])]
    q = np.quantile(residuals[idx], 1 - alpha, method="higher")
    return pred[t] - q, pred[t] + q, q, len(idx)


def run(alpha=0.1):
    data = load_weather()
    y = data["y"].to_numpy(dtype=float)
    residuals, pred = seasonal_residuals(y)
    features = np.vstack([spectral_feature(y, i, window=24 * 14) for i in range(len(y))])

    alpha_aci = alpha
    alpha_pid = alpha
    alpha_dasc = alpha
    pid_state = {"integral": 0.0, "previous": 0.0}
    gamma = 0.015
    start = max(24 * 60, 1200)
    rows = []

    for t in range(start, len(y)):
        if not np.isfinite(pred[t]):
            continue

        lo, hi, q, ncal = rolling_interval(y, residuals, pred, t, alpha)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "rolling", miss, hi - lo, q, alpha, ncal, np.nan, np.nan))

        lo, hi, q, ncal = rolling_interval(y, residuals, pred, t, alpha_aci)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "adaptive", miss, hi - lo, q, alpha_aci, ncal, np.nan, np.nan))
        alpha_aci = float(np.clip(alpha_aci + gamma * (alpha - miss), 0.01, 0.35))

        lo, hi, q, ncal = rolling_interval(y, residuals, pred, t, alpha_pid)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "conformal_PID", miss, hi - lo, q, alpha_pid, ncal, np.nan, np.nan))
        alpha_pid = pid_update(alpha_pid, miss, alpha, pid_state)

        _, _, q, neff = exponential_weighted_conformal(
            y, residuals, t, alpha, window=24 * 90, decay=0.996
        )
        lo, hi = pred[t] - q, pred[t] + q
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "exp_weighted", miss, hi - lo, q, alpha, neff, np.nan, 24 * 90))

        lo, hi, q, neff = spectral_only(y, residuals, features, t, alpha, window=24 * 90, h=0.55)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "spectral_only", miss, hi - lo, q, alpha, neff, np.nan, np.nan))

        lo, hi, q, neff, drift, m_t, alpha_eff = dasc(
            y,
            residuals,
            features,
            t,
            alpha_dasc,
            window=24 * 90,
            h=0.55,
            drift_lambda=0.45,
            m_min=24 * 21,
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "DASC", miss, hi - lo, q, alpha_dasc, neff, drift, m_t, alpha_eff))
        alpha_dasc = float(np.clip(alpha_dasc + gamma * (alpha - miss), 0.01, 0.35))

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "real_weather_results.csv", index=False)
    summarize(results).to_csv(OUT / "real_weather_summary.csv", index=False)
    monthly_summary(results).to_csv(OUT / "real_weather_monthly_summary.csv", index=False)
    print(summarize(results).to_string(index=False))


def row(data, t, method, miss, width, q, alpha_t, neff, drift, window, alpha_eff=None):
    return {
        "t": t,
        "datetime": data["datetime"].iloc[t],
        "method": method,
        "miss": miss,
        "width": width,
        "q": q,
        "alpha_t": alpha_t,
        "alpha_eff": alpha_t if alpha_eff is None else alpha_eff,
        "neff": neff,
        "drift": drift,
        "window": window,
    }


def summarize(results):
    return (
        results.groupby("method")
        .agg(
            empirical_miscoverage=("miss", "mean"),
            empirical_coverage=("miss", lambda x: 1 - x.mean()),
            avg_width=("width", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
        )
        .reset_index()
    )


def monthly_summary(results):
    out = results.copy()
    out["month"] = pd.to_datetime(out["datetime"]).dt.to_period("M").astype(str)
    return (
        out.groupby(["method", "month"])
        .agg(
            coverage=("miss", lambda x: 1 - x.mean()),
            avg_width=("width", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    run()
