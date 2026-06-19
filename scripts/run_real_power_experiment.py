from pathlib import Path

import numpy as np
import pandas as pd

from run_first_simulation import (
    dasc,
    exponential_weighted_conformal,
    pid_update,
    spectral_feature,
    spectral_only,
    weighted_quantile,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real" / "uci_household_power"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_hourly_power(max_hours=12000):
    txt = DATA_DIR / "household_power_consumption.txt"
    if not txt.exists():
        raise FileNotFoundError(
            f"Missing {txt}. Run scripts/download_real_data.py first."
        )

    df = pd.read_csv(
        txt,
        sep=";",
        na_values="?",
        usecols=["Date", "Time", "Global_active_power"],
        low_memory=False,
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.assign(datetime=dt)
    df = df.dropna(subset=["datetime", "Global_active_power"])
    df = df.set_index("datetime").sort_index()
    hourly = df["Global_active_power"].resample("h").mean().dropna()
    hourly = hourly.iloc[:max_hours]
    return hourly.reset_index().rename(columns={"Global_active_power": "y"})


def seasonal_residuals(y, lag=24):
    pred = np.full(len(y), np.nan)
    pred[lag:] = y[:-lag]
    residuals = np.abs(y - pred)
    return residuals, pred


def rolling_interval(y, residuals, pred, t, alpha, window=24 * 30):
    idx = np.arange(max(24, t - window), t)
    idx = idx[np.isfinite(residuals[idx])]
    q = np.quantile(residuals[idx], 1 - alpha, method="higher")
    return pred[t] - q, pred[t] + q, q, len(idx)


def adaptive_interval(y, residuals, pred, t, alpha_t, window=24 * 30):
    return rolling_interval(y, residuals, pred, t, alpha_t, window)


def run(alpha=0.1):
    data = load_hourly_power()
    y = data["y"].to_numpy(dtype=float)
    residuals, pred = seasonal_residuals(y)
    features = np.vstack([spectral_feature(y, i, window=24 * 7) for i in range(len(y))])

    alpha_aci = alpha
    alpha_pid = alpha
    alpha_dasc = alpha
    pid_state = {"integral": 0.0, "previous": 0.0}
    gamma = 0.015
    start = max(24 * 45, 800)
    rows = []

    for t in range(start, len(y)):
        if not np.isfinite(pred[t]):
            continue

        lo, hi, q, ncal = rolling_interval(y, residuals, pred, t, alpha)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "t": t,
            "datetime": data["datetime"].iloc[t],
            "method": "rolling",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha,
            "neff": ncal,
            "drift": np.nan,
        })

        lo, hi, q, ncal = adaptive_interval(y, residuals, pred, t, alpha_aci)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "t": t,
            "datetime": data["datetime"].iloc[t],
            "method": "adaptive",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha_aci,
            "neff": ncal,
            "drift": np.nan,
        })
        alpha_aci = float(np.clip(alpha_aci + gamma * (alpha - miss), 0.01, 0.35))

        lo, hi, q, ncal = rolling_interval(y, residuals, pred, t, alpha_pid)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "t": t,
            "datetime": data["datetime"].iloc[t],
            "method": "conformal_PID",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha_pid,
            "neff": ncal,
            "drift": np.nan,
        })
        alpha_pid = pid_update(alpha_pid, miss, alpha, pid_state)

        lo, hi, q, neff = exponential_weighted_conformal(
            y, residuals, t, alpha, window=24 * 60, decay=0.995
        )
        lo, hi = pred[t] - q, pred[t] + q
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "t": t,
            "datetime": data["datetime"].iloc[t],
            "method": "exp_weighted",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha,
            "alpha_eff": alpha,
            "neff": neff,
            "drift": np.nan,
            "window": 24 * 60,
        })

        lo, hi, q, neff = spectral_only(
            y, residuals, features, t, alpha, window=24 * 60, h=0.55
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "t": t,
            "datetime": data["datetime"].iloc[t],
            "method": "spectral_only",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha,
            "neff": neff,
            "drift": np.nan,
        })

        lo, hi, q, neff, drift, m_t, alpha_eff = dasc(
            y,
            residuals,
            features,
            t,
            alpha_dasc,
            window=24 * 60,
            h=0.55,
            drift_lambda=0.45,
            m_min=24 * 14,
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "t": t,
            "datetime": data["datetime"].iloc[t],
            "method": "DASC",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha_dasc,
            "alpha_eff": alpha_eff,
            "neff": neff,
            "drift": drift,
            "window": m_t,
        })
        alpha_dasc = float(np.clip(alpha_dasc + gamma * (alpha - miss), 0.01, 0.35))

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "real_power_results.csv", index=False)
    summarize(results).to_csv(OUT / "real_power_summary.csv", index=False)
    monthly_summary(results).to_csv(OUT / "real_power_monthly_summary.csv", index=False)
    data.to_csv(OUT / "real_power_hourly_series.csv", index=False)
    print(summarize(results).to_string(index=False))
    print()
    print(monthly_summary(results).head(20).to_string(index=False))


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
