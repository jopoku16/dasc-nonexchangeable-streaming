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
DATA_DIR = ROOT / "data" / "real" / "fred_sp500"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_sp500():
    path = DATA_DIR / "sp500_daily_fred.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_finance_data.py first.")
    df = pd.read_csv(path, parse_dates=["observation_date"]).sort_values("observation_date")
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")
    df = df.dropna(subset=["SP500"])
    close = df["SP500"].astype(float).to_numpy()
    returns = 100 * np.diff(np.log(close))
    abs_returns = np.abs(returns)
    out = df.iloc[1:][["observation_date"]].copy()
    out["y"] = abs_returns
    out = out.rename(columns={"observation_date": "datetime"})
    return out.dropna().reset_index(drop=True)


def lag_residuals(y):
    pred = np.r_[np.nan, y[:-1]]
    residuals = np.abs(y - pred)
    return residuals, pred


def rolling_interval(y, residuals, pred, t, alpha, window=252):
    idx = np.arange(max(1, t - window), t)
    idx = idx[np.isfinite(residuals[idx])]
    q = np.quantile(residuals[idx], 1 - alpha, method="higher")
    return max(0.0, pred[t] - q), pred[t] + q, q, len(idx)


def run(alpha=0.1):
    data = load_sp500()
    y = data["y"].to_numpy(dtype=float)
    residuals, pred = lag_residuals(y)
    features = np.vstack([spectral_feature(y, i, window=64) for i in range(len(y))])

    alpha_aci = alpha
    alpha_pid = alpha
    alpha_dasc = alpha
    pid_state = {"integral": 0.0, "previous": 0.0}
    gamma = 0.015
    start = 420
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

        _, _, q, neff = exponential_weighted_conformal(y, residuals, t, alpha, window=504, decay=0.992)
        lo, hi = max(0.0, pred[t] - q), pred[t] + q
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "exp_weighted", miss, hi - lo, q, alpha, neff, np.nan, 504))

        lo, hi, q, neff = spectral_only(y, residuals, features, t, alpha, window=504, h=0.55)
        lo = max(0.0, lo)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "spectral_only", miss, hi - lo, q, alpha, neff, np.nan, np.nan))

        lo, hi, q, neff, drift, m_t, alpha_eff = dasc(
            y,
            residuals,
            features,
            t,
            alpha_dasc,
            window=504,
            h=0.55,
            drift_lambda=0.45,
            m_min=126,
        )
        lo = max(0.0, lo)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append(row(data, t, "DASC", miss, hi - lo, q, alpha_dasc, neff, drift, m_t, alpha_eff))
        alpha_dasc = float(np.clip(alpha_dasc + gamma * (alpha - miss), 0.01, 0.35))

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "real_finance_results.csv", index=False)
    summarize(results).to_csv(OUT / "real_finance_summary.csv", index=False)
    year_summary(results).to_csv(OUT / "real_finance_year_summary.csv", index=False)
    data.to_csv(OUT / "real_finance_abs_returns.csv", index=False)
    print(summarize(results).to_string(index=False))
    print()
    print(year_summary(results).tail(30).to_string(index=False))


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


def year_summary(results):
    out = results.copy()
    out["year"] = pd.to_datetime(out["datetime"]).dt.year
    return (
        out.groupby(["method", "year"])
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
