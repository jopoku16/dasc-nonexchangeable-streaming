from pathlib import Path

import numpy as np
import pandas as pd

from run_first_simulation import (
    adaptive_conformal,
    dasc,
    residuals_from_lag,
    rolling_conformal,
    simulate_stream,
    spectral_feature,
    spectral_only,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def run(seed=0, alpha=0.1):
    data = simulate_stream(seed=seed)
    y = data["y"].to_numpy()
    residuals, _ = residuals_from_lag(y)
    features = np.vstack([spectral_feature(y, i) for i in range(len(y))])

    alpha_adapt = alpha
    alpha_dasc = alpha
    alpha_no_gate = alpha
    gamma = 0.015
    rows = []

    for t in range(220, len(y)):
        pred = y[t - 1]

        for method, fn in [
            ("rolling", lambda: rolling_conformal(y, residuals, t, alpha)),
            ("adaptive_only", lambda: adaptive_conformal(y, residuals, t, alpha_adapt)),
            ("spectral_only", lambda: spectral_only(y, residuals, features, t, alpha)),
        ]:
            lo, hi, q, neff = fn()
            miss = int(y[t] < lo or y[t] > hi)
            rows.append({
                "seed": seed,
                "t": t,
                "method": method,
                "miss": miss,
                "width": hi - lo,
                "neff": neff,
                "drift": np.nan,
            })
            if method == "adaptive_only":
                alpha_adapt = float(np.clip(alpha_adapt + gamma * (alpha - miss), 0.01, 0.35))

        lo, hi, q, neff, drift, m_t, _ = dasc(
            y, residuals, features, t, alpha_no_gate, window=360, h=0.55, drift_lambda=1e9, m_min=360
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "DASC_no_drift_gate",
            "miss": miss,
            "width": hi - lo,
            "neff": neff,
            "drift": drift,
        })
        alpha_no_gate = float(np.clip(alpha_no_gate + gamma * (alpha - miss), 0.01, 0.35))

        lo, hi, q, neff, drift, m_t, _ = dasc(
            y, residuals, features, t, alpha_dasc, window=360, h=0.55, drift_lambda=0.45, m_min=80
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "full_DASC",
            "miss": miss,
            "width": hi - lo,
            "neff": neff,
            "drift": drift,
        })
        alpha_dasc = float(np.clip(alpha_dasc + gamma * (alpha - miss), 0.01, 0.35))

    return pd.DataFrame(rows)


def summarize(df):
    df = df.copy()
    df["regime"] = pd.cut(
        df["t"],
        bins=[219, 449, 899, 1400],
        labels=["recurring_A", "recurring_B", "drift_after_shift"],
    )
    overall = (
        df.groupby("method")
        .agg(
            coverage=("miss", lambda x: 1 - x.mean()),
            miscoverage=("miss", "mean"),
            avg_width=("width", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
        )
        .reset_index()
    )
    by_regime = (
        df.groupby(["method", "regime"], observed=False)
        .agg(
            coverage=("miss", lambda x: 1 - x.mean()),
            avg_width=("width", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
        )
        .reset_index()
    )
    return overall, by_regime


def main():
    all_runs = [run(seed) for seed in range(10)]
    df = pd.concat(all_runs, ignore_index=True)
    overall, by_regime = summarize(df)
    df.to_csv(OUT / "ablation_results.csv", index=False)
    overall.to_csv(OUT / "ablation_summary.csv", index=False)
    by_regime.to_csv(OUT / "ablation_regime_summary.csv", index=False)
    print(overall.to_string(index=False))
    print()
    print(by_regime.to_string(index=False))


if __name__ == "__main__":
    main()
