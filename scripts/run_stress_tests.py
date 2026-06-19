from pathlib import Path

import numpy as np
import pandas as pd

from run_first_simulation import (
    adaptive_conformal,
    dasc,
    exponential_weighted_conformal,
    pid_update,
    residuals_from_lag,
    rolling_conformal,
    spectral_feature,
    spectral_only,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def simulate_scenario(scenario, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base_freq = 0.04
    amp = 1.2 + 0.45 * ((t // 180) % 2)
    level = np.zeros(n)
    sigma = np.full(n, 0.45)
    freq = np.full(n, base_freq)
    noise = rng.normal(0, sigma)

    if scenario == "abrupt_shift":
        level[t >= 900] = 1.4
        sigma[t >= 900] = 0.8
        noise = rng.normal(0, sigma)
    elif scenario == "gradual_frequency":
        freq = base_freq + 0.00006 * np.maximum(0, t - 500)
    elif scenario == "heavy_tail":
        noise = 0.35 * rng.standard_t(df=3, size=n)
        sigma[:] = np.std(noise[:400])
        level[t >= 900] = 0.8
    elif scenario == "mixed_drift":
        level[t >= 850] = 1.0
        sigma[t >= 850] = 0.75
        freq = base_freq + 0.00004 * np.maximum(0, t - 700)
        noise = rng.normal(0, sigma)
    elif scenario == "weak_recurrence":
        amp = 0.25 + 0.05 * rng.normal(size=n)
        freq = 0.02 + 0.00003 * t
        level[t >= 850] = 0.8
        sigma[t >= 850] = 0.75
        noise = rng.normal(0, sigma)

    signal = level + amp * np.sin(2 * np.pi * freq * t)
    y = signal + noise
    return pd.DataFrame({"t": t, "y": y, "signal": signal, "sigma": sigma, "freq": freq})


def interval_score(y, lo, hi, alpha):
    width = hi - lo
    lower_penalty = (2 / alpha) * max(lo - y, 0)
    upper_penalty = (2 / alpha) * max(y - hi, 0)
    return width + lower_penalty + upper_penalty


def run_one(scenario, seed, alpha=0.1):
    data = simulate_scenario(scenario, seed=seed)
    y = data["y"].to_numpy()
    residuals, _ = residuals_from_lag(y)
    features = np.vstack([spectral_feature(y, i) for i in range(len(y))])

    alpha_aci = alpha
    alpha_pid = alpha
    alpha_dasc = alpha
    gamma = 0.015
    pid_state = {"integral": 0.0, "previous": 0.0}
    rows = []
    start = 240

    for t in range(start, len(y)):
        methods = []

        lo, hi, q, neff = rolling_conformal(y, residuals, t, alpha)
        methods.append(("rolling", lo, hi, q, alpha, neff, np.nan))

        lo, hi, q, neff = adaptive_conformal(y, residuals, t, alpha_aci)
        methods.append(("adaptive", lo, hi, q, alpha_aci, neff, np.nan))

        lo, hi, q, neff = rolling_conformal(y, residuals, t, alpha_pid)
        methods.append(("conformal_PID", lo, hi, q, alpha_pid, neff, np.nan))

        lo, hi, q, neff = exponential_weighted_conformal(y, residuals, t, alpha, window=360)
        methods.append(("exp_weighted", lo, hi, q, alpha, neff, np.nan))

        lo, hi, q, neff = spectral_only(y, residuals, features, t, alpha, window=360)
        methods.append(("spectral_only", lo, hi, q, alpha, neff, np.nan))

        lo, hi, q, neff, drift, _, alpha_eff = dasc(
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
        methods.append(("DASC", lo, hi, q, alpha_dasc, neff, drift))

        for method, lo, hi, q, alpha_t, neff, drift in methods:
            miss = int(y[t] < lo or y[t] > hi)
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "t": t,
                    "method": method,
                    "miss": miss,
                    "coverage": 1 - miss,
                    "width": hi - lo,
                    "interval_score": interval_score(y[t], lo, hi, alpha),
                    "alpha_t": alpha_t,
                    "neff": neff,
                    "drift": drift,
                }
            )
            if method == "adaptive":
                alpha_aci = float(np.clip(alpha_aci + gamma * (alpha - miss), 0.01, 0.35))
            elif method == "conformal_PID":
                alpha_pid = pid_update(alpha_pid, miss, alpha, pid_state)
            elif method == "DASC":
                alpha_dasc = float(np.clip(alpha_dasc + gamma * (alpha - miss), 0.01, 0.35))

    return pd.DataFrame(rows)


def summarize(results):
    return (
        results.groupby(["scenario", "method"])
        .agg(
            coverage=("coverage", "mean"),
            miscoverage=("miss", "mean"),
            avg_width=("width", "mean"),
            interval_score=("interval_score", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
        )
        .reset_index()
    )


def rank_summary(summary):
    out = summary.copy()
    out["coverage_error"] = (out["coverage"] - 0.90).abs()
    out["calibrated"] = out["coverage"].between(0.89, 0.91)
    out["score_rank"] = out.groupby("scenario")["interval_score"].rank(method="min")
    out["coverage_rank"] = out.groupby("scenario")["coverage_error"].rank(method="min")
    return out


def main():
    scenarios = [
        "abrupt_shift",
        "gradual_frequency",
        "heavy_tail",
        "mixed_drift",
        "weak_recurrence",
    ]
    frames = []
    for scenario in scenarios:
        for seed in range(8):
            frames.append(run_one(scenario, seed))
    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    ranked = rank_summary(summary)
    results.to_csv(OUT / "stress_test_results.csv", index=False)
    summary.to_csv(OUT / "stress_test_summary.csv", index=False)
    ranked.to_csv(OUT / "stress_test_rank_summary.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(ranked[ranked["method"] == "DASC"].to_string(index=False))


if __name__ == "__main__":
    main()
