from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def simulate_stream(n=1400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    freq = np.piecewise(
        t,
        [t < 450, (t >= 450) & (t < 900), t >= 900],
        [0.035, 0.055, lambda z: 0.055 + 0.00006 * (z - 900)],
    )
    amp = np.where((t // 180) % 2 == 0, 1.0, 1.7)
    level = np.where(t >= 900, 1.2, 0.0)
    sigma = np.where(t >= 900, 0.75, 0.45)
    signal = level + amp * np.sin(2 * np.pi * freq * t)
    y = signal + rng.normal(0, sigma)
    return pd.DataFrame({"t": t, "y": y, "signal": signal, "sigma": sigma, "freq": freq})


def lag_forecast(y, t, lag=1):
    return y[t - lag]


def residuals_from_lag(y):
    pred = np.r_[np.nan, y[:-1]]
    return np.abs(y - pred), pred


def spectral_feature(y, end, window=64):
    start = max(0, end - window)
    chunk = y[start:end]
    if len(chunk) < 8:
        return np.zeros(6)
    chunk = chunk - np.mean(chunk)
    power = np.abs(np.fft.rfft(chunk)) ** 2
    if len(power) < 7:
        return np.pad(power, (0, 7 - len(power)))[:6]
    bands = np.array_split(power[1:], 6)
    feature = np.array([b.mean() for b in bands])
    denom = np.linalg.norm(feature)
    return feature / denom if denom > 0 else feature


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cdf, q, side="left")]


def exponential_weighted_conformal(y, residuals, t, alpha, window=360, decay=0.985):
    idx = np.arange(max(1, t - window), t)
    ages = t - idx
    weights = decay ** ages
    weights = weights / weights.sum()
    q = weighted_quantile(residuals[idx], weights, 1 - alpha)
    pred = lag_forecast(y, t)
    neff = 1.0 / np.sum(weights**2)
    return pred - q, pred + q, q, neff


def rolling_conformal(y, residuals, t, alpha, window=180):
    idx = np.arange(max(1, t - window), t)
    q = np.quantile(residuals[idx], 1 - alpha, method="higher")
    pred = lag_forecast(y, t)
    return pred - q, pred + q, q, len(idx)


def adaptive_conformal(y, residuals, t, alpha_t, window=180):
    return rolling_conformal(y, residuals, t, alpha_t, window=window)


def spectral_only(y, residuals, features, t, alpha_t, window=360, h=0.55):
    current = features[t]
    pool = np.arange(max(1, t - window), t)
    distances = np.linalg.norm(features[pool] - current, axis=1)
    weights = np.exp(-(distances**2) / (h**2))
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    neff = 1.0 / np.sum(weights**2)
    q = weighted_quantile(residuals[pool], weights, 1 - alpha_t)
    pred = lag_forecast(y, t)
    return pred - q, pred + q, q, neff


def pid_update(alpha_t, error_t, target_alpha, state, kp=0.025, ki=0.003, kd=0.01):
    # Positive control signal means recent miscoverage is too low, so intervals can narrow.
    centered = target_alpha - error_t
    state["integral"] += centered
    derivative = centered - state["previous"]
    state["previous"] = centered
    update = kp * centered + ki * state["integral"] + kd * derivative
    return float(np.clip(alpha_t + update, 0.01, 0.35))


def dasc(
    y,
    residuals,
    features,
    t,
    alpha_t,
    window=360,
    h=0.55,
    drift_lambda=0.45,
    m_min=80,
    stability_relax=0.0,
):
    current = features[t]
    pool = np.arange(max(1, t - window), t)
    distances = np.linalg.norm(features[pool] - current, axis=1)
    spec_weights = np.exp(-(distances**2) / (h**2))
    if spec_weights.sum() == 0:
        spec_weights = np.ones_like(spec_weights)
    spec_weights = spec_weights / spec_weights.sum()

    recent = np.arange(max(1, t - 64), t)
    recent_feature = features[recent].mean(axis=0)
    weighted_feature = np.average(features[pool], axis=0, weights=spec_weights)
    drift = float(np.linalg.norm(recent_feature - weighted_feature))

    m_max = window
    m_t = int(m_max - (m_max - m_min) * min(1.0, drift / drift_lambda))
    gated = pool[pool >= t - m_t]
    if len(gated) < 20:
        gated = pool[-min(len(pool), 20):]

    distances = np.linalg.norm(features[gated] - current, axis=1)
    weights = np.exp(-(distances**2) / (h**2))
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    neff = 1.0 / np.sum(weights**2)

    stability = max(0.0, 1.0 - min(1.0, drift / drift_lambda))
    neff_factor = min(1.0, neff / 200.0)
    alpha_eff = float(np.clip(alpha_t + stability_relax * stability * neff_factor, 0.01, 0.35))

    q = weighted_quantile(residuals[gated], weights, 1 - alpha_eff)
    pred = lag_forecast(y, t)
    return pred - q, pred + q, q, neff, drift, m_t, alpha_eff


def run(
    seed=7,
    alpha=0.1,
    dasc_h=0.55,
    drift_lambda=0.45,
    dasc_m_min=80,
    stability_relax=0.0,
):
    data = simulate_stream(seed=seed)
    y = data["y"].to_numpy()
    residuals, pred = residuals_from_lag(y)
    features = np.vstack([spectral_feature(y, i) for i in range(len(y))])

    rows = []
    alpha_aci = alpha
    alpha_pid = alpha
    alpha_dasc = alpha
    gamma = 0.015
    pid_state = {"integral": 0.0, "previous": 0.0}
    start = 220

    for t in range(start, len(y)):
        lo, hi, q, ncal = rolling_conformal(y, residuals, t, alpha)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "rolling",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha,
            "neff": ncal,
            "drift": np.nan,
            "window": ncal,
        })

        lo, hi, q, ncal = adaptive_conformal(y, residuals, t, alpha_aci)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "adaptive",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha_aci,
            "neff": ncal,
            "drift": np.nan,
            "window": ncal,
        })
        alpha_aci = float(np.clip(alpha_aci + gamma * (alpha - miss), 0.01, 0.35))

        lo, hi, q, ncal = rolling_conformal(y, residuals, t, alpha_pid)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "conformal_PID",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha_pid,
            "alpha_eff": alpha_pid,
            "neff": ncal,
            "drift": np.nan,
            "window": ncal,
        })
        alpha_pid = pid_update(alpha_pid, miss, alpha, pid_state)

        lo, hi, q, neff = exponential_weighted_conformal(y, residuals, t, alpha, window=360)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "exp_weighted",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha,
            "alpha_eff": alpha,
            "neff": neff,
            "drift": np.nan,
            "window": 360,
        })

        lo, hi, q, neff = spectral_only(y, residuals, features, t, alpha)
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
            "method": "spectral_only",
            "miss": miss,
            "width": hi - lo,
            "q": q,
            "alpha_t": alpha,
            "alpha_eff": alpha,
            "neff": neff,
            "drift": np.nan,
            "window": 360,
        })

        lo, hi, q, neff, drift, m_t, alpha_eff = dasc(
            y,
            residuals,
            features,
            t,
            alpha_dasc,
            h=dasc_h,
            drift_lambda=drift_lambda,
            m_min=dasc_m_min,
            stability_relax=stability_relax,
        )
        miss = int(y[t] < lo or y[t] > hi)
        rows.append({
            "seed": seed,
            "t": t,
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

    return pd.DataFrame(rows), data


def summarize(results):
    late = results[results["t"] >= 220]
    summary = (
        late.groupby("method")
        .agg(
            empirical_miscoverage=("miss", "mean"),
            empirical_coverage=("miss", lambda x: 1 - x.mean()),
            avg_width=("width", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
        )
        .reset_index()
    )
    return summary


def add_regime_column(results):
    out = results.copy()
    out["regime"] = pd.cut(
        out["t"],
        bins=[219, 449, 899, 1400],
        labels=["recurring_A", "recurring_B", "drift_after_shift"],
    )
    return out


def regime_summary(results):
    with_regime = add_regime_column(results)
    return (
        with_regime.groupby(["method", "regime"], observed=False)
        .agg(
            empirical_miscoverage=("miss", "mean"),
            empirical_coverage=("miss", lambda x: 1 - x.mean()),
            avg_width=("width", "mean"),
            median_neff=("neff", "median"),
            avg_drift=("drift", "mean"),
            low_neff_rate=("neff", lambda x: (x < 80).mean()),
        )
        .reset_index()
    )


def main():
    all_results = []
    for seed in range(10):
        results, data = run(seed=seed)
        all_results.append(results)
        if seed == 0:
            data.to_csv(OUT / "synthetic_stream_seed0.csv", index=False)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT / "first_simulation_results.csv", index=False)
    summarize(combined).to_csv(OUT / "first_simulation_summary.csv", index=False)
    regime_summary(combined).to_csv(OUT / "first_simulation_regime_summary.csv", index=False)
    print(summarize(combined).to_string(index=False))
    print()
    print(regime_summary(combined).to_string(index=False))


if __name__ == "__main__":
    main()
