from dataclasses import dataclass

import numpy as np


@dataclass
class DASCConfig:
    alpha: float = 0.10
    window: int = 360
    spectral_window: int = 64
    bandwidth: float = 0.55
    drift_lambda: float = 0.45
    m_min: int = 80
    m_max: int = 360
    gamma: float = 0.015
    alpha_min: float = 0.01
    alpha_max: float = 0.35


def spectral_feature(y, end, window=64, n_bands=6):
    start = max(0, end - window)
    chunk = np.asarray(y[start:end], dtype=float)
    if len(chunk) < 8:
        return np.zeros(n_bands)
    chunk = chunk - np.mean(chunk)
    power = np.abs(np.fft.rfft(chunk)) ** 2
    bands = np.array_split(power[1:], n_bands)
    feature = np.array([b.mean() if len(b) else 0.0 for b in bands])
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cdf, q, side="left")]


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return float(1.0 / np.sum(weights**2))


def spectral_weights(features, current_feature, bandwidth):
    distances = np.linalg.norm(features - current_feature, axis=1)
    weights = np.exp(-(distances**2) / (bandwidth**2))
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    return weights / weights.sum(), distances


def drift_gate_length(drift, config):
    shrink = min(1.0, drift / config.drift_lambda)
    return int(config.m_max - (config.m_max - config.m_min) * shrink)


class DASC:
    """Drift-Aware Spectral Conformal predictor for one-step streaming intervals."""

    def __init__(self, config=None):
        self.config = config or DASCConfig()
        self.alpha_t = self.config.alpha

    def reset(self):
        self.alpha_t = self.config.alpha

    def predict(self, y, residuals, features, t, point_prediction):
        cfg = self.config
        current = features[t]
        pool = np.arange(max(1, t - cfg.window), t)

        prelim_weights, _ = spectral_weights(features[pool], current, cfg.bandwidth)
        recent = np.arange(max(1, t - cfg.spectral_window), t)
        recent_feature = features[recent].mean(axis=0)
        weighted_feature = np.average(features[pool], axis=0, weights=prelim_weights)
        drift = float(np.linalg.norm(recent_feature - weighted_feature))

        m_t = drift_gate_length(drift, cfg)
        gated = pool[pool >= t - m_t]
        if len(gated) < 20:
            gated = pool[-min(len(pool), 20):]

        weights, distances = spectral_weights(features[gated], current, cfg.bandwidth)
        neff = effective_sample_size(weights)
        qhat = weighted_quantile(residuals[gated], weights, 1 - self.alpha_t)
        lo, hi = point_prediction - qhat, point_prediction + qhat
        diagnostics = {
            "qhat": float(qhat),
            "alpha_t": float(self.alpha_t),
            "drift": drift,
            "neff": neff,
            "window": int(m_t),
            "mean_spectral_distance": float(np.average(distances, weights=weights)),
        }
        return lo, hi, diagnostics

    def update(self, covered):
        miss = 0 if covered else 1
        cfg = self.config
        self.alpha_t = float(
            np.clip(
                self.alpha_t + cfg.gamma * (cfg.alpha - miss),
                cfg.alpha_min,
                cfg.alpha_max,
            )
        )
        return self.alpha_t
