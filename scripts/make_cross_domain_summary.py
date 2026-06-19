from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


DATASETS = [
    ("Electricity", OUT / "real_power_summary.csv"),
    ("Weather", OUT / "real_weather_summary.csv"),
    ("Finance", OUT / "real_finance_summary.csv"),
]


def best_baseline(df):
    baselines = df[df["method"] != "DASC"].copy()
    calibrated = baselines[
        (baselines["empirical_coverage"] >= 0.89)
        & (baselines["empirical_coverage"] <= 0.97)
    ]
    if calibrated.empty:
        calibrated = baselines
    return calibrated.sort_values("avg_width").iloc[0]


def main():
    rows = []
    for dataset, path in DATASETS:
        df = pd.read_csv(path)
        dasc = df[df["method"] == "DASC"].iloc[0]
        best = best_baseline(df)
        width_reduction = 100 * (best["avg_width"] - dasc["avg_width"]) / best["avg_width"]
        rows.append({
            "dataset": dataset,
            "dasc_coverage": dasc["empirical_coverage"],
            "dasc_width": dasc["avg_width"],
            "best_baseline": best["method"],
            "best_baseline_coverage": best["empirical_coverage"],
            "best_baseline_width": best["avg_width"],
            "width_reduction_percent": width_reduction,
            "dasc_median_neff": dasc["median_neff"],
            "dasc_avg_drift": dasc["avg_drift"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "cross_domain_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
