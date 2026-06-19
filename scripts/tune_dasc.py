import itertools
from pathlib import Path

import pandas as pd

from run_first_simulation import run


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def score_row(row, target=0.90):
    coverage_gap = abs(row["empirical_coverage"] - target)
    undercoverage_penalty = max(0.0, target - row["empirical_coverage"]) * 20.0
    return row["avg_width"] + 15.0 * coverage_gap + undercoverage_penalty


def main():
    grid = {
        "h": [0.35, 0.45, 0.55],
        "drift_lambda": [0.45, 0.70],
        "m_min": [40, 80],
        "stability_relax": [0.0, 0.01, 0.02, 0.03],
    }

    rows = []
    full_results = []
    for h, drift_lambda, m_min, stability_relax in itertools.product(
        grid["h"], grid["drift_lambda"], grid["m_min"], grid["stability_relax"]
    ):
        per_seed = []
        for seed in range(5):
            results, _ = run(
                seed=seed,
                dasc_h=h,
                drift_lambda=drift_lambda,
                dasc_m_min=m_min,
                stability_relax=stability_relax,
            )
            dasc = results[results["method"] == "DASC"].copy()
            dasc["h"] = h
            dasc["drift_lambda"] = drift_lambda
            dasc["m_min"] = m_min
            dasc["stability_relax"] = stability_relax
            per_seed.append(dasc)
        combined = pd.concat(per_seed, ignore_index=True)
        row = {
            "h": h,
            "drift_lambda": drift_lambda,
            "m_min": m_min,
            "stability_relax": stability_relax,
            "empirical_miscoverage": combined["miss"].mean(),
            "empirical_coverage": 1 - combined["miss"].mean(),
            "avg_width": combined["width"].mean(),
            "median_neff": combined["neff"].median(),
            "avg_drift": combined["drift"].mean(),
        }
        row["score"] = score_row(row)
        rows.append(row)
        full_results.append(combined)

    tuning = pd.DataFrame(rows).sort_values("score")
    tuning.to_csv(OUT / "dasc_tuning_grid.csv", index=False)
    best = tuning.iloc[0]
    print(tuning.head(12).to_string(index=False))

    best_results = pd.concat(
        [
            r
            for r in full_results
            if float(r["h"].iloc[0]) == float(best["h"])
            and float(r["drift_lambda"].iloc[0]) == float(best["drift_lambda"])
            and int(r["m_min"].iloc[0]) == int(best["m_min"])
            and float(r["stability_relax"].iloc[0]) == float(best["stability_relax"])
        ],
        ignore_index=True,
    )
    best_results.to_csv(OUT / "dasc_best_tuned_results.csv", index=False)


if __name__ == "__main__":
    main()
