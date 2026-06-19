from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


STEPS = [
    "scripts/run_first_simulation.py",
    "scripts/run_stress_tests.py",
    "scripts/run_external_mapie_benchmarks.py",
    "scripts/run_ablation.py",
    "scripts/tune_dasc.py",
    "scripts/run_real_power_experiment.py",
    "scripts/run_real_weather_experiment.py",
    "scripts/run_real_finance_experiment.py",
    "scripts/make_cross_domain_summary.py",
    "scripts/make_diagnostic_figure.py",
    "scripts/make_result_figures.py",
]


def main():
    for step in STEPS:
        print(f"\n=== running {step} ===", flush=True)
        subprocess.run([sys.executable, step], cwd=ROOT, check=True)
    print("\nAll experiments and figures were regenerated.")


if __name__ == "__main__":
    main()
