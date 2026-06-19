from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

W, H = 1600, 1000
MARGIN = 120
INK = "#202124"
MUTED = "#5f6368"
GRID = "#d9dde3"
TARGET = "#9aa0a6"
COLORS = {
    "DASC": "#1b9e77",
    "full_DASC": "#1b9e77",
    "rolling": "#7570b3",
    "adaptive": "#d95f02",
    "adaptive_only": "#d95f02",
    "conformal_PID": "#e7298a",
    "exp_weighted": "#66a61e",
    "spectral_only": "#1f78b4",
    "DASC_no_drift_gate": "#a6761d",
}


def font(size, bold=False):
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def canvas(title, subtitle=None):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 45), title, fill=INK, font=font(44, True))
    if subtitle:
        d.text((72, 100), subtitle, fill=MUTED, font=font(24))
    return img, d


def axis(d, x0, y0, x1, y1, xlabel, ylabel):
    d.line((x0, y1, x1, y1), fill=INK, width=3)
    d.line((x0, y0, x0, y1), fill=INK, width=3)
    d.text(((x0 + x1) // 2 - 80, y1 + 55), xlabel, fill=INK, font=font(24))
    d.text((x0 - 95, y0 - 45), ylabel, fill=INK, font=font(24))


def label(d, xy, text, size=22, fill=INK, bold=False):
    d.text(xy, str(text), fill=fill, font=font(size, bold))


def save(img, name):
    path = FIGURES / name
    img.save(path, quality=95)
    print(path)


def coverage_width_tradeoff():
    files = [
        ("Synthetic", "first_simulation_summary.csv"),
        ("Electricity", "real_power_summary.csv"),
        ("Weather", "real_weather_summary.csv"),
        ("Finance", "real_finance_summary.csv"),
    ]
    frames = []
    for dataset, filename in files:
        df = pd.read_csv(RESULTS / filename)
        cov_col = "empirical_coverage"
        width_col = "avg_width"
        if cov_col not in df.columns:
            cov_col = "coverage"
        df = df.rename(columns={cov_col: "coverage", width_col: "width"})
        df["dataset"] = dataset
        frames.append(df[["dataset", "method", "coverage", "width"]])
    all_df = pd.concat(frames, ignore_index=True)

    img, d = canvas(
        "Coverage-width tradeoff across benchmarks",
        "Points closer to 0.90 coverage with smaller width are better; DASC is marked in green.",
    )
    x0, y0, x1, y1 = 150, 180, 1450, 860
    axis(d, x0, y0, x1, y1, "average interval width", "empirical coverage")
    d.line((x0, y0 + int((0.98 - 0.90) / 0.16 * (y1 - y0)), x1, y0 + int((0.98 - 0.90) / 0.16 * (y1 - y0))), fill=TARGET, width=2)
    label(d, (x1 - 175, y0 + int((0.98 - 0.90) / 0.16 * (y1 - y0)) - 34), "target 0.90", 20, TARGET)

    xmin, xmax = 1.8, max(all_df["width"]) * 1.05
    ymin, ymax = 0.84, 1.00
    for _, r in all_df.iterrows():
        x = x0 + int((r["width"] - xmin) / (xmax - xmin) * (x1 - x0))
        y = y1 - int((r["coverage"] - ymin) / (ymax - ymin) * (y1 - y0))
        color = COLORS.get(r["method"], "#444444")
        radius = 15 if r["method"] == "DASC" else 11
        d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
        if r["method"] == "DASC":
            label(d, (x + 16, y - 10), r["dataset"], 19, color, True)

    legend_y = 195
    for i, method in enumerate(["DASC", "adaptive", "exp_weighted", "rolling", "conformal_PID", "spectral_only"]):
        lx = 1060
        ly = legend_y + i * 38
        d.ellipse((lx, ly, lx + 22, ly + 22), fill=COLORS[method])
        label(d, (lx + 34, ly - 4), method.replace("_", " "), 21)
    save(img, "figure2_coverage_width_tradeoff.png")


def cross_domain_gain():
    df = pd.read_csv(RESULTS / "cross_domain_summary.csv")
    img, d = canvas(
        "DASC width change versus the best calibrated baseline",
        "Positive values mean DASC is narrower; the finance case is retained as an honest hard case.",
    )
    x0, y0, x1, y1 = 260, 210, 1420, 820
    d.line((x0, y1, x1, y1), fill=INK, width=3)
    d.line((x0, y0, x0, y1), fill=INK, width=3)
    vals = df["width_reduction_percent"].to_numpy()
    vmin, vmax = -10, 50
    zero_y = y1 - int((0 - vmin) / (vmax - vmin) * (y1 - y0))
    d.line((x0, zero_y, x1, zero_y), fill=TARGET, width=2)
    bar_w = 210
    gap = 150
    for i, r in df.iterrows():
        cx = x0 + 130 + i * (bar_w + gap)
        val = float(r["width_reduction_percent"])
        top = y1 - int((val - vmin) / (vmax - vmin) * (y1 - y0))
        color = "#1b9e77" if val >= 0 else "#b3261e"
        d.rectangle((cx, min(top, zero_y), cx + bar_w, max(top, zero_y)), fill=color)
        label(d, (cx + 18, min(top, zero_y) - 38), f"{val:.1f}%", 28, color, True)
        label(d, (cx + 10, y1 + 28), r["dataset"], 25, INK, True)
        label(d, (cx - 8, y1 + 66), f"vs {r['best_baseline']}".replace("_", " "), 20, MUTED)
    label(d, (80, y0 - 20), "width reduction (%)", 24)
    save(img, "figure3_cross_domain_width_reduction.png")


def synthetic_regime_heatmap():
    df = pd.read_csv(RESULTS / "first_simulation_regime_summary.csv")
    methods = ["DASC", "adaptive", "conformal_PID", "exp_weighted", "rolling", "spectral_only"]
    regimes = ["recurring_A", "recurring_B", "drift_after_shift"]
    img, d = canvas(
        "Coverage by synthetic regime",
        "The post-shift column shows why the drift-aware gate matters.",
    )
    x0, y0 = 360, 230
    cell_w, cell_h = 300, 90
    for j, reg in enumerate(regimes):
        label(d, (x0 + j * cell_w + 20, y0 - 55), reg.replace("_", " "), 24, INK, True)
    for i, method in enumerate(methods):
        label(d, (95, y0 + i * cell_h + 28), method.replace("_", " "), 24, COLORS.get(method, INK), method == "DASC")
        for j, reg in enumerate(regimes):
            row = df[(df["method"] == method) & (df["regime"] == reg)].iloc[0]
            cov = float(row["empirical_coverage"])
            closeness = max(0, 1 - abs(cov - 0.90) / 0.08)
            green = int(110 + 120 * closeness)
            red = int(235 - 120 * closeness)
            color = (red, green, 150)
            x = x0 + j * cell_w
            y = y0 + i * cell_h
            d.rectangle((x, y, x + cell_w - 12, y + cell_h - 12), fill=color, outline="white", width=3)
            label(d, (x + 95, y + 27), f"{cov:.3f}", 28, INK, True)
    save(img, "figure4_synthetic_regime_coverage.png")


def ablation_summary():
    df = pd.read_csv(RESULTS / "ablation_summary.csv")
    order = ["rolling", "adaptive_only", "spectral_only", "DASC_no_drift_gate", "full_DASC"]
    df = df.set_index("method").loc[order].reset_index()
    img, d = canvas(
        "Ablation study: calibration, spectral weighting, and drift gating",
        "Full DASC gives near-nominal coverage while keeping the diagnostic layer active.",
    )
    x0, y0, x1, y1 = 180, 230, 1450, 790
    d.line((x0, y1, x1, y1), fill=INK, width=3)
    width_max = df["avg_width"].max() * 1.15
    for i, r in df.iterrows():
        y = y0 + i * 105
        bar_len = int(float(r["avg_width"]) / width_max * (x1 - x0 - 260))
        color = COLORS.get(r["method"], "#555555")
        label(d, (45, y + 12), r["method"].replace("_", " "), 22, color, r["method"] == "full_DASC")
        d.rectangle((x0, y, x0 + bar_len, y + 48), fill=color)
        label(d, (x0 + bar_len + 18, y + 8), f"width {r['avg_width']:.2f}; coverage {r['coverage']:.3f}", 22)
    save(img, "figure5_ablation_component_summary.png")


def real_stability_panels():
    datasets = [
        ("Electricity", "real_power_monthly_summary.csv", "month"),
        ("Weather", "real_weather_monthly_summary.csv", "month"),
        ("Finance", "real_finance_year_summary.csv", "year"),
    ]
    img, d = canvas(
        "Local coverage stability on real streams",
        "DASC is compared with the strongest calibrated baseline in each domain.",
    )
    panel_w, panel_h = 440, 560
    lefts = [100, 580, 1060]
    top = 250
    for (name, file, time_col), x0 in zip(datasets, lefts):
        df = pd.read_csv(RESULTS / file)
        cross = pd.read_csv(RESULTS / "cross_domain_summary.csv")
        baseline = cross[cross["dataset"] == name]["best_baseline"].iloc[0]
        sub = df[df["method"].isin(["DASC", baseline])].copy()
        times = list(dict.fromkeys(sub[time_col].astype(str).tolist()))
        if len(times) > 30:
            times = times[-30:]
        sub = sub[sub[time_col].astype(str).isin(times)]
        y0, y1 = top, top + panel_h
        d.rectangle((x0, y0, x0 + panel_w, y1), outline=GRID, width=2)
        label(d, (x0, y0 - 42), name, 28, INK, True)
        target_y = y1 - int((0.90 - 0.70) / 0.30 * panel_h)
        d.line((x0 + 45, target_y, x0 + panel_w - 25, target_y), fill=TARGET, width=2)
        for method in ["DASC", baseline]:
            pts = []
            vals = sub[sub["method"] == method]["coverage"].to_numpy()
            if len(vals) == 0:
                continue
            for i, cov in enumerate(vals):
                x = x0 + 45 + int(i / max(1, len(vals) - 1) * (panel_w - 80))
                y = y1 - int((float(cov) - 0.70) / 0.30 * panel_h)
                y = max(y0 + 8, min(y1 - 8, y))
                pts.append((x, y))
            if len(pts) > 1:
                d.line(pts, fill=COLORS.get(method, "#444444"), width=4)
            for x, y in pts[:: max(1, len(pts) // 10)]:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS.get(method, "#444444"))
        label(d, (x0 + 55, y1 + 18), "DASC", 20, COLORS["DASC"], True)
        label(d, (x0 + 145, y1 + 18), baseline.replace("_", " "), 20, COLORS.get(baseline, MUTED))
    save(img, "figure6_real_local_coverage.png")


def stress_test_heatmap():
    df = pd.read_csv(RESULTS / "stress_test_summary.csv")
    methods = ["DASC", "adaptive", "conformal_PID", "exp_weighted", "rolling", "spectral_only"]
    scenarios = [
        "abrupt_shift",
        "gradual_frequency",
        "heavy_tail",
        "mixed_drift",
        "weak_recurrence",
    ]
    img, d = canvas(
        "Stress-test coverage across hard synthetic regimes",
        "Cells closest to 0.90 are strongest; DASC stays calibrated in every stress regime.",
    )
    x0, y0 = 310, 230
    cell_w, cell_h = 235, 86
    for j, scenario in enumerate(scenarios):
        words = scenario.replace("_", " ")
        label(d, (x0 + j * cell_w + 8, y0 - 55), words, 19, INK, True)
    for i, method in enumerate(methods):
        label(d, (72, y0 + i * cell_h + 24), method.replace("_", " "), 23, COLORS.get(method, INK), method == "DASC")
        for j, scenario in enumerate(scenarios):
            row = df[(df["method"] == method) & (df["scenario"] == scenario)].iloc[0]
            cov = float(row["coverage"])
            err = abs(cov - 0.90)
            closeness = max(0, 1 - err / 0.04)
            green = int(110 + 120 * closeness)
            red = int(235 - 120 * closeness)
            color = (red, green, 150)
            x = x0 + j * cell_w
            y = y0 + i * cell_h
            d.rectangle((x, y, x + cell_w - 12, y + cell_h - 12), fill=color, outline="white", width=3)
            label(d, (x + 70, y + 25), f"{cov:.3f}", 27, INK, True)
    save(img, "figure7_stress_test_coverage.png")


def main():
    coverage_width_tradeoff()
    cross_domain_gain()
    synthetic_regime_heatmap()
    ablation_summary()
    real_stability_panels()
    stress_test_heatmap()


if __name__ == "__main__":
    main()
