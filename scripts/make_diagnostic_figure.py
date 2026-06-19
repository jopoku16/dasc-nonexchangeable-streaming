from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def rolling_mean(x, window=50):
    return pd.Series(x).rolling(window, min_periods=max(5, window // 5)).mean().to_numpy()


def scale(values, lo, hi, invert=False):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if not finite.any():
        return np.full_like(arr, (lo + hi) / 2, dtype=float)
    mn, mx = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if mx <= mn:
        out[:] = (lo + hi) / 2
    else:
        out = lo + (arr - mn) * (hi - lo) / (mx - mn)
    if invert:
        out = hi - (out - lo)
    return out


def scale_fixed(values, vmin, vmax, lo, hi, invert=False):
    arr = np.asarray(values, dtype=float)
    out = lo + (arr - vmin) * (hi - lo) / (vmax - vmin)
    out = np.clip(out, min(lo, hi), max(lo, hi))
    if invert:
        out = hi - (out - lo)
    return out


def wrap_text(text, max_chars=145):
    words = text.split()
    lines = []
    current = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_line(draw, xs, ys, color, width=3):
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pts) > 1:
        draw.line(pts, fill=color, width=width, joint="curve")


def draw_axes(draw, box, title, y_label=None):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(200, 205, 210), width=1)
    draw.text((x0, y0 - 24), title, fill=(30, 35, 40), font=FONT_BOLD)
    if y_label:
        draw.text((x0 + 8, y0 + 8), y_label, fill=(95, 100, 105), font=FONT_SMALL)
    for xpos in [450, 900]:
        x = x0 + (xpos - 220) * (x1 - x0) / (1399 - 220)
        draw.line((x, y0, x, y1), fill=(180, 70, 70), width=2)
        draw.text((x + 4, y0 + 5), f"t={xpos}", fill=(150, 55, 55), font=FONT_SMALL)


def draw_legend(draw, items, x, y):
    for label, color in items:
        draw.line((x, y + 7, x + 28, y + 7), fill=color, width=4)
        draw.text((x + 36, y), label, fill=(55, 60, 65), font=FONT_SMALL)
        x += 190


def build_figure_data():
    stream = pd.read_csv(RESULTS / "synthetic_stream_seed0.csv")
    results = pd.read_csv(RESULTS / "first_simulation_results.csv")
    seed0 = results[results["seed"] == 0].copy()

    rows = []
    for t in range(220, int(stream["t"].max()) + 1):
        row = {
            "t": t,
            "y": float(stream.loc[stream["t"] == t, "y"].iloc[0]),
            "signal": float(stream.loc[stream["t"] == t, "signal"].iloc[0]),
        }
        for method in ["rolling", "spectral_only", "adaptive", "conformal_PID", "DASC"]:
            m = seed0[(seed0["t"] == t) & (seed0["method"] == method)]
            if len(m):
                row[f"{method}_miss"] = int(m["miss"].iloc[0])
                row[f"{method}_width"] = float(m["width"].iloc[0])
                row[f"{method}_neff"] = float(m["neff"].iloc[0])
        d = seed0[(seed0["t"] == t) & (seed0["method"] == "DASC")]
        row["DASC_drift"] = float(d["drift"].iloc[0]) if len(d) else np.nan
        row["DASC_window"] = float(d["window"].iloc[0]) if len(d) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    for method in ["rolling", "spectral_only", "adaptive", "conformal_PID", "DASC"]:
        df[f"{method}_local_miscoverage"] = rolling_mean(df[f"{method}_miss"], 75)
        df[f"{method}_local_coverage"] = 1 - df[f"{method}_local_miscoverage"]
    df["DASC_drift_smooth"] = rolling_mean(df["DASC_drift"], 25)
    df["DASC_neff_smooth"] = rolling_mean(df["DASC_neff"], 25)
    out = RESULTS / "figure1_diagnostic_timeseries.csv"
    df.to_csv(out, index=False)
    return df, out


def make_png(df):
    width, height = 1500, 1200
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_x = 95
    panel_w = width - 2 * margin_x
    panel_h = 220
    gap = 65
    top = 80
    boxes = []
    for i in range(4):
        y0 = top + i * (panel_h + gap)
        boxes.append((margin_x, y0, margin_x + panel_w, y0 + panel_h))

    draw.text((margin_x, 25), "Figure 1. DASC detects drift before coverage breaks", fill=(20, 25, 30), font=FONT_TITLE)

    t = df["t"].to_numpy()
    x = scale(t, margin_x, margin_x + panel_w)

    # Panel 1
    box = boxes[0]
    draw_axes(draw, box, "Synthetic streaming series", "response")
    ymin = min(df["y"].min(), df["signal"].min())
    ymax = max(df["y"].max(), df["signal"].max())
    y_y = scale_fixed(df["y"], ymin, ymax, box[1] + 18, box[3] - 18, invert=True)
    y_signal = scale_fixed(df["signal"], ymin, ymax, box[1] + 18, box[3] - 18, invert=True)
    draw_line(draw, x, y_y, (90, 110, 135), width=2)
    draw_line(draw, x, y_signal, (20, 120, 110), width=4)
    draw_legend(draw, [("observed", (90, 110, 135)), ("signal", (20, 120, 110))], box[0] + 15, box[3] - 30)

    # Panel 2
    box = boxes[1]
    draw_axes(draw, box, "Local coverage over time (75-point rolling window)", "coverage")
    y_target = scale_fixed([0.90], 0.75, 1.0, box[1] + 18, box[3] - 18, invert=True)[0]
    draw.line((box[0], y_target, box[2], y_target), fill=(80, 80, 80), width=2)
    draw.text((box[0] + 8, y_target - 22), "90% target", fill=(80, 80, 80), font=FONT_SMALL)
    for method, color in [
        ("rolling", (210, 80, 65)),
        ("spectral_only", (140, 85, 180)),
        ("adaptive", (70, 110, 210)),
        ("DASC", (20, 150, 100)),
    ]:
        vals = df[f"{method}_local_coverage"].clip(0.75, 1.0)
        y = scale_fixed(vals, 0.75, 1.0, box[1] + 18, box[3] - 18, invert=True)
        draw_line(draw, x, y, color, width=3 if method != "DASC" else 5)
    draw_legend(draw, [
        ("rolling", (210, 80, 65)),
        ("spectral-only", (140, 85, 180)),
        ("adaptive", (70, 110, 210)),
        ("DASC", (20, 150, 100)),
    ], box[0] + 15, box[3] - 30)

    # Panel 3
    box = boxes[2]
    draw_axes(draw, box, "DASC transport drift score", "drift")
    y = scale(df["DASC_drift_smooth"], box[1] + 18, box[3] - 18, invert=True)
    draw_line(draw, x, y, (200, 120, 30), width=5)

    # Panel 4
    box = boxes[3]
    draw_axes(draw, box, "DASC effective sample size", "n_eff")
    neff_values = df["DASC_neff_smooth"]
    neff_min = min(50, float(np.nanmin(neff_values)))
    neff_max = max(260, float(np.nanmax(neff_values)))
    y = scale_fixed(neff_values, neff_min, neff_max, box[1] + 18, box[3] - 18, invert=True)
    draw_line(draw, x, y, (35, 95, 170), width=5)
    threshold_y = scale_fixed([80], neff_min, neff_max, box[1] + 18, box[3] - 18, invert=True)[0]
    draw.line((box[0], threshold_y, box[2], threshold_y), fill=(120, 120, 120), width=2)
    draw.text((box[0] + 8, threshold_y - 22), "fragility threshold", fill=(90, 90, 90), font=FONT_SMALL)

    caption = (
        "Spectral-only conformal prediction is calibrated during recurring regimes but undercovers after drift. "
        "DASC uses the drift score and effective sample size to gate calibration while maintaining near-nominal coverage."
    )
    for i, line in enumerate(wrap_text(caption)):
        draw.text((margin_x, height - 58 + i * 20), line, fill=(50, 55, 60), font=FONT_SMALL)

    out = FIGURES / "figure1_dasc_diagnostic_triangle.png"
    img.save(out)
    return out


try:
    FONT_TITLE = ImageFont.truetype("arial.ttf", 32)
    FONT_BOLD = ImageFont.truetype("arialbd.ttf", 22)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 17)
except Exception:
    FONT_TITLE = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def main():
    df, csv_path = build_figure_data()
    png_path = make_png(df)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
