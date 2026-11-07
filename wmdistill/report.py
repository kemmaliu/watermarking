"""Collect per-job metrics into results.csv and plot metrics vs lambda (one row per watermark)."""
import json

from .common import read_jsonl, run_dir, student_dir, sweep_jobs

METRICS = [("detect_rate", "detection rate (p < alpha)"), ("mean_z_per_token", "mean z per token"),
           ("teacher_ppl", "teacher perplexity"), ("rep3_rate", "3-gram repetition rate")]
STUDENT_COLOR, INK, MUTED, GRID = "#2a78d6", "#1a1a19", "#6b6a64", "#e4e3dd"


def write_report(cfg):
    import pandas as pd

    d = run_dir(cfg)
    rows, missing = [], []
    for wm, lam in sweep_jobs(cfg):
        p = student_dir(cfg, wm, lam) / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            m.pop("final_train", None)
            rows.append(m)
        else:
            missing.append((wm, lam))
    controls = json.loads((d / "controls.json").read_text()) if (d / "controls.json").exists() else {}
    for wm, c in controls.items():
        for name, m in c.items():
            rows.append({"watermark": wm, "lambda": None, "control": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(d / "results.csv", index=False)
    print(df.to_string(index=False))
    if missing:
        print(f"\nmissing {len(missing)} jobs: {missing}")
    if not df.empty:
        _plot(cfg, df, controls, d / "lambda_sweep.png")


def _plot(cfg, df, controls, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wms = cfg["watermarks"]
    metrics = [(k, t) for k, t in METRICS if k in df.columns]
    fig, axes = plt.subplots(len(wms), len(metrics), figsize=(3.6 * len(metrics), 2.9 * len(wms)),
                             squeeze=False)
    for r, wm in enumerate(wms):
        sub = df[(df["watermark"] == wm) & df["lambda"].notna()].sort_values("lambda")
        for c, (key, title) in enumerate(metrics):
            ax = axes[r, c]
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_axisbelow(True)
            if not sub.empty:
                ax.plot(sub["lambda"], sub[key], color=STUDENT_COLOR, linewidth=2, marker="o",
                        markersize=5, markeredgecolor="white", markeredgewidth=1.5, label="trained student")
            for ctrl, style, label in (("teacher_watermarked", "--", "teacher (wm)"),
                                       ("student_base", ":", "base student")):
                v = controls.get(wm, {}).get(ctrl, {}).get(key)
                if v is not None:
                    ax.axhline(v, color=MUTED, linestyle=style, linewidth=1.2, label=label)
            ax.set_title(f"{wm}: {title}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("λ", fontsize=8, color=MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
