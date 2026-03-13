"""
Plot stress, kappa, and damage progression from tensile test case_reports.
Uses full-tmp series to set the x-axis limit.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_ORDER = [
    ("local", "tmp"),
    ("local", "frozen"),
    ("full", "tmp"),
    ("full", "frozen"),
]
CASE_COLORS = {"tmp": "#1f77b4", "frozen": "#ff7f0e"}
CASE_LINESTYLES = {"local": "-", "full": "--"}

Y_SPECS = [
    ("max_von_mises", "Max von Mises stress"),
    ("max_kappa", "Max kappa"),
    ("max_damage", "Max damage d"),
]


def _x_limit_from_full_tmp(case_reports):
    full_tmp = case_reports.get(("full", "tmp"))
    if full_tmp is None:
        return None
    rows = full_tmp.get("progress_rows", [])
    if not rows:
        return None
    return max(int(row["step"]) for row in rows)


def plot_mesh_progress(mesh_name, case_reports, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    x_max = _x_limit_from_full_tmp(case_reports)

    for damage_label, update_mode in CASE_ORDER:
        key = (damage_label, update_mode)
        report = case_reports.get(key)
        if report is None:
            continue
        rows = report.get("progress_rows", [])
        if not rows:
            continue

        x = [int(row["step"]) for row in rows]
        label = f"{damage_label}-{update_mode}"
        color = CASE_COLORS[update_mode]
        linestyle = CASE_LINESTYLES[damage_label]

        for ax, (y_key, y_title) in zip(axes, Y_SPECS):
            y = [float(row[y_key]) for row in rows]
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.6)
            ax.set_ylabel(y_title)
            ax.grid(True, alpha=0.3)

    if x_max is not None:
        for ax in axes:
            ax.set_xlim(0, x_max)

    axes[-1].set_xlabel("Load step index")
    fig.suptitle(f"{mesh_name}: max response progression")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = Path(out_dir)
    fig_path = out_dir / f"{mesh_name}_max_progress.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path
