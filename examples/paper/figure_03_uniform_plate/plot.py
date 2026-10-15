import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MESH_COLUMNS = (
    ("coarse", "Coarse"),
    ("medium", "Refined"),
    ("non_uniform", "Non-uniform"),
)
CASE_STYLES = {
    "local": ("Local Monolithic", "#5b9bd5", "-", "^"),
    "local_staggered": ("Local Staggered", "#ed4b4b", "-", "s"),
    "monolithic": (
        "Gradient Enhanced Monolithic",
        "#5b9bd5",
        "--",
        "^",
    ),
    "staggered": (
        "Gradient Enhanced Staggered",
        "#ed4b4b",
        "--",
        "s",
    ),
}
METRICS = (
    ("max_von_mises", r"$\sigma_{\mathrm{vM,max}}$ [MPa]"),
    ("max_kappa", r"$\kappa_{\mathrm{max}}$ [MPa]"),
    ("max_damage", r"$d_{\mathrm{max}}$"),
)


def read_progress(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def identify_series(path):
    name = path.stem.lower()
    mesh = next(
        (mesh_name for mesh_name, _title in MESH_COLUMNS if mesh_name in name),
        None,
    )
    case = next(
        (
            case_name
            for case_name in (
                "local_staggered",
                "monolithic",
                "staggered",
                "local",
            )
            if case_name in name
        ),
        None,
    )
    return mesh, case


def collect_results(inputs):
    results = {}
    for input_path in inputs:
        mesh, case = identify_series(input_path)
        if mesh is not None and case is not None:
            results[(mesh, case)] = read_progress(input_path)
    return results


def add_mesh_thumbnail(axis, mesh, title, image_directory):
    image_path = image_directory / f"{mesh}_mesh.png"
    if image_path.exists():
        axis.imshow(plt.imread(image_path))
    else:
        axis.text(
            0.5,
            0.5,
            f"{title}\nmesh",
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )
    axis.set_title(title, fontsize=10, fontweight="bold", pad=3)
    axis.axis("off")


def plot_metric(axis, mesh_results, metric):
    for case, style in CASE_STYLES.items():
        rows = mesh_results.get(case)
        if not rows:
            continue
        label, color, line_style, marker = style
        displacement = [float(row["displacement"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        axis.plot(
            displacement,
            values,
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=1.25,
        )
        axis.plot(
            displacement[-1],
            values[-1],
            marker=marker,
            color=color,
            markeredgecolor="#202020",
            markeredgewidth=0.5,
            markersize=4.5,
        )
    axis.grid(color="#d8d8d8", linewidth=0.45, alpha=0.8)
    axis.tick_params(labelsize=7.5)


def plot_progress(inputs, output, image_directory):
    results = collect_results(inputs)
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(7.2, 9.2),
        gridspec_kw={"height_ratios": (0.9, 1.0, 1.0, 1.0)},
    )
    for column, (mesh, title) in enumerate(MESH_COLUMNS):
        add_mesh_thumbnail(
            axes[0, column],
            mesh,
            title,
            image_directory,
        )
        mesh_results = {
            case: rows
            for (result_mesh, case), rows in results.items()
            if result_mesh == mesh
        }
        for row, (metric, ylabel) in enumerate(METRICS, start=1):
            axis = axes[row, column]
            plot_metric(axis, mesh_results, metric)
            axis.set_box_aspect(1)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=8.5)
            else:
                axis.tick_params(labelleft=False)
            if row == 3:
                axis.set_xlabel(r"Prescribed $u$ [mm]", fontsize=8.5)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            fontsize=8,
            frameon=True,
            handlelength=3.0,
        )
    figure.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.97,
        bottom=0.11,
        hspace=0.12,
        wspace=0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mesh-image-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "meshes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    plot_progress(
        arguments.inputs,
        arguments.output,
        arguments.mesh_image_dir,
    )
