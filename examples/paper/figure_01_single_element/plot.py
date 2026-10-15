import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ETA_VALUES = (1.0, 2.0, 5.0, 20.0, 100.0)
KAPPA_VALUES = (0.0, 0.2, 0.6, 1.0, 2.0)
LINE_STYLES = (
    ("#5b9bd5", "-", 1.2),
    ("#ed4b4b", "-", 1.2),
    ("#202020", ":", 1.0),
    ("#5b9bd5", ":", 1.0),
    ("#ed4b4b", ":", 1.0),
)


def read_columns(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        name: [float(row[name]) for row in rows]
        for name in ("stretch", "sigma_11", "kappa", "damage")
    }


def classify_results(input_paths):
    eta_results = {}
    kappa_results = {}
    for input_path in input_paths:
        tokens = input_path.stem.split("_")
        eta = float(tokens[1])
        kappa = float(tokens[3])
        if kappa == 0.2:
            eta_results[eta] = read_columns(input_path)
        if eta == 2.0:
            kappa_results[kappa] = read_columns(input_path)
    return eta_results, kappa_results


def undamaged_stress(stretch):
    shear_modulus = 42.0 / (2.0 * (1.0 + 0.45))
    return shear_modulus * (stretch**2 - 1.0 / stretch)


def style_axis(axis, ylabel, ylim):
    axis.set_xlim(1.0, 1.5)
    axis.set_ylim(*ylim)
    axis.set_xlabel(r"local stretch $\lambda$ [-]", fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.tick_params(labelsize=8)
    axis.grid(color="#d8d8d8", linewidth=0.45, alpha=0.8)


def plot_sweep(axis, results, values, field, label_symbol, unit):
    for value, style in zip(values, LINE_STYLES):
        if value not in results:
            continue
        color, line_style, width = style
        axis.plot(
            results[value]["stretch"],
            results[value][field],
            color=color,
            linestyle=line_style,
            linewidth=width,
            label=rf"${label_symbol} = {value:.1f}$ {unit}",
        )


def plot_results(input_paths, output_path):
    eta_results, kappa_results = classify_results(input_paths)
    figure, axes = plt.subplots(3, 2, figsize=(7.4, 9.0))
    stretch = [1.0 + index / 200.0 for index in range(101)]
    for axis in axes[0]:
        axis.plot(
            stretch,
            undamaged_stress(np.asarray(stretch)),
            color="#202020",
            linestyle="--",
            linewidth=1.4,
            label="undamaged",
        )

    plot_sweep(
        axes[0, 0],
        eta_results,
        ETA_VALUES,
        "sigma_11",
        r"\eta_d",
        r"MPa$^{-1}$",
    )
    plot_sweep(
        axes[0, 1],
        kappa_results,
        KAPPA_VALUES,
        "sigma_11",
        r"\kappa_d",
        "MPa",
    )
    plot_sweep(
        axes[1, 0],
        eta_results,
        ETA_VALUES,
        "kappa",
        r"\eta_d",
        r"MPa$^{-1}$",
    )
    plot_sweep(
        axes[1, 1],
        kappa_results,
        KAPPA_VALUES,
        "kappa",
        r"\kappa_d",
        "MPa",
    )
    plot_sweep(
        axes[2, 0],
        eta_results,
        ETA_VALUES,
        "damage",
        r"\eta_d",
        r"MPa$^{-1}$",
    )
    plot_sweep(
        axes[2, 1],
        kappa_results,
        KAPPA_VALUES,
        "damage",
        r"\kappa_d",
        "MPa",
    )

    titles = (
        r"Varying $\eta_d$, fixed $\kappa_d = 0.2$ MPa",
        r"Varying $\kappa_d$, fixed $\eta_d = 2.0$ MPa$^{-1}$",
    )
    for column, title in enumerate(titles):
        for row in range(3):
            axes[row, column].set_title(title, fontsize=9, pad=4)
    for axis in axes[0]:
        style_axis(axis, r"Cauchy stress $\sigma_{11}$ [MPa]", (0.0, 18.0))
    style_axis(axes[1, 0], r"local damage variable $\kappa$ [MPa]", (0.0, 1.2))
    style_axis(axes[1, 1], r"local damage variable $\kappa$ [MPa]", (0.0, 2.4))
    for axis in axes[2]:
        style_axis(axis, r"damage $d = 1-f_d(\kappa)$ [-]", (0.0, 1.0))
    for axis in axes[0]:
        axis.legend(loc="upper right", fontsize=6.4, framealpha=0.9)
    for axis in axes[1:].flat:
        axis.legend(loc="upper left", fontsize=6.4, framealpha=0.9)

    for label, row in zip(("a", "b", "c"), range(3)):
        axes[row, 0].text(
            -0.14,
            1.08,
            label,
            transform=axes[row, 0].transAxes,
            fontsize=11,
            fontweight="bold",
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    plot_results(arguments.inputs, arguments.output)
