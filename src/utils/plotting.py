"""Plotting helpers for control-response experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_specialist_response_plot(
    raw_trace: dict[str, np.ndarray],
    controller_trace: dict[str, np.ndarray],
    destination: str | Path,
    *,
    title: str,
    controller_label: str = "SAC",
) -> Path:
    """Save the canonical p_c / p_ref / p_raw / p_SAC comparison."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray(controller_trace["time_s"], dtype=float)
    figure, (response_axis, force_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    response_axis.plot(time_s, np.rad2deg(controller_trace["p_command_rad_s"]), label="p_c", linewidth=1.5)
    response_axis.plot(time_s, np.rad2deg(controller_trace["p_reference_rad_s"]), label="p_ref", linewidth=2.0)
    response_axis.plot(time_s, np.rad2deg(raw_trace["p_rad_s"]), label="p_raw", linewidth=1.5)
    response_axis.plot(
        time_s,
        np.rad2deg(controller_trace["p_rad_s"]),
        label=f"p_{controller_label}",
        linewidth=2.0,
    )
    response_axis.set_title(title)
    response_axis.set_ylabel("Roll rate (deg/s)")
    response_axis.grid(alpha=0.25)
    response_axis.legend(ncol=4)

    force_axis.plot(time_s, raw_trace["f_as_n"], label="raw F_as", linewidth=1.2)
    if "requested_f_as_n" in controller_trace:
        force_axis.plot(
            time_s,
            controller_trace["requested_f_as_n"],
            label=f"{controller_label} requested F_as",
            linewidth=1.0,
            alpha=0.7,
        )
    force_axis.plot(
        time_s,
        controller_trace["f_as_n"],
        label=f"{controller_label} F_as",
        linewidth=1.5,
    )
    force_axis.set_xlabel("Time (s)")
    force_axis.set_ylabel("F_as (N)")
    force_axis.grid(alpha=0.25)
    force_axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_specialist_evaluation_grid(
    traces: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]],
    destination: str | Path,
    *,
    title: str,
    controller_label: str = "SAC",
) -> Path:
    """Save response and force traces for every held-out command without cherry-picking."""

    if not traces:
        raise ValueError("specialist evaluation grid requires at least one trace")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(traces),
        2,
        figsize=(13, 3.1 * len(traces)),
        squeeze=False,
        layout="constrained",
    )
    for row, (command_id, raw, controlled) in enumerate(traces):
        response_axis, force_axis = axes[row]
        time_s = controlled["time_s"]
        response_axis.plot(
            time_s,
            np.rad2deg(controlled["p_reference_rad_s"]),
            color="black",
            linestyle="--",
            linewidth=1.7,
            label="p_ref",
        )
        response_axis.plot(
            time_s,
            np.rad2deg(raw["p_rad_s"]),
            color="#2ca02c",
            linewidth=1.2,
            label="command-force baseline",
        )
        response_axis.plot(
            time_s,
            np.rad2deg(controlled["p_rad_s"]),
            color="#d62728",
            linewidth=1.7,
            label=controller_label,
        )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        response_axis.grid(alpha=0.25)
        response_axis.legend(loc="best")

        if "requested_f_as_n" in controlled:
            force_axis.plot(
                time_s,
                controlled["requested_f_as_n"],
                color="#9467bd",
                linewidth=1.0,
                alpha=0.7,
                label="requested",
            )
        force_axis.plot(
            time_s,
            controlled["f_as_n"],
            color="#ff7f0e",
            linewidth=1.5,
            label="applied",
        )
        force_axis.set_ylabel("F_as (N)")
        force_axis.grid(alpha=0.25)
        force_axis.legend(loc="best")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_controller_comparison_grid(
    traces: list[
        tuple[
            str,
            dict[str, np.ndarray],
            dict[str, dict[str, np.ndarray]],
        ]
    ],
    destination: str | Path,
    *,
    title: str,
) -> Path:
    """Compare raw, reference, and multiple controllers on every command."""

    if not traces:
        raise ValueError("controller comparison grid requires at least one trace")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ("#2878b5", "#c82423", "#7a5195", "#ef9b20")
    figure, axes = plt.subplots(
        len(traces),
        2,
        figsize=(14, 3.2 * len(traces)),
        squeeze=False,
        layout="constrained",
    )
    for row, (command_id, raw, controllers) in enumerate(traces):
        if not controllers:
            raise ValueError("each comparison row requires at least one controller")
        response_axis, force_axis = axes[row]
        time_s = np.asarray(raw["time_s"], dtype=float)
        response_axis.plot(
            time_s,
            np.rad2deg(raw["p_command_rad_s"]),
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="p_c",
        )
        response_axis.plot(
            time_s,
            np.rad2deg(raw["p_reference_rad_s"]),
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="p_ref",
        )
        response_axis.plot(
            time_s,
            np.rad2deg(raw["p_rad_s"]),
            color="#2f8f46",
            linewidth=1.2,
            label="Raw",
        )
        for color, (label, controlled) in zip(colors, controllers.items(), strict=False):
            controlled_time = np.asarray(controlled["time_s"], dtype=float)
            response_axis.plot(
                controlled_time,
                np.rad2deg(controlled["p_rad_s"]),
                color=color,
                linewidth=1.8,
                label=label,
            )
            force_axis.plot(
                controlled_time,
                controlled["requested_f_as_n"],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label=f"{label} requested",
            )
            force_axis.plot(
                controlled_time,
                controlled["f_as_n"],
                color=color,
                linewidth=1.5,
                label=f"{label} applied",
            )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        response_axis.grid(alpha=0.25)
        response_axis.legend(loc="best", ncol=3)
        force_axis.set_ylabel("F_as (N)")
        force_axis.grid(alpha=0.25)
        force_axis.legend(loc="best", ncol=2)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_controlled_response_error_grid(
    traces: list[
        tuple[
            str,
            dict[str, np.ndarray],
            dict[str, dict[str, np.ndarray]],
        ]
    ],
    destination: str | Path,
    *,
    title: str,
) -> Path:
    """Plot controlled responses, tracking errors, and forces without Raw scaling."""

    if not traces:
        raise ValueError("controlled comparison grid requires at least one trace")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "PID": "#2878b5",
        "RL Teacher": "#c82423",
        "Student": "#7a5195",
    }
    figure, axes = plt.subplots(
        len(traces),
        3,
        figsize=(18, 3.2 * len(traces)),
        squeeze=False,
        layout="constrained",
    )
    for row, (command_id, raw, controllers) in enumerate(traces):
        response_axis, error_axis, force_axis = axes[row]
        time_s = np.asarray(raw["time_s"], dtype=float)
        reference_deg_s = np.rad2deg(raw["p_reference_rad_s"])
        response_axis.plot(
            time_s,
            np.rad2deg(raw["p_command_rad_s"]),
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            label="p_c",
        )
        response_axis.plot(
            time_s,
            reference_deg_s,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="p_ref",
        )
        for label, controlled in controllers.items():
            color = colors.get(label, "#ef9b20")
            controlled_time = np.asarray(controlled["time_s"], dtype=float)
            response_deg_s = np.rad2deg(controlled["p_rad_s"])
            response_axis.plot(
                controlled_time,
                response_deg_s,
                color=color,
                linewidth=1.7,
                label=label,
            )
            error_axis.plot(
                controlled_time,
                response_deg_s - reference_deg_s,
                color=color,
                linewidth=1.4,
                label=label,
            )
            force_axis.plot(
                controlled_time,
                controlled["f_as_n"],
                color=color,
                linewidth=1.3,
                label=label,
            )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        response_axis.grid(alpha=0.25)
        response_axis.legend(loc="best", ncol=2)
        error_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        error_axis.set_ylabel("p - p_ref (deg/s)")
        error_axis.grid(alpha=0.25)
        error_axis.legend(loc="best")
        force_axis.set_ylabel("Applied F_as (N)")
        force_axis.grid(alpha=0.25)
        force_axis.legend(loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
