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
