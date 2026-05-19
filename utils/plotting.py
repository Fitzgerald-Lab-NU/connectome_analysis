"""Plotting helpers for columnar connectome notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,  SymLogNorm
from matplotlib.patches import RegularPolygon

### HELPER FUNCTIONS###
def pq_to_xy(p, q, flip_p=True, hexsize: float = 1.0, rotate: float | None = -np.pi / 6):
    """Convert Zhao-style hex coordinates to 2D xy coordinates."""
    a = q
    if flip_p:
        b = -p
    else:
        b = p
    x = hexsize * np.sqrt(3) * (a + b / 2)
    y = hexsize * 1.5 * b

    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        xr = c * x - s * y
        yr = s * x + c * y
        return xr, yr

    return x, y

def hex_distance(dp, dq):
    return np.maximum.reduce([
        np.abs(dp),
        np.abs(dq),
        np.abs(dq - dp)
    ])


def hex_offsets(radius):
    rows = []

    for dp in range(-radius, radius + 1):
        for dq in range(-radius, radius + 1):
            if hex_distance(dp, dq) <= radius:
                rows.append((dp, dq))

    return pd.DataFrame(rows, columns=["dp", "dq"])


### PLOTTING FUNCTIONS ####
def plot_spatial_kernel_hex(
    kernel,
    value_col="mean_signed_syn_count",
    *,
    radius=None,
    hexsize=1.0,
    cmap="bwr",
    norm=None,
    vmax=None,
    use_input_relative_to_target=True,   # <- important
    mirror_x=True,                      # <- visual-only
    mirror_y=True,                      # <- visual-only
    plot_labels=True,
    title=None,
    ax=None,
    add_colorbar=False,
    fs=14
):
    """
    Plot spatial kernel using dp/dq relative hex coordinates.

    Assumes stored convention:
        dp = post_p - pre_p
        dq = post_q - pre_q

    If use_input_relative_to_target=True, plots the input location relative
    to the target, i.e. uses (-dp, -dq).

    mirror_x / mirror_y are display-only perspective flips in xy space.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    kernel = kernel.copy()
    kernel = kernel.dropna(subset=["dp", "dq", value_col])

    kernel["dp"] = kernel["dp"].astype(int)
    kernel["dq"] = kernel["dq"].astype(int)

    # This is the mathematically correct conversion
    if use_input_relative_to_target:
        plot_dp = -kernel["dp"].to_numpy()
        plot_dq = -kernel["dq"].to_numpy()
    else:
        plot_dp = kernel["dp"].to_numpy()
        plot_dq = kernel["dq"].to_numpy()

    if radius is not None:
        dist = hex_distance(plot_dp, plot_dq)
        keep = dist <= radius
        kernel = kernel.iloc[np.where(keep)[0]].copy()
        plot_dp = plot_dp[keep]
        plot_dq = plot_dq[keep]

    ax.set_aspect("equal")
    cmap_obj = plt.get_cmap(cmap)

    vals = kernel[value_col].to_numpy()

    if norm is None:
        vmax = np.nanmax(np.abs(vals))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    x, y = pq_to_xy(
        plot_dp,
        plot_dq,
        hexsize=hexsize,
        rotate=-np.pi / 6
    )

    # Visual perspective flips only
    if mirror_x:
        x = -x
    if mirror_y:
        y = -y

    for xi, yi, val in zip(x, y, vals):
        ax.add_patch(
            RegularPolygon(
                (xi, yi),
                numVertices=6,
                radius=hexsize,
                orientation=np.pi / 6,
                facecolor=cmap_obj(norm(val)),
                edgecolor="black",
                linewidth=0.6
            )
        )

        if plot_labels:
            ax.text(
                xi, yi,
                f"{val:.1f}",
                fontsize=7,
                ha="center",
                va="center",
                color="black"
            )

    # Draw center hex ONCE, not inside the loop
    ax.add_patch(
        RegularPolygon(
            (0, 0),
            numVertices=6,
            radius=hexsize,
            orientation=np.pi / 6,
            facecolor="none",
            edgecolor="darkgray",
            linewidth=2.5,
            hatch="///",
        )
    )

    if len(x) > 0:
        pad = 2 * hexsize
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if title is None:
        title = value_col
    ax.set_title(title)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=value_col)

    return ax