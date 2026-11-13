"""Plotting helpers for columnar connectome notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,  SymLogNorm
from matplotlib.patches import RegularPolygon
import matplotlib as mpl
import math
import seaborn as sns
from .hex_geometry import pq_to_xy, hex_distance

### TODO: 
### Frame it as a connectome_view class object. 

### DF VECTOR TO KERNEL FOR PLOTTING


def plot_stats(plot_df, ax, color,metric):
   
    sns.violinplot(
    data=plot_df,
    x="pre_type",
    y="value",
    inner=None,
    cut=0,
    color= color,
    ax= ax
    )
    
    sns.pointplot(
    data=plot_df,
    x="pre_type",
    y="value",
    estimator=np.median,
    errorbar=None,
    markers="_",
    linestyles="none",
    color="black",
    # scale=1.5,
    ax=ax
    )
    
    # ax.set_title(f"{metric} distribution")
    # ax.set_xlabel("Input type")
    ax.set_ylabel(metric)


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
    emphasize_center=False,
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
    if emphasize_center:
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

## Helper to get random ids for plotting
def _get_sample_ids(full_kernels, n_samples, seed):
    valid_post_cells = (
            full_kernels
            .dropna(subset=["post", "target_p", "target_q"])
            [["post", "target_p", "target_q"]]
            .drop_duplicates()
        )

    n_samples = min(n_samples, len(valid_post_cells))

    post_ids = (
        valid_post_cells["post"]
        .sample(n=n_samples, random_state=seed)
        .to_list()
    )
    return post_ids

def plot_example_RFs(kernels, 
                     n_samples = 20,
                     max_cols = 6,
                     seed=1,
                     emphasize_center=False,
                     signed_norm = True,
                     value_col = "signed_count",
                     plot_labels = False,
                     cmap = "bwr"):
    
    post_ids  = _get_sample_ids(kernels, n_samples, seed=seed)
    # only values from sampled cells
    plot_df = kernels[kernels["post"].isin(post_ids)].copy()

    vmax = np.abs(plot_df[value_col]).max()
    if signed_norm:
        shared_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else: 
        shared_norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    n = len(post_ids)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    fig.suptitle(
        f"{plot_df['pre_cell_type'].iloc[0]} to "
        f"{plot_df['post_cell_type'].iloc[0]} example kernels",
        fontsize=18,
    )

    axes_flat = axes.ravel()
    for ax, post_id in zip(axes_flat, post_ids):
        ax.axis("off")
        cell_df = plot_df[plot_df['post']==post_id]
        plot_spatial_kernel_hex(cell_df, value_col=value_col, 
                            radius=4, cmap=cmap, 
                            norm = shared_norm, 
                            add_colorbar=False,
                            plot_labels=plot_labels, 
                            ax=ax,use_input_relative_to_target=True,
                            mirror_x=True,
                            mirror_y=True,
                            title=post_id,
                            emphasize_center=emphasize_center)
    for ax in axes_flat[n:]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=shared_norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])

    cax = fig.add_axes([0.9, 0.25, 0.02, 0.5]) # [left, bottom, width, height]
    cbar = fig.colorbar(
        sm,
        cax=cax,
        shrink=0.4,
        location="right",
        label="Mean signed synapse count",
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(value_col, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) #[left, bottom, right, top]
    # plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.show()
    
def plot_kernels_on_eye(full_kernels, n_samples, 
                        fontsize = 16, 
                        size= (8, 8),
                        seed=1):
        df = full_kernels.copy()

        post_ids  = _get_sample_ids(full_kernels, n_samples, seed=seed)
        plot_df = full_kernels[full_kernels["post"].isin(post_ids)].copy()
        # Absolute presynaptic/input column coordinates
        plot_df["input_p"] = plot_df["target_p"] - plot_df["dp"]
        plot_df["input_q"] = plot_df["target_q"] - plot_df["dq"]

        # Collapse in case multiple rows map to same input column for same post cell
        plot_df = (
            plot_df
            .groupby(["post", "target_p", "target_q", "input_p", "input_q"], as_index=False)
            .agg(signed_count=("signed_count", "sum"))
        )

        # Background eye map: all T5 home columns in this full table
        eye = (
            df[["target_p", "target_q"]]
            .dropna()
            .drop_duplicates()
            .rename(columns={"target_p": "p", "target_q": "q"})
        )

        x_eye, y_eye = pq_to_xy(
            eye["p"].to_numpy(),
            eye["q"].to_numpy(),
            hexsize=1.0,
            rotate=-np.pi / 6,
        )

        eye["x_plot"] = x_eye
        eye["y_plot"] = y_eye

        # Input columns for the selected cells
        x_in, y_in = pq_to_xy(
            plot_df["input_p"].to_numpy(),
            plot_df["input_q"].to_numpy(),
            hexsize=1.0,
            rotate=-np.pi / 6,
        )

        plot_df["x_plot"] = x_in
        plot_df["y_plot"] = y_in

        # Home columns for the selected cells
        homes = (
            plot_df[["post", "target_p", "target_q"]]
            .drop_duplicates()
            .rename(columns={"target_p": "p", "target_q": "q"})
        )

        x_home, y_home = pq_to_xy(
            homes["p"].to_numpy(),
            homes["q"].to_numpy(),
            hexsize=1.0,
            rotate=-np.pi / 6,
        )

        homes["x_plot"] = x_home
        homes["y_plot"] = y_home

        mirror_y = True
        if mirror_y:
            eye["y_plot"] = -eye["y_plot"]
            plot_df["y_plot"] = -plot_df["y_plot"]
            homes["y_plot"] = -homes["y_plot"]
        hexsize = 1.0
        cmap = plt.get_cmap("bwr")
        vmax = np.nanmax(np.abs(plot_df['signed_count']))

        norm = TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0,
            vmax=vmax
        )

        # norm = SymLogNorm(
        #     linthresh=1,      # values between -1 and 1 are linear
        #     linscale=1,
        #     vmin=-vmax,
        #     vmax=vmax,
        #     base=10
        # )

        fig, ax = plt.subplots(figsize=size)

        # 1. Plot all T columns as pale background
        for _, row in eye.iterrows():
            ax.add_patch(
                RegularPolygon(
                    (row["x_plot"], row["y_plot"]),
                    numVertices=6,
                    radius=hexsize * 0.95,
                    orientation=np.pi / 6,
                    facecolor="lightgray",
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=0.6,
                )
            )

        # 2. Plot selected input columns colored by synapse count
        # Draw only nonzero input bins if desired
        nonzero_inputs = plot_df[plot_df["signed_count"] != 0].copy()

        for _, row in nonzero_inputs.iterrows():
            ax.add_patch(
                RegularPolygon(
                    (row["x_plot"], row["y_plot"]),
                    numVertices=6,
                    radius=hexsize * 0.95,
                    orientation=np.pi / 6,
                    facecolor=cmap(norm(row["signed_count"])),
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.9,
                )
            )

        # 3. Highlight the three T home columns by darkening/outline
        for _, row in homes.iterrows():
            ax.add_patch(
                RegularPolygon(
                    (row["x_plot"], row["y_plot"]),
                    numVertices=6,
                    radius=hexsize * 0.95,
                    orientation=np.pi / 6,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=2.5,
                    hatch="///",
                )
            )

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.75, label="input synapse count")

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")

        fig.suptitle(
            f"{full_kernels['pre_cell_type'].iloc[0]} to "
            f"{full_kernels['post_cell_type'].iloc[0]} example kernels",
            fontsize=fontsize,
        )

        plt.show()
        
def receptive_fields_grid(kernels, value_col, 
                          input_types, post_type, 
                          plot_labels=False,
                          emphasize_center=True):
    all_vals = np.concatenate([
        k[value_col].to_numpy()
        for k in kernels.values()
    ])

    vmax = np.nanmax(np.abs(all_vals))*0.8

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    # norm = SymLogNorm(
    #     linthresh=1,      # values between -1 and 1 are linear
    #     linscale=1,
    #     vmin=-vmax,
    #     vmax=vmax,
    #     base=10
    # )

    import math
    n = len(input_types)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * ncols, 2 * nrows),
        squeeze=False
    )

    axes_flat = axes.flatten()

    for ax, pre_type in zip(axes_flat, input_types):
        plot_spatial_kernel_hex(
            kernels[pre_type],
            value_col=value_col,
            radius=2,
            cmap="bwr",
            norm=norm,
            ax=ax,
            title=f"{pre_type} → {post_type}",
            plot_labels=plot_labels, 
            use_input_relative_to_target=True,
            mirror_x=True,
            mirror_y=True,
            emphasize_center=emphasize_center
        )


    for ax in axes_flat:
        ax.axis("off")

    # Common colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("bwr"))
    sm.set_array([])
    plt.tight_layout(rect=[0, 0, 0.88, 1]) #[left, bottom, right, top]

    cax = fig.add_axes([0.97, 0.25, 0.02, 0.5]) # [left, bottom, width, height]
    cbar = fig.colorbar(
        sm,
        cax=cax,
        shrink=0.4,
        location="right",
        label="Mean signed synapse count"
    )


    fig.tight_layout()
    plt.show()