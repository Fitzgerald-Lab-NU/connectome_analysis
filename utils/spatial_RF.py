### SPATIAL RF UTILS ####
from .plotting import pq_to_xy, hex_distance, hex_offsets
def _get_post_units(edges, df, post_type, same_side):
    post_units = (
        edges.query("post_cell_type == @post_type")
        [["post", "target_p", "target_q", "post_side"]]
        .dropna(subset=["post", "target_p", "target_q", "post_side"])
        .drop_duplicates()
        .copy()
    )

    if same_side:
        valid_sides = df["post_side"].dropna().unique()
        post_units = post_units[
            post_units["post_side"].isin(valid_sides)
        ].copy()

    return post_units


def _get_ct1_post_units(df):
    post_units = (
        df[["target_p", "target_q", "post_side"]]
        .dropna(subset=["target_p", "target_q", "post_side"])
        .drop_duplicates()
        .copy()
    )

    post_units["target_p"] = post_units["target_p"].astype(int)
    post_units["target_q"] = post_units["target_q"].astype(int)

    post_units["post"] = (
        post_units["post_side"].astype(str)
        + "_"
        + post_units["target_p"].astype(str)
        + "_"
        + post_units["target_q"].astype(str)
    )

    return post_units


# def _get_observed(df, group_cols):
#     observed = (
#         df
#         .dropna(subset=group_cols)
#         .copy()
#     )

#     observed["dp"] = observed["dp"].astype(int)
#     observed["dq"] = observed["dq"].astype(int)

#     return (
#         observed
#         .groupby(group_cols, as_index=False)
#         .agg(
#             count=("count", "sum"),
#             signed_count=("signed_count", "sum"),
#         )
#     )


def _add_ct1_post_ids(df):
    df = (
        df
        .dropna(subset=["target_p", "target_q", "post_side"])
        .copy()
    )

    df["target_p"] = df["target_p"].astype(int)
    df["target_q"] = df["target_q"].astype(int)

    df["post"] = (
        df["post_side"].astype(str)
        + "_"
        + df["target_p"].astype(str)
        + "_"
        + df["target_q"].astype(str)
    )

    return df

def _zero_fill_kernel(post_units, offsets, observed, merge_keys=("post", "dp", "dq")):
    full_index = (
        post_units
        .assign(key=1)
        .merge(offsets.assign(key=1), on="key")
        .drop(columns="key")
    )

    full = full_index.merge(
        observed,
        on=list(merge_keys),
        how="left",
    )

    return full

##### COMPUTE KERNEL MEAN FUNCTION ### WIP 
def compute_kernel(
    edges,
    pre_type="Mi1",
    post_type="T4b",
    same_side=True,
    radius=2
):
    """
    Compute a zero-filled mean spatial connectivity kernel from pre_type to post_type.

    For ordinary postsynaptic cell types, each postsynaptic cell is treated as one
    target unit. For CT1, each target column/side is treated as a pseudo-postsynaptic
    compartment.
    """
    df = edges.query(
        "pre_cell_type == @pre_type and post_cell_type == @post_type"
    ).copy()

    if same_side:
        df = df.query("pre_side == post_side").copy()

    df["signed_count"] = df["count"] * df["sign"]

    offsets = hex_offsets(radius)

    if post_type == "CT1":
        post_units = _get_ct1_post_units(df)
        df = _add_ct1_post_ids(df)

    else:
        post_units = _get_post_units(edges, df, post_type, same_side)

        valid_posts = set(post_units["post"])
        df = df[df["post"].isin(valid_posts)].copy()

    observed = (
            df
            .dropna(subset=["post", "dp", "dq"])
            .assign(
                dp=lambda d: d["dp"].astype(int),
                dq=lambda d: d["dq"].astype(int),
            )
            .groupby(["post", "dp", "dq"], as_index=False)
            .agg(
                count=("count", "sum"),
                signed_count=("signed_count", "sum"),
            )
        )

    full = _zero_fill_kernel(
        post_units=post_units,
        offsets=offsets,
        observed=observed,
        merge_keys=("post", "dp", "dq"),
    )

    full["count"] = full["count"].fillna(0)
    full["signed_count"] = full["signed_count"].fillna(0)

    # Reconstruct absolute input/source column coordinates.
    # Check sign convention: this assumes dp = source_p - target_p. #Not strictyl right!
    full["source_p"] = full["target_p"] + full["dp"]
    full["source_q"] = full["target_q"] + full["dq"]

    full["pre_cell_type"] = pre_type
    full["post_cell_type"] = post_type

    kernel = (
        full
        .groupby(["dp", "dq"], as_index=False)
        .agg(
            mean_count=("count", "mean"),
            mean_signed_count=("signed_count", "mean"),
        )
        .sort_values("mean_signed_count", ascending=False)
    )

    return kernel, full