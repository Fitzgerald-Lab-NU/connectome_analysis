import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def summarize_vector(x, name):
    x = pd.Series(x).dropna()

    return {
        f"{name}_n": len(x),
        f"{name}_mean": x.mean(),
        f"{name}_std": x.std(),
        f"{name}_var": x.var(),
        f"{name}_cv": x.std() / x.mean() if x.mean() != 0 else np.nan,
        f"{name}_median": x.median(),
        f"{name}_iqr": x.quantile(0.75) - x.quantile(0.25),
        f"{name}_min": x.min(),
        f"{name}_max": x.max(),
        f"{name}_p05": x.quantile(0.05),
        f"{name}_p25": x.quantile(0.25),
        f"{name}_p75": x.quantile(0.75),
        f"{name}_p95": x.quantile(0.95),
    }
    
def stats_ctype_pair(full_kernels, input_types, post_type,
                     mean_thrs=0.0):
    ### Analyze input kernels
    hist_rows = []
    cell_metric_rows = []
    similarity_rows = []
    summary_rows = []

    kernel_maps = {} 
    for pre_type in input_types:
        pair_kernels = full_kernels[pre_type]

        pair_Matrix = (
            pair_kernels
            .assign(
                offset=lambda d: (
                    d["dp"].astype(int).astype(str)
                    + "_"
                    + d["dq"].astype(int).astype(str)
                )
            )
            .pivot_table(
                index="post",
                columns="offset",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .rename_axis(columns=None)
        )

        mean_vec = pair_Matrix.mean(axis=0)
        std_vec = pair_Matrix.std(axis=0)
        var_vec = pair_Matrix.var(axis=0)
        fano_vec = var_vec / mean_vec.replace(0, np.nan)
        
        fano_vec = fano_vec.fillna(0)
        fano_vec = fano_vec.where(mean_vec > mean_thrs, 0)
       
        total_input = pair_Matrix.sum(axis=1)

        max_offset_input = pair_Matrix.max(axis=1)
        dominance = max_offset_input / total_input.replace(0, np.nan)
        dominance = dominance.fillna(0)

        P = (
            pair_Matrix.loc[total_input > 0]
            .div(total_input.loc[total_input > 0], axis=0)
        )
        mean_P = P.mean(axis=0)


        sim = cosine_similarity(P)
        i, j = np.triu_indices_from(sim, k=1)
        sim_vals = sim[i, j]
        

        posts = P.index.to_numpy()
        post_i = posts[i]
        post_j = posts[j]

        # Cosine similarity of each post cell to the mean normalized kernel
        cosine_to_mean = cosine_similarity(
            P.values,
            mean_P.values.reshape(1, -1)
        ).ravel()    
        cosine_to_mean = pd.Series(
        cosine_to_mean,
        index=P.index,
        name="cosine_similarity_to_mean"
        )
        # -------------------------
        # 1. Cell-level metrics
        # -------------------------
        cell_df = pd.DataFrame({
            "pre_type": pre_type,
            "post_type": post_type,
            "post": pair_Matrix.index,
            "total_input": total_input.values,
            "dominance": dominance.values,
            "max_offset_input": max_offset_input.values,
        })
        
        cell_df["cosine_similarity_to_mean"] = (
        cosine_to_mean
        .reindex(pair_Matrix.index)
        .fillna(0)
        .values
        )

        cell_metric_rows.append(cell_df)

        # -------------------------
        # 2. Long-format histogram values
        # good for violin/strip/swarm plots
        # -------------------------
        hist_rows.append(pd.DataFrame({
            "pre_type": pre_type,
            "post_type": post_type,
            "metric": "total_input",
            "value": total_input.values,
            "post": total_input.index,
        }))

        hist_rows.append(pd.DataFrame({
            "pre_type": pre_type,
            "post_type": post_type,
            "metric": "dominance",
            "value": dominance.values,
            "post": dominance.index,
        }))

        hist_rows.append(pd.DataFrame({
            "pre_type": pre_type,
            "post_type": post_type,
            "metric": "pairwise_cosine_similarity",
            "value": sim_vals,
            "post": np.nan,
        }))

        hist_rows.append(pd.DataFrame({
            "pre_type": pre_type,
            "post_type": post_type,
            "metric": "cosine_similarity_to_mean",
            "value": cosine_to_mean.values,
            "post": cosine_to_mean.index,
        }))

        # -------------------------
        # 3. Pairwise similarity table
        # useful if you want to track which cells differ
        # -------------------------
        sim_df = pd.DataFrame({
            "pre_type": pre_type,
            "post_type": post_type,
            "post_i": post_i,
            "post_j": post_j,
            "cosine_similarity": sim_vals,
            # "cosine_sim_to_mean: 
        })

        similarity_rows.append(sim_df)

        # -------------------------
        # 4. Summary row
        # -------------------------
        row = {
            "pre_type": pre_type,
            "post_type": post_type,
            "n_post_cells": pair_Matrix.shape[0],
            "n_offsets": pair_Matrix.shape[1],
        }

        row.update(summarize_vector(total_input, "total_input"))
        row.update(summarize_vector(dominance, "dominance"))
        row.update(summarize_vector(sim_vals, "cosine_similarity"))
        row.update(summarize_vector(cosine_to_mean, "cosine_similarity_to_mean"))

        summary_rows.append(row)
        # Store plotting data
        kernel_maps[pre_type] = {
            "Mean count": mean_vec,
            "Std count": std_vec,
            "Fano Factor \n (var/mean)": fano_vec,
            "total_input": total_input,
            "dominance": dominance,
            "sim_vals": sim_vals,
            "cosine_to_mean": cosine_to_mean,
            "mean_P": mean_P,
            "pair_Matrix": pair_Matrix,
        }
        
        cell_metrics_df = pd.concat(cell_metric_rows, ignore_index=True)
        hist_df = pd.concat(hist_rows, ignore_index=True)
        similarity_df = pd.concat(similarity_rows, ignore_index=True)
        summary_df = pd.DataFrame(summary_rows)
    return cell_metrics_df, hist_df, similarity_df, summary_df, kernel_maps
