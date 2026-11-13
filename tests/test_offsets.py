"""dp = target_p - source_p; every partner column must equal its home column in nodes."""
from pathlib import Path
import pandas as pd
import pytest
from utils.spatial_RF import compute_kernel

DATA = Path(__file__).parents[1] / "results/connectome"
VERSIONS = ["", "_T4T5norm"]

def load(s):
    return (pd.read_feather(DATA / f"edgelist_OL_columnar{s}.feather"),
            pd.read_feather(DATA / f"nodes_OL_columnar{s}.feather"))

@pytest.mark.parametrize("s", VERSIONS)
def test_dp_is_target_minus_source(s):
    e = load(s)[0].dropna(subset=["dp", "dq", "source_p", "target_p"])
    assert (e.dp == e.target_p - e.source_p).all() and (e.dq == e.target_q - e.source_q).all()

@pytest.mark.parametrize("s", VERSIONS)
@pytest.mark.parametrize("role,col", [("post", "target"), ("pre", "source")])
def test_partner_columns_match_nodes(s, role, col):
    e, n = load(s); home = n.set_index("malecns_09_id")
    e = e[(e[f"{role}_cell_type"] != "CT1") & e[role].isin(home.index)].dropna(subset=[f"{col}_p"])
    bad = (e[f"{col}_p"].to_numpy() != home.loc[e[role], "p"].to_numpy()) | \
          (e[f"{col}_q"].to_numpy() != home.loc[e[role], "q"].to_numpy())
    assert not bad.any(), f"{bad.sum()} rows disagree"

@pytest.mark.parametrize("s", VERSIONS)
def test_compute_kernel_recovers_sources(s):
    e = load(s)[0]
    _, full = compute_kernel(e, "Mi1", "T4b", radius=2)
    assert not full.duplicated(["post", "dp", "dq"]).any(), "a post cell has >1 home column"
    truth = e.query("pre_cell_type == 'Mi1' and post_cell_type == 'T4b'")
    obs = full[full["count"] > 0].merge(
        truth[["post", "dp", "dq", "source_p", "source_q"]].drop_duplicates(["post", "dp", "dq"]),
        on=["post", "dp", "dq"], suffixes=("", "_true"))
    assert len(obs) and (obs.source_p == obs.source_p_true).all() and (obs.source_q == obs.source_q_true).all()