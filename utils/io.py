"""Input/output helpers for connectome analysis notebooks."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd

VALID_FILE_TYPES = {
    "meta",
    "edgelist",
    "edgelist_simple",
    "synapses",
    "skeletons",
}

REPO = Path(__file__).resolve().parents[1]          # .../male_cns_var
DATA = REPO / "connectome_data"                     # raw + processed (untracked)
RESULTS = REPO / "results" / "connectome"           # tracked outputs

def setup_gcs(token: str = "google_default"):
    """Return a Google Cloud Storage filesystem."""
    import gcsfs

    return gcsfs.GCSFileSystem(token=token)


def construct_path(
    data_root: str,
    dataset: str,
    file_type: str = "meta",
    space_suffix: str | None = None,
    subset: str | None = None,
) -> str:
    """Construct a dataset path for local or ``gs://`` connectome data.

    Parameters
    ----------
    data_root
        Root data directory, for example ``connectome_data`` or ``gs://bucket/data``.
    dataset
        Dataset name with version, for example ``banc_746``.
    file_type
        One of ``meta``, ``edgelist``, ``edgelist_simple``, ``synapses``, or
        ``skeletons``.
    space_suffix
        Skeleton-space suffix. Defaults to ``{dataset_name}_space``.
    subset
        Optional subset directory/name, such as ``mushroom_body``.
    """
    if file_type not in VALID_FILE_TYPES:
        choices = ", ".join(sorted(VALID_FILE_TYPES))
        raise ValueError(f"Unknown file_type: {file_type}. Choose one of: {choices}")

    dataset_name = dataset.split("_", maxsplit=1)[0]
    extension = {
        "meta": ".feather",
        "edgelist": ".feather",
        "edgelist_simple": ".feather",
        "synapses": ".parquet" if subset is None else ".feather",
        "skeletons": "",
    }[file_type]

    if subset is not None:
        subdir = f"{dataset_name}/{subset}"
        if file_type == "skeletons":
            space_suffix = space_suffix or f"{dataset_name}_space"
            suffix = "l2_swc" if dataset_name == "banc" else "swc"
            filename = f"{dataset_name}_{subset}_{space_suffix}_{suffix}{extension}"
        elif file_type == "edgelist_simple":
            filename = f"{dataset}_{subset}_simple_edgelist{extension}"
        else:
            filename = f"{dataset}_{subset}_{file_type}{extension}"
    else:
        subdir = dataset_name
        if file_type == "skeletons":
            space_suffix = space_suffix or f"{dataset_name}_space"
            suffix = "l2_swc" if dataset_name == "banc" else "swc"
            filename = f"{dataset_name}_{space_suffix}_{suffix}{extension}"
        elif file_type == "edgelist_simple":
            filename = f"{dataset}_simple_edgelist{extension}"
        else:
            filename = f"{dataset}_{file_type}{extension}"

    return f"{data_root.rstrip('/')}/{subdir}/{filename}"


def read_feather_gcs(
    path: str | os.PathLike[str],
    gcs_fs=None,
    cache_dir: str | os.PathLike[str] = ".cache",
    use_cache: bool = True,
):
    """Read a Feather file from local disk or Google Cloud Storage."""
    import pandas as pd
    import pyarrow.feather as feather

    path_str = str(path)
    if not path_str.startswith("gs://"):
        df = pd.read_feather(path_str)
        print(f"Loaded {len(df):,} rows")
        return df

    if gcs_fs is None:
        raise ValueError("gcs_fs is required for gs:// paths")

    cache_path = _cache_path(path_str, cache_dir)
    if use_cache and cache_path.exists():
        df = pd.read_feather(cache_path)
        print(f"Loaded {len(df):,} rows from cache")
        return df

    content = _read_gcs_file(gcs_fs, path_str)
    df = feather.read_feather(io.BytesIO(content))
    print(f"Loaded {len(df):,} rows from GCS")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(cache_path)
        print(f"Cached to {cache_path}")

    return df


def read_parquet_gcs(
    path: str | os.PathLike[str],
    gcs_fs=None,
    columns: Sequence[str] | None = None,
    cache_dir: str | os.PathLike[str] = ".cache",
    use_cache: bool = True,
):
    """Read a Parquet file from local disk or Google Cloud Storage."""
    import pandas as pd
    import pyarrow.parquet as pq

    path_str = str(path)
    if not path_str.startswith("gs://"):
        df = pd.read_parquet(path_str, columns=columns)
        print(f"Loaded {len(df):,} rows")
        return df

    if gcs_fs is None:
        raise ValueError("gcs_fs is required for gs:// paths")

    cache_path = _cache_path(path_str, cache_dir)
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path, columns=columns)
        print(f"Loaded {len(df):,} rows from cache")
        return df

    content = _read_gcs_file(gcs_fs, path_str)
    df = pq.read_table(io.BytesIO(content), columns=columns).to_pandas()
    print(f"Loaded {len(df):,} rows from GCS")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pq.read_table(io.BytesIO(content)).to_pandas() if columns else df
        cache_df.to_parquet(cache_path)
        print(f"Cached to {cache_path}")

    return df


def read_swc_from_gcs(gcs_fs, swc_path: str):
    """Read a single SWC file from Google Cloud Storage using navis."""
    import navis

    with gcs_fs.open(_strip_gs_prefix(swc_path), "rb") as handle:
        return navis.read_swc(io.BytesIO(handle.read()))


def batch_read_swc_from_gcs(
    gcs_fs,
    directory: str,
    filenames: Iterable[str],
    show_progress: bool = True,
):
    """Read multiple SWC files from Google Cloud Storage into a NeuronList."""
    import navis
    from tqdm.auto import tqdm

    iterator = tqdm(filenames, desc="Reading neurons") if show_progress else filenames
    neurons = []

    for filename in iterator:
        swc_path = f"{directory.rstrip('/')}/{filename}"
        try:
            neurons.append(read_swc_from_gcs(gcs_fs, swc_path))
        except Exception as exc:
            if show_progress:
                print(f"Error reading {filename}: {exc}")

    return navis.NeuronList(neurons)


def read_obj_from_gcs(gcs_fs, obj_path: str):
    """Read an OBJ mesh file from Google Cloud Storage."""
    import trimesh

    with gcs_fs.open(_strip_gs_prefix(obj_path), "rb") as handle:
        content = handle.read()

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return trimesh.load_mesh(tmp_path)
    finally:
        os.unlink(tmp_path)


def save_figure(fig, filename: str | os.PathLike[str], format: str = "png", **kwargs) -> None:
    """Save a matplotlib or plotly figure."""
    if hasattr(fig, "write_html"):
        if format == "html":
            fig.write_html(filename, **kwargs)
        elif format == "png":
            fig.write_image(filename, **kwargs)
        else:
            raise ValueError(f"Unsupported plotly format: {format}")
    else:
        fig.savefig(filename, format=format, **kwargs)

    print(f"Saved figure: {filename}")


def save_plot(
    fig,
    name: str,
    img_dir: str | os.PathLike[str] | None = None,
    format: str = "png",
    **kwargs,
) -> None:
    """Save a figure under ``img_dir`` using ``name`` and ``format``."""
    if img_dir is None:
        import sys

        img_dir = sys._getframe(1).f_globals.get("IMG_DIR", ".")

    filename = Path(img_dir) / f"{name}.{format}"
    save_figure(fig, filename, format=format, **kwargs)


def _read_gcs_file(gcs_fs, path: str) -> bytes:
    from tqdm.auto import tqdm

    gcs_path = _strip_gs_prefix(path)
    file_size = None

    try:
        file_size = gcs_fs.info(gcs_path).get("size")
    except Exception:
        pass

    with gcs_fs.open(gcs_path, "rb") as handle:
        if not file_size:
            return handle.read()

        chunks = []
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            while chunk := handle.read(1024 * 1024):
                chunks.append(chunk)
                pbar.update(len(chunk))

    return b"".join(chunks)


def _cache_path(path: str, cache_dir: str | os.PathLike[str]) -> Path:
    cache_filename = _strip_gs_prefix(path).replace("/", "_")
    return Path(cache_dir) / cache_filename


def _strip_gs_prefix(path: str) -> str:
    return path.removeprefix("gs://")

# utils/io.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # repo root, works from any folder
CONNECTOME_DIR = PROJECT_ROOT / "results" / "connectome"

def load_columnar_connectome(norm: bool = True):
    """Return (edges, nodes) for the optic-lobe columnar connectome.

    norm=True uses the T4/T5 column assignment based on normalized input counts
    (*_T4T5norm files); norm=False uses the raw-count assignment.
    """
    suffix = "_T4T5norm" if norm else ""
    edges = pd.read_feather(CONNECTOME_DIR / f"edgelist_OL_columnar{suffix}.feather")
    nodes = pd.read_feather(CONNECTOME_DIR / f"nodes_OL_columnar{suffix}.feather")
    return edges, nodes
