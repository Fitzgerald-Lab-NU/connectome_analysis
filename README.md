# Male CNS Connectome Analysis

Notebook-based analyses for male CNS optic lobe connectome data.

This repository contains exploratory notebooks, reusable Python helpers, and
derived columnar connectome tables used to build analysis-ready edge lists for
T4/T5 motion pathway analyses. The main goal is to assign column identities to
neurons and synapses that do not come with explicit column annotations, enabling
spatial kernel analyses of optic lobe connectivity.

![alt text](/images/T5b_input_RF.png)


## Project Overview

The male CNS connectome provides column identities for a subset of optic-lobe
cell types. This project uses those annotated cell types as anchors to infer
additional columnar structure, including:

- column identities for T4 and T5 neurons;
- CT1-associated column assignments based on synapse locations;
- analysis-ready edge lists with pre- and postsynaptic cell types, column
  coordinates, synapse counts, and spatial offsets;
- spatial input kernels onto T4/T5 subtypes.

These derived tables are saved under `results/` and are used for downstream
visualization and analysis of columnar connectivity.

**Goals:**
- Quantify variability
- Priors for modeling. 

## Repository Layout

```text
.
├── utils/                         # Reusable notebook helpers
│   ├── io.py                      # Local/GCS data loading and cache helpers
│   └── plotting.py                # Coordinate conversion and plotting helpers
├── results/connectome/            # Derived Feather tables
├── create_analysis_edgelist.ipynb # Build analysis-ready edge lists
├── optic_columnar_connectome.ipynb
├── T4_T5_cols.ipynb               # Infer T4/T5 column identities
├── ct1_edgelist_columnar.ipynb    # Assign CT1 synapses to columns
├── ct1_edgelist_full.ipynb        # Full CT1 edge-list construction
└── visualize_kernels.ipynb        # Plot spatial input kernels
