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
- Quantify variability in inputs to to motion pathway
  - Specific subtypes?
  - Correlations in the variability:
    - How the spatial pattern of input A relates back to the pattern of input B. 
- Priors for modeling. 
  - Mean spatial kernel
  - Allowed variation from mean, either on strength or shape. 

## Repository Layout

```text
.
├── utils/                         # Reusable notebook helpers
│   ├── io.py                      # Local/GCS data loading and cache helpers
│   └── plotting.py                # Coordinate conversion and plotting helpers
├── results/connectome/            # Derived Feather tables
├── optic_col_connectome/          # Notebooks to create the OL columnar connectome w/CT1
  ├── T4_T5_cols.ipynb                 # Infer T4/T5 column identities
  ├── ct1_edgelist_columnar.ipynb      # Assign CT1 synapses to columns
  ├── ct1_edgelist_full.ipynb          # Full CT1 edge-list construction
  ├── optic_columnar_connectome.ipynb  # Build analysis-ready edge lists
└── visualize_kernels.ipynb        # Plot spatial input kernels
```

## Install 
``` text
conda create -n connectome python=3.11 pip -y
conda activate connectome
cd /home/guardomayas/NUIN/sjbcs/male_cns_var
python -m pip install -e ".[analysis,notebooks]"
python -m ipykernel install --user --name connectome --display-name "Python (connectome)"
``` text 