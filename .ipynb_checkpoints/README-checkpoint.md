# GAIA — lstm2 Experimental Series

Experimental validation of the syndrome algebra framework for three
measurable failure modes in large language models.

This repository contains the code and results for the lstm2
experimental series accompanying:

> **Three Measurable Failure Modes of Large Language Models**
> Marek Hubka, 2026. [Zenodo DOI — PLACEHOLDER]

> **A Syndrome Algebra for Differentiable Parametric Systems**
> Marek Hubka, 2026. [Zenodo DOI — PLACEHOLDER]

## Overview

The lstm2 series validates the GAIA syndrome algebra on a controlled
synthetic LSTM architecture. Six experiments plus one post-hoc
analysis cover:

| Script | What it tests |
|--------|---------------|
| `lstm2_1_modes_vs_k.py` | Three failure modes across k=1..10 domains |
| `lstm2_2_arch_sweep.py` | Depth/width/paired capacity sweeps |
| `lstm2_3_detectability.py` | Syndrome identification accuracy and Jacobian variance |
| `lstm2_4_causal_localisation.py` | First confirmation of Definition 4.2 |
| `lstm2_5_correction.py` | Oracle correction, crossing error, practical correction |
| `lstm2_6_singlecell_vs_multicell.py` | Specialist cells vs generalist model |

**Post-hoc analysis:**

| Script | What it computes |
|--------|-----------------|
| `postanalysis_vj_acc_correlation.py` | Within-k V_j vs identification accuracy correlation |

## Architecture

All experiments use a shared LSTM module (`lstm2_model.py`):
- D_MODEL = 256, NUM_LAYERS = 10, VOCAB_SIZE = 256
- Domain family: arithmetic slope sequences,
  `token_t = (slope_i × t + phase) % 256`
- 6 seeds: [42, 2311, 9744, 9037, 8919, 3163]

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+. GPU recommended (CUDA) but not required.

## Quick Start

```bash
# Run the shared module self-test first
python lstm2_model.py

# Run any experiment in quick mode to verify installation
python lstm2_1_modes_vs_k.py --quick

# Full run (takes several hours per script on GPU)
python lstm2_1_modes_vs_k.py
```

## Results

Pre-computed results JSON files are provided in `results/`.
Place the JSON files from your run (or downloaded from Zenodo)
in the corresponding subdirectory before running
`postanalysis_vj_acc_correlation.py`.

Saved model weights (30 × .pt files from lstm2_3) are not
included due to size (~2GB). They are regenerated deterministically
by running `lstm2_3_detectability.py` with the canonical seeds.
Scripts lstm2_4, lstm2_5, and lstm2_6 load from this directory.

## Reproducing Results

All experiments are fully deterministic given the fixed seeds.
To reproduce the complete series:

```bash
python lstm2_1_modes_vs_k.py          # ~16h on GTX 1660 Ti
python lstm2_2_arch_sweep.py --sweep C # ~8h
python lstm2_2_arch_sweep.py --sweep A # ~6h
python lstm2_2_arch_sweep.py --sweep B # ~6h
python lstm2_3_detectability.py        # ~12h (saves model weights)
python lstm2_4_causal_localisation.py  # ~2h (loads from lstm2_3)
python lstm2_5_correction.py           # ~4h (loads from lstm2_3)
python lstm2_6_singlecell_vs_multicell.py --N 5   # ~8h
python lstm2_6_singlecell_vs_multicell.py --N 10  # ~16h
python postanalysis_vj_acc_correlation.py \
    --json results/lstm2_3_detectability/lstm2_3_detectability_results.json
```

## Analysis Documents

The `analysis/` directory contains detailed analysis of each
experiment's results, including interpretation, revised predictions,
and recommendations for publication. These accompany but do not
replace the papers.

## Citation

If you use this code or data, please cite:

```bibtex
@software{hubka2026gaia,
  author    = {Hubka, Marek},
  title     = {GAIA lstm2 Experimental Series},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {PLACEHOLDER},
  url       = {https://github.com/[USERNAME]/gaia-lstm2}
}
```

## License

Code: MIT License — see LICENSE file.
Data (results JSONs, analysis documents): CC BY 4.0.
Papers: CC BY 4.0 — see Zenodo deposit.
