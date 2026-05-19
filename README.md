# Reproducible Syndrome Algebra Experiments on a Controlled LSTM Architecture

Experimental validation of the syndrome algebra framework for three
measurable failure modes in large language models.

This repository contains the code, results, and educational materials
for the lstm2 series accompanying:


> **Three Measurable Failure Modes of Large Language Models**  
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20127318.svg)](https://doi.org/10.5281/zenodo.20127318)

> **A Syndrome Algebra for Differentiable Parametric Systems**  
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20127537.svg)](https://doi.org/10.5281/zenodo.20127537)


---

## Overview

The lstm2 series validates the syndrome algebra on a controlled synthetic LSTM architecture. Six experiments plus one post-hoc analysis test specific predictions of the framework.

### Experimental scripts

| Script | What it tests |
|--------|---------------|
| `lstm2_1_modes_vs_k.py` | Three failure modes across k=1..10 domains |
| `lstm2_2_arch_sweep.py` | Depth/width/paired capacity sweeps |
| `lstm2_3_detectability.py` | Syndrome identification and Jacobian variance |
| `lstm2_4_causal_localisation.py` | Causal layer localisation (Definition 4.2) |
| `lstm2_5_correction.py` | Oracle correction, crossing error, practical correction |
| `lstm2_6_singlecell_vs_multicell.py` | Specialist cells vs generalist model |

### Post-hoc analysis

| Script | What it computes |
|--------|-----------------|
| `postanalysis_vj_acc_correlation.py` | Within-k V_j vs identification accuracy |

### Educational notebook

| File | What it demonstrates |
|------|---------------------|
| `syndrome_algebra_notebook.ipynb` | Step-by-step syndrome table construction for 3×3 and 3×4 matrices — symbolic and numerical |

---

## Architecture

All experiments use a shared LSTM module (`lstm2_model.py`):
- D_MODEL = 256, NUM_LAYERS = 10, VOCAB_SIZE = 256
- Domain family: arithmetic slope sequences,
  `token_t = (slope_i × t + phase) % 256`
- 6 canonical seeds: `[42, 2311, 9744, 9037, 8919, 3163]`

---

## Requirements

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`,
`matplotlib>=3.7.0`. The educational notebook additionally requires
`sympy>=1.12` and `jupyter`.

Tested with Python 3.10+. GPU recommended (CUDA) but not required.

---

## Quick Start

```bash
# Verify installation
python lstm2_model.py

# Run any experiment in quick mode
python lstm2_1_modes_vs_k.py --quick

# Full run (see release notes for runtimes)
python lstm2_1_modes_vs_k.py
```

---

## Reproducing the Full Series

Scripts must be run in order — lstm2_3 saves model weights that
lstm2_4, lstm2_5, and lstm2_6 load.

```bash
python lstm2_1_modes_vs_k.py
python lstm2_2_arch_sweep.py --sweep C
python lstm2_2_arch_sweep.py --sweep A
python lstm2_2_arch_sweep.py --sweep B
python lstm2_3_detectability.py          # saves models to results/lstm2_3_detectability/models/
python lstm2_4_causal_localisation.py    # loads from lstm2_3 models
python lstm2_5_correction.py             # loads from lstm2_3 models
python lstm2_6_singlecell_vs_multicell.py --N 5
python lstm2_6_singlecell_vs_multicell.py --N 10
python postanalysis_vj_acc_correlation.py \
    --json results/lstm2_3_detectability/lstm2_3_detectability_results.json
```

---

## Results

Pre-computed results are in `results/`. Each subdirectory contains
a `*_results.json` (complete numerical data) and a `*_report.html`
(human-readable summary with figures). See `results/README.md`.

**Model weights** (~2GB) are not included. They are regenerated
deterministically by running `lstm2_3_detectability.py`.

---

## Educational Notebook

`syndrome_algebra_notebook.ipynb` walks through the complete syndrome
algebra construction step by step:

- **Part I** — 3×3 matrix: Gram metric, SVD, symbolic Jacobian,
  syndrome table (linear and nonlinear)
- **Part II** — 3×4 matrix: guaranteed null space by dimension
  counting, null direction syndrome near zero
- **Part III** — perturbation injection, syndrome identification,
  oracle correction, crossing error
- **Part IV** — null space covert channel: invisible on in-distribution
  inputs, active on out-of-distribution inputs

Open in Google Colab or run locally with `jupyter notebook`.

---

## Citation

If you use this code or data, please cite:

```bibtex
@software{hubka2026lstm2,
  author    = {Hubka, Marek},
  title     = {Reproducible Syndrome Algebra Experiments on a Controlled LSTM Architecture},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {PLACEHOLDER},
  url       = {https://github.com/MarcusSkynet/lstm2}
}
```

And the accompanying papers:

```bibtex
@article{hubka2026threemodes,
  author    = {Hubka, Marek},
  title     = {Three Measurable Failure Modes of Large Language Models},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20127318}
}

@article{hubka2026syndrome,
  author    = {Hubka, Marek},
  title     = {A Syndrome Algebra for Differentiable Parametric Systems},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20127537}
}
```

---

## License

Code: MIT License — see `LICENSE`  
Data (results JSONs): CC BY 4.0  
Papers: CC BY 4.0 — see Zenodo deposit
