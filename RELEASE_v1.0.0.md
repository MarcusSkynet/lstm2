# # Reproducible Syndrome Algebra Experiments on a Controlled LSTM Architecture
version: `v1.0.0`

## Initial Release — May 2026

First public release of the lstm2 experimental series, accompanying
two papers:

> **Three Measurable Failure Modes of Large Language Models**  
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20127318.svg)](https://doi.org/10.5281/zenodo.20127318)

> **A Syndrome Algebra for Differentiable Parametric Systems**  
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20127537.svg)](https://doi.org/10.5281/zenodo.20127537)

---

## What is included

### Code — experimental series

- `lstm2_model.py` — shared LSTM module: architecture, domain
  generator, training loop, perturbation utilities, syndrome
  measurement protocol
- `lstm2_1_modes_vs_k.py` — three failure modes across k=1..10
- `lstm2_2_arch_sweep.py` — width, depth, and paired capacity sweeps
- `lstm2_3_detectability.py` — syndrome identification and Jacobian
  variance; saves model weights for downstream scripts
- `lstm2_4_causal_localisation.py` — first experimental confirmation
  of causal layer localisation (Definition 4.2)
- `lstm2_5_correction.py` — oracle correction, crossing error, and
  practical correction quality
- `lstm2_6_singlecell_vs_multicell.py` — specialist cells vs
  generalist model (N=5 and N=10)
- `postanalysis_vj_acc_correlation.py` — post-hoc per-direction
  V_j vs identification accuracy correlation

### Code — educational notebook

- `syndrome_algebra_notebook.ipynb` — step-by-step construction of
  the syndrome table for 3×3 and 3×4 matrices. Symbolic derivation
  (linear case) and numerical evaluation (tanh nonlinearity). Covers
  Gram metric, SVD, Jacobian averaging, syndrome table construction,
  perturbation injection, identification, oracle correction, crossing
  error, and null space covert channel. Runs in Google Colab or any
  standard Jupyter environment.

### Results

Pre-computed results JSON files and HTML reports for all six
experiments. Each `results/` subdirectory contains:
- `*_results.json` — complete numerical results, all seeds, all metrics
- `*_report.html` — human-readable summary with figures and
  confirmation criteria, generated directly by the script

See `results/README.md` for file placement instructions.

---

## Key experimental results

| Experiment | Key result |
|---|---|
| lstm2_1 | CS_gap 0.273→0.067 (k=1..10); r(DE, CS_unknown)=0.9896 |
| lstm2_2 | Depth phase transition L=2→L=4; two-regime structure |
| lstm2_3 | Multi-layer identification 40–99× random baseline; JUP floor σ_min=0.194 |
| lstm2_4 | Causal localisation 100% accuracy 180/180 trials; pre/post ~2×10⁸ |
| lstm2_5 | Oracle cosine=1.000000 (36K trials); conditional split 0.099 vs 1.941 |
| lstm2_6 | Two complementary profiles diverge with N; Singleton bound 0.158→0.310 |

---

## What is not included

**Model weights** (~2GB, 30 × .pt files from lstm2_3) are not
included due to size. Regenerate deterministically by running
`lstm2_3_detectability.py` with the canonical seeds
`[42, 2311, 9744, 9037, 8919, 3163]`.

Scripts lstm2_4, lstm2_5, and lstm2_6 load from
`results/lstm2_3_detectability/models/`. Run lstm2_3 first.

---

## Hardware and runtime

All experiments: NVIDIA GeForce GTX 1660 Ti.

| Script | Runtime |
|---|---|
| lstm2_1 (k=1..10, 6 seeds) | ~16h |
| lstm2_2 Sweep A | ~6h |
| lstm2_2 Sweep B | ~6h |
| lstm2_2 Sweep C | ~8h |
| lstm2_3 | ~12h |
| lstm2_4 | ~2h |
| lstm2_5 | ~4h |
| lstm2_6 N=5 | ~8h |
| lstm2_6 N=10 | ~16h |

All scripts support `--quick` mode (~5–15 min per script).

---

## Reproducibility

All experiments are fully deterministic given the fixed seeds.
Architecture: D_MODEL=256, NUM_LAYERS=10, VOCAB_SIZE=256.
Domain family: arithmetic slope sequences,
`token_t = (slope_i × t + phase) % 256`.
Training: SEQS_PER_DOMAIN=2000, EPOCHS=200, LR=1e-3.

One convergence failure in the N=10 specialist series
(domain 7, seeds 9037 and 3163 — even slope creates sparse
sequence structure). Four of six seeds converged; aggregate
results are robust. All reported numbers use the canonical
6-seed protocol `[42, 2311, 9744, 9037, 8919, 3163]`.

---

## Citation

```bibtex
@software{hubka2026lstm2,
  author    = {Hubka, Marek},
  title     = {Reproducible Syndrome Algebra Experiments on a Controlled LSTM Architecture},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {PLACEHOLDER},
  url       = {https://github.com/MarcusSkynet/lstm2}
}

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

Code: MIT License  
Data (results JSONs): CC BY 4.0
