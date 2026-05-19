# Results

Pre-computed results for all six lstm2 experiments.

Each subdirectory contains:
- `*_results.json` — complete numerical output: all seeds, all
  metrics, per-run records, aggregate tables, confirmation criteria
- `*_report.html` — human-readable summary with figures, generated
  directly by the script from the results JSON

---

## Directory structure

```
results/
├── lstm2_1_modes_vs_k/
│   ├── lstm2_1_modes_vs_k_results.json
│   └── lstm2_1_modes_vs_k_report.html
│
├── lstm2_2_arch_sweep/
│   ├── lstm2_2_arch_sweep_A_results.json
│   ├── lstm2_2_arch_sweep_B_results.json
│   ├── lstm2_2_arch_sweep_C_results.json
│   └── lstm2_2_arch_sweep_report.html
│
├── lstm2_3_detectability/
│   ├── lstm2_3_detectability_results.json
│   ├── lstm2_3_detectability_report.html
│   ├── postanalysis_vj_acc_correlation.json
│   └── models/              ← model weights (not included, regenerate)
│
├── lstm2_4_causal_localisation/
│   ├── lstm2_4_causal_localisation_results.json
│   └── lstm2_4_causal_localisation_report.html
│
├── lstm2_5_correction/
│   ├── lstm2_5_correction_results.json
│   └── lstm2_5_correction_report.html
│
└── lstm2_6_singlecell_vs_multicell/
    ├── lstm2_6_N5_results.json
    ├── lstm2_6_N10_results.json
    └── lstm2_6_singlecell_vs_multicell_report.html
```

---

## Model weights

The `results/lstm2_3_detectability/models/` directory is not included
(~2GB, 30 × .pt files). Regenerate by running:

```bash
python lstm2_3_detectability.py
```

with the canonical seeds `[42, 2311, 9744, 9037, 8919, 3163]`.
All subsequent scripts (lstm2_4, lstm2_5, lstm2_6) load from this
directory automatically.

---

## Key results summary

| Experiment | Primary finding |
|---|---|
| lstm2_1 | CS_gap 0.273→0.067 (k=1..10); r(DE, CS_unknown)=0.9896 |
| lstm2_2 | Depth phase transition L=2→L=4; framework holds for L≥4 |
| lstm2_3 | Multi-layer identification 40–99× random; JUP floor σ_min=0.194 |
| lstm2_4 | Causal localisation 100% (180/180); pre/post ratio ~2×10⁸ |
| lstm2_5 | Oracle cosine=1.000000 (36K trials); 0.099 vs 1.941 split |
| lstm2_6 | Two profiles diverge with N; Singleton bound 0.158→0.310 |

---

## License

All results JSON files: CC BY 4.0.  
Cite the associated Zenodo deposit when using this data.
