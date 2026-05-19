# Results

This directory contains pre-computed results JSON files from the
lstm2 experimental series.

## File placement

After running an experiment or downloading from Zenodo, place
results files here:

```
results/
├── lstm2_1_modes_vs_k/
│   └── lstm2_1_modes_vs_k_results.json
├── lstm2_2_arch_sweep/
│   ├── lstm2_2_arch_sweep_A_results.json
│   ├── lstm2_2_arch_sweep_B_results.json
│   └── lstm2_2_arch_sweep_C_results.json
├── lstm2_3_detectability/
│   ├── lstm2_3_detectability_results.json
│   └── postanalysis_vj_acc_correlation.json
├── lstm2_4_causal_localisation/
│   └── lstm2_4_causal_localisation_results.json
├── lstm2_5_correction/
│   └── lstm2_5_correction_results.json
└── lstm2_6_singlecell_vs_multicell/
    ├── lstm2_6_N5_results.json
    └── lstm2_6_N10_results.json
```

## Data license

All results JSON files are released under CC BY 4.0.
Cite the associated Zenodo deposit when using this data.

## Note on model weights

Saved model weights (.pt files, ~2GB total) are not included
in this repository. They are regenerated deterministically by
running `lstm2_3_detectability.py` with the canonical seeds.
