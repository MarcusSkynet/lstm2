"""
lstm2_5_correction.py
=======================
Empirical confirmation of Theorem 9(i), Theorem 3, and Theorem 9(ii)
on the lstm2 model family. Loads saved (k, seed) models from lstm2_3.

Three independent measurement blocks:
  A. Oracle correction   — restore weights, verify cosine ≈ 1.000
  B. Crossing error      — any wrong correction strictly worsens output
  C. Practical correction — syndrome-guided correction, V_j-dependent
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
import argparse
import datetime
import base64

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from lstm2_model import (
    VOCAB_SIZE, D_MODEL, NUM_LAYERS, SEQ_LEN,
    SEEDS_BASE, GeneralistLSTM,
    make_domain_sequences, make_probe_test_split,
    get_weight_matrix, compute_svd_basis,
    perturb_weight, restore_weight,
    measure_syndrome, build_multilayer_syndrome,
    set_seed, get_device, make_results_dir, save_json,
)


# ===========================================================================
# Config
# ===========================================================================

SCRIPT_NAME = 'lstm2_5_correction'
RESULTS_DIR = make_results_dir(SCRIPT_NAME)
MODELS_DIR  = 'results/lstm2_3_detectability/models'
LSTM2_3_JSON = ('results/lstm2_3_detectability/'
                 'lstm2_3_detectability_results.json')

K_TEST   = [1, 5, 10]
SEEDS    = SEEDS_BASE

EPS_MIN, EPS_MAX = 1.0, 5.0

# Measurement A — oracle
N_DIRS_ORACLE = 200
N_EPS_ORACLE  = 10

# Measurement B — crossing error
N_DIRS_CROSS  = 200
N_WRONG_DIRS  = 10

# Measurement C — practical correction
N_DIRS_PRAC   = 200
N_PROBE_DICT  = 50
EPS_DICT      = 3.0
N_TEST_PRAC   = 100

# Quick
K_TEST_QUICK     = [1, 10]
SEEDS_QUICK      = SEEDS_BASE[:2]
N_DIRS_ORACLE_Q  = 20
N_EPS_ORACLE_Q   = 5
N_DIRS_CROSS_Q   = 20
N_WRONG_DIRS_Q   = 3
N_DIRS_PRAC_Q    = 20

ORACLE_THRESHOLD = 0.9999


# ===========================================================================
# Model loading and V_j lookup
# ===========================================================================

def load_model(k: int, seed: int, models_dir: str,
               device: torch.device) -> GeneralistLSTM:
    path = os.path.join(models_dir, f'k{k}_seed{seed}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Model not found: {path}\n'
            f'Run lstm2_3_detectability.py first to generate saved models.')
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model = GeneralistLSTM()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def load_vj_table(json_path: str) -> dict:
    """Return {(k_str, seed_str): list[float of length n_dirs]} or {}."""
    if not os.path.exists(json_path):
        print(f"  [warn] {json_path} not found — V_j unavailable",
              flush=True)
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        print(f"  [warn] could not parse {json_path}: {e!r}", flush=True)
        return {}
    out = {}
    runs = d.get('runs', {})
    for k_str, by_seed in runs.items():
        for s_str, rec in by_seed.items():
            vj = rec.get('vj_per_direction')
            if isinstance(vj, list):
                out[(k_str, s_str)] = [float(v) for v in vj]
    return out


# ===========================================================================
# Logit / cosine / MSE primitives — operate on GPU tensors when given them
# ===========================================================================

def _forward(model, x, with_layers=False):
    with torch.no_grad():
        if with_layers:
            logits, layers = model(x, return_all_layers=True)
            return logits.detach(), [lo.detach() for lo in layers]
        return model(x).detach()


def cosine_per_token(a_g: torch.Tensor, b_g: torch.Tensor) -> float:
    """Mean cosine similarity per token between [B, T, V] tensors on GPU."""
    a_flat = a_g.reshape(-1, a_g.shape[-1])
    b_flat = b_g.reshape(-1, b_g.shape[-1])
    a_norm = a_flat / (a_flat.norm(dim=-1, keepdim=True) + 1e-12)
    b_norm = b_flat / (b_flat.norm(dim=-1, keepdim=True) + 1e-12)
    return float((a_norm * b_norm).sum(dim=-1).mean().item())


def mse_per_token(a_g: torch.Tensor, b_g: torch.Tensor) -> float:
    """Mean per-token squared L2 distance between [B, T, V] tensors."""
    diff = (a_g - b_g).reshape(-1, a_g.shape[-1])
    return float((diff ** 2).sum(dim=-1).mean().item())


# ===========================================================================
# Measurement A — Oracle correction
# ===========================================================================

def measure_oracle(model, W, Vh, test_seqs_d, n_dirs, n_eps, seed, device):
    rng = np.random.default_rng(seed)
    clean_g = _forward(model, test_seqs_d)
    cosines = []
    failures = 0
    for d in range(n_dirs):
        direction = Vh[d]
        for _ in range(n_eps):
            eps = float(rng.uniform(EPS_MIN, EPS_MAX))
            original = perturb_weight(W, direction, eps)
            try:
                # Oracle correction = restore weights, then forward.
                restore_weight(W, original)
                corrected_g = _forward(model, test_seqs_d)
            finally:
                # restore_weight already returned us to the original;
                # extra defensive copy in case forward did anything weird
                W.data.copy_(original)
            cos = cosine_per_token(corrected_g, clean_g)
            cosines.append(cos)
            if cos < ORACLE_THRESHOLD:
                failures += 1
    cosines_arr = np.asarray(cosines)
    return {
        'n_trials':     int(cosines_arr.size),
        'mean_cosine':  float(cosines_arr.mean()),
        'min_cosine':   float(cosines_arr.min()),
        'std_cosine':   float(cosines_arr.std()),
        'n_failures':   int(failures),
        'failure_rate': float(failures / max(cosines_arr.size, 1)),
        'cosines':      cosines_arr.tolist(),
    }


# ===========================================================================
# Measurement B — Crossing error
# ===========================================================================

def measure_crossing(model, W, Vh, test_seqs_d, n_dirs, n_wrong, seed,
                       device):
    n_dirs_total = int(Vh.shape[0])
    clean_g = _forward(model, test_seqs_d)
    trials = []
    eps_inj_list = []
    eps_corr_list = []
    err_before_list = []
    err_after_list = []
    crossing_list = []
    violations = 0

    for d in range(n_dirs):
        rng = np.random.default_rng(seed + d)
        eps_inject = float(rng.uniform(EPS_MIN, EPS_MAX))
        direction_d = Vh[d]

        # Step 1: inject perturbation in d
        original = perturb_weight(W, direction_d, eps_inject)
        try:
            pert_g = _forward(model, test_seqs_d)
            error_before = mse_per_token(pert_g, clean_g)

            # Step 2: try N_WRONG_DIRS wrong corrections
            wrong_pool = [j for j in range(n_dirs_total) if j != d]
            n_wrong_eff = min(n_wrong, len(wrong_pool))
            wrong_indices = rng.choice(wrong_pool, size=n_wrong_eff,
                                         replace=False)
            for j in wrong_indices:
                eps_wrong = float(rng.uniform(EPS_MIN, EPS_MAX))
                direction_j = Vh[int(j)]
                # Apply wrong correction on top of injection
                original_j = perturb_weight(W, direction_j, -eps_wrong)
                try:
                    wrong_g = _forward(model, test_seqs_d)
                    error_after = mse_per_token(wrong_g, clean_g)
                finally:
                    restore_weight(W, original_j)
                cross = float(error_after - error_before)
                trials.append({
                    'true_dir':     int(d),
                    'wrong_dir':    int(j),
                    'eps_inject':   eps_inject,
                    'eps_correct':  eps_wrong,
                    'error_before': error_before,
                    'error_after':  error_after,
                    'crossing_err': cross,
                    'violation':    bool(cross <= 0),
                })
                eps_inj_list.append(eps_inject)
                eps_corr_list.append(eps_wrong)
                err_before_list.append(error_before)
                err_after_list.append(error_after)
                crossing_list.append(cross)
                if cross <= 0:
                    violations += 1
        finally:
            restore_weight(W, original)

    cross_arr = np.asarray(crossing_list)
    err_before = np.asarray(err_before_list)
    err_after  = np.asarray(err_after_list)
    n = int(cross_arr.size)
    if n == 0:
        return {
            'n_trials': 0, 'mean_crossing_err': 0.0,
            'min_crossing_err': 0.0, 'std_crossing_err': 0.0,
            'n_violations': 0, 'violation_rate': 0.0,
            'mean_error_before': 0.0, 'mean_error_after': 0.0,
            'mean_error_ratio': 0.0,
            'crossings': [],
        }
    ratios = err_after / np.maximum(err_before, 1e-12)
    return {
        'n_trials':           n,
        'mean_crossing_err':  float(cross_arr.mean()),
        'min_crossing_err':   float(cross_arr.min()),
        'std_crossing_err':   float(cross_arr.std()),
        'n_violations':       int(violations),
        'violation_rate':     float(violations / n),
        'mean_error_before':  float(err_before.mean()),
        'mean_error_after':   float(err_after.mean()),
        'mean_error_ratio':   float(ratios.mean()),
        'crossings':          cross_arr.tolist(),
    }


# ===========================================================================
# Measurement C — Practical correction
# ===========================================================================

def build_dictionary(model, W, Vh, n_dirs, probe_seqs, device):
    """Build multi-layer syndrome dictionary using same protocol as lstm2_3."""
    probe_seqs_d = probe_seqs.to(device)
    clean_logits_g, clean_layers_g = _forward(model, probe_seqs_d,
                                                with_layers=True)

    dict_multi_dim = NUM_LAYERS * D_MODEL + VOCAB_SIZE
    dict_multi = np.zeros((n_dirs, dict_multi_dim), dtype=np.float32)
    for d in range(n_dirs):
        original = perturb_weight(W, Vh[d], EPS_DICT)
        try:
            pert_logits_g, pert_layers_g = _forward(model, probe_seqs_d,
                                                       with_layers=True)
        finally:
            restore_weight(W, original)
        # Averaged delta logits
        delta_l = (pert_logits_g - clean_logits_g).mean(dim=(0, 1)) \
            .cpu().numpy().astype(np.float32)
        n_l = float(np.linalg.norm(delta_l))
        logit_syn = delta_l / n_l if n_l > 1e-12 else delta_l
        layer_syns = []
        for L in range(NUM_LAYERS):
            dL = (pert_layers_g[L] - clean_layers_g[L]).mean(dim=(0, 1)) \
                .cpu().numpy().astype(np.float32)
            n_L = float(np.linalg.norm(dL))
            layer_syns.append(dL / n_L if n_L > 1e-12 else dL)
        full = build_multilayer_syndrome(logit_syn, layer_syns,
                                          injection_layer=0)
        dict_multi[d] = full.astype(np.float32)

    norms = np.linalg.norm(dict_multi, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    dict_multi /= norms
    return dict_multi


def measure_practical(model, W, Vh, test_seqs_d, probe_seqs_d,
                       n_dirs, seed, device):
    # Build dictionary from same probe sequences
    clean_test_g = _forward(model, test_seqs_d)
    dict_multi = build_dictionary(model, W, Vh, n_dirs,
                                    probe_seqs_d, device)

    # Cached probe state for syndrome of the test perturbation
    clean_probe_logits_g, clean_probe_layers_g = _forward(
        model, probe_seqs_d, with_layers=True)

    per_direction = []
    rng = np.random.default_rng(seed + 10000)
    for d in range(n_dirs):
        eps_inject = float(rng.uniform(EPS_MIN, EPS_MAX))
        direction_d = Vh[d]

        # 1. Inject perturbation
        original = perturb_weight(W, direction_d, eps_inject)
        try:
            # Pert on test set: compute error_perturbed
            pert_test_g = _forward(model, test_seqs_d)
            error_perturbed = mse_per_token(pert_test_g, clean_test_g)

            # Pert on probe set: compute syndrome
            pert_probe_logits_g, pert_probe_layers_g = _forward(
                model, probe_seqs_d, with_layers=True)
            delta_l = (pert_probe_logits_g - clean_probe_logits_g).mean(
                dim=(0, 1)).cpu().numpy().astype(np.float32)
            n_l = float(np.linalg.norm(delta_l))
            logit_syn = delta_l / n_l if n_l > 1e-12 else delta_l
            layer_syns = []
            for L in range(NUM_LAYERS):
                dL = (pert_probe_layers_g[L] - clean_probe_layers_g[L]) \
                    .mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
                n_L = float(np.linalg.norm(dL))
                layer_syns.append(dL / n_L if n_L > 1e-12 else dL)
            test_multi = build_multilayer_syndrome(
                logit_syn, layer_syns, injection_layer=0
            ).astype(np.float32)
            tn = float(np.linalg.norm(test_multi))
            test_multi = test_multi / tn if tn > 1e-12 else test_multi

            cosines = dict_multi @ test_multi
            d_hat = int(np.argmax(cosines))
            id_correct = bool(d_hat == d)
        finally:
            restore_weight(W, original)

        # 2. Apply estimated correction (eps_hat = EPS_DICT) on top of
        #    the still-restored W, but the SPEC's intent is corrected
        #    weights rather than residually perturbed. We re-apply the
        #    injection then add the wrong-direction correction:
        eps_hat = EPS_DICT
        original = perturb_weight(W, direction_d, eps_inject)
        try:
            direction_hat = Vh[d_hat]
            original_corr = perturb_weight(W, direction_hat, -eps_hat)
            try:
                corrected_g = _forward(model, test_seqs_d)
                error_corrected = mse_per_token(corrected_g, clean_test_g)
            finally:
                restore_weight(W, original_corr)
        finally:
            restore_weight(W, original)

        ratio = (error_corrected / error_perturbed
                 if error_perturbed > 1e-12 else float('inf'))
        per_direction.append({
            'direction':         int(d),
            'eps_inject':        float(eps_inject),
            'id_correct':        bool(id_correct),
            'd_hat':             int(d_hat),
            'error_perturbed':   float(error_perturbed),
            'error_corrected':   float(error_corrected),
            'correction_ratio':  float(ratio),
            'crossing_occurred': bool(ratio > 1.0),
        })

    # Aggregate
    ratios = np.asarray([r['correction_ratio'] for r in per_direction])
    id_correct = np.asarray([r['id_correct'] for r in per_direction], dtype=bool)
    crossings = int(np.sum(ratios > 1.0))
    n = len(per_direction)
    correct_ratios = ratios[id_correct]
    wrong_ratios   = ratios[~id_correct]
    return {
        'n_trials':              int(n),
        'id_accuracy':           float(id_correct.mean()) if n else 0.0,
        'mean_correction_ratio': float(np.mean(np.minimum(ratios, 1e9))) if n else 0.0,
        'n_crossings':           crossings,
        'crossing_rate':         float(crossings / n) if n else 0.0,
        'mean_ratio_correct_id': (float(correct_ratios.mean())
                                    if correct_ratios.size > 0 else float('nan')),
        'mean_ratio_wrong_id':   (float(wrong_ratios.mean())
                                    if wrong_ratios.size > 0 else float('nan')),
        'per_direction':         per_direction,
    }


def pearson_r(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2: return float('nan')
    xm = x - x.mean(); ym = y - y.mean()
    denom = float(np.sqrt((xm**2).sum() * (ym**2).sum()))
    if denom < 1e-12: return float('nan')
    return float(xm @ ym / denom)


# ===========================================================================
# Aggregation
# ===========================================================================

def aggregate_runs(runs: dict) -> dict:
    agg = {}
    for k_str, by_seed in runs.items():
        # Oracle
        ms_o   = [r['oracle']['mean_cosine']   for r in by_seed.values()]
        mn_o   = [r['oracle']['min_cosine']    for r in by_seed.values()]
        nf_o   = sum(r['oracle']['n_failures'] for r in by_seed.values())
        nt_o   = sum(r['oracle']['n_trials']   for r in by_seed.values())

        # Crossing
        mx_c   = [r['crossing']['mean_crossing_err'] for r in by_seed.values()]
        mn_c   = [r['crossing']['min_crossing_err']  for r in by_seed.values()]
        nv_c   = sum(r['crossing']['n_violations']    for r in by_seed.values())
        nt_c   = sum(r['crossing']['n_trials']        for r in by_seed.values())
        rt_c   = [r['crossing']['mean_error_ratio']  for r in by_seed.values()]

        # Practical
        ia     = [r['practical']['id_accuracy']           for r in by_seed.values()]
        mr     = [r['practical']['mean_correction_ratio'] for r in by_seed.values()]
        cr     = [r['practical']['crossing_rate']         for r in by_seed.values()]
        rc     = [r['practical']['mean_ratio_correct_id'] for r in by_seed.values()]
        rw     = [r['practical']['mean_ratio_wrong_id']   for r in by_seed.values()]
        prv    = [r['practical']['pearson_r_vj_ratio']    for r in by_seed.values()]

        def _ms(arr):
            arr = [a for a in arr if a is not None
                   and (not isinstance(a, float) or np.isfinite(a))]
            if not arr: return {'mean': float('nan'), 'std': float('nan')}
            arr = np.asarray(arr, float)
            return {'mean': float(arr.mean()), 'std': float(arr.std())}

        agg[k_str] = {
            'oracle': {
                'mean_cosine':  _ms(ms_o),
                'min_cosine':   _ms(mn_o),
                'n_failures':   int(nf_o),
                'n_trials':     int(nt_o),
                'failure_rate': (float(nf_o / nt_o) if nt_o else 0.0),
            },
            'crossing': {
                'mean_crossing_err': _ms(mx_c),
                'min_crossing_err':  _ms(mn_c),
                'n_violations':      int(nv_c),
                'n_trials':          int(nt_c),
                'violation_rate':    (float(nv_c / nt_c) if nt_c else 0.0),
                'mean_error_ratio':  _ms(rt_c),
            },
            'practical': {
                'id_accuracy':           _ms(ia),
                'mean_correction_ratio': _ms(mr),
                'crossing_rate':         _ms(cr),
                'mean_ratio_correct_id': _ms(rc),
                'mean_ratio_wrong_id':   _ms(rw),
                'pearson_r_vj_ratio':    _ms(prv),
            },
        }
    return agg


# ===========================================================================
# Confirmation
# ===========================================================================

def evaluate_confirmation(aggregate: dict, k_test: list) -> dict:
    def status(p): return 'CONFIRMED' if p else 'NOT MET'
    klo_str = '1' if 1 in k_test else str(k_test[0])
    khi_str = '10' if 10 in k_test else str(k_test[-1])

    # R1: oracle mean_cosine > 0.9999 at all k
    r1_pass = True
    r1_min = 1.0
    for k in k_test:
        a = aggregate.get(str(k), {}).get('oracle', {})
        m = a.get('mean_cosine', {}).get('mean', 0.0)
        if m < ORACLE_THRESHOLD: r1_pass = False
        r1_min = min(r1_min, m)

    # R2: zero oracle failures
    r2_total = sum(aggregate.get(str(k), {}).get('oracle', {})
                    .get('n_failures', 0) for k in k_test)
    r2_pass = r2_total == 0

    # R3, R4: zero crossing violations at k=1 and k=10
    a_lo = aggregate.get(klo_str, {}).get('crossing', {})
    a_hi = aggregate.get(khi_str, {}).get('crossing', {})
    r3_n = a_lo.get('n_violations', 0)
    r4_n = a_hi.get('n_violations', 0)
    r3_pass = r3_n == 0
    r4_pass = r4_n == 0

    # R5: mean_error_ratio > 1.0 at all k
    r5_pass = True
    r5_vals = []
    for k in k_test:
        a = aggregate.get(str(k), {}).get('crossing', {})
        m = a.get('mean_error_ratio', {}).get('mean', 0.0)
        r5_vals.append(m)
        if m <= 1.0: r5_pass = False

    # R6: ratio|correct_id < 1.0 at k=1
    p_lo = aggregate.get(klo_str, {}).get('practical', {})
    r6_value = p_lo.get('mean_ratio_correct_id', {}).get('mean', float('nan'))
    r6_pass = (np.isfinite(r6_value) and r6_value < 1.0)

    # R7: ratio|wrong_id > 1.0 at k=1
    r7_value = p_lo.get('mean_ratio_wrong_id', {}).get('mean', float('nan'))
    r7_pass = (np.isfinite(r7_value) and r7_value > 1.0)

    # R8: Pearson r(V_j, ratio) > 0 at k=1
    r8_value = p_lo.get('pearson_r_vj_ratio', {}).get('mean', float('nan'))
    r8_pass = (np.isfinite(r8_value) and r8_value > 0.0)

    return {
        'R1': {'status': status(r1_pass), 'value': float(r1_min),
               'criterion': 'mean_cosine > 0.9999 (all k)'},
        'R2': {'status': status(r2_pass), 'value': int(r2_total),
               'criterion': 'zero oracle failures'},
        'R3': {'status': status(r3_pass), 'value': int(r3_n),
               'criterion': 'zero crossing violations at k=' + klo_str},
        'R4': {'status': status(r4_pass), 'value': int(r4_n),
               'criterion': 'zero crossing violations at k=' + khi_str},
        'R5': {'status': status(r5_pass), 'value': float(np.mean(r5_vals)),
               'criterion': 'mean_error_ratio > 1.0 (all k)'},
        'R6': {'status': status(r6_pass), 'value': float(r6_value),
               'criterion': 'ratio|correct_id < 1.0 at k=' + klo_str},
        'R7': {'status': status(r7_pass), 'value': float(r7_value),
               'criterion': 'ratio|wrong_id > 1.0 at k=' + klo_str},
        'R8': {'status': status(r8_pass), 'value': float(r8_value),
               'criterion': 'r(V_j, correction_ratio) > 0 at k=' + klo_str},
    }


# ===========================================================================
# Figures
# ===========================================================================

def _save_fig(fig, base):
    fig.savefig(os.path.join(RESULTS_DIR, base + '.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(RESULTS_DIR, base + '.png'),
                bbox_inches='tight', dpi=120)
    plt.close(fig)


def fig_oracle_cosine(runs: dict):
    if '1' not in runs:
        return
    cosines = []
    for s, rec in runs['1'].items():
        cosines.extend(rec['oracle'].get('cosines', []))
    if not cosines:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cosines, bins=60, color='0.4', edgecolor='black')
    ax.axvline(1.0, color='black', linestyle='--', lw=1.2,
               label='cosine = 1.0')
    lo = max(0.9995, min(cosines))
    ax.set_xlim(lo, 1.0001)
    ax.set_xlabel('Oracle cosine similarity')
    ax.set_ylabel('Count')
    ax.set_title('Oracle Correction Cosine Distribution (k=1)')
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, 'fig_oracle_cosine')


def fig_crossing_error(runs: dict, k_test: list):
    panels = [k for k in k_test if str(k) in runs]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                              squeeze=False)
    for ax, k in zip(axes[0], panels):
        crosses = []
        for s, rec in runs[str(k)].items():
            crosses.extend(rec['crossing'].get('crossings', []))
        if not crosses:
            ax.set_axis_off()
            continue
        crosses = np.asarray(crosses)
        # log scale: include only positive values; flag count of <=0
        pos = crosses[crosses > 0]
        nonpos = int(np.sum(crosses <= 0))
        if pos.size > 0:
            ax.hist(pos, bins=60, color='0.45', edgecolor='black',
                    log=True)
            ax.set_xscale('log')
        ax.set_xlabel('Crossing error (log scale)')
        ax.set_ylabel('Count')
        ax.set_title(f'k={k} — Crossing Error\n(violations: {nonpos})')
        ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, 'fig_crossing_error')


def fig_practical_correction_vj(runs: dict):
    if '1' not in runs:
        return
    vj_all = []; ratio_all = []; cross_all = []
    for s, rec in runs['1'].items():
        for r in rec['practical']['per_direction']:
            vj = r.get('vj')
            if vj is None or not np.isfinite(vj):
                continue
            ratio = r['correction_ratio']
            if not np.isfinite(ratio):
                continue
            vj_all.append(vj)
            ratio_all.append(ratio)
            cross_all.append(r['crossing_occurred'])
    fig, ax = plt.subplots(figsize=(8, 6))
    if not vj_all:
        ax.text(0.5, 0.5, 'No V_j data — lstm2_3 JSON unavailable',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save_fig(fig, 'fig_practical_correction_vj')
        return
    vj_arr = np.asarray(vj_all); ratio_arr = np.asarray(ratio_all)
    cross_arr = np.asarray(cross_all)
    ax.scatter(vj_arr[~cross_arr], ratio_arr[~cross_arr],
               s=12, color='#1f77b4', alpha=0.6, edgecolors='none',
               label='Improved (ratio < 1)')
    ax.scatter(vj_arr[cross_arr], ratio_arr[cross_arr],
               s=12, color='#d62728', alpha=0.6, edgecolors='none',
               label='Crossing (ratio > 1)')
    ax.axhline(1.0, color='black', linestyle='--', lw=1.0)
    if vj_arr.size >= 2 and np.std(vj_arr) > 1e-12:
        slope, intercept = np.polyfit(vj_arr, ratio_arr, 1)
        xs = np.linspace(vj_arr.min(), vj_arr.max(), 50)
        ax.plot(xs, slope * xs + intercept, 'k--', lw=1.4)
    r = pearson_r(vj_arr, ratio_arr)
    ax.set_xlabel('Jacobian Variance $V_j$')
    ax.set_ylabel('Correction ratio (corrected / perturbed)')
    ax.set_title(f'Practical Correction Quality vs $V_j$ '
                 f'(k=1, Pearson r = {r:.3f})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig_practical_correction_vj')


def fig_correction_id_split(runs: dict, k_test: list):
    panels = [k for k in k_test if str(k) in runs]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5),
                              squeeze=False)
    for ax, k in zip(axes[0], panels):
        correct = []; wrong = []
        for s, rec in runs[str(k)].items():
            for r in rec['practical']['per_direction']:
                ratio = r['correction_ratio']
                if not np.isfinite(ratio):
                    continue
                if r['id_correct']:
                    correct.append(ratio)
                else:
                    wrong.append(ratio)
        positions = []; data = []; labels = []
        if correct:
            positions.append(1); data.append(correct); labels.append('correct ID')
        if wrong:
            positions.append(2); data.append(wrong); labels.append('wrong ID')
        if data:
            try:
                ax.violinplot(data, positions=positions, showmeans=True,
                              showextrema=False)
            except Exception:
                ax.boxplot(data, positions=positions)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
        ax.set_yscale('log')
        ax.axhline(1.0, color='black', linestyle='--', lw=1.0)
        ax.set_ylabel('Correction ratio (log)')
        ax.set_title(f'k={k}')
        ax.grid(alpha=0.3, which='both')
    fig.suptitle('Correction Quality: correct vs wrong identification')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, 'fig_correction_id_split')


# ===========================================================================
# HTML report
# ===========================================================================

def _img_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def write_html_report(meta: dict, aggregate: dict,
                       confirmation: dict, k_test: list):
    out = os.path.join(RESULTS_DIR, f'{SCRIPT_NAME}_report.html')
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
             f'<title>{SCRIPT_NAME}</title>',
             '<style>'
             'body{font-family:-apple-system,sans-serif;max-width:1100px;'
             'margin:2em auto;padding:0 1em;}'
             'table{border-collapse:collapse;margin:1em 0;}'
             'th,td{border:1px solid #ccc;padding:0.4em 0.8em;text-align:right;}'
             'th{background:#eee;text-align:center;}'
             'td.label{text-align:left;font-weight:500;}'
             '.confirmed{color:#1a7a1a;font-weight:bold;}'
             '.notmet{color:#b00020;font-weight:bold;}'
             'img{max-width:100%;border:1px solid #ddd;margin:0.5em 0;}'
             'h2{border-bottom:1px solid #ccc;padding-bottom:0.2em;}'
             '</style></head><body>']
    parts.append(f"<h1>{SCRIPT_NAME} — {meta['date']}</h1>")
    parts.append(f"<p>Device: {meta['device']} | Mode: "
                 f"{'QUICK' if meta['quick_mode'] else 'FULL'} | k tested: "
                 f"{meta['k_test']} | seeds: {len(meta['seeds'])}</p>")

    parts.append('<h2>Confirmation summary</h2><table>'
                 '<tr><th>ID</th><th>Criterion</th><th>Value</th>'
                 '<th>Status</th></tr>')
    for cid in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        c = confirmation[cid]
        cls = 'confirmed' if c['status'] == 'CONFIRMED' else 'notmet'
        v = c['value']
        if isinstance(v, float):
            vs = f'{v:.4f}' if abs(v) > 1e-3 else f'{v:.2e}'
        else:
            vs = str(v)
        parts.append(f'<tr><td class="label">{cid}</td>'
                     f'<td class="label">{c["criterion"]}</td>'
                     f'<td>{vs}</td>'
                     f'<td class="{cls}">{c["status"]}</td></tr>')
    parts.append('</table>')

    parts.append('<h2>Oracle correction</h2><table>'
                 '<tr><th>k</th><th>mean_cosine</th><th>min_cosine</th>'
                 '<th>n_trials</th><th>failures</th></tr>')
    for k in k_test:
        a = aggregate.get(str(k), {}).get('oracle', {})
        mc = a.get('mean_cosine', {})
        mn = a.get('min_cosine', {})
        parts.append(f"<tr><td>{k}</td>"
                     f"<td>{mc.get('mean', 0):.6f}</td>"
                     f"<td>{mn.get('mean', 0):.6f}</td>"
                     f"<td>{a.get('n_trials', 0)}</td>"
                     f"<td>{a.get('n_failures', 0)}</td></tr>")
    parts.append('</table>')

    parts.append('<h2>Crossing error</h2><table>'
                 '<tr><th>k</th><th>mean_cross_err</th><th>min_cross_err</th>'
                 '<th>n_trials</th><th>violations</th>'
                 '<th>error_ratio</th></tr>')
    for k in k_test:
        a = aggregate.get(str(k), {}).get('crossing', {})
        mc = a.get('mean_crossing_err', {})
        mn = a.get('min_crossing_err', {})
        rt = a.get('mean_error_ratio', {})
        parts.append(f"<tr><td>{k}</td>"
                     f"<td>{mc.get('mean', 0):.4e}</td>"
                     f"<td>{mn.get('mean', 0):.4e}</td>"
                     f"<td>{a.get('n_trials', 0)}</td>"
                     f"<td>{a.get('n_violations', 0)}</td>"
                     f"<td>{rt.get('mean', 0):.3f}</td></tr>")
    parts.append('</table>')

    parts.append('<h2>Practical correction</h2><table>'
                 '<tr><th>k</th><th>id_acc</th><th>mean_ratio</th>'
                 '<th>crossings</th><th>r(V_j,ratio)</th>'
                 '<th>ratio|correct</th><th>ratio|wrong</th></tr>')
    for k in k_test:
        a = aggregate.get(str(k), {}).get('practical', {})
        ia = a.get('id_accuracy', {})
        mr = a.get('mean_correction_ratio', {})
        cr = a.get('crossing_rate', {})
        prv = a.get('pearson_r_vj_ratio', {})
        rc = a.get('mean_ratio_correct_id', {})
        rw = a.get('mean_ratio_wrong_id', {})
        parts.append(f"<tr><td>{k}</td>"
                     f"<td>{ia.get('mean', 0):.3f}</td>"
                     f"<td>{mr.get('mean', 0):.3f}</td>"
                     f"<td>{cr.get('mean', 0):.3f}</td>"
                     f"<td>{prv.get('mean', 0):.3f}</td>"
                     f"<td>{rc.get('mean', 0):.3f}</td>"
                     f"<td>{rw.get('mean', 0):.3f}</td></tr>")
    parts.append('</table>')

    for title, base in [('Oracle cosine distribution', 'fig_oracle_cosine'),
                         ('Crossing error', 'fig_crossing_error'),
                         ('Practical correction vs V_j',
                          'fig_practical_correction_vj'),
                         ('Correct vs wrong identification',
                          'fig_correction_id_split')]:
        png = os.path.join(RESULTS_DIR, base + '.png')
        if os.path.exists(png):
            b64 = _img_b64(png)
            parts.append(f'<h2>{title}</h2>'
                         f'<img src="data:image/png;base64,{b64}">')

    parts.append('</body></html>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ===========================================================================
# RESULTS SUMMARY
# ===========================================================================

def print_results_summary(meta, aggregate, confirmation, k_test):
    print()
    print('=' * 55)
    print('RESULTS SUMMARY — lstm2_5_correction')
    print('=' * 55)
    print(f"Date:    {meta['date']}")
    print(f"Device:  {meta['device']}")
    print(f"Mode:    {'QUICK' if meta['quick_mode'] else 'FULL'}")
    print(f"k values: {k_test}")
    print(f"Seeds:   {meta['seeds']}")
    print(f"Target:  lstm_layers[0].weight_ih_l0")

    print()
    print('-- MEASUREMENT A: ORACLE CORRECTION --------------------')
    print(f"{'k':<5}{'mean_cosine':<15}{'min_cosine':<15}"
          f"{'n_trials':<10}{'failures':<10}")
    for k in k_test:
        a = aggregate.get(str(k), {}).get('oracle', {})
        mc = a.get('mean_cosine', {}).get('mean', 0.0)
        mn = a.get('min_cosine', {}).get('mean', 0.0)
        nt = a.get('n_trials', 0); nf = a.get('n_failures', 0)
        print(f"{k:<5}{mc:<15.6f}{mn:<15.6f}{nt:<10}{nf:<10}")

    print()
    print('-- MEASUREMENT B: CROSSING ERROR -----------------------')
    print(f"{'k':<5}{'mean_cross_err':<18}{'min_cross_err':<18}"
          f"{'n_trials':<10}{'violations':<10}")
    for k in k_test:
        a = aggregate.get(str(k), {}).get('crossing', {})
        mx = a.get('mean_crossing_err', {}).get('mean', 0.0)
        mn = a.get('min_crossing_err', {}).get('mean', 0.0)
        nt = a.get('n_trials', 0); nv = a.get('n_violations', 0)
        print(f"{k:<5}{mx:<18.4e}{mn:<18.4e}{nt:<10}{nv:<10}")
    print()
    ratios_line = '  '.join(
        f"k={k}:{aggregate.get(str(k), {}).get('crossing', {}).get('mean_error_ratio', {}).get('mean', 0):.3f}"
        for k in k_test)
    print(f"Mean error ratio (error_after / error_before):  {ratios_line}")

    print()
    print('-- MEASUREMENT C: PRACTICAL CORRECTION -----------------')
    print(f"{'k':<5}{'id_acc':<10}{'mean_ratio':<14}"
          f"{'crossings':<14}{'r(V_j,ratio)':<14}")
    for k in k_test:
        a = aggregate.get(str(k), {}).get('practical', {})
        ia = a.get('id_accuracy', {}).get('mean', 0.0)
        mr = a.get('mean_correction_ratio', {}).get('mean', 0.0)
        cr = a.get('crossing_rate', {}).get('mean', 0.0)
        prv = a.get('pearson_r_vj_ratio', {}).get('mean', float('nan'))
        prv_s = f"{prv:.3f}" if np.isfinite(prv) else 'nan'
        print(f"{k:<5}{ia:<10.3f}{mr:<14.3f}{cr:<14.3f}{prv_s:<14}")
    print()
    print('Conditional means:')
    print(f"{'k':<5}{'ratio|correct_id':<22}{'ratio|wrong_id':<22}")
    for k in k_test:
        a = aggregate.get(str(k), {}).get('practical', {})
        rc = a.get('mean_ratio_correct_id', {}).get('mean', float('nan'))
        rw = a.get('mean_ratio_wrong_id', {}).get('mean', float('nan'))
        rcs = f"{rc:.3f}" if np.isfinite(rc) else 'nan'
        rws = f"{rw:.3f}" if np.isfinite(rw) else 'nan'
        print(f"{k:<5}{rcs:<22}{rws:<22}")

    print()
    print('-- CONFIRMATION ----------------------------------------')
    c = confirmation
    print(f"R1  mean_cosine > 0.9999 (all k)       "
          f"[{c['R1']['status']}]   min mean_cosine={c['R1']['value']:.6f}")
    print(f"R2  zero oracle failures               "
          f"[{c['R2']['status']}]   n={c['R2']['value']}")
    print(f"R3  zero crossing violations (k=lo)    "
          f"[{c['R3']['status']}]   n={c['R3']['value']}")
    print(f"R4  zero crossing violations (k=hi)    "
          f"[{c['R4']['status']}]   n={c['R4']['value']}")
    print(f"R5  mean_error_ratio > 1.0 (all k)     "
          f"[{c['R5']['status']}]   value={c['R5']['value']:.3f}")
    print(f"R6  ratio|correct_id < 1.0 (k=lo)      "
          f"[{c['R6']['status']}]   value={c['R6']['value']:.3f}")
    print(f"R7  ratio|wrong_id > 1.0 (k=lo)        "
          f"[{c['R7']['status']}]   value={c['R7']['value']:.3f}")
    print(f"R8  r(V_j, ratio) > 0 (k=lo)           "
          f"[{c['R8']['status']}]   r={c['R8']['value']:.4f}")

    print()
    all_met = all(v['status'] == 'CONFIRMED' for v in c.values())
    print('-- OVERALL ---------------------------------------------')
    print(f"All confirmations met: {'YES' if all_met else 'NO'}")
    oracle_pass = c['R1']['status'] == 'CONFIRMED' and c['R2']['status'] == 'CONFIRMED'
    cross_pass  = (c['R3']['status'] == 'CONFIRMED'
                    and c['R4']['status'] == 'CONFIRMED'
                    and c['R5']['status'] == 'CONFIRMED')
    prac_pass   = (c['R6']['status'] == 'CONFIRMED'
                    and c['R7']['status'] == 'CONFIRMED'
                    and c['R8']['status'] == 'CONFIRMED')
    print(f"Oracle (Thm 9i):     {'CONFIRMED' if oracle_pass else 'NOT MET'}")
    print(f"Crossing (Thm 3):    {'CONFIRMED' if cross_pass else 'NOT MET'}")
    print(f"Practical (Thm 9ii): {'CONFIRMED' if prac_pass else 'NOT MET'}")
    print(f"Output: results/{SCRIPT_NAME}/")
    print('=' * 55)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        k_test = K_TEST_QUICK; seeds = SEEDS_QUICK
        n_dirs_oracle = N_DIRS_ORACLE_Q; n_eps_oracle = N_EPS_ORACLE_Q
        n_dirs_cross = N_DIRS_CROSS_Q;   n_wrong = N_WRONG_DIRS_Q
        n_dirs_prac  = N_DIRS_PRAC_Q
        print('[QUICK MODE]')
    else:
        k_test = K_TEST; seeds = SEEDS
        n_dirs_oracle = N_DIRS_ORACLE; n_eps_oracle = N_EPS_ORACLE
        n_dirs_cross = N_DIRS_CROSS;   n_wrong = N_WRONG_DIRS
        n_dirs_prac  = N_DIRS_PRAC

    device = get_device()
    print(f"Script: {SCRIPT_NAME}")
    print(f"k tested: {k_test}    seeds: {seeds}")
    print(f"Oracle: {n_dirs_oracle}d × {n_eps_oracle}eps  "
          f"Cross: {n_dirs_cross}d × {n_wrong}wrong  "
          f"Prac: {n_dirs_prac}d")

    vj_table = load_vj_table(LSTM2_3_JSON)

    runs: dict = {}
    for k in k_test:
        runs[str(k)] = {}
        for seed in seeds:
            t0 = time.time()
            try:
                model = load_model(k, seed, MODELS_DIR, device)
            except FileNotFoundError as e:
                print(str(e), file=sys.stderr)
                return 1
            print(f"[loaded] k={k} seed={seed}", flush=True)

            set_seed(seed)
            W = model.lstm_layers[0].weight_ih_l0
            W_cpu = get_weight_matrix(model, 0)
            _, Vh = compute_svd_basis(W_cpu)
            Vh = Vh.to(device)

            probe_seqs, test_seqs = make_probe_test_split(
                domain_idx=0, n_probe=N_PROBE_DICT, n_test=N_TEST_PRAC)
            probe_seqs_d = probe_seqs.to(device)
            test_seqs_d  = test_seqs.to(device)

            # Measurement A
            t_a = time.time()
            oracle_res = measure_oracle(
                model, W, Vh, test_seqs_d,
                n_dirs_oracle, n_eps_oracle, seed, device)
            print(f"[A] k={k:<3}seed={seed:<6}n={oracle_res['n_trials']:<5} "
                  f"mean_cos={oracle_res['mean_cosine']:.6f}  "
                  f"min_cos={oracle_res['min_cosine']:.6f}  "
                  f"failures={oracle_res['n_failures']}  "
                  f"({time.time()-t_a:.0f}s)",
                  flush=True)

            # Measurement B
            t_b = time.time()
            cross_res = measure_crossing(
                model, W, Vh, test_seqs_d, n_dirs_cross, n_wrong,
                seed, device)
            print(f"[B] k={k:<3}seed={seed:<6}n={cross_res['n_trials']:<5} "
                  f"mean_cross={cross_res['mean_crossing_err']:+.3e}  "
                  f"min_cross={cross_res['min_crossing_err']:+.3e}  "
                  f"violations={cross_res['n_violations']}  "
                  f"err_ratio={cross_res['mean_error_ratio']:.3f}  "
                  f"({time.time()-t_b:.0f}s)",
                  flush=True)

            # Measurement C
            t_c = time.time()
            prac_res = measure_practical(
                model, W, Vh, test_seqs_d, probe_seqs_d,
                n_dirs_prac, seed, device)
            # Attach V_j per direction from lstm2_3 JSON if available
            vj_list = vj_table.get((str(k), str(seed)))
            for r in prac_res['per_direction']:
                d = r['direction']
                if vj_list is not None and d < len(vj_list):
                    r['vj'] = float(vj_list[d])
                else:
                    r['vj'] = None
            # Pearson r within (k, seed)
            ratios = np.asarray([r['correction_ratio']
                                  for r in prac_res['per_direction']])
            vjs = np.asarray([r['vj'] if r['vj'] is not None else np.nan
                               for r in prac_res['per_direction']])
            ok = np.isfinite(vjs) & np.isfinite(ratios)
            if int(ok.sum()) >= 2:
                prac_res['pearson_r_vj_ratio'] = pearson_r(
                    vjs[ok], ratios[ok])
            else:
                prac_res['pearson_r_vj_ratio'] = float('nan')
            print(f"[C] k={k:<3}seed={seed:<6}n={prac_res['n_trials']:<5} "
                  f"id_acc={prac_res['id_accuracy']:.3f}  "
                  f"mean_ratio={prac_res['mean_correction_ratio']:.3f}  "
                  f"crossings={prac_res['n_crossings']}  "
                  f"r(Vj,ratio)={prac_res['pearson_r_vj_ratio']:+.3f}  "
                  f"({time.time()-t_c:.0f}s)",
                  flush=True)

            runs[str(k)][str(seed)] = {
                'oracle':    oracle_res,
                'crossing':  cross_res,
                'practical': prac_res,
            }
            print(f"  total ({time.time()-t0:.0f}s)", flush=True)

    aggregate = aggregate_runs(runs)
    confirmation = evaluate_confirmation(aggregate, k_test)

    meta = {
        'script': SCRIPT_NAME,
        'date': datetime.datetime.now().isoformat(),
        'device': str(device),
        'k_test': k_test,
        'seeds': seeds,
        'quick_mode': args.quick,
        'config': {
            'n_dirs_oracle': n_dirs_oracle, 'n_eps_oracle': n_eps_oracle,
            'n_dirs_cross':  n_dirs_cross,  'n_wrong_dirs':  n_wrong,
            'n_dirs_prac':   n_dirs_prac,   'n_probe_dict': N_PROBE_DICT,
            'eps_dict':      EPS_DICT,
            'eps_min': EPS_MIN, 'eps_max': EPS_MAX,
            'perturbation_target': 'lstm_layers[0].weight_ih_l0',
            'models_dir': MODELS_DIR,
        },
    }
    output = {
        'meta':         meta,
        'runs':         runs,
        'aggregate':    aggregate,
        'confirmation': confirmation,
    }
    save_json(output, os.path.join(RESULTS_DIR,
                                     f'{SCRIPT_NAME}_results.json'))

    fig_oracle_cosine(runs)
    fig_crossing_error(runs, k_test)
    fig_practical_correction_vj(runs)
    fig_correction_id_split(runs, k_test)
    write_html_report(meta, aggregate, confirmation, k_test)

    print_results_summary(meta, aggregate, confirmation, k_test)
    return 0


if __name__ == '__main__':
    sys.exit(main())
