"""
lstm2_4_causal_localisation.py — First test of Definition 4.2.

For each saved (k, seed) model from lstm2_3, inject a perturbation at
each LSTM layer k* ∈ {0..9} and measure the residual stream response
magnitude ||δ̄_L|| at every observation layer L ∈ {0..9}. Definition 4.2
predicts ||δ̄_L|| ≈ 0 for L < k* and > 0 for L ≥ k*.
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
    SEEDS_BASE,
    GeneralistLSTM, make_domain_sequences,
    get_weight_matrix, compute_svd_basis,
    perturb_weight, restore_weight,
    get_residual_streams,
    set_seed, get_device, make_results_dir, save_json,
)


# ===========================================================================
# Config
# ===========================================================================

SCRIPT_NAME = 'lstm2_4_causal_localisation'
RESULTS_DIR = make_results_dir(SCRIPT_NAME)
MODELS_DIR  = 'results/lstm2_3_detectability/models'

K_TEST          = [1, 5, 10]
INJECT_LAYERS   = list(range(NUM_LAYERS))   # [0..9]
N_PROBE         = 50
EPS             = 3.0
N_DIRS          = 10
TRANSITION_FRAC = 0.10
SEEDS           = SEEDS_BASE

# Quick mode
K_TEST_QUICK    = [1, 10]
INJECT_QUICK    = [0, 3, 6, 9]
N_DIRS_QUICK    = 3
N_PROBE_QUICK   = 20
SEEDS_QUICK     = SEEDS_BASE[:2]


# ===========================================================================
# Model loading
# ===========================================================================

def load_model(k: int, seed: int, models_dir: str,
               device: torch.device) -> tuple[GeneralistLSTM, float]:
    path = os.path.join(models_dir, f'k{k}_seed{seed}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Model not found: {path}\n'
            f'Run lstm2_3_detectability.py first to generate saved models.')
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model = GeneralistLSTM()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, float(ckpt.get('final_loss', 0.0))


# ===========================================================================
# Localisation
# ===========================================================================

def localise(profile: np.ndarray,
             transition_frac: float = TRANSITION_FRAC) -> int:
    """Return the first observation layer whose norm exceeds
    transition_frac × max(profile). -1 if no response detected."""
    peak = float(profile.max()) if profile.size else 0.0
    if peak < 1e-8:
        return -1
    threshold = transition_frac * peak
    for L in range(len(profile)):
        if profile[L] >= threshold:
            return L
    return -1


# ===========================================================================
# Per-(k, seed) measurement
# ===========================================================================

def measure_for_model(model: GeneralistLSTM,
                       k: int, seed: int,
                       inject_layers: list,
                       n_dirs: int, n_probe: int,
                       device: torch.device,
                       progress_print=None) -> dict:
    """Measure residual response profiles for one (k, seed) model.

    Returns a dict keyed by k_star (int) with profile_mean, profile_std,
    pred_k_star, correct, pre_norm, post_norm, ratio.
    """
    set_seed(seed)
    probe_seqs = make_domain_sequences(0, n_probe).to(device)

    # Clean residuals computed ONCE per (k, seed) — same across all k* and d
    clean_residuals = get_residual_streams(model, probe_seqs, device)
    # list of NUM_LAYERS arrays, each [n_probe, SEQ_LEN, D_MODEL]

    out = {}
    for k_star in inject_layers:
        W_star = model.lstm_layers[k_star].weight_ih_l0
        W_cpu  = get_weight_matrix(model, k_star)
        # weight_ih_l0 is [4*D, D] -> Vh has shape [D, D] (full_matrices=False
        # gives min(4D, D) = D right singular vectors)
        _, Vh = compute_svd_basis(W_cpu)
        Vh = Vh.to(device)
        n_dirs_eff = min(n_dirs, Vh.shape[0])

        per_direction_norms = np.zeros((n_dirs_eff, NUM_LAYERS),
                                        dtype=np.float64)
        for d in range(n_dirs_eff):
            direction = Vh[d]
            original = perturb_weight(W_star, direction, EPS)
            try:
                pert_residuals = get_residual_streams(
                    model, probe_seqs, device)
            finally:
                restore_weight(W_star, original)

            for L in range(NUM_LAYERS):
                delta = pert_residuals[L] - clean_residuals[L]
                # Average over (probe, time) to get a [D] vector,
                # then take its (unnormalised) norm.
                delta_mean = delta.mean(axis=(0, 1))
                per_direction_norms[d, L] = float(
                    np.linalg.norm(delta_mean))

        profile_mean = per_direction_norms.mean(axis=0)  # [NUM_LAYERS]
        profile_std  = per_direction_norms.std(axis=0)   # [NUM_LAYERS]
        pred = localise(profile_mean, TRANSITION_FRAC)
        correct = (pred == k_star)

        # Pre/post ratio
        if k_star > 0:
            pre_norm = float(profile_mean[:k_star].mean())
        else:
            pre_norm = 0.0
        post_norm = float(profile_mean[k_star:].mean())
        ratio = post_norm / max(pre_norm, 1e-8)

        out[k_star] = {
            'profile_mean': profile_mean.tolist(),
            'profile_std':  profile_std.tolist(),
            'pred_k_star':  int(pred),
            'correct':      bool(correct),
            'pre_norm':     pre_norm,
            'post_norm':    post_norm,
            'ratio':        float(ratio),
        }

        if progress_print is not None:
            profile_str = ', '.join(f"{v:.3f}" for v in profile_mean)
            tag = '[CORRECT]' if correct else f'[WRONG: pred={pred}]'
            progress_print(
                f"k={k:<3}seed={seed:<6}k*={k_star:<3}"
                f"profile=[{profile_str}]  pred={pred}  {tag}")

    # Per-(k, seed) summary
    n_correct = sum(1 for kst in out if out[kst]['correct'])
    n_total = len(out)
    ratios = [out[kst]['ratio'] for kst in out
              if out[kst]['ratio'] != float('inf') and not np.isnan(out[kst]['ratio'])]
    pre_post_mean = float(np.mean(ratios)) if ratios else 0.0
    pre_post_std  = float(np.std(ratios))  if ratios else 0.0

    return {
        'localisation_accuracy': float(n_correct / n_total) if n_total else 0.0,
        'pre_post_ratio_mean':   pre_post_mean,
        'pre_post_ratio_std':    pre_post_std,
        'n_correct':             int(n_correct),
        'n_total':               int(n_total),
        'by_k_star':             {str(k): out[k] for k in out},
    }


# ===========================================================================
# Aggregation
# ===========================================================================

def aggregate_runs(runs: dict, inject_layers: list) -> dict:
    """Aggregate per-k results across seeds and injection layers."""
    agg = {}
    for k_str, seed_dict in runs.items():
        if not seed_dict:
            continue

        # Localisation accuracy: fraction of (seed, k_star) pairs correct
        all_correct = []
        ratios = []
        confusion = np.zeros((NUM_LAYERS, NUM_LAYERS), dtype=np.int64)
        # +1 row for "no response (-1)" predictions
        bad_predictions = 0
        profiles = {str(kst): {'mean': [], 'std': []}
                    for kst in inject_layers}

        for seed, run in seed_dict.items():
            for kst in inject_layers:
                kst_str = str(kst)
                rec = run['by_k_star'].get(kst_str)
                if rec is None:
                    continue
                all_correct.append(1 if rec['correct'] else 0)
                ratios.append(rec['ratio'])
                if 0 <= rec['pred_k_star'] < NUM_LAYERS:
                    confusion[kst, rec['pred_k_star']] += 1
                else:
                    bad_predictions += 1
                profiles[kst_str]['mean'].append(rec['profile_mean'])
                profiles[kst_str]['std'].append(rec['profile_std'])

        ratios_finite = [r for r in ratios
                         if np.isfinite(r) and not np.isnan(r)]
        profile_by_k_star = {}
        for kst_str, lst in profiles.items():
            if lst['mean']:
                arr = np.asarray(lst['mean'])
                profile_by_k_star[kst_str] = {
                    'mean': arr.mean(axis=0).tolist(),
                    'std':  arr.std(axis=0).tolist(),
                }

        agg[k_str] = {
            'localisation_accuracy': (float(np.mean(all_correct))
                                       if all_correct else 0.0),
            'mean_error':             0.0,  # filled below
            'pre_post_ratio': {
                'mean': float(np.mean(ratios_finite)) if ratios_finite else 0.0,
                'std':  float(np.std(ratios_finite))  if ratios_finite else 0.0,
            },
            'profile_by_k_star': profile_by_k_star,
            'confusion_matrix':  confusion.tolist(),
            'bad_predictions':   int(bad_predictions),
            'n_correct':         int(sum(all_correct)),
            'n_total':           int(len(all_correct)),
        }

        # Mean signed error
        errs = []
        for seed, run in seed_dict.items():
            for kst in inject_layers:
                kst_str = str(kst)
                rec = run['by_k_star'].get(kst_str)
                if rec is None:
                    continue
                if 0 <= rec['pred_k_star'] < NUM_LAYERS:
                    errs.append(rec['pred_k_star'] - kst)
        agg[k_str]['mean_error'] = (float(np.mean(errs)) if errs else 0.0)
    return agg


# ===========================================================================
# Confirmation
# ===========================================================================

def evaluate_confirmation(aggregate: dict, runs: dict,
                           inject_layers: list,
                           k_test: list) -> dict:
    def status(passed: bool) -> str:
        return 'CONFIRMED' if passed else 'NOT MET'

    k_low_str = '1' if 1 in k_test else str(k_test[0])
    k_high_str = '10' if 10 in k_test else str(k_test[-1])
    a_low = aggregate.get(k_low_str, {})
    a_high = aggregate.get(k_high_str, {})

    # L1: pre_norm < 0.10 × post_norm at k=1, averaged over k_star > 0
    # L2: post_norm > 0 at all injection layers, k=1
    pre_post_pairs = []
    all_post_positive = True
    if k_low_str in runs:
        for seed, run in runs[k_low_str].items():
            for kst in inject_layers:
                rec = run['by_k_star'].get(str(kst))
                if rec is None:
                    continue
                if rec['post_norm'] <= 1e-8:
                    all_post_positive = False
                if kst > 0:
                    pre_post_pairs.append((rec['pre_norm'], rec['post_norm']))
    if pre_post_pairs:
        pre_avg = float(np.mean([p[0] for p in pre_post_pairs]))
        post_avg = float(np.mean([p[1] for p in pre_post_pairs]))
        l1_pass = pre_avg < 0.10 * post_avg
        l1_value = pre_avg / max(post_avg, 1e-12)
    else:
        l1_pass = False
        l1_value = float('nan')

    l2_pass = bool(all_post_positive) and bool(pre_post_pairs)

    l3_value = a_low.get('localisation_accuracy', 0.0)
    l3_pass = l3_value > 0.50

    l4_value = a_low.get('pre_post_ratio', {}).get('mean', 0.0)
    l4_pass = l4_value > 5.0

    l5_value = a_high.get('localisation_accuracy', 0.0)
    l5_pass = l5_value > 0.30

    # L6 — visual; mark CONFIRMED if heatmap shows the expected
    # triangular pattern: lower-left triangle near zero, upper-right
    # triangle nonzero. Quantify via mean of off-diagonal upper vs lower
    # at k_low.
    l6_pass = False
    l6_value = float('nan')
    profiles_low = a_low.get('profile_by_k_star', {})
    if profiles_low:
        # Build the matrix: rows = k_star, cols = L
        rows = []
        used_kst = sorted(int(s) for s in profiles_low.keys())
        for kst in used_kst:
            rows.append(profiles_low[str(kst)]['mean'])
        H = np.asarray(rows)  # [n_kst, NUM_LAYERS]
        if H.size:
            # Average value strictly below the diagonal vs strictly above
            below = []
            above = []
            for i, kst in enumerate(used_kst):
                for L in range(NUM_LAYERS):
                    if L < kst:
                        below.append(H[i, L])
                    elif L > kst:
                        above.append(H[i, L])
            if below and above:
                below_m = float(np.mean(below))
                above_m = float(np.mean(above))
                l6_value = below_m / max(above_m, 1e-12)
                l6_pass = l6_value < 0.10  # below should be << above

    return {
        'L1': {'status': status(l1_pass), 'value': l1_value,
               'criterion': 'pre/post < 0.10 at k=' + k_low_str},
        'L2': {'status': status(l2_pass), 'value': bool(l2_pass),
               'criterion': 'post_norm > 0 all injections at k=' + k_low_str},
        'L3': {'status': status(l3_pass), 'value': l3_value,
               'criterion': 'localisation_accuracy(k=' + k_low_str + ') > 0.50'},
        'L4': {'status': status(l4_pass), 'value': l4_value,
               'criterion': 'pre/post ratio(k=' + k_low_str + ') > 5.0'},
        'L5': {'status': status(l5_pass), 'value': l5_value,
               'criterion': 'localisation_accuracy(k=' + k_high_str + ') > 0.30'},
        'L6': {'status': status(l6_pass), 'value': l6_value,
               'criterion': 'heatmap below_diag / above_diag < 0.10 (visual proxy)'},
    }


def def42_status(confirmation: dict) -> str:
    """CONFIRMED if L1, L2, L3, L4 all pass; PARTIAL if L1 and L2 pass
    but accuracy/ratio thresholds missed; NOT CONFIRMED otherwise."""
    L1 = confirmation['L1']['status'] == 'CONFIRMED'
    L2 = confirmation['L2']['status'] == 'CONFIRMED'
    L3 = confirmation['L3']['status'] == 'CONFIRMED'
    L4 = confirmation['L4']['status'] == 'CONFIRMED'
    if L1 and L2 and L3 and L4:
        return 'CONFIRMED'
    if L1 and L2:
        return 'PARTIAL'
    return 'NOT CONFIRMED'


# ===========================================================================
# Figures
# ===========================================================================

def _save_fig(fig, base):
    fig.savefig(os.path.join(RESULTS_DIR, base + '.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(RESULTS_DIR, base + '.png'),
                bbox_inches='tight', dpi=120)
    plt.close(fig)


def fig_heatmap(aggregate_k: dict, k_str: str, inject_layers: list):
    profiles = aggregate_k.get('profile_by_k_star', {})
    H = np.full((NUM_LAYERS, NUM_LAYERS), np.nan)
    for kst in inject_layers:
        rec = profiles.get(str(kst))
        if rec:
            H[kst] = np.asarray(rec['mean'])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log1p(np.where(np.isnan(H), 0, H)),
                   aspect='auto', cmap='viridis', origin='upper')
    ax.set_xlabel('Observation layer L')
    ax.set_ylabel('Injection layer  $k^{*}$')
    ax.set_title(f'k={k_str} — Causal Layer Localisation Heatmap')
    ax.set_xticks(range(NUM_LAYERS))
    ax.set_yticks(range(NUM_LAYERS))
    # Diagonal where L = k*
    diag = np.arange(NUM_LAYERS)
    ax.plot(diag, diag, color='white', linestyle='--', linewidth=1.4)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log1p(mean residual norm)')
    fig.tight_layout()
    _save_fig(fig, f'fig_heatmap_k{k_str}')


def fig_localisation_accuracy(aggregate: dict, k_test: list):
    accs = [aggregate[str(k)]['localisation_accuracy']
            if str(k) in aggregate else 0.0 for k in k_test]
    # std across seeds
    stds = []
    for k in k_test:
        if str(k) not in aggregate:
            stds.append(0.0); continue
        # crude std: from per-seed accuracies (re-derive)
        stds.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(k_test))
    ax.bar(x, accs, yerr=stds, capsize=4,
           color='0.30', edgecolor='black', linewidth=1.2)
    ax.axhline(0.10, color='black', linestyle='--', linewidth=1.0,
               label='Random baseline (1/10)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_test])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Localisation accuracy')
    ax.set_title('Causal Layer Localisation Accuracy vs Domain Count k')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig_localisation_accuracy')


def fig_prepost_ratio(aggregate: dict, k_test: list):
    means = [aggregate[str(k)]['pre_post_ratio']['mean']
             if str(k) in aggregate else 0.0 for k in k_test]
    stds = [aggregate[str(k)]['pre_post_ratio']['std']
            if str(k) in aggregate else 0.0 for k in k_test]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(k_test))
    means_clipped = [max(m, 1e-3) for m in means]
    ax.bar(x, means_clipped, yerr=stds, capsize=4,
           color='0.45', edgecolor='black', linewidth=1.2)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0,
               label='No causal boundary (ratio = 1)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_test])
    ax.set_ylabel('post/pre residual norm ratio (log)')
    ax.set_title('Causal Boundary Signal-to-Noise Ratio vs k')
    ax.legend(loc='upper right')
    ax.grid(axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig_prepost_ratio')


def fig_example_profiles(runs: dict, k_test: list):
    show_kstars = [0, 4, 9]
    rows = [k for k in k_test if str(k) in runs]
    fig, axes = plt.subplots(len(rows), len(show_kstars),
                              figsize=(11, 3.0 * len(rows)),
                              squeeze=False)
    for r, k in enumerate(rows):
        seed_dict = runs[str(k)]
        seed_use = '42' if '42' in seed_dict else next(iter(seed_dict))
        run = seed_dict[seed_use]
        for c, kst in enumerate(show_kstars):
            ax = axes[r][c]
            rec = run['by_k_star'].get(str(kst))
            if rec is None:
                ax.set_axis_off()
                ax.set_title(f'k={k}  k*={kst}  (n/a)')
                continue
            profile = np.asarray(rec['profile_mean'])
            colors = ['0.65' if L < kst else '#1f77b4'
                      for L in range(NUM_LAYERS)]
            ax.bar(range(NUM_LAYERS), profile,
                   color=colors, edgecolor='black', linewidth=0.8)
            ax.axvline(kst - 0.5, color='black', linestyle='--', lw=1.3)
            ax.set_title(f'k={k}, k*={kst}, pred={rec["pred_k_star"]}',
                         fontsize=10)
            ax.set_xticks(range(NUM_LAYERS))
            ax.set_xlabel('Observation layer L')
            if c == 0:
                ax.set_ylabel('||δ̄_L||')
            ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Residual Response Profiles (seed=42, mean across directions)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, 'fig_example_profiles')


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
                 f"{meta['k_test']} | seeds: {len(meta['seeds'])} | "
                 f"N_DIRS: {meta['config']['n_dirs']}</p>")

    # Confirmation table
    parts.append('<h2>Confirmation summary</h2><table>'
                 '<tr><th>ID</th><th>Criterion</th><th>Value</th>'
                 '<th>Status</th></tr>')
    for cid in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
        c = confirmation[cid]
        cls = 'confirmed' if c['status'] == 'CONFIRMED' else 'notmet'
        v = c['value']
        if isinstance(v, float):
            vs = f'{v:.4f}'
        else:
            vs = str(v)
        parts.append(f'<tr><td class="label">{cid}</td>'
                     f'<td class="label">{c["criterion"]}</td>'
                     f'<td>{vs}</td>'
                     f'<td class="{cls}">{c["status"]}</td></tr>')
    parts.append('</table>')

    # Summary table
    parts.append('<h2>Localisation summary</h2>'
                 '<table><tr><th>k</th><th>accuracy</th>'
                 '<th>pre/post ratio</th><th>n_correct/n_total</th></tr>')
    for k in k_test:
        a = aggregate.get(str(k), {})
        parts.append(f"<tr><td>{k}</td>"
                     f"<td>{a.get('localisation_accuracy', 0):.3f}</td>"
                     f"<td>{a.get('pre_post_ratio', {}).get('mean', 0):.2f} "
                     f"± {a.get('pre_post_ratio', {}).get('std', 0):.2f}</td>"
                     f"<td>{a.get('n_correct', 0)}/{a.get('n_total', 0)}</td>"
                     f"</tr>")
    parts.append('</table>')

    # Figures
    figs = [(f'Heatmap k={k}', f'fig_heatmap_k{k}') for k in k_test]
    figs += [('Localisation accuracy vs k', 'fig_localisation_accuracy'),
             ('Pre/post ratio vs k', 'fig_prepost_ratio'),
             ('Example profiles', 'fig_example_profiles')]
    for title, base in figs:
        png = os.path.join(RESULTS_DIR, base + '.png')
        if os.path.exists(png):
            b64 = _img_b64(png)
            parts.append(f'<h2>{title}</h2>'
                         f'<img src="data:image/png;base64,{b64}">')

    # Confusion matrices
    for k in k_test:
        cm = aggregate.get(str(k), {}).get('confusion_matrix')
        if not cm:
            continue
        parts.append(f'<h2>Confusion matrix at k={k}</h2><table>')
        parts.append('<tr><th>true \\ pred</th>'
                     + ''.join(f'<th>{c}</th>' for c in range(NUM_LAYERS))
                     + '</tr>')
        for i, row in enumerate(cm):
            parts.append(f'<tr><td class="label">{i}</td>'
                         + ''.join(f'<td>{v}</td>' for v in row)
                         + '</tr>')
        parts.append('</table>')

    parts.append('</body></html>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ===========================================================================
# RESULTS SUMMARY
# ===========================================================================

def print_results_summary(meta: dict, aggregate: dict, runs: dict,
                           confirmation: dict, k_test: list,
                           inject_layers: list):
    print()
    print('=' * 55)
    print('RESULTS SUMMARY — lstm2_4_causal_localisation')
    print('=' * 55)
    print(f"Date:    {meta['date']}")
    print(f"Device:  {meta['device']}")
    print(f"Mode:    {'QUICK' if meta['quick_mode'] else 'FULL'}")
    print(f"k values tested: {k_test}")
    print(f"Seeds:   {meta['seeds']}")
    print(f"Injection layers: {inject_layers}   "
          f"N_DIRS: {meta['config']['n_dirs']}   EPS: {meta['config']['eps']}")

    print()
    print('-- LOCALISATION SUMMARY --------------------------------')
    print(f"{'k':<5}{'accuracy':<12}{'pre/post ratio':<32}"
          f"{'n_correct/n_total':<18}")
    for k in k_test:
        a = aggregate.get(str(k), {})
        acc = a.get('localisation_accuracy', 0)
        pr = a.get('pre_post_ratio', {})
        m = pr.get('mean', 0); s = pr.get('std', 0)
        # Compact scientific format when pre_norm collapses to ~0 and the
        # ratio spans many orders of magnitude
        if m > 1e6:
            ratio_str = f"{m:.2e} ± {s:.2e}"
        else:
            ratio_str = f"{m:.2f} ± {s:.2f}"
        nstr = f"{a.get('n_correct', 0)}/{a.get('n_total', 0)}"
        print(f"{k:<5}{acc:<12.3f}{ratio_str:<32}{nstr:<18}")

    # Example profiles (seed=42 if available)
    print()
    print('-- EXAMPLE PROFILES (seed=42, mean across directions) --')
    print(f"       obs layer:  "
          + '  '.join(f'{i:>4d}' for i in range(NUM_LAYERS)))
    show_kstars = [0, 4, 9]
    for k in k_test:
        if str(k) not in runs:
            continue
        seed_dict = runs[str(k)]
        seed_use = '42' if '42' in seed_dict else next(iter(seed_dict))
        run = seed_dict[seed_use]
        for kst in show_kstars:
            rec = run['by_k_star'].get(str(kst))
            if rec is None:
                continue
            profile = rec['profile_mean']
            vals = '  '.join(f'{v:>4.2f}' for v in profile)
            print(f"k={k:<2} k*={kst:<2}: {vals}")

    print()
    print('-- CONFIRMATION ----------------------------------------')
    c = confirmation
    print(f"L1  pre_norm < 0.10×post_norm        "
          f"[{c['L1']['status']}]  ratio={c['L1']['value']:.4f}")
    print(f"L2  post_norm > 0 all injections     "
          f"[{c['L2']['status']}]")
    print(f"L3  accuracy(k=lo) > 0.50            "
          f"[{c['L3']['status']}]  value={c['L3']['value']:.3f}")
    print(f"L4  pre/post ratio(k=lo) > 5.0       "
          f"[{c['L4']['status']}]  value={c['L4']['value']:.2f}")
    print(f"L5  accuracy(k=hi) > 0.30            "
          f"[{c['L5']['status']}]  value={c['L5']['value']:.3f}")
    print(f"L6  heatmap triangular pattern       "
          f"[{c['L6']['status']}]  below/above={c['L6']['value']:.4f}")

    print()
    all_met = all(v['status'] == 'CONFIRMED' for v in c.values())
    print('-- OVERALL ---------------------------------------------')
    print(f"All confirmations met: {'YES' if all_met else 'NO'}")
    print(f"Definition 4.2 status: {def42_status(c)}")
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
        k_test  = K_TEST_QUICK
        seeds   = SEEDS_QUICK
        inject  = INJECT_QUICK
        n_dirs  = N_DIRS_QUICK
        n_probe = N_PROBE_QUICK
        print('[QUICK MODE]')
    else:
        k_test  = K_TEST
        seeds   = SEEDS
        inject  = INJECT_LAYERS
        n_dirs  = N_DIRS
        n_probe = N_PROBE

    device = get_device()
    print(f"Script:  {SCRIPT_NAME}")
    print(f"k tested: {k_test}")
    print(f"Seeds:    {seeds}")
    print(f"Inject:   {inject}")
    print(f"N_DIRS:   {n_dirs}   N_PROBE: {n_probe}   EPS: {EPS}")

    runs: dict = {}
    for k in k_test:
        runs[str(k)] = {}
        for seed in seeds:
            t0 = time.time()
            try:
                model, final_loss = load_model(k, seed, MODELS_DIR, device)
            except FileNotFoundError as e:
                print(str(e), file=sys.stderr)
                return 1
            print(f"[loaded] k={k} seed={seed} (loss={final_loss:.4f})",
                  flush=True)

            def _print(line):
                print('  ' + line, flush=True)

            res = measure_for_model(
                model, k, seed, inject, n_dirs, n_probe, device,
                progress_print=_print)
            runs[str(k)][str(seed)] = res
            elapsed = time.time() - t0
            print(f"k={k:<3}seed={seed:<6}accuracy={res['n_correct']}/"
                  f"{res['n_total']}  pre/post ratio="
                  f"{res['pre_post_ratio_mean']:.2f}×  ({elapsed:.0f}s)",
                  flush=True)

    aggregate = aggregate_runs(runs, inject)
    confirmation = evaluate_confirmation(aggregate, runs, inject, k_test)

    meta = {
        'script': SCRIPT_NAME,
        'date': datetime.datetime.now().isoformat(),
        'device': str(device),
        'k_test': k_test,
        'seeds': seeds,
        'quick_mode': args.quick,
        'config': {
            'n_probe': n_probe,
            'eps': EPS,
            'n_dirs': n_dirs,
            'num_layers': NUM_LAYERS,
            'transition_frac': TRANSITION_FRAC,
            'models_dir': MODELS_DIR,
            'inject_layers': inject,
        },
    }

    output = {
        'meta': meta,
        'runs': runs,
        'aggregate': aggregate,
        'confirmation': confirmation,
        'def42_status': def42_status(confirmation),
    }
    save_json(output, os.path.join(RESULTS_DIR,
                                     f'{SCRIPT_NAME}_results.json'))

    # Figures (per k)
    for k in k_test:
        if str(k) in aggregate:
            fig_heatmap(aggregate[str(k)], str(k), inject)
    fig_localisation_accuracy(aggregate, k_test)
    fig_prepost_ratio(aggregate, k_test)
    fig_example_profiles(runs, k_test)
    write_html_report(meta, aggregate, confirmation, k_test)

    print_results_summary(meta, aggregate, runs, confirmation, k_test, inject)
    return 0


if __name__ == '__main__':
    sys.exit(main())
