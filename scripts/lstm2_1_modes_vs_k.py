"""
lstm2_1_modes_vs_k.py — Three Failure Modes Across k Domains
=============================================================
Trains a fresh generalist LSTM for each (k, seed) combination and measures
all three failure modes. Establishes the k-sweep baseline.

All model/utility code is imported from lstm2_model.py.
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
import datetime
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from lstm2_model import (
    VOCAB_SIZE, D_MODEL, NUM_LAYERS, SEQ_LEN, SEQS_PER_DOMAIN,
    BATCH_SIZE, EPOCHS, LR, K_VALUES, SEEDS_BASE, SEEDS_EXT,
    PREFIX_LEN, GEN_LEN, ERR_POS, EPS_ERR,
    GeneralistLSTM, domain_params, make_domain_sequences,
    make_training_data, train_model, get_weight_matrix,
    compute_svd_basis, compute_dimensional_excess,
    set_seed, get_device, make_results_dir, save_json,
)


SCRIPT_NAME = 'lstm2_1_modes_vs_k'
RESULTS_DIR = make_results_dir(SCRIPT_NAME)

N_CS_PER_DOMAIN  = 50
N_KNOWN_DOMAINS  = 5
N_UNKNOWN_DOMS   = 3
N_ENTROPY        = 80

# SPECTRUM_K = [1, 5, 20, 50]
SPECTRUM_K = [1, 5, 10] 
# SEEDS = SEEDS_EXT
SEEDS = SEEDS_BASE

K_VALUES_QUICK = [1, 5, 10]
SEEDS_QUICK    = SEEDS_EXT[:3]
N_CS_QUICK     = 20
N_ENTROPY_QUICK = 30


# ===========================================================================
# Sampled domain selection
# ===========================================================================

def sampled_known_domains(k: int, n_max: int) -> list[int]:
    """Sample domain indices from training set [0, k-1].
    If k <= n_max, return all; else evenly-spaced subset of length n_max."""
    if k <= n_max:
        return list(range(k))
    return [int(round(i * (k - 1) / (n_max - 1))) for i in range(n_max)]


# ===========================================================================
# Measurement A — Correction Sensitivity (CS)
# ===========================================================================

def cs_for_domain(model: GeneralistLSTM, domain_idx: int, n_trials: int,
                  device: torch.device) -> float:
    """Measure CS for a single domain.

    For each of n_trials sequences:
      - Build clean prefix from sequence (PREFIX_LEN tokens)
      - Build corrupt prefix: clone with ERR_POS replaced by max(2, (tok+11) % VOCAB_SIZE)
      - Greedy generate GEN_LEN tokens from each
      - cs_j = mean disagreement on the GEN_LEN generated tokens
    Returns mean over trials.
    """
    seqs = make_domain_sequences(domain_idx, n_trials, seq_len=PREFIX_LEN + 2)
    seqs = seqs.to(device)
    cs_vals = []
    for j in range(n_trials):
        prefix_clean = seqs[j:j + 1, :PREFIX_LEN]
        prefix_corrupt = prefix_clean.clone()
        tok = int(prefix_corrupt[0, ERR_POS].item())
        new_tok = max(2, (tok + 11) % VOCAB_SIZE)
        prefix_corrupt[0, ERR_POS] = new_tok
        gen_clean = model.generate_greedy(prefix_clean, GEN_LEN)
        gen_corrupt = model.generate_greedy(prefix_corrupt, GEN_LEN)
        gen_clean_tail = gen_clean[0, PREFIX_LEN:].cpu().numpy()
        gen_corrupt_tail = gen_corrupt[0, PREFIX_LEN:].cpu().numpy()
        cs_j = float(np.mean(gen_clean_tail != gen_corrupt_tail))
        cs_vals.append(cs_j)
    return float(np.mean(cs_vals))


def measure_cs(model: GeneralistLSTM, k: int, n_cs: int,
               device: torch.device) -> dict:
    known_idxs = sampled_known_domains(k, N_KNOWN_DOMAINS)
    cs_by_known = {}
    for d in known_idxs:
        cs_by_known[str(d)] = cs_for_domain(model, d, n_cs, device)

    unknown_idxs = [k, k + 1, k + 2]
    cs_by_unknown = {}
    for d in unknown_idxs:
        cs_by_unknown[str(d)] = cs_for_domain(model, d, n_cs, device)

    cs_known = float(np.mean(list(cs_by_known.values()))) if cs_by_known else 0.0
    cs_unknown = float(np.mean(list(cs_by_unknown.values()))) if cs_by_unknown else 0.0
    cs_gap = cs_unknown - cs_known
    return {
        'cs_known': cs_known,
        'cs_unknown': cs_unknown,
        'cs_gap': cs_gap,
        'cs_by_known_domain': cs_by_known,
        'cs_by_unknown_domain': cs_by_unknown,
    }


# ===========================================================================
# Measurement B — Dimensional Excess (DE)
# ===========================================================================

def measure_de(model: GeneralistLSTM, k: int) -> dict:
    targets = ['output'] + list(range(NUM_LAYERS))
    per_matrix = {}
    de_vals = []
    for target in targets:
        W = get_weight_matrix(model, target)
        info = compute_dimensional_excess(W)
        key = 'output' if target == 'output' else f'lstm_{target}'
        per_matrix[key] = info
        de_vals.append(info['de'])
    sv_spectrum = None
    if k in SPECTRUM_K:
        W_out = get_weight_matrix(model, 'output')
        info_out = compute_dimensional_excess(W_out)
        sv_spectrum = info_out['singular_values']
    return {
        'mean_de': float(np.mean(de_vals)),
        'per_matrix': per_matrix,
        'sv_spectrum': sv_spectrum,
    }


# ===========================================================================
# Measurement C — Output Entropy (H)
# ===========================================================================

def entropy_for_domain(model: GeneralistLSTM, domain_idx: int, n_seqs: int,
                       device: torch.device) -> float:
    seqs = make_domain_sequences(domain_idx, n_seqs, seq_len=PREFIX_LEN + 2)
    seqs = seqs.to(device)
    model.eval()
    h_vals = []
    with torch.no_grad():
        for j in range(n_seqs):
            seq = seqs[j:j + 1, :PREFIX_LEN].clone()
            step_entropies = []
            for _ in range(GEN_LEN):
                logits = model(seq)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                H_t = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                step_entropies.append(float(H_t.item()))
                next_tok = logits.argmax(-1, keepdim=True)
                seq = torch.cat([seq, next_tok], dim=1)
            h_vals.append(float(np.mean(step_entropies)))
    return float(np.mean(h_vals))


def measure_h(model: GeneralistLSTM, k: int, n_entropy: int,
              device: torch.device) -> dict:
    known_idxs = sampled_known_domains(k, 3)
    h_known_vals = [entropy_for_domain(model, d, n_entropy, device) for d in known_idxs]
    h_known = float(np.mean(h_known_vals)) if h_known_vals else 0.0

    unknown_idxs = [k, k + 1, k + 2]
    h_unknown_vals = [entropy_for_domain(model, d, n_entropy, device) for d in unknown_idxs]
    h_unknown = float(np.mean(h_unknown_vals))

    h_ratio = h_unknown / max(h_known, 1e-12)
    return {
        'h_known': h_known,
        'h_unknown': h_unknown,
        'h_ratio': h_ratio,
    }


# ===========================================================================
# Aggregation and interaction
# ===========================================================================

def aggregate_runs(runs: dict) -> dict:
    agg = {}
    for k_str, seed_results in runs.items():
        cs_known = [r['cs']['cs_known'] for r in seed_results.values()]
        cs_unknown = [r['cs']['cs_unknown'] for r in seed_results.values()]
        cs_gap = [r['cs']['cs_gap'] for r in seed_results.values()]
        de_mean = [r['de']['mean_de'] for r in seed_results.values()]
        h_ratio = [r['h']['h_ratio'] for r in seed_results.values()]
        agg[k_str] = {
            'cs_known':   {'mean': float(np.mean(cs_known)),   'std': float(np.std(cs_known))},
            'cs_unknown': {'mean': float(np.mean(cs_unknown)), 'std': float(np.std(cs_unknown))},
            'cs_gap':     {'mean': float(np.mean(cs_gap)),     'std': float(np.std(cs_gap))},
            'de_mean':    {'mean': float(np.mean(de_mean)),    'std': float(np.std(de_mean))},
            'h_ratio':    {'mean': float(np.mean(h_ratio)),    'std': float(np.std(h_ratio))},
        }
    return agg


def pearson_r(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_interaction(aggregate: dict, k_values: list[int]) -> dict:
    de_means = [aggregate[str(k)]['de_mean']['mean'] for k in k_values]
    cs_unknown_means = [aggregate[str(k)]['cs_unknown']['mean'] for k in k_values]
    cs_gap_means = [aggregate[str(k)]['cs_gap']['mean'] for k in k_values]
    return {
        'pearson_r_de_vs_cs_unknown': pearson_r(de_means, cs_unknown_means),
        'pearson_r_de_vs_cs_gap':     pearson_r(de_means, cs_gap_means),
    }


# ===========================================================================
# Confirmation criteria
# ===========================================================================

def evaluate_confirmation(aggregate: dict, interaction: dict,
                          k_values: list[int]) -> dict:
    k1 = str(k_values[0])
    k_last = str(k_values[-1])

    def status(passed: bool) -> str:
        return 'CONFIRMED' if passed else 'NOT MET'

    # k=1 might not be in a quick run; use first k as proxy "k1"
    k1_cs_gap = aggregate[k1]['cs_gap']['mean']
    k1_cs_floor = aggregate[k1]['cs_known']['mean']
    k_last_cs_floor = aggregate[k_last]['cs_known']['mean']
    k_last_cs_gap = aggregate[k_last]['cs_gap']['mean']
    k1_de = aggregate[k1]['de_mean']['mean']
    k_last_de = aggregate[k_last]['de_mean']['mean']
    k1_h_ratio = aggregate[k1]['h_ratio']['mean']
    k_last_h_ratio = aggregate[k_last]['h_ratio']['mean']
    r_de_csu = interaction['pearson_r_de_vs_cs_unknown']

    return {
        'C1': {'status': status(k1_cs_gap > 0.25),
               'value': k1_cs_gap, 'criterion': '> 0.25'},
        'C2': {'status': status(k_last_cs_floor > k1_cs_floor * 1.5),
               'k1': k1_cs_floor, 'k_last': k_last_cs_floor,
               'criterion': f'CS_floor(k={k_last}) > CS_floor(k={k1}) x 1.5'},
        'C3': {'status': status(k_last_cs_gap < k1_cs_gap),
               'k1': k1_cs_gap, 'k_last': k_last_cs_gap,
               'criterion': f'CS_gap(k={k_last}) < CS_gap(k={k1})'},
        'C4': {'status': status(k1_de > 5.0),
               'value': k1_de, 'criterion': '> 5.0'},
        'C5': {'status': status(k_last_de > k1_de),
               'k1': k1_de, 'k_last': k_last_de,
               'criterion': f'DE(k={k_last}) > DE(k={k1})'},
        'C6': {'status': status(k1_h_ratio > 3.0),
               'value': k1_h_ratio, 'criterion': '> 3.0x'},
        'C7': {'status': status(k_last_h_ratio < k1_h_ratio),
               'k1': k1_h_ratio, 'k_last': k_last_h_ratio,
               'criterion': f'H_ratio(k={k_last}) < H_ratio(k={k1})'},
        'C8': {'status': status(r_de_csu > 0),
               'value': r_de_csu, 'criterion': 'r > 0'},
    }


# ===========================================================================
# Figures
# ===========================================================================

def _save_fig(fig, base: str):
    fig.savefig(os.path.join(RESULTS_DIR, base + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(RESULTS_DIR, base + '.png'), bbox_inches='tight', dpi=120)
    plt.close(fig)


def fig_three_modes(aggregate: dict, k_values: list[int]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cs_gap_mean = [aggregate[str(k)]['cs_gap']['mean'] for k in k_values]
    cs_gap_std  = [aggregate[str(k)]['cs_gap']['std'] for k in k_values]
    de_mean     = [aggregate[str(k)]['de_mean']['mean'] for k in k_values]
    de_std      = [aggregate[str(k)]['de_mean']['std'] for k in k_values]
    h_mean      = [aggregate[str(k)]['h_ratio']['mean'] for k in k_values]
    h_std       = [aggregate[str(k)]['h_ratio']['std'] for k in k_values]

    axes[0].errorbar(k_values, cs_gap_mean, yerr=cs_gap_std, marker='o',
                     color='#d62728', capsize=3)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('k (training domains)')
    axes[0].set_ylabel('CS gap (unknown - known)')
    axes[0].set_title('Mode 1: CS gap vs k')
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(k_values, de_mean, yerr=de_std, marker='s',
                     color='#1f77b4', capsize=3)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('k (training domains)')
    axes[1].set_ylabel('DE mean (across all weight matrices)')
    axes[1].set_title('Mode 2: DE vs k')
    axes[1].grid(alpha=0.3)

    axes[2].errorbar(k_values, h_mean, yerr=h_std, marker='^',
                     color='#2ca02c', capsize=3)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('k (training domains)')
    axes[2].set_ylabel('H ratio (unknown / known)')
    axes[2].set_title('Mode 3: H ratio vs k')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, 'fig_three_modes')


def fig_cs_decomposition(aggregate: dict, k_values: list[int]):
    fig, ax = plt.subplots(figsize=(8, 5))
    cs_known = [aggregate[str(k)]['cs_known']['mean'] for k in k_values]
    cs_known_std = [aggregate[str(k)]['cs_known']['std'] for k in k_values]
    cs_unknown = [aggregate[str(k)]['cs_unknown']['mean'] for k in k_values]
    cs_unknown_std = [aggregate[str(k)]['cs_unknown']['std'] for k in k_values]

    ax.fill_between(k_values, cs_known, cs_unknown, color='#888', alpha=0.2,
                    label='CS gap')
    ax.errorbar(k_values, cs_known, yerr=cs_known_std, marker='o',
                color='#1f77b4', label='CS_known (floor)', capsize=3)
    ax.errorbar(k_values, cs_unknown, yerr=cs_unknown_std, marker='s',
                color='#d62728', label='CS_unknown (ceiling)', capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('k (training domains)')
    ax.set_ylabel('Correction Sensitivity')
    ax.set_title('CS decomposition: rising floor with k (k-specificity)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, 'fig_cs_decomposition')


def fig_svd_spectrum(runs: dict, k_values: list[int]):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('viridis')
    overlay_ks = [k for k in SPECTRUM_K if k in k_values]
    if not overlay_ks:
        ax.text(0.5, 0.5, 'No spectrum k overlap with run', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        _save_fig(fig, 'fig_svd_spectrum')
        return
    seed_first = sorted(runs[str(overlay_ks[0])].keys())[0]
    for i, k in enumerate(overlay_ks):
        seeds_for_k = sorted(runs[str(k)].keys())
        seed_use = seed_first if seed_first in seeds_for_k else seeds_for_k[0]
        sv = runs[str(k)][seed_use]['de'].get('sv_spectrum')
        if sv is None:
            continue
        color = cmap(i / max(len(overlay_ks) - 1, 1))
        ax.plot(range(1, len(sv) + 1), sv, color=color, label=f'k={k}')
    ax.set_yscale('log')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (log scale)')
    ax.set_title('Output layer SVD spectrum overlay')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, 'fig_svd_spectrum')


def fig_mode_interaction(runs: dict, k_values: list[int]):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    xs_all, ys_all = [], []
    for i, k in enumerate(k_values):
        color = cmap(i / max(len(k_values) - 1, 1))
        xs, ys = [], []
        for seed_str, run in runs[str(k)].items():
            xs.append(run['de']['mean_de'])
            ys.append(run['cs']['cs_unknown'])
        ax.scatter(xs, ys, color=color, label=f'k={k}', alpha=0.7, s=30)
        xs_all.extend(xs)
        ys_all.extend(ys)

    if len(xs_all) >= 2 and np.std(xs_all) > 1e-12:
        slope, intercept = np.polyfit(xs_all, ys_all, 1)
        xline = np.linspace(min(xs_all), max(xs_all), 100)
        ax.plot(xline, slope * xline + intercept, 'k--', alpha=0.6,
                label='regression')
        r = pearson_r(xs_all, ys_all)
    else:
        r = 0.0

    ax.set_xlabel('DE_mean (per (k, seed))')
    ax.set_ylabel('CS_unknown (per (k, seed))')
    ax.set_title(f'Mode interaction — Pearson r = {r:.3f}')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    fig.tight_layout()
    _save_fig(fig, 'fig_mode_interaction')


# ===========================================================================
# HTML report
# ===========================================================================

def _img_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def write_html_report(meta: dict, aggregate: dict, interaction: dict,
                      confirmation: dict, k_values: list[int]):
    parts = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    parts.append('<title>lstm2_1_modes_vs_k</title>')
    parts.append("""<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: right; }
th { background: #eee; text-align: center; }
td.label { text-align: left; font-weight: 500; }
.confirmed { color: #1a7a1a; font-weight: bold; }
.notmet { color: #b00020; font-weight: bold; }
img { max-width: 100%; border: 1px solid #ddd; margin: 0.5em 0; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
</style></head><body>""")
    parts.append(f"<h1>lstm2_1_modes_vs_k — {meta['date']}</h1>")
    parts.append(f"<p>Device: {meta['device']} | Mode: {'QUICK' if meta['quick_mode'] else 'FULL'} | "
                 f"k values: {meta['k_values']} | seeds: {len(meta['seeds'])}</p>")

    # Summary table
    parts.append('<h2>Confirmation summary</h2><table><tr><th>ID</th><th>Check</th>'
                 '<th>Status</th><th>Detail</th></tr>')
    labels = {
        'C1': 'k=1 CS gap > 0.25',
        'C2': 'CS floor rises x1.5',
        'C3': 'CS gap narrows',
        'C4': 'k=1 DE > 5.0',
        'C5': 'DE rises with k',
        'C6': 'k=1 H ratio > 3x',
        'C7': 'H ratio falls with k',
        'C8': 'Pearson r(DE, CS_unknown) > 0',
    }
    for cid in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        c = confirmation[cid]
        cls = 'confirmed' if c['status'] == 'CONFIRMED' else 'notmet'
        if 'value' in c:
            detail = f"value = {c['value']:.4f}"
        else:
            detail = f"k1 = {c.get('k1', 0):.4f}, k_last = {c.get('k_last', 0):.4f}"
        parts.append(f'<tr><td class="label">{cid}</td>'
                     f'<td class="label">{labels[cid]}</td>'
                     f'<td class="{cls}">{c["status"]}</td>'
                     f'<td class="label">{detail}</td></tr>')
    parts.append('</table>')

    # Figures
    for title, base in [('Three modes vs k', 'fig_three_modes'),
                        ('CS decomposition', 'fig_cs_decomposition'),
                        ('SVD spectrum overlay', 'fig_svd_spectrum'),
                        ('Mode interaction', 'fig_mode_interaction')]:
        png = os.path.join(RESULTS_DIR, base + '.png')
        if os.path.exists(png):
            b64 = _img_b64(png)
            parts.append(f'<h2>{title}</h2><img src="data:image/png;base64,{b64}">')

    # Aggregate table
    parts.append('<h2>Aggregate table</h2><table><tr>'
                 '<th>k</th><th>CS_known</th><th>CS_unknown</th>'
                 '<th>CS_gap</th><th>DE_mean</th><th>H_ratio</th></tr>')
    for k in k_values:
        a = aggregate[str(k)]
        parts.append(f"<tr><td>{k}</td>"
                     f"<td>{a['cs_known']['mean']:.4f} ± {a['cs_known']['std']:.4f}</td>"
                     f"<td>{a['cs_unknown']['mean']:.4f} ± {a['cs_unknown']['std']:.4f}</td>"
                     f"<td>{a['cs_gap']['mean']:.4f} ± {a['cs_gap']['std']:.4f}</td>"
                     f"<td>{a['de_mean']['mean']:.3f} ± {a['de_mean']['std']:.3f}</td>"
                     f"<td>{a['h_ratio']['mean']:.3f} ± {a['h_ratio']['std']:.3f}</td></tr>")
    parts.append('</table>')

    parts.append(f"<h2>Mode interaction</h2><p>Pearson r(DE, CS_unknown) = "
                 f"{interaction['pearson_r_de_vs_cs_unknown']:.4f}<br>"
                 f"Pearson r(DE, CS_gap) = {interaction['pearson_r_de_vs_cs_gap']:.4f}</p>")
    parts.append('</body></html>')

    out = os.path.join(RESULTS_DIR, f'{SCRIPT_NAME}_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ===========================================================================
# RESULTS SUMMARY
# ===========================================================================

def print_results_summary(meta: dict, aggregate: dict, interaction: dict,
                          confirmation: dict, k_values: list[int]):
    print()
    print('=' * 55)
    print('RESULTS SUMMARY — lstm2_1_modes_vs_k')
    print('=' * 55)
    print(f"Date:    {meta['date']}")
    print(f"Device:  {meta['device']}")
    print(f"Mode:    {'QUICK' if meta['quick_mode'] else 'FULL'}")
    print(f"k values tested: {k_values}")
    print(f"Seeds:   {meta['seeds']}")
    print()
    print('-- AGGREGATE TABLE (mean ± std across seeds) ----------')
    print(f"{'k':<5}{'CS_known':<12}{'CS_unknown':<12}{'CS_gap':<10}"
          f"{'DE_mean':<12}{'H_ratio':<12}")
    for k in k_values:
        a = aggregate[str(k)]
        print(f"{k:<5}{a['cs_known']['mean']:.4f}±{a['cs_known']['std']:.3f}  "
              f"{a['cs_unknown']['mean']:.4f}±{a['cs_unknown']['std']:.3f}  "
              f"{a['cs_gap']['mean']:.4f}±{a['cs_gap']['std']:.3f}  "
              f"{a['de_mean']['mean']:7.3f}±{a['de_mean']['std']:.2f}  "
              f"{a['h_ratio']['mean']:7.3f}±{a['h_ratio']['std']:.2f}")
    print()
    print('-- CONFIRMATION ----------------------------------------')
    c = confirmation
    print(f"C1  k=1 CS gap > 0.25              [{c['C1']['status']}]  "
          f"value={c['C1']['value']:.4f}")
    print(f"C2  CS floor rises ×1.5            [{c['C2']['status']}]  "
          f"k1={c['C2']['k1']:.4f}  k_last={c['C2']['k_last']:.4f}")
    print(f"C3  CS gap narrows                 [{c['C3']['status']}]  "
          f"k1={c['C3']['k1']:.4f}  k_last={c['C3']['k_last']:.4f}")
    print(f"C4  k=1 DE > 5.0                   [{c['C4']['status']}]  "
          f"value={c['C4']['value']:.3f}")
    print(f"C5  DE rises with k                [{c['C5']['status']}]  "
          f"k1={c['C5']['k1']:.3f}  k_last={c['C5']['k_last']:.3f}")
    print(f"C6  k=1 H ratio > 3×               [{c['C6']['status']}]  "
          f"value={c['C6']['value']:.3f}")
    print(f"C7  H ratio falls with k           [{c['C7']['status']}]  "
          f"k1={c['C7']['k1']:.3f}  k_last={c['C7']['k_last']:.3f}")
    print(f"C8  Pearson r(DE, CS_unknown) > 0  [{c['C8']['status']}]  "
          f"r={c['C8']['value']:.4f}")
    print()
    all_met = all(v['status'] == 'CONFIRMED' for v in confirmation.values())
    print('-- OVERALL ---------------------------------------------')
    print(f"All confirmations met: {'YES' if all_met else 'NO'}")
    print(f"Output: results/{SCRIPT_NAME}/")
    print('=' * 55)


# ===========================================================================
# Main
# ===========================================================================

def main():
    quick = '--quick' in sys.argv
    if quick:
        k_values = K_VALUES_QUICK
        seeds = SEEDS_QUICK
        n_cs = N_CS_QUICK
        n_entropy = N_ENTROPY_QUICK
        epochs = EPOCHS  # keep training quality
        print('[QUICK MODE]')
    else:
        k_values = K_VALUES
        seeds = SEEDS
        n_cs = N_CS_PER_DOMAIN
        n_entropy = N_ENTROPY
        epochs = EPOCHS

    device = get_device()
    print(f"Script: {SCRIPT_NAME}")
    print(f"k values: {k_values}")
    print(f"Seeds: {seeds}")
    print(f"Quick: {quick}")

    partial_path = os.path.join(RESULTS_DIR,
                                 f'{SCRIPT_NAME}_runs_partial{"_quick" if quick else ""}.json')
    runs: dict = {}
    if os.path.exists(partial_path):
        try:
            with open(partial_path, 'r', encoding='utf-8') as f:
                runs = json.load(f)
            print(f"[resume] loaded partial runs from {partial_path}")
        except Exception as e:
            print(f"[resume] could not load partial runs: {e!r}")
            runs = {}

    for k in k_values:
        runs.setdefault(str(k), {})
        for seed in seeds:
            if str(seed) in runs[str(k)]:
                print(f"k={k:<3} seed={seed:<5} [resume: already done]")
                continue
            t0 = time.time()
            set_seed(seed)
            data = make_training_data(k=k, seqs_per_domain=SEQS_PER_DOMAIN,
                                       seq_len=SEQ_LEN)
            model = GeneralistLSTM()
            model, final_loss = train_model(
                model, data, epochs=epochs, lr=LR,
                batch_size=BATCH_SIZE, device=device, verbose=False)
            if final_loss > 1.0:
                print(f"  [WARNING] training may not have converged "
                      f"(loss={final_loss:.3f})")
            cs = measure_cs(model, k, n_cs, device)
            de = measure_de(model, k)
            h = measure_h(model, k, n_entropy, device)
            runs[str(k)][str(seed)] = {'cs': cs, 'de': de, 'h': h,
                                        'final_loss': final_loss}
            elapsed = time.time() - t0
            print(f"k={k:<3} seed={seed:<5} [done {epochs}ep, {elapsed:.0f}s, "
                  f"loss={final_loss:.3f}]  "
                  f"CS_gap={cs['cs_gap']:.3f}  DE={de['mean_de']:.2f}  "
                  f"H_ratio={h['h_ratio']:.2f}x")
            try:
                save_json(runs, partial_path)
            except Exception as e:
                print(f"  [warn] could not save partial: {e!r}")

    aggregate = aggregate_runs(runs)
    interaction = compute_interaction(aggregate, k_values)
    confirmation = evaluate_confirmation(aggregate, interaction, k_values)

    meta = {
        'script': SCRIPT_NAME,
        'date': datetime.datetime.now().isoformat(),
        'device': str(device),
        'k_values': k_values,
        'seeds': seeds,
        'quick_mode': quick,
        'model_config': {
            'vocab_size': VOCAB_SIZE, 'd_model': D_MODEL,
            'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN,
            'seqs_per_domain': SEQS_PER_DOMAIN,
            'total_seqs_per_k': {str(k): k * SEQS_PER_DOMAIN
                                 for k in k_values},
            'epochs': epochs, 'lr': LR,
        },
        'measurement_config': {
            'n_cs_per_domain': n_cs,
            'n_known_domains': N_KNOWN_DOMAINS,
            'n_unknown_domains': N_UNKNOWN_DOMS,
            'n_entropy': n_entropy,
            'err_pos': ERR_POS,
            'gen_len': GEN_LEN,
            'prefix_len': PREFIX_LEN,
        },
    }

    output = {
        'meta': meta,
        'runs': runs,
        'aggregate': aggregate,
        'interaction': interaction,
        'confirmation': confirmation,
    }
    save_json(output, os.path.join(RESULTS_DIR, f'{SCRIPT_NAME}_results.json'))

    fig_three_modes(aggregate, k_values)
    fig_cs_decomposition(aggregate, k_values)
    fig_svd_spectrum(runs, k_values)
    fig_mode_interaction(runs, k_values)
    write_html_report(meta, aggregate, interaction, confirmation, k_values)

    print_results_summary(meta, aggregate, interaction, confirmation, k_values)


if __name__ == '__main__':
    main()
