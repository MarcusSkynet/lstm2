"""
lstm2_2_arch_sweep.py — Architecture Capacity Sweep
====================================================
Trains models across width / depth / paired-scaling sweeps and measures
how the breakdown point of the three failure modes scales with capacity.

All model/utility code imported from lstm2_model.py. Each sweep saves
incrementally to results/lstm2_2_arch_sweep/lstm2_2_arch_sweep_{sweep}_partial.json
so an interrupted run resumes from the next (config, k, seed).
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
import datetime
import base64
import argparse

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
    BATCH_SIZE, LR, SEEDS_BASE,
    PREFIX_LEN, GEN_LEN, ERR_POS, EPS_ERR,
    GeneralistLSTM, domain_params, make_domain_sequences,
    make_training_data, train_model, get_weight_matrix,
    compute_svd_basis, compute_dimensional_excess,
    set_seed, get_device, make_results_dir, save_json,
)


SCRIPT_NAME = 'lstm2_2_arch_sweep'
RESULTS_DIR = make_results_dir(SCRIPT_NAME)


# ===========================================================================
# 2. Architecture configurations
# ===========================================================================

SWEEP_A = [
    {'name': 'A-XS', 'd_model':  64, 'num_layers': 2},
    {'name': 'A-S',  'd_model':  96, 'num_layers': 2},
    {'name': 'A-M',  'd_model': 128, 'num_layers': 2},
    {'name': 'A-L',  'd_model': 192, 'num_layers': 2},
    {'name': 'A-XL', 'd_model': 256, 'num_layers': 2},
]

SWEEP_B = [
    {'name': 'B-2',  'd_model': 256, 'num_layers':  2},
    {'name': 'B-4',  'd_model': 256, 'num_layers':  4},
    {'name': 'B-6',  'd_model': 256, 'num_layers':  6},
    {'name': 'B-8',  'd_model': 256, 'num_layers':  8},
    {'name': 'B-10', 'd_model': 256, 'num_layers': 10},
]

SWEEP_C = [
    {'name': 'C-XS', 'd_model':  64, 'num_layers':  2},
    {'name': 'C-S',  'd_model':  96, 'num_layers':  2},
    {'name': 'C-M',  'd_model': 128, 'num_layers':  4},
    {'name': 'C-L',  'd_model': 192, 'num_layers':  6},
    {'name': 'C-XL', 'd_model': 256, 'num_layers': 10},
]

SWEEPS = {'A': SWEEP_A, 'B': SWEEP_B, 'C': SWEEP_C}
SWEEP_LABEL = {
    'A': 'Width only',
    'B': 'Depth only',
    'C': 'Paired scaling',
}


def count_params(d_model: int, num_layers: int,
                 vocab_size: int = VOCAB_SIZE) -> int:
    """Count total trainable parameters for GeneralistLSTM variant."""
    embed  = vocab_size * d_model
    lstm   = num_layers * 4 * (d_model * d_model + d_model * d_model
                                + 2 * d_model)
    output = vocab_size * d_model
    return embed + lstm + output


# ===========================================================================
# 3. Training schedule
# ===========================================================================

EPOCHS_BY_SIZE   = {'XS': 300, 'S': 300, 'M': 250, 'L': 200, 'XL': 200}
EPOCHS_BY_LAYERS = {2: 300, 4: 250, 6: 225, 8: 200, 10: 200}


def epochs_for(config: dict, sweep: str) -> int:
    """Choose epoch count from sweep type and config name/layer count."""
    if sweep == 'B':
        return EPOCHS_BY_LAYERS[config['num_layers']]
    suffix = config['name'].split('-')[1]
    return EPOCHS_BY_SIZE[suffix]


# ===========================================================================
# 4. Quick / full configuration
# ===========================================================================

K_VALUES        = [1, 2, 3, 5, 10]
SEEDS           = SEEDS_BASE
N_CS            = 50
N_ENTROPY       = 80

K_VALUES_QUICK  = [1, 5, 10]
SEEDS_QUICK     = SEEDS_BASE[:3]
N_CS_QUICK      = 15
N_ENTROPY_QUICK = 20

N_KNOWN_DOMAINS = 5
SPECTRUM_K      = [1, 5, 10]


# ===========================================================================
# 5. Measurements (CS, DE, H) — copied from lstm2_1_modes_vs_k.py
# ===========================================================================

def sampled_known_domains(k: int, n_max: int) -> list[int]:
    if k <= n_max:
        return list(range(k))
    return [int(round(i * (k - 1) / (n_max - 1))) for i in range(n_max)]


def cs_for_domain(model: GeneralistLSTM, domain_idx: int, n_trials: int,
                  device: torch.device) -> float:
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
        gc = gen_clean[0, PREFIX_LEN:].cpu().numpy()
        gp = gen_corrupt[0, PREFIX_LEN:].cpu().numpy()
        cs_vals.append(float(np.mean(gc != gp)))
    return float(np.mean(cs_vals))


def measure_cs(model: GeneralistLSTM, k: int, n_cs: int,
               device: torch.device) -> dict:
    known_idxs = sampled_known_domains(k, N_KNOWN_DOMAINS)
    cs_by_known = {str(d): cs_for_domain(model, d, n_cs, device)
                   for d in known_idxs}
    unknown_idxs = [k, k + 1, k + 2]
    cs_by_unknown = {str(d): cs_for_domain(model, d, n_cs, device)
                     for d in unknown_idxs}
    cs_known = float(np.mean(list(cs_by_known.values()))) if cs_by_known else 0.0
    cs_unknown = float(np.mean(list(cs_by_unknown.values())))
    return {
        'cs_known': cs_known,
        'cs_unknown': cs_unknown,
        'cs_gap': cs_unknown - cs_known,
        'cs_by_known_domain': cs_by_known,
        'cs_by_unknown_domain': cs_by_unknown,
    }


def measure_de(model: GeneralistLSTM, k: int, num_layers: int) -> dict:
    targets = ['output'] + list(range(num_layers))
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
        sv_spectrum = per_matrix['output']['singular_values']
    return {
        'mean_de': float(np.mean(de_vals)),
        'per_matrix': per_matrix,
        'sv_spectrum': sv_spectrum,
    }


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
    h_known_vals = [entropy_for_domain(model, d, n_entropy, device)
                    for d in known_idxs]
    h_known = float(np.mean(h_known_vals)) if h_known_vals else 0.0
    unknown_idxs = [k, k + 1, k + 2]
    h_unknown_vals = [entropy_for_domain(model, d, n_entropy, device)
                      for d in unknown_idxs]
    h_unknown = float(np.mean(h_unknown_vals))
    h_ratio = h_unknown / max(h_known, 1e-12)
    return {
        'h_known': h_known,
        'h_unknown': h_unknown,
        'h_ratio': h_ratio,
    }


# ===========================================================================
# 6. Breakdown point detection
# ===========================================================================

def find_breakdown_k(aggregate_by_k: dict,
                     threshold: float = 0.05) -> int | None:
    for k_str in sorted(aggregate_by_k.keys(), key=int):
        if aggregate_by_k[k_str]['cs_gap']['mean'] < threshold:
            return int(k_str)
    return None


def find_k_half(aggregate_by_k: dict) -> int | None:
    ks = sorted(aggregate_by_k.keys(), key=int)
    if not ks:
        return None
    k1 = ks[0]
    base = aggregate_by_k[k1]['cs_gap']['mean']
    half = base / 2.0
    for k_str in ks:
        if aggregate_by_k[k_str]['cs_gap']['mean'] < half:
            return int(k_str)
    return None


# ===========================================================================
# 7. Aggregation
# ===========================================================================

def aggregate_runs(runs: dict) -> dict:
    """runs[config][k][seed] -> {cs, de, h, final_loss, converged}."""
    agg = {}
    for cname, kdict in runs.items():
        agg[cname] = {}
        for k_str, sdict in kdict.items():
            cs_known = [r['cs']['cs_known'] for r in sdict.values()]
            cs_unk   = [r['cs']['cs_unknown'] for r in sdict.values()]
            cs_gap   = [r['cs']['cs_gap'] for r in sdict.values()]
            de_mean  = [r['de']['mean_de'] for r in sdict.values()]
            h_ratio  = [r['h']['h_ratio'] for r in sdict.values()]
            n_conv   = sum(1 for r in sdict.values() if r.get('converged'))
            agg[cname][k_str] = {
                'cs_known':   {'mean': float(np.mean(cs_known)),
                               'std':  float(np.std(cs_known))},
                'cs_unknown': {'mean': float(np.mean(cs_unk)),
                               'std':  float(np.std(cs_unk))},
                'cs_gap':     {'mean': float(np.mean(cs_gap)),
                               'std':  float(np.std(cs_gap))},
                'de_mean':    {'mean': float(np.mean(de_mean)),
                               'std':  float(np.std(de_mean))},
                'h_ratio':    {'mean': float(np.mean(h_ratio)),
                               'std':  float(np.std(h_ratio))},
                'n_converged': int(n_conv),
                'n_failed':    int(len(sdict) - n_conv),
            }
    return agg


def compute_breakdown(aggregate: dict, configs: list) -> dict:
    out = {}
    for cfg in configs:
        cname = cfg['name']
        if cname not in aggregate:
            continue
        out[cname] = {
            'k_star': find_breakdown_k(aggregate[cname]),
            'k_half': find_k_half(aggregate[cname]),
            'n_params': count_params(cfg['d_model'], cfg['num_layers']),
        }
    return out


# ===========================================================================
# 8. Cross-validation
# ===========================================================================

REF_CS_GAP   = 0.273
REF_CS_TOL   = 0.10
REF_DE       = 48.06
REF_DE_TOL   = 5.0
CV_CONFIGS   = {'C-XL', 'B-10'}


def cross_validation(aggregate: dict, sweep: str) -> dict:
    """Returns the cross-validation block for the JSON output."""
    if sweep == 'A':
        return {'config': None, 'status': 'N/A',
                'cs_gap': None, 'de': None,
                'reference_cs_gap': REF_CS_GAP, 'reference_de': REF_DE}
    cv_cfg = 'C-XL' if sweep == 'C' else 'B-10'
    if cv_cfg not in aggregate or '1' not in aggregate[cv_cfg]:
        return {'config': cv_cfg, 'status': 'N/A',
                'cs_gap': None, 'de': None,
                'reference_cs_gap': REF_CS_GAP, 'reference_de': REF_DE}
    cs_gap = aggregate[cv_cfg]['1']['cs_gap']['mean']
    de = aggregate[cv_cfg]['1']['de_mean']['mean']
    ok = (abs(cs_gap - REF_CS_GAP) <= REF_CS_TOL and
          abs(de - REF_DE) <= REF_DE_TOL)
    status = 'OK' if ok else 'FAIL'
    return {'config': cv_cfg, 'status': status,
            'cs_gap': cs_gap, 'de': de,
            'reference_cs_gap': REF_CS_GAP, 'reference_de': REF_DE}


# ===========================================================================
# 9. Confirmation criteria
# ===========================================================================

def _kh(value):
    """Treat None as +infinity for comparison purposes."""
    return float('inf') if value is None else value


def evaluate_confirmation(sweep: str, aggregate: dict, breakdown: dict,
                           cv_block: dict, configs: list) -> dict:
    """Per-sweep confirmation criteria as defined in the SPEC."""
    def status(passed: bool) -> str:
        return 'CONFIRMED' if passed else 'NOT MET'

    def k1_metric(cname, key):
        if cname in aggregate and '1' in aggregate[cname]:
            return aggregate[cname]['1'][key]['mean']
        return None

    out = {}

    if sweep == 'A':
        kh_xl = _kh(breakdown['A-XL']['k_half']) if 'A-XL' in breakdown else None
        kh_s  = _kh(breakdown['A-S']['k_half']) if 'A-S' in breakdown else None
        gap_xs = k1_metric('A-XS', 'cs_gap')
        gap_xl = k1_metric('A-XL', 'cs_gap')
        de_xs = k1_metric('A-XS', 'de_mean')
        de_xl = k1_metric('A-XL', 'de_mean')
        out['A1'] = {'status': status(kh_xl is not None and kh_s is not None
                                       and kh_xl >= kh_s),
                     'value': {'A-XL': breakdown.get('A-XL', {}).get('k_half'),
                                'A-S':  breakdown.get('A-S',  {}).get('k_half')},
                     'criterion': 'k_half(A-XL) >= k_half(A-S)'}
        out['A2'] = {'status': status(gap_xs is not None and gap_xl is not None
                                       and gap_xs > gap_xl),
                     'value': {'A-XS': gap_xs, 'A-XL': gap_xl},
                     'criterion': 'CS_gap(k=1, A-XS) > CS_gap(k=1, A-XL)'}
        out['A3'] = {'status': status(de_xs is not None and de_xl is not None
                                       and de_xl > de_xs),
                     'value': {'A-XS': de_xs, 'A-XL': de_xl},
                     'criterion': 'DE(k=1, A-XL) > DE(k=1, A-XS)'}

    elif sweep == 'B':
        kh_2  = _kh(breakdown.get('B-2',  {}).get('k_half'))
        kh_10 = _kh(breakdown.get('B-10', {}).get('k_half'))
        b2_gap = k1_metric('B-2', 'cs_gap')
        all_gaps = {c['name']: k1_metric(c['name'], 'cs_gap') for c in configs}
        de_2  = k1_metric('B-2',  'de_mean')
        de_10 = k1_metric('B-10', 'de_mean')
        within = (b2_gap is not None and
                   all(g is not None and abs(g - b2_gap) <= 0.10
                       for g in all_gaps.values()))
        out['B1'] = {'status': status(kh_10 != float('inf')
                                       and kh_2 != float('inf')
                                       and kh_10 <= kh_2 * 1.5),
                     'value': {'B-2': breakdown.get('B-2', {}).get('k_half'),
                                'B-10': breakdown.get('B-10', {}).get('k_half')},
                     'criterion': 'k_half(B-10) <= k_half(B-2) * 1.5'}
        out['B2'] = {'status': status(within),
                     'value': all_gaps,
                     'criterion': 'CS_gap(k=1) within ±0.10 of B-2'}
        out['B3'] = {'status': status(de_10 is not None and de_2 is not None
                                       and de_10 > de_2),
                     'value': {'B-2': de_2, 'B-10': de_10},
                     'criterion': 'DE(k=1, B-10) > DE(k=1, B-2)'}
        out['B4'] = {'status': status(cv_block['status'] == 'OK'),
                     'value': cv_block['status'],
                     'criterion': 'cross-validation B-10 within tolerance'}

    elif sweep == 'C':
        kh_xs = _kh(breakdown.get('C-XS', {}).get('k_half'))
        kh_xl = _kh(breakdown.get('C-XL', {}).get('k_half'))
        gap_xs = k1_metric('C-XS', 'cs_gap')
        gap_xl = k1_metric('C-XL', 'cs_gap')
        de_xs = k1_metric('C-XS', 'de_mean')
        de_xl = k1_metric('C-XL', 'de_mean')
        # H-ratio inverted-U on the larger configs
        inverted = False
        for cname in ['C-L', 'C-XL']:
            if cname in aggregate:
                hk = aggregate[cname]
                # Need k=3 or k=5 and k=10 measured
                ks = {3, 5, 10} & set(int(k) for k in hk.keys())
                if {3, 10}.issubset(ks) or {5, 10}.issubset(ks):
                    peak = max((hk[str(k)]['h_ratio']['mean']
                                for k in ks if k in (3, 5)),
                                default=None)
                    h10 = hk['10']['h_ratio']['mean']
                    if peak is not None and h10 < peak:
                        inverted = True
                        break
        out['C1'] = {'status': status(cv_block['status'] == 'OK'),
                     'value': cv_block['status'],
                     'criterion': 'cross-validation C-XL within tolerance'}
        out['C2'] = {'status': status(kh_xs != float('inf')
                                       and kh_xs < kh_xl),
                     'value': {'C-XS': breakdown.get('C-XS', {}).get('k_half'),
                                'C-XL': breakdown.get('C-XL', {}).get('k_half')},
                     'criterion': 'k_half(C-XS) < k_half(C-XL)'}
        out['C3'] = {'status': status(gap_xs is not None and gap_xl is not None
                                       and gap_xs > gap_xl),
                     'value': {'C-XS': gap_xs, 'C-XL': gap_xl},
                     'criterion': 'CS_gap(k=1, C-XS) > CS_gap(k=1, C-XL)'}
        out['C4'] = {'status': status(de_xs is not None and de_xl is not None
                                       and de_xl > de_xs),
                     'value': {'C-XS': de_xs, 'C-XL': de_xl},
                     'criterion': 'DE(k=1, C-XL) > DE(k=1, C-XS)'}
        out['C5'] = {'status': status(inverted),
                     'value': {'inverted_u_present': inverted},
                     'criterion': 'H_ratio peaks at k=3-5, falls at k=10 in C-L or C-XL'}
    return out


# ===========================================================================
# 10. Figures
# ===========================================================================

def _save_fig(fig, base: str):
    fig.savefig(os.path.join(RESULTS_DIR, base + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(RESULTS_DIR, base + '.png'),
                 bbox_inches='tight', dpi=120)
    plt.close(fig)


def _config_shades(configs: list) -> dict:
    """Light-to-dark greyscale per config order."""
    n = len(configs)
    return {c['name']: plt.cm.viridis(0.15 + 0.75 * i / max(n - 1, 1))
            for i, c in enumerate(configs)}


def fig_cs_gap(sweep: str, aggregate: dict, breakdown: dict,
               configs: list, k_values: list[int]):
    fig, ax = plt.subplots(figsize=(8, 5))
    shades = _config_shades(configs)
    for cfg in configs:
        cname = cfg['name']
        if cname not in aggregate:
            continue
        ks = sorted(aggregate[cname].keys(), key=int)
        ks = [int(k) for k in ks]
        means = [aggregate[cname][str(k)]['cs_gap']['mean'] for k in ks]
        stds  = [aggregate[cname][str(k)]['cs_gap']['std']  for k in ks]
        np_ = breakdown.get(cname, {}).get('n_params', 0)
        label = f'{cname} ({np_/1e3:,.0f}K)'
        ax.errorbar(ks, means, yerr=stds, marker='o', capsize=3,
                     color=shades[cname], label=label)
        kh = breakdown.get(cname, {}).get('k_half')
        if kh is not None and kh in ks:
            i = ks.index(kh)
            ax.plot(kh, means[i], marker='|', color=shades[cname],
                     markersize=18, markeredgewidth=2)
    ax.axhline(0.05, color='black', linestyle=':', lw=1.0,
                label='breakdown threshold (0.05)')
    ax.set_xscale('log')
    ax.set_xlabel('k (training domains)')
    ax.set_ylabel('CS_gap (mean ± std)')
    ax.set_title(f'Sweep {sweep} — CS_gap vs k  ({SWEEP_LABEL[sweep]})')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save_fig(fig, f'fig_sweep_{sweep}_cs_gap')


def fig_k_half(sweep: str, breakdown: dict, configs: list):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs, ys, names = [], [], []
    open_xs, open_ys, open_names = [], [], []
    for cfg in configs:
        cname = cfg['name']
        if cname not in breakdown:
            continue
        if sweep == 'A':
            x = cfg['d_model']
        elif sweep == 'B':
            x = cfg['num_layers']
        else:
            x = float(np.log10(breakdown[cname]['n_params']))
        kh = breakdown[cname]['k_half']
        if kh is None:
            open_xs.append(x); open_ys.append(10); open_names.append(cname)
        else:
            xs.append(x); ys.append(kh); names.append(cname)
    if xs:
        ax.plot(xs, ys, 'o-', color='black', markersize=8,
                 markerfacecolor='0.30', label='k_half (finite)')
    if open_xs:
        ax.plot(open_xs, open_ys, marker='o', linestyle='none',
                 color='black', markerfacecolor='white', markersize=10,
                 label='k_half = None (plotted at 10)')
    # Annotate names
    for x, y, n in zip(xs + open_xs, ys + open_ys, names + open_names):
        ax.annotate(n, (x, y), textcoords='offset points', xytext=(6, 6),
                     fontsize=8)
    # Linear fit if ≥3 finite
    if len(xs) >= 3:
        slope, intercept = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, slope * xline + intercept, '--', color='0.4',
                 label=f'fit slope = {slope:.3f}')
    if sweep == 'A':
        ax.set_xlabel('D_MODEL')
    elif sweep == 'B':
        ax.set_xlabel('NUM_LAYERS')
    else:
        ax.set_xlabel('log10(n_params)')
    ax.set_ylabel('k_half')
    ax.set_title(f'Sweep {sweep} — k_half vs capacity')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, f'fig_sweep_{sweep}_k_half')


def fig_k1_modes(sweep: str, aggregate: dict, breakdown: dict,
                 configs: list):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xs, gaps, gap_e, des, de_e, hs, h_e = [], [], [], [], [], [], []
    for cfg in configs:
        cname = cfg['name']
        if cname not in aggregate or '1' not in aggregate[cname]:
            continue
        if sweep == 'A':
            x = cfg['d_model']
        elif sweep == 'B':
            x = cfg['num_layers']
        else:
            x = float(np.log10(breakdown[cname]['n_params']))
        a1 = aggregate[cname]['1']
        xs.append(x)
        gaps.append(a1['cs_gap']['mean']); gap_e.append(a1['cs_gap']['std'])
        des.append(a1['de_mean']['mean']); de_e.append(a1['de_mean']['std'])
        hs.append(a1['h_ratio']['mean']);  h_e.append(a1['h_ratio']['std'])
    axes[0].errorbar(xs, gaps, yerr=gap_e, marker='o', color='#d62728', capsize=3)
    axes[0].set_ylabel('CS_gap (k=1)'); axes[0].set_title('CS_gap')
    axes[1].errorbar(xs, des,  yerr=de_e,  marker='s', color='#1f77b4', capsize=3)
    axes[1].set_ylabel('DE_mean (k=1)'); axes[1].set_title('DE')
    axes[2].errorbar(xs, hs,   yerr=h_e,   marker='^', color='#2ca02c', capsize=3)
    axes[2].set_ylabel('H_ratio (k=1)'); axes[2].set_title('H_ratio')
    for ax in axes:
        if sweep == 'A':
            ax.set_xlabel('D_MODEL')
        elif sweep == 'B':
            ax.set_xlabel('NUM_LAYERS')
        else:
            ax.set_xlabel('log10(n_params)')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Sweep {sweep} — Three modes at k=1', fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, f'fig_sweep_{sweep}_k1_modes')


def fig_overlay_C(aggregate: dict, breakdown: dict, configs: list):
    fig, ax = plt.subplots(figsize=(8, 5))
    shades = _config_shades(configs)
    for cfg in configs:
        cname = cfg['name']
        if cname not in aggregate:
            continue
        ks = sorted(aggregate[cname].keys(), key=int)
        ks = [int(k) for k in ks]
        means = [aggregate[cname][str(k)]['cs_gap']['mean'] for k in ks]
        stds  = [aggregate[cname][str(k)]['cs_gap']['std']  for k in ks]
        np_ = breakdown.get(cname, {}).get('n_params', 0)
        label = f'{cname} ({np_/1e6:.2f}M)' if np_ > 1e6 else f'{cname} ({np_/1e3:.0f}K)'
        ax.errorbar(ks, means, yerr=stds, marker='o', capsize=3,
                     color=shades[cname], label=label)
    ax.axhline(0.05, color='black', linestyle=':', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel('k (training domains)')
    ax.set_ylabel('CS_gap (mean ± std)')
    ax.set_title('Sweep C — Paired Scaling Overlay')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, 'fig_sweep_C_overlay')


# ===========================================================================
# 11. HTML report
# ===========================================================================

def _img_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def write_html_report(sweep: str, meta: dict, aggregate: dict,
                       breakdown: dict, cv_block: dict,
                       confirmation: dict, runs: dict, configs: list):
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
             f'<title>lstm2_2_arch_sweep {sweep}</title>',
             """<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: right; }
th { background: #eee; text-align: center; }
td.label { text-align: left; font-weight: 500; }
.confirmed { color: #1a7a1a; font-weight: bold; }
.notmet { color: #b00020; font-weight: bold; }
.ok { color: #1a7a1a; font-weight: bold; }
.fail { color: #b00020; font-weight: bold; }
img { max-width: 100%; border: 1px solid #ddd; margin: 0.5em 0; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
</style></head><body>"""]
    parts.append(f"<h1>lstm2_2_arch_sweep — Sweep {sweep} ({SWEEP_LABEL[sweep]})</h1>")
    parts.append(f"<p>{meta['date']} | Device: {meta['device']} | "
                  f"Mode: {'QUICK' if meta['quick_mode'] else 'FULL'} | "
                  f"k values: {meta['k_values']} | seeds: {len(meta['seeds'])}</p>")

    # Cross-validation
    parts.append('<h2>Cross-validation</h2>')
    if cv_block['status'] == 'N/A':
        parts.append('<p>Not applicable for this sweep.</p>')
    else:
        cls = 'ok' if cv_block['status'] == 'OK' else 'fail'
        parts.append(
            f"<p>Config: <b>{cv_block['config']}</b> at k=1<br>"
            f"CS_gap = {cv_block['cs_gap']:.3f}  "
            f"(reference {cv_block['reference_cs_gap']:.3f} ± {REF_CS_TOL})<br>"
            f"DE = {cv_block['de']:.2f}  "
            f"(reference {cv_block['reference_de']:.2f} ± {REF_DE_TOL})<br>"
            f"Status: <span class='{cls}'>{cv_block['status']}</span></p>")

    # Confirmation table
    parts.append('<h2>Confirmation criteria</h2><table><tr>'
                  '<th>ID</th><th>Criterion</th><th>Status</th>'
                  '<th>Value</th></tr>')
    for cid in sorted(confirmation.keys()):
        c = confirmation[cid]
        cls = 'confirmed' if c['status'] == 'CONFIRMED' else 'notmet'
        parts.append(
            f"<tr><td class='label'>{cid}</td>"
            f"<td class='label'>{c['criterion']}</td>"
            f"<td class='{cls}'>{c['status']}</td>"
            f"<td class='label'>{c.get('value', '')}</td></tr>")
    parts.append('</table>')

    # Breakdown table
    parts.append('<h2>Breakdown points</h2><table><tr>'
                  '<th>Config</th><th>n_params</th><th>k*</th><th>k_half</th>'
                  '<th>CS_gap(k=1)</th><th>DE(k=1)</th><th>H_ratio(k=1)</th>'
                  '</tr>')
    for cfg in configs:
        cname = cfg['name']
        if cname not in breakdown:
            continue
        bp = breakdown[cname]
        a1 = aggregate.get(cname, {}).get('1', {})
        parts.append(
            f"<tr><td class='label'><b>{cname}</b></td>"
            f"<td>{bp['n_params']:,}</td>"
            f"<td>{bp['k_star']}</td>"
            f"<td>{bp['k_half']}</td>"
            f"<td>{a1.get('cs_gap', {}).get('mean', float('nan')):.3f} ± "
            f"{a1.get('cs_gap', {}).get('std', 0):.3f}</td>"
            f"<td>{a1.get('de_mean', {}).get('mean', float('nan')):.2f} ± "
            f"{a1.get('de_mean', {}).get('std', 0):.2f}</td>"
            f"<td>{a1.get('h_ratio', {}).get('mean', float('nan')):.2f} ± "
            f"{a1.get('h_ratio', {}).get('std', 0):.2f}</td></tr>")
    parts.append('</table>')

    # Figures
    fig_titles = [
        ('CS_gap vs k',   f'fig_sweep_{sweep}_cs_gap'),
        ('k_half vs capacity', f'fig_sweep_{sweep}_k_half'),
        ('Three modes at k=1', f'fig_sweep_{sweep}_k1_modes'),
    ]
    if sweep == 'C':
        fig_titles.append(('Paired scaling overlay', 'fig_sweep_C_overlay'))
    for title, base in fig_titles:
        png = os.path.join(RESULTS_DIR, base + '.png')
        if os.path.exists(png):
            b64 = _img_b64(png)
            parts.append(f'<h2>{title}</h2>'
                         f'<img src="data:image/png;base64,{b64}">')

    # Convergence failures
    failures = []
    for cname, kdict in runs.items():
        for k, sdict in kdict.items():
            for seed, r in sdict.items():
                if not r.get('converged', True):
                    failures.append((cname, k, seed, r.get('final_loss')))
    parts.append('<h2>Convergence failures</h2>')
    if failures:
        parts.append('<table><tr><th>Config</th><th>k</th><th>seed</th>'
                      '<th>final_loss</th></tr>')
        for cname, k, seed, loss in failures:
            parts.append(f"<tr><td>{cname}</td><td>{k}</td><td>{seed}</td>"
                         f"<td>{loss:.3f}</td></tr>")
        parts.append('</table>')
    else:
        parts.append('<p>None.</p>')
    parts.append('</body></html>')

    out = os.path.join(RESULTS_DIR, f'{SCRIPT_NAME}_{sweep}_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ===========================================================================
# 12. RESULTS SUMMARY
# ===========================================================================

def print_results_summary(sweep: str, meta: dict, aggregate: dict,
                           breakdown: dict, cv_block: dict,
                           confirmation: dict, runs: dict, configs: list,
                           k_values: list[int]):
    bar = '=' * 63
    print()
    print(bar)
    print(f'RESULTS SUMMARY — lstm2_2_arch_sweep  [SWEEP {sweep}]')
    print(bar)
    print(f"Date:    {meta['date']}")
    print(f"Device:  {meta['device']}")
    print(f"Mode:    {'QUICK' if meta['quick_mode'] else 'FULL'}")
    print(f"Sweep:   {sweep}: {SWEEP_LABEL[sweep]}")
    print(f"k values: {k_values}")
    print(f"Seeds:   {meta['seeds']}")

    print()
    print('-- CROSS-VALIDATION (Sweep C/B only) ----------------------')
    if cv_block['status'] == 'N/A':
        print('N/A')
    else:
        print(f"{cv_block['config']} k=1: CS_gap={cv_block['cs_gap']:.3f}  "
              f"DE={cv_block['de']:.2f}  [{cv_block['status']}]")
        print(f"Reference:           CS_gap={REF_CS_GAP:.3f}±{REF_CS_TOL}  "
              f"DE={REF_DE:.2f}±{REF_DE_TOL}")

    print()
    print('-- BREAKDOWN POINT SUMMARY --------------------------------')
    print(f"{'Config':<8}{'n_params':>12}  {'k*':>6}  {'k_half':>7}  "
          f"{'CS_gap(k=1)':>13}  {'DE(k=1)':>10}  {'H_ratio(k=1)':>12}")
    for cfg in configs:
        cname = cfg['name']
        if cname not in breakdown:
            continue
        bp = breakdown[cname]
        a1 = aggregate.get(cname, {}).get('1', {})
        cs1 = a1.get('cs_gap', {}).get('mean', float('nan'))
        de1 = a1.get('de_mean', {}).get('mean', float('nan'))
        h1  = a1.get('h_ratio', {}).get('mean', float('nan'))
        kstar = '-' if bp['k_star'] is None else str(bp['k_star'])
        khalf = '-' if bp['k_half'] is None else str(bp['k_half'])
        print(f"{cname:<8}{bp['n_params']:>12,}  {kstar:>6}  {khalf:>7}  "
              f"{cs1:>13.4f}  {de1:>10.3f}  {h1:>12.3f}")

    print()
    print('-- CS_GAP TRAJECTORY (mean across seeds) ------------------')
    head = f"{'Config':<8}" + ''.join(f"{f'k={k}':>9}" for k in k_values)
    print(head)
    for cfg in configs:
        cname = cfg['name']
        if cname not in aggregate:
            continue
        row = f"{cname:<8}"
        for k in k_values:
            v = aggregate[cname].get(str(k), {}).get('cs_gap', {}).get('mean')
            row += '   -    ' if v is None else f'{v:>8.4f} '
        print(row)

    print()
    print('-- CONFIRMATION -------------------------------------------')
    for cid in sorted(confirmation.keys()):
        c = confirmation[cid]
        print(f"{cid}  {c['criterion']:<55}  [{c['status']}]")

    print()
    print('-- CONVERGENCE FAILURES -----------------------------------')
    failures = []
    for cname, kdict in runs.items():
        for k, sdict in kdict.items():
            for seed, r in sdict.items():
                if not r.get('converged', True):
                    failures.append((cname, k, seed, r.get('final_loss')))
    if failures:
        for cname, k, seed, loss in failures:
            print(f"  {cname:<6} k={k:<3} seed={seed}  loss={loss:.3f}")
    else:
        print('  None.')

    print()
    print('-- OVERALL ------------------------------------------------')
    all_met = all(c['status'] == 'CONFIRMED' for c in confirmation.values())
    print(f"All confirmations met: {'YES' if all_met else 'NO'}")
    print(f"Output: results/{SCRIPT_NAME}/")
    print(bar)


# ===========================================================================
# 13. Main runner
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', choices=['A', 'B', 'C', 'all'],
                   default='C',
                   help='Which sweep to run (default: C)')
    p.add_argument('--quick', action='store_true',
                   help='Quick mode: reduced k, seeds, and trials')
    return p.parse_args()


def run_sweep(sweep: str, quick: bool, device: torch.device):
    print(f"\n=== SWEEP {sweep}: {SWEEP_LABEL[sweep]} ===\n")
    configs = SWEEPS[sweep]
    k_values = K_VALUES_QUICK if quick else K_VALUES
    seeds    = SEEDS_QUICK    if quick else SEEDS
    n_cs     = N_CS_QUICK     if quick else N_CS
    n_ent    = N_ENTROPY_QUICK if quick else N_ENTROPY

    partial_path = os.path.join(
        RESULTS_DIR,
        f'{SCRIPT_NAME}_{sweep}_partial{"_quick" if quick else ""}.json')

    runs: dict = {}
    if os.path.exists(partial_path):
        try:
            with open(partial_path, 'r', encoding='utf-8') as f:
                runs = json.load(f)
            print(f"[resume] loaded partial runs from {partial_path}")
        except Exception as e:
            print(f"[resume] could not load partial runs: {e!r}")
            runs = {}

    for cfg in configs:
        cname = cfg['name']
        runs.setdefault(cname, {})
        n_params = count_params(cfg['d_model'], cfg['num_layers'])
        epochs = epochs_for(cfg, sweep)
        for k in k_values:
            runs[cname].setdefault(str(k), {})
            for seed in seeds:
                if str(seed) in runs[cname][str(k)]:
                    print(f"[{cname:<5}] k={k:<3} seed={seed:<5} "
                          f"[resume: already done]")
                    continue
                t0 = time.time()
                set_seed(seed)
                data = make_training_data(k=k,
                                            seqs_per_domain=SEQS_PER_DOMAIN,
                                            seq_len=SEQ_LEN)
                model = GeneralistLSTM(d_model=cfg['d_model'],
                                        num_layers=cfg['num_layers'])
                model, final_loss = train_model(
                    model, data, epochs=epochs, lr=LR,
                    batch_size=BATCH_SIZE, device=device, verbose=False)
                if final_loss > 1.0:
                    print(f"  [WARNING] training may not have converged "
                          f"(loss={final_loss:.3f})")
                cs = measure_cs(model, k, n_cs, device)
                de = measure_de(model, k, cfg['num_layers'])
                h  = measure_h(model, k, n_ent, device)
                runs[cname][str(k)][str(seed)] = {
                    'cs': cs, 'de': de, 'h': h,
                    'final_loss': float(final_loss),
                    'converged': bool(final_loss < 1.0),
                }
                elapsed = time.time() - t0
                print(f"[{cname:<5}] k={k:<3} seed={seed:<5} "
                      f"[done {epochs}ep, {elapsed:.0f}s, "
                      f"loss={final_loss:.3f}]  "
                      f"CS_gap={cs['cs_gap']:.3f}  "
                      f"DE={de['mean_de']:.2f}  "
                      f"H_ratio={h['h_ratio']:.2f}x")
                try:
                    save_json(runs, partial_path)
                except Exception as e:
                    print(f"  [warn] could not save partial: {e!r}")

        # Per-config summary line
        agg_so_far = aggregate_runs({cname: runs[cname]})
        bk = compute_breakdown(agg_so_far, [cfg]).get(cname, {})
        kstar = bk.get('k_star')
        khalf = bk.get('k_half')
        cv_note = ''
        if cname in CV_CONFIGS and '1' in agg_so_far[cname]:
            cs1 = agg_so_far[cname]['1']['cs_gap']['mean']
            de1 = agg_so_far[cname]['1']['de_mean']['mean']
            ok = (abs(cs1 - REF_CS_GAP) <= REF_CS_TOL
                  and abs(de1 - REF_DE) <= REF_DE_TOL)
            cv_note = '  [CROSS-VALIDATION OK]' if ok else \
                      '  [CROSS-VALIDATION FAIL]'
        print(f"[{cname:<5}] COMPLETE  k*={kstar}  k_half={khalf}  "
              f"n_params={n_params:,}{cv_note}")

    # Aggregation
    aggregate = aggregate_runs(runs)
    breakdown = compute_breakdown(aggregate, configs)
    cv_block  = cross_validation(aggregate, sweep)
    confirmation = evaluate_confirmation(sweep, aggregate, breakdown,
                                          cv_block, configs)

    meta = {
        'script': SCRIPT_NAME,
        'sweep': sweep,
        'date': datetime.datetime.now().isoformat(),
        'device': str(device),
        'k_values': k_values,
        'seeds': seeds,
        'quick_mode': quick,
        'configs': [
            {'name': c['name'], 'd_model': c['d_model'],
             'num_layers': c['num_layers'],
             'n_params': count_params(c['d_model'], c['num_layers']),
             'epochs': epochs_for(c, sweep)}
            for c in configs
        ],
    }

    output = {
        'meta': meta,
        'cross_validation': cv_block,
        'runs': runs,
        'aggregate': aggregate,
        'breakdown': breakdown,
        'confirmation': confirmation,
    }
    save_json(output, os.path.join(RESULTS_DIR,
                                     f'{SCRIPT_NAME}_{sweep}_results.json'))

    # Figures
    fig_cs_gap(sweep, aggregate, breakdown, configs, k_values)
    fig_k_half(sweep, breakdown, configs)
    fig_k1_modes(sweep, aggregate, breakdown, configs)
    if sweep == 'C':
        fig_overlay_C(aggregate, breakdown, configs)

    write_html_report(sweep, meta, aggregate, breakdown, cv_block,
                       confirmation, runs, configs)
    print_results_summary(sweep, meta, aggregate, breakdown, cv_block,
                           confirmation, runs, configs, k_values)


def main():
    args = parse_args()
    device = get_device()
    sweeps = ['C', 'A', 'B'] if args.sweep == 'all' else [args.sweep]
    for s in sweeps:
        run_sweep(s, args.quick, device)


if __name__ == '__main__':
    main()
