"""
postanalysis_vj_acc_correlation.py
===================================
Post-hoc within-(k, seed) correlation between per-direction Jacobian
variance V_j and per-direction identification accuracy.

Reads lstm2_3_detectability_results.json. Trains nothing, loads no
models, runs no perturbations.
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------

try:
    from scipy.stats import pearsonr as _pearsonr
    from scipy.stats import spearmanr as _spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def pearson_manual(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float('nan')
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    if denom < 1e-12:
        return float('nan')
    return float(xm @ ym / denom)


def spearman_manual(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float('nan')

    def _rank(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        # average ranks for ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.shape, dtype=float)
        np.add.at(sums, inv, ranks)
        means = sums / counts
        return means[inv]

    rx = _rank(x); ry = _rank(y)
    return pearson_manual(rx, ry)


def pearson(x, y) -> float:
    if _HAS_SCIPY:
        try:
            r, _ = _pearsonr(x, y)
            return float(r) if np.isfinite(r) else float('nan')
        except Exception:
            return pearson_manual(x, y)
    return pearson_manual(x, y)


def spearman(x, y) -> float:
    if _HAS_SCIPY:
        try:
            r, _ = _spearmanr(x, y)
            return float(r) if np.isfinite(r) else float('nan')
        except Exception:
            return spearman_manual(x, y)
    return spearman_manual(x, y)


def _mean_std(values):
    values = [float(v) for v in values
              if v is not None and np.isfinite(v)]
    if not values:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    arr = np.asarray(values, dtype=float)
    return {'mean': float(arr.mean()), 'std': float(arr.std()),
            'n': int(arr.size)}


# ---------------------------------------------------------------------------
# Per-direction accuracy from a confusion matrix
# ---------------------------------------------------------------------------

def per_direction_accuracy(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (accuracy, row_sum). accuracy[d] = cm[d,d] / row_sum[d]
    when row_sum[d] > 0, else 0. row_sum[d] = total trials with true=d."""
    cm = np.asarray(cm, dtype=np.int64)
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm).astype(np.float64)
    safe = np.maximum(row_sum, 1)
    acc = diag / safe
    acc[row_sum == 0] = 0.0
    return acc, row_sum


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

VJ_THRESHOLDS  = [0.1, 0.5, 0.7]
PERCENTILES    = [10, 25, 50, 75, 90]
SCATTER_K      = ['1', '5', '10']


def parse_args():
    p = argparse.ArgumentParser(
        description='Per-direction V_j vs accuracy correlation analysis')
    p.add_argument('--json', required=True,
                   help='Path to lstm2_3_detectability_results.json')
    p.add_argument('--out_dir', default=None,
                   help='Output directory (default: same dir as JSON)')
    return p.parse_args()


def main():
    args = parse_args()
    in_path = os.path.abspath(args.json)
    out_dir = (os.path.abspath(args.out_dir) if args.out_dir
               else os.path.dirname(in_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    runs = data.get('runs', {})
    meta_in = data.get('meta', {})
    n_dirs = int(meta_in.get('config', {}).get('n_dirs', 200))
    k_values = sorted(runs.keys(), key=lambda s: int(s))
    seeds = sorted(set(s for k in runs for s in runs[k].keys()),
                   key=lambda s: int(s))

    # ---- per-(k, seed) correlations ---------------------------------
    per_run: dict = {}
    k_with_confusion: list = []
    pooled_pearson_multi   = []
    pooled_spearman_multi  = []

    for k in k_values:
        per_run[k] = {}
        any_confusion_for_k = False
        for seed, rec in runs[k].items():
            vj_list = rec.get('vj_per_direction')
            if not vj_list:
                continue
            vj = np.asarray(vj_list, dtype=float)

            cm_multi = rec.get('confusion_multi')
            cm_single = rec.get('confusion_single')

            entry = {
                'pearson_r_multi':    None,
                'spearman_rho_multi': None,
                'pearson_r_single':   None,
                'spearman_rho_single': None,
                'n_valid_dirs':       0,
            }

            if cm_multi is not None:
                any_confusion_for_k = True
                cm_m = np.asarray(cm_multi, dtype=np.int64)
                acc_m, row_m = per_direction_accuracy(cm_m)
                valid = row_m > 0
                n_valid = int(valid.sum())
                entry['n_valid_dirs'] = n_valid
                if n_valid >= 2:
                    pr = pearson(vj[valid], acc_m[valid])
                    sr = spearman(vj[valid], acc_m[valid])
                    entry['pearson_r_multi']    = pr
                    entry['spearman_rho_multi'] = sr
                    if np.isfinite(pr):
                        pooled_pearson_multi.append(pr)
                    if np.isfinite(sr):
                        pooled_spearman_multi.append(sr)

            if cm_single is not None:
                cm_s = np.asarray(cm_single, dtype=np.int64)
                acc_s, row_s = per_direction_accuracy(cm_s)
                valid = row_s > 0
                if int(valid.sum()) >= 2:
                    entry['pearson_r_single']    = pearson(vj[valid], acc_s[valid])
                    entry['spearman_rho_single'] = spearman(vj[valid], acc_s[valid])

            per_run[k][seed] = entry
        if any_confusion_for_k:
            k_with_confusion.append(k)

    # ---- aggregate by k ---------------------------------------------
    aggregate_by_k: dict = {}
    for k in k_values:
        prs_m  = [per_run[k][s]['pearson_r_multi']    for s in per_run[k]]
        srs_m  = [per_run[k][s]['spearman_rho_multi'] for s in per_run[k]]
        prs_s  = [per_run[k][s]['pearson_r_single']   for s in per_run[k]]
        srs_s  = [per_run[k][s]['spearman_rho_single'] for s in per_run[k]]
        ndirs  = [per_run[k][s]['n_valid_dirs']       for s in per_run[k]]

        any_multi = any(v is not None for v in prs_m)
        any_single = any(v is not None for v in prs_s)
        agg_entry = {
            'pearson_r_multi':    _mean_std(prs_m) if any_multi else None,
            'spearman_rho_multi': _mean_std(srs_m) if any_multi else None,
            'pearson_r_single':   _mean_std(prs_s) if any_single else None,
            'spearman_rho_single': _mean_std(srs_s) if any_single else None,
            'n_seeds':     int(sum(1 for v in prs_m if v is not None)),
            'n_dirs_mean': float(np.mean(ndirs)) if ndirs else 0.0,
        }
        aggregate_by_k[k] = agg_entry

    grand = {
        'pearson_r_multi_all_k':    _mean_std(pooled_pearson_multi),
        'spearman_rho_multi_all_k': _mean_std(pooled_spearman_multi),
    }

    # ---- V_j threshold analysis & percentiles -----------------------
    high_frac_by_k: dict = {}
    percentiles_by_k: dict = {}
    for k in k_values:
        all_vjs = []
        for seed in runs[k]:
            vj_list = runs[k][seed].get('vj_per_direction')
            if vj_list:
                all_vjs.append(np.asarray(vj_list, dtype=float))
        if not all_vjs:
            continue
        # per-seed high-fraction averaged
        per_seed_high = []
        for vj_arr in all_vjs:
            per_seed_high.append([float(np.mean(vj_arr > t))
                                   for t in VJ_THRESHOLDS])
        high_frac_by_k[k] = [float(np.mean([row[i]
                                              for row in per_seed_high]))
                              for i in range(len(VJ_THRESHOLDS))]

        pooled = np.concatenate(all_vjs)
        pcts = np.percentile(pooled, PERCENTILES)
        percentiles_by_k[k] = {f'p{p}': float(v)
                                for p, v in zip(PERCENTILES, pcts)}

    # ---- Scatter data -----------------------------------------------
    scatter_data: dict = {}
    for k in SCATTER_K:
        if k not in runs:
            continue
        vj_pool = []
        acc_pool = []
        seed_pool = []
        for seed in seeds:
            rec = runs[k].get(seed)
            if not rec:
                continue
            cm_multi = rec.get('confusion_multi')
            vj_list = rec.get('vj_per_direction')
            if cm_multi is None or not vj_list:
                continue
            cm_m = np.asarray(cm_multi, dtype=np.int64)
            acc_m, row_m = per_direction_accuracy(cm_m)
            vj = np.asarray(vj_list, dtype=float)
            valid = row_m > 0
            vj_pool.extend(vj[valid].tolist())
            acc_pool.extend(acc_m[valid].tolist())
            seed_pool.extend([seed] * int(valid.sum()))
        scatter_data[k] = {
            'vj': vj_pool, 'acc': acc_pool, 'seed': seed_pool,
        }

    # ---- Figure -----------------------------------------------------
    fig_path = os.path.join(out_dir, 'postanalysis_vj_acc_scatter.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    seed_to_color = {s: plt.cm.tab10(i % 10)
                      for i, s in enumerate(seeds)}
    for ax, k in zip(axes, SCATTER_K):
        sd = scatter_data.get(k)
        if not sd or not sd['vj']:
            ax.set_title(f'k={k}  (no confusion data)')
            ax.set_xlabel('Jacobian Variance $V_j$')
            ax.set_ylabel('Per-direction Accuracy (multi-layer)')
            continue
        vj_arr  = np.asarray(sd['vj'])
        acc_arr = np.asarray(sd['acc'])
        seed_arr = np.asarray(sd['seed'])
        for s in seeds:
            mask = seed_arr == s
            if not mask.any():
                continue
            ax.scatter(vj_arr[mask], acc_arr[mask],
                       s=12, alpha=0.55, color=seed_to_color[s], label=s,
                       edgecolors='none')
        # regression line
        if vj_arr.size >= 2 and np.std(vj_arr) > 1e-12:
            slope, intercept = np.polyfit(vj_arr, acc_arr, 1)
            xs = np.linspace(vj_arr.min(), vj_arr.max(), 50)
            ax.plot(xs, slope * xs + intercept, 'k--', lw=1.5)
        r = pearson(vj_arr, acc_arr)
        ax.set_title(f'k={k},  Pearson r = {r:.3f}')
        ax.set_xlabel('Jacobian Variance $V_j$')
        ax.set_ylabel('Per-direction Accuracy (multi-layer)')
        ax.grid(alpha=0.3)

    handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                          color=seed_to_color[s], label=f'seed {s}')
               for s in seeds]
    fig.legend(handles=handles, loc='lower center', ncol=len(seeds),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- JSON output ------------------------------------------------
    out_json = {
        'meta': {
            'script': 'postanalysis_vj_acc_correlation',
            'input_json': in_path,
            'date': datetime.datetime.now().isoformat(),
            'k_values_with_confusion': k_with_confusion,
            'seeds': seeds,
            'n_dirs': n_dirs,
            'scipy_available': _HAS_SCIPY,
        },
        'per_run': per_run,
        'aggregate_by_k': aggregate_by_k,
        'grand_aggregate': grand,
        'vj_threshold_analysis': {
            'thresholds': VJ_THRESHOLDS,
            'high_frac_by_k': high_frac_by_k,
            'percentiles_by_k': percentiles_by_k,
        },
        'scatter_data': {
            k: {'vj': scatter_data[k]['vj'], 'acc': scatter_data[k]['acc']}
            for k in scatter_data
        },
    }
    out_json_path = os.path.join(out_dir,
                                  'postanalysis_vj_acc_correlation.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2)

    # ---- Pretty RESULTS SUMMARY -------------------------------------
    def fmt_ms(d):
        if d is None:
            return '   n/a       '
        if d['n'] == 0:
            return '   n/a       '
        return f"{d['mean']:+.3f} ± {d['std']:.3f}"

    print()
    print('=' * 55)
    print('RESULTS SUMMARY — postanalysis_vj_acc_correlation')
    print('=' * 55)
    print(f"Input:   {os.path.basename(in_path)}")
    print(f"k values with confusion matrices: "
          f"{[int(k) for k in k_with_confusion]}")
    print(f"Seeds:   {len(seeds)}   N_DIRS: {n_dirs}")
    if not _HAS_SCIPY:
        print("(scipy not found — using manual Pearson/Spearman)")

    # Per-direction correlation, multi-layer
    print()
    print('-- PER-DIRECTION CORRELATION (within k, multi-layer) ----')
    print(f"{'k':<5}{'Pearson r':<18}{'Spearman rho':<18}{'n_seeds':<8}")
    for k in k_with_confusion:
        a = aggregate_by_k.get(k, {})
        pr = a.get('pearson_r_multi')
        sr = a.get('spearman_rho_multi')
        print(f"{k:<5}{fmt_ms(pr):<18}{fmt_ms(sr):<18}"
              f"{a.get('n_seeds', 0):<8}")
    print()
    g_pr = grand['pearson_r_multi_all_k']
    g_sr = grand['spearman_rho_multi_all_k']
    print(f"Grand (all k pooled):")
    print(f"  Pearson r  = {g_pr['mean']:+.3f} ± {g_pr['std']:.3f}  "
          f"(n={g_pr['n']} runs)")
    print(f"  Spearman ρ = {g_sr['mean']:+.3f} ± {g_sr['std']:.3f}")

    # Per-direction correlation, single-layer
    print()
    print('-- PER-DIRECTION CORRELATION (within k, single-layer) ---')
    print(f"{'k':<5}{'Pearson r':<18}{'Spearman rho':<18}")
    for k in k_with_confusion:
        a = aggregate_by_k.get(k, {})
        pr = a.get('pearson_r_single')
        sr = a.get('spearman_rho_single')
        print(f"{k:<5}{fmt_ms(pr):<18}{fmt_ms(sr):<18}")

    # V_j threshold analysis
    print()
    print('-- V_j THRESHOLD ANALYSIS ------------------------------')
    print(f"         "
          f"{'thresh=0.10':<14}{'thresh=0.50':<14}{'thresh=0.70':<14}")
    for k in k_values:
        if k not in high_frac_by_k:
            continue
        h = high_frac_by_k[k]
        print(f"k={k:<6}{h[0]:<14.2f}{h[1]:<14.2f}{h[2]:<14.2f}")

    # V_j percentiles
    print()
    print('-- V_j PERCENTILES -------------------------------------')
    print(f"{'k':<5}{'p10':<10}{'p25':<10}{'p50':<10}{'p75':<10}{'p90':<10}")
    for k in k_values:
        if k not in percentiles_by_k:
            continue
        p = percentiles_by_k[k]
        print(f"{k:<5}{p['p10']:<10.4f}{p['p25']:<10.4f}"
              f"{p['p50']:<10.4f}{p['p75']:<10.4f}{p['p90']:<10.4f}")

    # I6 reassessment block
    print()
    print('-- I6 REASSESSMENT -------------------------------------')
    pooled_orig = data.get('pearson_r_vj_acc')
    if isinstance(pooled_orig, (int, float)):
        orig_str = f'{pooled_orig:+.4f}'
    else:
        orig_str = '(unavailable in JSON)'
    print(f"Original (pooled across k):    r = {orig_str}  [confounded]")
    if g_pr['n'] > 0:
        print(f"Within-k Pearson r (multi):    "
              f"{g_pr['mean']:+.3f} ± {g_pr['std']:.3f}")
        print(f"Within-k Spearman rho (multi): "
              f"{g_sr['mean']:+.3f} ± {g_sr['std']:.3f}")
        if g_pr['mean'] < -0.05:
            direction = 'NEGATIVE'
            jup = 'SUPPORTED'
        elif g_pr['mean'] > 0.05:
            direction = 'POSITIVE'
            jup = 'NOT SUPPORTED'
        else:
            direction = 'NEAR-ZERO'
            jup = 'NOT SUPPORTED'
        print(f"Direction of within-k effect:  {direction}")
        print(f"JUP prediction (r < 0):        {jup}")
    else:
        print("Within-k Pearson r (multi):    (no confusion matrices found)")

    print()
    print('-- OUTPUT ----------------------------------------------')
    print(f"  {os.path.relpath(out_json_path)}")
    print(f"  {os.path.relpath(fig_path)}")
    print('=' * 55)


if __name__ == '__main__':
    sys.exit(main())
