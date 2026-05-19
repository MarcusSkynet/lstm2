"""
lstm2_6_singlecell_vs_multicell.py
====================================
Test of Corollary 1: N specialist cells (k=1 each) vs one generalist (k=N).

Per (N, seed):
  - Train (or load) N specialist cells, one per domain in {0..N-1}.
  - Load the corresponding k=N generalist from lstm2_3 saved models.
  - Evaluate both architectures with oracle routing using identical
    inputs and report CS, DE, H, identification, practical correction
    and crossing violation metrics side by side.
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
    VOCAB_SIZE, D_MODEL, NUM_LAYERS, SEQ_LEN,
    SEQS_PER_DOMAIN, EPOCHS, LR, BATCH_SIZE,
    SEEDS_BASE, GeneralistLSTM,
    make_domain_sequences, make_training_data,
    make_probe_test_split, train_model,
    get_weight_matrix, compute_svd_basis,
    compute_dimensional_excess,
    perturb_weight, restore_weight,
    measure_syndrome, build_multilayer_syndrome,
    set_seed, get_device, make_results_dir, save_json,
)


# ===========================================================================
# Config
# ===========================================================================

SCRIPT_NAME = 'lstm2_6_singlecell_vs_multicell'
RESULTS_DIR = make_results_dir(SCRIPT_NAME)
SPECIALIST_MODELS_DIR = os.path.join(RESULTS_DIR, 'specialist_models')
os.makedirs(SPECIALIST_MODELS_DIR, exist_ok=True)
GENERALIST_MODELS_DIR = 'results/lstm2_3_detectability/models'

N_VALUES = [5, 10]

# CS / H / generation
N_CS_PER_DOMAIN  = 50
PREFIX_LEN       = 8
GEN_LEN          = 12
ERR_POS          = 5
EPS_ERR          = 0.1

# Identification / correction
N_PROBE_DICT     = 50
N_DIRS           = 200
EPS_DICT         = 3.0
N_TEST_ID        = 100
N_TEST_CORR      = 100
EPS_MIN, EPS_MAX = 1.0, 5.0
N_WRONG_DIRS     = 3

SEEDS = SEEDS_BASE
LSTM2_1_K1_CS_GAP = 0.273    # baseline for fig_singleton_bound

# Quick
N_CS_QUICK         = 15
N_TEST_ID_QUICK    = 30
N_TEST_CORR_QUICK  = 30
N_DIRS_QUICK       = 30
SEEDS_QUICK        = SEEDS_BASE[:2]


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, choices=[5, 10], default=5)
    p.add_argument('--quick', action='store_true')
    return p.parse_args()


# ===========================================================================
# Model loading / training
# ===========================================================================

def load_or_train_specialist(d: int, seed: int,
                              models_dir: str,
                              device: torch.device):
    """Return (model_in_eval, final_loss). Cache on disk by (d, seed).
    For d == 0, prefer the lstm2_3 k=1 checkpoint if available."""
    if d == 0:
        lstm2_3_path = os.path.join(GENERALIST_MODELS_DIR,
                                      f'k1_seed{seed}.pt')
        if os.path.exists(lstm2_3_path):
            ckpt  = torch.load(lstm2_3_path, map_location='cpu',
                                weights_only=False)
            model = GeneralistLSTM()
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device).eval()
            return model, float(ckpt.get('final_loss', 0.0))

    path = os.path.join(models_dir, f'specialist_d{d}_seed{seed}.pt')
    if os.path.exists(path):
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        model = GeneralistLSTM()
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        return model, float(ckpt.get('final_loss', 0.0))

    set_seed(seed)
    data = make_domain_sequences(d, SEQS_PER_DOMAIN, seq_len=SEQ_LEN)
    model = GeneralistLSTM()
    model, final_loss = train_model(
        model, data, epochs=EPOCHS, lr=LR,
        batch_size=BATCH_SIZE, device=device, verbose=False)
    torch.save({
        'model_state_dict': model.state_dict(),
        'domain': int(d), 'seed': int(seed),
        'final_loss': float(final_loss),
        'config': {'d_model': D_MODEL, 'num_layers': NUM_LAYERS,
                   'vocab_size': VOCAB_SIZE, 'seq_len': SEQ_LEN,
                   'seqs_per_domain': SEQS_PER_DOMAIN, 'epochs': EPOCHS},
    }, path)
    model.eval()
    return model, float(final_loss)


def load_generalist(N: int, seed: int, device: torch.device):
    path = os.path.join(GENERALIST_MODELS_DIR, f'k{N}_seed{seed}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Generalist model not found: {path}\n'
            f'Run lstm2_3_detectability.py first.')
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model = GeneralistLSTM()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, float(ckpt.get('final_loss', 0.0))


# ===========================================================================
# Per-domain measurement primitives
# ===========================================================================

def _forward(model, x, with_layers=False):
    with torch.no_grad():
        if with_layers:
            logits, layers = model(x, return_all_layers=True)
            return logits.detach(), [lo.detach() for lo in layers]
        return model(x).detach()


def cs_for_domain(model, domain_idx: int, n_trials: int,
                   device: torch.device) -> float:
    seqs = make_domain_sequences(domain_idx, n_trials,
                                  seq_len=PREFIX_LEN + 2).to(device)
    cs_vals = []
    for j in range(n_trials):
        prefix_clean = seqs[j:j + 1, :PREFIX_LEN]
        prefix_corrupt = prefix_clean.clone()
        tok = int(prefix_corrupt[0, ERR_POS].item())
        prefix_corrupt[0, ERR_POS] = max(2, (tok + 11) % VOCAB_SIZE)
        gen_clean = model.generate_greedy(prefix_clean, GEN_LEN)
        gen_corrupt = model.generate_greedy(prefix_corrupt, GEN_LEN)
        gc = gen_clean[0, PREFIX_LEN:].cpu().numpy()
        gp = gen_corrupt[0, PREFIX_LEN:].cpu().numpy()
        cs_vals.append(float(np.mean(gc != gp)))
    return float(np.mean(cs_vals))


def de_mean(model: GeneralistLSTM) -> float:
    targets = ['output'] + list(range(NUM_LAYERS))
    de_vals = []
    for t in targets:
        W = get_weight_matrix(model, t)
        info = compute_dimensional_excess(W)
        de_vals.append(info['de'])
    return float(np.mean(de_vals))


def entropy_for_domain(model, domain_idx: int, n_seqs: int,
                        device: torch.device) -> float:
    seqs = make_domain_sequences(domain_idx, n_seqs,
                                  seq_len=PREFIX_LEN + 2).to(device)
    h_vals = []
    model.eval()
    with torch.no_grad():
        for j in range(n_seqs):
            seq = seqs[j:j + 1, :PREFIX_LEN].clone()
            steps = []
            for _ in range(GEN_LEN):
                logits = model(seq)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                H_t = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                steps.append(float(H_t.item()))
                next_tok = logits.argmax(-1, keepdim=True)
                seq = torch.cat([seq, next_tok], dim=1)
            h_vals.append(float(np.mean(steps)))
    return float(np.mean(h_vals))


def build_dict_for_domain(model, domain_idx: int, n_dirs: int,
                            device: torch.device) -> dict:
    """Build dict_single, dict_multi for this model using probe sequences
    drawn from the given domain. Reuses optimised batched approach."""
    probe_seqs = make_domain_sequences(
        domain_idx, N_PROBE_DICT, seq_len=SEQ_LEN).to(device)

    W = model.lstm_layers[0].weight_ih_l0
    W_cpu = get_weight_matrix(model, 0)
    _, Vh = compute_svd_basis(W_cpu)
    Vh = Vh.to(device)
    n_dirs_eff = min(n_dirs, Vh.shape[0])

    clean_logits_g, clean_layers_g = _forward(model, probe_seqs,
                                                with_layers=True)
    d_multi = NUM_LAYERS * D_MODEL + VOCAB_SIZE
    dict_single = np.zeros((n_dirs_eff, VOCAB_SIZE), dtype=np.float32)
    dict_multi  = np.zeros((n_dirs_eff, d_multi),    dtype=np.float32)

    for d in range(n_dirs_eff):
        original = perturb_weight(W, Vh[d], EPS_DICT)
        try:
            pert_logits_g, pert_layers_g = _forward(model, probe_seqs,
                                                       with_layers=True)
        finally:
            restore_weight(W, original)
        delta_l = (pert_logits_g - clean_logits_g).mean(
            dim=(0, 1)).cpu().numpy().astype(np.float32)
        n_l = float(np.linalg.norm(delta_l))
        logit_syn = delta_l / n_l if n_l > 1e-12 else delta_l
        dict_single[d] = logit_syn
        layer_syns = []
        for L in range(NUM_LAYERS):
            dL = (pert_layers_g[L] - clean_layers_g[L]).mean(
                dim=(0, 1)).cpu().numpy().astype(np.float32)
            n_L = float(np.linalg.norm(dL))
            layer_syns.append(dL / n_L if n_L > 1e-12 else dL)
        dict_multi[d] = build_multilayer_syndrome(
            logit_syn, layer_syns, injection_layer=0
        ).astype(np.float32)

    norms_s = np.linalg.norm(dict_single, axis=1, keepdims=True)
    norms_s = np.where(norms_s < 1e-12, 1.0, norms_s)
    dict_single /= norms_s
    norms_m = np.linalg.norm(dict_multi, axis=1, keepdims=True)
    norms_m = np.where(norms_m < 1e-12, 1.0, norms_m)
    dict_multi /= norms_m

    return {
        'Vh': Vh, 'dict_single': dict_single, 'dict_multi': dict_multi,
        'probe_seqs': probe_seqs, 'clean_probe_logits_g': clean_logits_g,
        'clean_probe_layers_g': clean_layers_g,
    }


def cosine_per_token(a_g: torch.Tensor, b_g: torch.Tensor) -> float:
    a_flat = a_g.reshape(-1, a_g.shape[-1])
    b_flat = b_g.reshape(-1, b_g.shape[-1])
    a_norm = a_flat / (a_flat.norm(dim=-1, keepdim=True) + 1e-12)
    b_norm = b_flat / (b_flat.norm(dim=-1, keepdim=True) + 1e-12)
    return float((a_norm * b_norm).sum(dim=-1).mean().item())


def mse_per_token(a_g: torch.Tensor, b_g: torch.Tensor) -> float:
    diff = (a_g - b_g).reshape(-1, a_g.shape[-1])
    return float((diff ** 2).sum(dim=-1).mean().item())


def run_id_test(model, domain_idx: int, dict_single, dict_multi, Vh,
                  n_dirs: int, n_test: int, seed: int,
                  device: torch.device) -> dict:
    test_seqs = make_domain_sequences(domain_idx, n_test,
                                        seq_len=SEQ_LEN).to(device)
    W = model.lstm_layers[0].weight_ih_l0
    clean_logits_g, clean_layers_g = _forward(model, test_seqs,
                                                with_layers=True)
    rng = np.random.default_rng(seed + 10 * domain_idx + 1)
    correct_single = 0
    correct_multi  = 0
    for trial in range(n_test):
        true_dir = int(rng.integers(0, n_dirs))
        eps_test = float(rng.uniform(EPS_MIN, EPS_MAX))
        original = perturb_weight(W, Vh[true_dir], eps_test)
        try:
            pert_logits_g, pert_layers_g = _forward(model, test_seqs,
                                                       with_layers=True)
        finally:
            restore_weight(W, original)
        delta_l = (pert_logits_g - clean_logits_g).mean(
            dim=(0, 1)).cpu().numpy().astype(np.float32)
        n_l = float(np.linalg.norm(delta_l))
        logit_syn = delta_l / n_l if n_l > 1e-12 else delta_l
        layer_syns = []
        for L in range(NUM_LAYERS):
            dL = (pert_layers_g[L] - clean_layers_g[L]).mean(
                dim=(0, 1)).cpu().numpy().astype(np.float32)
            n_L = float(np.linalg.norm(dL))
            layer_syns.append(dL / n_L if n_L > 1e-12 else dL)
        ts = logit_syn
        tm = build_multilayer_syndrome(logit_syn, layer_syns,
                                         injection_layer=0).astype(np.float32)
        tn = float(np.linalg.norm(tm))
        if tn > 1e-12:
            tm = tm / tn
        if int(np.argmax(dict_single @ ts)) == true_dir:
            correct_single += 1
        if int(np.argmax(dict_multi @ tm)) == true_dir:
            correct_multi += 1
    return {
        'acc_single': correct_single / max(n_test, 1),
        'acc_multi':  correct_multi  / max(n_test, 1),
    }


def run_practical(model, domain_idx: int, dict_multi, Vh,
                    n_dirs: int, n_test: int, seed: int,
                    device: torch.device) -> dict:
    test_seqs = make_domain_sequences(domain_idx, n_test,
                                        seq_len=SEQ_LEN).to(device)
    W = model.lstm_layers[0].weight_ih_l0
    clean_test_g = _forward(model, test_seqs)
    # Probe set for syndrome match — same as dict probe set: from domain_idx
    probe_seqs = make_domain_sequences(domain_idx, N_PROBE_DICT,
                                         seq_len=SEQ_LEN).to(device)
    clean_probe_logits_g, clean_probe_layers_g = _forward(
        model, probe_seqs, with_layers=True)

    rng = np.random.default_rng(seed + 100 * domain_idx + 5)
    n_dirs_eff = min(n_dirs, dict_multi.shape[0])
    correct_ratios = []
    wrong_ratios = []
    crossings = 0
    correct_id = 0
    n_done = 0
    for d in range(min(n_test, n_dirs_eff)):
        eps_inject = float(rng.uniform(EPS_MIN, EPS_MAX))
        original = perturb_weight(W, Vh[d], eps_inject)
        try:
            pert_test_g = _forward(model, test_seqs)
            error_perturbed = mse_per_token(pert_test_g, clean_test_g)
            pert_probe_logits_g, pert_probe_layers_g = _forward(
                model, probe_seqs, with_layers=True)
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
            if tn > 1e-12:
                test_multi = test_multi / tn
            d_hat = int(np.argmax(dict_multi @ test_multi))
            id_correct = bool(d_hat == d)
        finally:
            restore_weight(W, original)

        # Apply correction: re-inject and apply -EPS_DICT × Vh[d_hat]
        original = perturb_weight(W, Vh[d], eps_inject)
        try:
            original_corr = perturb_weight(W, Vh[d_hat], -EPS_DICT)
            try:
                corrected_g = _forward(model, test_seqs)
                error_corrected = mse_per_token(corrected_g, clean_test_g)
            finally:
                restore_weight(W, original_corr)
        finally:
            restore_weight(W, original)

        ratio = (error_corrected / error_perturbed
                  if error_perturbed > 1e-12 else float('inf'))
        if np.isfinite(ratio):
            if id_correct:
                correct_ratios.append(ratio); correct_id += 1
            else:
                wrong_ratios.append(ratio)
            if ratio > 1.0:
                crossings += 1
        n_done += 1

    all_ratios = np.asarray(correct_ratios + wrong_ratios)
    return {
        'n_trials':           int(n_done),
        'id_acc':             float(correct_id / n_done) if n_done else 0.0,
        'mean_ratio':         (float(np.mean(np.minimum(all_ratios, 1e9)))
                                if all_ratios.size else 0.0),
        'mean_ratio_correct': (float(np.mean(correct_ratios))
                                if correct_ratios else float('nan')),
        'mean_ratio_wrong':   (float(np.mean(wrong_ratios))
                                if wrong_ratios else float('nan')),
        'crossing_rate':      (float(crossings / n_done) if n_done else 0.0),
    }


def run_violation(model, domain_idx: int, Vh, n_test: int,
                    n_wrong: int, seed: int,
                    device: torch.device) -> dict:
    test_seqs = make_domain_sequences(domain_idx, n_test,
                                        seq_len=SEQ_LEN).to(device)
    W = model.lstm_layers[0].weight_ih_l0
    n_dirs_total = int(Vh.shape[0])
    clean_g = _forward(model, test_seqs)
    rng = np.random.default_rng(seed + 1000 * domain_idx + 7)
    n_dirs_eff = min(n_test, n_dirs_total)
    violations = 0
    n_trials = 0
    for d in range(n_dirs_eff):
        eps_inject = float(rng.uniform(EPS_MIN, EPS_MAX))
        original = perturb_weight(W, Vh[d], eps_inject)
        try:
            pert_g = _forward(model, test_seqs)
            error_before = mse_per_token(pert_g, clean_g)
            wrong_pool = [j for j in range(n_dirs_total) if j != d]
            n_w = min(n_wrong, len(wrong_pool))
            wrong_indices = rng.choice(wrong_pool, size=n_w, replace=False)
            for j in wrong_indices:
                eps_wrong = float(rng.uniform(EPS_MIN, EPS_MAX))
                original_j = perturb_weight(W, Vh[int(j)], -eps_wrong)
                try:
                    wrong_g = _forward(model, test_seqs)
                    error_after = mse_per_token(wrong_g, clean_g)
                finally:
                    restore_weight(W, original_j)
                if error_after - error_before <= 0:
                    violations += 1
                n_trials += 1
        finally:
            restore_weight(W, original)
    return {
        'n_trials':       int(n_trials),
        'violation_rate': float(violations / n_trials) if n_trials else 0.0,
    }


# ===========================================================================
# Per-(N, seed) full evaluation
# ===========================================================================

def evaluate_seed(N: int, seed: int, device: torch.device,
                   n_cs: int, n_test_id: int, n_test_corr: int,
                   n_dirs: int, log) -> dict:
    domains_train = list(range(N))
    domain_unknown = N

    # Train/load all specialists
    specialists = {}
    specialist_losses = {}
    log(f"  [N={N}] training/loading specialists for seed={seed}...")
    for d in domains_train:
        t0 = time.time()
        m, loss = load_or_train_specialist(d, seed,
                                            SPECIALIST_MODELS_DIR, device)
        specialists[d] = m
        specialist_losses[d] = loss
        log(f"    d={d:<3}seed={seed:<6}loss={loss:.4f}  "
            f"({time.time()-t0:.0f}s)")

    # Load generalist
    log(f"  [N={N}] loading k={N} generalist seed={seed}...")
    generalist, gen_loss = load_generalist(N, seed, device)
    log(f"    generalist loss={gen_loss:.4f}")

    # Per-domain metrics
    multi_per_domain = {}
    gen_per_domain   = {}

    # Cache generalist syndrome state per domain for ID + correction
    # (built per domain inside the loop because probe set is domain-specific)
    log(f"  [N={N}] evaluating per-domain (multicell + generalist)...")
    for d in domains_train:
        t_d = time.time()
        m_spec = specialists[d]
        # CS
        cs_known_s   = cs_for_domain(m_spec,    d,             n_cs, device)
        cs_unknown_s = cs_for_domain(m_spec,    domain_unknown, n_cs, device)
        cs_known_g   = cs_for_domain(generalist, d,             n_cs, device)
        cs_unknown_g = cs_for_domain(generalist, domain_unknown, n_cs, device)

        # H
        h_known_s    = entropy_for_domain(m_spec,    d, max(n_cs // 2, 8), device)
        h_unknown_s  = entropy_for_domain(m_spec,    domain_unknown,
                                            max(n_cs // 2, 8), device)
        h_known_g    = entropy_for_domain(generalist, d, max(n_cs // 2, 8), device)
        h_unknown_g  = entropy_for_domain(generalist, domain_unknown,
                                            max(n_cs // 2, 8), device)

        # Build dictionaries per-(model, domain)
        spec_dict = build_dict_for_domain(m_spec, d, n_dirs, device)
        gen_dict  = build_dict_for_domain(generalist, d, n_dirs, device)

        # Identification
        id_s = run_id_test(m_spec, d,
                            spec_dict['dict_single'], spec_dict['dict_multi'],
                            spec_dict['Vh'], n_dirs, n_test_id, seed, device)
        id_g = run_id_test(generalist, d,
                            gen_dict['dict_single'], gen_dict['dict_multi'],
                            gen_dict['Vh'], n_dirs, n_test_id, seed, device)

        # Practical correction
        prac_s = run_practical(m_spec, d, spec_dict['dict_multi'],
                                 spec_dict['Vh'], n_dirs, n_test_corr,
                                 seed, device)
        prac_g = run_practical(generalist, d, gen_dict['dict_multi'],
                                 gen_dict['Vh'], n_dirs, n_test_corr,
                                 seed, device)

        # Violation
        viol_s = run_violation(m_spec, d, spec_dict['Vh'], n_test_corr,
                                 N_WRONG_DIRS, seed, device)
        viol_g = run_violation(generalist, d, gen_dict['Vh'], n_test_corr,
                                 N_WRONG_DIRS, seed, device)

        multi_per_domain[d] = {
            'cs_known': cs_known_s, 'cs_unknown': cs_unknown_s,
            'cs_gap': cs_unknown_s - cs_known_s,
            'h_known': h_known_s, 'h_unknown': h_unknown_s,
            'h_ratio': h_unknown_s / max(h_known_s, 1e-12),
            'id_acc_single': id_s['acc_single'],
            'id_acc_multi':  id_s['acc_multi'],
            'corr_ratio':            prac_s['mean_ratio'],
            'corr_ratio_correct':    prac_s['mean_ratio_correct'],
            'corr_ratio_wrong':      prac_s['mean_ratio_wrong'],
            'crossing_rate':         prac_s['crossing_rate'],
            'violation_rate':        viol_s['violation_rate'],
        }
        gen_per_domain[d] = {
            'cs_known': cs_known_g, 'cs_unknown': cs_unknown_g,
            'cs_gap': cs_unknown_g - cs_known_g,
            'h_known': h_known_g, 'h_unknown': h_unknown_g,
            'h_ratio': h_unknown_g / max(h_known_g, 1e-12),
            'id_acc_single': id_g['acc_single'],
            'id_acc_multi':  id_g['acc_multi'],
            'corr_ratio':            prac_g['mean_ratio'],
            'corr_ratio_correct':    prac_g['mean_ratio_correct'],
            'corr_ratio_wrong':      prac_g['mean_ratio_wrong'],
            'crossing_rate':         prac_g['crossing_rate'],
            'violation_rate':        viol_g['violation_rate'],
        }
        log(f"    domain={d}: spec CS_gap={multi_per_domain[d]['cs_gap']:.3f}  "
            f"gen CS_gap={gen_per_domain[d]['cs_gap']:.3f}  "
            f"({time.time()-t_d:.0f}s)")

    # DE — per specialist (domain index doesn't change DE), per generalist
    de_specs = [de_mean(specialists[d]) for d in domains_train]
    de_gen   = de_mean(generalist)

    def _avg(records, key):
        vals = [r[key] for r in records.values()
                if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float('nan')

    multicell = {
        'cs_known':            _avg(multi_per_domain, 'cs_known'),
        'cs_unknown':          _avg(multi_per_domain, 'cs_unknown'),
        'cs_gap':              _avg(multi_per_domain, 'cs_gap'),
        'de_mean':             float(np.mean(de_specs)),
        'h_known':             _avg(multi_per_domain, 'h_known'),
        'h_unknown':           _avg(multi_per_domain, 'h_unknown'),
        'h_ratio':             _avg(multi_per_domain, 'h_ratio'),
        'id_acc_single':       _avg(multi_per_domain, 'id_acc_single'),
        'id_acc_multi':        _avg(multi_per_domain, 'id_acc_multi'),
        'corr_ratio':          _avg(multi_per_domain, 'corr_ratio'),
        'corr_ratio_correct':  _avg(multi_per_domain, 'corr_ratio_correct'),
        'corr_ratio_wrong':    _avg(multi_per_domain, 'corr_ratio_wrong'),
        'crossing_rate':       _avg(multi_per_domain, 'crossing_rate'),
        'violation_rate':      _avg(multi_per_domain, 'violation_rate'),
        'specialist_losses':   {str(k): float(v)
                                  for k, v in specialist_losses.items()},
        'per_domain':          {str(k): v
                                  for k, v in multi_per_domain.items()},
    }
    gen = {
        'final_loss':          gen_loss,
        'cs_known':            _avg(gen_per_domain, 'cs_known'),
        'cs_unknown':          _avg(gen_per_domain, 'cs_unknown'),
        'cs_gap':              _avg(gen_per_domain, 'cs_gap'),
        'de_mean':             de_gen,
        'h_known':             _avg(gen_per_domain, 'h_known'),
        'h_unknown':           _avg(gen_per_domain, 'h_unknown'),
        'h_ratio':             _avg(gen_per_domain, 'h_ratio'),
        'id_acc_single':       _avg(gen_per_domain, 'id_acc_single'),
        'id_acc_multi':        _avg(gen_per_domain, 'id_acc_multi'),
        'corr_ratio':          _avg(gen_per_domain, 'corr_ratio'),
        'corr_ratio_correct':  _avg(gen_per_domain, 'corr_ratio_correct'),
        'corr_ratio_wrong':    _avg(gen_per_domain, 'corr_ratio_wrong'),
        'crossing_rate':       _avg(gen_per_domain, 'crossing_rate'),
        'violation_rate':      _avg(gen_per_domain, 'violation_rate'),
        'per_domain':          {str(k): v
                                  for k, v in gen_per_domain.items()},
    }
    return {'multicell': multicell, 'generalist': gen}


# ===========================================================================
# Aggregation
# ===========================================================================

# Metrics where lower is better — delta is gen − multi (positive = multi wins)
LOWER_IS_BETTER = {'cs_known', 'de_mean', 'h_known', 'corr_ratio',
                    'corr_ratio_correct', 'corr_ratio_wrong',
                    'crossing_rate', 'violation_rate'}
# Metrics where higher is better — delta is multi − gen
HIGHER_IS_BETTER = {'cs_unknown', 'cs_gap', 'h_unknown', 'h_ratio',
                     'id_acc_single', 'id_acc_multi'}

ALL_METRICS = sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)


def _ms(arr):
    arr = [a for a in arr
           if a is not None and (not isinstance(a, float) or np.isfinite(a))]
    if not arr: return {'mean': float('nan'), 'std': float('nan')}
    arr = np.asarray(arr, float)
    return {'mean': float(arr.mean()), 'std': float(arr.std())}


def aggregate_runs(runs: dict) -> dict:
    multi = {m: [] for m in ALL_METRICS}
    gen   = {m: [] for m in ALL_METRICS}
    delta = {m: [] for m in ALL_METRICS}
    for seed, rec in runs.items():
        for m in ALL_METRICS:
            mv = rec['multicell'].get(m)
            gv = rec['generalist'].get(m)
            multi[m].append(mv)
            gen[m].append(gv)
            if (mv is None or gv is None
                    or not np.isfinite(mv) or not np.isfinite(gv)):
                continue
            if m in LOWER_IS_BETTER:
                delta[m].append(gv - mv)
            else:
                delta[m].append(mv - gv)
    return {
        'multicell':                         {m: _ms(multi[m]) for m in ALL_METRICS},
        'generalist':                        {m: _ms(gen[m])   for m in ALL_METRICS},
        'delta_multicell_minus_generalist':  {m: _ms(delta[m]) for m in ALL_METRICS},
    }


# ===========================================================================
# Singleton bound
# ===========================================================================

def count_params() -> int:
    """Approx parameter count of GeneralistLSTM at default config."""
    embed  = VOCAB_SIZE * D_MODEL
    lstm   = NUM_LAYERS * 4 * (D_MODEL * D_MODEL + D_MODEL * D_MODEL
                                 + 2 * D_MODEL)
    output = VOCAB_SIZE * D_MODEL + VOCAB_SIZE
    return embed + lstm + output


def singleton_gap_advantage(N: int, n_params: int) -> float:
    d_generalist = (n_params - N) / 2 + 1
    d_specialist = n_params
    return float((d_specialist - d_generalist) / n_params)


# ===========================================================================
# Confirmation
# ===========================================================================

def evaluate_confirmation(aggregate: dict) -> dict:
    def status(p): return 'CONFIRMED' if p else 'NOT MET'
    m = aggregate['multicell']; g = aggregate['generalist']

    def _v(d, k): return d.get(k, {}).get('mean', float('nan'))

    M1_pass = np.isfinite(_v(m, 'cs_known')) and _v(m, 'cs_known') < _v(g, 'cs_known')
    M2_pass = np.isfinite(_v(m, 'cs_gap'))   and _v(m, 'cs_gap')   > _v(g, 'cs_gap')
    M3_pass = np.isfinite(_v(m, 'de_mean'))  and _v(m, 'de_mean')  < _v(g, 'de_mean')
    M4_pass = np.isfinite(_v(m, 'corr_ratio_correct')) and \
              _v(m, 'corr_ratio_correct') < _v(g, 'corr_ratio_correct')
    M5_pass = np.isfinite(_v(m, 'violation_rate')) and \
              _v(m, 'violation_rate') < _v(g, 'violation_rate')

    # M6 — finding, no pass/fail
    id_m = _v(m, 'id_acc_multi'); id_g = _v(g, 'id_acc_multi')
    if not np.isfinite(id_m) or not np.isfinite(id_g):
        winner = 'unknown'; delta = float('nan')
    elif abs(id_m - id_g) < 0.01:
        winner = 'tied'; delta = id_m - id_g
    elif id_m > id_g:
        winner = 'multicell'; delta = id_m - id_g
    else:
        winner = 'generalist'; delta = id_m - id_g

    return {
        'M1': {'status': status(M1_pass),
               'value':  _v(m, 'cs_known'),
               'criterion': 'specialist CS_known < generalist CS_known',
               'multicell': _v(m, 'cs_known'),
               'generalist': _v(g, 'cs_known')},
        'M2': {'status': status(M2_pass),
               'value':  _v(m, 'cs_gap'),
               'criterion': 'specialist CS_gap > generalist CS_gap',
               'multicell': _v(m, 'cs_gap'),
               'generalist': _v(g, 'cs_gap')},
        'M3': {'status': status(M3_pass),
               'value':  _v(m, 'de_mean'),
               'criterion': 'specialist DE < generalist DE',
               'multicell': _v(m, 'de_mean'),
               'generalist': _v(g, 'de_mean')},
        'M4': {'status': status(M4_pass),
               'value':  _v(m, 'corr_ratio_correct'),
               'criterion': 'ratio|correct_id(spec) < ratio|correct_id(gen)',
               'multicell': _v(m, 'corr_ratio_correct'),
               'generalist': _v(g, 'corr_ratio_correct')},
        'M5': {'status': status(M5_pass),
               'value':  _v(m, 'violation_rate'),
               'criterion': 'violation_rate(spec) < violation_rate(gen)',
               'multicell': _v(m, 'violation_rate'),
               'generalist': _v(g, 'violation_rate')},
        'M6': {'status': 'FINDING',
               'winner': winner,
               'delta':  float(delta),
               'criterion': 'identification accuracy winner (no pass/fail)'},
    }


def corollary1_status(confirmation: dict) -> str:
    pf = [confirmation[c]['status'] == 'CONFIRMED'
          for c in ('M1', 'M2', 'M3', 'M4', 'M5')]
    if all(pf): return 'CONFIRMED'
    if sum(pf) >= 3: return 'PARTIAL'
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


def fig_comparison(aggregate: dict, N: int):
    panels = [
        ('CS_gap',           'cs_gap',              '↑ better'),
        ('CS_floor',         'cs_known',            '↓ better'),
        ('DE_mean',          'de_mean',             '↓ better'),
        ('H_ratio',          'h_ratio',             'mixed'),
        ('id_acc_multi',     'id_acc_multi',        '↑ better'),
        ('ratio|correct_id', 'corr_ratio_correct',  '↓ better'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (title, key, direction) in zip(axes.flat, panels):
        m = aggregate['multicell'].get(key, {})
        g = aggregate['generalist'].get(key, {})
        means = [m.get('mean', 0.0), g.get('mean', 0.0)]
        stds  = [m.get('std',  0.0), g.get('std',  0.0)]
        x = np.arange(2)
        ax.bar(x, means, yerr=stds, capsize=4,
               color=['#1f77b4', '#ff7f0e'],
               edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(['Multicell', 'Generalist'])
        ax.set_title(f'{title} ({direction})')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Multicell ({N} specialists) vs Generalist (k={N})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, f'fig_comparison_N{N}')


def fig_per_domain(runs: dict, N: int):
    # Mean ± std of CS_gap per domain across seeds, both arches
    domains = list(range(N))
    multi_means = []; multi_stds = []
    gen_means   = []; gen_stds   = []
    for d in domains:
        m_vals = []; g_vals = []
        for seed, rec in runs.items():
            pd_m = rec['multicell']['per_domain'].get(str(d), {})
            pd_g = rec['generalist']['per_domain'].get(str(d), {})
            if 'cs_gap' in pd_m: m_vals.append(pd_m['cs_gap'])
            if 'cs_gap' in pd_g: g_vals.append(pd_g['cs_gap'])
        multi_means.append(np.mean(m_vals) if m_vals else np.nan)
        multi_stds.append(np.std(m_vals)  if m_vals else 0.0)
        gen_means.append(np.mean(g_vals)  if g_vals else np.nan)
        gen_stds.append(np.std(g_vals)    if g_vals else 0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.asarray(domains)
    ax.errorbar(x, multi_means, yerr=multi_stds, marker='o',
                 color='#1f77b4', label='Multicell (specialist_d)',
                 capsize=3, lw=1.5)
    ax.errorbar(x, gen_means, yerr=gen_stds, marker='s',
                 color='#ff7f0e', label=f'Generalist (k={N})',
                 capsize=3, lw=1.5)
    ax.set_xticks(domains)
    ax.set_xlabel('Domain index d')
    ax.set_ylabel('CS_gap (unknown − known)')
    ax.set_title(f'Per-domain CS_gap, N={N}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, f'fig_per_domain_N{N}')


def fig_singleton_bound():
    """Combine N=5 and N=10 results plus lstm2_1 k=1 baseline."""
    json5  = os.path.join(RESULTS_DIR, 'lstm2_6_N5_results.json')
    json10 = os.path.join(RESULTS_DIR, 'lstm2_6_N10_results.json')
    if not (os.path.exists(json5) and os.path.exists(json10)):
        return False
    with open(json5,  'r', encoding='utf-8') as f: d5  = json.load(f)
    with open(json10, 'r', encoding='utf-8') as f: d10 = json.load(f)

    Ns = [1, 5, 10]
    multi_means = [LSTM2_1_K1_CS_GAP,
                    d5 ['aggregate']['multicell']['cs_gap']['mean'],
                    d10['aggregate']['multicell']['cs_gap']['mean']]
    multi_stds  = [0.0,
                    d5 ['aggregate']['multicell']['cs_gap']['std'],
                    d10['aggregate']['multicell']['cs_gap']['std']]
    gen_means   = [LSTM2_1_K1_CS_GAP,
                    d5 ['aggregate']['generalist']['cs_gap']['mean'],
                    d10['aggregate']['generalist']['cs_gap']['mean']]
    gen_stds    = [0.0,
                    d5 ['aggregate']['generalist']['cs_gap']['std'],
                    d10['aggregate']['generalist']['cs_gap']['std']]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(Ns, multi_means, yerr=multi_stds, marker='o',
                 color='#1f77b4', label='Multicell (specialists)',
                 capsize=3, lw=2.0)
    ax.errorbar(Ns, gen_means, yerr=gen_stds, marker='s', linestyle='--',
                 color='#ff7f0e', label='Generalist (k=N)',
                 capsize=3, lw=2.0)
    ax.fill_between(Ns,
                     [a - b for a, b in zip(multi_means, multi_stds)],
                     [a + b for a, b in zip(gen_means, gen_stds)],
                     color='#aaaaaa', alpha=0.20, label='observed advantage')
    ax.set_xticks(Ns)
    ax.set_xlabel('N (number of domains)')
    ax.set_ylabel('CS_gap')
    ax.set_title('Singleton Bound: CS_gap, multicell vs generalist')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig_singleton_bound')
    return True


# ===========================================================================
# HTML
# ===========================================================================

def _img_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def write_html_report(meta: dict, aggregate: dict, confirmation: dict,
                       N: int):
    out = os.path.join(RESULTS_DIR,
                        f'lstm2_6_N{N}_report.html')
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
             f'<title>{SCRIPT_NAME} N={N}</title>',
             '<style>'
             'body{font-family:-apple-system,sans-serif;max-width:1100px;'
             'margin:2em auto;padding:0 1em;}'
             'table{border-collapse:collapse;margin:1em 0;}'
             'th,td{border:1px solid #ccc;padding:0.4em 0.8em;'
             'text-align:right;}'
             'th{background:#eee;text-align:center;}'
             'td.label{text-align:left;font-weight:500;}'
             '.confirmed{color:#1a7a1a;font-weight:bold;}'
             '.notmet{color:#b00020;font-weight:bold;}'
             '.finding{color:#665500;font-weight:bold;}'
             'img{max-width:100%;border:1px solid #ddd;margin:0.5em 0;}'
             'h2{border-bottom:1px solid #ccc;padding-bottom:0.2em;}'
             '</style></head><body>']
    parts.append(f'<h1>{SCRIPT_NAME} — N={N} — {meta["date"]}</h1>')
    parts.append(f'<p>Device: {meta["device"]} | Mode: '
                 f'{"QUICK" if meta["quick_mode"] else "FULL"} | '
                 f'seeds: {len(meta["seeds"])} | '
                 f'Routing: oracle (always queries the correct specialist)</p>')

    pred = meta.get('singleton_prediction', {})
    parts.append('<h2>Singleton bound prediction (Corollary 1)</h2><pre>'
                 + json.dumps(pred, indent=2) + '</pre>')

    parts.append('<h2>Confirmation</h2><table>'
                 '<tr><th>ID</th><th>Criterion</th>'
                 '<th>Multicell</th><th>Generalist</th>'
                 '<th>Status</th></tr>')
    for cid in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        c = confirmation[cid]
        if cid == 'M6':
            cls = 'finding'
            status_html = (f'FINDING (winner: {c["winner"]}, '
                            f'delta={c["delta"]:+.4f})')
            mc = '—'; gn = '—'
        else:
            cls = 'confirmed' if c['status'] == 'CONFIRMED' else 'notmet'
            status_html = c['status']
            mc = f'{c.get("multicell", float("nan")):.4f}'
            gn = f'{c.get("generalist", float("nan")):.4f}'
        parts.append(f'<tr><td class="label">{cid}</td>'
                     f'<td class="label">{c["criterion"]}</td>'
                     f'<td>{mc}</td><td>{gn}</td>'
                     f'<td class="{cls}">{status_html}</td></tr>')
    parts.append('</table>')

    # Aggregate table
    parts.append('<h2>Aggregate (mean ± std across seeds)</h2><table>'
                 '<tr><th>Metric</th><th>Multicell</th><th>Generalist</th>'
                 '<th>Δ (multi wins +)</th></tr>')
    for m in ALL_METRICS:
        a_m = aggregate['multicell'].get(m, {})
        a_g = aggregate['generalist'].get(m, {})
        d   = aggregate['delta_multicell_minus_generalist'].get(m, {})
        parts.append(
            f'<tr><td class="label">{m}</td>'
            f'<td>{a_m.get("mean", 0):+.4f} ± {a_m.get("std", 0):.4f}</td>'
            f'<td>{a_g.get("mean", 0):+.4f} ± {a_g.get("std", 0):.4f}</td>'
            f'<td>{d.get("mean", 0):+.4f} ± {d.get("std", 0):.4f}</td>'
            f'</tr>')
    parts.append('</table>')

    # Figures
    figs = [(f'Side-by-side comparison (N={N})', f'fig_comparison_N{N}'),
            (f'Per-domain CS_gap (N={N})', f'fig_per_domain_N{N}')]
    for title, base in figs:
        png = os.path.join(RESULTS_DIR, base + '.png')
        if os.path.exists(png):
            b64 = _img_b64(png)
            parts.append(f'<h2>{title}</h2>'
                         f'<img src="data:image/png;base64,{b64}">')
    sb_png = os.path.join(RESULTS_DIR, 'fig_singleton_bound.png')
    if os.path.exists(sb_png):
        parts.append('<h2>Singleton bound (multicell vs generalist '
                     'across N)</h2>'
                     f'<img src="data:image/png;base64,'
                     f'{_img_b64(sb_png)}">')
    else:
        parts.append('<p><em>fig_singleton_bound: requires both N=5 '
                     'and N=10 to be present.</em></p>')

    parts.append('<h2>Identification accuracy finding (M6)</h2>')
    m6 = confirmation['M6']
    parts.append(f'<p>Reported as a finding (no pass/fail). Winner: '
                 f'<b>{m6["winner"]}</b>, delta='
                 f'<b>{m6["delta"]:+.4f}</b> (multicell − generalist).</p>')
    parts.append('</body></html>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ===========================================================================
# RESULTS SUMMARY
# ===========================================================================

def print_results_summary(meta, aggregate, confirmation, N, runs):
    pred = meta.get('singleton_prediction', {}).get(
        'predicted_cs_gap_advantage', float('nan'))
    obs_delta = aggregate['delta_multicell_minus_generalist'] \
        .get('cs_gap', {})

    def _fmt_pair(metric):
        m = aggregate['multicell'].get(metric, {})
        g = aggregate['generalist'].get(metric, {})
        d = aggregate['delta_multicell_minus_generalist'].get(metric, {})
        if metric in LOWER_IS_BETTER:
            winner = ('multi' if d.get('mean', 0) > 0
                       else ('gen' if d.get('mean', 0) < 0 else 'tied'))
        else:
            winner = ('multi' if d.get('mean', 0) > 0
                       else ('gen' if d.get('mean', 0) < 0 else 'tied'))
        return (f"{m.get('mean', 0):+.4f}±{m.get('std', 0):.3f}",
                f"{g.get('mean', 0):+.4f}±{g.get('std', 0):.3f}",
                f"{d.get('mean', 0):+.4f}",
                winner)

    print()
    print('=' * 55)
    print(f'RESULTS SUMMARY — lstm2_6_singlecell_vs_multicell [N={N}]')
    print('=' * 55)
    print(f"Date:    {meta['date']}")
    print(f"Device:  {meta['device']}")
    print(f"Mode:    {'QUICK' if meta['quick_mode'] else 'FULL'}")
    print(f"N:       {N}")
    print(f"Domains: {{0,...,{N-1}}}   Routing: oracle")
    print(f"Seeds:   {meta['seeds']}")

    print()
    print('-- AGGREGATE COMPARISON (mean ± std across seeds) ------')
    print(f"{'Metric':<22}{'Multicell':<22}{'Generalist':<22}"
          f"{'Delta':<14}{'Winner':<6}")
    show = ['cs_known', 'cs_gap', 'de_mean', 'h_ratio',
            'id_acc_multi', 'id_acc_single',
            'corr_ratio_correct', 'corr_ratio_wrong',
            'crossing_rate', 'violation_rate']
    for metric in show:
        a, b, dlt, w = _fmt_pair(metric)
        print(f"{metric:<22}{a:<22}{b:<22}{dlt:<14}{w:<6}")

    print()
    print('-- SINGLETON BOUND CHECK -------------------------------')
    print(f"Predicted CS_gap advantage (Corollary 1): {pred:.4f}")
    obs_mean = obs_delta.get('mean', float('nan'))
    obs_std  = obs_delta.get('std', float('nan'))
    print(f"Observed CS_gap delta (multicell − generalist): "
          f"{obs_mean:+.4f} ± {obs_std:.4f}")
    same_dir = (np.isfinite(pred) and np.isfinite(obs_mean)
                 and (pred * obs_mean > 0))
    print(f"Direction matches prediction: {'YES' if same_dir else 'NO'}")

    print()
    print('-- CONFIRMATION ----------------------------------------')
    c = confirmation
    print(f"M1  Specialist CS_floor < generalist     [{c['M1']['status']}]   "
          f"multi={c['M1']['multicell']:.4f}  gen={c['M1']['generalist']:.4f}")
    print(f"M2  Specialist CS_gap > generalist       [{c['M2']['status']}]   "
          f"multi={c['M2']['multicell']:.4f}  gen={c['M2']['generalist']:.4f}")
    print(f"M3  Specialist DE < generalist           [{c['M3']['status']}]   "
          f"multi={c['M3']['multicell']:.3f}  gen={c['M3']['generalist']:.3f}")
    print(f"M4  Specialist correction better         [{c['M4']['status']}]   "
          f"multi={c['M4']['multicell']:.4f}  gen={c['M4']['generalist']:.4f}")
    print(f"M5  Specialist violation rate lower      [{c['M5']['status']}]   "
          f"multi={c['M5']['multicell']:.4f}  gen={c['M5']['generalist']:.4f}")
    print(f"M6  Identification accuracy winner       [FINDING: "
          f"{c['M6']['winner']}  delta={c['M6']['delta']:+.4f}]")

    # Convergence
    print()
    print('-- CONVERGENCE (specialist training) -------------------')
    failed = []
    for seed, rec in runs.items():
        for d_str, loss in rec['multicell'].get('specialist_losses', {}).items():
            if float(loss) >= 1.0:
                failed.append((seed, d_str, loss))
    if failed:
        for seed, d, loss in failed:
            print(f"  d={d:>2}  seed={seed:>5}  loss={loss:.3f}  [FAILED]")
    else:
        print('  none')

    print()
    pf = [c[k]['status'] == 'CONFIRMED' for k in ('M1', 'M2', 'M3', 'M4', 'M5')]
    all_met = all(pf)
    print('-- OVERALL ---------------------------------------------')
    print(f"All pass/fail confirmations met: {'YES' if all_met else 'NO'}")
    print(f"Corollary 1 status: {corollary1_status(c)}")
    print(f"Output: results/{SCRIPT_NAME}/")
    print('=' * 55)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    N = int(args.N)
    if args.quick:
        seeds = SEEDS_QUICK
        n_cs = N_CS_QUICK; n_test_id = N_TEST_ID_QUICK
        n_test_corr = N_TEST_CORR_QUICK; n_dirs = N_DIRS_QUICK
        print('[QUICK MODE]')
    else:
        seeds = SEEDS
        n_cs = N_CS_PER_DOMAIN; n_test_id = N_TEST_ID
        n_test_corr = N_TEST_CORR; n_dirs = N_DIRS

    device = get_device()
    print(f"Script: {SCRIPT_NAME}")
    print(f"N={N}    Seeds: {seeds}")
    print(f"n_cs={n_cs}  n_test_id={n_test_id}  "
          f"n_test_corr={n_test_corr}  n_dirs={n_dirs}")

    n_params = count_params()
    pred_adv = singleton_gap_advantage(N, n_params)
    print(f"n_params={n_params}  Singleton predicted CS_gap advantage="
          f"{pred_adv:.4e}")

    # ---- Per-(N, seed) eval --------------------------------------
    runs: dict = {}
    for seed in seeds:
        t0 = time.time()
        log = lambda msg: print(msg, flush=True)
        rec = evaluate_seed(N, seed, device, n_cs, n_test_id, n_test_corr,
                              n_dirs, log)
        runs[str(seed)] = rec
        m = rec['multicell']; g = rec['generalist']
        log(f"  seed={seed} summary:")
        log(f"    multicell:   CS_gap={m['cs_gap']:.3f}  DE={m['de_mean']:.2f}  "
            f"H_ratio={m['h_ratio']:.2f}x  id_acc={m['id_acc_multi']:.3f}  "
            f"ratio|correct={m['corr_ratio_correct']:.3f}")
        log(f"    generalist:  CS_gap={g['cs_gap']:.3f}  DE={g['de_mean']:.2f}  "
            f"H_ratio={g['h_ratio']:.2f}x  id_acc={g['id_acc_multi']:.3f}  "
            f"ratio|correct={g['corr_ratio_correct']:.3f}")
        log(f"  ({time.time()-t0:.0f}s)")

    aggregate = aggregate_runs(runs)
    confirmation = evaluate_confirmation(aggregate)

    meta = {
        'script': SCRIPT_NAME,
        'N': N,
        'date': datetime.datetime.now().isoformat(),
        'device': str(device),
        'seeds': seeds,
        'quick_mode': args.quick,
        'domains': list(range(N)),
        'config': {
            'n_cs_per_domain': n_cs, 'n_test_id': n_test_id,
            'n_test_corr': n_test_corr, 'n_dirs': n_dirs,
            'eps_dict': EPS_DICT, 'n_wrong_dirs': N_WRONG_DIRS,
            'oracle_routing': True,
            'generalist_models_dir': GENERALIST_MODELS_DIR,
            'specialist_models_dir': SPECIALIST_MODELS_DIR,
        },
        'singleton_prediction': {
            'N': N,
            'd_specialist': 'n',
            'd_generalist_bound': '(n-N)/2 + 1',
            'predicted_cs_gap_advantage': pred_adv,
            'n_params': n_params,
        },
    }

    output = {
        'meta': meta,
        'runs': runs,
        'aggregate': aggregate,
        'confirmation': confirmation,
        'corollary1_status': corollary1_status(confirmation),
    }
    save_json(output, os.path.join(RESULTS_DIR,
                                     f'lstm2_6_N{N}_results.json'))

    fig_comparison(aggregate, N)
    fig_per_domain(runs, N)
    sb_done = fig_singleton_bound()
    write_html_report(meta, aggregate, confirmation, N)

    if not sb_done:
        print('  [note] fig_singleton_bound deferred until both N=5 and '
              'N=10 results exist.')

    print_results_summary(meta, aggregate, confirmation, N, runs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
