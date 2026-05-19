"""
lstm2_3_detectability.py — Syndrome Identification Across k Domains
====================================================================
Trains the same (k, seed) models as lstm2_1 (deterministic via seeds),
saves the weights for downstream scripts, then measures:
  - single-layer vs multi-layer syndrome identification accuracy
  - Jacobian variance V_j per direction

All model/utility code imported from lstm2_model.py.

Critical detail: perturbation target is `lstm_layers[0].weight_ih_l0` —
NEVER `output.weight`. Output-only perturbation gives a degenerate
multi-layer signal because hidden states do not depend on the output
projection.
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
    VOCAB_SIZE, D_MODEL, NUM_LAYERS, SEQ_LEN,
    BATCH_SIZE, LR, SEEDS_BASE,
    PREFIX_LEN, GEN_LEN, ERR_POS,
    GeneralistLSTM, make_domain_sequences, make_training_data,
    train_model, get_weight_matrix, compute_svd_basis,
    measure_syndrome, build_multilayer_syndrome,
    make_probe_test_split,
    perturb_weight, restore_weight,
    set_seed, get_device, make_results_dir, save_json,
)


SCRIPT_NAME = 'lstm2_3_detectability'
RESULTS_DIR = make_results_dir(SCRIPT_NAME)
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ===========================================================================
# 2. Configuration
# ===========================================================================

SEQS_PER_DOMAIN = 2000
EPOCHS          = 200

N_DIRS    = 200
N_PROBE   = 50
EPS_DICT  = 3.0

N_TEST    = 3000
EPS_MIN   = 1.0
EPS_MAX   = 5.0

N_PROBE_JV = 50

VJ_THRESHOLD = 0.1

CONFUSION_K = [1, 5, 10]

CV_CS_GAP_REF = 0.273
CV_CS_GAP_TOL = 0.10

SEEDS    = SEEDS_BASE
K_VALUES = [1, 2, 3, 5, 10]

# Quick mode
K_VALUES_QUICK = [1, 5, 10]
SEEDS_QUICK    = SEEDS_BASE[:3]
N_TEST_QUICK   = 500
N_DIRS_QUICK   = 50


# ===========================================================================
# 3. Perturbation target
# ===========================================================================

def get_target_weight(model: GeneralistLSTM) -> torch.nn.Parameter:
    """Return the perturbation target weight parameter.

    lstm_layers[0].weight_ih_l0 — input-to-hidden weight of the first LSTM
    layer, shape [4 * D_MODEL, D_MODEL]. SVD gives D_MODEL right singular
    vectors that propagate through all subsequent layers.
    """
    return model.lstm_layers[0].weight_ih_l0


# ===========================================================================
# 4. Cross-validation: quick CS-gap measurement at k=1
# ===========================================================================

def quick_cs_gap(model: GeneralistLSTM, device: torch.device,
                  n_trials: int = 50) -> float:
    """50-trial CS measurement: domain 0 = known, domain 1 = unknown.

    Returns CS_unknown - CS_known.
    """
    def _cs(domain_idx):
        seqs = make_domain_sequences(domain_idx, n_trials,
                                       seq_len=PREFIX_LEN + 2)
        seqs = seqs.to(device)
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
    return _cs(1) - _cs(0)


# ===========================================================================
# 5. Model training / loading
# ===========================================================================

def get_or_train_model(k: int, seed: int, device: torch.device
                        ) -> tuple[GeneralistLSTM, float, bool]:
    """Load saved model if present; else train and save.

    Returns: (model, final_loss, was_loaded)
    """
    model_path = os.path.join(MODELS_DIR, f'k{k}_seed{seed}.pt')
    if os.path.exists(model_path):
        model = GeneralistLSTM()
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        return model, float(ckpt.get('final_loss', 0.0)), True

    set_seed(seed)
    data = make_training_data(k=k, seqs_per_domain=SEQS_PER_DOMAIN,
                                seq_len=SEQ_LEN)
    model = GeneralistLSTM()
    model, final_loss = train_model(model, data, epochs=EPOCHS,
                                     lr=LR, batch_size=BATCH_SIZE,
                                     device=device, verbose=False)
    torch.save({
        'model_state_dict': model.state_dict(),
        'k': k, 'seed': seed, 'final_loss': float(final_loss),
        'config': {
            'd_model': D_MODEL, 'num_layers': NUM_LAYERS,
            'vocab_size': VOCAB_SIZE, 'seq_len': SEQ_LEN,
            'seqs_per_domain': SEQS_PER_DOMAIN, 'epochs': EPOCHS,
        },
    }, model_path)
    return model, float(final_loss), False


# ===========================================================================
# 6. Dictionary construction
# ===========================================================================

def build_dictionaries(model: GeneralistLSTM, Vh: torch.Tensor,
                        n_dirs: int, probe_seqs: torch.Tensor,
                        device: torch.device
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Build single-layer and multi-layer syndrome dictionaries."""
    W = get_target_weight(model)
    d_multi = NUM_LAYERS * D_MODEL + VOCAB_SIZE
    dict_single = np.zeros((n_dirs, VOCAB_SIZE), dtype=np.float32)
    dict_multi  = np.zeros((n_dirs, d_multi),    dtype=np.float32)
    probe_seqs_d = probe_seqs.to(device)
    for d in range(n_dirs):
        direction = Vh[d]
        logit_syn, layer_syns = measure_syndrome(
            model, W, direction, EPS_DICT,
            probe_seqs_d, device,
            return_layer_deltas=True,
        )
        dict_single[d] = logit_syn.astype(np.float32)
        dict_multi[d]  = build_multilayer_syndrome(
            logit_syn, layer_syns, injection_layer=0
        ).astype(np.float32)
    # Re-normalise rows (defensive — components from measure_syndrome are
    # already normalised, this guards numerical drift).
    s_norm = np.linalg.norm(dict_single, axis=1, keepdims=True)
    s_norm = np.where(s_norm < 1e-12, 1.0, s_norm)
    dict_single = dict_single / s_norm
    m_norm = np.linalg.norm(dict_multi, axis=1, keepdims=True)
    m_norm = np.where(m_norm < 1e-12, 1.0, m_norm)
    dict_multi = dict_multi / m_norm
    return dict_single, dict_multi


# ===========================================================================
# 7. Jacobian variance
# ===========================================================================

def measure_vj(model: GeneralistLSTM, direction: torch.Tensor,
               eps: float, probe_seqs: torch.Tensor,
               device: torch.device) -> tuple[float, np.ndarray]:
    """Variance of the per-probe syndrome around its mean.

    Returns (vj, syns) — vj is a scalar; syns is [N_PROBE_JV, VOCAB_SIZE].
    """
    W = get_target_weight(model)
    syns = []
    for i in range(probe_seqs.shape[0]):
        s = measure_syndrome(model, W, direction, eps,
                              probe_seqs[i:i + 1], device,
                              return_layer_deltas=False)
        syns.append(s)
    syns = np.stack(syns).astype(np.float32)
    s_mean = syns.mean(axis=0)
    vj = float(np.mean(np.sum((syns - s_mean) ** 2, axis=1)))
    return vj, syns


# ===========================================================================
# 8. Identification test
# ===========================================================================

def run_identification_test(model: GeneralistLSTM, Vh: torch.Tensor,
                             dict_single: np.ndarray, dict_multi: np.ndarray,
                             test_seqs: torch.Tensor, n_dirs: int,
                             n_test: int, seed: int,
                             store_confusion: bool, device: torch.device
                             ) -> dict:
    W = get_target_weight(model)
    rng = np.random.default_rng(seed)
    test_seqs_d = test_seqs.to(device)

    correct_single = 0
    correct_multi  = 0
    wrong_cosines_single = []
    wrong_cosines_multi  = []

    confusion_single = (np.zeros((n_dirs, n_dirs), dtype=np.int32)
                         if store_confusion else None)
    confusion_multi  = (np.zeros((n_dirs, n_dirs), dtype=np.int32)
                         if store_confusion else None)

    # Per-direction accumulators (for V_j vs accuracy correlation)
    dir_correct_multi = np.zeros(n_dirs, dtype=np.int64)
    dir_total         = np.zeros(n_dirs, dtype=np.int64)

    progress_every = max(n_test // 4, 1)
    t_loop = time.time()

    for trial in range(n_test):
        if trial > 0 and trial % progress_every == 0:
            print(f"    test trials: {trial}/{n_test} "
                  f"({time.time()-t_loop:.0f}s)  "
                  f"running single={correct_single/trial:.3f}  "
                  f"multi={correct_multi/trial:.3f}", flush=True)
        true_dir = int(rng.integers(0, n_dirs))
        eps_test = float(rng.uniform(EPS_MIN, EPS_MAX))

        logit_syn, layer_syns = measure_syndrome(
            model, W, Vh[true_dir], eps_test,
            test_seqs_d, device,
            return_layer_deltas=True,
        )

        ts = logit_syn.astype(np.float32)
        ts_norm = float(np.linalg.norm(ts))
        if ts_norm < 1e-12:
            ts = ts
        else:
            ts = ts / ts_norm

        tm = build_multilayer_syndrome(
            logit_syn, layer_syns, injection_layer=0
        ).astype(np.float32)

        pred_single = int(np.argmax(dict_single @ ts))
        pred_multi  = int(np.argmax(dict_multi  @ tm))

        if pred_single == true_dir:
            correct_single += 1
        else:
            wrong_cosines_single.append(float(dict_single[true_dir] @ ts))
        if pred_multi == true_dir:
            correct_multi += 1
            dir_correct_multi[true_dir] += 1
        else:
            wrong_cosines_multi.append(float(dict_multi[true_dir] @ tm))
        dir_total[true_dir] += 1

        if store_confusion:
            confusion_single[true_dir, pred_single] += 1
            confusion_multi[true_dir, pred_multi]   += 1

    return {
        'acc_single':  correct_single / n_test,
        'acc_multi':   correct_multi  / n_test,
        'conf_single': (float(np.mean(wrong_cosines_single))
                        if wrong_cosines_single else 1.0),
        'conf_multi':  (float(np.mean(wrong_cosines_multi))
                        if wrong_cosines_multi else 1.0),
        'confusion_single': (confusion_single.tolist()
                              if confusion_single is not None else None),
        'confusion_multi':  (confusion_multi.tolist()
                              if confusion_multi is not None else None),
        'dir_correct_multi': dir_correct_multi.tolist(),
        'dir_total':         dir_total.tolist(),
    }


# ===========================================================================
# 8b. Optimized inner loops — batched V_j + amortized clean forward
# ===========================================================================
#
# The original implementation calls measure_syndrome once per probe and
# once per trial. measure_syndrome computes a CLEAN forward every time
# even though the model and inputs do not change. For:
#   - dictionary build: 200 dirs × (1 clean + 1 pert) = 400 forwards
#   - V_j:              200 dirs × 50 probes × 2 = 20 000 tiny forwards
#   - identification:   3000 trials × (1 clean + 1 pert) = 6000 forwards
#
# Optimized:
#   - Compute the clean forward ONCE on the probe set, then for each
#     direction perform a single perturbed forward and compute both the
#     averaged delta (for the dictionary) and the per-probe delta (for
#     V_j) from the same output. 200 forwards for dict + V_j combined.
#   - Same trick for the identification test: clean forward on the test
#     set ONCE, then 3000 perturbed forwards.
#   - Keep clean state on GPU and compute deltas on GPU, transferring
#     only the small averaged vectors (256-d logit + 10 × 256-d layer)
#     to CPU. Avoids 1+ GB PCIe transfers per trial.
#
# Numerically equivalent to the original implementation: same averages,
# same normalisation, same multi-layer concatenation.

def _forward_with_layers(model: GeneralistLSTM, x: torch.Tensor):
    """One forward pass returning (logits, list of layer outputs) on GPU,
    detached from autograd."""
    with torch.no_grad():
        logits, layer_outs = model(x, return_all_layers=True)
    return logits.detach(), [lo.detach() for lo in layer_outs]


def _normalize_np(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


def build_dict_and_vj(model: GeneralistLSTM, Vh: torch.Tensor,
                       n_dirs: int, probe_seqs: torch.Tensor,
                       device: torch.device,
                       progress_print: callable = None
                       ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Combined dictionary construction and V_j measurement.

    For each direction:
      1. Perturb the target weight by EPS_DICT * direction.
      2. Single forward on the full probe batch.
      3. Compute averaged delta (used for dict) AND per-probe delta
         (used for V_j) from the same output.

    Returns (dict_single, dict_multi, vj_per_direction). Equivalent to
    the original build_dictionaries + measure_vj loop, but ~50× fewer
    kernel launches.
    """
    model.eval()
    W = get_target_weight(model)
    probe_seqs_d = probe_seqs.to(device)

    # Clean forward — once
    clean_logits_g, clean_layers_g = _forward_with_layers(model, probe_seqs_d)

    n_probe = probe_seqs.shape[0]
    d_multi = NUM_LAYERS * D_MODEL + VOCAB_SIZE
    dict_single = np.zeros((n_dirs, VOCAB_SIZE), dtype=np.float32)
    dict_multi  = np.zeros((n_dirs, d_multi),    dtype=np.float32)
    vj_per_direction: list[float] = []

    for d in range(n_dirs):
        original = perturb_weight(W, Vh[d], EPS_DICT)
        try:
            pert_logits_g, pert_layers_g = _forward_with_layers(
                model, probe_seqs_d)
        finally:
            restore_weight(W, original)

        # Logit delta — averaged (for dict) and per-probe (for V_j)
        diff_logits = pert_logits_g - clean_logits_g           # [B, T, V]
        # Averaged over (B, T) -> [V]
        delta_avg = diff_logits.mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
        # Per-probe -> [B, V]
        delta_per_probe = diff_logits.mean(dim=1).cpu().numpy().astype(np.float32)

        # Dict single-layer entry
        n_avg = float(np.linalg.norm(delta_avg))
        logit_syn = (delta_avg / n_avg) if n_avg > 1e-12 else delta_avg
        dict_single[d] = logit_syn

        # Layer deltas — averaged for dict, only [D] per layer transferred
        layer_syns = []
        for L in range(NUM_LAYERS):
            dL_avg = (pert_layers_g[L] - clean_layers_g[L]).mean(
                dim=(0, 1)).cpu().numpy().astype(np.float32)        # [D]
            n_L = float(np.linalg.norm(dL_avg))
            ls = (dL_avg / n_L) if n_L > 1e-12 else dL_avg
            layer_syns.append(ls)

        # Multi-layer syndrome (already final-normalised inside builder)
        dict_multi[d] = build_multilayer_syndrome(
            logit_syn, layer_syns, injection_layer=0
        ).astype(np.float32)

        # V_j: variance of per-probe (logit) syndrome around its mean
        syns = _normalize_np(delta_per_probe, axis=1)               # [B, V]
        s_mean = syns.mean(axis=0)
        vj = float(np.mean(np.sum((syns - s_mean) ** 2, axis=1)))
        vj_per_direction.append(vj)

        if progress_print is not None and (d + 1) % max(n_dirs // 4, 1) == 0:
            progress_print(d + 1, n_dirs)

    # Defensive renormalise rows
    dict_single = _normalize_np(dict_single, axis=1)
    dict_multi  = _normalize_np(dict_multi,  axis=1)
    return dict_single, dict_multi, vj_per_direction


def run_test_optimized(model: GeneralistLSTM, Vh: torch.Tensor,
                        dict_single: np.ndarray, dict_multi: np.ndarray,
                        test_seqs: torch.Tensor, n_dirs: int,
                        n_test: int, seed: int,
                        store_confusion: bool, device: torch.device,
                        progress_print: callable = None
                        ) -> dict:
    """Identification test with the clean forward computed once.

    Numerically equivalent to run_identification_test: same RNG draws,
    same metric definitions, same confusion matrix; ~2× fewer forwards
    because we do not re-compute clean every trial.
    """
    model.eval()
    W = get_target_weight(model)
    test_seqs_d = test_seqs.to(device)

    # Clean forward — once
    clean_logits_g, clean_layers_g = _forward_with_layers(model, test_seqs_d)

    rng = np.random.default_rng(seed)
    correct_single = 0
    correct_multi  = 0
    wrong_cosines_single = []
    wrong_cosines_multi  = []
    confusion_single = (np.zeros((n_dirs, n_dirs), dtype=np.int32)
                         if store_confusion else None)
    confusion_multi  = (np.zeros((n_dirs, n_dirs), dtype=np.int32)
                         if store_confusion else None)
    dir_correct_multi = np.zeros(n_dirs, dtype=np.int64)
    dir_total         = np.zeros(n_dirs, dtype=np.int64)

    progress_every = max(n_test // 4, 1)
    t_loop = time.time()

    for trial in range(n_test):
        true_dir = int(rng.integers(0, n_dirs))
        eps_test = float(rng.uniform(EPS_MIN, EPS_MAX))

        original = perturb_weight(W, Vh[true_dir], eps_test)
        try:
            pert_logits_g, pert_layers_g = _forward_with_layers(
                model, test_seqs_d)
        finally:
            restore_weight(W, original)

        # Averaged logit delta -> [V] on GPU, transfer only this small vec
        delta_logits = (pert_logits_g - clean_logits_g).mean(
            dim=(0, 1)).cpu().numpy().astype(np.float32)
        n_l = float(np.linalg.norm(delta_logits))
        logit_syn = (delta_logits / n_l) if n_l > 1e-12 else delta_logits

        layer_syns = []
        for L in range(NUM_LAYERS):
            dL = (pert_layers_g[L] - clean_layers_g[L]).mean(
                dim=(0, 1)).cpu().numpy().astype(np.float32)
            n_L = float(np.linalg.norm(dL))
            layer_syns.append((dL / n_L) if n_L > 1e-12 else dL)

        ts = logit_syn  # already normalised (or zero)
        tm = build_multilayer_syndrome(
            logit_syn, layer_syns, injection_layer=0
        ).astype(np.float32)

        pred_single = int(np.argmax(dict_single @ ts))
        pred_multi  = int(np.argmax(dict_multi  @ tm))

        if pred_single == true_dir:
            correct_single += 1
        else:
            wrong_cosines_single.append(float(dict_single[true_dir] @ ts))
        if pred_multi == true_dir:
            correct_multi += 1
            dir_correct_multi[true_dir] += 1
        else:
            wrong_cosines_multi.append(float(dict_multi[true_dir] @ tm))
        dir_total[true_dir] += 1

        if store_confusion:
            confusion_single[true_dir, pred_single] += 1
            confusion_multi[true_dir, pred_multi]   += 1

        if progress_print is not None and trial > 0 and trial % progress_every == 0:
            progress_print(trial, n_test, time.time() - t_loop,
                           correct_single / trial, correct_multi / trial)

    return {
        'acc_single':  correct_single / n_test,
        'acc_multi':   correct_multi  / n_test,
        'conf_single': (float(np.mean(wrong_cosines_single))
                        if wrong_cosines_single else 1.0),
        'conf_multi':  (float(np.mean(wrong_cosines_multi))
                        if wrong_cosines_multi else 1.0),
        'confusion_single': (confusion_single.tolist()
                              if confusion_single is not None else None),
        'confusion_multi':  (confusion_multi.tolist()
                              if confusion_multi is not None else None),
        'dir_correct_multi': dir_correct_multi.tolist(),
        'dir_total':         dir_total.tolist(),
    }


# ===========================================================================
# 9. Per-(k, seed) measurement
# ===========================================================================

def run_one_cell(k: int, seed: int, n_dirs: int, n_test: int,
                  device: torch.device, quick: bool) -> dict:
    t0 = time.time()
    print(f"k={k:<3}seed={seed:<6}START", flush=True)

    t_model = time.time()
    model, final_loss, was_loaded = get_or_train_model(k, seed, device)
    converged = bool(final_loss < 1.0)
    if was_loaded:
        print(f"  [k={k} seed={seed}] model loaded "
              f"({time.time()-t_model:.1f}s)", flush=True)
    else:
        print(f"  [k={k} seed={seed}] model trained "
              f"({EPOCHS}ep, {time.time()-t_model:.0f}s, "
              f"loss={final_loss:.3f})", flush=True)

    # Cross-validation at k=1
    cv_ok = None
    cv_cs_gap = None
    if k == 1:
        t_cv = time.time()
        cv_cs_gap = quick_cs_gap(model, device, n_trials=50)
        cv_ok = bool(abs(cv_cs_gap - CV_CS_GAP_REF) <= CV_CS_GAP_TOL)
        print(f"  [k={k} seed={seed}] CV {'OK' if cv_ok else 'FAIL'}: "
              f"CS_gap={cv_cs_gap:.3f} (ref {CV_CS_GAP_REF}±{CV_CS_GAP_TOL})  "
              f"({time.time()-t_cv:.1f}s)", flush=True)

    # SVD basis on the perturbation target
    W_cpu = get_weight_matrix(model, 0)
    S, Vh = compute_svd_basis(W_cpu)
    Vh = Vh.to(device)

    # Probe / test split (non-overlapping by construction in lstm2_model)
    probe_seqs, test_seqs = make_probe_test_split(
        domain_idx=0, n_probe=N_PROBE, n_test=n_test
    )

    # Dictionary + V_j combined (optimized: clean forward done once,
    # per-direction work yields BOTH the averaged delta for the dict and
    # the per-probe delta for V_j)
    t_dv = time.time()
    print(f"  [k={k} seed={seed}] building {n_dirs} dict entries + V_j "
          f"(combined, batched)...", flush=True)
    def _dv_progress(done, total):
        print(f"    dict+V_j: {done}/{total} done "
              f"({time.time()-t_dv:.0f}s)", flush=True)
    dict_single, dict_multi, vj_per_direction = build_dict_and_vj(
        model, Vh, n_dirs, probe_seqs, device,
        progress_print=_dv_progress,
    )
    vj_arr = np.asarray(vj_per_direction)
    vj_mean = float(vj_arr.mean())
    vj_std  = float(vj_arr.std())
    vj_high_frac = float(np.mean(vj_arr > VJ_THRESHOLD))
    print(f"  [k={k} seed={seed}] dict + V_j done ({time.time()-t_dv:.0f}s)  "
          f"vj_mean={vj_mean:.4f}  vj_high_frac={vj_high_frac:.2f}",
          flush=True)

    # Identification test (optimized: clean forward done once on test set,
    # then 1 perturbed forward per trial instead of 2)
    t_test = time.time()
    print(f"  [k={k} seed={seed}] running {n_test} identification trials...",
          flush=True)
    store_confusion = (k in CONFUSION_K)
    def _test_progress(trial, total, elapsed, run_s, run_m):
        print(f"    test trials: {trial}/{total} ({elapsed:.0f}s)  "
              f"running single={run_s:.3f}  multi={run_m:.3f}", flush=True)
    test_res = run_test_optimized(
        model, Vh, dict_single, dict_multi, test_seqs,
        n_dirs, n_test, seed, store_confusion, device,
        progress_print=_test_progress,
    )
    print(f"  [k={k} seed={seed}] test done ({time.time()-t_test:.0f}s)  "
          f"single={test_res['acc_single']:.3f}  "
          f"multi={test_res['acc_multi']:.3f}", flush=True)

    elapsed = time.time() - t0
    record = {
        'final_loss': float(final_loss),
        'converged':  converged,
        'cv_ok':      cv_ok,
        'cv_cs_gap':  cv_cs_gap,
        'was_loaded': was_loaded,
        'acc_single':  test_res['acc_single'],
        'acc_multi':   test_res['acc_multi'],
        'conf_single': test_res['conf_single'],
        'conf_multi':  test_res['conf_multi'],
        'confusion_single': test_res['confusion_single'],
        'confusion_multi':  test_res['confusion_multi'],
        'dir_correct_multi': test_res['dir_correct_multi'],
        'dir_total':         test_res['dir_total'],
        'vj_mean': vj_mean,
        'vj_std':  vj_std,
        'vj_high_frac': vj_high_frac,
        'vj_per_direction': vj_per_direction,
        'elapsed_s': float(elapsed),
    }

    # Progress print
    cv_tag = ''
    if cv_ok is not None:
        cv_tag = '  [CV-OK]' if cv_ok else '  [CV-FAIL]'
    train_tag = ('[loaded]' if was_loaded else
                  f'[trained {EPOCHS}ep, loss={final_loss:.3f}]')
    print(f"k={k:<3}seed={seed:<6}{train_tag}{cv_tag}")
    if cv_cs_gap is not None:
        print(f"     CV CS_gap={cv_cs_gap:.3f} "
              f"(ref {CV_CS_GAP_REF}±{CV_CS_GAP_TOL})")
    print(f"     dict built  single-acc={test_res['acc_single']:.3f}  "
          f"multi-acc={test_res['acc_multi']:.3f}  ({elapsed:.0f}s)")
    print(f"     vj_mean={vj_mean:.4f}  "
          f"vj_high_frac={vj_high_frac:.2f}")
    if final_loss > 1.0:
        print(f"     [WARNING] training may not have converged "
              f"(loss={final_loss:.3f})")
    return record


# ===========================================================================
# 10. Aggregation
# ===========================================================================

def aggregate_runs(runs: dict, k_values: list[int]) -> dict:
    agg = {}
    for k in k_values:
        k_str = str(k)
        if k_str not in runs or not runs[k_str]:
            continue
        rs = runs[k_str]
        acc_s = [r['acc_single'] for r in rs.values()]
        acc_m = [r['acc_multi']  for r in rs.values()]
        conf_s = [r['conf_single'] for r in rs.values()]
        conf_m = [r['conf_multi']  for r in rs.values()]
        vj_m  = [r['vj_mean']    for r in rs.values()]
        vj_h  = [r['vj_high_frac'] for r in rs.values()]
        n_conv = sum(1 for r in rs.values() if r.get('converged'))
        n_cv_ok = sum(1 for r in rs.values()
                       if r.get('cv_ok') is True)
        agg[k_str] = {
            'acc_single':   {'mean': float(np.mean(acc_s)),
                              'std':  float(np.std(acc_s))},
            'acc_multi':    {'mean': float(np.mean(acc_m)),
                              'std':  float(np.std(acc_m))},
            'conf_single':  {'mean': float(np.mean(conf_s)),
                              'std':  float(np.std(conf_s))},
            'conf_multi':   {'mean': float(np.mean(conf_m)),
                              'std':  float(np.std(conf_m))},
            'vj_mean':      {'mean': float(np.mean(vj_m)),
                              'std':  float(np.std(vj_m))},
            'vj_high_frac': {'mean': float(np.mean(vj_h)),
                              'std':  float(np.std(vj_h))},
            'multi_minus_single': float(np.mean(acc_m) - np.mean(acc_s)),
            'n_converged': int(n_conv),
            'n_failed':    int(len(rs) - n_conv),
            'n_cv_ok':     int(n_cv_ok),
        }
    return agg


def compute_pearson_vj_acc(runs: dict) -> float:
    """Pool per-direction (V_j, accuracy) pairs across k and seeds."""
    xs, ys = [], []
    for k_str, sdict in runs.items():
        for seed_str, r in sdict.items():
            vj  = np.asarray(r['vj_per_direction'])
            cor = np.asarray(r['dir_correct_multi'], dtype=np.float64)
            tot = np.asarray(r['dir_total'], dtype=np.float64)
            mask = tot > 0
            if not mask.any():
                continue
            acc = np.zeros_like(tot)
            acc[mask] = cor[mask] / tot[mask]
            xs.append(vj[mask])
            ys.append(acc[mask])
    if not xs:
        return 0.0
    x = np.concatenate(xs); y = np.concatenate(ys)
    if x.std() < 1e-12 or y.std() < 1e-12 or x.size < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ===========================================================================
# 11. Confirmation criteria
# ===========================================================================

def evaluate_confirmation(aggregate: dict, pearson_r: float,
                           k_values: list[int]) -> dict:
    def status(passed: bool) -> str:
        return 'CONFIRMED' if passed else 'NOT MET'

    k1 = str(k_values[0])
    k_last = str(k_values[-1])
    a1 = aggregate.get(k1, {})
    aL = aggregate.get(k_last, {})

    acc_m_k1 = a1.get('acc_multi', {}).get('mean')
    acc_s_k1 = a1.get('acc_single', {}).get('mean')
    acc_m_kL = aL.get('acc_multi', {}).get('mean')
    vj_k1    = a1.get('vj_mean', {}).get('mean')
    vj_kL    = aL.get('vj_mean', {}).get('mean')

    gap_k1 = (acc_m_k1 - acc_s_k1) if (acc_m_k1 is not None and acc_s_k1 is not None) else None

    return {
        'I1': {'status': status(acc_m_k1 is not None and acc_s_k1 is not None
                                 and acc_m_k1 > acc_s_k1),
               'value': {'acc_multi': acc_m_k1, 'acc_single': acc_s_k1,
                          'gap': gap_k1},
               'criterion': 'acc_multi(k=1) > acc_single(k=1)'},
        'I2': {'status': status(acc_m_k1 is not None and acc_m_k1 > 0.50),
               'value': acc_m_k1,
               'criterion': 'acc_multi(k=1) > 0.50'},
        'I3': {'status': status(gap_k1 is not None and gap_k1 > 0.10),
               'value': gap_k1,
               'criterion': 'gap(k=1) > 0.10'},
        'I4': {'status': status(acc_m_k1 is not None and acc_m_kL is not None
                                 and acc_m_kL < acc_m_k1),
               'value': {'k1': acc_m_k1, 'k_last': acc_m_kL},
               'criterion': f'acc_multi(k={k_last}) < acc_multi(k={k1})'},
        'I5': {'status': status(vj_k1 is not None and vj_kL is not None
                                 and vj_kL > vj_k1),
               'value': {'k1': vj_k1, 'k_last': vj_kL},
               'criterion': f'vj_mean(k={k_last}) > vj_mean(k={k1})'},
        'I6': {'status': status(pearson_r < 0),
               'value': pearson_r,
               'criterion': 'Pearson r(V_j, acc) < 0'},
    }


# ===========================================================================
# 12. Figures
# ===========================================================================

def _save_fig(fig, base: str):
    fig.savefig(os.path.join(RESULTS_DIR, base + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(RESULTS_DIR, base + '.png'),
                 bbox_inches='tight', dpi=120)
    plt.close(fig)


def fig_identification_accuracy(aggregate: dict, k_values: list[int]):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [k for k in k_values if str(k) in aggregate]
    if not ks:
        plt.close(fig)
        return
    s_mean = [aggregate[str(k)]['acc_single']['mean'] for k in ks]
    s_std  = [aggregate[str(k)]['acc_single']['std']  for k in ks]
    m_mean = [aggregate[str(k)]['acc_multi']['mean']  for k in ks]
    m_std  = [aggregate[str(k)]['acc_multi']['std']   for k in ks]
    ax.fill_between(ks, s_mean, m_mean, color='#888', alpha=0.2,
                     label='multi-layer benefit')
    ax.errorbar(ks, s_mean, yerr=s_std, marker='o', color='#1f77b4',
                 capsize=3, label='single-layer')
    ax.errorbar(ks, m_mean, yerr=m_std, marker='s', color='#d62728',
                 capsize=3, label='multi-layer')
    ax.axhline(0.005, color='k', linestyle=':', lw=1.0,
                label='random (1/200)')
    ax.set_xlabel('k (training domains)')
    ax.set_ylabel('Identification accuracy')
    ax.set_title('Syndrome identification accuracy vs k')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    _save_fig(fig, 'fig_identification_accuracy')


def fig_jacobian_variance(aggregate: dict, k_values: list[int]):
    ks = [k for k in k_values if str(k) in aggregate]
    if not ks:
        return
    vj_m = [aggregate[str(k)]['vj_mean']['mean']      for k in ks]
    vj_s = [aggregate[str(k)]['vj_mean']['std']       for k in ks]
    vj_h = [aggregate[str(k)]['vj_high_frac']['mean'] for k in ks]
    vj_he = [aggregate[str(k)]['vj_high_frac']['std'] for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(ks, vj_m, yerr=vj_s, marker='o', color='#1f77b4',
                      capsize=3)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('V_j mean')
    axes[0].set_title('Jacobian variance — mean across directions')
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(ks, vj_h, yerr=vj_he, marker='s', color='#d62728',
                      capsize=3)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel(f'fraction with V_j > {VJ_THRESHOLD}')
    axes[1].set_title('High-variance direction fraction')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig_jacobian_variance')


def fig_vj_vs_accuracy(runs: dict, k_values: list[int],
                        pearson_r: float):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    for i, k in enumerate(k_values):
        k_str = str(k)
        if k_str not in runs:
            continue
        color = cmap(i / max(len(k_values) - 1, 1))
        for seed_str, r in runs[k_str].items():
            vj  = np.asarray(r['vj_per_direction'])
            cor = np.asarray(r['dir_correct_multi'], dtype=np.float64)
            tot = np.asarray(r['dir_total'], dtype=np.float64)
            mask = tot > 0
            if not mask.any():
                continue
            acc = np.zeros_like(tot)
            acc[mask] = cor[mask] / tot[mask]
            ax.scatter(vj[mask], acc[mask], color=color, s=12, alpha=0.5,
                        label=f'k={k}' if seed_str == sorted(runs[k_str])[0] else None)
    ax.set_xlabel('V_j (per direction)')
    ax.set_ylabel('Multi-layer identification accuracy (per direction)')
    ax.set_title(f'V_j vs identification accuracy  (Pearson r = {pearson_r:.3f})')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, 'fig_vj_vs_accuracy')


def fig_confusion(runs: dict, k: int):
    """Average confusion matrices across seeds at one k value."""
    k_str = str(k)
    if k_str not in runs:
        return
    cms_single = []
    cms_multi  = []
    for r in runs[k_str].values():
        if r.get('confusion_single') is not None:
            cms_single.append(np.asarray(r['confusion_single'],
                                          dtype=np.int64))
        if r.get('confusion_multi') is not None:
            cms_multi.append(np.asarray(r['confusion_multi'],
                                         dtype=np.int64))
    if not cms_single or not cms_multi:
        return
    cm_s = np.mean(np.stack(cms_single), axis=0)
    cm_m = np.mean(np.stack(cms_multi),  axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(np.log1p(cm_s), cmap='gray_r', aspect='auto')
    axes[0].set_title(f'Single-layer confusion  (k={k})')
    axes[0].set_xlabel('Predicted dir')
    axes[0].set_ylabel('True dir')
    fig.colorbar(im0, ax=axes[0], label='log1p(count)')
    im1 = axes[1].imshow(np.log1p(cm_m), cmap='gray_r', aspect='auto')
    axes[1].set_title(f'Multi-layer confusion  (k={k})')
    axes[1].set_xlabel('Predicted dir')
    axes[1].set_ylabel('True dir')
    fig.colorbar(im1, ax=axes[1], label='log1p(count)')
    fig.tight_layout()
    _save_fig(fig, f'fig_confusion_k{k}')


# ===========================================================================
# 13. HTML report
# ===========================================================================

def _img_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def write_html_report(meta: dict, aggregate: dict, runs: dict,
                       confirmation: dict, pearson_r: float,
                       k_values: list[int]):
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
             '<title>lstm2_3_detectability</title>',
             """<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: right; }
th { background: #eee; text-align: center; }
td.label { text-align: left; font-weight: 500; }
.confirmed { color: #1a7a1a; font-weight: bold; }
.notmet { color: #b00020; font-weight: bold; }
img { max-width: 100%; border: 1px solid #ddd; margin: 0.5em 0; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
</style></head><body>"""]
    parts.append('<h1>lstm2_3_detectability</h1>')
    parts.append(f"<p>{meta['date']} | Device: {meta['device']} | "
                  f"Mode: {'QUICK' if meta['quick_mode'] else 'FULL'}<br>"
                  f"k values: {k_values} | seeds: {len(meta['seeds'])} | "
                  f"perturbation: lstm_layers[0].weight_ih_l0<br>"
                  f"N_DIRS = {meta['config']['n_dirs']}, "
                  f"N_TEST = {meta['config']['n_test']}, "
                  f"EPS_DICT = {meta['config']['eps_dict']}</p>")

    # Cross-validation per seed
    parts.append('<h2>Cross-validation (k=1)</h2><table><tr>'
                  '<th>seed</th><th>CS_gap</th><th>status</th></tr>')
    if '1' in runs:
        for seed_str, r in sorted(runs['1'].items(), key=lambda x: int(x[0])):
            cv_ok = r.get('cv_ok')
            cls = 'confirmed' if cv_ok else ('notmet' if cv_ok is False else '')
            label = 'OK' if cv_ok else ('FAIL' if cv_ok is False else 'N/A')
            cs = r.get('cv_cs_gap')
            cs_s = f"{cs:.3f}" if cs is not None else '—'
            parts.append(f"<tr><td>{seed_str}</td><td>{cs_s}</td>"
                          f"<td class='{cls}'>{label}</td></tr>")
    parts.append('</table>')

    # Confirmation
    parts.append('<h2>Confirmation</h2><table><tr><th>ID</th>'
                  '<th>Criterion</th><th>Status</th><th>Value</th></tr>')
    for cid in sorted(confirmation.keys()):
        c = confirmation[cid]
        cls = 'confirmed' if c['status'] == 'CONFIRMED' else 'notmet'
        parts.append(f"<tr><td class='label'>{cid}</td>"
                      f"<td class='label'>{c['criterion']}</td>"
                      f"<td class='{cls}'>{c['status']}</td>"
                      f"<td class='label'>{c['value']}</td></tr>")
    parts.append('</table>')

    # Aggregate table
    parts.append('<h2>Aggregate</h2><table><tr>'
                  '<th>k</th><th>acc_single</th><th>acc_multi</th>'
                  '<th>gap</th><th>vj_mean</th>'
                  '<th>vj_high_frac</th></tr>')
    for k in k_values:
        a = aggregate.get(str(k))
        if not a:
            continue
        gap = a['multi_minus_single']
        parts.append(
            f"<tr><td>{k}</td>"
            f"<td>{a['acc_single']['mean']:.3f} ± "
            f"{a['acc_single']['std']:.3f}</td>"
            f"<td>{a['acc_multi']['mean']:.3f} ± "
            f"{a['acc_multi']['std']:.3f}</td>"
            f"<td>{gap:+.3f}</td>"
            f"<td>{a['vj_mean']['mean']:.4f} ± "
            f"{a['vj_mean']['std']:.4f}</td>"
            f"<td>{a['vj_high_frac']['mean']:.2f} ± "
            f"{a['vj_high_frac']['std']:.2f}</td></tr>")
    parts.append('</table>')
    parts.append(f'<p>Pearson r(V_j, accuracy) = {pearson_r:.4f}</p>')

    # Figures
    fig_specs = [
        ('Identification accuracy vs k', 'fig_identification_accuracy'),
        ('Jacobian variance vs k', 'fig_jacobian_variance'),
        ('V_j vs accuracy', 'fig_vj_vs_accuracy'),
    ]
    for k in CONFUSION_K:
        fig_specs.append((f'Confusion matrices (k={k})', f'fig_confusion_k{k}'))
    for title, base in fig_specs:
        png = os.path.join(RESULTS_DIR, base + '.png')
        if os.path.exists(png):
            b64 = _img_b64(png)
            parts.append(f"<h2>{title}</h2>"
                          f"<img src='data:image/png;base64,{b64}'>")

    # Convergence failures
    parts.append('<h2>Convergence failures</h2>')
    fails = [(k, s, r['final_loss']) for k, sd in runs.items()
             for s, r in sd.items() if not r.get('converged', True)]
    if fails:
        parts.append('<table><tr><th>k</th><th>seed</th><th>loss</th></tr>')
        for k, s, l in fails:
            parts.append(f"<tr><td>{k}</td><td>{s}</td><td>{l:.3f}</td></tr>")
        parts.append('</table>')
    else:
        parts.append('<p>None.</p>')
    parts.append('</body></html>')

    out = os.path.join(RESULTS_DIR, f'{SCRIPT_NAME}_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ===========================================================================
# 14. RESULTS SUMMARY
# ===========================================================================

def print_results_summary(meta: dict, aggregate: dict, runs: dict,
                           confirmation: dict, pearson_r: float,
                           k_values: list[int]):
    bar = '=' * 55
    print()
    print(bar)
    print('RESULTS SUMMARY — lstm2_3_detectability')
    print(bar)
    print(f"Date:    {meta['date']}")
    print(f"Device:  {meta['device']}")
    print(f"Mode:    {'QUICK' if meta['quick_mode'] else 'FULL'}")
    print(f"k values: {k_values}")
    print(f"Seeds:   {meta['seeds']}")
    print('Perturbation target: lstm_layers[0].weight_ih_l0')
    print(f"N_DIRS: {meta['config']['n_dirs']}   "
          f"N_TEST: {meta['config']['n_test']}   "
          f"EPS_DICT: {meta['config']['eps_dict']}")

    print()
    print('-- CROSS-VALIDATION (k=1) ---------------------------------')
    print(f"{'seed':<8}{'CS_gap':<10}{'status'}")
    if '1' in runs:
        for seed_str, r in sorted(runs['1'].items(), key=lambda x: int(x[0])):
            cs = r.get('cv_cs_gap')
            cv_ok = r.get('cv_ok')
            cs_s = f"{cs:.3f}" if cs is not None else '—'
            status = 'OK' if cv_ok else ('FAIL' if cv_ok is False else 'N/A')
            print(f"{seed_str:<8}{cs_s:<10}{status}")

    print()
    print('-- AGGREGATE TABLE (mean ± std across seeds) --------------')
    head = f"{'k':<5}{'acc_single':<18}{'acc_multi':<18}{'gap':<10}{'vj_mean':<14}{'vj_high_frac'}"
    print(head)
    for k in k_values:
        a = aggregate.get(str(k))
        if not a:
            continue
        gap = a['multi_minus_single']
        print(f"{k:<5}"
              f"{a['acc_single']['mean']:.3f}±{a['acc_single']['std']:.3f}    "
              f"{a['acc_multi']['mean']:.3f}±{a['acc_multi']['std']:.3f}    "
              f"{gap:+.3f}    "
              f"{a['vj_mean']['mean']:.4f}±{a['vj_mean']['std']:.4f}  "
              f"{a['vj_high_frac']['mean']:.2f}±{a['vj_high_frac']['std']:.2f}")

    print()
    print('-- CONFIRMATION -------------------------------------------')
    c = confirmation
    def _f(v, fmt='.3f'):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)
    i1 = c['I1']; i2 = c['I2']; i3 = c['I3']
    i4 = c['I4']; i5 = c['I5']; i6 = c['I6']
    print(f"I1  acc_multi(k=1) > acc_single(k=1)         "
          f"[{i1['status']}]  gap={_f(i1['value']['gap'])}")
    print(f"I2  acc_multi(k=1) > 0.50                    "
          f"[{i2['status']}]  value={_f(i2['value'])}")
    print(f"I3  gap(k=1) > 0.10                          "
          f"[{i3['status']}]  value={_f(i3['value'])}")
    print(f"I4  acc_multi(k_last) < acc_multi(k=1)       "
          f"[{i4['status']}]  k1={_f(i4['value']['k1'])}  "
          f"k_last={_f(i4['value']['k_last'])}")
    print(f"I5  vj_mean(k_last) > vj_mean(k=1)           "
          f"[{i5['status']}]  k1={_f(i5['value']['k1'], '.4f')}  "
          f"k_last={_f(i5['value']['k_last'], '.4f')}")
    print(f"I6  Pearson r(V_j, acc) < 0                  "
          f"[{i6['status']}]  r={pearson_r:.4f}")

    print()
    print('-- CONVERGENCE FAILURES -----------------------------------')
    fails = [(k, s, r['final_loss']) for k, sd in runs.items()
             for s, r in sd.items() if not r.get('converged', True)]
    if fails:
        for k, s, l in fails:
            print(f"  k={k} seed={s}  loss={l:.3f}")
    else:
        print('  None.')

    print()
    print('-- OVERALL ------------------------------------------------')
    all_met = all(v['status'] == 'CONFIRMED' for v in confirmation.values())
    n_models = sum(1 for f in os.listdir(MODELS_DIR)
                    if f.endswith('.pt'))
    print(f"All confirmations met: {'YES' if all_met else 'NO'}")
    print(f"Models saved: {MODELS_DIR}/  ({n_models} files)")
    print(f"Output: results/{SCRIPT_NAME}/")
    print(bar)


# ===========================================================================
# 15. Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true',
                   help='Quick mode: reduced k, seeds, dirs, test trials')
    return p.parse_args()


def main():
    args = parse_args()
    quick = args.quick

    if quick:
        k_values = K_VALUES_QUICK
        seeds    = SEEDS_QUICK
        n_dirs   = N_DIRS_QUICK
        n_test   = N_TEST_QUICK
        print('[QUICK MODE]')
    else:
        k_values = K_VALUES
        seeds    = SEEDS
        n_dirs   = N_DIRS
        n_test   = N_TEST

    device = get_device()
    print(f"Script: {SCRIPT_NAME}")
    print(f"k values: {k_values}")
    print(f"Seeds: {seeds}")
    print(f"N_DIRS={n_dirs}  N_TEST={n_test}  EPS_DICT={EPS_DICT}")

    partial_path = os.path.join(
        RESULTS_DIR,
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
                print(f"k={k:<3}seed={seed:<6}[resume: already done]")
                continue
            record = run_one_cell(k, seed, n_dirs, n_test, device, quick)
            runs[str(k)][str(seed)] = record
            try:
                save_json(runs, partial_path)
            except Exception as e:
                print(f"  [warn] could not save partial: {e!r}")

    # Aggregate and confirm
    aggregate = aggregate_runs(runs, k_values)
    pearson_r = compute_pearson_vj_acc(runs)
    confirmation = evaluate_confirmation(aggregate, pearson_r, k_values)

    meta = {
        'script': SCRIPT_NAME,
        'date': datetime.datetime.now().isoformat(),
        'device': str(device),
        'k_values': k_values,
        'seeds': seeds,
        'quick_mode': quick,
        'config': {
            'n_dirs': n_dirs, 'n_probe': N_PROBE, 'eps_dict': EPS_DICT,
            'n_test': n_test, 'eps_min': EPS_MIN, 'eps_max': EPS_MAX,
            'perturbation_target': 'lstm_layers[0].weight_ih_l0',
            'seqs_per_domain': SEQS_PER_DOMAIN, 'epochs': EPOCHS,
        },
    }
    output = {
        'meta': meta,
        'runs': runs,
        'aggregate': aggregate,
        'pearson_r_vj_acc': pearson_r,
        'confirmation': confirmation,
    }
    save_json(output,
              os.path.join(RESULTS_DIR, f'{SCRIPT_NAME}_results.json'))

    # Figures
    fig_identification_accuracy(aggregate, k_values)
    fig_jacobian_variance(aggregate, k_values)
    fig_vj_vs_accuracy(runs, k_values, pearson_r)
    for k in CONFUSION_K:
        if k in k_values:
            fig_confusion(runs, k)

    write_html_report(meta, aggregate, runs, confirmation, pearson_r,
                       k_values)
    print_results_summary(meta, aggregate, runs, confirmation, pearson_r,
                           k_values)


if __name__ == '__main__':
    main()
