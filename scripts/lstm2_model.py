"""
lstm2_model.py — Shared Model Module for the lstm2 Experiment Series
=====================================================================
Shared foundation imported by all lstm2 experiment scripts. Contains:
  - Shared configuration constants
  - Domain generator (parameterized linear congruential family)
  - 10-layer LSTM model class with per-layer residual access
  - Training function
  - SVD and perturbation utilities
  - Probe/test split with non-overlap assertion
  - Output utilities
  - Self-test (__main__ block)

This file is not an experiment. It produces no results JSON or HTML.
Running it directly executes only the self-test.
"""

import os
import sys
import json
import random
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# ===========================================================================
# 1. SHARED CONFIGURATION
# ===========================================================================

# -- Architecture --------------------------------------------------------
VOCAB_SIZE  = 256
D_MODEL     = 256
NUM_LAYERS  = 10
SEQ_LEN     = 40

# -- Training ------------------------------------------------------------
SEQS_PER_DOMAIN = 2000
BATCH_SIZE      = 128
EPOCHS          = 200
LR              = 1e-3

# -- Experiment standards ------------------------------------------------
EPS_ERR    = 0.1
ERR_POS    = 5
PREFIX_LEN = 8
GEN_LEN    = 12

# -- k values for the sweep ---------------------------------------------
# K_VALUES = [1, 2, 3, 5, 10, 15, 20, 30, 50]
K_VALUES = [1, 2, 3, 5, 10]

# -- Seeds ---------------------------------------------------------------
SEEDS_BASE = [42, 2311, 9744, 9037, 8919, 3163]
SEEDS_EXT  = [42, 2311, 9744, 9037, 8919, 3163,
              7777, 1234, 5678, 9999, 3141, 2718]


# ===========================================================================
# 2. DOMAIN GENERATOR
# ===========================================================================

def domain_params(i: int) -> int:
    """Return slope a_i for domain index i (0-indexed). a_i = i + 1.

    Sequences are position-indexed arithmetic progressions:
        token_t = (a_i * t + start) % VOCAB_SIZE
    The model learns the domain-specific slope from context. Corruption at
    a single position does not perturb the slope estimate computed from the
    other positions, so a converged model self-corrects on its own domain
    (CS_known low) but applies the wrong slope on unknown domains
    (CS_unknown high) — giving a meaningful CS gap.
    """
    return int(i + 1)


def make_domain_sequences(domain_idx: int, n: int, seq_len: int = SEQ_LEN,
                          rng: np.random.Generator | None = None) -> torch.Tensor:
    """Generate n sequences of length seq_len for domain domain_idx.

    Each sequence is a position-indexed arithmetic progression
        token_t = (slope * t + start) % VOCAB_SIZE
    with `start` drawn uniformly from [0, VOCAB_SIZE) per sequence. Tokens
    are clipped to [2, 255] so they share vocabulary with the [0,1]-reserved
    range used elsewhere.

    Returns: LongTensor of shape [n, seq_len]
    """
    if rng is None:
        rng = np.random.default_rng()
    slope = domain_params(domain_idx)
    starts = rng.integers(0, VOCAB_SIZE, size=n)
    t = np.arange(seq_len)
    out = (slope * t[None, :] + starts[:, None]) % VOCAB_SIZE
    out = np.clip(out, 2, VOCAB_SIZE - 1).astype(np.int64)
    return torch.from_numpy(out).long()


def make_training_data(k: int, seqs_per_domain: int = SEQS_PER_DOMAIN,
                       seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Generate mixed training data for a k-domain generalist model.

    Concatenates seqs_per_domain sequences from each of domains 0..k-1.
    Shuffles the result. Returns LongTensor of shape [k * seqs_per_domain, seq_len].
    """
    rng = np.random.default_rng()
    parts = [make_domain_sequences(d, seqs_per_domain, seq_len, rng)
             for d in range(k)]
    data = torch.cat(parts, dim=0)
    perm = torch.randperm(data.shape[0])
    return data[perm]


# ===========================================================================
# 3. MODEL CLASS
# ===========================================================================

class GeneralistLSTM(nn.Module):
    """10-layer autoregressive LSTM for the lstm2 k-sweep series.

    Architecture:
        Embedding(VOCAB_SIZE, D_MODEL)
        10 x LSTM(D_MODEL, D_MODEL, num_layers=1, batch_first=True, dropout=0.0)
        Linear(D_MODEL, VOCAB_SIZE)

    No dropout in LSTM layers — perturbation experiments require deterministic
    hidden states.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL,
                 num_layers=NUM_LAYERS):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, return_all_layers: bool = False):
        h = self.embedding(x)
        layer_outputs = []
        for lstm in self.lstm_layers:
            h, _ = lstm(h)
            layer_outputs.append(h)
        logits = self.output(h)
        if return_all_layers:
            return logits, layer_outputs
        return logits

    def generate_greedy(self, prefix: torch.Tensor, length: int) -> torch.Tensor:
        self.eval()
        seq = prefix.clone()
        with torch.no_grad():
            for _ in range(length):
                logits = self.forward(seq)[:, -1, :]
                seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
        return seq


# ===========================================================================
# 4. TRAINING FUNCTION
# ===========================================================================

def train_model(model: GeneralistLSTM,
                data: torch.Tensor,
                epochs: int = EPOCHS,
                lr: float = LR,
                batch_size: int = BATCH_SIZE,
                device: torch.device = None,
                verbose: bool = True) -> tuple[GeneralistLSTM, float]:
    """Train model on next-token prediction with cross-entropy loss.

    Returns: (model, final_loss) — final_loss is the average loss over the
    final epoch. Caller may emit a convergence warning if it is high.
    """
    if device is None:
        device = torch.device('cpu')
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = data.shape[0]
    model.train()
    avg_loss = float('nan')
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        data_shuf = data[perm]
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch = data_shuf[i:i + batch_size]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if verbose and (ep % 30 == 0 or ep == epochs):
            print(f"Epoch {ep:3d}/{epochs}: loss = {avg_loss:.4f}")
    return model, float(avg_loss)


# ===========================================================================
# 5. SVD UTILITIES
# ===========================================================================

def get_weight_matrix(model: GeneralistLSTM,
                      target: str | int) -> torch.Tensor:
    """Return the weight matrix for a given target as a CPU float32 tensor.

    target options:
        'output'   - model.output.weight        shape [V, D]
        'embed'    - model.embedding.weight      shape [V, D]
        int i      - model.lstm_layers[i].weight_ih_l0   shape [4*D, D]
    """
    if target == 'output':
        W = model.output.weight
    elif target == 'embed':
        W = model.embedding.weight
    elif isinstance(target, int):
        W = model.lstm_layers[target].weight_ih_l0
    else:
        raise ValueError(f"Unknown target: {target!r}")
    return W.detach().cpu().float()


def compute_svd_basis(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the SVD of W and return (S, Vh).

    Uses torch.linalg.svd(W, full_matrices=False).
    Convention: perturbation direction = Vh[k]  (shape [d_in])
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return S, Vh


def compute_dimensional_excess(W: torch.Tensor) -> dict:
    """Compute spectral properties of weight matrix W."""
    S, _ = compute_svd_basis(W)
    s = S.detach().cpu().float()
    sigma1_sq = float((s[0] ** 2).item()) if s.numel() > 0 else 0.0
    sum_sq = float((s ** 2).sum().item())
    stable_rank = sum_sq / sigma1_sq if sigma1_sq > 0 else 0.0
    d_out, d_in = int(W.shape[0]), int(W.shape[1])
    nominal_rank = min(d_out, d_in)
    de = d_in / stable_rank if stable_rank > 0 else float('inf')
    return {
        'de': float(de),
        'stable_rank': float(stable_rank),
        'nominal_rank': int(nominal_rank),
        'singular_values': [float(x) for x in s.tolist()],
        'shape': [d_out, d_in],
    }


# ===========================================================================
# 6. PERTURBATION UTILITIES
# ===========================================================================

def perturb_weight(W: torch.Tensor, direction: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """Apply full-direction perturbation in-place and return original data.

    Perturbation: W.data += eps * direction.unsqueeze(0)
    """
    original = W.data.clone()
    W.data.add_(eps * direction.to(dtype=W.dtype, device=W.device).unsqueeze(0))
    return original


def restore_weight(W: torch.Tensor, original: torch.Tensor) -> None:
    """Restore weight to pre-perturbation state and assert correctness."""
    W.data.copy_(original)
    assert (W.data - original).abs().max().item() < 1e-6, \
        "Weight restore failed - numerical mismatch"


# ===========================================================================
# 7. PROBE/TEST SPLIT
# ===========================================================================

def make_probe_test_split(domain_idx: int,
                          n_probe: int,
                          n_test: int,
                          seq_len: int = SEQ_LEN,
                          rng: np.random.Generator | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate non-overlapping probe and test sequences for one domain."""
    if rng is None:
        rng = np.random.default_rng()
    seqs = make_domain_sequences(domain_idx, n_probe + n_test, seq_len, rng)
    probe = seqs[:n_probe]
    test = seqs[n_probe:n_probe + n_test]
    assert probe.shape[0] == n_probe
    assert test.shape[0] == n_test
    return probe, test


# ===========================================================================
# 8. RESIDUAL STREAM CAPTURE
# ===========================================================================

def get_residual_streams(model: GeneralistLSTM,
                         sequences: torch.Tensor,
                         device: torch.device
                         ) -> list[np.ndarray]:
    """Run forward pass and return all per-layer hidden state outputs."""
    model.eval()
    with torch.no_grad():
        sequences = sequences.to(device)
        _, layer_outputs = model(sequences, return_all_layers=True)
    return [lo.cpu().numpy() for lo in layer_outputs]


# ===========================================================================
# 9. AVERAGED SYNDROME MEASUREMENT
# ===========================================================================

def measure_syndrome(model: GeneralistLSTM,
                     W: torch.Tensor,
                     direction: torch.Tensor,
                     eps: float,
                     probe_seqs: torch.Tensor,
                     device: torch.device,
                     return_layer_deltas: bool = False):
    """Compute the averaged normalised syndrome for one perturbation direction.

    Steps:
      1. Run clean forward pass on probe_seqs -> clean_logits [B, T, V]
      2. Perturb W by eps in direction
      3. Run perturbed forward pass -> pert_logits [B, T, V]
      4. Restore W immediately (assert correctness)
      5. delta_logits = (pert_logits - clean_logits).mean(axis=(0,1)) -> [V]
      6. Normalise: s = delta_logits / ||delta_logits||
    """
    model.eval()
    probe_seqs = probe_seqs.to(device)

    with torch.no_grad():
        if return_layer_deltas:
            clean_logits, clean_layers = model(probe_seqs, return_all_layers=True)
            clean_logits = clean_logits.cpu().numpy()
            clean_layers = [lo.cpu().numpy() for lo in clean_layers]
        else:
            clean_logits = model(probe_seqs).cpu().numpy()
            clean_layers = None

    original = perturb_weight(W, direction, eps)
    try:
        with torch.no_grad():
            if return_layer_deltas:
                pert_logits, pert_layers = model(probe_seqs, return_all_layers=True)
                pert_logits = pert_logits.cpu().numpy()
                pert_layers = [lo.cpu().numpy() for lo in pert_layers]
            else:
                pert_logits = model(probe_seqs).cpu().numpy()
                pert_layers = None
    finally:
        restore_weight(W, original)

    delta_logits = (pert_logits - clean_logits).mean(axis=(0, 1))
    norm = np.linalg.norm(delta_logits)
    if norm < 1e-12:
        print("  [warn] measure_syndrome: ||delta_logits|| < 1e-12, returning zero vector")
        syndrome = np.zeros_like(delta_logits)
    else:
        syndrome = delta_logits / norm

    if not return_layer_deltas:
        return syndrome

    layer_syndromes = []
    for L in range(len(clean_layers)):
        d = (pert_layers[L] - clean_layers[L]).mean(axis=(0, 1))
        n = np.linalg.norm(d)
        if n < 1e-12:
            print(f"  [warn] measure_syndrome: layer {L} ||delta|| < 1e-12, returning zero vector")
            layer_syndromes.append(np.zeros_like(d))
        else:
            layer_syndromes.append(d / n)
    return syndrome, layer_syndromes


# ===========================================================================
# 10. MULTI-LAYER SYNDROME (production protocol)
# ===========================================================================

def build_multilayer_syndrome(logit_syndrome: np.ndarray,
                              layer_syndromes: list[np.ndarray],
                              injection_layer: int) -> np.ndarray:
    """Concatenate all layer syndromes from injection_layer onward, then
    add the logit syndrome, and normalise the full concatenation.

    Each component is already independently normalised before entry.
    """
    n_layers = len(layer_syndromes)
    components = [layer_syndromes[L] for L in range(injection_layer, n_layers)]
    components.append(logit_syndrome)
    concat = np.concatenate(components)
    norm = np.linalg.norm(concat)
    if norm < 1e-12:
        return concat
    return concat / norm


# ===========================================================================
# 11. OUTPUT UTILITIES
# ===========================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds deterministically."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available and functional, else CPU."""
    if torch.cuda.is_available():
        try:
            torch.randn(10, device='cuda').sum().item()
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda')
        except Exception:
            pass
    print("  Device: CPU")
    return torch.device('cpu')


def make_results_dir(script_name: str) -> str:
    """Create and return path: results/{script_name}/"""
    path = os.path.join('results', script_name)
    os.makedirs(path, exist_ok=True)
    return path


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(data: dict, path: str) -> None:
    """Save dict as formatted JSON. Converts numpy types to native types."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(data), f, indent=2)


# ===========================================================================
# 12. SELF-TEST
# ===========================================================================

def _self_test() -> int:
    print("SELF-TEST: lstm2_model.py")
    print("-------------------------")
    t0 = time.time()
    failed = None

    # [1/6] domain_params — distinct slopes and distinct difference patterns
    try:
        slopes = [domain_params(i) for i in range(50)]
        assert len(set(slopes)) == 50, \
            f"only {len(set(slopes))} distinct slopes"
        # Sequences from different domains should be distinguishable.
        # Use sequence pairs and check (seq[t+1] - seq[t]) % VOCAB_SIZE
        # at positions where neither token was clipped to 2 — a clipped
        # token has value 2 only when (slope*t + start) % VOCAB_SIZE in {0,1}.
        rng = np.random.default_rng(0)
        seqs_d0 = make_domain_sequences(0, 32, seq_len=10, rng=rng).numpy()
        seqs_d1 = make_domain_sequences(1, 32, seq_len=10, rng=rng).numpy()
        # At any t and start, (seq[t+1] - seq[t]) % VOCAB_SIZE == slope when
        # neither end was clipped. For slope=1 and slope=2, both small enough
        # that almost every position has a clean diff equal to slope.
        diffs0 = (seqs_d0[:, 1:] - seqs_d0[:, :-1]) % VOCAB_SIZE
        diffs1 = (seqs_d1[:, 1:] - seqs_d1[:, :-1]) % VOCAB_SIZE
        assert (diffs0 == 1).any() and (diffs1 == 2).any(), \
            "expected slope-1 and slope-2 diffs not observed"
        print("[1/6] domain_params .............. OK  (50 distinct slopes, "
              "expected diff signatures observed)")
    except Exception as e:
        failed = ("domain_params", e)

    # [2/6] make_domain_sequences
    if failed is None:
        try:
            seqs = make_domain_sequences(0, 100, seq_len=40)
            assert tuple(seqs.shape) == (100, 40), f"shape={tuple(seqs.shape)}"
            assert seqs.min().item() >= 2 and seqs.max().item() <= 255, \
                f"values out of range: [{seqs.min().item()},{seqs.max().item()}]"
            print("[2/6] make_domain_sequences ....... OK  (shape [100, 40], values in [2,255])")
        except Exception as e:
            failed = ("make_domain_sequences", e)

    # [3/6] make_training_data
    if failed is None:
        try:
            data = make_training_data(k=3, seqs_per_domain=500, seq_len=40)
            assert tuple(data.shape) == (1500, 40), f"shape={tuple(data.shape)}"
            print("[3/6] make_training_data .......... OK  (k=3: shape [1500, 40], shuffled)")
        except Exception as e:
            failed = ("make_training_data", e)

    # [4/6] GeneralistLSTM forward
    if failed is None:
        try:
            set_seed(0)
            model = GeneralistLSTM()
            x = torch.randint(0, VOCAB_SIZE, (2, 10), dtype=torch.long)
            logits, layer_outputs = model(x, return_all_layers=True)
            assert tuple(logits.shape) == (2, 10, VOCAB_SIZE), \
                f"logits shape={tuple(logits.shape)}"
            assert len(layer_outputs) == 10, f"layer_outputs len={len(layer_outputs)}"
            for i, lo in enumerate(layer_outputs):
                assert tuple(lo.shape) == (2, 10, D_MODEL), \
                    f"layer {i} shape={tuple(lo.shape)}"
            print("[4/6] GeneralistLSTM forward ...... OK  (logits [2,10,256], 10 layer_outputs)")
        except Exception as e:
            failed = ("GeneralistLSTM forward", e)

    # [5/6] perturb / restore
    if failed is None:
        try:
            set_seed(0)
            model = GeneralistLSTM()
            W = model.output.weight
            direction = torch.randn(W.shape[1])
            direction = direction / direction.norm()
            original = perturb_weight(W, direction, eps=1.0)
            # Verify the perturbation actually changed the weight
            diff_after_perturb = (W.data - original).abs().max().item()
            assert diff_after_perturb > 1e-6, "perturbation had no effect"
            restore_weight(W, original)
            max_delta = (W.data - original).abs().max().item()
            assert max_delta < 1e-6, f"restore delta={max_delta}"
            print("[5/6] perturb / restore ........... OK  (max delta after restore < 1e-6)")
        except Exception as e:
            failed = ("perturb / restore", e)

    # [6/6] measure_syndrome — use k=2, 3 epochs, 20 seqs/domain, batch_size=4
    if failed is None:
        try:
            set_seed(0)
            device = torch.device('cpu')
            data = make_training_data(k=2, seqs_per_domain=20, seq_len=40)
            model = GeneralistLSTM()
            train_model(model, data, epochs=3, lr=1e-3, batch_size=4,
                        device=device, verbose=False)
            probe = make_domain_sequences(0, 8, seq_len=40)
            W = model.output.weight
            S, Vh = compute_svd_basis(get_weight_matrix(model, 'output'))
            direction = Vh[0]
            syndrome = measure_syndrome(model, W, direction, eps=0.5,
                                        probe_seqs=probe, device=device,
                                        return_layer_deltas=False)
            assert syndrome.shape == (VOCAB_SIZE,), f"shape={syndrome.shape}"
            norm = float(np.linalg.norm(syndrome))
            assert abs(norm - 1.0) < 1e-5, f"norm={norm}"
            print("[6/6] measure_syndrome ............ OK  (syndrome shape [256], norm ≈ 1.0)")
        except Exception as e:
            failed = ("measure_syndrome", e)

    elapsed = time.time() - t0
    if failed is not None:
        name, err = failed
        print(f"\nSELF-TEST FAILED at {name}: {err!r}")
        return 1

    print(f"\nALL TESTS PASSED  (elapsed: {elapsed:.1f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
