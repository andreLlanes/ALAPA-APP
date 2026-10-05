""" Shared training loop for the LSTM and GNN: teacher-forcing decay, early stopping once it
    reaches 0, learning-rate decay, optional MMD alignment, and resumable checkpoints.
"""

import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import config
from utils import runlog
from utils.progress import load_checkpoint, save_checkpoint

def seed_everything(seed: int):
    """ Seed Python, NumPy and torch (weights, dropout); optionally force deterministic GPU kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: an op with no deterministic GPU version warns instead of stopping the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

def to_device(batch, device):
    """ Turn a NumPy batch from data/batch.py into model inputs on the device; ids stay NumPy.
    """
    x_enc, x_dec, y, *graph, ids = batch
    inputs = {k: torch.from_numpy(v).to(device, non_blocking=True)
              for k, v in (("x_enc", x_enc), ("x_dec", x_dec), ("y", y))}
    if graph:
        inputs["edge_index"] = torch.from_numpy(graph[0]).long().to(device)
        inputs["edge_weight"] = torch.from_numpy(graph[1]).to(device)
    return inputs, ids

def teacher_forcing(epoch: int, autoregressive: bool) -> float:
    """ Return the share of steps fed the true previous hour: 1.0 falling by TF_DECAY_RATE per epoch.
    """
    if not autoregressive:
        return 0.0
    return max(0.0, round(1.0 - config.TF_DECAY_RATE * epoch, 10))

def evaluate(model, batches, device) -> float:
    """ Compute RMSE (normalized units) over all forecast hours, with no teacher forcing.
    """
    model.eval()
    sq, n = torch.zeros((), device=device), 0
    with torch.no_grad():
        for batch in batches():
            b, _ = to_device(batch, device)
            pred, _ = model(**b)
            sq += ((pred - b["y"]) ** 2).sum()
            n += b["y"].numel()
    return math.sqrt(sq.item() / max(n, 1))

def predict(model, batches, device):
    """ Forecast every window of the batches; return (predictions [n,72] normalized, window ids).
    """
    model.eval()
    preds, ids = [], []
    with torch.no_grad():
        for batch in batches():
            b, batch_ids = to_device(batch, device)
            preds.append(model(**b)[0].cpu().numpy())
            ids.append(batch_ids)
    return np.concatenate(preds), np.concatenate(ids)

def _target_reps(model, target_batches, rng, device):
    """ Yield encoder representations of target batches forever (inputs only, never targets).
    """
    while True:
        for batch in target_batches(rng):
            b, _ = to_device(batch, device)
            yield model.encode(**b)[-1]

def fit(model, train_batches, val_batches, lr, seed, device="cpu", align=None, ckpt=None,
        verbose=True, decay=True, label="TRAINING"):
    """ Train with early stopping and restore the best epoch; return the history.
        train_batches(rng) / val_batches() yield NumPy batches. align = (MMDAlignment,
        target_batches) adds lambda * MMD between source and target encoder representations.
        ckpt is a checkpoint path: rerunning resumes, and a finished run returns immediately.
        Aligned runs are not checkpointed: their target-batch order cannot be resumed exactly.
        verbose=False hides the epoch lines (used while tuning). decay=False keeps teacher forcing
        at 0 from the first epoch (fine-tuning a pretrained decoder). The full history is written
        to the run log under `label`.
    """
    if align:
        ckpt = None
    autoregressive = model.autoregressive and decay
    if autoregressive and math.ceil(1 / config.TF_DECAY_RATE) >= config.MAX_EPOCHS:
        raise ValueError("MAX_EPOCHS must exceed the epochs teacher forcing needs to reach 0")
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    # factor 1.0 means no decay (ReduceLROnPlateau itself rejects it).
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=config.LR_DECAY_FACTOR, patience=config.LR_DECAY_PATIENCE)
        if config.LR_DECAY_FACTOR < 1.0 else None)
    rng = np.random.default_rng(seed)               # batch order
    gen = torch.Generator().manual_seed(seed)       # teacher-forcing flips, drawn on the CPU
    mmd = align[0].to(device) if align else None
    state = {"epoch": 0, "best": math.inf, "best_state": None, "bad": 0, "history": [],
             "done": False}

    saved = load_checkpoint(ckpt)
    if saved:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        if sched:
            sched.load_state_dict(saved["sched"])
        if mmd is not None:
            mmd.load_state_dict(saved["mmd"])
        rng.bit_generator.state = saved["rng"]
        gen.set_state(saved["gen"])
        state = saved["state"]
        if verbose:
            print(f"    Resumed at epoch {state['epoch']}")

    targets = _target_reps(model, align[1], rng, device) if align else None
    while not state["done"] and state["epoch"] < config.MAX_EPOCHS:
        epoch = state["epoch"]
        tf = teacher_forcing(epoch, autoregressive)
        model.train()
        sq, n, mmd_sum, mmd_n = torch.zeros((), device=device), 0, 0.0, 0
        for batch in train_batches(rng):
            b, _ = to_device(batch, device)
            teacher = (torch.rand(b["y"].shape, generator=gen) < tf).to(device) if tf > 0 else None
            pred, rep = model(**b, teacher=teacher)
            loss = F.mse_loss(pred, b["y"])  # every forecast hour weighted equally
            if targets is not None:
                target_rep = next(targets)
                if len(rep) > 1 and len(target_rep) > 1:  # the unbiased MMD needs 2 rows a side
                    term, raw = mmd(rep, target_rep)
                    loss = loss + term
                    mmd_sum, mmd_n = mmd_sum + raw.item(), mmd_n + 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sq += ((pred.detach() - b["y"]) ** 2).sum()
            n += b["y"].numel()

        train = math.sqrt(sq.item() / max(n, 1))
        val = evaluate(model, val_batches, device)
        # Early stopping and LR decay only start once teacher forcing has reached 0.
        if tf == 0:
            if val < state["best"]:
                state.update(best=val, best_state=copy.deepcopy(model.state_dict()), bad=0)
            else:
                state["bad"] += 1
            if sched:
                sched.step(val)
        state["history"].append({"epoch": epoch + 1, "train": train, "val": val, "tf": tf,
                                 "lr": opt.param_groups[0]["lr"],
                                 "mmd2": mmd_sum / mmd_n if mmd_n else float("nan")})
        line = f"    EPOCH {epoch + 1}/{config.MAX_EPOCHS}: train = {train:.4f}, val = {val:.4f}"
        # The no-improve counter only exists once teacher forcing has reached 0.
        if verbose:
            print(line + (f" (no improve: {state['bad']}/{config.PATIENCE})" if tf == 0 else ""))
        state["epoch"] = epoch + 1
        state["done"] = state["bad"] >= config.PATIENCE
        if ckpt:
            save_checkpoint(ckpt, {"model": model.state_dict(), "opt": opt.state_dict(),
                                   "sched": sched.state_dict() if sched else None,
                                   "mmd": mmd.state_dict() if mmd is not None else None,
                                   "rng": rng.bit_generator.state, "gen": gen.get_state(),
                                   "state": state})

    model.load_state_dict(state["best_state"])
    best = min((h for h in state["history"] if h["tf"] == 0), key=lambda h: h["val"])
    runlog.detail(label, {
        "best epoch": f"{best['epoch']} (val {best['val']:.4f})",
        "epochs run": f"{len(state['history'])} of {config.MAX_EPOCHS}"
                      + (" (early stop)" if state["done"] else ""),
        "learning rate": lr, "seed": seed,
        "trainable parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "mmd lambda / warmup": f"{mmd.lambda_} / {mmd.warmup_steps}" if mmd is not None else "none"},
        state["history"])
    return state["history"]
