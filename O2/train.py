""" Tune, train and score one configuration for every seed (and every block under ablation).
    PYTHONPATH=.. python train.py --model lstm --city mm
    PYTHONPATH=.. python train.py --model gnn --city mm --transfer true --source bk
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from Common.schema import DECODER_COLS, ENCODER_COLS
from Common.splits import TEST, TRAIN, VAL
from data.batch import GraphBatches, lstm_batches
from data.database import CITIES, copy_path
from data.load import load_city
from evals import climatology, persistence
from evals.eval import compare, print_table
from models import transfer
from models.gbt import GBTForecaster
from models.gnn import GNNForecaster
from models.lstm import LSTMForecaster
from models.mmd import MMDAlignment
from models.trainer import fit, predict, seed_everything
from tuning.grid import load_grid, point_name, search
from utils import runlog
from utils.artifacts import ROOT, config_dir, load_json, save_json, save_scores, settings_tag

METHODS = ("param", "mmd")

def boolean(text: str) -> bool:
    """ Parse true/false arguments.
    """
    if text.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return text.lower() == "true"

def parse_args():
    """ Read the command line.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "gnn", "gbt"])
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--transfer", type=boolean, default=False)
    ap.add_argument("--source", choices=list(CITIES))
    ap.add_argument("--training-variant", default="all",
                    choices=list(transfer.VARIANTS + transfer.GBT_VARIANTS) + ["all"])
    ap.add_argument("--autoregressive", type=boolean, default=False)
    ap.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP,
                    choices=config.LONGEST_GAP_CHOICES)
    ap.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS,
                    choices=config.COMPLETENESS_CHOICES)
    ap.add_argument("--data-ablation", type=int, default=100, choices=config.ABLATION_CHOICES)
    ap.add_argument("--block", default="all", choices=["0", "1", "2", "all"])
    ap.add_argument("--seed", type=lambda t: [int(s) for s in t.split(",")],
                    default=[42, 1234, 2026])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.transfer and (args.source is None or args.source == args.city):
        ap.error("--transfer true needs a --source different from --city")
    allowed = transfer.GBT_VARIANTS if args.model == "gbt" else transfer.VARIANTS
    if args.training_variant not in allowed + ("all",):
        ap.error(f"--training-variant for {args.model}: {', '.join(allowed)} or all")
    return args

def header(args, block, seed, variant=None) -> str:
    """ Title line of one run, e.g. [LSTM] Bangkok -> Metro Manila, mmd_frozen (...).
    """
    where = (f"{CITIES[args.source]} -> {CITIES[args.city]}, {variant}" if args.transfer
             else CITIES[args.city])
    return (f"[{args.model.upper()}] {where} (longest gap: {args.longest_gap}, station "
            f"completeness: {args.completeness}, ablation: {args.data_ablation}, "
            f"block: {block}, seed: {seed})")

def run_tag(args, block) -> str:
    """ Name this run's data settings (gap, completeness, ablation, block).
    """
    return settings_tag(args.longest_gap, args.completeness, args.data_ablation, block)

def tuned(args, path, score, grid) -> dict:
    """ Return the best grid point at `path` ({params, val, model, seed}), searching the grid on
        this run's own data first if needed: every setting gets its own tuned hyperparameters.
    """
    best = search(grid, score, path, args.seed[0])
    points = load_json(path)["points"]
    runlog.detail(f"TUNING {Path(path).parent.name}/{Path(path).name}",
                  {"best": f"{best['params']} (val {best['val']:.4f}, seed {best['seed']})"},
                  pd.DataFrame([{**p["params"], "val": p["val"]} for p in points]))
    return best

def reusable(best, seed, suffix=""):
    """ Return the tuned model of the best grid point if this seed would retrain it identically,
        else None.
    """
    if seed != best["seed"]:
        return None
    path = Path(best["model"] + suffix)
    return path if path.exists() else None

def report(out, data, ids, pred):
    """ Save and print one run's test scores.
    """
    s = save_scores(out, data, ids, pred)
    runlog.detail(f"TEST {Path(out).parent.name}/{Path(out).name}",
                  {k: v for k, v in s.items() if not isinstance(v, dict)},
                  pd.read_csv(Path(out) / "stations.csv")[["location_key", "seen", "scored",
                                                          "n_windows", "rmse", "mae", "mbe"]])
    print(f"    TEST: rmse = {s['rmse']:.3f} (seen = {s['seen'].get('rmse', float('nan')):.3f}, "
          f"unseen = {s['unseen'].get('rmse', float('nan')):.3f})")

# LSTM / GNN

_loaders = {}

def loaders(model: str, data, params: dict):
    """ Return (train(rng), val(), test()) batch factories for one dataset (cached).
    """
    key = (model, id(data), params.get("k"), params["batch_size"])
    if key in _loaders:
        return _loaders[key]
    if model == "lstm":
        ids = {s: data.split_window_ids(s) for s in (TRAIN, VAL, TEST)}
        out = (lambda rng: lstm_batches(data, ids[TRAIN], params["batch_size"], rng),
               lambda: lstm_batches(data, ids[VAL], config.EVAL_BATCH),
               lambda: lstm_batches(data, ids[TEST], config.EVAL_BATCH))
    else:
        graphs = {s: GraphBatches(data, s, params["k"], params["batch_size"])
                  for s in (TRAIN, VAL, TEST)}
        out = (graphs[TRAIN], lambda: graphs[VAL](), lambda: graphs[TEST]())
    _loaders[key] = out
    return out

def build_nn(args, params):
    """ Build an untrained LSTM/GNN with the given architecture.
    """
    cls = LSTMForecaster if args.model == "lstm" else GNNForecaster
    return cls(len(ENCODER_COLS), len(DECODER_COLS), params["hidden"], params["layers"],
               config.DROPOUT, args.autoregressive)

def load_nn(args, params, path):
    """ Rebuild an LSTM/GNN and load saved weights.
    """
    model = build_nn(args, params)
    model.load_state_dict(torch.load(path))
    return model.to(args.device)

def train_nn(args, params, data, seed, ckpt=None, init=None, freeze=False, lr=None,
             align=None, verbose=True, decay=True, save_to=None, label="TRAINING"):
    """ Build and train an LSTM/GNN; return (model, best val RMSE). save_to keeps its weights.
    """
    seed_everything(seed)
    model = build_nn(args, params)
    if init is not None:
        model.load_state_dict(init)
    if freeze:
        transfer.freeze_encoder(model)
    train, val, _ = loaders(args.model, data, params)
    history = fit(model, train, val, lr or params["lr"], seed, args.device, align, ckpt, verbose,
                  decay, label)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(save_to).with_suffix(".pt"))
    return model, min(h["val"] for h in history if h["tf"] == 0)

def pretrain(args, arch, src, tgt, method, lam, seed, cdir, verbose):
    """ Pretrain on the source, plainly or aligned to the target's training inputs; cached per
        architecture and seed. Plain pretraining depends only on the source, so it is cached with
        the source; MMD pretraining also sees the target's training inputs, so it stays per run.
    """
    name = f"{point_name(arch)}_seed{seed}.pt"
    if method == "param":
        path = config_dir(args.model, args.source, None, args.autoregressive,
                          settings_tag(args.longest_gap, args.completeness)) / "pretrain" / name
    else:
        path = Path(cdir) / "pretrain" / f"mmd_lambda{lam}_{name}"
    if path.exists():
        return torch.load(path)
    if verbose:
        print(f"    Pretraining on {CITIES[args.source]} ({method})")
        runlog.detail("PRETRAINING SETUP", {"method": method, "architecture": arch, "seed": seed,
                                      "mmd lambda": lam, "saved to": path})
    align = None
    if method == "mmd":
        align = (MMDAlignment(lam, warmup_steps=config.MMD_WARMUP_STEPS),
                 loaders(args.model, tgt, arch)[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = None if align else path.with_suffix(".ckpt")
    model, _ = train_nn(args, arch, src, seed, ckpt=ckpt, align=align, verbose=verbose,
                        label=f"PRETRAINING {method} on {CITIES[args.source]}")
    torch.save(model.state_dict(), path)
    return model.state_dict()

def finetune(args, arch, src, tgt, method, variant, ft, seed, cdir, ckpt=None, verbose=True,
             save_to=None):
    """ Fine-tune a pretrained model on the target; return (model, best val RMSE). Teacher
        forcing stays at 0: the pretrained decoder already runs on its own predictions.
    """
    init = pretrain(args, arch, src, tgt, method, ft.get("lambda"), seed, cdir, verbose)
    if verbose:
        print(f"    Fine-tuning on {CITIES[args.city]} ({variant})")
    return train_nn(args, arch, tgt, seed, ckpt, init=init, freeze=variant == "frozen",
                    lr=ft["fine_tune_lr"], verbose=verbose, decay=False, save_to=save_to,
                    label=f"FINE-TUNING {method}_{variant} on {CITIES[args.city]} {ft}")

# GBT

def gbt_arrays(data, ids):
    """ Return (lookback, x_dec, y) for window ids.
    """
    _, x_dec, y = data.gather(ids)
    return data.lookback[ids], x_dec, y

def train_gbt(args, params, data, seed, extra=None, ckpt=None, verbose=True, save_to=None,
              label="TREES"):
    """ Train a GBT on the training windows (plus weighted source windows); return (model, val RMSE).
        extra = (lookback, x_dec, y, weight) appended for transfer; target windows weigh 1.
    """
    p = dict(params)
    rounds = p.pop("num_boost_round")
    lb, xd, y = gbt_arrays(data, data.split_window_ids(TRAIN))
    weight = None
    if extra is not None:
        weight = np.concatenate([np.ones(len(y)), extra[3]])
        lb, xd, y = (np.concatenate([a, b]) for a, b in zip((lb, xd, y), extra[:3]))
    val = gbt_arrays(data, data.split_window_ids(VAL))
    model = GBTForecaster(p, rounds, args.autoregressive, args.device, seed)
    model.fit(lb, xd, y, weight, val if verbose else None, ckpt, verbose, label)
    if save_to is not None:
        model.save(save_to)
    return model, float(np.sqrt(np.mean((model.predict(*val[:2]) - val[2]) ** 2)))

def load_gbt(args, params, seed, path):
    """ Rebuild a GBT and load its saved models.
    """
    p = dict(params)
    rounds = p.pop("num_boost_round")
    return GBTForecaster(p, rounds, args.autoregressive, args.device, seed).load(path)

def gbt_test(model, data):
    """ Predict the test windows; return (pred, ids).
    """
    ids = data.split_window_ids(TEST)
    lb, xd, _ = gbt_arrays(data, ids)
    return model.predict(lb, xd), ids

# Configurations

def run_plain(args, block):
    """ No transfer: tune on the city, then train and score every seed.
    """
    data = load_city(args.city, args.longest_gap, args.completeness, args.data_ablation, block)
    cdir = config_dir(args.model, args.city, None, args.autoregressive, run_tag(args, block))
    gbt = args.model == "gbt"
    trainer = train_gbt if gbt else train_nn
    print(f"[TUNE] {args.model.upper()} {CITIES[args.city]}")
    best = tuned(args, cdir / "tuning.json",
                 lambda p, m, s: trainer(args, p, data, s, verbose=False, save_to=m,
                                         label=f"TUNING POINT {p}")[1],
                 load_grid()[args.model])
    params = best["params"]

    for seed in args.seed:
        out = cdir / "none" / f"seed{seed}"
        print(header(args, block, seed))
        if (out / "metrics.json").exists():
            print("    Done (skipped)")
            continue
        saved = reusable(best, seed, "" if gbt else ".pt")
        if saved:
            print("    Reusing the model trained while tuning (same seed and data)")
        if gbt:
            model = (load_gbt(args, params, seed, saved) if saved else
                     train_gbt(args, params, data, seed, ckpt=out / "trees")[0])
            pred, ids = gbt_test(model, data)
        else:
            model = (load_nn(args, params, saved) if saved else
                     train_nn(args, params, data, seed, ckpt=out / "checkpoint.pt")[0])
            pred, ids = predict(model, loaders(args.model, data, params)[2], args.device)
        save_json(out / "params.json", params)
        report(out, data, ids, pred)

def run_transfer_nn(args, block):
    """ LSTM/GNN transfer: architecture tuned on the source; param and MMD pretraining, each
        fine-tuned frozen and/or full on the target.
    """
    src = load_city(args.source, args.longest_gap, args.completeness)
    tgt = load_city(args.city, args.longest_gap, args.completeness, args.data_ablation, block,
                    stats=src.stats)
    # The source is never ablated: its architecture is tuned like a plain source run's.
    src_dir = config_dir(args.model, args.source, None, args.autoregressive,
                         settings_tag(args.longest_gap, args.completeness))
    cdir = config_dir(args.model, args.city, args.source, args.autoregressive, run_tag(args, block))
    grid = load_grid()

    print(f"[TUNE] {args.model.upper()} {CITIES[args.source]} (source architecture)")
    arch = tuned(args, src_dir / "tuning.json",
                 lambda p, m, s: train_nn(args, p, src, s, verbose=False, save_to=m,
                                          label=f"TUNING POINT (source) {p}")[1],
                 grid[args.model])["params"]

    variants = transfer.VARIANTS if args.training_variant == "all" else (args.training_variant,)
    for method in METHODS:
        for variant in variants:
            name = f"{method}_{variant}"
            ft_grid = {**grid["fine_tune"], **(grid["mmd"] if method == "mmd" else {})}
            print(f"[TUNE] {args.model.upper()} {CITIES[args.source]} -> {CITIES[args.city]}, {name}")
            best = tuned(args, cdir / name / "tuning.json",
                         lambda p, m, s: finetune(args, arch, src, tgt, method, variant, p,
                                                  s, cdir, verbose=False, save_to=m)[1],
                         ft_grid)
            for seed in args.seed:
                out = cdir / name / f"seed{seed}"
                print(header(args, block, seed, name))
                if (out / "metrics.json").exists():
                    print("    Done (skipped)")
                    continue
                saved = reusable(best, seed, ".pt")
                if saved:
                    print("    Reusing the model trained while tuning (same seed and data)")
                    model = load_nn(args, arch, saved)
                else:
                    model = finetune(args, arch, src, tgt, method, variant, best["params"], seed,
                                     cdir, ckpt=out / "checkpoint.pt")[0]
                pred, ids = predict(model, loaders(args.model, tgt, arch)[2], args.device)
                save_json(out / "params.json", {**arch, **best["params"]})
                report(out, tgt, ids, pred)

def source_distances(args, src, tgt, cdir) -> dict:
    """ Return each source station's MMD distance to the target's training rows, measured once per
        run setting (the target's training rows change with gap, completeness and ablation block).
    """
    path = Path(cdir) / "distances.json"
    saved = load_json(path)
    if saved is not None:
        return {k: np.asarray(v) for k, v in saved.items()}
    print(f"    Measuring station distances: {CITIES[args.source]} -> {CITIES[args.city]}")
    # The fixed bandwidth is pooled from every city except the target, as in the distance study.
    pool = {c: src if c == args.source else load_city(c, args.longest_gap, args.completeness)
            for c in CITIES if c != args.city}
    dist = transfer.station_distances(src, tgt, pool)
    save_json(path, {k: v.tolist() for k, v in dist.items()})
    return dist

def run_transfer_gbt(args, block):
    """ GBT transfer: the target's training windows plus the source's, either pooled (every source
        window weighs 1) or weighted (each source station by exp(-d_s / tau), from its MMD distance
        to the target). Each variant tunes its own tree settings on the data it trains on; the
        weighted variant tunes tau jointly with them, for this source-target pair only.
    """
    src = load_city(args.source, args.longest_gap, args.completeness)
    tgt = load_city(args.city, args.longest_gap, args.completeness, args.data_ablation, block,
                    stats=src.stats)
    cdir = config_dir("gbt", args.city, args.source, args.autoregressive, run_tag(args, block))
    grid = load_grid()
    ids = src.split_window_ids(TRAIN)
    source_arrays = gbt_arrays(src, ids)
    trees = lambda p: {k: v for k, v in p.items() if k != "tau_scale"}

    variants = transfer.GBT_VARIANTS if args.training_variant == "all" else (args.training_variant,)
    for variant in variants:
        if variant == "pooled":
            space, tau_of = grid["gbt"], lambda p: None
            weight_of = lambda p: np.ones(len(ids))
        else:
            dist = source_distances(args, src, tgt, cdir)
            median = transfer.tau_grid(dist, [1.0])[0]   # tau = median station distance x scale
            space, tau_of = {**grid["gbt"], "tau_scale": grid["tau_scales"]}, lambda p: median * p["tau_scale"]
            weight_of = lambda p: transfer.gbt_weights(src, ids, dist, tau_of(p))

        print(f"[TUNE] GBT {CITIES[args.source]} -> {CITIES[args.city]}, {variant}")
        best = tuned(args, cdir / variant / "tuning.json",
                     lambda p, m, s: train_gbt(args, trees(p), tgt, s, (*source_arrays, weight_of(p)),
                                               verbose=False, save_to=m,
                                               label=f"TUNING POINT {variant} {p}")[1], space)
        params = best["params"]
        weights = weight_of(params)
        station_weight = pd.Series(weights).groupby(src.station[ids]).first()
        table = pd.DataFrame({"station": src.keys[station_weight.index],
                              "windows": pd.Series(src.station[ids]).value_counts()[station_weight.index].to_numpy(),
                              "weight": station_weight.round(6).to_numpy()})
        if variant == "weighted":
            d = pd.DataFrame({k: v for k, v in dist.items() if k != "station"},
                             index=src.keys[dist["station"]])
            table = table.join(d.round(6), on="station")
        runlog.detail(f"SOURCE WEIGHTS {CITIES[args.source]} -> {CITIES[args.city]}, {variant}",
                      {"tau": tau_of(params), "tree settings": trees(params),
                       "source windows": len(ids), "effective source windows": round(float(weights.sum()), 1)},
                      table)
        for seed in args.seed:
            out = cdir / variant / f"seed{seed}"
            print(header(args, block, seed, variant))
            if (out / "metrics.json").exists():
                print("    Done (skipped)")
                continue
            saved = reusable(best, seed)
            if saved:
                print("    Reusing the model trained while tuning (same seed and data)")
            model = (load_gbt(args, trees(params), seed, saved) if saved else
                     train_gbt(args, trees(params), tgt, seed, (*source_arrays, weight_of(params)),
                               ckpt=out / "trees")[0])
            pred, test_ids = gbt_test(model, tgt)
            save_json(out / "params.json", {**params, "tau": tau_of(params)})
            report(out, tgt, test_ids, pred)

def baselines_and_comparison(args):
    """ Score persistence and climatology on the same test windows (once per filter setting),
        then print this city's comparison.
    """
    tag = settings_tag(args.longest_gap, args.completeness)
    for name, module in (("persistence", persistence), ("climatology", climatology)):
        if not (ROOT / "baselines" / name / args.city / tag / "test" / "metrics.json").exists():
            module.run(args.city, args.longest_gap, args.completeness)
    table = compare(args.city)
    print_table(table[table["filters"] == f"gap{args.longest_gap}_comp{args.completeness}"])

def main():
    args = parse_args()
    pair = f"{args.source}-{args.city}" if args.transfer else args.city
    path = runlog.start(ROOT / "logs", f"{args.model}_{pair}_{'ar' if args.autoregressive else 'direct'}_"
                        f"{settings_tag(args.longest_gap, args.completeness, args.data_ablation)}")
    try:
        runlog.detail("RUN", {"command": " ".join(sys.argv), **vars(args), **runlog.environment(),
                              **{f"copy of {c}": f"{copy_path(c)} (modified "
                                 f"{datetime.fromtimestamp(copy_path(c).stat().st_mtime):%Y-%m-%d %H:%M:%S})"
                                 for c in CITIES if copy_path(c).exists()}})
        runlog.detail("CONFIG", {k: v for k, v in vars(config).items() if k.isupper()})
        run(args)
    finally:
        runlog.stop()
        print(f"Log: {path}")

def run(args):
    """ Run every block of the configuration, then the baselines and the comparison.
    """
    blocks = [0] if args.data_ablation == 100 else (
        [0, 1, 2] if args.block == "all" else [int(args.block)])
    for block in blocks:
        _loaders.clear()  # drop the previous block's data and batches
        if not args.transfer:
            run_plain(args, block)
        elif args.model == "gbt":
            run_transfer_gbt(args, block)
        else:
            run_transfer_nn(args, block)
    baselines_and_comparison(args)

if __name__ == "__main__":
    main()
