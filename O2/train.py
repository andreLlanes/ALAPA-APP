""" Tune, train and score one configuration for every seed (and every block under ablation).
    python train.py --model lstm --city mm
    python train.py --model gnn --city mm --transfer true --source bk+la
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from baselines import climatology, persistence, ridge
from baselines.dlinear import DLinear
from Common.schema import DECODER_COLS, ENCODER_COLS
from Common.splits import TEST, TRAIN, VAL
from data.batch import GraphBatches, edited, lstm_batches
from data.database import CITIES
from data.load import load_city
from evals.eval import all_runs, compare, print_table
from models import transfer
from models.gbt import GBTForecaster
from models.gnn import GNNForecaster
from models.lstm import LSTMForecaster
from models.mmd import MMDAlignment
from models.trainer import fit, predict, seed_everything
from tuning.grid import load_grid, point_name, search
from utils import runlog
from utils.args import add_arguments, blocks, config_folder, names, read_arguments, variants
from utils.artifacts import ROOT, config_dir, load_json, save_json, save_scores, settings_tag

def header(args, block, seed, variant) -> str:
    """ Title line of one run, e.g. [LSTM] Bangkok -> Metro Manila, mmd_frozen (...).
    """
    target = CITIES[args.city]
    where = (target if not args.transfer else
             f"Zeroshot: {names(args.sources)} -> {target}" if variant == "zeroshot" else
             f"Pooled: {names(args.sources)} + {target}" if variant == "pooled" else
             f"{names(args.sources)} -> {target}, {variant}")
    return (f"[{args.model.upper()}] {where} (longest gap: {args.longest_gap}, station "
            f"completeness: {args.completeness}, ablation: {args.data_ablation}, "
            f"block: {block}, seed: {seed})")

def tuned(title, path, score, grid) -> dict:
    """ Return the best grid point at `path` ({params, val}), searching the grid on this run's own
        data first if needed: every setting gets its own tuned hyperparameters.
    """
    print(f"[TUNE] {title} (seed {config.TUNING_SEED})")
    best = search(grid, score, path)
    points = load_json(path)["points"]
    runlog.detail(f"TUNING {Path(path).parent.name}/{Path(path).name}",
                  {"best": f"{best['params']} (val {best['val']:.4f})"},
                  pd.DataFrame([{**p["params"], "val": p["val"]} for p in points]))
    return best

def report(out, data, ids, pred, val_ids, val_pred):
    """ Save and print one run's test scores.
    """
    s = save_scores(out, data, ids, pred, val_ids, val_pred)
    runlog.detail(f"TEST {Path(out).parent.name}/{Path(out).name}",
                  {k: v for k, v in s.items() if not isinstance(v, dict)},
                  pd.read_csv(Path(out) / "stations.csv")[["location_key", "seen", "scored",
                                                          "n_windows", "rmse", "mae", "mbe"]])
    print(f"    TEST: rmse = {s['rmse']:.3f} (seen = {s['seen'].get('rmse', float('nan')):.3f}, "
          f"unseen = {s['unseen'].get('rmse', float('nan')):.3f})")

def run_seed(args, block, variant, seed, out, data, params, train):
    """ Train one seed unless it is done, then save its scores. train(out) returns the model;
        params are saved with the run and describe its architecture for the batch loaders.
    """
    print(header(args, block, seed, variant))
    if (out / "metrics.json").exists():
        print("    Done (skipped)")
        return
    if variant == "zeroshot":
        print(f"    No fine-tuning: the model trained on {names(args.sources)} is applied to "
              f"{CITIES[args.city]} as is")
    model = train(out)
    pred, ids = predict_split(args, model, data, params, TEST)
    val_pred, val_ids = (predict_split(args, model, data, params, VAL)
                         if config.SAVE_VAL_PREDICTIONS else (None, None))
    save_json(out / "params.json", params)
    if args.model == "gbt":
        model.importance().to_csv(out / "feature_importance.csv", index=False)
    report(out, data, ids, pred, val_ids, val_pred)

def predict_split(args, model, data, params, split, edit=None):
    """ Predict a split's windows; return (normalized predictions, window ids). edit(x_enc, x_dec,
        ids) -> (x_enc, x_dec) changes the inputs first (the trees have no x_enc: None).
    """
    if args.model == "gbt":
        ids = data.split_window_ids(split)
        lookback, x_dec, _ = gbt_arrays(data, ids)
        return model.predict(lookback, x_dec if edit is None else edit(None, x_dec, ids)[1]), ids
    _, val, test = loaders(args.model, data, params)
    batches = test if split == TEST else val[0]
    return predict(model, batches if edit is None else edited(batches, edit), args.device)

def pooled_data(args, block):
    """ Return the source cities and the target as one dataset that trains on all of them and
        stops and scores on the target.
    """
    return load_city([*args.sources, args.city], args.longest_gap, args.completeness,
                     args.data_ablation, block).only_scored_on(args.city)

def target_data(args, block, stats=None):
    """ Return the target city (ablated), normalized with the source statistics in transfer.
    """
    return load_city(args.city, args.longest_gap, args.completeness, args.data_ablation, block,
                     stats=stats)

def variant_data(args, block, variant):
    """ Return the data a saved variant was scored on.
    """
    if variant == "pooled" and args.model != "gbt":
        return pooled_data(args, block)
    stats = load_city(args.sources, args.longest_gap, args.completeness).stats if args.transfer else None
    return target_data(args, block, stats)

# LSTM / GNN / DLinear

LOADERS = {}

def loaders(model: str, data, params: dict):
    """ Return (train(rng), [val() per validation city], test()) batch factories for one dataset
        (cached).
    """
    key = (model, id(data), params.get("k"), params["batch_size"])
    if key not in LOADERS:
        train, val, test = (data.split_window_ids(TRAIN), data.split_by_city(VAL),
                            data.split_window_ids(TEST))
        if model == "gnn":
            graphs = lambda ids: GraphBatches(data, ids, params["k"], params["batch_size"])
            LOADERS[key] = (graphs(train), [graphs(ids) for ids in val], graphs(test))
        else:
            LOADERS[key] = (lambda rng: lstm_batches(data, train, params["batch_size"], rng),
                             [partial(lstm_batches, data, ids, config.EVAL_BATCH) for ids in val],
                             partial(lstm_batches, data, test, config.EVAL_BATCH))
    return LOADERS[key]

def build_nn(args, params):
    """ Build an untrained model with the given architecture.
    """
    if args.model == "dlinear":
        return DLinear(params["kernel"])
    cls = LSTMForecaster if args.model == "lstm" else GNNForecaster
    return cls(len(ENCODER_COLS), len(DECODER_COLS), params["hidden"], params["layers"],
               config.DROPOUT, args.autoregressive)

def load_nn(args, params, path):
    """ Rebuild a model and load its saved weights.
    """
    model = build_nn(args, params)
    model.load_state_dict(torch.load(path, map_location=args.device))
    return model.to(args.device)

def train_nn(args, params, data, seed, ckpt=None, init=None, freeze=False, lr=None,
             align=None, verbose=True, decay=True, save_to=None, label="TRAINING"):
    """ Build and train a model; return (model, best val RMSE). save_to keeps its weights and
        replaces the resume checkpoint.
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
        torch.save(model.state_dict(), save_to)
        if ckpt:
            Path(ckpt).unlink(missing_ok=True)
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
        return torch.load(path, map_location=args.device)
    if verbose:
        print(f"    Pretraining on {names(args.sources)} ({method})")
        runlog.detail("PRETRAINING SETUP", {"method": method, "architecture": arch, "seed": seed,
                                            "mmd lambda": lam, "saved to": path})
    align = None
    if method == "mmd":
        align = (MMDAlignment(lam, warmup_steps=config.MMD_WARMUP_STEPS),
                 loaders(args.model, tgt, arch)[0])
    model, _ = train_nn(args, arch, src, seed, ckpt=None if align else path.with_suffix(".ckpt"),
                        align=align, verbose=verbose, save_to=path,
                        label=f"PRETRAINING {method} on {names(args.sources)}")
    return model.state_dict()

def finetune(args, arch, src, tgt, method, mode, ft, seed, cdir, ckpt=None, verbose=True,
             save_to=None):
    """ Fine-tune a pretrained model on the target; return (model, best val RMSE). Teacher
        forcing stays at 0: the pretrained decoder already runs on its own predictions.
    """
    init = pretrain(args, arch, src, tgt, method, ft.get("lambda"), seed, cdir, verbose)
    if verbose:
        print(f"    Fine-tuning on {CITIES[args.city]} ({mode})")
    return train_nn(args, arch, tgt, seed, ckpt, init=init, freeze=mode == "frozen",
                    lr=ft["fine_tune_lr"], verbose=verbose, decay=False, save_to=save_to,
                    label=f"FINE-TUNING {method}_{mode} on {CITIES[args.city]} {ft}")

def zero_shot(args, arch, src, tgt, seed, cdir, out):
    """ Return the plainly pretrained source model, saved as this run's model.
    """
    model = build_nn(args, arch)
    model.load_state_dict(pretrain(args, arch, src, tgt, "param", None, seed, cdir, True))
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    return model.to(args.device)

def load_model(args, params, seed, out):
    """ Load a finished run's model from its folder.
    """
    return (load_gbt(args, params, seed, out / "trees") if args.model == "gbt"
            else load_nn(args, params, out / "model.pt"))

# GBT

def gbt_arrays(data, ids):
    """ Return (lookback, x_dec, y) for window ids.
    """
    _, x_dec, y = data.gather(ids)
    return data.lookback[ids], x_dec, y

def train_gbt(args, params, data, seed, extra=None, ckpt=None, verbose=True, label="TREES"):
    """ Train a GBT on the training windows (plus weighted source windows); return (model, val
        RMSE, the mean over validation cities). extra = (lookback, x_dec, y, weight) appended for
        transfer; target windows weigh 1.
    """
    p = dict(params)
    rounds = p.pop("num_boost_round")
    lb, xd, y = gbt_arrays(data, data.split_window_ids(TRAIN))
    weight = None
    if extra is not None:
        weight = np.concatenate([np.ones(len(y)), extra[3]])
        lb, xd, y = (np.concatenate([a, b]) for a, b in zip((lb, xd, y), extra[:3]))
    val_ids = data.split_window_ids(VAL)
    val = gbt_arrays(data, val_ids)
    model = GBTForecaster(p, rounds, args.autoregressive, args.device, seed)
    model.fit(lb, xd, y, weight, val if verbose else None, ckpt, verbose, label)
    error = (model.predict(*val[:2]) - val[2]) ** 2
    city = data.city[data.station[val_ids]]
    return model, float(np.mean([np.sqrt(error[city == c].mean()) for c in np.unique(city)]))

def load_gbt(args, params, seed, path):
    """ Rebuild a GBT and load its saved models.
    """
    p = dict(params)
    rounds = p.pop("num_boost_round")
    return GBTForecaster(p, rounds, args.autoregressive, args.device, seed).load(path)

# Configurations

def run_plain(args, block):
    """ No transfer: tune on the city, then train and score every seed.
    """
    data = target_data(args, block)
    cdir = config_folder(args, block)
    gbt = args.model == "gbt"
    trainer = train_gbt if gbt else train_nn
    best = tuned(f"{args.model.upper()} {CITIES[args.city]}", cdir / "tuning.json",
                 lambda p, s: trainer(args, p, data, s, verbose=False,
                                      label=f"TUNING POINT {p}")[1],
                 load_grid()[args.model])
    params = best["params"]
    for seed in args.seed:
        train = ((lambda out: train_gbt(args, params, data, seed, ckpt=out / "trees")[0]) if gbt else
                 (lambda out: train_nn(args, params, data, seed, ckpt=out / "checkpoint.pt",
                                       save_to=out / "model.pt")[0]))
        run_seed(args, block, "none", seed, cdir / "none" / f"seed{seed}", data, params, train)

def nn_variant(args, variant, src, tgt, arch, cdir, block):
    """ Return (scored data, architecture and fine-tuning settings, train(seed, out)) for one
        LSTM / GNN transfer variant, tuning what the variant needs.
    """
    grid = load_grid()
    if variant == "zeroshot":
        return tgt, arch, lambda seed, out: zero_shot(args, arch, src, tgt, seed, cdir, out)
    title = f"{args.model.upper()} {names(args.sources)}"
    if variant == "pooled":
        data = pooled_data(args, block)
        best = tuned(f"{title} + {CITIES[args.city]}, pooled", cdir / variant / "tuning.json",
                     lambda p, s: train_nn(args, p, data, s, verbose=False,
                                           label=f"TUNING POINT pooled {p}")[1], grid[args.model])
        train = lambda seed, out: train_nn(args, best["params"], data, seed,
                                           ckpt=out / "checkpoint.pt", save_to=out / "model.pt")[0]
        return data, best["params"], train
    method, mode = variant.split("_")
    ft_grid = {**grid["fine_tune"], **(grid["mmd"] if method == "mmd" else {})}
    best = tuned(f"{title} -> {CITIES[args.city]}, {variant}", cdir / variant / "tuning.json",
                 lambda p, s: finetune(args, arch, src, tgt, method, mode, p, s, cdir,
                                       verbose=False)[1], ft_grid)
    train = lambda seed, out: finetune(args, arch, src, tgt, method, mode, best["params"], seed,
                                       cdir, ckpt=out / "checkpoint.pt",
                                       save_to=out / "model.pt")[0]
    return tgt, {**arch, **best["params"]}, train

def run_transfer_nn(args, block):
    """ LSTM/GNN transfer: the source architecture is tuned once; each variant then tunes its own
        settings. Zero-shot applies the pretrained source model as is, pooled trains from scratch
        on source and target together, and the rest fine-tune a param or MMD pretrained model.
    """
    src = load_city(args.sources, args.longest_gap, args.completeness)
    tgt = target_data(args, block, src.stats)
    cdir = config_folder(args, block)
    todo = variants(args)
    arch = None
    if todo != ["pooled"]:
        # The source is never ablated: its architecture is tuned like a plain source run's.
        src_dir = config_dir(args.model, args.source, None, args.autoregressive,
                             settings_tag(args.longest_gap, args.completeness))
        arch = tuned(f"{args.model.upper()} {names(args.sources)} (source architecture)",
                     src_dir / "tuning.json",
                     lambda p, s: train_nn(args, p, src, s, verbose=False,
                                           label=f"TUNING POINT (source) {p}")[1],
                     load_grid()[args.model])["params"]
    for variant in todo:
        data, params, train = nn_variant(args, variant, src, tgt, arch, cdir, block)
        for seed in args.seed:
            run_seed(args, block, variant, seed, cdir / variant / f"seed{seed}", data, params,
                     partial(train, seed))

def source_distances(args, src, tgt, cdir) -> dict:
    """ Return each source station's MMD distance to the target's training rows, measured once per
        run setting (the target's training rows change with gap, completeness and ablation block).
    """
    path = Path(cdir) / "distances.json"
    saved = load_json(path)
    if saved is not None:
        return {k: np.asarray(v) for k, v in saved.items()}
    print(f"    Measuring station distances: {names(args.sources)} -> {CITIES[args.city]}")
    # The fixed bandwidth is pooled from every city except the target, as in the distance study.
    pool = {c: transfer.training_rows(src, c)[0] if c in args.sources else
            transfer.training_rows(load_city(c, args.longest_gap, args.completeness))[0]
            for c in CITIES if c != args.city}
    dist = transfer.station_distances(src, tgt, pool)
    save_json(path, {k: v.tolist() for k, v in dist.items()})
    return dist

def gbt_variant(args, variant, src, tgt, cdir):
    """ Return (saved settings, train(seed, out)) for one GBT transfer variant, tuning what it
        needs. Zero-shot trains on the source alone; pooled adds every source window at weight 1;
        weighted adds each source station at exp(-d / tau), tuning tau jointly with the trees for
        this source-target pair.
    """
    grid = load_grid()
    title = f"GBT {names(args.sources)}"
    if variant == "zeroshot":
        src_dir = config_dir("gbt", args.source, None, args.autoregressive,
                             settings_tag(args.longest_gap, args.completeness))
        best = tuned(f"{title} (source trees)", src_dir / "tuning.json",
                     lambda p, s: train_gbt(args, p, src, s, verbose=False,
                                            label=f"TUNING POINT (source) {p}")[1], grid["gbt"])
        return best["params"], lambda seed, out: train_gbt(args, best["params"], src, seed,
                                                           ckpt=out / "trees")[0]
    ids = src.split_window_ids(TRAIN)
    source_arrays = gbt_arrays(src, ids)
    trees = lambda p: {k: v for k, v in p.items() if k != "tau_scale"}
    if variant == "pooled":
        space, tau_of = grid["gbt"], lambda p: None
        weight_of = lambda p: np.ones(len(ids))
    else:
        dist = source_distances(args, src, tgt, cdir)
        median = transfer.tau_grid(dist, [1.0])[0]   # tau = median station distance x scale
        space = {**grid["gbt"], "tau_scale": grid["tau_scales"]}
        tau_of = lambda p: median * p["tau_scale"]
        weight_of = lambda p: transfer.gbt_weights(src, ids, dist, tau_of(p))
    best = tuned(f"{title} -> {CITIES[args.city]}, {variant}", cdir / variant / "tuning.json",
                 lambda p, s: train_gbt(args, trees(p), tgt, s, (*source_arrays, weight_of(p)),
                                        verbose=False, label=f"TUNING POINT {variant} {p}")[1],
                 space)
    params, weights = best["params"], weight_of(best["params"])
    station_weight = pd.Series(weights).groupby(src.station[ids]).first()
    table = pd.DataFrame({"station": src.keys[station_weight.index],
                          "windows": pd.Series(src.station[ids]).value_counts()[station_weight.index].to_numpy(),
                          "weight": station_weight.round(6).to_numpy()})
    if variant == "weighted":
        table = table.join(pd.DataFrame({k: v for k, v in dist.items() if k != "station"},
                                        index=src.keys[dist["station"]]).round(6), on="station")
    runlog.detail(f"SOURCE WEIGHTS {names(args.sources)} -> {CITIES[args.city]}, {variant}",
                  {"tau": tau_of(params), "tree settings": trees(params),
                   "source windows": len(ids),
                   "effective source windows": round(float(weights.sum()), 1)}, table)
    train = lambda seed, out: train_gbt(args, trees(params), tgt, seed,
                                        (*source_arrays, weights), ckpt=out / "trees")[0]
    return {**params, "tau": tau_of(params)}, train

def run_transfer_gbt(args, block):
    """ GBT transfer: each variant tunes its own tree settings on the data it trains on.
    """
    src = load_city(args.sources, args.longest_gap, args.completeness)
    tgt = target_data(args, block, src.stats)
    cdir = config_folder(args, block)
    for variant in variants(args):
        params, train = gbt_variant(args, variant, src, tgt, cdir)
        for seed in args.seed:
            run_seed(args, block, variant, seed, cdir / variant / f"seed{seed}", tgt, params,
                     partial(train, seed))

def baselines_and_comparison(args):
    """ Score the cheap baselines on the same test windows (once per filter setting), then print
        this city's comparison.
    """
    tag = settings_tag(args.longest_gap, args.completeness)
    for name, module in (("persistence", persistence), ("climatology", climatology),
                         ("ridge", ridge)):
        if not (ROOT / "baselines" / name / args.city / tag / "test" / "metrics.json").exists():
            module.run(args.city, args.longest_gap, args.completeness)
    table = compare(all_runs(args.city))
    print_table(table[table["filters"] == f"gap{args.longest_gap}_comp{args.completeness}"])

def run(args):
    """ Run every block of the configuration, then the baselines and the comparison.
    """
    for block in blocks(args):
        LOADERS.clear()  # drop the previous block's data and batches
        if not args.transfer:
            run_plain(args, block)
        elif args.model == "gbt":
            run_transfer_gbt(args, block)
        else:
            run_transfer_nn(args, block)
    baselines_and_comparison(args)

def main():
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    ap.add_argument("--seed", type=lambda t: [int(s) for s in t.split(",")],
                    default=config.DEFAULT_SEEDS)
    args = read_arguments(ap)
    if config.TUNING_SEED in args.seed:
        ap.error(f"seed {config.TUNING_SEED} tunes the hyperparameters and is not a result seed")
    with runlog.logged(f"{args.model}_{args.source + '-' if args.transfer else ''}{args.city}", args):
        run(args)

if __name__ == "__main__":
    main()
