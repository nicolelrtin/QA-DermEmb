"""S8 Tune: how the ridge penalty is chosen, shown in full (PROJECT.md, Plan item A).

For one model family (ridge by default), on the outer folds of the S6 design (same splits, inner splits, estimator and
seeds), every value of a fine penalty grid is fitted and scored inside each outer training set (inner CV) and on the
outer test fold. Three selection rules are then compared, each choosing one C per outer fold from the inner scores only:
  logloss   lowest inner log loss (what S6 does)
  primary   highest inner AUROC for benign vs malignant, highest inner balanced accuracy for the multiclass tasks
  one_se    strongest penalty within one standard error of the lowest inner log loss
A fourth row, s6_grid, applies the log-loss rule on the 7 values S6 searched. It must give S6's choices and S6's
outer scores exactly; this is the stage's check that it reproduces the S6 design.

lambda = 1 / C. scikit-learn minimises C x (sum of class-weighted log losses) + 0.5 x |w|^2 on standardised answers.

Outputs: inner_curves.csv (per outer fold, C and inner fold), fold_curves.csv (per outer fold and C: inner means and
the outer score), selections.csv (per outer fold and rule), rules_summary.csv, repeats.csv (pooled score per repeat
and rule), oof_predictions.parquet (per rule), [s6_check.csv], figures/, summary.md.
"""
import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models import plots
from ..models.metrics import task_metrics
from ..models.tuning import penalty_curves, select
from .s5_model import feature_mask
from .s6_classify import MODEL_NAMES, SOURCE_NAMES, define_tasks, load_labels, manifest_frame

VERSION = 2
RULES = ["logloss", "primary", "one_se", "s6_grid"]
RULE_NAMES = {"logloss": "lowest inner log loss (as in S6)", "primary": "best inner AUROC / balanced accuracy",
              "one_se": "strongest penalty within 1 SE of lowest log loss", "s6_grid": "S6's 7-value grid, log loss"}
S6_GRID = {"l2": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1], "l1": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}


def upstream(cfg) -> list[str]:
    return ["manifest", "embed", "classify"] if cfg.tune.check_classify else ["manifest", "embed"]


def key(cfg, ctx) -> dict:
    c = cfg.classify
    return {"tune": cfg.tune, "outer": [c.outer_folds, c.outer_repeats], "label_columns": c.label_columns,
            "exclude_kinds": cfg.model.exclude_kinds, "exclude_scopes": cfg.model.exclude_scopes, "seed": cfg.seed,
            "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest), "sens_target": c.sens_target,
            "sens_at_spec": cfg.model.sens_at_spec, "ece_bins": cfg.model.ece_bins}


def primary(task: str) -> str:
    return "auroc" if task == "s1_2c" else "balanced_accuracy"


def grid_values(tc) -> list[float]:
    """1, 2, 3 and 5 per decade from 10^lo to 10^hi (C), so S6's 1-3-10 grid is contained in it."""
    g = [m * 10.0 ** d for d in range(int(tc.log10_C[0]), int(tc.log10_C[1])) for m in (1, 2, 3, 5)]
    return sorted(set(np.round(g + [10.0 ** int(tc.log10_C[1])] + S6_GRID.get(tc.model, []), 12)))


def run(cfg, out, inputs, ctx) -> dict:
    tc, cc = cfg.tune, cfg.classify
    man = manifest_frame(cfg, inputs)
    lab = load_labels(cfg, man)
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qk = qt[feature_mask(qt, cfg.model, patch=False)].reset_index(drop=True)
    qids = list(qk.question_id)
    sources = {s: pd.read_parquet(inputs["embed"] / f"X_image_{s}.parquet").loc[list(man.image_id)]
               for s in tc.sources}
    tasks = define_tasks(lab, list(tc.tasks))
    groups = lab.lesion.values
    grid = grid_values(tc)
    spec, sens, bins = cfg.model.sens_at_spec, cc.sens_target, cfg.model.ece_bins
    (out / "figures").mkdir(exist_ok=True)

    specs, meta = {}, {}
    for s, Xdf in sources.items():
        X = Xdf[qids].values.astype(np.float64)
        for t, T in tasks.items():
            rows, cls = T["rows"], T["classes"]
            yi = np.array([cls.index(v) for v in lab[T["col"]].values[rows]])
            meta[(s, t)] = (rows, cls, yi)
            specs[(s, t)] = (tc.model, X[rows], yi, lab.c10.values[rows], groups[rows], cfg.seed)
    res = penalty_curves(specs, grid, cc, cc.inner_folds, tc.n_jobs)

    inner_rows, curve_rows, sel_rows, rep_rows, pred_frames = [], [], [], [], []
    for (s, t), R in res.items():
        rows, cls, yi = meta[(s, t)]
        pm = primary(t)
        oof = {rule: np.full((cc.outer_repeats, len(yi), len(cls)), np.nan) for rule in RULES}
        for si, (r, f, tr, te) in enumerate(R["splits"]):
            base = {"source": s, "task": t, "repeat": r, "fold": f}
            fc = []
            for C in grid:
                fit = R["fits"][(si, C)]
                inn = np.array([x[1:] for x in fit["inner"]])
                for i, ll, pr in fit["inner"]:
                    inner_rows.append({**base, "C": C, "inner_fold": i, "log_loss": ll, f"inner_{pm}": pr})
                met = task_metrics(yi[te], fit["P"], cls, spec, sens, bins)
                fc.append({**base, "C": C, "lambda": 1 / C, "ll_mean": inn[:, 0].mean(),
                           "ll_se": inn[:, 0].std(ddof=1) / np.sqrt(len(inn)), "prim_mean": inn[:, 1].mean(),
                           "n_iter": fit["n_iter"], "n_nonzero": fit["n_nonzero"],
                           **{f"outer_{k}": met[k] for k in (pm, "log_loss", "macro_f1", f"spec_at_sens{int(sens * 100)}")
                              if k in met}})
            fc = pd.DataFrame(fc)
            curve_rows.append(fc)
            for rule in RULES:
                sub = fc[fc.C.isin(S6_GRID[tc.model])] if rule == "s6_grid" else fc
                C = select(sub, "logloss" if rule == "s6_grid" else rule)
                P = R["fits"][(si, C)]["P"]
                oof[rule][r, te] = P
                met = task_metrics(yi[te], P, cls, spec, sens, bins)
                sel_rows.append({**base, "rule": rule, "C": C, "lambda": 1 / C,
                                 **{k: v for k, v in met.items() if k != "confusion_matrix"}})
        for rule in RULES:
            for r in range(cc.outer_repeats):
                met = task_metrics(yi, oof[rule][r], cls, spec, sens, bins)
                rep_rows.append({"source": s, "task": t, "rule": rule, "repeat": r,
                                 **{k: v for k, v in met.items() if k != "confusion_matrix"}})
                pr = pd.DataFrame(oof[rule][r], columns=[f"p::{c}" for c in cls])
                pr.insert(0, "image_id", lab.image_id.values[rows])
                pred_frames.append(pr.assign(source=s, task=t, rule=rule, repeat=r, true=np.array(cls)[yi]))

    inner = pd.DataFrame(inner_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    sel = pd.DataFrame(sel_rows)
    reps = pd.DataFrame(rep_rows)
    inner.to_csv(out / "inner_curves.csv", index=False)
    curves.to_csv(out / "fold_curves.csv", index=False)
    sel.to_csv(out / "selections.csv", index=False)
    reps.to_csv(out / "repeats.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(out / "oof_predictions.parquet", index=False)

    summ = rules_summary(sel, reps, tasks, sens)
    summ.to_csv(out / "rules_summary.csv", index=False)
    check = s6_check(inputs, sel, tc.model) if "classify" in inputs else None
    if check is not None:
        check.to_csv(out / "s6_check.csv", index=False)
    figures(out / "figures", curves, sel, reps, tasks, list(tc.sources), tc.model)
    write_summary(out, cfg, lab, tasks, qk, grid, curves, sel, reps, summ, check)
    best = summ[(summ.task == "s1_2c") & (summ.rule == "logloss")]
    return {"model": tc.model, "grid_size": len(grid), "C_range": [min(grid), max(grid)],
            "sources": list(tc.sources), "tasks": list(tc.tasks),
            "s6_reproduced": bool(check.match.all()) if check is not None else None,
            "key_metric": ("; ".join(f"{r.source} s1_2c auroc={r.fold_mean:.3f}" for r in best.itertuples())
                           if len(best) else "")}


def rules_summary(sel, reps, tasks, sens) -> pd.DataFrame:
    rows = []
    for (s, t, rule), g in sel.groupby(["source", "task", "rule"], sort=False):
        pm = primary(t)
        rp = reps[(reps.source == s) & (reps.task == t) & (reps.rule == rule)][pm]
        lam = g["lambda"]
        mode = lam.round(6).mode().iloc[0]
        row = {"source": s, "task": t, "rule": rule, "metric": pm, "fold_mean": g[pm].mean(), "fold_sd": g[pm].std(),
               "pooled_mean": rp.mean(), "pooled_sd_over_repeats": rp.std(), "pooled_min": rp.min(),
               "pooled_max": rp.max(), "lambda_median": lam.median(), "lambda_mode": mode,
               "lambda_mode_share": (lam.round(6) == mode).mean(), "lambda_min": lam.min(), "lambda_max": lam.max()}
        if t == "s1_2c":
            k = f"spec_at_sens{int(sens * 100)}"
            row[f"{k}_fold_mean"] = g[k].mean()
            row["log_loss_fold_mean"] = g["log_loss"].mean()
        else:
            row["macro_f1_fold_mean"] = g["macro_f1"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def s6_check(inputs, sel, model) -> pd.DataFrame | None:
    """The s6_grid rule against S6's own chosen C and outer-fold score, fold by fold."""
    d = inputs["classify"]
    bp = pd.read_csv(d / "best_params.csv")
    mf = pd.read_csv(d / "metrics_folds.csv")
    bp, mf = bp[bp.model == model], mf[mf.model == model]
    me = sel[sel.rule == "s6_grid"]
    rows = []
    for x in me.itertuples():
        pm = primary(x.task)
        b = bp[(bp.source == x.source) & (bp.task == x.task) & (bp.repeat == x.repeat) & (bp.fold == x.fold)]
        m = mf[(mf.source == x.source) & (mf.task == x.task) & (mf.repeat == x.repeat) & (mf.fold == x.fold)]
        if b.empty or m.empty:
            continue
        s6C, s6m = float(b["clf__C"].iloc[0]), float(m[pm].iloc[0])
        rows.append({"source": x.source, "task": x.task, "repeat": x.repeat, "fold": x.fold, "C_s6": s6C,
                     "C_here": x.C, f"{pm}_s6": s6m, f"{pm}_here": getattr(x, pm),
                     "match": bool(np.isclose(s6C, x.C) and abs(s6m - getattr(x, pm)) < 1e-6)})
    return pd.DataFrame(rows) if rows else None


def figures(fdir, curves, sel, reps, tasks, sources, model):
    plots.penalty_curves(curves, sel, list(tasks), sources, primary, SOURCE_NAMES,
                         f"{MODEL_NAMES[model]}: inner and outer scores against the penalty", fdir / "penalty_curves.png")
    plots.chosen_lambda(sel, list(tasks), sources, RULES, SOURCE_NAMES,
                        f"{MODEL_NAMES[model]}: penalty chosen in each of the outer folds, by selection rule",
                        fdir / "chosen_lambda.png")
    plots.repeat_spread(reps, list(tasks), sources, RULES, primary, SOURCE_NAMES,
                        "Score of each repeat of the 5-fold outer split (pooled over its 5 test folds), by selection rule",
                        fdir / "repeat_spread.png")


def _f(v, d=3):
    return f"{v:.{d}f}"


def _lam(v):
    return f"{v:,.0f}" if v >= 10 else f"{v:.2g}"


def write_summary(out, cfg, lab, tasks, qk, grid, curves, sel, reps, summ, check):
    tc, cc = cfg.tune, cfg.classify
    L = [f"# S8 tune: choosing the {MODEL_NAMES[tc.model]} penalty", ""]
    L += [f"{len(lab)} images, {len(qk)} questions (artefact questions and the control question left out, as in S6). "
          f"Outer design as in S6: {cc.outer_repeats} repeats of {cc.outer_folds}-fold, stratified on the 10 subtypes and "
          f"grouped by lesion; {cc.inner_folds}-fold inner CV inside each outer training set. "
          f"Penalty grid: {len(grid)} values of C from {min(grid):g} to {max(grid):g} "
          f"(lambda = 1/C from {_lam(1 / max(grid))} to {_lam(1 / min(grid))}).", ""]
    L += ["## How to read this", "",
          "- *Inner* scores are what a selection rule sees: the model is fitted on two thirds of an outer training set "
          "and scored on the remaining third, averaged over the three inner folds.",
          "- *Outer* scores are on the outer test fold, images the rule never saw. The outer curve over all penalties is "
          "drawn only to show how flat or peaked it is; picking the penalty from it would be cheating.",
          "- *Fold mean* is the mean over the 25 outer test folds (about 160 images each for benign vs malignant). "
          "*Pooled* is the score of one repeat with its 5 test folds put together (all 800 or 1,000 images once); its "
          "spread over the 5 repeats is what a single 5x3 nested CV, without repeats, would vary by.", ""]
    if check is not None:
        n, ok = len(check), int(check.match.sum())
        L += ["## Check against S6", "",
              f"Using S6's 7-value grid and log loss, this stage reproduces S6's chosen penalty and outer score in {ok} of "
              f"{n} outer folds{' (all)' if ok == n else ' — the stage does NOT reproduce S6; do not trust the rest'}.", ""]
    L += ["## Selection rules", ""]
    for t in tasks:
        pm = primary(t)
        L += [f"### {t} ({pm.replace('_', ' ')})", "",
              "| source | rule | fold mean ± sd | pooled per repeat: mean (min to max) | lambda chosen: median (range); most common (share of folds) |",
              "|---|---|---|---|---|"]
        for x in summ[summ.task == t].itertuples():
            L.append(f"| {SOURCE_NAMES.get(x.source, x.source)} | {RULE_NAMES[x.rule]} | {_f(x.fold_mean)} ± {_f(x.fold_sd)} | "
                     f"{_f(x.pooled_mean)} ({_f(x.pooled_min)} to {_f(x.pooled_max)}) | {_lam(x.lambda_median)} "
                     f"({_lam(x.lambda_min)} to {_lam(x.lambda_max)}); {_lam(x.lambda_mode)} ({x.lambda_mode_share:.0%}) |")
        L.append("")
    L += ["## Shape of the curves", "",
          "Mean over the 25 outer folds at each penalty: inner log loss, inner primary score, and the outer primary score "
          "(for reference only). The best value on each curve and how much the score changes over a factor of 10 in "
          "lambda around it.", "",
          "| source | task | lambda at lowest inner log loss | lambda at best inner score | lambda at best outer score | outer score: best / at 10x weaker / at 10x stronger penalty |",
          "|---|---|---|---|---|---|"]
    for (s, t), g in curves.groupby(["source", "task"], sort=False):
        pm = primary(t)
        m = g.groupby("C")[["ll_mean", "prim_mean", f"outer_{pm}"]].mean().sort_index()
        lam = 1 / m.index.values
        o = m[f"outer_{pm}"].values
        ib = int(np.argmax(o))

        def at(factor):
            j = int(np.argmin(np.abs(np.log10(lam) - np.log10(lam[ib] * factor))))
            return o[j]
        L.append(f"| {SOURCE_NAMES.get(s, s)} | {t} | {_lam(lam[int(np.argmin(m.ll_mean.values))])} | "
                 f"{_lam(lam[int(np.argmax(m.prim_mean.values))])} | {_lam(lam[ib])} | "
                 f"{_f(o[ib])} / {_f(at(0.1))} / {_f(at(10))} |")
    L += ["", "Figures: `figures/penalty_curves.png`, `figures/chosen_lambda.png`, `figures/repeat_spread.png`.", ""]
    (out / "summary.md").write_text("\n".join(L))
