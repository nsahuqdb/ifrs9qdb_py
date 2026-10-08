"""
Why the provision moved -- contract by contract, at any level.

``ecl_bridge`` walks every contract from the previous run to the current one,
pricing it the way each run's own report priced it (etl/report.py: the same
engine frame, the same pricing context) and changing one ingredient at a time:

    f0  the previous run, as it was
    f1  + the current balance, EAD curve and remaining term   -> Exposure
    f2  + the current stage                                    -> Stage migration
    f3  + the current rating, on the previous run's PD curves  -> Rating migration
    f4  + the current macro inputs, on the previous model      -> Macro variables
    f5  + the current model                                    -> Model
    f6  + the current collateral, LGD and EIR                  -> LGD & collateral

f0 is the previous run's model ECL and f6 the current run's, so the steps
telescope to the move in the model ECL. The change in the post-model overlay is
its own step, and "Other" is what the report holds that the repricing does not
-- zero when the engine reproduces the report -- together with the whole move
of a contract that cannot be repriced in both runs. A contract that left is
Derecognised, one that arrived New business. Every contract's components
therefore sum EXACTLY to its change in the reports, and a customer, a facility,
an account type, a segment or a stage is a sum of contracts.

Macro and model are told apart by pricing once more on PD curves built from the
current run's macro inputs and the previous run's model, both read from the
runs' frozen ``config_used``. What counts as which:

    model  model.yml (the MEV models, the TTC anchor, the horizons), the model
           chosen (config.yml run.internal_model), the TTC PD table, and how
           the MEV models combine (model_inputs.yml mev_model_weights)
    macro  the rest of model_inputs.yml (MEV forecasts, scenario weights, the
           GCC paths), the scenario severities and the GDP histories

When only one of them changed it takes the whole PD-curve move and the other is
zero by construction. Without a frozen config on both sides the two cannot be
told apart and the move is shown as one step, "PD curves (macro and model)".

The order is fixed, as in any sequential decomposition: another order moves
the interaction terms between the factors without changing the total.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["BRIDGE_COMPONENTS", "BRIDGE_LEVELS", "BridgeRun", "load_bridge_run",
           "config_changes", "ecl_bridge", "bridge_view", "bridge_by",
           "bridge_members"]

# component -> label, in waterfall order
BRIDGE_COMPONENTS = {
    "derecognised": "Derecognised",
    "moved_out": "Moved out",
    "moved_in": "Moved in",
    "new_business": "New business",
    "exposure": "Exposure",
    "stage": "Stage migration",
    "rating": "Rating migration",
    "macro": "Macro variables",
    "model": "Model",
    "pd_curves": "PD curves (macro and model)",
    "lgd": "LGD & collateral",
    "overlay": "Overlay",
    "other": "Other",
}
# what a contract carries (moving between groups is a group's, not a contract's)
CONTRACT_COMPONENTS = ("derecognised", "new_business", "exposure", "stage",
                       "rating", "macro", "model", "pd_curves", "lgd",
                       "overlay", "other")

# level -> the contract attribute that places a contract in a group
BRIDGE_LEVELS = {"book": None, "customer": "customer", "facility": "contract",
                 "account_type": "account_type", "segment": "portfolio",
                 "stage": "stage", "rating": "rating"}

# model and macro parts of a run's frozen config (see the module docstring)
_MODEL_STATIC = ("ttc_pd_table",)
_MACRO_STATIC = ("scenario_severity", "non_oil_gdp_history",
                 "gcc_real_gdp_growth", "gcc_gdp_current_prices")
_MODEL_INPUT_KEYS = ("mev_model_weights",)


@dataclass
class BridgeRun:
    """One side of a comparison: the report, and what priced it."""
    run_dir: Path
    ok: bool
    reason: str = ""
    out_dir: Path | None = None
    report: pd.DataFrame | None = None   # normalised; one row per position
    frame: pd.DataFrame | None = None    # the engine frame, one row per position
    ctx: object = None                   # etl.report._Ctx
    model_cfg: dict | None = None
    model_inputs: dict | None = None
    static: object = None
    model_id: str | None = None
    extract_date: str | None = None
    has_config: bool = False


def _yaml(path: Path):
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_bridge_run(run_dir, report_path=None) -> BridgeRun:
    """A run's report and the inputs it was priced with.

    ``report_path`` is the report to explain (an overlaid one, say); the
    default is the run's FinalEclReport.csv. The config and the reference are
    the run's own frozen copies when it has them, as its report used.
    """
    from ..etl.pipeline import load_model_config
    from ..etl.report import _Ctx, _portfolio_map, engine_frame
    from ..etl.static_ref import load_static_reference
    from ..inputs import as_id
    from .profile import normalise

    run_dir = Path(run_dir)
    out_dir = run_dir / "Output" if (run_dir / "Output").is_dir() else run_dir
    rp = Path(report_path) if report_path else out_dir / "FinalEclReport.csv"
    if not rp.is_file():
        return BridgeRun(run_dir, False, f"No ECL report in {out_dir}")
    raw = pd.read_csv(rp, low_memory=False)
    rep = normalise(raw)
    if len(rep) == 0:
        return BridgeRun(run_dir, False, f"{rp.name} has no contracts")
    names = {c.lower().replace(" ", ""): c for c in raw.columns}
    if "customername" in names:
        nm = pd.read_csv(rp, low_memory=False, usecols=[names["customername"],
                                                        names["contractid"]])
        nm.columns = ["name", "contract"]
        nm["contract"] = as_id(nm["contract"]).astype("string")
        rep = rep.merge(nm.drop_duplicates("contract"), on="contract", how="left")
    else:
        rep["name"] = pd.NA

    cu = run_dir / "config_used"
    has_cfg = (cu / "config").is_dir() and (cu / "static").is_dir()
    try:
        static = load_static_reference(cu / "static" if has_cfg else None)
    except Exception:
        static, has_cfg = None, False
    model_cfg, model_inputs = load_model_config(cu / "config" if has_cfg else None)
    if model_cfg is None:
        has_cfg = False
    rc = _yaml(cu / "config" / "config.yml") if has_cfg else None
    run = (rc or {}).get("run") if isinstance(rc, dict) else None
    model_id = (run or {}).get("internal_model") if isinstance(run, dict) else None

    ctx = _Ctx(out_dir, model_cfg=model_cfg)
    pmap = _portfolio_map(static)
    parts = [engine_frame(out_dir, s, pmap) for s in ("_1", "_2")]
    parts = [p for p in parts if p is not None]
    if not parts:
        return BridgeRun(run_dir, False, f"No AccountMaster in {out_dir}")
    frame = pd.concat(parts, ignore_index=True)
    # the run's own id keys its collateral and EAD-curve tables; the
    # normalised one joins the frame to the report and the two runs
    frame["cid"] = frame["contract"]
    frame["contract"] = as_id(frame["contract"]).astype("string")

    # One row per position: a contract id is not unique (an investment held in
    # two positions), and both the report and the engine frame keep the
    # AccountMaster's order, so the n-th row of a contract in one is the n-th
    # in the other.
    rep["occ"] = rep.groupby("contract").cumcount()
    frame["occ"] = frame.groupby("contract").cumcount()
    rep["key"] = rep["contract"].astype(str) + "#" + rep["occ"].astype(str)
    frame["key"] = frame["contract"].astype(str) + "#" + frame["occ"].astype(str)
    ext = pd.to_datetime(rep["extract_date"], errors="coerce", format="mixed")
    ext = ext.dropna()
    return BridgeRun(
        run_dir, True, out_dir=out_dir, report=rep,
        frame=frame.drop_duplicates("key").set_index("key"), ctx=ctx,
        model_cfg=model_cfg, model_inputs=model_inputs, static=static,
        model_id=model_id,
        extract_date=(ext.mode().iloc[0].strftime("%Y-%m-%d") if len(ext) else None),
        has_config=has_cfg)


# --------------------------------------------------------- what changed ----
def _flat(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k in x:
            out.update(_flat(x[k], f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(_flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = x
    return out


def _dict_diff(a, b, label: str) -> list[dict]:
    fa, fb = _flat(a or {}), _flat(b or {})
    out = []
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            out.append({"item": f"{label}: {k}", "before": fa.get(k),
                        "after": fb.get(k)})
    return out


def _table_diff(a, b, label: str) -> list[dict]:
    if a is None and b is None:
        return []
    if a is None or b is None:
        return [{"item": label, "before": "absent" if a is None else f"{len(a)} rows",
                 "after": "absent" if b is None else f"{len(b)} rows"}]
    ra, rb = a.reset_index(drop=True), b.reset_index(drop=True)
    if ra.shape == rb.shape and list(ra.columns) == list(rb.columns):
        same = ra.astype(str).eq(rb.astype(str))
        if bool(same.all().all()):
            return []
        n = int((~same.all(axis=1)).sum())
        return [{"item": label, "before": f"{len(ra)} rows",
                 "after": f"{n} row(s) changed"}]
    return [{"item": label, "before": f"{len(ra)} rows", "after": f"{len(rb)} rows"}]


def config_changes(a: BridgeRun, b: BridgeRun) -> dict:
    """What changed in the model and in the macro inputs between two runs,
    read from their frozen config. ``known`` is False when either run has
    none, and then neither can be said to have changed or not."""
    if not (a.has_config and b.has_config):
        return {"known": False, "model_changed": None, "macro_changed": None,
                "model": [], "macro": []}
    model = _dict_diff(a.model_cfg, b.model_cfg, "model.yml")
    if (a.model_id or "") != (b.model_id or ""):
        model.append({"item": "config.yml: run.internal_model",
                      "before": a.model_id or "(default)",
                      "after": b.model_id or "(default)"})
    for k in _MODEL_INPUT_KEYS:
        model += _dict_diff((a.model_inputs or {}).get(k),
                            (b.model_inputs or {}).get(k), f"model_inputs.yml: {k}")
    for k in _MODEL_STATIC:
        model += _table_diff(a.static.get(k), b.static.get(k), f"{k}.csv")
    ma = {k: v for k, v in (a.model_inputs or {}).items() if k not in _MODEL_INPUT_KEYS}
    mb = {k: v for k, v in (b.model_inputs or {}).items() if k not in _MODEL_INPUT_KEYS}
    macro = _dict_diff(ma, mb, "model_inputs.yml")
    for k in _MACRO_STATIC:
        macro += _table_diff(a.static.get(k), b.static.get(k), f"{k}.csv")
    return {"known": True, "model_changed": bool(model),
            "macro_changed": bool(macro), "model": model, "macro": macro}


def _hybrid_ctx(a: BridgeRun, b: BridgeRun):
    """The current run's pricing context on PD curves built from ITS macro
    inputs and the PREVIOUS run's model."""
    from ..etl.macro import build_stpd_from_static
    from ..etl.report import _Ctx

    static_h = copy.copy(b.static)
    for k in _MODEL_STATIC:
        static_h[k] = a.static.get(k)
    inputs_h = copy.deepcopy(b.model_inputs) or {}
    for k in _MODEL_INPUT_KEYS:
        if k in (a.model_inputs or {}):
            inputs_h[k] = copy.deepcopy(a.model_inputs[k])
        else:
            inputs_h.pop(k, None)
    stpd = build_stpd_from_static(static_h, a.model_cfg, inputs_h,
                                  b.extract_date or "", model_id=a.model_id)
    ctx = copy.copy(b.ctx)
    ctx.stpd = _Ctx._stpd(stpd)
    return ctx


# ------------------------------------------------------------ the bridge ----
def _price(ctx, stage: int, ht, bal: float, lgd: float, cum, eir) -> float:
    """One contract, as the report prices it (etl/report.py _segment)."""
    from ..engine import sum_marginal_ecl
    if stage == 3:
        if ctx.stage3 == "zero":
            return 0.0
        return bal if bal > 0 else 0.0
    if cum is None or ht is None:
        return float("nan")
    curve, H, _ = ht
    tot = sum_marginal_ecl(curve, lgd, cum, eir, H)
    if np.isfinite(tot) and ctx.cap and bal > 0:
        tot = min(tot, bal)
    return float(tot)


def _num0(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def ecl_bridge(a: BridgeRun, b: BridgeRun) -> dict:
    """Every contract's move from run ``a`` to run ``b``, split into causes.

    Returns ``rows`` -- one row per position, with both runs' attributes
    (customer, account type, segment, rating, stage, exposure, ECL, PD, LGD)
    and the components, which sum to ``ecl_b - ecl_a`` -- with what changed
    in the config, whether macro and model could be told apart, and notes.
    """
    from ..engine import compute_lgd
    from ..etl.report import _ead_curve

    if not (a.ok and b.ok):
        return {"ok": False, "reason": a.reason or b.reason}
    changes = config_changes(a, b)
    notes = []
    split = changes["known"]
    if not split:
        notes.append("A run has no frozen config (config_used), so the move in "
                     "the PD curves is one step, not split into macro and model.")
        ctx_h = None
    elif changes["macro_changed"] and changes["model_changed"]:
        try:
            ctx_h = _hybrid_ctx(a, b)
        except Exception as exc:
            split, ctx_h = False, None
            notes.append("The PD curves for the current macro inputs on the "
                         f"previous model could not be built ({exc}); the move "
                         "is one step.")
    elif changes["macro_changed"]:
        ctx_h = b.ctx          # the model did not change: macro takes it all
    elif changes["model_changed"]:
        ctx_h = a.ctx          # the macro inputs did not: the model takes it
    else:
        ctx_h = None           # neither changed: any curve difference is Other

    ra = a.report.set_index("key")
    rb = b.report.set_index("key")
    keys_a, keys_b = ra.index, rb.index
    both = keys_a.intersection(keys_b)
    only_a = keys_a.difference(keys_b)
    only_b = keys_b.difference(keys_a)

    cols = ["contract", "customer", "name", "account_type", "portfolio",
            "rating", "stage", "exposure", "exposure_off", "ecl", "overlay",
            "pd", "lgd", "dpd"]
    left = ra.reindex(keys_a.union(keys_b))[cols].add_suffix("_a")
    right = rb.reindex(keys_a.union(keys_b))[cols].add_suffix("_b")
    rows = pd.concat([left, right], axis=1)
    rows["contract"] = rows["contract_b"].fillna(rows["contract_a"])
    rows["name"] = rows["name_b"].fillna(rows["name_a"])
    rows["status"] = "continuing"
    rows.loc[only_a, "status"] = "derecognised"
    rows.loc[only_b, "status"] = "new"
    for c in ("ecl_a", "ecl_b", "overlay_a", "overlay_b", "exposure_a",
              "exposure_b"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce").fillna(0.0)
    for c in CONTRACT_COMPONENTS:
        rows[c] = 0.0
    rows["repriced"] = False
    rows.loc[only_a, "derecognised"] = -rows.loc[only_a, "ecl_a"]
    rows.loc[only_b, "new_business"] = rows.loc[only_b, "ecl_b"]

    fa, fb = a.frame, b.frame
    ca, cb = a.ctx, b.ctx
    eff = {k: np.zeros(len(both)) for k in ("exposure", "stage", "rating",
                                           "macro", "model", "pd_curves", "lgd",
                                           "other")}
    ok = np.zeros(len(both), dtype=bool)
    not_split_rating = 0
    for i, key in enumerate(both):
        if key not in fa.index or key not in fb.index:
            continue
        ea, eb = fa.loc[key], fb.loc[key]
        # priced on the engine frame's own stage, rating and segment -- what
        # the report priced with
        sa, sb = int(ea["stage"]), int(eb["stage"])
        ba, bb = _num0(ea["on_balance"]), _num0(eb["on_balance"])
        na = _num0(ca.collnet.get(ea["cid"], 0.0))
        nb = _num0(cb.collnet.get(eb["cid"], 0.0))
        la = compute_lgd(ba, na, base=ca.lgd_base, unsecured_floor=ca.lgd_floor)
        lb = compute_lgd(bb, nb, base=cb.lgd_base, unsecured_floor=cb.lgd_floor)
        ra_, rb_ = ea["rating"], eb["rating"]
        rta = "External" if str(ea["rating_type"]) == "External" else "Internal"
        rtb = "External" if str(eb["rating_type"]) == "External" else "Internal"
        pfa, pfb = ea["portfolio"], eb["portfolio"]
        cum_a = ca.pd_curve(pfa, ra_, rta)
        cum_r = ca.pd_curve(pfb, rb_, rtb)
        cum_b = cb.pd_curve(pfb, rb_, rtb)
        cum_h = ctx_h.pd_curve(pfb, rb_, rtb) if ctx_h is not None else None

        def ht(ctx, e, stage, bal):
            return _ead_curve(ctx, e["cid"], stage, bal, int(e["months"]),
                              e["payment_type"], e["portfolio"], e["nir"],
                              e["deferral"], e["pay_freq"])

        h_a = ht(ca, ea, sa, ba)
        h_ba = ht(cb, eb, sa, bb)
        h_bb = h_ba if sb == sa else ht(cb, eb, sb, bb)
        eir_a, eir_b = ea["eir"], eb["eir"]

        f0 = _price(ca, sa, h_a, ba, la, cum_a, eir_a)
        f1 = _price(cb, sa, h_ba, bb, la, cum_a, eir_a)
        f2 = _price(cb, sb, h_bb, bb, la, cum_a, eir_a)
        if cum_r is None and cum_a is not None and sb != 3:
            # the previous run had no curve for the current rating: the rating
            # move cannot be priced on its own and travels with the next step
            f3 = f2
            not_split_rating += 1
        else:
            f3 = _price(cb, sb, h_bb, bb, la, cum_r, eir_a)
        f5 = _price(cb, sb, h_bb, bb, la, cum_b, eir_a)
        f4 = _price(cb, sb, h_bb, bb, la, cum_h, eir_a) if ctx_h is not None else f5
        f6 = _price(cb, sb, h_bb, bb, lb, cum_b, eir_b)
        f = [f0, f1, f2, f3, f4, f5, f6]
        if not all(np.isfinite(x) for x in f):
            continue
        ok[i] = True
        eff["exposure"][i] = f1 - f0
        eff["stage"][i] = f2 - f1
        eff["rating"][i] = f3 - f2
        if split and ctx_h is not None:
            eff["macro"][i] = f4 - f3
            eff["model"][i] = f5 - f4
        elif split:
            eff["other"][i] += f5 - f3           # neither changed
        else:
            eff["pd_curves"][i] = f5 - f3
        eff["lgd"][i] = f6 - f5
        model_a = rows.at[key, "ecl_a"] - rows.at[key, "overlay_a"]
        model_b = rows.at[key, "ecl_b"] - rows.at[key, "overlay_b"]
        eff["other"][i] += (model_b - f6) - (model_a - f0)

    cont = rows.loc[both]
    move = (cont["ecl_b"] - cont["ecl_a"]).to_numpy()
    ovl = (cont["overlay_b"] - cont["overlay_a"]).to_numpy()
    for k, v in eff.items():
        rows.loc[both, k] = np.where(ok, v, 0.0)
    rows.loc[both, "overlay"] = np.where(ok, ovl, 0.0)
    rows.loc[both, "other"] = np.where(ok, eff["other"], move)
    rows.loc[both, "repriced"] = ok
    n_not = int((~ok).sum())
    if n_not:
        notes.append(f"{n_not:,} contract(s) in both runs could not be repriced "
                     "in both (no PD curve, EAD curve or EIR); their whole move "
                     "is under Other.")
    if not_split_rating:
        notes.append(f"{not_split_rating:,} contract(s) moved to a rating the "
                     "previous run had no PD curve for; their rating move is "
                     "shown with the next step.")
    for side in ("a", "b"):
        rows[f"stage_{side}"] = pd.to_numeric(rows[f"stage_{side}"],
                                              errors="coerce").astype("Int64")
    rows.index.name = "key"
    rows = rows.reset_index()
    total = rows[list(CONTRACT_COMPONENTS)].sum(axis=1)
    resid = float((total - (rows["ecl_b"] - rows["ecl_a"])).abs().max())
    return {"ok": True, "rows": rows, "changes": changes, "split": split,
            "notes": notes, "residual": resid,
            "runs": {"a": {"run": a.run_dir.name, "extract_date": a.extract_date,
                           "model_id": a.model_id},
                     "b": {"run": b.run_dir.name, "extract_date": b.extract_date,
                           "model_id": b.model_id}}}


# ------------------------------------------------------------- the views ----
def _attr(rows: pd.DataFrame, level: str, side: str) -> pd.Series:
    col = BRIDGE_LEVELS[level]
    s = rows["contract"] if col == "contract" else rows[f"{col}_{side}"]
    if col == "stage":
        return s.astype("Int64").astype("string")
    return s.astype("string")


def _masks(rows, level, members):
    if level == "book" or BRIDGE_LEVELS.get(level) is None:
        return (rows["status"] != "new"), (rows["status"] != "derecognised")
    want = {str(m) for m in (members or [])}
    in_a = _attr(rows, level, "a").isin(want).fillna(False) & (rows["status"] != "new")
    in_b = _attr(rows, level, "b").isin(want).fillna(False) & \
        (rows["status"] != "derecognised")
    return in_a.astype(bool), in_b.astype(bool)


def _profile(rows: pd.DataFrame, mask: pd.Series, side: str) -> dict:
    d = rows[mask]
    exp = pd.to_numeric(d[f"exposure_{side}"], errors="coerce").fillna(0.0)
    ecl = pd.to_numeric(d[f"ecl_{side}"], errors="coerce").fillna(0.0)
    off = pd.to_numeric(d[f"exposure_off_{side}"], errors="coerce").fillna(0.0)
    ovl = pd.to_numeric(d[f"overlay_{side}"], errors="coerce").fillna(0.0)
    tot = float(exp.sum())

    def wavg(col):
        v = pd.to_numeric(d[col], errors="coerce")
        w = exp.where(v.notna(), 0.0)
        return float((v.fillna(0.0) * w).sum() / w.sum()) if w.sum() > 0 else None

    stage = d[f"stage_{side}"]
    by_stage = {str(int(s)): float(exp[stage == s].sum())
                for s in sorted(stage.dropna().unique())}
    rating = d[f"rating_{side}"].astype("string")
    rmix = (pd.DataFrame({"r": rating, "e": exp}).groupby("r")["e"].sum()
            .sort_values(ascending=False))
    ratings = [{"rating": str(k), "exposure": float(v)} for k, v in rmix.head(5).items()]
    dpd = pd.to_numeric(d[f"dpd_{side}"], errors="coerce")
    return {
        "contracts": int(d["contract"].nunique()),
        "customers": int(d[f"customer_{side}"].nunique()),
        "exposure": tot, "exposure_off": float(off.sum()),
        "ecl": float(ecl.sum()), "overlay": float(ovl.sum()),
        "coverage": 100 * float(ecl.sum()) / tot if tot > 0 else None,
        "pd": wavg(f"pd_{side}"), "lgd": wavg(f"lgd_{side}"),
        "worst_stage": int(stage.max()) if stage.notna().any() else None,
        "stage_exposure": by_stage,
        "rating": (str(rmix.index[0]) if len(rmix) == 1
                   else f"{len(rmix)} ratings" if len(rmix) else None),
        "ratings": ratings,
        "dpd_max": float(dpd.max()) if dpd.notna().any() else None,
    }


def bridge_view(rows: pd.DataFrame, level: str = "book", members=None,
                top: int = 200) -> dict:
    """The waterfall for a group of contracts, with its before and after.

    ``level`` is one of BRIDGE_LEVELS and ``members`` the values chosen (one
    or more customers, facilities, account types, segments, stages or
    ratings). A contract belongs to the group in a run when its attribute in
    THAT run is one of them, so the opening and closing are each run's own
    figures for the group. A contract that left the group for another is
    "Moved out" at its previous ECL; one that joined is "Moved in" at its
    previous ECL, and its own causes then follow, so the steps still sum
    exactly to the closing.
    """
    if level not in BRIDGE_LEVELS:
        raise ValueError(f"level must be one of {list(BRIDGE_LEVELS)}")
    if rows is None or len(rows) == 0:
        return {}
    in_a, in_b = _masks(rows, level, members)
    cont = rows["status"] == "continuing"
    steps = dict.fromkeys(BRIDGE_COMPONENTS, 0.0)
    steps["derecognised"] = float(rows.loc[in_a & (rows["status"] == "derecognised"),
                                           "derecognised"].sum())
    steps["new_business"] = float(rows.loc[in_b & (rows["status"] == "new"),
                                           "new_business"].sum())
    steps["moved_out"] = -float(rows.loc[in_a & cont & ~in_b, "ecl_a"].sum())
    steps["moved_in"] = float(rows.loc[~in_a & cont & in_b, "ecl_a"].sum())
    stay = in_b & cont
    for k in ("exposure", "stage", "rating", "macro", "model", "pd_curves",
              "lgd", "overlay", "other"):
        steps[k] = float(rows.loc[stay, k].sum())
    opening = float(rows.loc[in_a, "ecl_a"].sum())
    closing = float(rows.loc[in_b, "ecl_b"].sum())

    table = [{"key": "opening", "label": "Opening", "amount": opening,
              "kind": "total"}]
    table += [{"key": k, "label": BRIDGE_COMPONENTS[k], "amount": v,
               "kind": "delta"} for k, v in steps.items()]
    table.append({"key": "closing", "label": "Closing", "amount": closing,
                  "kind": "total"})

    sel = rows[in_a | in_b].copy()
    sel["change"] = sel["ecl_b"] - sel["ecl_a"]
    sel = sel.reindex(sel["change"].abs().sort_values(ascending=False).index)
    keep = ["contract", "customer_b", "customer_a", "name", "status",
            "account_type_a", "account_type_b", "portfolio_a", "portfolio_b",
            "rating_a", "rating_b", "stage_a", "stage_b", "exposure_a",
            "exposure_b", "ecl_a", "ecl_b", "change",
            *CONTRACT_COMPONENTS]
    return {
        "level": level, "members": [str(m) for m in (members or [])],
        "opening": opening, "closing": closing,
        "steps": pd.DataFrame(table),
        "residual": opening + sum(steps.values()) - closing,
        "before": _profile(rows, in_a, "a"),
        "after": _profile(rows, in_b, "b"),
        "counts": {"in_previous": int(in_a.sum()), "in_current": int(in_b.sum()),
                   "derecognised": int((in_a & (rows["status"] == "derecognised")).sum()),
                   "new": int((in_b & (rows["status"] == "new")).sum()),
                   "moved_out": int((in_a & cont & ~in_b).sum()),
                   "moved_in": int((~in_a & cont & in_b).sum())},
        "contracts": sel[keep].head(top).reset_index(drop=True),
    }


def _labels(rows: pd.DataFrame, level: str, values) -> list[str]:
    """A customer's or a facility's id with its name, from either run; any
    other group as it is."""
    values = [str(v) for v in values]
    if BRIDGE_LEVELS[level] not in ("customer", "contract"):
        return values
    ga, gb = _attr(rows, level, "a"), _attr(rows, level, "b")
    named = pd.DataFrame({"v": gb.fillna(ga), "n": rows["name"]}).dropna()
    named = named[named["n"].astype(str).str.strip() != ""]
    names = named.drop_duplicates("v").set_index("v")["n"]
    return [f"{v} — {names[v]}" if v in names.index else v for v in values]


def bridge_by(rows: pd.DataFrame, level: str) -> pd.DataFrame:
    """Every group of a level, with its opening, its causes and its closing:
    bridge_view() for each group at once, so a level of thousands of
    customers or facilities is one pass over the contracts. ``label`` is the
    group with a customer's or a facility's name."""
    if level not in BRIDGE_LEVELS or BRIDGE_LEVELS[level] is None:
        raise ValueError("bridge_by needs a level other than the whole book")
    status = rows["status"]
    cont = status == "continuing"
    # each contract's group in each run, and none in a run it is not in
    ga = _attr(rows, level, "a").where(status != "new")
    gb = _attr(rows, level, "b").where(status != "derecognised")
    groups = sorted(set(ga.dropna()) | set(gb.dropna()), key=str)
    if not groups:
        return pd.DataFrame()
    moved = cont & ~(ga == gb).fillna(False).astype(bool)

    def total(values, keys, mask=None):
        m = keys.notna() if mask is None else (mask & keys.notna())
        m = m.to_numpy(dtype=bool)
        s = pd.Series(pd.to_numeric(values, errors="coerce").fillna(0.0)
                      .to_numpy(dtype=float)[m], index=keys.to_numpy()[m])
        return s.groupby(level=0).sum().reindex(groups, fill_value=0.0).to_numpy()

    out = pd.DataFrame({"group": groups, "opening": total(rows["ecl_a"], ga)})
    out["derecognised"] = total(rows["derecognised"], ga, status == "derecognised")
    out["moved_out"] = 0.0 - total(rows["ecl_a"], ga, moved)
    out["moved_in"] = total(rows["ecl_a"], gb, moved)
    out["new_business"] = total(rows["new_business"], gb, status == "new")
    for k in ("exposure", "stage", "rating", "macro", "model", "pd_curves",
              "lgd", "overlay", "other"):
        out[k] = total(rows[k], gb, cont)
    out["closing"] = total(rows["ecl_b"], gb)
    out["change"] = out["closing"] - out["opening"]
    out["label"] = _labels(rows, level, groups)
    out = out[["group", "label", "opening", *BRIDGE_COMPONENTS, "closing", "change"]]
    order = np.argsort(-out["change"].abs().to_numpy(), kind="stable")
    return out.iloc[order].reset_index(drop=True)


def bridge_members(rows: pd.DataFrame, level: str, query: str | None = None,
                   limit: int | None = None) -> pd.DataFrame:
    """What can be chosen at a level, largest move first: the value, a label
    (a customer's name), and its ECL in each run. ``limit`` keeps the first
    so many; all of them by default."""
    if level not in BRIDGE_LEVELS or BRIDGE_LEVELS[level] is None:
        return pd.DataFrame()
    ga, gb = _attr(rows, level, "a"), _attr(rows, level, "b")
    a = pd.DataFrame({"value": ga, "ecl": rows["ecl_a"]})[rows["status"] != "new"]
    b = pd.DataFrame({"value": gb, "ecl": rows["ecl_b"]})[rows["status"] != "derecognised"]
    sa = a.groupby("value")["ecl"].sum()
    sb = b.groupby("value")["ecl"].sum()
    idx = sa.index.union(sb.index)
    out = pd.DataFrame({"value": idx,
                        "ecl_a": sa.reindex(idx).fillna(0.0).to_numpy(),
                        "ecl_b": sb.reindex(idx).fillna(0.0).to_numpy()})
    out["change"] = out["ecl_b"] - out["ecl_a"]
    out["label"] = _labels(rows, level, out["value"])
    if query:
        q = str(query).lower()
        out = out[out["label"].str.lower().str.contains(q, regex=False)]
    out = out.iloc[np.argsort(-out["change"].abs().to_numpy(), kind="stable")]
    if limit:
        out = out.head(limit)
    return out.reset_index(drop=True)
