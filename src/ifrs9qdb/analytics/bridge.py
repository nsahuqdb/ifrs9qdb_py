"""
Why the provision moved -- contract by contract, at any level.

``ecl_bridge`` walks every contract from the previous run to the current one,
pricing it the way each run's own report priced it (etl/report.py: the same
engine frame, the same pricing context) and changing one ingredient at a time:

    f0  the previous run, repriced on this engine                -> Method change
    f1  + the current balance, EAD curve and remaining term   -> Exposure
    f2  + the current stage                                    -> Stage migration
    f3  + the current rating, on the previous run's PD curves  -> Rating migration
    f4  + the current macro inputs, on the previous model      -> Macro variables
    f5' + the current ECL settings                             -> ECL settings
    f5  + the current PD model                                 -> Model
    f6  + the current collateral, LGD and EIR                  -> LGD & collateral

Method change is the previous run's model ECL against f0: the previous run on
this version of the calculation, with its own data, config and -- rebuilt from
its frozen config -- PD curves. It is zero when the previous run was priced by
the same calculation, and otherwise holds a change in the method itself (the
month count, the EAD payment dates, the scenario weighting) rather than leaving
it to the steps it touches. f6 is the current run's model ECL, so the steps
telescope to the move in the model ECL. The change in the post-model overlay is
its own step, and "Other" is what the current report holds that the repricing
does not -- zero when this engine priced it -- together with the whole move of
a contract that cannot be repriced in both runs. A contract that left is
Derecognised, one that arrived New business. Every contract's components
therefore sum EXACTLY to its change in the reports, and a customer, a facility,
an account type, a segment or a stage is a sum of contracts.

The ECL settings in model.yml (its ``ecl`` block: the EAD fallback rules for a
contract with no repayment schedule, the Stage 3 method, the exposure cap) are
not the PD model and are their own step, ECL settings: the steps before it are
priced on the previous run's settings, and it switches to the current run's.
Model is then the PD model alone, so a quarter in which only the settings
changed shows no Model step.

Macro and model are told apart by pricing once more on PD curves built from the
current run's macro inputs and the previous run's model, both read from the
runs' frozen ``config_used``. What counts as which:

    model  model.yml except its ``ecl`` block (the MEV models, the TTC anchor,
           the horizons), the model chosen (config.yml run.internal_model),
           the TTC PD table, and how the MEV models combine
           (model_inputs.yml mev_model_weights)
    settings  model.yml's ``ecl`` block
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
    "method": "Method change",
    "derecognised": "Derecognised",
    "new_business": "New business",
    "exposure": "Exposure",
    "stage": "Stage migration",
    "rating": "Rating migration",
    "macro": "Macro variables",
    "model": "Model",
    "settings": "ECL settings",
    "pd_curves": "PD curves (macro and model)",
    "lgd": "LGD & collateral",
    "overlay": "Overlay",
    "other": "Other",
}
# what a contract carries
CONTRACT_COMPONENTS = ("method", "derecognised", "new_business", "exposure",
                       "stage", "rating", "macro", "settings", "model",
                       "pd_curves", "lgd", "overlay", "other")

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
    ctx: object = None                   # etl.report.EclContext
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
    from ..etl.report import _portfolio_map, build_ecl_context, ecl_frame
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

    # what the run's report was priced on: its context and contract facts
    # (etl.report), one row per position -- see ecl_frame
    ctx = build_ecl_context(out_dir, model_cfg=model_cfg)
    frame = ecl_frame(out_dir, _portfolio_map(static))
    if frame is None:
        return BridgeRun(run_dir, False, f"No AccountMaster in {out_dir}")
    rep["occ"] = rep.groupby("contract").cumcount()
    rep["key"] = rep["contract"].astype(str) + "#" + rep["occ"].astype(str)
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


_SHAPE_WORDS = {"bullet": "Bullet", "linear": "Equal instalments",
                "annuity": "Annuity"}
_SETTING_WORDS = {
    "stage3_method": "Stage 3 provision method",
    "cap_ecl_at_exposure": "Cap the provision at the exposure",
    "ead_fallback.default": "Repayment shape when no rule matches "
                            "(contracts with no repayment schedule)",
}


def settings_changes(ecl_a, ecl_b) -> list[dict]:
    """The ECL settings (model.yml's ``ecl`` block) that differ, in words.

    The EAD fallback rules are matched on portfolio and payment type, not on
    their position in the list, so inserting a rule does not read as every
    later rule changing:

        Repayment shape, Al Dhameen, payment type 4   Bullet -> Equal instalments
    """
    a, b = dict(ecl_a or {}), dict(ecl_b or {})
    out = []

    def word(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if v is None:
            return "(not set)"
        return _SHAPE_WORDS.get(str(v), str(v))

    def rules(cfg):
        got = {}
        for r in ((cfg.get("ead_fallback") or {}).get("rules") or []):
            if isinstance(r, dict):
                key = (str(r.get("portfolio") or "any portfolio"),
                       str(r.get("payment_type") or "any"))
                got[key] = r.get("shape")
        return got

    ra, rb = rules(a), rules(b)
    for key in sorted(set(ra) | set(rb)):
        if ra.get(key) != rb.get(key):
            pf, pt = key
            out.append({"item": f"Repayment shape, {pf}, payment type {pt} "
                                "(no repayment schedule)",
                        "before": word(ra.get(key)) if key in ra else "(no rule)",
                        "after": word(rb.get(key)) if key in rb else "(no rule)"})
    strip = lambda c: {k: (v if k != "ead_fallback" else
                           {kk: vv for kk, vv in (v or {}).items() if kk != "rules"})
                       for k, v in c.items()}
    for d in _dict_diff(strip(a), strip(b), ""):
        k = d["item"].removeprefix(": ")
        out.append({"item": _SETTING_WORDS.get(k, k.replace("_", " ")),
                    "before": word(d["before"]), "after": word(d["after"])})
    return out


def config_changes(a: BridgeRun, b: BridgeRun) -> dict:
    """What changed in the model and in the macro inputs between two runs,
    read from their frozen config. ``known`` is False when either run has
    none, and then neither can be said to have changed or not."""
    if not (a.has_config and b.has_config):
        return {"known": False, "model_changed": None, "macro_changed": None,
                "settings_changed": None, "model": [], "macro": [],
                "settings": []}
    # model.yml's ecl block is the ECL settings, not the PD model
    ecl_a = (a.model_cfg or {}).get("ecl") if isinstance(a.model_cfg, dict) else None
    ecl_b = (b.model_cfg or {}).get("ecl") if isinstance(b.model_cfg, dict) else None
    settings = settings_changes(ecl_a, ecl_b)
    strip = lambda m: {k: v for k, v in (m or {}).items() if k != "ecl"} \
        if isinstance(m, dict) else m
    model = _dict_diff(strip(a.model_cfg), strip(b.model_cfg), "model.yml")
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
            "macro_changed": bool(macro), "settings_changed": bool(settings),
            "model": model, "macro": macro, "settings": settings}


def _hybrid_ctx(a: BridgeRun, b: BridgeRun):
    """The current run's pricing context on PD curves built from ITS macro
    inputs and the PREVIOUS run's model."""
    from ..etl.macro import build_stpd_from_static
    from ..etl.report import EclContext

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
    return b.ctx.copy(stpd=EclContext._stpd(stpd))


def _rebuilt_ctx(r: BridgeRun):
    """A run's pricing context on PD curves rebuilt by THIS engine from the
    run's own frozen config: what it would have priced with on the current
    calculation."""
    from ..etl.macro import build_stpd_from_static
    from ..etl.report import EclContext

    stpd = build_stpd_from_static(r.static, r.model_cfg, r.model_inputs,
                                  r.extract_date or "", model_id=r.model_id)
    return r.ctx.copy(stpd=EclContext._stpd(stpd))


def _same_curves(x: dict, y: dict, tol: float = 1e-9) -> bool:
    """Whether two runs' PD curves are the same calculation: equal to well
    inside what any change in method moves them by (and R and Python build
    them to 1e-12 of each other)."""
    if x.keys() != y.keys():
        return False
    for k, v in x.items():
        w = y[k]
        if v.shape != w.shape or not np.allclose(v, w, rtol=0.0, atol=tol):
            return False
    return True


def _with_settings(ctx, of):
    """``ctx`` (the current run's data) on the ECL settings of ``of``: its EAD
    fallback rules, Stage 3 method and exposure cap."""
    return ctx.copy(rules=of.rules, stage3=of.stage3, cap=of.cap)


# ------------------------------------------------------------ the bridge ----
def ecl_bridge(a: BridgeRun, b: BridgeRun) -> dict:
    """Every contract's move from run ``a`` to run ``b``, split into causes.

    Returns ``rows`` -- one row per position, with both runs' attributes
    (customer, account type, segment, rating, stage, exposure, ECL, PD, LGD)
    and the components, which sum to ``ecl_b - ecl_a`` -- with what changed
    in the config, whether macro and model could be told apart, and notes.
    """
    from ..etl.report import compute_ecl

    if not (a.ok and b.ok):
        return {"ok": False, "reason": a.reason or b.reason}
    changes = config_changes(a, b)
    notes = []
    # The previous run on THIS engine: its PD curves rebuilt from its own
    # frozen config, so that a change in how the curves are calculated (the
    # scenario weighting, say) is Method change rather than Macro or Model.
    # Only when the current run's curves are this engine's: had both runs come
    # from an earlier version, restating one of them would only show the same
    # change twice, once each way.
    ca_m = a.ctx
    if a.has_config and b.has_config:
        try:
            if _same_curves(_rebuilt_ctx(b).stpd, b.ctx.stpd):
                rebuilt = _rebuilt_ctx(a)
                if not _same_curves(rebuilt.stpd, a.ctx.stpd):
                    ca_m = rebuilt
            else:
                notes.append("The current run's PD curves are not what this "
                             "engine builds from its config: it was priced by "
                             "another version of the calculation, and repricing "
                             "does not reproduce it (Other).")
        except Exception as exc:
            notes.append(f"The PD curves could not be rebuilt on this engine "
                         f"({exc}); a change in how they are calculated is in "
                         "the PD-curve steps.")
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
        ctx_h = ca_m           # the macro inputs did not: the model takes it
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
    # the current run's data on the previous run's ECL settings, up to Model
    cb_a = _with_settings(cb, ca)
    same_settings = (ca.rules == cb.rules and ca.stage3 == cb.stage3
                     and ca.cap == cb.cap)
    eff = {k: np.zeros(len(both)) for k in ("method", "exposure", "stage",
                                           "rating", "macro", "settings",
                                           "model", "pd_curves", "lgd",
                                           "other")}
    ok = np.zeros(len(both), dtype=bool)
    not_split_rating = 0
    pos = np.array([k in fa.index and k in fb.index for k in both], dtype=bool)
    keys = [k for k, p_ in zip(both, pos) if p_]
    if keys:
        # Every price is etl.report.compute_ecl() -- the function that priced
        # both reports -- on one run's context, with the contract facts, LGD or
        # PD curve held at the other's where a step has not reached them yet.
        FA, FB = fa.loc[keys], fb.loc[keys]
        sa = FA["stage"].astype(int).to_numpy()
        sb = FB["stage"].astype(int).to_numpy()
        eir_a = FA["eir"].to_numpy()

        def curves(ctx, F):
            return [ctx.pd_curve(p_, r_, t_) for p_, r_, t_ in
                    zip(F["portfolio"], F["rating"], F["rating_type"])]

        cum_a, cum_r = curves(ca_m, FA), curves(ca_m, FB)
        cum_h = curves(ctx_h, FB) if ctx_h is not None else None
        # the current balance, EAD curve and term, on the previous stage and EIR
        FBa = FB.assign(stage=sa, eir=eir_a)
        FBb = FB.assign(eir=eir_a)            # ... and on the current stage

        def price(ctx, F, lgd=None, cum=None):
            return compute_ecl(ctx, F, lgd=lgd, cum=cum)

        p0 = price(ca_m, FA)
        f0 = p0["ecl"].to_numpy()
        la = p0["lgd"].to_numpy()             # the previous LGD, held to f5
        f1 = price(cb_a, FBa, la, cum_a)["ecl"].to_numpy()
        f2 = price(cb_a, FBb, la, cum_a)["ecl"].to_numpy()
        f3 = price(cb_a, FBb, la, cum_r)["ecl"].to_numpy()
        # the previous run had no curve for the current rating: the rating
        # move cannot be priced on its own and travels with the next step
        hold = np.array([r_ is None and a_ is not None and s_ != 3
                         for r_, a_, s_ in zip(cum_r, cum_a, sb)], dtype=bool)
        f3 = np.where(hold, f2, f3)
        f5 = price(cb, FBb, la)["ecl"].to_numpy()
        if ctx_h is not None:
            f4 = price(cb_a, FBb, la, cum_h)["ecl"].to_numpy()
            # the current ECL settings, still on the previous PD model
            f4s = f4 if same_settings else price(cb, FBb, la, cum_h)["ecl"].to_numpy()
        else:
            # no macro step: the curves are the current run's from here, and
            # the settings switch on their own
            f4 = f5 if same_settings else price(cb_a, FBb, la)["ecl"].to_numpy()
            f4s = f5
        f6 = price(cb, FB)["ecl"].to_numpy()

        good = np.all(np.isfinite(np.vstack([f0, f1, f2, f3, f4, f4s, f5, f6])),
                      axis=0)
        not_split_rating = int((hold & good).sum())
        idx = np.where(pos)[0]
        ok[idx] = good
        z = lambda v: np.where(good, v, 0.0)
        eff["exposure"][idx] = z(f1 - f0)
        eff["stage"][idx] = z(f2 - f1)
        eff["rating"][idx] = z(f3 - f2)
        eff["settings"][idx] = z(f4s - f4)
        if split and ctx_h is not None:
            eff["macro"][idx] = z(f4 - f3)
            eff["model"][idx] = z(f5 - f4s)
        elif split:
            eff["other"][idx] += z(f4 - f3)          # neither changed
        else:
            eff["pd_curves"][idx] = z(f4 - f3)
        eff["lgd"][idx] = z(f6 - f5)
        kk = list(keys)
        model_a = (rows.loc[kk, "ecl_a"] - rows.loc[kk, "overlay_a"]).to_numpy()
        model_b = (rows.loc[kk, "ecl_b"] - rows.loc[kk, "overlay_b"]).to_numpy()
        eff["method"][idx] = z(f0 - model_a)
        eff["other"][idx] += z(model_b - f6)

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
    meth = float(np.where(ok, eff["method"], 0.0).sum())
    if abs(meth) >= 0.5:
        notes.append(f"Repricing the previous run on this engine, with its own "
                     f"data and config, moves it by {meth:,.0f}: it was priced "
                     "by another version of the calculation. That is shown as "
                     "Method change, not as a change of model.")
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


def _group(rows: pd.DataFrame, level: str) -> pd.Series:
    """The ONE group a contract is counted in at a level: its group in the
    current run, or -- for a contract that left -- in the previous run. A
    contract is never in two groups, so a group's movement is just the sum of
    its contracts' causes; there is no transfer between groups."""
    ga, gb = _attr(rows, level, "a"), _attr(rows, level, "b")
    return gb.where(rows["status"] != "derecognised", ga)


def _masks(rows, level, members):
    """The selection's contracts, as they stood in each run."""
    if level == "book" or BRIDGE_LEVELS.get(level) is None:
        member = pd.Series(True, index=rows.index)
    else:
        want = {str(m) for m in (members or [])}
        member = _group(rows, level).isin(want).fillna(False).astype(bool)
    in_a = member & (rows["status"] != "new")
    in_b = member & (rows["status"] != "derecognised")
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
    ratings). A contract is counted in the group it is in NOW (a contract
    that left: the group it was in), so the opening is the previous ECL of
    these contracts and the closing their current ECL. For customers,
    facilities, segments and account types that is each run's own figure for
    the group; for a stage or a rating the opening is the previous ECL of the
    contracts in that stage or rating now -- a loan that moved from Stage 1 to
    Stage 2 is in the Stage 2 view, its whole move under Stage migration.
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
    stay = in_b & cont
    for k in ("method", "exposure", "stage", "rating", "macro", "settings",
              "model", "pd_curves", "lgd", "overlay", "other"):
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
                   "new": int((in_b & (rows["status"] == "new")).sum())},
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
    # each contract in one group (see _group); none in a run it is not in
    g = _group(rows, level)
    ga = g.where(status != "new")
    gb = g.where(status != "derecognised")
    groups = sorted(set(g.dropna()), key=str)
    if not groups:
        return pd.DataFrame()

    def total(values, keys, mask=None):
        m = keys.notna() if mask is None else (mask & keys.notna())
        m = m.to_numpy(dtype=bool)
        s = pd.Series(pd.to_numeric(values, errors="coerce").fillna(0.0)
                      .to_numpy(dtype=float)[m], index=keys.to_numpy()[m])
        return s.groupby(level=0).sum().reindex(groups, fill_value=0.0).to_numpy()

    out = pd.DataFrame({"group": groups, "opening": total(rows["ecl_a"], ga)})
    out["derecognised"] = total(rows["derecognised"], ga, status == "derecognised")
    out["new_business"] = total(rows["new_business"], gb, status == "new")
    for k in ("method", "exposure", "stage", "rating", "macro", "settings",
              "model", "pd_curves", "lgd", "overlay", "other"):
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
    g = _group(rows, level)
    a = pd.DataFrame({"value": g, "ecl": rows["ecl_a"]})[rows["status"] != "new"]
    b = pd.DataFrame({"value": g, "ecl": rows["ecl_b"]})[rows["status"] != "derecognised"]
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
