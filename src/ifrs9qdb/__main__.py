"""Command line entry point for the calculation package.

    python -m ifrs9qdb etl --input IN --runs runs/
    python -m ifrs9qdb stpd --out StPD.csv --date 12/31/2025
    python -m ifrs9qdb validate --run runs/run_00001
    python -m ifrs9qdb reconcile --produced out/ --expected runs/run_00001/Output
    python -m ifrs9qdb version

This package calculates; it does not serve. The web interface lives in the
ifrs9_app_py repository and imports this package, so a figure on a screen and a
figure in a notebook come from the same call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_etl(args) -> int:
    from .etl.pipeline import run_etl
    result = run_etl(Path(args.input).expanduser(), Path(args.runs).expanduser())
    print(f"  run       {getattr(result, 'run_id', '?')}")
    for line in getattr(result, "summary_lines", []) or []:
        print(f"  {line}")
    pending = getattr(result, "pending", None)
    if pending:
        print("  pending:  " + ", ".join(pending))
    return 0


def _cmd_stpd(args) -> int:
    import yaml

    from .etl.macro import build_stpd_from_static
    from .etl.static_ref import load_static_reference

    cfg = Path(args.config).expanduser() if args.config else None
    static = load_static_reference(args.static and Path(args.static).expanduser())
    if cfg is None:
        pkg = Path(__file__).parent / "config"
        cfg = pkg
    model = yaml.safe_load((cfg / "model.yml").read_text(encoding="utf-8"))
    inputs = yaml.safe_load((cfg / "model_inputs.yml").read_text(encoding="utf-8"))
    # config.yml, when the config folder carries one, names the model
    from .etl.model_registry import model_id_from_run_config
    rc_path = cfg / "config.yml"
    rc = yaml.safe_load(rc_path.read_text(encoding="utf-8")) \
        if rc_path.is_file() else None
    out = build_stpd_from_static(static, model, inputs, args.date,
                                 model_id=model_id_from_run_config(rc))
    out.to_csv(args.out, index=False)
    print(f"  wrote {args.out}  ({len(out):,} rows)")
    return 0


def _cmd_validate(args) -> int:
    from .validation import validate_run
    issues = validate_run(Path(args.run).expanduser())
    frame = getattr(issues, "to_frame", None)
    rows = frame() if callable(frame) else issues
    n = len(rows)
    print(f"  {n} finding{'s' if n != 1 else ''}")
    if args.out:
        rows.to_csv(args.out, index=False)
        print(f"  wrote {args.out}")
    return 0


def _cmd_reconcile(args) -> int:
    from .etl import reconciliation_report
    print(reconciliation_report(Path(args.produced).expanduser(),
                                Path(args.expected).expanduser()))
    return 0


def _cmd_version(args) -> int:
    from . import __version__
    print(__version__)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ifrs9qdb",
        description="IFRS 9 ECL calculation engine and ETL (QDB).")
    sub = p.add_subparsers(dest="command")

    e = sub.add_parser("etl", help="Read the extracts and build a run.")
    e.add_argument("--input", required=True, help="Directory holding the extracts.")
    e.add_argument("--runs", default="runs", help="Where the run folder is written.")
    e.set_defaults(func=_cmd_etl)

    s = sub.add_parser("stpd", help="Build StPD.csv from the static reference.")
    s.add_argument("--out", default="StPD.csv")
    s.add_argument("--date", required=True, help="Extract date, e.g. 12/31/2025")
    s.add_argument("--config", help="Config directory (default: the packaged one).")
    s.add_argument("--static", help="Static directory (default: the packaged one).")
    s.set_defaults(func=_cmd_stpd)

    v = sub.add_parser("validate", help="Run the validation suite over a run.")
    v.add_argument("--run", required=True)
    v.add_argument("--out", help="Write the findings to this CSV.")
    v.set_defaults(func=_cmd_validate)

    r = sub.add_parser("reconcile", help="Compare a produced output against a reference.")
    r.add_argument("--produced", required=True)
    r.add_argument("--expected", required=True)
    r.set_defaults(func=_cmd_reconcile)

    sub.add_parser("version", help="Print the engine version.").set_defaults(
        func=_cmd_version)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 2
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
