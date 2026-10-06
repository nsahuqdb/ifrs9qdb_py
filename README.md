# ifrs9qdb (Python)

The IFRS 9 expected credit loss engine used by QDB's Financial Risk Management
function: reads the bank's portfolio extracts, applies the staging rules and
the satellite macro model, builds PD term structures and monthly EAD curves,
computes ECL, and writes the standardised CSV set consumed by EY's LIC
Solution. Also provides the analytics and stress-testing layer over a
completed run.

This is a port of the `ifrs9qdb` R package, validated against it rather than
written to resemble it: every output is compared with the file the R pipeline
produced from the same source extracts, so "done" means identical, not
plausible.

The package calculates and nothing else. The web interface lives in
`ifrs9_app_py` and imports this package, so a figure on a screen can be
reproduced in a notebook with the same call.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

## Where the numbers stand

| Component | Against the R reference |
|---|---|
| ECL engine, 4 EY reconciliation contracts | exact (ratio 1.000000) |
| ECL engine, 12 run-324 contracts | within 0.01% of LIC |
| **StPD, all 75,600 rows, both rating scales** | **exact (max abs diff 6.7e-15)** |
| LifeTimeParameterOther, 82,478 rows | match |
| 12 of 18 LIC input files | match |
| Collateral netting, 5,932 allocations | resolve (were silently missing all) |
| Overlay applied to a run | matches the R output to the cent |
| **Validation suite** | **114 checks, R's ids, same INPUT/TRANSFORM/DERIVED split** |

The StPD figure holds against two independent reference runs
(`run_00001` at 12/31/2025 and `run_00002` at 9/30/2025) using only the
packaged config and static reference — both of which are byte-identical to the
R package's `inst/`.

`PORT_STATUS.md` records what is done, what is outstanding, and the findings
that cost the most to reach.

## Using it

```python
from ifrs9qdb.etl import run_etl, reconciliation_report

run_etl("extracts/", "runs/")
print(reconciliation_report("runs/run_00002/Output", "runs/run_00001/Output"))
```

From the command line:

```bash
python -m ifrs9qdb etl --input extracts/ --runs runs/
python -m ifrs9qdb stpd --date 12/31/2025 --out StPD.csv
python -m ifrs9qdb validate --run runs/run_00001
python -m ifrs9qdb reconcile --produced out/ --expected runs/run_00001/Output
```

## What is here

```
src/ifrs9qdb/
  engine.py       LGD, EAD curves, the marginal-loss sum, Stage 3, the cap
  inputs.py       loading a run's inputs
  etl/            extracts in, the eighteen LIC files out
    read_inputs   format detection, the twelve extracts
    transform     collateral, allocation, staging flags, origination
    lending       AccountMaster and the investment book
    lifetime      EAD curves from the repayment schedule
    macro         the PD chain: MEVs to scaling factors to StPD
    model_registry  which PD model config.yml names, resolved as R does
    report        FinalEclReport
    pipeline      run_etl end to end, with a manifest; run_etl_phase1 /
                  run_etl_phase2 pause for customer overrides between
    reconcile     compare a produced run against a reference, and
                  write out the rows that differ
  analytics/      walk, staging, concentration, data quality, profiles
  stress.py       what-if, packages, reverse stress, roll-forward
  validation/     148 run checks with the R engine's ids -- input,
                  transform, derived, pricing readiness (READY) and report
                  -- plus 15 pre-flight checks on the config and static
                  reference, and the suppressions file (reason, approver,
                  expiry; ended, never deleted). Findings can also be
                  accepted for one run only (accepted_findings=): the run
                  records every finding accepted in it, and why, in
                  reports/accepted_findings.csv. See PRICING_READINESS.md
  prerun.py       the pre-run check and the pricing-readiness dry run: which
                  contracts would get no ECL, or a blank in LIC, before a run
  runs.py         the runs folder: list, manifest, validation, readiness,
                  overrides, outputs
  audit_log.py    the project audit log, in the R engine's events
  governance.py   two-stage sign-off, audit log, config snapshots
  run_status.py   maker-checker, in the R engine's reports/run_status.yml
  snapshots.py    frozen, versioned config + static, with its own lifecycle
  calculator_versions.py  the code-version registry and its fingerprint
  code_version.py the git state a run was produced from
  acquisition.py  getting a quarter's extracts in: zip upload, data drops,
                  a structural check, and recording where they came from
  ids.py          ids that join, whatever dtype the column was read as
  dates.py        dates read exactly as the R engine reads them: the
                  schema's typing, the checks' and the transforms' parsers
  overlays.py     management overlays: the engine, the bundle a
                  person authors, its approval trail, and applying
                  one to a completed run without touching it
  config/         model.yml, model_inputs.yml, overlays.yml
  static/         rating scales, TTC PDs, scenario severities, mappings
```

## Tests and reference data

The fixtures under `tests/fixtures/` ship with the repository, so the suite
means something on a fresh clone with nothing configured:

```bash
pytest             # 677 passed, 11 skipped with reference runs configured
                   #   (below); without them the data tests skip
```

The reconciliation tests need a real run, and a real run is real portfolio
data — customer identifiers, exposures, provisions. That is deliberately not
in git. Point the environment at a copy instead:

```bash
export IFRS9_REF_RUN=/path/to/runs/run_00001     # holds Output/
export IFRS9_SRC_INPUTS=/path/to/extracts        # the raw Oracle files
export IFRS9_REF_RUN_PREV=/path/to/runs/run_00002   # optional, the run before
pytest -q                                        # the data tests now run too
```

Without them those tests skip and name the variable that was missing, rather
than failing or — worse — passing silently on absent data.

The config and static reference are not in that category: they ship inside the
package, so anything needing only those runs everywhere.
