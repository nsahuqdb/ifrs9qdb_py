# Will every row get a number? Pre-run validation and pricing readiness

The ETL can accept every file, pass every structural check and still hand LIC a
book in which some contracts get no ECL, come out blank, or are priced from
incomplete inputs — without an error anywhere. The case that started this: a
collateral id that `AccountCollateralAllocation` references but `Collateral`
does not carry makes LIC return NaN coverage, and **the whole contract's ECL
comes out blank**.

Both engines now answer, before a run starts: *will every contract be priced,
and priced from complete inputs?* The same assessment runs inside every run and
is filed with it. Identical in R (`R/readiness.R`) and Python
(`ifrs9qdb/validation/readiness.py`); this file is kept identical in both
repositories.

## What runs, and when

| Stage | Checks | Runs in | Stops a run |
|---|---:|---|---|
| PREFLIGHT (config + static) | 15 | pre-run check | unsuppressed ERROR blocks Start |
| INPUT | 68 | Validate inputs (preview), pre-run check, every run | pre-run: unsuppressed ERROR blocks Start; in a run: `run.on_validation_error` |
| TRANSFORM | 28 | every run | `run.on_validation_error` |
| DERIVED | 29 | every run | `run.on_validation_error` |
| READY | 20 | pre-run readiness, every run | pre-run: unsuppressed ERROR blocks Start; in a run: `run.on_validation_error` |
| REPORT | 2 | every run, after pricing | records only |

162 check ids, the same in both engines (`tests/r_validator_ids.txt` in the
Python repository is the register). A suppression — a reason, an approver, an
optional expiry, written to the audit log — turns a finding into INFO so it
stops blocking; the checks marked *not suppressible* below cannot be accepted
that way.

**Pre-run readiness** (`pre_run_readiness()` in both engines) runs the
pipeline — load, validate, transform, EAD and PD curves, write the LIC files —
into a temporary folder, assesses every contract, and stops before pricing.
Nothing is added to `runs/`; the audit log records one `pre_run_readiness`
event. About 30–60 seconds on the June 2026 book.

## The four outcomes

| Outcome | What it means |
|---|---|
| **No ECL** | Neither engine can price the contract: no PD curve resolves, or there is no EIR to discount with. |
| **Blank in LIC** | The engines price it, LIC will not: an allocation points at a collateral record that is not there, so LIC's coverage is NaN and the contract's ECL is blank. |
| **Priced — check** | Priced, but from an input known to be wrong or missing; the number is not to be trusted without looking. |
| **Priced** | Priced from complete inputs. |

## The field-impact map

What each input gap does to the number, which check flags it and what fixes
it. Every row is a READY check; the per-contract answer is in
`reports/readiness.csv` (`Outcome`, and the reason codes in `Reasons`).

| Field / link | If it is missing or wrong | Check | Severity | Outcome | Fix |
|---|---|---|---|---|---|
| ContractId (AccountMaster, AccountMasterInvestments) | appears twice: priced and counted twice | READY_contract_unique | ERROR | Priced — check | one row per contract at source |
| Rating → master_rating_scale, account type → portfolio | no PD curve for the rating and portfolio: blank in LIC and both engines | READY_pd_curve | ERROR | **No ECL** | add the rating to `master_rating_scale.csv`, check the portfolio mapping |
| EIR | blank: nothing to discount with | READY_eir_present | ERROR | **No ECL** | supply EIR; the account-type fallback needs one rate per type |
| EIR | outside 0.1%–30%: wrong units, discounting wrong | READY_eir_range | WARN | Priced — check | the extract carries percent (4.2 = 4.2%) |
| OnBalance | blank: priced as zero exposure | READY_exposure_present | ERROR | Priced — check | supply ONBALANCE |
| OnBalance | negative: negative provision | READY_exposure_nonneg | WARN | Priced — check | confirm the credit balance |
| Allocation → CollateralId → Collateral | the collateral record is absent: LIC NaN coverage, **whole contract blank** | READY_collateral_allocation_links | ERROR | **Blank in LIC** | add the collateral record, or remove the allocation |
| Collateral type → CollateralType | no haircut: no benefit in R/Python; LIC's treatment undocumented | READY_collateral_type_mapped | WARN | Priced — check | add the type with its QCB haircut |
| Collateral value | blank on a type that would reduce the loss: benefit lost, LGD up | READY_collateral_value_present | WARN | Priced — check | fix the appraisal value |
| AllocationPercentage | blank or outside 0–100%: benefit lost or overstated | READY_allocation_share_valid | WARN | Priced — check | correct the share at source |
| Allocation → ContractId | the row names no contract: dropped | READY_allocation_contract_present | WARN | — | correct the allocation |
| MaturityDate | unusable: horizon floored to 3 months, Stage 2 lifetime loss understated | READY_maturity_present | WARN | Priced — check | supply MATURITYDATE |
| MaturityDate | on or before the extract date: priced as a 3-month bullet, as LIC does | READY_maturity_after_extract | INFO | Priced | none; listed so expired facilities are visible |
| RepaymentSchedule dates | curve stops > 12 months before maturity with balance outstanding (two-digit years read as 19xx): lifetime ECL understated | READY_ead_curve_complete | ERROR | Priced — check | re-extract the schedule with four-digit years |
| CustomerStagingFlag row | missing: watchlist, default and local flags unknown, stage rests on DPD alone | READY_staging_flags_present | WARN | Priced — check | add the customer to the extract |
| PastDueDays | blank: treated as not past due | READY_dpd_present | WARN | Priced — check | supply PASTDUEDAYS |
| OnBalance = 0 with a schedule | ECL comes from the schedule, not capped | READY_zero_exposure_schedule | INFO | Priced | drop a closed facility's schedule |
| Schedule and fallback rule | neither: the EAD is held flat (bullet) | READY_payment_type_rule | INFO | Priced | add an `ecl.ead_fallback` rule in `model.yml` |
| AccountType → product_portfolio_mapping | unmapped: priced as Business Finance | READY_portfolio_mapped | WARN | Priced — check | add the account type to the mapping |
| every raw row | a row that never reaches the LIC files is a contract never priced | READY_rows_carried | ERROR | — | see the row funnel |

After pricing, two REPORT checks confirm the prediction: `REPORT_rows_complete`
(every contract has exactly one report row) and `REPORT_ecl_populated` (every
row carries an ECL, and every blank one was predicted by a READY finding — an
unpredicted blank says "investigate").

The links compound: collateral benefit needs the allocation, the collateral
record, its value and a mapped type all at once, and each missing link gives a
different answer — a blank contract (record missing), a lost benefit (value or
type missing), or a wrong one (share out of range). The map above is the full
chain.

### The reporting date

`INPUT_extract_date_plausible` (ERROR, not suppressible, new in both engines)
refuses a reporting date that is before any contract's OPENDATE or after
today. The reporting date anchors every output stamp, maturity extension, EAD
and PD curve, and a wrong one fails no other check when every file carries it —
a stale or mistyped EXTRACTDA, or one stray earlier row (see R1 below). On the
June 2026 book no contract opens after the extract date, so it passes.

## The row funnel

`reports/readiness_funnel.csv` follows every input file from raw rows, through
the repeated-header rows the reader strips, to the rows written to the LIC
files, and states the rows lost and their exposure. A lost row is a contract
LIC never sees; `READY_rows_carried` fails on any.

## Where it shows

| | Python app | R app |
|---|---|---|
| Before a run | **Run the pipeline → 3. Pre-run findings**: tiles (contracts, no ECL, blank in LIC, priced — check), the verdict, the READY findings, reasons and fixes, the funnel, the contracts with a gap. Start stays disabled while any unsuppressed ERROR stands. | **Runs → Run pipeline → 3. Pre-run findings**: the same, and the same gate on Start. |
| After a run | **Browse runs → Readiness** tab; **Validation** page | **Browse runs → Readiness** tab |
| Audit log | "Pricing readiness: N contracts, n with no ECL, n blank in LIC, n priced with a gap" | the same sentence |
| Files | `reports/readiness.csv`, `readiness_funnel.csv`, `readiness.md` | the same |

Helpers: `readiness_reasons()`, `readiness_table_summary()` and
`read_run_readiness()` in R; `ifrs9qdb.runs.read_run_readiness()` and the
Python app's readiness payload return the same table.

## Proof both engines agree

Datasets: the June 2026 extract as delivered; the same with the schedule dates
repaired; a copy with defects injected into every file; a copy with one stray
EXTRACTDA row; a copy with EXTRACTDA written DD-MON-YYYY. Config variants: a
different internal model with a null MEV weight; `mev_model_weights: auto_p_value`.

- **Pre-run check** (83 checks: config, static, input): identical pass/fail and
  identical messages on all five datasets.
- **Full runs**: all 147 run checks, `readiness.csv`, `readiness_funnel.csv`
  and all 29 output files identical, R against Python, on the injected-defect,
  stray-date, switched-model and auto-p-value runs; the delivered and repaired
  runs identical except for the check added after the R runs were made.
- **Switched model**: ECL 1,972,982,425 in both; **auto p-value weights**:
  1,960,080,755 in both (before this round Python silently ignored both
  settings).
- The pricing-readiness reason table computed by the R app and by the Python
  app on the same run: identical, row for row.

## Found and fixed in this round

Found by driving both apps in a browser against the engines, then proving the
fix R against Python.

| # | Finding | Where | Effect before | Now |
|---|---|---|---|---|
| 1 | The config checks tested a different path list from R: `output_dir` and the legacy `model_config` were "required" | Python | spurious ERRORs blocked runs R allows | R's list: `static_dir`, `variable_dictionary`, `models`, `model_inputs`; optional `input_dir`, `reference_outputs`, `data_drop_root` |
| 2 | The pre-run check did not apply the inputs' EXTRACTDA to `run.extract_date` before the config checks | Python | the shipped `config.yml` (empty `extract_date`, by design) blocked every run | applied first, as R's `pre_run_check()` does; `internal_model` is checked too |
| 3 | The PD model was hardcoded to `internal_v4_production`; `run.internal_model` and `mev_model_weights.mode` were ignored, a null weight was not derived from the p-values, an unknown model was not refused | Python | a config version switching either priced on the old model, silently | `etl/model_registry.py` mirrors R's `resolve_model()`; the manifest records the resolved model; the run freezes `config.yml` with a `config_used.yml` marker |
| 4 | Outputs, EAD and PD curves were dated from the latest EXTRACTDA | Python | a mixed-date bundle priced on a different date from R | R's rule: `resolve_input_extract_date()` for the run date; each transform's own latest date for maturity extension |
| 5 | Collateral files stamped with their own dates; `AllocationPercentage` always divided by 100 | Python | differed from R on a mixed-date or fraction-scaled file | the run's date; R's scale detection |
| 6 | No check caught a wrong reporting date | both | a stale or stray EXTRACTDA re-dated the book silently | `INPUT_extract_date_plausible` |
| 7 | An overlay moved the provision but the indicative attribution split it over EAD/PD/LGD (or showed all zeros) | both | factors did not sum to the move | an **Overlay** line; the factors explain the model move |
| 8 | The MEV forecast and weight tables read the default model | both apps | wrong components after a model switch | the run's own model, with R's resolved weights |
| 9 | The Assistant read the model report while the pages read the overlaid one | Python app | two different provisions for one run | the same report as the pages, and it says so |

## R behaviours mirrored, not changed

These are decisions for the model owners; both engines behave identically so a
figure never depends on which one produced it.

- **R1 — one stray row can set the reporting date.** `resolve_input_extract_date()`
  counts distinct *spellings* of EXTRACTDA, not rows, and breaks ties to the
  earliest date. 7,000 rows dated 1 July and one dated 9 June give a run dated
  9 June. Flagged by `INPUT_extract_date_matches_run_cfg` (ERROR, when a file's
  first row disagrees), `INPUT_consistent_extract_date` (WARN) and now
  `INPUT_extract_date_plausible` (ERROR). Recommendation: count rows.
- **R2 — DD-MON-YYYY in the helper parsers.** `.parse_any_date()` and
  `normalise_extract_date()` try `%d-%b-%y` before `%d-%b-%Y` and ignore trailing
  text, so "31-DEC-2025" reads as 2020-12-31. Latent: the schema layer types
  every date column first with anchored patterns, so the pipeline never hands
  these parsers the raw string (a DD-MON-YYYY extract run through both engines'
  pre-run check and phase 1 was dated 2026-06-09, correctly). Recommendation: anchor the formats as the schema layer does.
- **R3 — allocation scale by detection.** `AllocationPercentage` is divided by
  100 only if some value exceeds 1; a percent file whose every allocation is at
  most 1% would be read as fractions, 100 times too large. Recommendation: state
  the unit in config.
- **Known difference, deliberate:** Python's schema layer parses a date column
  with the format that fits the most values; R's tries guarded formats value by
  value. Identical on every extract seen; a DD/MM/YYYY column with days above
  12 would differ (R returns NA for those values).
