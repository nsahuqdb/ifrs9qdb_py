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
| INPUT | 69 | Validate inputs (preview), pre-run check, every run | pre-run: unsuppressed ERROR blocks Start; in a run: `run.on_validation_error` |
| TRANSFORM | 28 | every run | `run.on_validation_error` |
| DERIVED | 29 | every run | `run.on_validation_error` |
| READY | 20 | pre-run readiness, every run | pre-run: unsuppressed ERROR blocks Start; in a run: `run.on_validation_error` |
| REPORT | 2 | every run, after pricing | records only |

163 check ids, the same in both engines (`tests/r_validator_ids.txt` in the
Python repository is the register). A suppression — a reason, an approver, an
optional expiry, written to the audit log — turns a finding into INFO so it
stops blocking; the checks marked *not suppressible* below cannot be accepted
that way.

**Accepting a finding.** A blocking finding can be accepted in two ways, each
with a reason, and neither for a check marked *not suppressible*:

* **For this run only** -- on the pipeline page (*Accept for this run...*). The
  acceptance is passed to the pre-run check and to the run
  (`accepted_findings =` on `pre_run_check()`, `pre_run_readiness()`,
  `run_etl()` and `run_etl_phase1()`); nothing is written to
  `validation_suppressions.yml`, so the next run asks again. Changing the
  inputs, the config version or the run type clears it.
* **Standing** -- a suppression in `validation_suppressions.yml` (*Validation
  suppressions* page), which applies to every run until it expires or is
  removed. Removing one (`remove_suppression()`) ends it -- `valid_until` set to
  yesterday, with who removed it, when and why -- rather than deleting it, so
  the history stays.

Every run records what was accepted in it: `reports/accepted_findings.csv`
(check, severity, *run* or *standing*, reason, who, when, expiry, and whether
it took effect -- a standing suppression is listed only when its check
failed), an *Accepted findings* section at the end of `validation.md`,
`accepted_findings` in `manifest.json`, and one `finding_accepted` event per
finding in the audit log. `read_run_accepted_findings()` in R and
`ifrs9qdb.runs.read_run_accepted_findings()` in Python read it back; for a run
made before the record existed they rebuild it from `validation.csv` and the
run's copy of the suppressions file (`recorded = FALSE`).

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

The run's reporting date is the AccountMaster EXTRACTDA that most **rows**
carry (a tie goes to the earliest), in both engines. It anchors every output
stamp, the maturity extension, the EAD and PD curves -- one date for the whole
run (`run_reporting_date()` in R). Three checks guard it:

* `INPUT_extract_date_matches_run_cfg` (ERROR): every row of every file carries
  that date, and the message names the rows that do not ("AccountMaster: 3
  row(s) dated 2026-05-31; Collateral: 2 row(s) dated 2026-06-10").
* `INPUT_consistent_extract_date` (WARN): the distinct dates across the files.
* `INPUT_extract_date_plausible` (ERROR, not suppressible): the date is not
  before any contract's OPENDATE, nor after today -- a stale or mistyped
  EXTRACTDA fails no other check when every file carries it. On the June 2026
  book no contract opens after the extract date, so it passes.

### Values the typing cannot read

Every input column is typed (date, number, flag) before any check or the
calculation sees it, with the same rules in both engines: a date is read only
in the shapes the extracts use (M/D/YYYY, YYYY-MM-DD, DD-MON-YY, DD-MON-YYYY,
M/D/YY, an Excel serial), and a date outside 1900-2200 is blank. A value the
typing cannot read -- "31/02/2020", "1,234.5", "4.46%", "Y" in a 0/1 flag --
becomes blank, and the run treats it as missing. `INPUT_values_typed` (WARN)
names each one by file and column; the field checks (ONBALANCE, MATURITYDATE,
the allocation share...) count the blanks with their own severity.

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
| Accepted findings | **Run the pipeline → 3. Pre-run findings**: *Accept for this run...* and *Remove...* for a standing suppression; listed on the paused and the finished run, the **Validation → Accepted findings** tab, **Browse runs**, the **Approval queue**, and the **Audit log** ("Finding accepted", "Suppression removed") | the same: **Runs → Run pipeline** card 3, the paused and finished run, **Browse runs**, the **Approval queue**, the **Audit log**; **Config → Validation suppressions** to remove a standing one |

Helpers: `readiness_reasons()`, `readiness_table_summary()` and
`read_run_readiness()` in R; `ifrs9qdb.runs.read_run_readiness()` and the
Python app's readiness payload return the same table.

## Proof both engines agree

Datasets: the June 2026 extract as delivered; the same with the schedule dates
repaired; a copy with defects injected into every file; a copy with mixed
EXTRACTDA dates; a copy with EXTRACTDA written DD-MON-YYYY; and two built for
the decisions below -- three AccountMaster rows dated 31 May and two Collateral
rows dated 10 June (**stray rows**), and unreadable values in five files
(**garbage**: "31/02/2020" and "2020-13-01" as opening dates,
"1,234.5" as a balance, "2030-02-30" as a maturity, "12,000" as a collateral
value, "4.46%" as an allocation, "Y" as a watch-list flag). Config variants: a
different internal model with a null MEV weight; `mev_model_weights:
auto_p_value`.

- **Pre-run check** (84 checks: config, static, input): identical pass/fail
  and identical messages, R against Python, on all seven datasets.
- **Full runs**: all 148 run checks, `readiness.csv`, `readiness_funnel.csv`
  and all 29 output files identical, R against Python, on every dataset and
  config variant above.
- **Stray rows** price exactly as the clean book (1,916,701,233.09 in both
  engines): the reporting date stays 9 June. Before, R dated that run 31 May.
- **Garbage**: `INPUT_values_typed` names the seven injected values and only
  those, in both engines; the delivered June extract has none.
- **Switched model**: ECL 1,972,982,425 in both; **auto p-value weights**:
  1,960,080,755 in both.
- The pricing-readiness reason table computed by the R app and by the Python
  app on the same run: identical, row for row.

## Found and fixed: the previous round

Found by driving both apps in a browser against the engines, then proving the
fix R against Python.

| # | Finding | Where | Effect before | Now |
|---|---|---|---|---|
| 1 | The config checks tested a different path list from R: `output_dir` and the legacy `model_config` were "required" | Python | spurious ERRORs blocked runs R allows | R's list: `static_dir`, `variable_dictionary`, `models`, `model_inputs`; optional `input_dir`, `reference_outputs`, `data_drop_root` |
| 2 | The pre-run check did not apply the inputs' EXTRACTDA to `run.extract_date` before the config checks | Python | the shipped `config.yml` (empty `extract_date`, by design) blocked every run | applied first, as R's `pre_run_check()` does; `internal_model` is checked too |
| 3 | The PD model was hardcoded to `internal_v4_production`; `run.internal_model` and `mev_model_weights.mode` were ignored, a null weight was not derived from the p-values, an unknown model was not refused | Python | a config version switching either priced on the old model, silently | `etl/model_registry.py` mirrors R's `resolve_model()`; the manifest records the resolved model; the run freezes `config.yml` with a `config_used.yml` marker |
| 4 | Outputs, EAD and PD curves were dated from the latest EXTRACTDA | Python | a mixed-date bundle priced on a different date from R | R's rule: `resolve_input_extract_date()` for the run date; each transform's own latest date for maturity extension |
| 5 | Collateral files stamped with their own dates; `AllocationPercentage` always divided by 100 | Python | differed from R on a mixed-date or fraction-scaled file | the run's date; R's rule (divided by 100 when any value exceeds 1) |
| 6 | No check caught a wrong reporting date | both | a stale or stray EXTRACTDA re-dated the book silently | `INPUT_extract_date_plausible` |
| 7 | An overlay moved the provision but the indicative attribution split it over EAD/PD/LGD (or showed all zeros) | both | factors did not sum to the move | an **Overlay** line; the factors explain the model move |
| 8 | The MEV forecast and weight tables read the default model | both apps | wrong components after a model switch | the run's own model, with R's resolved weights |
| 9 | The Assistant read the model report while the pages read the overlaid one | Python app | two different provisions for one run | the same report as the pages, and it says so |

## Decided and fixed: this round

The open items of the previous round were model-owner decisions. They were
taken as recommended, and both engines changed together, so a figure still
never depends on which one produced it.

| # | Was | Where | Effect before | Now |
|---|---|---|---|---|
| R1 | The reporting date counted distinct *spellings* of EXTRACTDA and broke ties to the earliest | both | one stray row with an earlier date re-dated the whole run (7,000 rows of 1 July and one of 9 June gave 9 June) | the date most **rows** carry; a tie goes to the earliest |
| R1b | `INPUT_extract_date_matches_run_cfg` compared each file's first row | both | a stray row anywhere else passed | every row of every file, with the rows named |
| R1c | The maturity extension anchored on each file's own latest EXTRACTDA | both | a stray later row moved every lapsed maturity | the run's reporting date, as the stamps, EAD and PD curves (`run_reporting_date()`) |
| R2 | The date parsers tried an unanchored list of formats | both | "31-DEC-2025" read as 2020-12-31, "31-12-2025" as the year 31, "6/9/26" as the year 26 | each format only on values of its shape; any date outside 1900-2200 is blank, so the checks report it |
| R3 | `AllocationPercentage` is divided by 100 only if some value exceeds 1 | both | -- | kept, by decision: the extract delivers percentages (0-100), so a file always has values above 1 and is divided by 100; no setting, no new check |
| D | Python typed dates with the format that fitted most of a column; R value by value | Python | "31/12/2025", "2025/12/31" typed in Python, blank in R | R's schema rules, value for value (`ifrs9qdb.dates`) |

And found on the way, while proving those:

| # | Finding | Where | Effect before | Now |
|---|---|---|---|---|
| 10 | The OPENDATE parse checks read the schema-typed column, where an unreadable date is already blank | both | `INPUT_AccountMaster_opendate_parses` (ERROR) could never fail | they count what the typing could not read; `INPUT_values_typed` (WARN, new) names every unreadable value in any date or number column |
| 11 | R typed an ISO-looking date with `as.Date()` and no format | R | one malformed ISO date first in a column stopped the typing of the whole file | an explicit format; the bad value is blank and reported |
| 12 | The Python transforms parsed maturity, opening and schedule dates with a lenient parser | Python | a date R leaves blank was priced in Python | R's schema rules |
| 13 | The Origination schema read columns 1-5, but the file carries EXTRACTDA first | both | the "origination PD" held the contract id (1,065 values `INPUT_values_typed` would call unreadable) | positions 2-6; no number moves (the columns are empty and written blank) |
| 14 | `pre_run_check(snapshot =)` read a version's suppressions from `config/config/` | R | a version's suppressions ignored before the run, applied during it | read where the run reads them |
| 15 | A config version's input, drop and runs folders resolved inside the version | R pre-run check | every version pick warned the input and drop folders missing | the project's folders, in the pre-run check as in the run (`snapshot_run_paths()`); the Python app takes them the same way |
| 16 | Cancel at the pause left the partial run folder | R app | a cancelled run listed under Browse runs | removed, and `run_cancelled` logged, as in the Python app |
| 17 | The R assistant looked for `output/`; runs write `Output/` | R app | no file found on a case-sensitive file system | `Output/`, and the overlaid report when a run carries one |
| 18 | The Python reader read "1,234.5" in an HTML extract as 1234.5 | Python | R blanked it, Python priced it | read as R reads it, and both report it |
| 19 | Python wrote `AllocationPercentage` at full precision; R writes four decimals | Python | a share below 0.00005 (a percentage below 0.005%) priced differently | four decimals, as R: the file is byte-for-byte R's |
