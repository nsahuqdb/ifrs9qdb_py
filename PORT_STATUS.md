# The port — where it stands

Every output is compared against the file the R pipeline produced from the
SAME source extracts, so "done" means identical, not plausible.

**The ECL report now reproduces the R engine exactly.** On both reference
runs, every contract matches to the cent:

| Run | Contracts | R report | Python | Max abs difference |
| --- | --- | --- | --- | --- |
| run_00001 | 6,709 | 2,283,041,268.74 | 2,283,041,268.74 | 0.000000 |
| run_00002 | 6,481 | 2,193,510,515.24 | 2,193,510,515.24 | 0.000000 |

LGD matches on every contract, and the staging rule disagrees on none. The two
defects that stood in the way are below, under *Collateral netting* and *The
EAD fallback*.

```python
from ifrs9qdb.etl import reconciliation_report
print(reconciliation_report("out/", "runs/run_00001/Output"))
```

## From raw extracts to every file, R against Python — and the run itself

The ETL now reproduces the R engine from the raw extracts: every one of the 29
files a run writes, all 148 run checks, the pricing-readiness report and its
row funnel, identical R against Python on the June 2026 book as delivered, with
its dates repaired, with defects injected into every file, with mixed and
stray EXTRACTDA rows, with unreadable values in five files, and under two model
configurations (a switched internal model with a null weight;
`mev_model_weights: auto_p_value`). The pre-run check -- 84 config, static and
input checks -- gives identical results and messages on all of them. Details, and what was found and fixed on the way, in
`PRICING_READINESS.md`.

What the Python engine now does as R does, that it did not:

* **The run itself** — `run_etl_phase1()` pauses with the customer view for
  rating, stage and restructuring overrides; `run_etl_phase2()` applies them
  and finishes. The project audit log (`audit_log.py`), run discovery and the
  run record (`runs.py`), and the pre-run check and readiness dry run
  (`prerun.py`) match R's events, columns and schema.
* **The model** — `run.internal_model` selects the PD model and
  `mev_model_weights.mode` the MEV weights (`etl/model_registry.py`, R's
  `resolve_model()`); the run freezes `config.yml` beside `config/` and
  records the resolved model in its manifest.
* **The reporting date** — the EXTRACTDA most rows carry dates the run, and
  the maturity extension anchors on it too (decided with R this round: it
  used to count spellings, so one stray row could re-date the run).
* **Dates and typing** — `ifrs9qdb.dates` reads a date exactly as R's schema,
  checks and transforms do, value for value; the schema records what it could
  not read (`INPUT_values_typed`).
* **The allocation share** — divided by 100 when any value is above 1, as R
  does, and written with R's four decimals.
* **The config checks** — R's path and `run:` block rules, with the inputs'
  EXTRACTDA applied first.

## StPD — now exact, and the earlier diagnosis was wrong

`build_stpd_from_static()` reproduces the reference StPD to floating point:
**max absolute difference 6.7e-15 across all 75,600 rows**, on both rating
scales, against two independent reference runs (12/31/2025 and 9/30/2025),
using only the packaged config and static reference.

A previous version of this file reported a mean absolute difference of 0.0058
on the internal scale, attributed it to a "calibration gap", and concluded
that settling it needed the reference run's frozen `config_used/` because the
implied scaling factor was "about 0.11 too low at every rating".

**That diagnosis was wrong in both parts, and it is withdrawn.**

* The frozen `config_used/` was eventually obtained. Its `model.yml`,
  `model_inputs.yml` and every static table are **byte-identical** to the
  packaged ones. There was no calibration difference to find, and no
  recalibrated TTC table: solving the reference for an implied TTC PD of
  0.1400 against the packaged 0.1264 was fitting noise from the real error.
* The scaling-factor chain was never wrong. The SF table printed here
  (Significant Downturn 0.4289, Slight Downturn 0.1905 …) was stale output
  from an earlier revision. The current code returns 0.3159 and 0.1729, which
  is what a line-by-line reproduction of the R returns from the same config.

The actual causes were three, all of them plumbing rather than modelling, and
none visible from the output alone:

1. **`build_stpd_from_static` fell back to EQUAL scenario weights.** With no
   `scenario_weights` argument it used `1/n` per scenario — a fifth each —
   instead of resolving the configured `auto_non_oil_gdp_cdf` block. The
   pipeline passed weights correctly, so a full `run_etl()` was right and
   every direct call, including the test that produced the 0.0058 figure, was
   not. That one default accounted for the entire internal-scale difference.
   Fixing it took the internal scale to exact.
2. **The external scale was given the INTERNAL weights.** The external book is
   weighted from where the REGIONAL forecast sits on the regional history, and
   with a different vector FOR EACH YEAR — the R has a whole
   `apply_scenario_weights_per_year` for it. One flat domestic vector was
   being applied to both.
3. **The GCC history started in 1980, and the forecast was weighted by the
   wrong table.** The R takes the regional series from 1982 (`yrs >= 1982`),
   and requires at least two countries in a year before it will call the
   result an average. Including 1980-81 moves the standard deviation from
   4.4912 to 4.3950, and that SD is the unit every scenario shift is measured
   in. Separately, the forecast was being GDP-weighted by the STATIC table's
   latest year — one fixed weight vector — where the R weights each year by
   the `external_gcc_country_prices` block that sits beside the growth block
   in `model_inputs.yml`. Year 1 came out 4.249 against the correct 4.2569.

Each of the three is pinned by a test.

The lesson worth keeping: a silent default is worse than a missing argument.
Equal weights are a plausible-looking number that is never the right answer,
and they produced a curve set that was monotonic, bounded, correctly shaped
and wrong — which is exactly the shape of thing the shape tests pass.

## Collateral resolved to zero for the whole book

The collateral stress lever moved the provision by exactly 0.00 at every
setting, including "remove all collateral". That is not a small number; it is
the signature of a join returning nothing.

`AccountCollateralAllocation` carries a blank `ContractId`, so pandas reads the
column as float64 rather than int64, and `astype(str)` renders the ids as
`"548840.0"`. The account master's column has no blank, reads as int64, and
renders the same id as `"548840"`. **All 5,932 allocations missed.** Collateral
netted to zero for every contract, so LGD used none of it and the lever had
nothing to scale.

Nothing failed. The dictionary was built, it had 5,933 entries, every lookup
returned the default, and the default is a perfectly reasonable 0.0.

The R is not exposed to this: `as.character()` on an R integer has no trailing
`.0`. It is a pandas-specific hazard and it will recur wherever an id is used
as a key, because whether a column has a blank is a property of the QUARTER,
not of the schema. So ids now go through `ifrs9qdb.ids.as_id()`, which
normalises int, float and text forms to the same string and blanks a missing
value rather than letting it become the key `"nan"`.

Worth knowing: `Series.astype(str)` turns a missing value into the STRING
`"nan"` on pandas 2 and keeps it as `NaN` on pandas 3. The pandas 2 behaviour
is the more dangerous of the two, because `"nan"` is a usable dictionary key,
so every missing id collides into one bucket instead of being dropped.
`as_id()` handles both, with a test for each.

After the fix: 5,932 of 5,933 allocations resolve (the one left is the blank
row, correctly dropped), halving collateral costs 12.3m and removing it
entirely costs 87.6m.

The same pattern has been applied to the ETL's id keys in `report.py`,
`lifetime.py`, `lending.py` and `customer.py`. Those sites are NOT yet verified
against raw extracts — that needs the extract set, which is not in the
reference material available here — so they are a hardening, not a confirmed
fix.

## Validation — the full suite, with the R engine's ids

The port carried 22 checks against the R engine's 114. It now carries all 114,
grouped as the R report groups them, and the counts line up exactly with what
R's own `validation.md` prints:

    Total checks: 114
    INPUT (57)   the raw extracts, cross-file agreement, config coverage
    TRANSFORM (28)   the book after shaping, before pricing
    DERIVED (29)     the curves and weights the engine prices against

plus 15 pre-flight checks on the static reference and `config.yml`, which in
the R run separately from the 114.

**The ids are the contract, and they match R's exactly.** Suppressions are
recorded by validator id, so a Python suite with its own ids (`IN001`,
`TR002`) would silently stop honouring every exception the bank had approved:
the finding returns as a failure, with nothing to say why. Ported ids mean an
exception approved against either engine applies to both, and a finding can be
compared across the two.

A test asserts the parity directly, against a list extracted from the R
sources and bundled at `tests/r_validator_ids.txt`.

### Suppressions

`validation/suppressions.py` mirrors the R module: a suppressed finding still
RUNS and is still recorded, and what changes is that its effective severity
becomes INFO so it stops gating. Each entry carries a reason, an approver and
an optional expiry; nothing is ever deleted, because the file is the audit
trail. Omitting the approver falls back to the logged-in user, but passing a
blank one raises — putting a name in an audit trail that nobody chose is worse
than refusing.

### What the checks are for

The three StPD defects above are the argument for this suite. Every shape test
passed while the curves were out by a mean of 0.0058: they stayed monotonic,
bounded and correctly sized, because a wrong weighting produces a perfectly
well-formed curve set. So the DERIVED checks assert the things a wrong
weighting does NOT preserve — that the weights sum to one, that the two scales
differ from each other, that the curves order correctly by rating, and that a
zero-TTC bucket still exists.

Verified against the R engine's own output: all 29 DERIVED checks pass on
`run_00001`, with one INFO finding — 1,849 EAD curves that rise above their
opening balance, which is the documented revolving-facility case and is pinned
as informational precisely so nobody "fixes" it.

The ETL now runs the suite as a step of its own and writes `reports/
validation.csv` and `reports/validation.md` in the R format, column for column.

## Reproducibility — calculator versions, code state, input acquisition

Three R modules had no Python equivalent at all. All three are now ported.

**`calculator_versions.py`.** Deployment does not always go through git -- a
bank release can be a copied directory -- so the code version is tracked in a
registry rather than inferred from a SHA. Each run records the version it
SELECTED and a FINGERPRINT of the code that actually executed. Either alone is
insufficient: the version can claim what the code is not, and a bare hash names
nothing. Together they make drift visible, and a test asserts exactly that --
register v1.0, edit a file, and `matches_registered` turns False.

The fingerprint is an md5 over the sources with their paths, so a RENAMED file
changes it as much as an edited one, and `__pycache__` is excluded so an
installed package fingerprints the same as a checkout.

One deliberate difference from the R. The R can execute an archived version by
sourcing its files into a fresh environment. Python has no clean equivalent --
rebinding a live package's modules mid-process leaves a state matching neither
version -- so an archived version here is a directory you install or put on the
path, and `calc_version_code_dir()` returns it.

**`code_version.py`.** The git SHA, branch, last commit and whether the tree is
dirty. `dirty` is the one that matters at close: a run produced from a modified
working tree cannot be reproduced from its SHA, and the record should say so.
Everything degrades quietly -- a deployment need not be a checkout, and a
missing SHA is a gap in the record rather than a reason to refuse a run. An
unknown SHA reads as unknown, never as a mismatch.

**`acquisition.py`.** Getting a quarter's extracts in: extract an uploaded zip,
list the data team's dated drop folders, run a cheap structural check, and
record where the inputs came from into the run's own `reports/input_source.yml`.
Without that last one a run records what it produced and not what it read.

Two details worth keeping:

* A new directory per upload, never a reused one. Two uploads in a session must
  not be able to mix, and a half-overwritten bundle produces a run nobody can
  explain afterwards.
* **The zip extraction refuses a member that would write outside the
  destination.** `utils::unzip` in the R does not check this, and a zip can name
  a member `../../etc/passwd`. This is a hardening the R does not have.

The manifest now carries the calculator record and the code state alongside the
engine version and the validation summary.

## Maker-checker, in the R engine's own file

The two engines are meant to share a `runs/` folder — the app README says so —
and that only works if a run written by one is readable and approvable by the
other. The Python had its own two-stage sign-off in `approval.json`; it now
also writes and reads `reports/run_status.yml`, in the R format, with the same
state machine:

    official run   -> pending_checker -> approved
                                      -> rejected
    unofficial run -> unofficial                     (terminal, no approval)

Only `pending_checker` transitions. A rejected run is re-run, not re-argued.

Two rules carry the control, and both REFUSE rather than warn, because a
warning that can be scrolled past is not a gate:

* **A reason is required.** An approval with no reason records that somebody
  clicked, not that somebody decided.
* **Separation of duties.** Whoever ran the pipeline cannot also approve it,
  when the config asks for it. Rejection is always allowed — the person who
  built a run has to be able to withdraw it, and making them find someone else
  to do that achieves nothing.

A run with no status file is listed as `unknown` and SURFACES in the queue
rather than being filtered out. It is the case most worth seeing: something
wrote a run and did not record that it needs approving.

Verified against the R engine's own runs: `run_00001` and `run_00002` both
read as unofficial, weighted, run by nsahu, and are correctly excluded from
the pending queue.

**One interop fix this turned up.** The R engine writes `reports/manifest.json`
and the Python was writing `manifest.json` at the run root, so neither could
find the other's maker — which is what separation of duties is checked
against. The Python now writes where the R reads, and readers accept both.

## Config snapshots — the frozen copy, with its lifecycle

A snapshot freezes the config and the static reference together under a label.
It exists because "what changed?" is the first question asked of a provision
that moved, and the answer has to be a file rather than a memory.

The lifecycle is the R's, exactly:

    draft ──► tested ──► pending_final ──► approved ──► archived
      ▲         │            │   │
      └─────────┘            │   └──► rejected ──► (clone to a new draft)
                             └──────► tested

Three things in it are easy to get wrong and all three are pinned by tests:

* **Only a DRAFT is editable.** That is what `tested` is for — the creator
  locks their own snapshot before impact-testing it, so the numbers being
  tested cannot move underneath the test.
* **Separation of duties applies to the FINAL approval only.** The creator may
  mark their own snapshot tested and submit it; those are their own work. Only
  the approval is gated.
* **A child starts from its PARENT, not from live.** "v3 based on v2" begins
  as v2's frozen content at any depth of the chain. A test changes the live
  config after v1 is cut and asserts v2 still carries v1's value.

**The detail that would have been lost silently:** the static tables carry
their provenance in leading `#` lines — which variable, which source, when it
was last refreshed. A plain read-then-write round trip through pandas deletes
every one of them. `read_static_csv_with_header` and its writer carry the
header separately and put it back, and a test round-trips the R app's own
`non_oil_gdp_history.csv` to prove the seven header lines survive.

The copied `config.yml` is rewritten so its model paths point at the
snapshot's own frozen files rather than at `config/` — without that, a frozen
copy reads the LIVE model files, which is the one thing it must not do.

Verified against the R app's own `config_snapshots/ddd`: it reads, lists, and
its editable file set and comment headers come back intact.

## Overlays — the bundle, the approval trail, and applying one to a run

The Python had the overlay ENGINE (the three types, the selectors, the
overlap check) and nothing around it. It now has the rest of the R module:
the bundle as a person authors it, its approval trail, and applying one to a
completed run.

A bundle is one management adjustment in the form somebody writes it: an id,
an owner, a status and a list of RULES, each picking a level, a target, a
method and a value. It is flattened into single overlays before the engine
sees it. The separation is not cosmetic -- a bundle is reviewed by people and
lives in config, while the flattened overlays exist only for the length of one
calculation.

`apply_overlay_to_run` writes `FinalEclReport_overlay_<id>.csv` and
`OverlayAuditLog_<id>.csv` beside the model report and never modifies it, so
the model number and the adjusted one both stay on disk and the difference
between them is a file anybody can open.

**Verified against the R app's own applied overlay.** Reconstructing `ov1` (a
10% uplift on Business Finance) from its audit log and applying it to the same
report reproduces the R output to the cent:

| | Python | R |
|---|---:|---:|
| Ecl Model Onbal | 2,283,041,268.74 | 2,283,041,268.74 |
| Overlay Amount | 75,821,039.70 | 75,821,039.70 |
| Ecl Final Onbal | 2,358,862,308.44 | 2,358,862,308.44 |
| Contracts matched | 4,966 | 4,966 |

Same 6,709 x 80 shape, and the audit log carries the R's fifteen columns in
the R's order.

Three behaviours are pinned because each is a decision rather than an
accident: `whole_book` means everything EXCEPT Stage 3 (booked manually, so an
overlay on top would double-count); overlapping overlays are REFUSED rather
than compounded (otherwise the order of application decides the provision);
and the approval trail is append-only, so an overlay rejected and later
approved reads as exactly that.

## Reconciliation — which rows differ, not just which files

`compare_outputs` said which files and columns differed. That is enough to
know there is a problem and never enough to fix one: the next question is
always "which rows, and what do they hold?".

`dump_mismatches` answers it, writing up to three files per output:

    <name>_unmatched_actual.csv      keys produced here the reference lacks
    <name>_unmatched_reference.csv   the other way round
    <name>_value_diffs.csv           keys in both, values differing, with the
                                     two values side by side

Only files with a natural key are dumped. Without one the rows can only be
lined up positionally, and a positional "difference" on a file written in a
different order is noise that buries the real ones. The key registry was also
four files short of the R's — `FxRate`, `PortfolioRatingType`, `RatingTypes`
and `CollateralType` were being compared positionally.

`write_reconciliation_markdown` writes the same comparison as a document, for
a reviewer rather than a console.

Two things the tests pin: ids are compared as TEXT, so a file written in a
different order reconciles rather than reporting every row as changed; and a
run reconciles against itself across all 24 files, which is the check that the
comparison is not simply reporting everything as different.

## Reading the extracts — complete

All twelve read. Format is detected from the file's bytes, not its extension:
four arrive as SQL*Plus HTML with an `.xls` name, and which files come in which
format has changed between extract runs.

## The ETL is complete end to end

`run_etl()` goes from the raw Oracle extracts to a priced ECL report in about
22 seconds, writing all 18 LIC input files plus `FinalEclReport.csv` and a
manifest.

    Read inputs         12 of 12 files read
    Lending accounts    7,257 contracts
    Investment accounts 73 accounts
    Collateral          4,034 items, 117,183 allocations
    Customers           705 lending, 73 investment
    EAD curves          5,520 contracts, 82,478 monthly points
    PD curves           21 buckets x 6 portfolios x 600 months
    ECL report          7,330 contracts

### Three joins that are not on the account rows

Each produced a plausible-looking run that priced almost nothing:

* **The portfolio is a LOOKUP from the product type.** Without
  `product_portfolio_mapping` every product becomes its own portfolio and no
  PD curve resolves, because the curves are keyed on the six real portfolios.
  The run completes and prices 347 contracts out of 7,330.
* **The lending rating is a CUSTOMER attribute**, not on the account rows at
  all. It comes from the customer master.
* **A missing maturity date reads as a date object.** `NaT` passes an
  attribute check and fails on conversion partway through the book, so the
  fallback has to test the value, not the type.

## Outputs — 12 of 18 match, and the other 6 are accounted for

| Match (12) | |
|---|---|
| AccountMaster_2, Collateral, AccountCollateralAllocation | |
| CustomerMaster_1, CustomerMaster_2, Origination_2 | |
| LifeTimeParameterOther (82,478 rows) | |
| Portfolios, PortfolioRatingType, Ratings, RatingTypes, FxRate | |

**StPD now joins these** — see above.

### The six that differ, and why

**Three are stale reference data, not port defects.**

* `AccountMaster_1` — 7,257 rows against the reference's 7,250. The input
  file's own SQL*Plus footer reads **"7257 rows selected."**, and the seven
  extra rows are the last seven in the file with full data. The extract has
  grown since the reference run; the port reads it correctly.
* `Origination_1` — the same seven, since it is filtered to the account list.
* `CustomerStagingFlag_2` — two rows carry `IsLocal1 = TRUE` in the reference.
  The R writer now emits `rep(FALSE, n)` unconditionally, so the current R
  would also disagree with its own reference file. The port matches the CODE.

**Two are real and small.**

* `CustomerStagingFlag_1` — one customer out of 616. The STAGE uses the
  watchlist flag on the ACCOUNT records while the IsWatchlist COLUMN comes
  from the staging extract, and for customer 46073 the two disagree.
* `CollateralType` — one column, a haircut written to a different precision.

**One is outstanding.**

* `AccountMaster_1`'s trailing block — 7 rows out of 7,250 sit in a short
  block the export appends after the last page, with a shifted column layout.
  The rule that excludes them has not been found, and guessing one that fits
  this quarter's file would break on the next.

## What comparison caught that reading code would not

Each of these was a wrong answer that looked right:

* **Allocation percentages were a hundredfold out.** The source writes `10.09`
  for ten per cent; LIC wants `0.1009`. This appears as collateral coverage
  above 100%, not as a failure.
* **Truncated SQL aliases map by POSITION.** `CustomerStagingFlag` arrives as
  `I, ISWATCHLIST, I.1, I.2, ISLOCAL1, I.3 …` where every `I.n` is a different
  flag. Name-matching files IsInsolvency under IsDefaultInGCC.
* **CustomerMaster is written almost entirely BLANK.** LIC derives its own
  names, limits and ratings and overwrites anything supplied.
* **The staging flags are DERIVED, not copied.** `IsDefault` is the customer's
  worst DPD over 90 — computed, and blank at source.
* **Investment customers get surrogate ids 1..73**, keyed on the account: 73
  rows against 62 distinct counterparties.
* **Dates stay `M/D/YYYY`.** ISO looked tidier and did not match.

## The staging rule, ported rather than inferred

Every flag derives from the customer's FINAL STAGE:

    is_default    stage is Stage 3
    is_watchlist  the source flag, any facility flags the customer
    is_local1     restructured
    is_local3     stage is Stage 2

and the stage is

    DPD > 90                                            -> Stage 3
    restructured, watchlisted, or threshold < DPD <= 90  -> Stage 2
    otherwise                                            -> Stage 1

applied in that order, so a defaulted customer cannot be pulled back to
Stage 2 by also being watchlisted.

## LifeTimeParameterOther — match, 82,478 rows

* **EAD at a scheduled payment is the balance BEFORE it is applied**
  (`BALANCE + REPAYMENT`). Using the balance after understates exposure by one
  instalment at every point on the curve.
* **The month bound is EXCLUSIVE**: months `0 .. end_month - 1`. A contract
  whose whole schedule falls in the current month produces no curve, which is
  right — there is no future exposure to price.
* **Month 0 is today's outstanding from the account master**, not the
  schedule's first figure. The two differ whenever a payment falls in the
  current month.

**3,324 of 5,520 curves RISE above the current balance**, most sharply at
month 1. That is the schedule carrying committed but undrawn amounts, and it
is why the engine caps ECL at exposure. It is pinned as a test so nobody later
"fixes" a rising curve.

## The PD chain

* **The point-in-time shift is in PROBIT space**:
  `PD_t = Phi(Phi^-1(TTC) + SF_t)`.
* **Beyond the forecast horizon the curve reverts to TTC in LOG space.**
* **Cumulative PD is a SURVIVAL product**, `1 - prod(1 - m)`, not a running sum.
* **Scenarios are weighted on the MARGINALS**, then re-cumulated, and the
  weights are **NOT normalised** — V4's explicit weights sum to 1.0003, a
  rounded snapshot of the computed ones, and the R carries that through.
* **The monthly conversion uses a running SUM**, not the survival formula.
  That is inconsistent with the annual curve, and it is what LIC expects — a
  test says so, because it is exactly the kind of thing someone would "fix".
* **The two scales use OPPOSITE sign conventions.** Internal is the plain
  probit shift; external is Basel ASRF, in which the factor is SUBTRACTED:

      internal   PD = Phi( Phi^-1(TTC) + SF )
      external   PD = Phi( (Phi^-1(TTC) - sqrt(R) * SF) / sqrt(1 - R) )
      with       R  = 0.24 - 0.12 * (1 - exp(-50*PD)) / (1 - exp(-50))

  Both come out right. A test fails if the internal formula is ever applied to
  the external book.
* **A TTC PD of ZERO must be kept.** The engine short-circuits it to a zero
  curve, but the rating still needs a bucket in the output. Filtering on `> 0`
  instead of `>= 0` loses three external grades and 3,600 rows.
* **Only Non-Oil GDP carries weight** in the production model — real estate and
  domestic credit are weighted 0.0. Shocking either changes nothing, and that
  is the model, not a fault. Pinned as a test so a property-price stress that
  returns zero is not mistaken for a broken tool.

### The two scales use opposite factor conventions, and both are right

An earlier version of this file claimed a downturn LOWERS the provision on the
external scale. That was wrong, and it was wrong because the two scales carry
factors of different kinds:

* the internal chain produces a **stress factor** — the probit gap between the
  fitted PD and the anchor — where **positive means worse**, and the formula
  adds it;
* the external chain produces a **shift factor** — the probit of where growth
  sits in its own history — where **positive means better**, and the Vasicek
  form subtracts it.

Each formula consumes its own factor with the matching sign, so both raise the
provision in a downturn. Measured on the reference run, a significant downturn
takes a 2% external rating from 0.0093 to 0.0412 and a 14.6% internal rating
from 0.1504 to 0.2303. Nothing needs deciding; the naming needed fixing, and
it is recorded as M2 in `METHODOLOGY_ISSUES.md`.

## Where the ECL numbers stand

Exact. Both reference runs reproduce to the cent, on every contract, with the
stage and the LGD agreeing on every row too (the table at the top of this
file). Getting there took two fixes, each silent and each worth more than a
percent of the provision.

### Collateral netting — the allocation and the haircut were both missing

The netting formula is

    sum over allocations of
        CollateralValue x AllocationPercentage x (1 - HaircutGeneral)

joining `AccountCollateralAllocation` to `Collateral` to `CollateralType`. The
loader summed raw `CollateralValue`: no allocation share, no haircut.

Both factors are large. An allocation is a SHARE of a collateral record,
commonly a fifth of it. And most QDB collateral types carry
`HaircutGeneral = 1.00` — corporate cheques, comfort assignments, corporate
guarantees are worth nothing for provisioning. Summing the raw value credited
contracts with security they do not have, drove their LGD to the 0.225 floor,
and understated the provision.

It failed in the quiet direction. Nothing raised, every figure looked
plausible, and the damage was concentrated where nobody was looking:

| | Before | After |
| --- | --- | --- |
| Whole book | −2.56% | −0.05% |
| Stage 1 | −0.35% | −0.001% |
| Stage 2 | **−9.65%** | −0.19% |
| Stage 3 | exact | exact |
| Contracts whose LGD differed | 981 | **0** |

### The EAD fallback — shape by portfolio, and the maturity floor

Two things were wrong in the parametric curve used by the 24% of the book with
no supplied schedule.

The shape was resolved from the payment type alone — type 3 bullet, anything
else linear. LIC resolves it from the portfolio AND the payment type, and
defaults to bullet. Al Dhameen type 4 is a bullet there, not a straight line.

More consequential: a facility whose maturity is at or before the extract date
has had its remaining term floored to three months, and it has no remaining
amortisation schedule to run. LIC prices those as bullets over the floored
horizon. Amortising them instead charges two thirds of the exposure where LIC
charges three thirds — a third of the provision on each of them, and it
accounted for 50 of the last 54 differing contracts.

The EY reconciliation keeps the other half of the rule honest: with no
portfolio to resolve on, type 4 must still amortise. A flat curve gives
roughly 1.8x the LIC figure on their contract 11, which is the point their
prose answer got wrong and their own workbook settles.

Both rules are pinned in `tests/test_collateral_and_ead.py`, along with the
contract-by-contract equality against the reference report.

### A contract id is not unique

The October book holds one investment security in two positions, so
`XS2908723328` appears twice in the report AND twice in `AccountMaster`. Three
places assumed the id was a key, and each failed differently:

* the overlay writer raised `InvalidIndexError` on the lookup, and would have
  given both rows the same figure had it not — it now aligns by row;
* the repricing join squared it, two rows against two rows being four, so the
  security was priced twice in every stress and what-if total — the account
  master is now de-duplicated before the join;
* a parity test compared through a merge, which fanned the comparison out —
  it now compares row by row.

None of them errored in a way anybody would have noticed on a book without a
repeated id, which is most books.

The walk was a fourth: it kept the first position and dropped the second, so
its opening fell short of the report by that position's ECL and the security
counted as continuing when one position had left. Positions are now keyed by contract and occurrence, matched in order --
the same key the ECL bridge uses -- in the walk, its drill-down, the flows and
the stage transitions, in both engines.

## The analytics layer — ported

The R package's analytics are now matched function for function, in six
modules under `ifrs9qdb.analytics` plus the stress additions.

| Module | What it answers |
| --- | --- |
| `profile` | where the provision sits: segments, concentration, staging, data quality |
| `walk` | how it moved, reconciling exactly |
| `bridge` | why it moved, contract by contract and cause by cause, at any level (below) |
| `attribution` | *why* it moved — the indicative split, the coverage bridge, and an exact engine-level decomposition |
| `staging` | why names sit where they sit, and who migrated |
| `risk` | the parameters themselves: PD curves, LGD floor, collateral, EAD run-off |
| `scenarios` | the five per-scenario runs, compared and reweighted |
| `model_view` | what the run froze: its config, scenarios, MEV forecast |

Four results are worth recording, because each is a statement about the port
rather than about the book:

* **`staging_consistency` finds 0 mismatches over 13,188 contracts.** It
  re-applies this port's `classify_stage` to two finished R runs and compares
  against the stage the R engine wrote. Every contract agrees. That is the
  strongest available evidence that the staging rule was ported rather than
  approximated.
* **`factor_attribution_exact` reconciles to machine precision.** It reprices
  each contract five times through the engine, substituting one ingredient at
  a time, so the four effects sum to the move by construction. 4,797 of 4,797
  matched contracts priced; nothing fell into `uncovered`.
* **`mev_stress` rebuilds the whole PD chain from the run's frozen config and,
  with no edit, reproduces the run's own provision to 1e-6.** The rebuild is
  therefore measuring the macro path and not itself.
* **A domestic MEV shock leaves Investments and Banks and FIs untouched**, as
  it must: those price off GCC growth. The two rating scales stay separate all
  the way through the rebuilt chain.

### A second sign question for Risk

Shocking Non-Oil GDP down 2pp across every forecast year:

| Weight mode | Change in provision |
| --- | --- |
| `hold` — weights pinned, PD effect only | **+1.16%** |
| `auto` — weights follow the forecast, as configured | **−0.78%** |

The PD effect alone behaves as expected: a worse path raises the provision.
With the weights on `auto_non_oil_gdp_cdf` — which is how the production
config runs — the weighting effect is larger than the PD effect and points the
other way, so a downturn *reduces* the provision.

This is the same class of finding as the external-scale sign above: the port
reproduces it rather than correcting it, and it needs a decision from Risk.
The `weight_mode` switch exists precisely so the two effects can be shown
apart in a review.

## The ECL bridge — why it moved, at any level

`analytics.bridge` reprices every contract from the previous run to the
current one, one ingredient at a time, on each run's own frozen config and
the report's own pricing path:

| Step | What changes |
| --- | --- |
| Exposure | balance, EAD curve and remaining term, at the old stage, rating, curves and LGD |
| Stage migration | the stage: 12-month or lifetime, or Stage 3 at the balance |
| Rating migration | the rating, on the previous run's PD curves |
| Macro variables | PD curves from the current macro inputs on the previous model |
| Model | the model's own change: model.yml, the model chosen, the TTC table, how the MEV models combine |
| LGD & collateral | collateral, so LGD, and the EIR |
| Overlay | the post-model overlay |

plus Derecognised and New business, and Moved out / Moved in when a level is
chosen and a contract changed group. Every contract's steps sum exactly to its
change, so any customer, facility, account type, segment, stage or rating is a
sum of contracts and closes on its own figures (`bridge_view`, `bridge_by`,
`bridge_members`).

What it was checked against:

* **September to December 2025, as priced:** Other is 0 on all 4,794
  continuing contracts, so the repricing reproduces both reports.
* **A copy of a run with only its MEV forecasts changed** moves only Macro
  variables; **only a coefficient changed** moves only Model; **both** split
  with Macro equal to the macro-only copy's, because Macro is measured on the
  old model.
* **R and Python agree to 3e-8 per contract** in all of those cases, and to
  5e-6 on group totals in the billions, at every level.

`bridge_by` is one pass over the contracts, so a level of 8,393 facilities is
0.14 s rather than a bridge per facility.

## The surface is complete

Every public function in the R package's analytics and stress surface has a
Python counterpart, and every one of them is reachable from the app's API.
The R Shiny app's twenty-four analytics tabs all have a page here, and the
app carries one thing the R app has that took porting rather than translating:
the assistant.

Four defects turned up while wiring the last pieces, each silent in its own
way:

* the ETL pipeline weighted the external rating scale with the internal
  scenario weights (up to 1.2e-2 of cumulative PD);
* a value sitting exactly on a band's top edge was dropped from its chart
  (114 and 183 contracts on the two runs, in the most severe LGD bucket);
* three of the app's endpoints — reconcile, output summary and run export —
  imported the engine with a relative path that reaches beyond their package,
  so every one raised on its first call while the page that uses them rendered
  perfectly. A render check is not a functional check;
* the file reconciliation indexed both sides on a key that is not unique,
  which made a comparison of the real pair of runs raise rather than report.

A contract id turned out not to be a key in four separate places: the overlay
writer, the repricing join, the file reconciliation, and a parity test. None
of them errored in a way anybody would notice on a book without a repeated id,
which is most books.

## Model issues are tracked separately

The port reproduces the R engine exactly, which says nothing about whether the
MODEL is right. `METHODOLOGY_ISSUES.md` is the register of what needs
reworking: fourteen items, five of them wrong in a direction that matters,
each with its evidence and each pinned by a characterisation test in
`tests/test_methodology_issues.py` so a fix cannot land unnoticed.

The sign question recorded below is M1 there. What was previously a
second one turned out to be a naming collision, now M2.

## Next

1. The `AccountMaster_1` trailing block, which also closes both remaining
   `CustomerStagingFlag` differences. It no longer affects the ECL — the
   report is exact — but the input files still differ by those rows.
2. The two sign questions above, together, with Risk: the external scale's
   growth-to-PD direction, and the scenario weighting outrunning the PD effect
   under `auto`.
3. A second pair of runs, to confirm the exact reproduction holds on a quarter
   these fixes were not measured against.
