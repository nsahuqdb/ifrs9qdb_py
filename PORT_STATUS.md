# The port — where it stands

Every output is compared against the file the R pipeline produced from the
SAME source extracts, so "done" means identical, not plausible.

```python
from ifrs9qdb.etl import reconciliation_report
print(reconciliation_report("out/", "runs/run_00001/Output"))
```

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

### One question for Risk, not a defect

On the external scale, higher growth gives a higher percentile, a higher SF
and therefore a HIGHER PD — so a downturn LOWERS the provision on Investments
and Banks and FIs. The port reproduces the reference rather than correcting
it, because changing the sign would make the Python disagree with the R and
with every signed figure to date. It needs a decision, not a patch.

## Where the ECL numbers stand

Against the R report: exposure agrees to 0.1% (13.94bn against 13.96bn) and
Stage 3 counts nearly (274 against 276). ECL and the Stage 1/2 split still
differ. The StPD half of that gap is now closed; what remains traces to the
staging inputs, which is the `AccountMaster_1` trailing block above.

## Next

1. Re-measure the ECL report now that StPD is exact — the previous figure was
   taken with the equal-weights StPD underneath it and is not a fair reading.
2. The `AccountMaster_1` trailing block, which also closes both remaining
   `CustomerStagingFlag` differences.
3. The governance surface. Calculator versions, the code fingerprint, input
   acquisition, maker-checker and config snapshots are now ported (below).
4. The analytics layer is the largest remaining gap by line count.
