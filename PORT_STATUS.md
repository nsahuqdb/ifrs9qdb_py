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
3. The governance surface: snapshots, approvals and calculator versions are
   thinner here than in the R package.
