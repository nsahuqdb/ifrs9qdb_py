# Input-side issues, and how Excel, R and Python each handle them

A companion to `METHODOLOGY_ISSUES.md`. That register covers the MODEL. This one
covers the **extract** — what arrives from Oracle before any calculation — and,
for each item, what the three implementations do with it: the Excel tool, the R
package, and the Python port.

**Nothing here has been changed in the code.**

The distinction matters because the two categories need different owners. A
methodology item is a decision for Risk. An input item is a fix in the source
extract or in validation, and until it is fixed every downstream engine
inherits it — the Excel tool, the R package and the Python port alike, because
all three faithfully implement the same rule on the same bad data.

Evidence is the raw extract of **2026-06-09**: `RepaymentSchedule.xlsx`
(85,160 rows, 5,705 contracts), `AccountMaster.xlsx` (7,259 contracts), and the
delivered runs for 2025-09-30 and 2025-12-31.

| | Issue | Severity |
| --- | --- | --- |
| I1 | Schedule dates lose their century: 2030–2043 arrive as 1930–1943 | **A** |
| I2 | No validator looks at a schedule date at all | **A** |
| I3 | The extract contradicts itself on maturity, and nothing compares the two | **A** |
| I4 | `PastDueDays` is blank on contracts holding 450m, and blank reads as current | **B** |
| I5 | The collateral over-allocation check is on the wrong axis | **B** |
| I6 | Ten AccountMaster columns arrive identically zero | **B** |
| I7 | Zero and negative outstanding balances | **C** |
| X1 | Excel and R/Python disagree on the months before the first payment | **B** |
| X2 | R has an annuity shape; Python does not | **C** |
| X3 | The `ead_fallback` config block is live in R and inert in Python | **B** |
| X4 | Python carries a fallback rule the R config does not — dormant, not fixed | **B** |

---

## The worked example, end to end

An eighteen-year annual-payment facility, which is the case that exposes all of
this at once. Contract `599073`, from the 2026-06-09 extract:

* `ONBALANCE` **121,636,306**
* `MATURITYDAT` **2043-01-30** → **199 months** remaining
* `PAYMENTFREQUENCY` 12 (annual), 17 scheduled payments

Its schedule arrives like this:

```
START_DAT   MonthLifetime     REPAYMENT       BALANCE
2027-01-30              7    2,594,948    119,332,783     <- survives
2028-01-30             19    2,594,948    119,157,638     <- survives
2029-01-30             31    2,594,948    118,985,562     <- survives
1930-01-30         -1,157    3,594,948    117,803,377     <- dropped
1931-01-30         -1,145    4,594,948    115,597,219     <- dropped
   ...                ...          ...            ...
1943-01-30         -1,001   17,867,151              0     <- dropped
```

Fourteen of the seventeen payments carry a 1930s year. Restore the century and
the schedule is perfect: 2027, 2028, … 2043, ending **2043-01-30 at a balance of
0.00** — the same date `AccountMaster` already gives as the maturity.

**What each implementation does with it — all three agree, and all three are wrong:**

| | Excel tool | R package | Python port |
| --- | --- | --- | --- |
| end month | `MAX(MonthLifetime)` = **31** | same = **31** | same = **31** |
| curve emitted | months 0–30 | months 0–30 | months 0–30 |
| months priced | **31 of 199** | **31 of 199** | **31 of 199** |
| balance where the curve stops | **121,580,510** | same | same |

**100.0% of the exposure is still outstanding when the curve ends.** The
facility is back-loaded — instalments run from 2.59m up to 17.87m — so the three
surviving years repay 2.6m of 121.6m. Everything else is priced at zero.

This is worse than the intuition that prompted the check. It is not "100m
becomes 90m and then the curve stops"; here the curve stops with the balance
essentially untouched, because the payments that actually retire the principal
are exactly the ones whose dates were mangled.

**Where the three implementations differ on this contract:** months 1–6.

| month | Excel | R and Python |
| --- | --- | --- |
| 0 | 121,636,306 | 121,636,306 |
| 1–6 | **121,927,731** | **121,636,306** |
| 7–18 | 121,752,586 | 121,752,586 |
| 19–30 | 121,580,510 | 121,580,510 |

See **X1**. Six months differ; the curve sum differs by **1,748,551** — the
largest single-contract divergence in the book.

---

## I1 — Schedule dates lose their century · **A**

**25,466 of 85,160 rows (29.9%), across 1,124 contracts**, carry `START_DAT`
between 1930-01-01 and 1943-01-30. The non-wrapped rows span 2026–2029 and
nothing else. That range is exactly the two-digit-year pivot: `30`–`99` →
1930–1999, `00`–`29` → 2000–2029. They are payments due **2030 to 2043**.

Four checks, all passed by adding a century and failed without it:

| check | repaired | as delivered |
| --- | --- | --- |
| balance path monotone non-increasing | holds | fails |
| roll-forward identity on every row | holds | fails |
| last payment equals `AccountMaster.MATURITYDAT` | **1,124 / 1,124** | **0 / 1,124** |
| final balance at that payment is 0.00 | holds | 2.4bn outstanding |

Median error in the implied end month, as delivered: **39 months**.

**The cheapest possible detection:** all 25,466 wrapped rows have `START_DAT`
**earlier than `POST_DATE`** — a scheduled payment dated before the schedule was
written. One comparison, zero false positives on this extract.

The R source calls these "placeholder historical schedule entries … a sentinel
date for the final balloon payment of a fully-amortising loan". They are not
sentinels: the median affected contract has **19** of them (max 126), and only
**55 of 1,124** contracts have a single wrapped row.

**Handling:**

* **Excel tool** — keeps the rows for `MIN`/`MAX`, so they set
  `first_schedule_month` to a large negative number but never win the
  forward lookup. `MAX(MonthLifetime)` therefore stops at the last pre-2030
  payment and the curve ends there.
* **R package** — `R/lifetime_parameter_other.R:94` drops `month_lifetime < 0`
  before computing `MIN`/`MAX`, so the same truncation happens one step earlier.
* **Python port** — `src/ifrs9qdb/etl/lifetime.py` filters `month >= 0`
  identically. Byte-for-byte the same curve as R.

All three then take the ECL horizon from the curve's own length
(`R/ecl_ead_curve.R:260`, `resolve_ead_curve` in both ports) and never consult
`MATURITYDAT`, so the shortened curve becomes the contract's lifetime. See
**M15** for the provision impact: Stage 2 understated by 13.7m–17.8m on
run_00001 and 9.4m–13.1m on run_00002.

**18 contracts have every row wrapped.** They get no curve at all and fall to
the parametric fallback (on-balance 3,119,064), where the shape is guessed from
the payment type — see **M5**.

---

## I2 — No validator looks at a schedule date at all · **A**

`build_input_validators()` in `R/validators_input.R` defines **28** rules. Not
one of them reads `RepaymentSchedule.START_DAT`. There is no parse check, no
range check, no ordering check, no comparison against `POST_DATE` or
`MATURITYDAT`.

The only schedule rule is `INPUT_RS_coverage`, which checks that every
AccountMaster contract **has rows**. It says nothing about what is in them. It
passes on this extract.

For contrast, `AccountMaster.OPENDATE` does get a parse check
(`INPUT_AccountMaster_opendate_parses`, ERROR) and a
`maturity_date >= open_date` check (WARN). The schedule — the file that sets the
ECL horizon for 4,225 contracts — has neither.

**Handling:** the Excel tool has no validation layer at all. R runs the 28 rules
and passes. The Python port reproduces the same rule set, so it also passes.
**All three accept the extract without comment.**

The rules that would have caught I1, cheapest first:

1. `START_DAT >= POST_DATE` — catches all 25,466 rows.
2. `YEAR(START_DAT) >= YEAR(extract_date)` — same.
3. `MAX(START_DAT) per contract == AccountMaster.MATURITYDAT` (±1 month) — see **I3**.

---

## I3 — The extract contradicts itself on maturity · **A**

`AccountMaster` carries the **correct** maturity and `RepaymentSchedule` the
**corrupted** one, in the same extract, for the same contract.

`AccountMaster.MATURITYDAT` ranges 2013-06-15 to **2043-01-30**, has no missing
values, no year below 1950, none before `OPENDATE`, and none more than forty
years out. It is clean. And 2043-01-30 is exactly contract `599073`'s true final
payment — the date the schedule delivers as 1943-01-30.

So the information needed to detect and repair I1 is already inside the file
that was shipped. Nothing compares the two.

**Handling:** no implementation performs this cross-check. Worse, the engine
consults `MATURITYDAT` only when a schedule is **absent**
(`resolve_ead_curve` reaches `months_to_mat` in the fallback branch alone). When
a schedule is present — the case where the two could be compared — the maturity
date is ignored entirely, in all three.

915 contracts have `MATURITYDAT` at or before the extract date, which is a
separate matter handled by the three-month floor (see **M4**).

---

## I4 — `PastDueDays` blank on contracts holding 450m · **B**

`PASTDUEDAYS` is missing on **131 of 7,259 contracts** (on-balance
**577,160,140**). What happens next depends on the contract's siblings, because
only the **customer's worst** DPD is written out — the per-contract value never
reaches the report:

```r
past_dues_days[is.na(past_dues_days)] <- 0        # R/transform_lending.R:292
worst_dpd <- ave(past_dues_days, customer_id, FUN = max)
```
```python
worst_dpd = out.groupby("customer_id")["past_dues_days"].transform("max")
out["past_dues_worst"] = worst_dpd.fillna(0)      # etl/lending.py:179-180
```

The two fill the blank at different points — R before the customer maximum,
Python after — but the answer is the same, because zero never raises a maximum.
**No divergence here.** Splitting the 131:

| | contracts | on-balance | outcome |
| --- | --- | --- | --- |
| customer has a sibling facility with a DPD | 8 | 127,015,867 | inherits the sibling's value — 3 of the 8 inherit a DPD above 30, one of 740 days. Handled correctly. |
| **every** facility of the customer is blank | **123** | **450,144,273** | **both ports land on 0 — fully current** |

So the exposure genuinely at risk is **450m, not 577m**, and for those 123
contracts a blank is being read as the most favourable value available: the
30-day backstop and the 90-day default trigger cannot fire. No validator checks
for a blank DPD.

A missing DPD is not evidence of a current account. Treating it as one on 450m
of exposure is a choice, and it is being made silently.

---

## I5 — The collateral over-allocation check is on the wrong axis · **B**

`INPUT_ACA_total_allocation_per_contract` sums `AllocationPercentage`
**per ContractId** and warns above 100.1%. Its stated rationale is that
"allocations are a partition of the contract's collateral coverage".

That premise is wrong. An allocation is a share of **one collateral record**, and
a contract may hold shares of many different assets — 50% of a property plus 80%
of a deposit is 130% and entirely correct. On this extract the rule fires on
**595 contracts, up to 2,083%**, every one of them legitimate.

The axis that would detect the failure it describes — the same asset pledged
more than once — is **per CollateralId**, and it is not checked. Tested
directly: the maximum total share of any collateral record is **1.0001**, and
**zero** records are over-allocated. The book is clean on the axis that matters
and noisy on the one that is watched.

**Handling:** R runs the rule and emits ~595 warnings. The Python port carries
the same rule set. The Excel tool has no equivalent. In all three the real check
is absent.

The scale conversion, incidentally, is correct: the raw column is in percent
(0–100, median 0.44) and the ETL divides by 100 before the netting formula
(run output: 0–1, median 0.0049). That was worth confirming, because feeding
percent into `CollateralValue × Allocation × (1 − Haircut)` would inflate
collateral a hundredfold.

---

## I6 — Ten AccountMaster columns arrive identically zero · **B**

Zero on all 7,259 rows:

| column | consequence |
| --- | --- |
| `STAGE`, `PD12M`, `PDLIFETIMEVALUE`, `LGDRATE`, `EAD`, `CCF`, `IMPAIRMENTAMOUNT` | **by design** — the ETL sends these blank so the engine computes them |
| `NOMINALINTERESTRATE` | R's annuity shape is `r = NIR/12`; at zero it degenerates to linear, so that shape can never produce curvature on this input |
| `OFFBALANCE` | no undrawn exposure reaches the model at all, including for the Off BS book |
| `LOANTOVALUE` | no independent check on collateral coverage |
| the twelve component columns (`PRINCIPAL`, `PRINCIPALOVERDUE`, `INTERESTACCRUED`, `INTERESTOVERDUE`, `FEE`, `PENALTY`, `COMMISSION`, `OTHER` and their overdue pairs) | they sum to **0.00** against an `ONBALANCE` total of **7,767,546,788** |

The last one has a visible consequence: `Cla Amount Principal`,
`Cla Amount Interest Accrued` and the other ten breakdown columns are **zero in
every row of both delivered reports**. That is not a code defect — the
input carries no breakdown to allocate. It does mean the provision cannot be
split by component for disclosure.

**Handling:** all three implementations pass the zeros straight through. R's
annuity branch is present and unreachable; Python does not implement it at all
(see **X2**). No validator flags an all-zero column.

---

## I7 — Zero and negative outstanding balances · **C**

`ONBALANCE` is **exactly zero on 214 contracts** and **negative on one**
(−0.0020).

**Handling:** `compute_lgd` assigns `zero_exposure_lgd = 1.0`, so a
zero-balance contract is booked at 100% LGD — harmless while the exposure it
multiplies is zero, but it distorts every LGD distribution and average the
analytics produce. The single negative balance is immaterial in value and
indicates a rounding artefact upstream. `INPUT_AccountMaster_onbalance_nonneg`
exists as a **WARN** and does catch the negative.

---

## X1 — Excel and R/Python disagree on the months before the first payment · **B**

The forward lookup is the same in all three. `first_schedule_month` is not.

* **Excel** takes `MIN(MonthLifetime)` over **every** row. With wrapped dates
  present that minimum is a large negative number, so the clause
  "if m < first_schedule_month use the current exposure" can never fire for any
  m ≥ 0. Months 1 onward therefore take the **first surviving payment's**
  `BALANCE + REPAYMENT`.
* **R and Python** drop `month_lifetime < 0` first, so `MIN` is the first
  surviving payment's month. Months between 1 and that month take
  **`OnBalance`** instead.

The R source is explicit that it knows Excel behaves differently — the `m == 0L`
special case exists because "first_schedule_month < 0 … would bypass the second
clause".

Measured over the whole extract:

| | |
| --- | --- |
| contracts whose curve **values** differ | **187** |
| months differing | 206 |
| their on-balance exposure | **584,655,155** |
| curve length differing | **0** contracts |
| net curve sum, Excel vs R | **−21,162,763** (−0.11%) |
| largest single contract (`599073`) | **+1,748,551** |

Only 187 of the 1,124 affected contracts differ, because most have their first
surviving payment at month 0 or 1, leaving no gap. The net direction is mixed,
not systematic. Curve length never differs, so `MAX` is unaffected.

This matters less for its size than for what it is: **the Excel tool and the two
ports do not agree, and the disagreement exists only because of I1.** Repair the
dates and it vanishes.

---

## X2 — R has an annuity shape; Python does not · **C**

`R/ecl_ead_curve.R:200-215` implements three shapes:

```r
coef <- switch(shape,
  linear  = pmax(0, 1 - prog),
  annuity = { r <- if (is.na(nir)) 0 else nir / 12
              if (r <= 1e-9) pmax(0, 1 - prog)      # degenerates to linear
              else { rp <- r * f
                     num <- (1 + rp)^n_pay - (1 + rp)^paid
                     den <- (1 + rp)^n_pay - 1
                     ifelse(paid >= n_pay, 0, pmax(0, num / den)) } },
  rep(1, H))
```

`src/ifrs9qdb/engine.py` implements two:

```python
if shape == "bullet":
    return np.full(H, float(on_balance), dtype=float)
# ... everything else is the linear ramp
```

Today this is invisible, twice over: no shipped rule selects `annuity`, and
`NOMINALINTERESTRATE` is zero everywhere (**I6**) so R's annuity would degenerate
to linear anyway. But the shapes are set in `config/model.yml`, which Risk can
edit without touching code. **The moment someone writes `shape: annuity` and the
extract carries a real interest rate, R produces a convex curve and Python
silently produces a straight line.**

---

## X3 — The `ead_fallback` config block is live in R and inert in Python · **B**

This is the one that makes X4 unfixable from the config file, and it is worse
than a divergence in behaviour: it is a divergence in what is *configurable*.

**R reads the block.** `ead_fallback_rules(cfg)` at `R/ecl_ead_curve.R:103`
pulls `cfg$ecl$ead_fallback`, honours both `default` and `rules`, and falls back
to the built-in defaults only when the node is absent.

**Python never reads it.** The package ships
`src/ifrs9qdb/config/model.yml` with the same `ead_fallback` block, and the only
occurrence of the string `ead_fallback` anywhere under `src/` **is that YAML
file**. The shapes come from the hardcoded `EAD_FALLBACK_RULES` tuple in
`engine.py`; `fallback_ead_curve` does not accept a `rules` argument at all, and
its single production caller (`engine.py:161`) passes none. The `rules`
parameter on `resolve_ead_shape` exists only so tests can inject a set.

So if Risk edits `ecl.ead_fallback` — adds a portfolio, changes a shape, moves
the default — **the R numbers move and the Python numbers do not**, with no
error, no warning and no sign in either output that the two are now running
different rules.

One thing does agree: `EAD_FALLBACK_DEFAULT = "bullet"` matches R's
`default: bullet`, and R's `switch` falls through to `rep(1, H)` — also bullet —
for an unrecognised shape name. So a typo like `shape: bulet` gives a flat curve
in R, and in Python gives whatever the hardcoded tuple says, because the typo
never reaches it. The defaults coincide; the mechanism does not.

---

## X4 — Python carries a fallback rule the R config does not · **B**

`EAD_FALLBACK_RULES` in `src/ifrs9qdb/engine.py` has **six** entries. The R
package's defaults and `config/model.yml` — including the copy frozen into both
delivered runs — have **five**. The extra one:

```python
(None, "4", "linear"),   # no portfolio named: payment type 4 amortises
```

R has no portfolio-less rule, so an unmatched type 4 falls to `default: bullet`.

**Is it live?** Not on the tested quarters. Resolving both rule sets against
every fallback contract in both delivered runs gives **zero** disagreements:
every (portfolio, payment type) pair there is matched by one of the five
portfolio rules, and `Off BS` and `Tasdeer` carry only payment type 3. That is
why the port reproduces the R report to the cent.

**Is it armed?** Yes. On the 2026-06-09 book, `Off BS` and `Tasdeer` **do** carry
type-4 facilities with no schedule:

| portfolio | type | R | Python | contracts | on-balance |
| --- | --- | --- | --- | --- | --- |
| Off BS | 4 | bullet | linear | 781 | 494,700,900 |
| Tasdeer | 4 | bullet | linear | 94 | 13,286,320 |
| | | | | **875** | **507,987,231** |

That is **34.6%** of the fallback population, and linear sums to about half of
bullet over the same horizon (0.54 at 12 months, 0.51 at 60). When this extract
is run, the two engines will disagree by roughly a factor of two on half a
billion of exposure.

The rule was added because it was needed to match EY's own workbook on their
reconciliation contract 11, where a flat curve gives about 1.8× the LIC figure.
That reasoning may well be right — but then **the R config is wrong and should
carry the same rule**, rather than the two diverging by accident on the next
quarter's book.

---

## What to fix, in order

1. **Repair the dates at the source** (I1). One rule — `year < 1950` in a 2026
   extract means the pivot wrapped — or, better, have the extract deliver a
   four-digit year.
2. **Validate the schedule** (I2, I3). `START_DAT >= POST_DATE`, and
   `MAX(START_DAT) per contract == MATURITYDAT`. Either would have caught I1 on
   the day it appeared.
3. **Reconcile the two rule sets** (X4) and make the Python side read the
   config (X3). Decide whether the portfolio-less type-4 rule is right, then put
   the same rules in both — which today means editing Python source, not a
   config file.
4. **Decide what a blank DPD means** (I4) for the 123 contracts whose customer
   has no DPD anywhere, and say so explicitly rather than defaulting to zero.
5. **Fix the allocation check's axis** (I5) so it watches per collateral, and
   stop 595 false warnings from hiding real ones.
6. **Ask the source for the component breakdown and off-balance** (I6), or
   record that the provision cannot be disclosed by component.

Items 1 to 3 change reported numbers. Items 4 to 6 change what can be seen.
