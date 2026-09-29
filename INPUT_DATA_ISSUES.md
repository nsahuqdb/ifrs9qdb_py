# Input-side issues, and how Excel, R and Python each handle them

A companion to `METHODOLOGY_ISSUES.md`. That register covers the MODEL. This one
covers the **extract** — what arrives from Oracle before any calculation — and,
for each item, what the three implementations do with it: the Excel tool, the R
package, and the Python port.

**How each input item is priced has not been changed.** What changed is that
each is now flagged before a run starts — the pre-run check and the
pricing-readiness check name every contract it touches, and why; see
`PRICING_READINESS.md`. The engine differences X2–X4 are fixed.

The distinction matters because the owners differ. A methodology item is a
decision for Risk. An input item is a fix in the source extract or in
validation, and until it lands every engine inherits it — all three faithfully
implement the same rule on the same bad data.

Evidence: the raw extract of **2026-06-09** (`RepaymentSchedule.xlsx`, 85,160
rows over 5,705 contracts; `AccountMaster.xlsx`, 7,259 contracts), the delivered
runs for 2025-09-30 and 2025-12-31, and those runs' own
`reports/validation.csv`.

| | Issue | Severity |
| --- | --- | --- |
| I1 | Schedule dates lose their century: 2030–2043 arrive as 1930–1943 | **A** |
| I2 | 97.6% of allocated collateral has no value, so LGD is unsecured | **A** |
| I3 | `Origination` arrives empty except the contract id | **A** |
| I4 | Three of five customer staging flags never arrive | **A** |
| I5 | `PastDueDays` is blank on contracts holding 450m, and blank reads as current | **B** |
| I6 | Ten AccountMaster columns arrive identically zero | **B** |
| I7 | 1,689 fully duplicated schedule rows, and 117 allocations with no contract id | **B** |
| I8 | Two collateral types are unmapped; the ports disagreed, now aligned | **B** |
| I9 | Zero and negative outstanding balances | **C** |
| X1 | Excel and R/Python disagree on the months before the first payment | **B** |
| X2 | The `ead_fallback` config block was live in R and inert in Python — **fixed** | **B** |
| X3 | Python carried a fallback rule the R config does not — **fixed** | **B** |
| X4 | R had an annuity shape; Python did not — **fixed** | **C** |

---

## 1. What I1 costs, in percent — reconciled

This section has been revised three times, and the figures moved each time. The
table below is the final position; the one after it says what was wrong with each
earlier figure, so the history is on the record rather than quietly replaced.

### The answer

| book | how it was measured | contracts truncated | **Stage 2 ECL** | **total ECL** |
| --- | --- | --- | --- | --- |
| **June 2026** (latest) | **the R package itself**, run on the raw extract twice — as delivered, and with only `START_DAT` repaired in the input file | 1,106 curves | **+6.81%** (+38.50m) | **+2.01%** |
| **December 2025** (delivered run) | every truncated curve extended with the **real repaired schedule** of the same loan from the June extract, recomputed through the engine | 861 | **+3.70%** (+22.02m) | **+0.96%** |

**June, in full** — R and the Python port now produce identical reports from the
raw extract (7,334 of 7,334 stages agree, largest per-contract difference 0.0002):

| | as delivered | dates repaired | change |
| --- | --- | --- | --- |
| total ECL | 1,916,701,233 | 1,955,197,029 | **+38,495,796 = +2.01%** |
| Stage 1 | 376,891,584 | 376,891,734 | +150 (+0.00%) |
| **Stage 2** | **565,397,063** | **603,892,709** | **+38,495,646 = +6.81%** |
| Stage 3 | 974,412,586 | 974,412,586 | 0 |

Stage 3 cannot move — it is booked at 100% of outstanding (**M12**). Stage 1 is
capped at twelve months. The whole effect is Stage 2.

**December, in full:** 2,283,041,269 → 2,305,060,413 (+0.96%); Stage 2
595,880,001 → 617,899,145 (+3.70%). 468 of the truncated curves stop exactly at
December 2029 and **393 stop earlier** — annual and quarterly payers whose last
surviving payment falls before the cliff, like the eighteen-year annual loan in §4.
The December figure is a close lower bound: 1,606 December contracts had matured
or closed by June, so their repaired schedules are not available (951 of them
Stage 2, 49.0m of Stage 2 ECL). Contracts that closed within six months are
short-dated and unlikely to have had post-2029 payments.

**Why June is larger than December.** In June **67%** of contracts are Stage 2,
against 54% in December, and the loans the date wrap truncates carry **79%** of
Stage 2 ECL, against 62%. June's Stage 2 book is both larger and far more
concentrated in exactly the long-dated lending this defect cuts short.

### What was wrong with each earlier figure

| figure given earlier | book | what was wrong |
| --- | --- | --- |
| Stage 2 **+2.30% to +2.99%**, total +0.60% to +0.78% | December | Found truncated contracts by looking for curves that stop **exactly at December 2029** — 506 of them. That misses every contract whose last surviving payment is earlier in 2029: **393 more**. It also ran the missing tail down in a straight line, which captured 87% of the true exposure. |
| Stage 2 **+7.03%**, total +2.14% | June | Ran through the **Python** ETL, which did not yet reproduce R from raw inputs. It priced 949 facilities at zero and dropped every maturity date (§5), which shrank the Stage 2 base the percentage is taken of. R's own figure is +6.81%. |
| total **+0.18%** | June | The Python ETL left EIR in percent, discounting every long-dated month to nothing (§5). |

The direction was never in doubt — every month the curve does not reach adds a
non-negative marginal loss — but two of the three earlier sizes were understated
and one was measured through code that did not yet match R. The figures above
were produced by R itself, and the Python port now matches R exactly.

## 2. Why none of this was flagged in validation

The runs do validate. `reports/validation.csv` in each one records **114
checks**, and in both runs **105 passed, 1 ERROR, 7 WARN, 1 INFO, 0 suppressed**.
`validation_suppressions.yml` is empty in both — nothing was silenced.

**Not one failing check in either run relates to the date corruption.** It is not
suppressed, not downgraded, not accepted. It is invisible.

Here is why, and it is structural rather than an oversight in one rule.

### The suite checks structure, not content

Categorising all 114 checks by what they actually assert:

| what it verifies | checks |
| --- | --- |
| range / sign | 22 |
| internal consistency (sums, counts, contiguity, monotonicity) | 21 |
| uniqueness | 17 |
| file present / schema | 16 |
| referential integrity and mapping coverage | 14 |
| other | 19 |
| logical rules (`dpd > 90 implies stage 3`) | 3 |
| **dates parse or are ordered** | **2** |
| **a column actually carries data** | **4** |

The two date checks are both on `AccountMaster.OPENDATE` — a parse check and
`maturity >= open`. **Neither touches `RepaymentSchedule`.** Searching every
check's id, description, rationale and remediation: `START_DAT` appears in none,
`POST_DATE` in none, "century" in none, "horizon" in none, "truncat" in none.

Of the four checks that verify a column carries data, two are the contract-id
column, one is a derived back-fill, and one asserts a column is *deliberately*
all-NA. **No input data column is checked for being populated at all.** A file
can arrive with only its key column filled and pass every check — which is
exactly what `Origination` does (**I3**).

### The one schedule rule checks the wrong thing

`INPUT_RS_coverage` is the only rule that looks at `RepaymentSchedule`. It checks
that every AccountMaster contract **has rows** and says nothing about their
contents. It fired in both runs — 2,390 and 1,463 contracts with no schedule —
which is a coverage warning, not a content one.

### The derived layer checks every property except length

Eight `DERIVED_LTPO_*` checks guard the EAD curve:

| check | severity | on these runs |
| --- | --- | --- |
| `schema` — seven columns | ERROR | passed |
| `extract_date_unique` | ERROR | passed |
| `ead_nonneg` — non-NA and ≥ 0 | ERROR | passed |
| `month_starts_at_zero` — every contract has a month-0 row | ERROR | passed |
| `months_contiguous` — months form a contiguous 0..N−1 | ERROR | passed |
| `contracts_subset_of_trans` | ERROR | passed |
| `total_month0_ead_reconciles` — month-0 sum equals on-balance | WARN | passed |
| `ead_nonincreasing` | **INFO** | **failed** |

**A truncated curve satisfies seven of the eight.** It is perfectly
well-formed — non-negative, starting at zero, contiguous, reconciling at month 0.
It is simply too short, and **nothing compares N to the contract's maturity.**

### The one check that did notice was pre-explained away

`DERIVED_LTPO_ead_nonincreasing` failed in both runs — **1,849** contracts in
run_00001 and **2,297** in run_00002 with a rising EAD. It is one of only **two
INFO-severity checks in the whole suite of 114**, so it never gates, and its
message supplies the reason before anyone asks:

> *expected for revolving/off-balance products and contracts with interest/fee
> accrual baked into the schedule*

That explanation is **half right**, which is the worst kind. About 45% of the
rises are genuine grace-period accrual (**M17**). The rest are the projected
interest being added to the balance (**M16**). The check found a real symptom of
a real defect and its own message classified it as normal.

### Two of the three implementations have no validation at all

The Excel tool runs no checks. The Python port reproduces the same rule set as
R, so it reaches the same verdict. **All three accept the extract without
comment.**

### The rules that would have caught I1, cheapest first

1. **`START_DAT >= POST_DATE`.** All 25,466 wrapped rows are dated before the
   schedule that contains them, and **no clean row is**. One comparison, zero
   false positives on this extract.
2. **`YEAR(START_DAT) >= YEAR(extract_date)`.** The same set.
3. **`MAX(START_DAT) per contract == AccountMaster.MaturityDate` (±1 month).**
   This one is free, and it is the strongest — see **I1**, where it matches
   1,124 of 1,124 contracts once the century is restored and 0 of 1,124 as
   delivered.
4. **`LAST(MonthLifetime) per contract >= months to maturity`** in the derived
   layer, at ERROR. This is the check whose absence let the truncation through
   even after the curve was built.
5. **`a column is not entirely blank`**, applied to every input column. One rule
   would have caught I3, I4 and half of I6.

### One caveat on gating

Both delivered runs carry `status: unofficial` ("no approval required"), so the
one ERROR-severity failure in each did not block them. Whether an ERROR halts an
**official** close is a separate question this evidence does not answer.

---

## 3. Why Excel and our code differ on months 1–6

The forward lookup is identical in all three implementations. What differs is a
single quantity: **`first_schedule_month`**.

The Excel rule, as the R source documents it from
`RepaymentScheduleTransform!T4`:

```
IF m = 0                      -> current exposure (AccountMaster.OnBalance)
IF m < first_schedule_month   -> current exposure
ELSE                          -> EAD at the first scheduled payment with month > m
```

`first_schedule_month` is `MIN(MonthLifetime)` over the contract's rows. The two
implementations compute that minimum over **different row sets**:

| | rows used for `MIN` | `first_schedule_month` on contract 599073 |
| --- | --- | --- |
| **Excel** | every row, including the wrapped ones | `MIN(-1157, …, 31)` = **−1,157** |
| **R and Python** | only rows with `month >= 0`, negatives dropped first | `MIN(7, 19, 31)` = **7** |

Now run the middle clause for month 1:

* **Excel**: is `1 < -1157`? **No.** So it falls to the third clause and looks up
  the first payment after month 1 — the 2027 payment at month 7 — giving
  `BALANCE + REPAYMENT` = **121,927,731**.
* **R and Python**: is `1 < 7`? **Yes.** So it uses the current exposure,
  `OnBalance` = **121,636,306**.

The same applies to months 2 through 6. From month 7 the two agree again,
because both are then in the third clause looking at the same surviving
payments.

**So the difference is not a different formula — it is the same formula fed a
different minimum, because R and Python drop the corrupted rows one step
earlier than Excel does.** R's own source comment shows the author knew:

> *The R=0 special case matters when a contract has placeholder schedule entries
> with bogus historical dates (negative month_lifetime), which would make
> first_schedule_month < 0 and bypass the second clause.*

The `m == 0` clause was added precisely so month 0 still gets `OnBalance` when
`first_schedule_month` goes negative. It fixes month 0 and leaves months 1
onward to diverge.

### How many contracts, and how many months?

**Not all of them, and for most of those it is one month.** The number of
differing months is exactly `first_surviving_payment_month − 1`, floored at zero,
so it depends entirely on how soon the contract's next real payment falls:

| first surviving payment at month | contracts | months that differ |
| --- | --- | --- |
| 0 | 568 | **0** |
| 1 | 321 | **0** |
| 2 | 206 | 1 |
| 3 | 9 | 2 |
| 7 | 2 | **6** |

| | |
| --- | --- |
| affected contracts with a surviving payment | 1,106 |
| **where no month differs** (payment at month 0 or 1) | **889 — 80.4%** |
| **where months do differ** | **217 — 19.6%** |
| differing months, total across the book | **236** |
| per differing contract | median **1**, maximum **6** |
| contracts whose curve **values** actually differ | **187** |
| their on-balance exposure | **584,655,155** |
| contracts whose curve **length** differs | **0** |
| net curve sum, Excel vs R/Python | **−21,162,763** (−0.11%), mixed sign |
| largest single contract (`599073`) | **+1,748,551** |

So the "months 1–6" of the worked example is the **extreme case** — only two
contracts in the book have a six-month gap. Four in five affected contracts have
no gap at all, because they still have a payment due this month or next.

The 187 is smaller than the 217 because on thirty of them `OnBalance` happens to
equal the first surviving payment's `BALANCE + REPAYMENT`, so the two clauses
return the same number. 217 is how many are structurally exposed; 187 is how many
actually print differently.

Curve **length** never differs, since `MAX` is unaffected by dropping negatives
as long as one non-negative row survives.

**This divergence exists only because of I1.** Repair the dates and
`first_schedule_month` is positive in all three, the second clause behaves
identically, and the difference vanishes. It is a symptom, not a separate
defect — which is why fixing the dates is the right response and reconciling the
two lookups is not.

---

## 4. The worked example, end to end

An eighteen-year annual-payment facility — the case that exposes everything at
once. Contract `599073`, from the 2026-06-09 extract:

* `ONBALANCE` **121,636,306**
* `MATURITYDAT` **2043-01-30** → **199 months** remaining
* `PAYMENTFREQUENCY` 12 (annual), 17 scheduled payments

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

Fourteen of seventeen payments carry a 1930s year. Restore the century and the
schedule is perfect: 2027, 2028, … 2043, ending **2043-01-30 at a balance of
0.00** — the date `AccountMaster` already gives as the maturity.

| | Excel tool | R package | Python port |
| --- | --- | --- | --- |
| end month | `MAX` = **31** | same | same |
| curve emitted | months 0–30 | months 0–30 | months 0–30 |
| months priced | **31 of 199** | **31 of 199** | **31 of 199** |
| balance where the curve stops | **121,580,510** | same | same |
| months 1–6 | 121,927,731 | 121,636,306 | 121,636,306 |

**100.0% of the exposure is still outstanding when the curve ends.** The facility
is back-loaded — instalments run from 2.59m up to 17.87m — so the three surviving
years retire 2.6m of 121.6m and **every payment that actually repays the loan is
one of the fourteen dropped.**

This is worse than "100m becomes 90m and then the curve stops". Here the curve
stops with the balance essentially untouched, because the truncation lands
exactly where the amortisation had not yet begun. A long project loan is the
profile where this is worst, and the profile most exposed to it.

---

## 5. R and Python consistency, and the Excel tool

### How this was checked

R was installed in this environment (4.3.3) and the R package run on its own
tests: **239 pass**, including all four EY reconciliation contracts and all
twelve run-324 goldens, which the R package documents as matching LIC exactly.
One R test fails — `test-analytics.R:219`, on the staging-threshold analytic — and
it is off the ECL path.

Then **both ETLs were run on the same raw extract** (2026-06-09), with the same
config and static data (the Python package's copies are byte-identical to the R
package's `inst/`), and every output file compared cell by cell.

That comparison had never been made. The engine had been proven against R, but
only from R's *output* files — so the Python ETL, raw extract to those files, was
never tested against anything. It disagreed on seven things.

### What differed, and what it cost

| # | divergence | Python did | R does | effect on the June book |
| --- | --- | --- | --- | --- |
| 1 | EIR scale | left the raw percent (3.0) | divides by 100, two-tier fallback for zero | discounting at 300% a year; ECL **−24%** |
| 2 | maturity column | looked for `MATURITYDATE` only | reads the truncated `MATURITYDAT` | every maturity blank; 1,567 fallback contracts floored; **−22.4m** |
| 3 | maturity extension | not implemented | matured → reporting date + 365 days | part of the −22.4m |
| 4 | rating chain | kept the literal "Unrated" | V4 Transformation!Z: sector rule by DPD, else segment fallback | **949 facilities priced at zero; −129.3m** |
| 5 | worst-grade ladder | hard-coded; missing QDB 1+, 6+, 6-, listing non-existent QDB 10–12 | master-scale hierarchy | none on this extract; latent |
| 6 | `Ratings.csv` | one external grade per bucket (42 rows) | all 62 grades | 2 investments rated BBB+ found no PD bucket |
| 7 | investment ids | 1..n surrogates | the extract's account id and counterparty name | traceability only |

Plus smaller ones, all fixed: `Origination` ids not substituted for 1,006
off-balance contracts; a SQL*Plus header row written into the allocation file as
data; dates zero-padded where R does not pad; a zero payment frequency written as
`0` rather than blank; StPD probabilities in scientific notation (`6.27e-05`)
where R writes sixteen fixed decimals — a CSV reader that rejects exponents would
misread the whole term structure.

Every one is now fixed to match R, and **both ETLs produce identical output**:
all 18 files agree cell for cell, and the final reports agree on all 7,334
contracts to within 0.0002.

**Several of these had comments asserting the wrong behaviour was required** —
"LIC wants the bucket once", "LIC will not accept a name as a key", "matching
the zero-padded reference is the requirement". Both delivered R runs, which are
what LIC actually received, contradict all three.

`tests/test_r_parity.py` now runs the R package itself whenever R is installed and
`IFRS9_SRC_INPUTS` is set, and fails if any output file or any contract's ECL
diverges. That is the test whose absence let seven defects through.

### What remains different, deliberately or cosmetically

* **Row order** in three files: R writes each EAD curve by ascending month, the
  port by descending, and StPD and the staging-flag file are grouped differently.
  LIC keys all three; the parity test compares by key.
* **Numeric formatting** otherwise: R writes fixed decimals in places, the port
  writes the value. Compared numerically, identical.
* **Balances at the fourth decimal.** R formats a column to seven significant
  figures of its smallest value, so 7852.9968 is written 7852.997. The port now
  does the same, so the two match — but it is worth knowing that R's writer
  rounds balances, by under 0.0005.

### What R does that the register should know

* **Origination PD and rating are always written blank**, whatever the extract
  holds (`.build_origination_rows`). So the relative SICR test is unavailable by
  construction, not only because `Origination` arrives empty (**I3**) — filling
  the extract would change nothing until the writer changes.
* **The maturity extension is strictly `<`.** `staging_thresholds.csv` describes
  it as "If Maturity Date **<=** Reporting Date". The code and the V4 formula it
  cites use `<`, so a facility maturing on the reporting date is not extended. The
  description is wrong, or the code is; they should not disagree.

### The Excel tool

**It could not be compared directly — the workbook is not in anything available
here.** The R package names it
(`Updated_ETL_File__Test__V4_20211029_-_New_server_link.xlsm`) and documents each
formula it replicates, cell by cell. What is verified:

* the R package reproduces the four EY contracts and twelve LIC run-324 contracts
  exactly, per its own golden tests, which pass here;
* the Python port reproduces the R package exactly, from raw extract to report;
* on the wrapped dates specifically, R and Excel **differ on months 1 to *n*
  before the first surviving payment** (§3) — 187 contracts on this extract — and
  on nothing else in the EAD derivation.

What is not verified: anything the workbook does that the R package does not
document. Closing that needs the workbook and the same extract run through it.

---

## I1 — Schedule dates lose their century · **A**

**25,466 of 85,160 rows (29.9%), across 1,124 contracts**, carry `START_DAT`
between 1930-01-01 and 1943-01-30. The non-wrapped rows span 2026–2029 and
nothing else. That is exactly the two-digit-year pivot: `30`–`99` → 1930–1999,
`00`–`29` → 2000–2029. They are payments due **2030 to 2043**.

Four checks, all passed by adding a century and failed without it:

| check | repaired | as delivered |
| --- | --- | --- |
| balance path monotone non-increasing | holds | fails |
| roll-forward identity on every row | holds | fails |
| last payment equals `AccountMaster.MATURITYDAT` | **1,124 / 1,124** | **0 / 1,124** |
| final balance at that payment is 0.00 | holds | 2.4bn outstanding |

Median error in the implied end month as delivered: **39 months**.

The R source calls these "placeholder historical schedule entries … a sentinel
date for the final balloon payment". They are not sentinels: the median affected
contract has **19** of them (max 126), and only **55 of 1,124** contracts have a
single wrapped row.

**Handling:**

* **Excel tool** — keeps the rows for `MIN`/`MAX`, so `MAX(MonthLifetime)` stops
  at the last pre-2030 payment and the curve ends there. `MIN` goes deeply
  negative, which is what produces the months 1–6 divergence in §3.
* **R package** — `R/lifetime_parameter_other.R:94` drops `month_lifetime < 0`
  before `MIN`/`MAX`, so the same truncation happens one step earlier.
* **Python port** — `src/ifrs9qdb/etl/lifetime.py` filters `month >= 0`
  identically. Byte-for-byte the same curve as R.

All three then take the ECL horizon from the curve's own length
(`R/ecl_ead_curve.R:260`, `resolve_ead_curve` in both ports) and never consult
`MaturityDate`, so the shortened curve becomes the contract's lifetime. **The
extract already carries the correct maturity** — `MATURITYDAT` is clean (range
2013-06-15 to 2043-01-30, no blanks, nothing below 1950, nothing before
`OPENDATE`) and its maximum is exactly `599073`'s true final payment. Nothing
compares the two, and the engine reads `MaturityDate` only in the branch where a
schedule is *absent* — the one case where the comparison is impossible.

**18 contracts have every row wrapped.** They get no curve at all and fall to the
parametric fallback (on-balance 3,119,064), where the shape is guessed from the
payment type — see **M5**.

---

## I2 — 97.6% of allocated collateral has no value · **A**

The netting formula is
`CollateralValue × AllocationPercentage × (1 − HaircutGeneral)`. The first
factor is almost always zero.

| | run_00001 | run_00002 |
| --- | --- | --- |
| allocated collateral records | 3,707 | 3,841 |
| **with zero or missing `CollateralValue`** | **3,618 (97.6%)** | **3,727 (97.0%)** |
| records actually carrying a value | **89** | **114** |
| total allocated collateral value | 1,514,381,667 | 1,843,596,345 |
| resulting `collateral_net` | 303,941,072 | 537,263,041 |

In the raw 2026-06-09 extract the same pattern: `COLLATERALVALUE` is zero on
**3,948 of 4,042** records. **The entire secured position of the book rests on
89 collateral records.** Every other secured contract is priced as if unsecured,
so its LGD sits at the 0.45 base instead of being reduced toward the 0.225 floor.

This one **is** flagged — `XFILE_collateral_value_valid`, WARN, in both runs,
with the correct rationale ("LGD silently rises to the unsecured 45%") and the
correct remediation ("send the listed collateral ids to the collateral unit").
It has been carried as a warning across at least two quarters.

The direction matters for reading the whole register: this is the one large item
that makes the provision **too high**, and it is far larger in exposure terms
than I1, which makes it too low. They do not cancel — they land on different
contracts — but neither can be judged without the other.

**Handling:** identical in all three. A zero value contributes zero; no
implementation substitutes an appraisal, a market value or a haircut default.
There is no collateral valuation date in the extract, so appraisal staleness
cannot be assessed at all.

---

## I3 — `Origination` arrives empty except the contract id · **A**

`Origination.xls` has 7,259 rows and 14 columns. **Twelve of the fourteen are
entirely empty** — every value blank on every row. Only `EXTRACTDA` and
`CONTRACTID` carry data.

The schema reads it positionally:

| position | field | populated |
| --- | --- | --- |
| 1 | `contract_id` | 7,259 |
| 2 | `origination_pd_12m` | **0** |
| 3 | `origination_rating` | **0** |
| 4 | `origination_dpd` | **0** |
| 5 | `origination_watchlist` | **0** |

Every one of those four is declared `required = FALSE`, so the loader accepts the
file without complaint.

**This is the root cause of M9.** That item records that no quantitative SICR
test exists; the reason is that the data a relative test needs — the 12-month PD
and the rating *at origination*, to compare against today's — **never arrives**.
IFRS 9's significant-increase test is inherently a comparison with origination,
and the origination side of it is blank.

Two checks touch this file — `INPUT_Origination_contractid_unique` and
`XFILE_AM_contract_in_origination`. Both verify the **join**. Both pass. Neither
notices the file carries no data.

**Handling:** all three implementations fall back to the absolute staging
triggers (DPD, watchlist, restructuring), and none warns that the relative test
is unavailable.

**Filling the extract would not fix it.** R writes `OriginationPD12M` and
`OriginationRating` **blank on every row regardless of the source**
(`.build_origination_rows`, one blank row per account), and both delivered runs
carry exactly that. The Python port used to pass the source values through; it
now matches R (§5). So the relative SICR test is unavailable by construction:
the source must send origination data **and** the writer must be changed to pass
it on.

---

## I4 — Three of five customer staging flags never arrive · **A**

`CustomerStagingFlag.xlsx` has 702 rows and 12 columns, of which **eight are
entirely empty**. Against the schema:

| position | field | populated | required |
| --- | --- | --- | --- |
| 3 | `is_default` | **0** | FALSE |
| 4 | `is_watchlist` | 702 (**112 set**) | TRUE |
| 5 | `is_insolvency` | **0** | FALSE |
| 6 | `is_default_in_gcc` | **0** | FALSE |
| 7 | `is_local1` (restructured) | 702 (**174 set**) | TRUE |

So `is_default`, `is_insolvency` and `is_default_in_gcc` — three Stage 3
triggers — **never arrive**, and because the schema marks them optional, nothing
objects.

Stage 3 therefore rests on **DPD > 90 alone** (plus manual override). And DPD is
itself blank on 131 contracts, 123 of which read as fully current (**I5**). A
customer who is legally insolvent but not yet 90 days down cannot be staged by
the data.

`TRANS_LENDPV_watchlist_implies_stage_2_or_3` and
`TRANS_LENDPV_restructured_implies_stage_2_or_3` both pass — they test the two
flags that *do* arrive. There is no equivalent check for the three that do not,
because a check on an absent column has nothing to assert.

---

## I5 — `PastDueDays` blank on contracts holding 450m · **B**

`PASTDUEDAYS` is missing on **131 of 7,259 contracts** (on-balance
**577,160,140**). Only the **customer's worst** DPD is written out; the
per-contract value never reaches the report:

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
**No divergence.** Splitting the 131:

| | contracts | on-balance | outcome |
| --- | --- | --- | --- |
| customer has a sibling facility with a DPD | 8 | 127,015,867 | inherits it — 3 of the 8 inherit a DPD above 30, one of 740 days. Correct. |
| **every** facility of the customer is blank | **123** | **450,144,273** | **both ports land on 0 — fully current** |

So the exposure genuinely at risk is **450m, not 577m**. For those 123 contracts
a blank is read as the most favourable value available: neither the 30-day
backstop nor the 90-day default trigger can fire. No validator checks for it.

---

## I6 — Ten AccountMaster columns arrive identically zero · **B**

Zero on all 7,259 rows:

| column | consequence |
| --- | --- |
| `STAGE`, `PD12M`, `PDLIFETIMEVALUE`, `LGDRATE`, `EAD`, `CCF`, `IMPAIRMENTAMOUNT` | **by design** — the ETL sends these blank for the engine to compute |
| `NOMINALINTERESTRATE` | R's annuity shape is `r = NIR/12`; at zero it degenerates to linear, so that shape can never produce curvature on this input |
| `OFFBALANCE` | no undrawn exposure reaches the model at all, including for the Off BS book |
| `LOANTOVALUE` | no independent check on collateral coverage — which matters given **I2** |
| the twelve component columns (`PRINCIPAL`, `INTERESTACCRUED`, … and their overdue pairs) | they sum to **0.00** against an `ONBALANCE` total of **7,767,546,788** |

The last has a visible consequence: `Cla Amount Principal` and its eleven
siblings are **zero in every row of both delivered reports**. Not a code
defect — the input carries no breakdown to allocate — but the provision cannot be
split by component for disclosure.

`AccountMasterInvestments` is worse: **25 of 40 columns** entirely empty.
`Origination` 12 of 14 (**I3**), `CustomerStagingFlag` 8 of 12 (**I4**),
`CustomerMasterInvestments` 7 of 10, `OriginationInvestments` 12 of 14.

No validator flags an all-zero or all-blank column anywhere.

---

## I7 — Duplicated schedule rows, and allocations with no contract · **B**

**1,689 of 85,160 schedule rows are fully identical** to another row — every
column, including the id, both dates and all four amounts — across **78
contracts**. 756 of them also carry a wrapped date.

**This corrects an earlier reading in this register.** The 933 cases where two
rows share a contract-month, which an earlier draft recorded as payments the
monthly grid loses, are **all 933 exact duplicates** — identical in
`PRINCE_DUE`, `PROJ_INT`, `REPAYMENT` and `BALANCE`. Not one is a genuinely
different payment. **Dropping the extra row is correct**, and the grid absorbing
them is accidentally right rather than wrong. Zero contract-months in this
extract hold two different payments.

That makes this an input hygiene item, not a model one: the extract should not
emit the same payment twice, and nothing checks that it hasn't.
`INPUT_ACA_pair_unique` does this for the allocation file; there is no
equivalent for the schedule.

Separately, **117 allocation rows have a blank `ContractId`**, carrying
**110,509,605** of gross collateral value. They can never be allocated to
anything, so that security is silently forfeited — conservative, and part of why
`INPUT_ACA_contract_fk` reports 704 orphans. The Python `as_id` helper already
documents this case; R has no id-normalisation helper in its join path.

---

## I8 — Two collateral types are unmapped; the ports disagreed, now aligned · **B**

`CONFIG_collateral_type_coverage` fires in both runs: collateral types **27 and
28** are absent from `collateral_types.csv`, so `HaircutGeneral` resolves to NA.
The two implementations then do **opposite** things.

**R** — `R/ecl_collateral.R:66-67`:
```r
contrib <- cval[a_clid] * a_pct * (1 - hc[ctyp[a_clid]])
contrib[!is.finite(contrib)] <- 0
```
`1 - NA` is NA, so the row is set to **zero**. An unmapped type contributes **no
collateral benefit** — conservative.

**Python** — `src/ifrs9qdb/inputs.py`:
```python
net_value[k] = float(v) * (1.0 - haircut.get(t, 0.0))
```
`haircut.get(t, 0.0)` defaults to **no haircut**, so `1 - 0 = 1` and the row
contributes its **full value** — aggressive.

These were exact opposites, and the comment beside the Python line claimed it was
"matching the engine", which was wrong.

**Which is right?** Neither, strictly — but Python's was indefensible. The
haircut table has 26 types: **23 carry `haircut_general = 1.00`** (worthless for
provisioning) and **exactly one carries 0.00** (bank letter of guarantee).
Defaulting an unknown type to `0.0` therefore treated it as the single most
generous category in the book. R's effective zero-benefit matches the 88%
majority.

**What the Excel tool does** settles the question differently: a VLOOKUP miss
returns `#N/A`, which propagates through the SUM, so the contract's collateral
becomes `#N/A` and somebody has to fix the mapping. Excel does not pick a side —
it refuses to compute. That is the right behaviour, and it is what
`CONFIG_collateral_type_coverage` firing should mean.

**Resolved.** Python now defaults an unmapped type to a **full haircut**, matching
R and the table's majority. Verified to change nothing today:
`collateral_net` is **303,941,071.60** and **537,263,040.63** on the two runs —
identical to R to the cent, because every type-27 and type-28 record carries zero
value (**I2**). The remaining gap is that both engines still compute silently
where Excel would refuse; the open item is to make an unmapped type an **ERROR**,
not to pick a better default.

---

## I9 — Zero and negative outstanding balances · **C**

`ONBALANCE` is exactly zero on **214 contracts** and negative on one (−0.0020).

**Handling:** `compute_lgd` assigns `zero_exposure_lgd = 1.0`, so a zero-balance
contract is booked at 100% LGD — harmless while the exposure it multiplies is
zero, but it distorts every LGD distribution the analytics produce.
`INPUT_AccountMaster_onbalance_nonneg` exists as a WARN and does catch the
negative.

---

## X1 — Excel and R/Python disagree on months 1 to 6 · **B**

Fully explained in §3 above. In summary: same formula, different
`first_schedule_month`, because Excel takes `MIN` over every row and the two
ports drop the wrapped rows first. 187 contracts, 206 months, 584,655,155 of
exposure, net −21,162,763. **It exists only because of I1 and disappears when the
dates are repaired.**

---

## X2 — The `ead_fallback` config block was live in R and inert in Python · **B** · FIXED

**Status: fixed.** `engine.ead_fallback_rules(model_cfg)` now reads
`ecl.ead_fallback` exactly as R's `ead_fallback_rules(cfg)` does
(`R/ecl_ead_curve.R:103`), honouring both `default` and `rules`. The report
builder prices with the run's model config, and the analytics and stress layers
reprice with the rules the run froze (`config_used/config/model.yml`), so
repricing a run uses the shapes its own report used. If Risk edits the block,
the R and Python numbers now move together. Pinned by `TestX2TheConfigBlockIsHonoured`.

What it was: the only occurrence of `ead_fallback` under `src/` used to be the
shipped YAML; the shapes came from a hard-coded tuple, so an edited config
moved R and left Python where it was, with nothing in either output to say
the two were running different rules.

---

## X3 — Python carried a fallback rule the R config does not · **B** · FIXED

**Status: fixed.** `EAD_FALLBACK_RULES` is now R's five portfolio rules, rule
for rule. The sixth, portfolio-free rule

```python
(None, "4", "linear"),   # no portfolio named: payment type 4 amortises
```

moved to `NO_PORTFOLIO_RULES` and applies only where no portfolio is known at
all — EY's worked examples, where it reproduces their workbook (a flat curve
gives about 1.8x the LIC figure on their contract 11). A priced book always
knows its portfolios, so it never fires there. Pinned by
`TestX3TheRuleSetsAgree`.

What it would have cost: on the June book the extra rule would have amortised
the type-4 facilities R prices as bullets —

| portfolio | type | R | Python before | contracts | on-balance |
| --- | --- | --- | --- | --- | --- |
| Off BS | 4 | bullet | linear | 781 | 494,700,900 |
| Tasdeer | 4 | bullet | linear | 94 | 13,286,320 |
| | | | | **875** | **507,987,231** |

— and a linear curve sums to about half a bullet over the same horizon. The
June as-is and repaired runs now reproduce R's FinalEclReport to the cent on
every contract.

---

## X4 — R had an annuity shape; Python did not · **C** · FIXED

**Status: fixed.** The report builder (`etl/report.py`) ports R's
`build_ead_fallback_curve` including the annuity branch, and the engine's
`fallback_ead_curve` (used by the analytics and stress layers) now does too: an
annuity retires level instalments at the nominal rate
(`NominalInterestRate / 12` per month, times the payment frequency), degenerates
to linear at a zero rate, and a shape R does not know is a bullet (R's `switch`
default) rather than the linear ramp it used to take. Pinned by
`TestX4TheAnnuityShape`. Still dormant: no shipped rule selects `annuity` and
`NOMINALINTERESTRATE` is zero everywhere (**I6**).

---

## What to fix, in order

1. **Repair the dates at source** (I1). A year below 1950 in a 2026 extract means
   the pivot wrapped; better, have the extract deliver four digits.
2. **Value the collateral** (I2). 97.6% of allocated records carry no value, and
   the whole secured position rests on 89 of them. This is the largest item by
   exposure and it pushes the provision the other way from I1.
3. **Add the five validators in §2**, particularly the two that compare the
   schedule to `MaturityDate` and the one that rejects an entirely blank column.
   Any of them would have caught I1, I3 or I4 on the day it appeared.
4. **Ask the source for `Origination` and the missing staging flags** (I3, I4),
   or record formally that the relative SICR test and three Stage 3 triggers are
   unavailable.
5. ~~Reconcile the two rule sets and make Python read the config~~ (X2, X3) — done.
6. **Decide what a blank DPD means** for the 123 contracts whose customer has
   none (I5), rather than defaulting to current.
7. **Make an unmapped collateral type an ERROR** (I8). The two ports are now
   aligned on a full haircut, but both still compute silently where Excel
   returns `#N/A` and forces the mapping to be fixed.
8. **Ask for the component breakdown and off-balance amounts** (I6), or record
   that the provision cannot be disclosed by component.

Items 1, 2, 5 and 7 change reported numbers. The rest change what can be seen.
