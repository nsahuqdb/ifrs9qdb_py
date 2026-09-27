# Input-side issues, and how Excel, R and Python each handle them

A companion to `METHODOLOGY_ISSUES.md`. That register covers the MODEL. This one
covers the **extract** — what arrives from Oracle before any calculation — and,
for each item, what the three implementations do with it: the Excel tool, the R
package, and the Python port.

**Nothing here has been changed in the code.**

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
| X2 | The `ead_fallback` config block is live in R and inert in Python | **B** |
| X3 | Python carries a fallback rule the R config does not — dormant, not fixed | **B** |
| X4 | R has an annuity shape; Python does not | **C** |

---

## 1. What I1 costs, in percent

**Measured directly, the way you asked: change `START_DAT` in the input file and
re-run everything.** Nothing else was touched — the same extract, the same
config, the same code, one column repaired (`+100 years` where the year is below
1950). Full ETL and engine on both, 2026-06-09 extract, 7,334 contracts,
**staging identical in both runs.**

| | as delivered | dates repaired | change |
| --- | --- | --- | --- |
| **total ECL** | 1,765,010,637 | 1,802,745,667 | **+37,735,030 = +2.14%** |
| Stage 1 | 256,822,194 | 257,009,880 | +187,686 (+0.07%) |
| **Stage 2** | **533,775,857** | **571,323,201** | **+37,547,344 = +7.03%** |
| Stage 3 | 974,412,586 | 974,412,586 | 0 (+0.00%) |
| the 598 contracts that moved | 438,323,391 | 476,058,421 | **+8.61%** |

**Stage 2 is understated by 7.0%, and the whole provision by 2.1%.**

Stage 3 cannot move: it is booked at 100% of outstanding regardless of the curve
(**M12**). Stage 1 barely moves, and only for 10 contracts whose curve was
shorter than twelve months before the repair. **Everything else lands in Stage
2** — which, on this book, is 4,919 of 7,334 contracts.

The repair lengthened **1,124 curves**, median gain **40 months**, maximum
**168**. `LifeTimeParameterOther` goes from 80,362 rows to 131,346 and its
longest curve from month 41 to month 198. The StPD term structures run to 600
months, so nothing else limits the horizon.

### This supersedes an earlier, lower estimate

An earlier version of this file reported **+0.60% to +0.78%** of total ECL and
**+2.30% to +2.99%** of Stage 2, measured on the two delivered quarters by
*reconstructing* the missing tail rather than repairing the source. Those figures
were too low, for two reasons:

1. **The reconstruction understated the exposure.** Running the residual balance
   down in a straight line captured **87%** of the true hidden EAD
   (94.9bn against 108.98bn summed over the hidden months), because these
   schedules are back-loaded — contract `590851` still carries 26.1m at month 60
   and 10.1m at month 90.
2. **More importantly, they were a different book.** The delivered quarters have
   **35.1%** of scheduled exposure in wrapped contracts; the June 2026 extract
   has **66.4%**, and far more of it sits in Stage 2. The two measurements are
   both right about their own quarter.

So: **+2.14% total and +7.03% of Stage 2 is the current, directly measured
figure**, and it is the one to use. The earlier bracket stands only as a
measurement of the December and September books, and even there is understated by
roughly a further 13% from the reconstruction.

### A defect found while doing this

The first attempt at this measurement returned **+0.18%**, which was wrong, and
the reason was a genuine bug in the Python ETL that this exercise exposed.

`AccountMaster.EIR` arrives as a **raw percent** — the values are 2.25, 2.50,
3.00, 3.50, 7.00. R divides by 100 (`R/transform_lending.R:336`,
`eir_raw / 100`). **The Python lending transform did not**, so 4,872 of 7,334
contracts carried a percent straight into the discount factor:

```
disc = (1 + eir) ** (t / 12)
```

At `eir = 3.0` that is `4 ** (t/12)`: a loss at month 42 divided by **128**
instead of 1.11, and at month 100 by **9,463**. It priced every long-dated month
at nothing — which is precisely the effect being measured, so the defect hid
itself.

It is now fixed, along with R's two-tier fallback for a zero or missing rate,
which the Python path also lacked. The port's EIR now matches R's scale exactly
(median 0.0439 against R's 0.0434, **maximum 0.0975 in both**). The effect on the
June book alone, before any date repair, is **1,425,123,668 → 1,765,010,637, or
+23.9%**.

This never touched the delivered runs: the parity test reads the R run's *output*
CSVs, where EIR is already a decimal, so `transform_lending` was never in that
path. The exact-reproduction test still passes.

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

**Handling:** all three implementations read the blanks as missing and fall back
to the absolute staging triggers (DPD, watchlist, restructuring). None warns that
the relative test is unavailable.

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

## X2 — The `ead_fallback` config block is live in R and inert in Python · **B**

**R reads it.** `ead_fallback_rules(cfg)` at `R/ecl_ead_curve.R:103` pulls
`cfg$ecl$ead_fallback` and honours both `default` and `rules`.

**Python never does.** The package ships `src/ifrs9qdb/config/model.yml` with the
same block, and the only occurrence of the string `ead_fallback` anywhere under
`src/` **is that YAML file**. The shapes come from the hardcoded
`EAD_FALLBACK_RULES` tuple in `engine.py`; `fallback_ead_curve` takes no `rules`
argument, and its single production caller (`engine.py:161`) passes none. The
`rules` parameter on `resolve_ead_shape` exists only so tests can inject a set.

So if Risk edits `ecl.ead_fallback`, **the R numbers move and the Python numbers
do not** — no error, no warning, nothing in either output to say the two are now
running different rules.

One thing agrees: `EAD_FALLBACK_DEFAULT = "bullet"` matches R's
`default: bullet`, and R's `switch` falls through to `rep(1, H)` — also bullet —
for an unrecognised shape name. The defaults coincide; the mechanism does not.

---

## X3 — Python carries a fallback rule the R config does not · **B**

`EAD_FALLBACK_RULES` has **six** entries. The R package's defaults and
`config/model.yml` — including the copy frozen into both delivered runs — have
**five**. The extra one:

```python
(None, "4", "linear"),   # no portfolio named: payment type 4 amortises
```

R has no portfolio-less rule, so an unmatched type 4 falls to `default: bullet`.

**Dormant on the tested quarters.** Resolving both rule sets against every
fallback contract in both runs gives **zero** disagreements: every
(portfolio, payment type) pair there is matched by one of the five portfolio
rules, and `Off BS` and `Tasdeer` carry only payment type 3. That is why the port
reproduces the R report to the cent.

**Armed on the June book**, where they do carry type-4 facilities without a
schedule:

| portfolio | type | R | Python | contracts | on-balance |
| --- | --- | --- | --- | --- | --- |
| Off BS | 4 | bullet | linear | 781 | 494,700,900 |
| Tasdeer | 4 | bullet | linear | 94 | 13,286,320 |
| | | | | **875** | **507,987,231** |

**34.6%** of the fallback population, and a linear curve sums to about half a
bullet over the same horizon (0.54 at 12 months, 0.51 at 60). When that extract
is run the two engines will differ by roughly a factor of two on half a billion.

The rule was added to match EY's own workbook on their reconciliation contract
11, where a flat curve gives about 1.8× the LIC figure. That may be right — in
which case **the R config is the thing to change**, rather than letting the two
diverge by accident on the next quarter's book. Note that because of **X2**,
changing the Python side means editing source, not config.

---

## X4 — R has an annuity shape; Python does not · **C**

`R/ecl_ead_curve.R:200-216` implements bullet, linear and annuity. Python
implements bullet and linear; everything that is not bullet takes the linear
ramp. Invisible today twice over — no shipped rule selects `annuity`, and
`NOMINALINTERESTRATE` is zero everywhere (**I6**) so R's annuity would degenerate
to linear anyway. But the shapes are set in config, which Risk can edit: the
moment someone writes `shape: annuity` and the extract carries a real rate, R
produces a convex curve and Python a straight line.

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
5. **Reconcile the two rule sets and make Python read the config** (X2, X3).
6. **Decide what a blank DPD means** for the 123 contracts whose customer has
   none (I5), rather than defaulting to current.
7. **Make an unmapped collateral type an ERROR** (I8). The two ports are now
   aligned on a full haircut, but both still compute silently where Excel
   returns `#N/A` and forces the mapping to be fixed.
8. **Ask for the component breakdown and off-balance amounts** (I6), or record
   that the provision cannot be disclosed by component.

Items 1, 2, 5 and 7 change reported numbers. The rest change what can be seen.
