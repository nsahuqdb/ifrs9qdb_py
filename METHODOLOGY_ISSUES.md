# Methodology issues — what needs reworking

This is a register of MODEL and CALCULATION problems, not software defects.
Everything here runs without error and produces plausible-looking numbers. That
is what makes the list worth having: none of it announces itself.

Except where an item is marked FIXED, nothing in this document has been
changed in the code. Each item states what
the engine does now, the evidence, the size of it, and what a correct treatment
would look like — so Risk can decide, rather than find a number has moved.

One entry, M2, was raised as a defect and is not one. It is kept rather than
deleted because the wrong reading had already reached two other files, and an
item that says "this looks wrong and is not" is worth more than a gap.

M15 to M17 came from reading the raw Oracle extracts the app is fed, which
closed the one link in the EAD chain this register could not previously check.
M15 is the largest item here and the only one whose cause is outside the model:
a date the ETL misreads, which the model then prices as fact.

M18 to M22 were added on 29 October 2026, from the Q3 2026 parallel run and
the IMF's October 2026 releases. M18 is the external-scale counterpart of M1
and the most urgent: the October GCC figures make it bite in Q4. M21 and M22
are decisions rather than defects.

Every figure below is reproducible from `tests/test_methodology_issues.py`,
which pins the CURRENT behaviour. Those tests are written to fail when an item
is fixed, so a fix cannot land silently.

**Severity** — **A**: the number is wrong now, and wrong in a direction that
matters. **B**: the number is wrong but small, or right by accident. **C**: not
wrong arithmetically, but below what IFRS 9 or a model validator would expect,
or a naming and documentation problem that keeps producing wrong readings.

| | Issue | Severity |
| --- | --- | --- |
| M1 | Scenario weights move the wrong way with the forecast — **fixed (V2, Oct 2026)** | **A** |
| M2 | The two scales' macro factors are opposite by design and share a name | **C** |
| M3 | Monthly PD accumulates by summing, so curves reach certain default | **A** |
| M4 | The amortisation step count floors, ending the curve before maturity — **fixed (Oct 2026)** | **A** |
| M5 | The EAD fallback does not reproduce the schedules it stands in for | **A** |
| M6 | Month 1's loss is not discounted | **B** |
| M7 | Annual PD is split evenly across months rather than as a hazard | **B** |
| M8 | Scenario weights do not sum to one — **restated (Oct 2026)** | **B** |
| M9 | No quantitative SICR test exists | **C** |
| M10 | LGD is a single static number, not forward-looking | **C** |
| M11 | Collateral above ~50% coverage is worth nothing | **C** |
| M12 | Stage 3 is booked at 100% with no recovery | **C** |
| M13 | Revolving and off-balance-sheet lifetime is contractual | **C** |
| M14 | Only one macroeconomic variable carries any weight | **C** |
| M15 | Every schedule beyond 2029 is silently discarded, cutting lifetime | **A** |
| M16 | The EAD curve adds each instalment's interest to the balance | **B** |
| M17 | The rising curve is half defect and half correct | **C** |
| M18 | External scenario weights still move the wrong way with the forecast | **A** |
| M19 | The external percentile rank uses a shorter history than its mean and sd (Excel) | **B** |
| M20 | Non-oil GDP entered as a fraction where the model is in points | **A** |
| M21 | Q4 non-oil GDP source: the IMF's own post-war path, and a revised history | **C** |
| M22 | What follows a model change: the weights' single variable and the stress slope range | **C** |

---

## M1 — Scenario weights move the wrong way with the forecast · **A** · FIXED

**Fixed in 0.29.0 (Oct 2026)**, in both engines, to the 2026 V2 method of
`IFRS_9_Scenarios_Probabilities_2026Q3.xlsx`: the band edges are fixed on the
history (`mu + sigma * z`, the 10/25/75/90th percentiles) and growth is
centred on the forecast, `norm.cdf(mu + sigma * z, loc=f, scale=sigma)`. At a
forecast equal to the historical mean the weights are exactly 10/15/50/15/10;
a lower forecast puts more weight on the downturns. On the Q3 2026 inputs
(history 2015–2025, forecasts −0.19% and 2.92%) the engine now reproduces the
workbook's 22.11 / 19.40 / 43.02 / 9.73 / 5.74 to 1e-9; the old code gave the
mirror, 5.74 / 9.73 / 43.02 / 19.40 / 22.11. The tests in
`TestM1ScenarioWeightsFollowTheForecast` pin the corrected direction. The
external (GCC) weights still use the V1 band construction — see the note at
the end of this section. The text below is the original finding.


*You raised this one. It is worse than "weird": the weights are an exact mirror
image of what they should be.*

`compute_internal_scenario_weights` in `etl/macro.py`.

The weights are meant to say how likely each macro scenario is, given the
forecast. As the non-oil GDP forecast falls, the weight on **Significant
Downturn** falls and the weight on **Significant Uptrend** rises:

| Non-oil GDP forecast | Sig. Downturn | Slight Downturn | Base | Slight Uptrend | Sig. Uptrend |
| --- | --- | --- | --- | --- | --- |
| +6.00% | 0.3365 | 0.2369 | 0.3641 | 0.0464 | 0.0161 |
| +4.44% *(the run's own)* | 0.2006 | 0.2075 | 0.4598 | 0.0897 | 0.0424 |
| +2.81% | 0.1011 | 0.1509 | 0.5000 | 0.1491 | 0.0989 |
| 0.00% | 0.0213 | 0.0565 | 0.3939 | 0.2323 | 0.2960 |
| −5.00% | 0.0004 | 0.0025 | 0.0766 | 0.1320 | 0.7886 |
| −10.00% | 0.0000 | 0.0000 | 0.0030 | 0.0132 | **0.9838** |

At a forecast of −10% growth — a depression — the model puts **98.4% weight on
Significant Uptrend and zero on Significant Downturn**.

**Why.** The band edges are computed as

```python
cdf = norm.cdf(f + sigma * z_sorted, loc=mu, scale=sigma)
```

The forecast `f` is inside the value being evaluated, while the distribution
stays anchored on the historical mean. So the thresholds slide down WITH the
forecast and never register that it moved. The two arguments are the wrong way
round. With the threshold fixed on history and the forecast as the new mean —

```python
cdf = norm.cdf(mu + sigma * z_sorted, loc=f, scale=sigma)
```

— the same code produces the exact reverse of the table above, which is the
direction everyone expects. The current output at any forecast equals the
corrected output read backwards; that symmetry is the fingerprint of a swapped
pair, not a modelling choice.

**What it costs.** This is the mechanism behind the macro result already
recorded in `PORT_STATUS.md`: shocking non-oil GDP down 2pp *reduces* the
provision by 0.78% when the weights are on `auto`, and *raises* it by 1.16%
when they are held. The weighting effect is larger than the PD effect and
points the other way, so the provision is currently anti-cyclical by
construction.

**Also affected.** `compute_external_scenario_weights` uses the same band
construction on GCC growth: confirmed and sized in M18, not yet changed.

---

## M2 — The two scales' macro factors are opposite by design · **C**

*Raised as a severity-A defect. It is not one — both scales are correct. What
is wrong is that two different quantities are both called "SF".*

| | Internal, TTC 14.6% | | External, TTC 2.0% | |
| --- | --- | --- | --- | --- |
| | factor | PD | factor | PD |
| Significant Downturn | **+0.316** | **0.2303** | **−1.150** | **0.0412** |
| Base Case | +0.019 | 0.1504 | +0.246 | 0.0093 |
| Significant Uptrend | −0.258 | 0.0948 | +1.774 | 0.0012 |

Both raise the provision in a downturn. There is nothing to fix in either
formula, and no reason to make them share one — the Vasicek form on the
external side is the standard treatment and the internal side is a direct
probit translation of a fitted PD. Forcing them into one functional form would
be effort spent on symmetry rather than on accuracy.

**The two factors are different quantities, and should be named differently.**

| | Internal | External |
| --- | --- | --- |
| Call it | **stress factor** | **shift factor** |
| It is | `Φ⁻¹(PD_fitted) − Φ⁻¹(TTC_anchor)` — a probit gap between a regression's fitted PD and the anchor | `Φ⁻¹(percentile rank of growth in its own history)` — a systematic-factor reading of the state of the world |
| Positive means | **worse**: PD increases | **better**: PD decreases |
| The formula | **adds** it: `Φ(Φ⁻¹(TTC) + stress)` | **subtracts** it: `Φ((Φ⁻¹(TTC) − √R·shift) / √(1−R))` |
| Scale | whatever the regression gives — roughly ±0.3 here | standard normal by construction |

Each formula consumes its own factor with the matching sign. Comparing the two
**at the same numeric value** is what makes them look contradictory, and it is
a comparison that means nothing: a stress factor of +1 and a shift factor of
+1 describe opposite states of the world.

**Why this is in the register at all.** That comparison was made twice, and the
wrong conclusion reached three files — the `external_combined_sf` docstring
carried a warning that a downturn *lowers* the provision, `PORT_STATUS.md`
repeated it as an open question for Risk, and this register opened it as a
severity-A defect. All three are now corrected. A single name covering two
opposite conventions will keep producing that error, in review as much as in
code, until the names are separated.

**What would actually have to change** if the Vasicek form were ever wanted on
both sides: not the sign, but the *kind* of factor. Vasicek assumes a standard
normal systematic factor. The internal stress factor spans about 0.6 in total,
so feeding it to the Vasicek form as-is collapses the scenario spread from
0.136 to 0.046 — the macro overlay all but disappears. It would have to be
rebuilt as a percentile-probit like the external one, which discards the
regression entirely, or standardised first. Recorded here so the option is not
re-explored from scratch; it is not recommended.

## M3 — Monthly PD accumulates by summing, so curves reach certain default · **A**

*The second half of your point (a).*

`convert_to_monthly_stpd` in `etl/macro.py`:

```python
monthly = annual_marg[idx] / 12.0
cum = np.minimum(np.cumsum(monthly), pd_cap)   # pd_cap = 1.0
```

Cumulative PD is a running **sum** of marginals. A probability accumulated by
summing grows without bound and has to be clipped at 1; accumulated correctly,
`1 − Π(1−h)`, it approaches 1 but never reaches it.

**Measured on the run's own StPD file (126 curves):**

- **32 curves reach cumulative PD = 1.000000** — certain default. The earliest
  at **month 114 (9.5 years)**; median month 346.
- At 10 years the sum overstates cumulative PD by a mean of **+0.163**, worst
  **+0.364**.
- At 50 years the mean overstatement is **+0.235**.
- All 32 saturating curves are ones where the survival accumulation stays below
  1 — the saturation is entirely an artefact of the method.

The error is concentrated in the weak grades. On `Al Dhameen|1` the two agree
to four decimals at every horizon; on the worst buckets a lifetime ECL is being
computed against a PD of exactly 1.

**It is also internally inconsistent.** `stress.conditional_pd`, used by the
roll-forward, applies the correct survival formula
`(cum(k+t) − cum(k)) / (1 − cum(k))`. The same codebase therefore treats the
same curve two different ways depending on which screen you are on.

The code comment says LIC expects the sum. That may be true of the file format,
but it means the **lifetime ECL for long-dated weak exposures is materially
overstated**, and it should be a documented, approved deviation rather than a
comment.

---

## M4 — The amortisation step count floors, ending the curve before maturity · **A** · FIXED

**Fixed in 0.29.0 (Oct 2026)**, in both engines. Payment dates are now
counted back from maturity — `N, N − f, N − 2f, …` — so there are
`ceil((N − D) / f)` of them and the last one falls at maturity. This is what
LIC does in its own fallback: on the Al Dhameen book (annual payments, no
schedule) it matches 122 of 128 contracts in run 324 and all 32 Stage 1/2
contracts in run 330, where counting forward from the start matches 81 of 128.
When `N − D` is a multiple of `f` the two give the same curve, which is why
the forward count matched every monthly and quarterly Business Finance case.
The original finding follows.


`fallback_ead_curve` in `engine.py`: `n_steps = max(1, amortising // f)`.

Integer division discards the remainder, so the balance runs to zero early:

| Term | Payment frequency | Steps | Balance reaches zero |
| --- | --- | --- | --- |
| 10m | 3m | 3 | month 10 — correct |
| 14m | 3m | 4 | **month 13 — 1 month early** |
| 23m | 12m | 1 | **month 13 — 10 months early** |

For a 23-month facility paying annually, **ten months of exposure disappear
from the ECL**. The contract is priced as if it were repaid in full at month 13.

1,194 contracts carry quarterly frequency and 152 annual, so this is not a
corner case. Affects only contracts on the fallback curve — but that is 24% of
the book and every contract the maturity lever touches.

---

## M5 — The EAD fallback does not reproduce the schedules it stands in for · **A**

*Back-tested against the R engine's own formula, swept across every parameter
it exposes. R itself is not installed here, so `build_ead_fallback_curve` was
transcribed line for line from `R/ecl_ead_curve.R` and run against the real
schedules.*

### First: what the "supplied schedule" actually is

The engine does **not** transform a schedule it is given.
`build_ead_schedule_curves` splits `EADLifetime` by contract and
`resolve_ead_curve` returns it untouched, truncated to twelve months for
Stage 1. The Python port does the same. That part is a clean pass-through.

But `LifeTimeParameterOther.csv` is **not the raw repayment schedule** — it is
derived from it by `build_lifetime_parameter_other`:

| month | EAD taken as |
| --- | --- |
| 0 | `AccountMaster.OnBalance` — the current exposure |
| below the first scheduled month | `AccountMaster.OnBalance` |
| otherwise | `BALANCE + REPAYMENT` at the **first** scheduled payment with month > m |

Verified against the data: **month 0 equals OnBalance on all 4,225 supplied
curves, with zero exceptions.** The forward step lookup is what produces the
staircase shape — 995 of the curves step every three months, 894 every month.

This also corrects an earlier reading in this register. The rise between
months 0 and 1 on 271 contracts is the `+ REPAYMENT` add-back, not a drawdown.
A further 1,171 curves rise again later, which the add-back does not explain.

Whether that derivation faithfully represents the source data has since been
tested against the raw `RepaymentSchedule` extract, and the answer is in three
parts. The derivation is faithful to its own rule — 94.4% of curve points equal
the `BALANCE + REPAYMENT` the schedule gives, and month 0 equals `OnBalance` on
**all 5,399 curves without exception**. But the rule itself adds each
instalment's interest to the balance (**M16**), the rising curve it produces is
part artefact and part real (**M17**), and, far more seriously, every payment falling after
2029 is discarded before the curve is built (**M15**). M15 is the reason 1,124
contracts are priced over a lifetime that ends in December 2029 whatever their
maturity date says.

### The horizon is wrong before the shape is considered

The fallback's length comes from `months_to_maturity`, not from the schedule:

| | run_00001 | run_00002 |
| --- | --- | --- |
| matches the schedule's own length | 67.4% | 70.5% |
| fallback horizon **longer** | 1,286 contracts, median **+18 months** | 1,325, median **+21** |
| fallback horizon shorter | none | none |

A third of the book would be priced over a year and a half more exposure than
its own schedule runs to.

### Every parameter R's formula exposes, swept

Shape, payment frequency, deferral and rate, built to the schedule's own
horizon so this isolates shape. 3,939 contracts on run_00001:

| shape | frequency | area within ±1% | area within ±10% | whole curve within ±1% |
| --- | --- | --- | --- | --- |
| bullet | any | 14.7% | 24.8% | 10.2% |
| linear | as-is | 15.2% | 34.8% | 12.9% |
| linear | quarterly | 14.6% | **40.6%** | 7.3% |
| annuity | as-is | 16.4% | 35.7% | **13.7%** |
| annuity | quarterly | 14.5% | **41.8%** | 7.2% |

**No parameterisation exceeds 42% within ±10%, and none reproduces more than
14% of the curves outright.** This is not a calibration gap that better
parameters would close — the supplied curve is a forward step lookup into a
repayment table, and a smooth parametric decline cannot express that shape at
any setting.

### In aggregate it is nearly unbiased, which is why nobody noticed

Pricing the 3,680 Stage 1 and 2 contracts that have a schedule, each way:

| EAD curve used | ECL | vs the real schedule |
| --- | --- | --- |
| real schedule | 665,623,988 | — |
| linear — what the rule table gives Business Finance pt 4 | 658,129,738 | **−1.1%** |
| annuity | 670,258,954 | +0.7% |
| bullet | 832,332,794 | **+25.0%** |

So the headline provision would barely move, while individual contracts are
wrong by a median 13% and a worst case of 245%. **The reported total is
defensible; nothing computed per contract is** — which is every stress lever,
every what-if, and every movement attribution.

### 1,181 contracts are priced on a shape nobody has ever observed

| portfolio | payment type | shape given | with a real schedule |
| --- | --- | --- | --- |
| Business Finance | 4 | linear | 4,174 — testable |
| Business Finance | 3 | bullet | 51 — testable |
| **Al Dhameen** | **4** | **bullet** | **0 — never observed** |
| Off BS | 3 | bullet | 0 — never observed |
| Tasdeer | 3 | bullet | 0 — never observed |
| Investments | 3 | bullet | 0 — never observed |
| Banks and Fis | 3 | bullet | 0 — never observed |

A bullet for payment type 3 is reasonable on its face. **Al Dhameen payment
type 4 is the one to question**: it is the same payment type as Business
Finance pt 4, which the same rule table sends to `linear`, and the ECL table
above puts the gap between those two choices at 25%. All 150 Al Dhameen
contracts take it, and not one of them can be checked.

## M6 — Month 1's loss is not discounted · **B**

`sum_marginal_ecl` builds `t = np.arange(H)`, so the month-1 marginal loss is
divided by `(1+EIR)^0 = 1`. A default during month 1 is discounted as though it
happened on the reporting date.

Across the 6,108 Stage 1 and 2 contracts:

| Convention | Provision | vs current |
| --- | --- | --- |
| `t` — current | 867,239,707 | — |
| `t + 0.5` — mid-period | 865,851,146 | −0.160% |
| `t + 1` — end of period | 864,465,051 | −0.320% |

About **1.4m to 2.8m of overstatement**. Small, systematic, and free to fix.
Mid-period is the usual convention where losses are assumed to arrive evenly
through the month.

---

## M7 — Annual PD is split evenly across months rather than as a hazard · **B**

*The first half of your point (a).*

`monthly = annual_marg[idx] / 12.0` gives every month of a year the same
marginal PD. The survival-consistent conversion is
`h = 1 − (1 − p_annual)^(1/12)`.

Taken alone this one is second-order: both reach the same value at the year
boundary, and the difference within a year is `≈ (11/288)·p²`. It matters for
two reasons rather than one:

- **Discounting.** A default assumed at a uniform rate is discounted
  differently from one following a declining-survival hazard. Combined with M6
  the two biases point the same way.
- **The 12-month horizon.** Stage 1 uses exactly the first year, so the shape
  inside that year is the whole calculation for 2,497 contracts.

Fixing M7 without M3 would be pointless — they are the same conversion — but
M3 is where the money is.

---

## M8 — Scenario weights do not sum to one · **B**

`internal_scenario_weights.explicit_weights` sums to **1.0003**. The engine
deliberately does not normalise, because normalising would disagree with the R
implementation, and that behaviour is pinned by a test.

Reproducing the reference is the right call for a port. As a model it means
every provision is scaled by 1.0003 — about **0.7m on the current book** — for
no reason anyone intended. Either the published weights should be restated to
sum to 1, or the normalisation should be applied and the change approved.

Note this only bites when the mode is `explicit`. The production config runs
`auto_non_oil_gdp_cdf`, where the weights are computed — and the computed
weights are M1.

**Restated (Oct 2026).** The shipped explicit weights are now the rounded V2
weights with Base Case taking the residual, so they sum to exactly 1.

---

## M9 — No quantitative SICR test exists · **C**

`classify_stage` takes `dpd, default_flag, watchlist, local_any, portfolio,
customer`. There is no origination PD, no current-versus-origination
comparison, no relative-deterioration threshold. Staging is days past due plus
binary flags plus contagion plus the Tasdeer collective rule.

IFRS 9 requires an assessment of *significant increase in credit risk since
initial recognition*. A 30-days-past-due backstop and a watchlist flag are
backstops **to** that assessment, not a substitute for it. As built, a borrower
whose PD has tripled since origination but who is paying on time and is not
watchlisted stays in Stage 1.

The report does not even carry an origination PD column, so the test cannot
currently be computed from the run output — the inputs would need to come
through the ETL first.

This is the largest gap against the standard in the register.

---

## M10 — LGD is a single static number, not forward-looking · **C**

`compute_lgd(on_balance, collateral_net, base, unsecured_floor,
zero_exposure_lgd)`. No scenario argument, no time index, no macro input.

Consequences:

- **The same LGD is used in every scenario.** The severe downturn changes PD
  only. In reality collateral values and recovery rates fall in a downturn,
  which is where a large part of downturn loss comes from.
- **LGD does not vary over the lifetime.** A default in month 3 and one in
  month 200 recover identically.
- **No discounting of recoveries**, and no realisation period, though
  `CollateralType.csv` carries a `RealizationPeriod` column that is read and
  never used.

The base rate is a flat 0.45 for everything unsecured. There is no segmentation
by product, seniority or collateral type beyond the haircut.

---

## M11 — Collateral above ~50% coverage is worth nothing · **C**

`LGD = base × max(unsecured_floor, (E − C)/E)` with `base = 0.45`,
`unsecured_floor = 0.5`. Once collateral covers half the exposure the floor
binds and additional collateral changes nothing.

**127 contracts carry coverage above 50%, holding 109.6m of exposure; 42 are
covered above 100%.** Each of them is provisioned exactly as if it were covered
at 50%.

A floor is a reasonable conservatism. A floor this high makes the collateral
data irrelevant for the contracts that are best secured, which is the opposite
of where precision is usually wanted, and it means any collateral revaluation
stress is inert for them — the what-if collateral lever genuinely cannot move
these contracts.

---

## M12 — Stage 3 is booked at 100% with no recovery · **C**

`stage3_method = "full_outstanding"`: the provision equals the on-balance
amount. No collateral, no recovery, no discounting, no time to resolution.
1,415,801,562 of the current 2,283,041,269 provision — **62%** — is set this
way.

This is QDB's stated basis and it diverges from LIC deliberately, so it is a
policy rather than a defect. It is in this register because:

- it makes 62% of the provision insensitive to every model input, every
  scenario and every stress lever;
- IFRS 9 measures Stage 3 as the present value of expected cash shortfalls,
  which for a secured defaulted exposure is not the full balance;
- 42 contracts are collateralised above 100% of exposure, and any that are in
  Stage 3 are being provisioned in full against collateral that covers them.

---

## M13 — Revolving and off-balance-sheet lifetime is contractual · **C**

Lifetime comes from `months_to_maturity(maturity_date, extract_date)`, floored
at 3 months.

| Portfolio | n | Median months to maturity |
| --- | --- | --- |
| Off BS | 808 | **4.7** |
| Business Finance | 5,528 | 8.0 |
| Al Dhameen | 150 | 12.0 |
| Tasdeer | 150 | 12.0 |

IFRS 9 requires the **behavioural** life for revolving facilities — the period
over which the entity is exposed to credit risk, which for a revolver that is
routinely renewed is longer than its contractual term. Off BS at a 4.7-month
median contractual life is almost certainly being measured over the wrong
horizon.

The three-month floor is a separate question: a matured facility gets three
months of exposure at the full balance (see M4), which is an assumption nobody
has documented.

---

## M14 — Only one macroeconomic variable carries any weight · **C**

`internal_v4_production` has three MEV components:

| MEV | Weight | Coefficient |
| --- | --- | --- |
| Non-oil GDP growth | **1.0** | −0.0414 |
| Qatar real estate index growth | 0.0 | −0.9005 |
| Qatar domestic credit growth | 0.0 | −4.5845 |

Real estate and domestic credit have large coefficients and zero weight, so
they contribute nothing. A property-price stress returns exactly zero, and the
app now says so on the Macro path screen rather than looking broken.

It means the entire forward-looking element of the provision rests on one
series, with **ten annual observations** behind its distribution
(mean 2.787, sd 3.738). Ten points is a thin basis for the percentile bands in
M1, and a single-variable model is thin for a model-validation review.

---

## M15 — Every schedule beyond 2029 is silently discarded, cutting lifetime · **A**

*This is the largest single finding in the register, and it was invisible until
the raw extract was read. It is a data-interpretation error in the ETL that the
model then treats as fact.*

### What the extract contains

`RepaymentSchedule.xlsx` (extract 2026-06-09, 85,160 rows, 5,705 contracts)
holds **25,466 rows — 29.9% of the file — with `START_DAT` between 1930-01-01
and 1943-01-30**, spread over **1,124 contracts**. `build_lifetime_parameter_other`
drops them, with this comment:

> Drop placeholder historical schedule entries (e.g., START_DAT = 1930-01-01)
> […] These are common when the source system uses a sentinel date for the
> final balloon payment of a fully-amortising loan

They are not sentinels and they are not placeholders. They are **2030 to 2043
read through a two-digit-year pivot** — the Excel convention where `30`–`99`
becomes 1930–1999 and `00`–`29` becomes 2000–2029. The observed range,
1930–1943, is exactly that boundary.

### Four independent proofs

Add a century to those dates and:

| check | result |
| --- | --- |
| the contract's balance path becomes monotone non-increasing | holds |
| the roll-forward identity holds on every row | 30/30 and 16/16 on the two worked contracts |
| the last scheduled payment lands on `AccountMaster.MaturityDate` | **1,124 of 1,124 exactly** |
| the final balance at that payment is 0.00 | holds |

Against the same test, the end month R actually keeps matches the contract's
own maturity date for **0 of 1,124** contracts, with a median error of **39
months**. Worked example, contract `520610`:

```
raw, sorted by START_DAT        repaired (+100y)
1930-03-17  bal 1,381,651.58    2026-06-17  bal 22,106,426.60
1930-06-17  bal         0.00    ...
2026-06-17  bal 22,106,426.60   2030-03-17  bal  1,381,651.58
...                             2030-06-17  bal         0.00
2029-12-17  bal  2,763,303.25
```

`AccountMaster.MaturityDate` for `520610` is **2030-06-17**. The repaired
schedule's last payment is 2030-06-17 at a zero balance. R keeps the schedule
only to 2029-12-17, where **2,763,303.25 is still outstanding**.

### Why a dropped row changes the provision

`resolve_ead_curve` takes the horizon from the curve, not from the maturity date:

```r
curve <- schedule_curves[[cid]]
if (!is.null(curve) && length(curve) > 0) {
  H <- length(curve)                     # the maturity date is never consulted
  if (stage == 1L) H <- min(12L, H)
  return(list(ead = curve[seq_len(H)], horizon = H, shape = NA_character_))
}
```

So dropping the post-2029 rows shortens `end_month`, which shortens `H`, which
ends the ECL sum early. The PD term structures run to 600 months, so nothing
else limits it — the truncated EAD curve is the binding constraint.

### It is live in both delivered runs

The fingerprint is unmistakable. December 2029 is month 47 from the
2025-12-31 extract and month 50 from 2025-09-30:

| | run_00001 (2025-12-31) | run_00002 (2025-09-30) |
| --- | --- | --- |
| longest supplied curve, any contract | **month 47** | **month 50** |
| curves ending exactly there | 506 | 557 |
| of those, maturity is later | **506 of 506** | **557 of 557** |
| their on-balance exposure | 1,914,805,490 | 2,074,643,215 |
| share of scheduled on-balance | 35.1% | 35.1% |
| months of life cut, median | 34 | 31 |
| stage 1 / 2 / 3 | 186 / 254 / 66 | 174 / 319 / 64 |

**No curve in either run extends past December 2029.** On a book whose own
report gives these contracts a median 78 months to maturity, that is not a
coincidence — it is the 2030 cliff.

### What it costs

**Superseded figures removed.** This section first reported +2.30% to +2.99% of
Stage 2 for December, from a method that found only the contracts ending exactly
at December 2029 and missed 393 others. The reconciled figures, measured by the R
package itself on the June extract and with real repaired schedules on December,
are in `INPUT_DATA_ISSUES.md` §1:

| book | Stage 2 ECL understated | total ECL understated |
| --- | --- | --- |
| **June 2026** — R, raw extract, dates repaired in the input | **+6.81%** (+38.50m) | **+2.01%** |
| **December 2025** — all 861 truncated curves, real schedules | **+3.70%** (+22.02m) | **+0.96%** |

Stage 1 is capped at twelve months and Stage 3 is booked at 100% (M12), so the
whole effect is Stage 2. The direction was never an estimate — every missing
month adds a non-negative marginal loss.

### The shape that does the damage

The loss is not proportional to the months cut, because the truncation lands
where the principal has barely moved. Contract `599073`, an annual payer:

| | |
| --- | --- |
| on balance | 121,636,306 |
| maturity | 2043-01-30 — **199 months** |
| payments in the schedule | 17, of which **14 carry a 1930s year** |
| curve emitted | months 0–30 |
| balance where the curve stops | **121,580,510 — 100.0% of the exposure** |

Its instalments run from 2.59m up to 17.87m, so the three surviving years retire
2.6m of 121.6m, and every payment that actually repays the loan is one of the
fourteen that were dropped. A back-loaded facility is exactly the profile where
this is worst, and exactly the profile a long project loan has.
`INPUT_DATA_ISSUES.md` traces this contract through all three implementations.

### What a correct treatment looks like

Read the source dates correctly. A schedule row whose year is below 1950 in an
extract taken in 2026 is not a historical payment; the pivot has wrapped it.
Either fix it at the read (`year < 1950` → `+100`), or, better, have the
extract deliver a four-digit year and make a pre-extract-date schedule row a
validation error rather than something to drop quietly. Separately,
`resolve_ead_curve` should not take a lifetime from a file's row count when
`AccountMaster.MaturityDate` is right there: a curve that ends before maturity
should be extended or rejected, not believed.
---

## M16 — The EAD curve adds each instalment's interest to the balance · **B**

*The one-line rule `EAD = BALANCE + REPAYMENT` is documented in both ports as
"the balance BEFORE this payment is applied". Against the source data it is not
that. It is the balance plus the interest portion of the next instalment.*

### The identity the data actually satisfies

`REPAYMENT` is principal **plus** interest, and `BALANCE` is the balance after
the payment. So the balance standing before payment *j* is
`BALANCE_j + REPAYMENT_j − PROJ_INT_j`. Tested on the 54,007 consecutive
payment pairs in the 2026-06-09 extract:

| candidate | holds |
| --- | --- |
| `prior = BALANCE + REPAYMENT − PROJ_INT` | **49,446 (91.6%)** |
| `prior = BALANCE + PRINCE_DUE` | 46,656 (58.7%) |
| `prior = BALANCE + REPAYMENT` — what the engine uses | **12,418 (23.0%)** |

`PRINCE_DUE` is not the amortisation either; on contract `504055` the balance
steps down by 200,525.61 a quarter while `PRINCE_DUE` reads 173,185.24, and
`200,525.61 + PROJ_INT` reproduces `REPAYMENT` to the cent.

### Size

Running the shipped derivation on the raw extract and comparing each curve
point to the payment it came from:

| | |
| --- | --- |
| curve points testable | 55,000 |
| equal to `BALANCE + REPAYMENT` | 51,921 (94.4%) — the code does what it says |
| equal to the true prior balance | 15,046 (27.4%) |
| overstatement, aggregate | **+0.43%** (375,583,006 across the curve) |
| per point | median +0.29%, p95 +1.26%, p99 +1.28%, max **+42.4%** |

That 94.4% is worth reading as a pass: the derivation is faithful to its own
documented rule, and both ports implement it identically. The rule is what is
wrong, not the code.

### An independent corroboration

An amortising balance cannot rise. On rows that carry a repayment,
`BALANCE + REPAYMENT` rises from one payment to the next **3,555 times**; the
true prior balance rises **369 times**. Nine in ten of those rises are the
projected interest being added, not anything in the loan.

### What a correct treatment looks like

`EAD = BALANCE + REPAYMENT − PROJ_INT` on rows that carry a repayment, and
`EAD = BALANCE` on rows that do not (see M17). IFRS 9 measures exposure at
default, so the balance outstanding is the right quantity; the instalment's
interest is not exposure, and it is being counted twice — once inside the EAD
and again through the discount rate. The effect is small and in the opposite
direction to M15, which is precisely why neither shows up in a total.

---

## M17 — The rising curve is half defect and half correct · **C**

*This item was originally raised as two findings. The first has been tested
against the source and **withdrawn** — it was my error. The second stands, and it
is the reason M16 must not be fixed the obvious way.*

### WITHDRAWN: payments sharing a month are not lost

An earlier draft recorded that where two payments fall in the same calendar
month, the grid holds one value and the second is never looked up — **933
payments across 78 contracts**.

Checked against the extract, **all 933 are exact duplicates**: identical in
`PRINCE_DUE`, `PROJ_INT`, `REPAYMENT` and `BALANCE`, not merely in month. Not one
contract-month in the extract holds two genuinely different payments. **Dropping
the extra row is correct**, and the grid absorbing it is right rather than wrong.

What is real is upstream: the extract emits **1,689 fully duplicated rows** across
78 contracts, and nothing checks that it hasn't. That is recorded as **I7** in
`INPUT_DATA_ISSUES.md`, where it belongs. The monthly grid is not the problem;
it happens to hide one.

A monthly grid would still lose a genuinely fortnightly schedule. This book does
not have one, so that is a latent limitation rather than a present defect.

### A quarter of the schedule is not a repayment at all

**14,687 of 59,694 future rows (24.6%)** carry `REPAYMENT = 0` with a **rising**
`BALANCE` — a facility in its grace or drawdown phase, capitalising interest.
Contract `609199` runs 21,857,894 → 22,931,091 over fourteen months with no
payment due. For those rows `BALANCE + REPAYMENT` reduces to `BALANCE`, which
is correct, and the rising curve is correct too: exposure genuinely grows.

This is the trap. **2,918 of the 6,473 rising steps (45%) are legitimate
accrual.** Forcing the curve to be non-increasing — the obvious guard against
M16 — would understate every project and construction facility on the book.
Whatever is done about M16 has to distinguish a repayment row from an accrual
row, and the `REPAYMENT = 0` flag is what does it.

### For scale

Across all 5,399 curves the delivered EAD sums to 149,662,922,630. Capped to be
non-increasing it sums to 148,531,385,630 — 0.76% lower. Most of that gap is
accrual that should be there.

### The tool already noticed, and explained it away

`DERIVED_LTPO_ead_nonincreasing` failed in both delivered runs — **1,849**
contracts in run_00001 and **2,297** in run_00002 with a rising EAD. It is one of
only **two INFO-severity checks in a suite of 114**, so it never gates, and its
message supplies the reason before anyone asks: *"expected for revolving/
off-balance products and contracts with interest/fee accrual baked into the
schedule"*.

That is half right, which is the worst kind. About 45% of the rises are the
genuine accrual it describes; the rest are M16's projected interest. The check
found a real symptom and its own message classified it as normal. See §2 of
`INPUT_DATA_ISSUES.md`.

---

## M18 — External scenario weights still move the wrong way with the forecast · **A**

*M1 for the external scale. M1 fixed the internal weights in 0.29.0 and noted
that the external ones use the same band construction; they do, and they were
not changed.*

`compute_external_scenario_weights` in `etl/macro.py`, called once per forecast
year by `compute_external_scenario_weights_per_year`. That part is right and is
the methodology: each of the five forecast years has its own weight vector,
from where that year's GDP-weighted GCC growth forecast falls on the 1982–
history, and maturities beyond year 5 take the average of the five
(`apply_scenario_weights_per_year`; the Excel ETL's Calculations D69:H74). The
monthly cumulative PD is then capped at 1 (M3). What is wrong is the band
construction inside each year, which is the V1 one M1 fixed for the internal
scale:

```python
cdf = norm.cdf(f + z_sorted * sigma, loc=mu, scale=sigma)    # now (V1)
cdf = norm.cdf(mu + sigma * z_sorted, loc=f, scale=sigma)    # V2, as M1
```

**Measured on the packaged GCC history** (1982–2024, mean 3.0217, sd 4.4912),
one forecast year:

| GCC growth forecast | Sig. Downturn | Slight Downturn | Base | Slight Uptrend | Sig. Uptrend |
| --- | --- | --- | --- | --- | --- |
| +9.28% *(Oct 2026 WEO, 2027)* | **0.5446** | 0.2194 | 0.2168 | 0.0156 | 0.0037 |
| +3.02% *(the history's mean)* | 0.0999 | 0.1499 | 0.5000 | 0.1501 | 0.1001 |
| 0.00% | 0.0253 | 0.0636 | 0.4117 | 0.2280 | 0.2713 |
| −5.02% *(Oct 2026 WEO, 2026)* | 0.0011 | 0.0058 | 0.1253 | 0.1732 | **0.6946** |
| −10.00% | 0.0000 | 0.0002 | 0.0129 | 0.0398 | **0.9472** |

A GCC recession of −5% puts 69% of the weight on Significant Uptrend; the
rebound after it puts 54% on Significant Downturn. V2 gives the same table read
backwards. On the 26q3-lic reference files (history to 2025) the October path
gives 69.7% Significant Uptrend in year 1 and 51.8% Significant Downturn in
year 2.

**What it costs**, investment curves on the 26q3-lic config and reference files,
V1 against V2 (cumulative PD, relative):

| Bucket | 12 months, Q3 path | 60 months, Q3 path | 12 months, Oct 2026 path | 60 months, Oct 2026 path |
| --- | --- | --- | --- | --- |
| 6 | −41.0% | +14.4% | −77.1% | −59.0% |
| 9 | −39.0% | +14.4% | −73.9% | −52.8% |
| 12 | −33.7% | +13.4% | −64.8% | −35.9% |
| 15 | −24.6% | +9.2% | −48.1% | −11.5% |
| 18 | −16.3% | +1.1% | −30.9% | −4.9% |

The Q3 path (1.82, 4.87, 4.00, 4.06, 3.78) sits just below the mean in year 1,
so V1 already understates the 12-month external PD by 16–41% and overstates the
long end. The October path (−5.02, 9.28, 6.31, 4.17, 3.60) makes it large:
the external Stage 1 provision would be understated by up to three quarters.

**The Excel ETL does the same** (V1, one vector a year, the average beyond
year 5), so LIC reproduces it; changing it is a methodology decision, as the
internal one was. A correct treatment is V2, with the per-year vectors and the
average kept. `TestM18ExternalWeightsFollowTheForecast` pins the current
direction.

---

## M19 — The external percentile rank uses a shorter history than its mean and sd · **B**

*Found reconciling LIC run 330's investment curves (Q3 2026 validation,
"External PD: Excel formula check").*

In the Excel ETL the external scaling factor is
`NORM.S.INV(PERCENTRANK.EXC('GCC GDP'!D28:AT28, growth))`: the rank is taken
against 1982–2024, while the history's mean and sd use `D28:AU28`, 1982–2025.
Column AU was added with its own formulas and the PERCENTRANK range in all six
sheets was not extended. The engine ranks against the same years it takes the
mean and sd from (`external_combined_sf` on `gcc_weighted_history`, 1982 to
the last year in the file).

**Size.** With the rank on 1982–2024 the engine reproduces the Excel StPD to
1e-6 in every year; with 1982–2025 it is up to 2.7% higher. On LIC run 330 the
investment book is +30,487 in Stage 1 and +7,291 in Stage 2 from this alone.

**Decision.** Copy the Excel's range (a config setting for the rank's last
year), or have the Excel corrected. The engine's reading is the consistent one.
`TestM19TheRankUsesTheWholeHistory` pins it.

---

## M20 — Non-oil GDP entered as a fraction where the model is in points · **A**

*Q3 2026: why run 14 and LIC run 330 overstate Stage 2.*

The V4 model is calibrated in percentage points: model.yml has non-oil GDP in
`percentage_points` with multiplier 1 and sd 9.8877, and the Excel ETL's
Inputs_Lending Portfolio W5:W9 hold −0.18876, 2.91929, 3.03143, 3.78499,
3.10496 (Base Case D9 `= W5 + H5 × z`, the logit
`−1.5474945 − 0.041414507 × 3.78499` gives D27 = 0.153911). LIC runs 326 and
330, and the 26q3-lic snapshot, were given the same numbers as fractions
(−0.0019, 0.0292, …). Nothing fails: the model reads 2.9% growth as 0.029%.

**Size.** The scaling factor is about 0.12 in every year instead of
0.125 / 0.054 / 0.051 / 0.034 / 0.049. QDB 3- (TTC 8.01%), V2 weights,
cumulative PD:

| Year | Points (correct) | Fractions | Difference |
| --- | --- | --- | --- |
| 1 | 11.59% | 11.50% | −0.7% |
| 2 | 20.55% | 21.57% | +5.0% |
| 3 | 28.50% | 30.40% | +6.7% |
| 4 | 35.37% | 38.16% | +7.9% |
| 5 | 41.70% | 44.99% | +7.9% |

LIC run 330: Stage 1 0.88m too low, Stage 2 8.98m too high. In the engine's
automatic weights the fraction also looks like zero growth against a history
in points and moves weight to the downturns (32.1 / 23.6 / 37.6 / 5.0 / 1.8
against the workbook's 22.1 / 19.4 / 43.0 / 9.7 / 5.7).

**Where it stands.** An input, not code: the engine warns
(`check_gdp_units`), the config update writes points, and the IMF's own figures
are in points. A correct treatment would make the warning a stop unless the
run states it is reproducing such a run. `TestM20AFractionReadsAsNoGrowth`
pins what the model does with a fraction.

---

## M21 — Q4 non-oil GDP source: the IMF's own post-war path · **C**

*A decision, not a defect.*

Q2 and Q3 2026 used a war-adjusted non-oil path derived in
`Qatar_NonOil_GDP_Update_2026-2030.xlsx`, because the IMF's regional outlook
still had its October 2025 numbers. The IMF's October 2026 Regional Economic
Outlook (22 October 2026; on FRED 23 October) now publishes its own:

| | 2026 | 2027 | 2028 | 2029 | 2030 |
| --- | --- | --- | --- | --- | --- |
| Q3 war-adjusted input | −0.19 | 2.92 | 3.03 | 3.78 | 3.10 |
| IMF October 2026 | −1.22 | 5.64 | 2.59 | 2.59 | 2.59 |

It also revises the history the weights are measured on: 2020 −4.49 → −3.17,
2024 3.40 → 4.43, 2025 4.44 → 3.10 (an estimate). On run 14's config the
internal V2 weights move from 22.1 / 19.4 / 43.0 / 9.7 / 5.7 to roughly
25.9 / 14.1 / 31.6 / 12.7 / 15.8 with both updated. The GCC paths change as
much (Qatar 2026 −8.6% → −15.6%, Kuwait −0.6% → −16.7%; see M18 for what that
does to the external weights) and the GCC history is revised (Saudi Arabia
2022 7.5% → 12.0%).

**Decision.** Whether Q4 takes the IMF's path and revised history (the config
update does it, by model, when asked) or keeps a derived path.

---

## M22 — What follows a model change · **C**

*Two places where a recalibration needs a decision, not only new numbers.*

1. **The scenario weights rest on one variable.** The V2 method is defined on
   one series. The engine takes the internal model's main variable (the largest
   weight in model.yml; `internal_scenario_weights.variable` can name another).
   With today's model that is non-oil GDP, the only weighted variable (M14). A
   model with several weighted variables needs the method to say which drives
   the weights, or how they combine.
2. **The stress slope range is absolute.** The sensitivity screen prices a
   scenario at slopes −0.019 and −0.048 a point (stress.yml `pd_link.range`),
   estimated from QDB's default rates 2014–2024 around V4's −0.0414. The slope
   itself is now read from model.yml, so a recalibration moves it; the range has
   to be re-estimated with it.

---

## A note on how these were found, and what it says about the suite

M15 to M17 needed the raw extracts. Pointing the suite at them for the first
time — `IFRS9_SRC_INPUTS` had never been set in any run of it — turned up
fifteen failures that had nothing to do with the model:

* Five inputs are SQL*Plus HTML behind an `.xls` extension and need `lxml`,
  which was simply not installed. They reported as "missing input".
* Eight tests compared the ETL's output against a run's output without
  checking that the two came from the **same extract**. A June extract against
  a December run fails on row counts and proves nothing either way.
* One hardcoded `reporting_date == "6/30/2026"`, and one required a
  `N rows selected.` footer that a clean Excel export does not have.

None of that changes a reported number, and none of it is in this register's
scope. It is recorded because the cause is worth naming: **a test that only
runs when an environment variable is set is a test that does not run.** The
assertions were written from the R package's documentation rather than from a
file, and nothing exercised them. The same shape of problem produced the three
broken imports found earlier in the reconcile endpoints — code paths that only
fire on a button press.

Those eight now skip with the reason stated (`IFRS9_SRC_INPUTS is extract X and
IFRS9_REF_RUN is Y`), the reporting date is derived from the extract instead of
hardcoded, and the two junk-row tests skip when the extract carries no junk.
No assertion about model behaviour was weakened. With a matched extract and run
they would run properly for the first time.

---

## What to test next

The EAD chain is now checked end to end: the engine passes the supplied curve
through untouched (M5), the derivation that builds that curve has been tested
against the raw extract (M15, M16, M17), and the fallback used where no
schedule exists has been swept across every parameter it exposes (M5). What
remains open:

1. **How much M15 really costs the two delivered quarters.** The figures in
   M15 come from extending each truncated curve to its reported maturity,
   which brackets the answer but does not settle it. The exact number needs
   the `RepaymentSchedule` extract for 2025-09-30 and 2025-12-31; only the
   2026-06-09 one is held. With those two files the whole report can be
   rebuilt from repaired dates and the difference read off directly.

2. **Whether the annual PD term structure itself is right.** M3 is about the
   monthly conversion; the annual curve that feeds it has not been back-tested
   against realised default experience. There is no default history in the run
   outputs to do it with.

3. **Whether the stress levers move the provision by the right amount**, as
   opposed to the right direction. M5 shows the EAD lever rests on an
   unvalidated curve. The PD, rating and collateral levers have no equivalent
   back-test because there is no realised outcome to compare against — a
   deliberate hold-out or a back-test against a prior quarter's actuals would
   be the way in.
