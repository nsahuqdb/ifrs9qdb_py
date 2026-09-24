"""
The PD term structure: through-the-cycle PDs to the monthly curve LIC prices on.

The chain, and why each step is shaped the way it is:

  1. **Point-in-time adjustment.** A through-the-cycle PD is shifted by the
     scenario's scaling factor in PROBIT space:
     ``PD_t = Phi(Phi^-1(TTC) + SF_t)``. Working in probit rather than on the
     probability directly is what keeps the result inside (0, 1) however severe
     the scenario -- a multiplicative shock would not.

  2. **Reversion beyond the forecast horizon.** Only the first ``n_forecasts``
     years have a macro view. After that the curve reverts to the TTC level,
     interpolated in LOG space between the anchor year's PD and the TTC PD.
     Linear reversion would revert too fast in the early years.

  3. **Cumulative from marginal.** ``1 - cumprod(1 - marginal)`` -- survival,
     not a running sum. Adding marginals would exceed 1 on a long enough
     horizon.

  4. **Scenario weighting** on the MARGINAL PDs, then re-cumulated. Weighting
     the cumulative curves instead would not be the same number.

  5. **Monthly conversion.** Each year's marginal PD is spread evenly across
     its twelve months and accumulated with a running SUM, then capped. This
     step is a deliberate simplification of the survival formula above, and it
     is what LIC expects; changing it to be internally consistent would stop
     the output matching.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["pit_pd_term_structure", "pit_pd_term_structure_external",
           "basel_asrf_pit", "cumulative_pd", "apply_scenario_weights",
           "convert_to_monthly_stpd", "build_stpd"]


def pit_pd_term_structure(ttc_pd: float, combined_sf, n_forecasts: int,
                          max_maturity: int) -> np.ndarray:
    """Point-in-time marginal PD by year.

    Forecast years are shifted in probit space; later years revert to the TTC
    level through a log-space interpolation.
    """
    sf = np.asarray(combined_sf, dtype=float)
    if sf.size < n_forecasts:
        raise ValueError(
            f"combined_sf has {sf.size} elements but n_forecasts={n_forecasts}")
    if ttc_pd == 0:
        return np.zeros(max_maturity)

    out = np.full(max_maturity, np.nan)
    inv_ttc = norm.ppf(ttc_pd)
    horizon = min(n_forecasts, max_maturity)
    out[:horizon] = norm.cdf(inv_ttc + sf[:horizon])

    anchor = out[n_forecasts - 1] if n_forecasts <= max_maturity else np.nan
    if not np.isfinite(anchor):
        return out

    span = max_maturity - n_forecasts
    if span > 0:
        exp_anchor = np.exp(anchor)
        exp_ttc = np.exp(ttc_pd)
        t = np.arange(n_forecasts + 1, max_maturity + 1)
        frac = (t - n_forecasts) / span
        out[n_forecasts:] = np.log(frac * (exp_ttc - exp_anchor) + exp_anchor)
    return out


def basel_asrf_pit(ttc_pd: float, sf: float) -> float:
    """Basel ASRF point-in-time PD, used by the EXTERNAL scale.

        R  = 0.24 - 0.12 * (1 - exp(-50 * PD)) / (1 - exp(-50))
        PD = Phi( (Phi^-1(TTC) - sqrt(R) * SF) / sqrt(1 - R) )

    Two differences from the internal formula, and both matter:

      * The factor is SUBTRACTED and scaled by the asset correlation, so a
        HIGHER factor gives a LOWER PD -- the opposite sign convention to the
        internal chain. Combined with the external factor rising with growth,
        the net effect is the expected one: weaker growth raises the provision.
      * The denominator sqrt(1 - R) widens the shift for low-PD names, since R
        falls as PD rises.
    """
    if ttc_pd is None or sf is None or not np.isfinite(ttc_pd) or not np.isfinite(sf):
        return float("nan")
    if ttc_pd <= 0:
        return 0.0
    if ttc_pd >= 1:
        return 1.0
    R = 0.24 - 0.12 * (1 - np.exp(-50 * ttc_pd)) / (1 - np.exp(-50))
    return float(norm.cdf((norm.ppf(ttc_pd) - np.sqrt(R) * sf) / np.sqrt(1 - R)))


def pit_pd_term_structure_external(ttc_pd: float, combined_sf,
                                   n_forecasts: int,
                                   max_maturity: int) -> np.ndarray:
    """The external scale's PiT curve: ASRF for the forecast years, then the
    same log-space reversion the internal curve uses."""
    sf = np.asarray(combined_sf, dtype=float)
    if sf.size < n_forecasts:
        raise ValueError(
            f"combined_sf has {sf.size} elements but n_forecasts={n_forecasts}")
    if ttc_pd == 0:
        return np.zeros(max_maturity)

    out = np.full(max_maturity, np.nan)
    horizon = min(n_forecasts, max_maturity)
    for t in range(horizon):
        out[t] = basel_asrf_pit(ttc_pd, sf[t])

    anchor = out[n_forecasts - 1] if n_forecasts <= max_maturity else np.nan
    if not np.isfinite(anchor):
        return out
    span = max_maturity - n_forecasts
    if span > 0:
        exp_anchor, exp_ttc = np.exp(anchor), np.exp(ttc_pd)
        t = np.arange(n_forecasts + 1, max_maturity + 1)
        frac = (t - n_forecasts) / span
        out[n_forecasts:] = np.log(frac * (exp_ttc - exp_anchor) + exp_anchor)
    return out


def cumulative_pd(marginal) -> np.ndarray:
    """Cumulative default probability from marginals, via survival.

    ``1 - prod(1 - m)``. A running sum would drift above 1 on a long horizon.
    """
    m = np.asarray(marginal, dtype=float)
    return 1.0 - np.cumprod(1.0 - m)


def apply_scenario_weights(annual: pd.DataFrame,
                           weights: dict | pd.Series) -> pd.DataFrame:
    """Weight the scenarios, on the MARGINAL PDs.

    Weighting cumulative curves instead gives a different -- and wrong --
    answer, because the cumulative is a product of the marginals rather than a
    linear function of them.
    """
    w = pd.Series(weights, dtype=float)
    # NOT normalised. The V4 explicit weights sum to 1.0003 -- a rounded
    # snapshot of the computed ones -- and the R engine carries that through
    # rather than rescaling it. Normalising here would silently disagree with
    # the production figures by that factor.
    if abs(float(w.sum()) - 1.0) > 0.01:
        warnings.warn(f"scenario weights sum to {float(w.sum()):.4f}, not ~1.0",
                      RuntimeWarning, stacklevel=2)
    d = annual.copy()
    d["_w"] = d["scenario"].map(w)
    if d["_w"].isna().any():
        missing = sorted(d.loc[d["_w"].isna(), "scenario"].unique())
        raise ValueError(f"no weight supplied for scenario(s): {missing}")
    d["_wm"] = d["marginal_pd"] * d["_w"]
    out = (d.groupby(["rating", "maturity"], as_index=False)["_wm"].sum()
           .rename(columns={"_wm": "weighted_marginal_pd"}))
    return out.sort_values(["rating", "maturity"]).reset_index(drop=True)


def apply_scenario_weights_per_year(annual: pd.DataFrame,
                                    weights_per_year: pd.DataFrame,
                                    weights_average) -> pd.DataFrame:
    """Weight the scenarios with a DIFFERENT weight vector for each year.

    The external scale works this way: its weights come from where that year's
    regional growth forecast sits on the historical distribution, and the
    forecast moves year to year, so year 1 and year 5 do not share a weight.
    Maturities past the forecast horizon take the average of the modelled
    years, which is what the workbook's last weight row is.

    Using one flat vector for both scales -- the easy mistake, since the
    internal scale genuinely has one -- puts the domestic weights on the
    regional book.
    """
    wpy = pd.DataFrame(weights_per_year).astype(float)
    avg = pd.Series(weights_average, dtype=float)
    missing = sorted(set(annual["scenario"]) - set(wpy.columns))
    if missing:
        raise ValueError(f"no per-year weight supplied for scenario(s): {missing}")

    n_years = len(wpy)
    max_mat = int(annual["maturity"].max())
    rows = [wpy.iloc[m - 1] if m <= n_years else avg
            for m in range(1, max_mat + 1)]
    lookup = pd.DataFrame(rows, index=range(1, max_mat + 1))[list(wpy.columns)]

    col = {s: i for i, s in enumerate(lookup.columns)}
    d = annual.copy()
    d["_w"] = lookup.to_numpy()[d["maturity"].to_numpy() - 1,
                                d["scenario"].map(col).to_numpy()]
    d["_wm"] = d["marginal_pd"] * d["_w"]
    out = (d.groupby(["rating", "maturity"], as_index=False)["_wm"].sum()
           .rename(columns={"_wm": "weighted_marginal_pd"}))
    return out.sort_values(["rating", "maturity"]).reset_index(drop=True)


def convert_to_monthly_stpd(weighted_annual: pd.DataFrame, max_month: int,
                            pd_cap: float = 1.0) -> pd.DataFrame:
    """Spread each year's marginal PD across its months and accumulate.

    The accumulation is a running SUM, not the survival formula used for the
    annual curve. That is an inconsistency in the model, not in this port: LIC
    expects it, and matching it is the requirement. Months beyond the last
    modelled year repeat that year's marginal.
    """
    frames = []
    months = np.arange(1, max_month + 1)
    year_idx = np.ceil(months / 12).astype(int)
    for rating, sub in weighted_annual.groupby("rating", sort=False):
        annual_marg = sub.sort_values("maturity")["weighted_marginal_pd"].to_numpy()
        idx = np.minimum(year_idx, len(annual_marg)) - 1
        monthly = annual_marg[idx] / 12.0
        cum = np.minimum(np.cumsum(monthly), pd_cap)
        frames.append(pd.DataFrame({
            "rating": rating,
            "month_lifetime": months,
            "monthly_marginal_pd": monthly,
            "pd_lifetime": cum,
        }))
    return pd.concat(frames, ignore_index=True)


def build_stpd(term_structure: pd.DataFrame, weights, portfolios,
               ratings: pd.DataFrame, extract_date: str,
               max_month: int = 600) -> pd.DataFrame:
    """The StPD file: one cumulative PD per portfolio, bucket and month.

    ``ratings`` maps a rating to its hierarchy number, which is the bucket LIC
    keys on. Every portfolio on the same rating scale shares the same curves,
    so the table is the curve set repeated per portfolio.
    """
    if isinstance(weights, dict) and "per_year" in weights:
        weighted = apply_scenario_weights_per_year(
            term_structure, weights["per_year"], weights["average"])
    else:
        weighted = apply_scenario_weights(term_structure, weights)
    monthly = convert_to_monthly_stpd(weighted, max_month)

    hierarchy = dict(zip(ratings["rating"].astype(str),
                         pd.to_numeric(ratings["hierarchy"], errors="coerce")))
    monthly["bucket"] = monthly["rating"].astype(str).map(hierarchy)
    monthly = monthly.dropna(subset=["bucket"])

    frames = []
    for pf in portfolios:
        f = monthly.copy()
        f.insert(0, "PortfolioCode", pf)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)

    return pd.DataFrame({
        "ExtractDate": extract_date,
        "PortfolioCode": out["PortfolioCode"],
        "PDBucketDim1": out["bucket"].astype(int),
        "PDBucketDim2": "",
        "MonthLifetime": out["month_lifetime"].astype(int),
        "PDLifetime": out["pd_lifetime"],
    })


# =============================================================================
# From macroeconomic forecasts to scaling factors.
#
# This is the step that turns a scenario into the SF the point-in-time
# adjustment consumes:
#
#     stress_mevs          shift each variable by severity x its own SD
#     compute_logit_pds    a per-variable regression, in logit space
#     compute_pds_from_logits   logistic transform back to a probability
#     compute_per_mev_sf   the probit gap against the through-the-cycle anchor
#     combine_sf           weight the per-variable factors into one
#
# Two details are easy to get wrong and change every number downstream:
#
#   * The stress unit multiplier is used TWICE, and in opposite directions --
#     multiplying the shift, then dividing the variable before the regression.
#     It exists because a variable may be modelled in different units from the
#     ones it is quoted in (percent versus fraction, say). Applying it once, or
#     in the same direction both times, silently rescales the shock.
#   * The per-variable SF is a difference of PROBITS, not of probabilities. It
#     has to be, because the adjustment it feeds is itself applied in probit
#     space.
# =============================================================================

__all__ += ["build_term_structure", "build_stpd_from_static", "stress_mevs", "compute_logit_pds", "compute_pds_from_logits",
            "compute_per_mev_sf", "combine_sf", "combined_sf_for_scenario",
            "external_combined_sf", "percentrank_exc", "truncate_significant",
            "compute_internal_scenario_weights", "gcc_weighted_history",
            "build_term_structure_external", "external_gcc_forecast",
            "apply_scenario_weights_per_year",
            "compute_external_scenario_weights",
            "compute_external_scenario_weights_per_year",
            "resolve_internal_scenario_weights",
            "resolve_external_scenario_weights"]


def stress_mevs(forecasts, severity_z: float, mev_specs) -> np.ndarray:
    """Shift each macro variable by ``severity_z`` of its own standard deviation.

    Each variable moves by its OWN volatility, so a one-sigma downturn shifts a
    stable series less than a volatile one. A single shared shift would
    overstate the stable variables and understate the volatile ones.
    """
    m = np.atleast_2d(np.asarray(forecasts, dtype=float))
    if m.shape[1] != len(mev_specs):
        raise ValueError(
            f"mev_specs has {len(mev_specs)} entries but the forecast matrix "
            f"has {m.shape[1]} columns")
    sd = np.array([s["standard_deviation"] for s in mev_specs], dtype=float)
    unit = np.array([s["stress_unit_multiplier"] for s in mev_specs], dtype=float)
    return m + severity_z * sd * unit


def compute_logit_pds(stressed, mev_specs) -> np.ndarray:
    """Per-variable regression, in logit space: ``intercept + coef * (mev / unit)``.

    The division by the unit multiplier undoes the multiplication applied in
    ``stress_mevs`` -- the shock is expressed in quoted units, the regression
    is fitted in modelled units.
    """
    m = np.atleast_2d(np.asarray(stressed, dtype=float))
    intercept = np.array([s["intercept"] for s in mev_specs], dtype=float)
    coef = np.array([s["coefficient"] for s in mev_specs], dtype=float)
    unit = np.array([s["stress_unit_multiplier"] for s in mev_specs], dtype=float)
    return intercept + coef * (m / unit)


def compute_pds_from_logits(logits) -> np.ndarray:
    """Logistic transform. Written as ``e/(e+1)`` to match the reference."""
    e = np.exp(np.asarray(logits, dtype=float))
    return e / (e + 1.0)


def compute_per_mev_sf(estimated_pds, ttc_anchor: float) -> np.ndarray:
    """The scaling factor per variable: a difference of PROBITS.

    Not of probabilities -- the adjustment this feeds is applied in probit
    space, so the gap has to be measured there too.
    """
    return norm.ppf(np.asarray(estimated_pds, dtype=float)) - norm.ppf(ttc_anchor)


def combine_sf(per_mev_sf, mev_weights) -> np.ndarray:
    """Weight the per-variable factors into one factor per forecast year."""
    sf = np.atleast_2d(np.asarray(per_mev_sf, dtype=float))
    w = np.asarray(mev_weights, dtype=float)
    if sf.shape[1] != w.size:
        raise ValueError(
            f"per_mev_sf has {sf.shape[1]} columns but {w.size} weights given")
    return sf @ w


def combined_sf_for_scenario(forecasts, severity_z: float, mev_specs,
                             mev_weights, ttc_anchor: float) -> np.ndarray:
    """The whole internal chain, for one scenario."""
    stressed = stress_mevs(forecasts, severity_z, mev_specs)
    pds = compute_pds_from_logits(compute_logit_pds(stressed, mev_specs))
    return combine_sf(compute_per_mev_sf(pds, ttc_anchor), mev_weights)


def truncate_significant(x: float, significance: int = 3) -> float:
    """Truncate -- not round -- to N significant digits.

    Excel's own behaviour, and it has to be reproduced: PERCENTRANK.EXC returns
    three significant digits by truncation, and rounding instead shifts the
    probit that follows.
    """
    if x is None or not np.isfinite(x) or x == 0:
        return x
    magnitude = np.floor(np.log10(abs(x)))
    factor = 10.0 ** (significance - 1 - magnitude)
    return float(np.sign(x) * np.floor(abs(x) * factor) / factor)


def percentrank_exc(data, x: float, significance: int | None = 3) -> float:
    """Excel's PERCENTRANK.EXC.

    Exclusive, so a value ranks in ``1/(n+1) .. n/(n+1)`` and never reaches 0 or
    1 -- which matters because the result is fed to a probit, and 0 or 1 would
    give minus or plus infinity. Values outside the observed range are clamped
    to those bounds rather than extrapolated.
    """
    d = np.sort(np.asarray(data, dtype=float))
    n = d.size
    if n == 0 or not np.isfinite(x):
        return float("nan")
    if x < d[0]:
        pr = 1.0 / (n + 1)
    elif x > d[-1]:
        pr = n / (n + 1)
    else:
        j = int(np.searchsorted(d, x, side="right"))   # 1-based interval index
        if j >= n:
            pr = n / (n + 1)
        elif d[j] == d[j - 1]:
            pr = j / (n + 1)
        else:
            frac = (x - d[j - 1]) / (d[j] - d[j - 1])
            pr = (j + frac) / (n + 1)
    return truncate_significant(pr, significance) if significance else pr


def external_combined_sf(stressed_gcc, gcc_history) -> np.ndarray:
    """The external scale's scaling factor, from GCC growth.

    Externally rated exposures are driven by REGIONAL growth rather than the
    domestic variables, which is why they carry their own curve set.

    The factor is the probit of where the stressed growth rate falls in the
    historical distribution -- a percentile rank, not a z-score. That
    distinction matters: a percentile rank makes no assumption about the shape
    of the distribution, and the two give visibly different answers.

    A WARNING, because this is faithful to the reference and looks wrong:

        higher growth -> higher percentile -> higher SF -> HIGHER PD

    so on this scale a downturn LOWERS the provision. On the QDB history
    (sd 8.65) a 1.28-sigma downturn gives a 2% rating a PD of 0.00006 while the
    matching uptrend gives 0.29 -- a factor of several thousand, in the
    direction opposite to the internal scale, where a downturn raises PD as
    expected.

    The port reproduces the reference rather than correcting it: changing the
    sign here would make the Python disagree with the R and with every signed
    figure to date. It is flagged in ETL_STATUS.md as a question for Risk. The
    two portfolios affected are Investments and Banks and FIs.
    """
    h = np.asarray(gcc_history, dtype=float)
    h = h[np.isfinite(h)]
    out = []
    for v in np.atleast_1d(np.asarray(stressed_gcc, dtype=float)):
        if not np.isfinite(v):
            out.append(np.nan)
            continue
        out.append(norm.ppf(percentrank_exc(h, float(v))))
    return np.array(out)


def build_term_structure(ttc_table: pd.DataFrame, scenarios: pd.DataFrame,
                         mev_specs, mev_weights, forecasts, ttc_anchor: float,
                         n_forecasts: int = 5,
                         max_maturity: int = 50) -> pd.DataFrame:
    """The annual term structure: one marginal PD per rating, scenario and year.

    Every rating on a scale sees the same scenario shift; what differs is where
    that shift starts from, which is the rating's own TTC PD.
    """
    rows = []
    for sc in scenarios.itertuples():
        sf = combined_sf_for_scenario(forecasts, float(sc.severity_z),
                                      mev_specs, mev_weights, ttc_anchor)
        for r in ttc_table.itertuples():
            pit = pit_pd_term_structure(float(r.ttc_pd), sf, n_forecasts,
                                        max_maturity)
            cum = cumulative_pd(pit)
            marginal = np.diff(np.concatenate([[0.0], cum]))
            rows.append(pd.DataFrame({
                "rating": r.rating,
                "scenario": sc.scenario,
                "maturity": np.arange(1, max_maturity + 1),
                "marginal_pd": marginal,
            }))
    return pd.concat(rows, ignore_index=True)


def build_stpd_from_static(static, model_cfg, model_inputs, extract_date: str,
                           max_month: int = 600,
                           scenario_weights=None) -> pd.DataFrame:
    """Build StPD.csv end to end from a run's reference data and config.

    The internal and external scales are built SEPARATELY and each is repeated
    across the portfolios that use it. A PD curve is a property of the rating
    scale, not of the portfolio, and combining the two scales would silently
    mix grades that share a hierarchy number but mean different things.
    """
    mc = model_cfg
    comp = mc["models"]["internal_v4_production"]["mev_components"]
    variables = mc.get("variables", {})
    specs = [{
        "standard_deviation": c["standard_deviation"],
        "stress_unit_multiplier":
            variables.get(c["variable"], {}).get("stress_unit_multiplier", 1),
        "intercept": c["intercept"],
        "coefficient": c["coefficient"],
    } for c in comp]
    weights = [c["weight"] for c in comp]

    fc_block = model_inputs["mev_forecasts"]["forecasts"]
    forecasts = np.array([np.asarray(fc_block[y], dtype=float)
                          for y in sorted(fc_block, key=lambda k: int(k))])
    anchor = float(mc["ttc_anchor_pd"])
    # These live under `horizons`, not at the top level. Defaulting them
    # silently would change every curve beyond the forecast window.
    horizons = mc.get("horizons", {}) or {}
    n_forecasts = int(horizons.get("n_forecasts",
                                   mc.get("n_forecasts", forecasts.shape[0])))
    max_maturity = int(horizons.get("max_maturity", mc.get("max_maturity", 50)))

    scen = static["scenario_severity"]

    portfolios = static["portfolios"]
    pf_col = "portfolio_code" if "portfolio_code" in portfolios.columns \
        else portfolios.columns[0]
    rt_col = next((c for c in portfolios.columns if "rating" in c.lower()), None)

    # The external scale is driven by REGIONAL growth, not the domestic MEVs,
    # so it gets its own scaling factors. Using the internal chain for both
    # produced errors five times larger on the external portfolios.
    gcc_hist = gcc_weighted_history(static.get("gcc_real_gdp_growth"),
                                    static.get("gcc_gdp_current_prices"))
    ext_fc = model_inputs.get("external_gcc_forecast")
    if ext_fc is None:
        ext_fc = external_gcc_forecast(model_inputs, static, n_forecasts)

    # The two scales are weighted SEPARATELY. The internal weights come from
    # where the domestic forecast sits on the domestic history; the external
    # ones from the regional forecast on the regional history, one vector per
    # year. An explicit `scenario_weights` overrides both, which is what the
    # stress pages pass when they reweight a run by hand.
    if scenario_weights is None:
        internal_weights = resolve_internal_scenario_weights(model_inputs, static)
        external_weights = resolve_external_scenario_weights(
            model_inputs, static, gcc_hist, ext_fc)
    else:
        internal_weights = external_weights = scenario_weights

    frames = []
    for rating_type in (1, 2):
        ttc = static.ttc_for(rating_type)
        if len(ttc) == 0:
            continue
        if rt_col is not None:
            pfs = portfolios.loc[
                _matching_type(portfolios[rt_col], rating_type), pf_col].tolist()
        else:
            pfs = portfolios[pf_col].tolist()
        if not pfs:
            continue
        if rating_type == 2 and gcc_hist.size and ext_fc is not None:
            ts = build_term_structure_external(ttc, scen, np.asarray(ext_fc,
                                               dtype=float), gcc_hist,
                                               max_maturity)
        else:
            ts = build_term_structure(ttc, scen, specs, weights, forecasts,
                                      anchor, n_forecasts, max_maturity)
        scale_weights = external_weights if rating_type == 2 else internal_weights
        frames.append(build_stpd(ts, scale_weights, pfs,
                                 static.ratings_for(rating_type),
                                 extract_date, max_month))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def external_gcc_forecast(model_inputs, static, n_forecasts: int):
    """Forward regional growth, GDP-weighted across the GCC, year by year.

    The weights come from ``external_gcc_country_prices`` in the model inputs,
    alongside the growth block, and they are PER YEAR:

        weighted[y] = sum_c growth[c, y] * price[c, y] / sum_c price[c, y]

    Two things this must not do, both of which look harmless:

      * Weight by the static GDP table's latest year instead. That is a
        different set of numbers from the forecast block, and it applies one
        fixed weight vector to every year, so the projected shift in the
        region's composition is thrown away.
      * Divide by the total of all weights while treating a missing growth
        figure as zero. A country with no forecast then drags the regional
        rate toward zero rather than dropping out of the average.
    """
    block = model_inputs.get("external_gcc_country_growth")
    if not block:
        h = gcc_weighted_history(static.get("gcc_real_gdp_growth"),
                                 static.get("gcc_gdp_current_prices"))
        return None if h.size == 0 else np.full(n_forecasts, float(h[-1]))

    countries = list(block)
    years = max(len(np.atleast_1d(block[c])) for c in countries)
    growth = np.full((years, len(countries)), np.nan)
    for j, c in enumerate(countries):
        v = np.atleast_1d(np.asarray(block[c], dtype=float))
        growth[:v.size, j] = v

    price_block = model_inputs.get("external_gcc_country_prices")
    weights = np.full((years, len(countries)), np.nan)
    if price_block:
        for j, c in enumerate(countries):
            v = np.atleast_1d(np.asarray(price_block.get(c, []), dtype=float))
            weights[:v.size, j] = v
    if not np.isfinite(weights).any():
        # No per-year prices in the inputs: fall back to the static table's
        # most recent year, one weight vector for every forecast year.
        prices = static.get("gcc_gdp_current_prices")
        fallback = np.ones(len(countries))
        if prices is not None and len(prices):
            low = {c.lower(): c for c in prices.columns}
            ccol = low.get("country")
            vcol = low.get("value", low.get("gdp"))
            ycol = low.get("year")
            if ccol and vcol:
                pf = prices.copy()
                if ycol:
                    pf = pf[pf[ycol] == pf[ycol].max()]
                lut = dict(zip(pf[ccol].astype(str),
                               pd.to_numeric(pf[vcol], errors="coerce")))
                cand = np.array([lut.get(c, np.nan) for c in countries],
                                dtype=float)
                if np.isfinite(cand).any():
                    fallback = np.nan_to_num(cand, nan=0.0)
        if fallback.sum() == 0:
            fallback = np.ones(len(countries))
        weights = np.tile(fallback, (years, 1))

    out = np.full(years, np.nan)
    for i in range(years):
        ok = np.isfinite(growth[i]) & np.isfinite(weights[i])
        if not ok.any():
            continue
        w = weights[i][ok]
        total = w.sum()
        if total == 0:
            out[i] = float(np.mean(growth[i][ok]))
        else:
            out[i] = float(np.sum(growth[i][ok] * w) / total)

    out = out[np.isfinite(out)]
    return out[:n_forecasts] if out.size >= n_forecasts else out


def build_term_structure_external(ttc_table: pd.DataFrame,
                                  scenarios: pd.DataFrame, gcc_forecast,
                                  gcc_history, max_maturity: int = 50
                                  ) -> pd.DataFrame:
    """Annual term structure for the externally rated book.

    Each scenario shifts the regional growth forecast by its severity times the
    history's own standard deviation, and the scaling factor is the probit of
    where that lands in the historical distribution.
    """
    sd = float(np.std(np.asarray(gcc_history, dtype=float), ddof=1))
    rows = []
    for sc in scenarios.itertuples():
        stressed = np.asarray(gcc_forecast, dtype=float) + float(sc.severity_z) * sd
        sf = external_combined_sf(stressed, gcc_history)
        for r in ttc_table.itertuples():
            pit = pit_pd_term_structure_external(float(r.ttc_pd), sf, len(sf),
                                                 max_maturity)
            cum = cumulative_pd(pit)
            rows.append(pd.DataFrame({
                "rating": r.rating,
                "scenario": sc.scenario,
                "maturity": np.arange(1, max_maturity + 1),
                "marginal_pd": np.diff(np.concatenate([[0.0], cum])),
            }))
    return pd.concat(rows, ignore_index=True)


def _matching_type(column, rating_type: int):
    from .static_ref import _matches_type
    return _matches_type(column, rating_type)


def compute_internal_scenario_weights(historical_series, forecasts,
                                      scenarios: pd.DataFrame) -> dict:
    """Scenario weights from where each forecast sits on the historical CDF.

    Each scenario owns a band of the distribution. The band edges are the
    forecast shifted by each severity, read through the historical normal, and
    a scenario's weight is the probability mass between its edge and its
    neighbour's.

    Two details that are easy to lose:

      * The CENTRAL scenario takes the RESIDUAL, ``1 - sum(others)``, rather
        than its own band. That is what guarantees the weights sum to one
        exactly instead of to something near it.
      * Bands are assigned in severity order, then the weights are put back in
        the scenarios' original order. Skipping the reordering silently pairs
        each weight with the wrong scenario.

    Averaging across forecast years is a plain mean, so every year in the
    horizon counts equally.
    """
    h = np.asarray(historical_series, dtype=float)
    h = h[np.isfinite(h)]
    mu, sigma = h.mean(), h.std(ddof=1)

    z = pd.to_numeric(scenarios["severity_z"], errors="coerce").to_numpy()
    order = np.argsort(z)
    z_sorted = z[order]
    n = z_sorted.size
    central = int(np.argmin(np.abs(z_sorted)))

    per_year = []
    for f in np.atleast_1d(np.asarray(forecasts, dtype=float)):
        cdf = norm.cdf(f + sigma * z_sorted, loc=mu, scale=sigma)
        p = np.zeros(n)
        for i in range(n):
            if i == central:
                continue
            if i < central:
                lower = 0.0 if i == 0 else cdf[i - 1]
                p[i] = cdf[i] - lower
            else:
                upper = 1.0 if i == n - 1 else cdf[i + 1]
                p[i] = upper - cdf[i]
        p[central] = 1.0 - p.sum()
        per_year.append(p)

    weights_sorted = np.mean(np.vstack(per_year), axis=0)
    weights = np.zeros(n)
    weights[order] = weights_sorted
    return dict(zip(scenarios["scenario"], weights))


def compute_external_scenario_weights(weighted_gcc_year: float,
                                      gcc_history, scenarios: pd.DataFrame) -> dict:
    """Scenario weights for ONE forecast year, from the regional GDP distribution.

    Same band construction as the internal weights, but read against the GCC
    history rather than the domestic one, and evaluated at a single year's
    forecast. The central scenario takes the residual so the row sums to
    exactly one.
    """
    h = np.asarray(gcc_history, dtype=float)
    h = h[np.isfinite(h)]
    mu, sigma = h.mean(), h.std(ddof=1)

    z = pd.to_numeric(scenarios["severity_z"], errors="coerce").to_numpy()
    order = np.argsort(z)
    z_sorted = z[order]
    n = z_sorted.size
    central = int(np.argmin(np.abs(z_sorted)))

    cdf = norm.cdf(float(weighted_gcc_year) + z_sorted * sigma, loc=mu, scale=sigma)
    w = np.zeros(n)
    for i in range(n):
        if i == central:
            continue
        if i < central:
            w[i] = cdf[i] - (0.0 if i == 0 else cdf[i - 1])
        else:
            w[i] = (1.0 if i == n - 1 else cdf[i + 1]) - cdf[i]
    w[central] = 1.0 - w.sum()

    out = np.zeros(n)
    out[order] = w
    return dict(zip(scenarios["scenario"], out))


def compute_external_scenario_weights_per_year(weighted_gcc_forecasts,
                                               gcc_history,
                                               scenarios: pd.DataFrame) -> dict:
    """The external weights for every forecast year, plus the average row.

    Returns ``{"per_year": DataFrame[year x scenario], "average": Series}``.
    The average applies to maturities beyond the forecast horizon, which is 45
    of the 50 years in every curve -- so getting it from the modelled years
    rather than repeating year 5 matters.
    """
    names = list(scenarios["scenario"])
    rows = [compute_external_scenario_weights(f, gcc_history, scenarios)
            for f in np.atleast_1d(np.asarray(weighted_gcc_forecasts, dtype=float))]
    per_year = pd.DataFrame(rows, columns=names)
    per_year.index = [f"year_{i}" for i in range(1, len(per_year) + 1)]
    bad = (per_year.sum(axis=1) - 1.0).abs() > 0.01
    if bad.any():
        warnings.warn("external per-year weight rows do not sum to 1: "
                      f"{list(per_year.index[bad])}", RuntimeWarning, stacklevel=2)
    return {"per_year": per_year, "average": per_year.mean(axis=0)}


def resolve_internal_scenario_weights(model_inputs, static) -> dict:
    """The internal weights a run should use, honouring the configured mode.

    ``auto_non_oil_gdp_cdf`` computes them from the domestic history and the
    forecast; ``explicit`` takes the block as written. Falling back to an equal
    split when neither resolves would be the worst outcome available -- it
    looks like a weighting and is not one -- so this raises instead.
    """
    block = (model_inputs.get("internal_scenario_weights") or {})
    mode = block.get("mode", "explicit")
    explicit = block.get("explicit_weights")

    if mode == "explicit":
        if not explicit:
            raise ValueError("internal_scenario_weights.mode='explicit' but no "
                             "explicit_weights were supplied")
        return dict(explicit)

    if mode == "auto_non_oil_gdp_cdf":
        fcb = model_inputs["mev_forecasts"]["forecasts"]
        n = int(block.get("n_forecast_years", 2))
        gdp = [float(fcb[y][0]) for y in sorted(fcb, key=lambda k: int(k))][:n]
        hist = np.asarray(static["non_oil_gdp_history"]["value"], dtype=float)
        return compute_internal_scenario_weights(hist, gdp,
                                                 static["scenario_severity"])

    raise ValueError(f"unknown internal_scenario_weights.mode: {mode!r} "
                     "(expected 'explicit' or 'auto_non_oil_gdp_cdf')")


def resolve_external_scenario_weights(model_inputs, static, gcc_history,
                                      gcc_forecasts) -> dict:
    """The external weights, as ``{"per_year": ..., "average": ...}``.

    ``explicit`` repeats one vector across every year, which is what the R does
    when the block is written out by hand; ``auto_gcc_cdf`` derives a vector
    per year from the regional forecast.
    """
    block = (model_inputs.get("external_scenario_weights") or {})
    mode = block.get("mode", "explicit")
    names = list(static["scenario_severity"]["scenario"])
    n_years = len(np.atleast_1d(np.asarray(gcc_forecasts, dtype=float)))

    if mode == "explicit":
        explicit = block.get("explicit_weights")
        if not explicit:
            raise ValueError("external_scenario_weights.mode='explicit' but no "
                             "explicit_weights were supplied")
        w = pd.Series({k: float(v) for k, v in explicit.items()}).reindex(names)
        if w.isna().any():
            raise ValueError("external explicit_weights is missing entries for: "
                             f"{sorted(w.index[w.isna()])}")
        per_year = pd.DataFrame([w] * max(n_years, 1), columns=names)
        per_year.index = [f"year_{i}" for i in range(1, len(per_year) + 1)]
        return {"per_year": per_year, "average": w}

    if mode == "auto_gcc_cdf":
        return compute_external_scenario_weights_per_year(
            gcc_forecasts, gcc_history, static["scenario_severity"])

    raise ValueError(f"unknown external_scenario_weights.mode: {mode!r} "
                     "(expected 'explicit' or 'auto_gcc_cdf')")


GCC_HISTORY_FIRST_YEAR = 1982


def gcc_weighted_history(growth: pd.DataFrame,
                         prices: pd.DataFrame | None = None,
                         first_year: int = GCC_HISTORY_FIRST_YEAR,
                         min_countries: int = 2) -> np.ndarray:
    """A GDP-weighted regional growth series from the per-country tables.

    Weighting by each country's GDP is the point: an unweighted pooled series
    treats a small economy's swing as equal to Saudi Arabia's, which inflates
    the dispersion and therefore every scenario shift built from it.

    Two bounds carried over from the R, both of which move every external
    number and neither of which is visible in the output:

      * The series starts at 1982. The tables reach back to 1980, but those
        two years are not in the modelled window; including them takes the
        standard deviation from 4.49 to 4.39, and that SD is what every
        scenario shift is measured in.
      * A year needs at least two countries with both a growth and a GDP
        figure. One country is not a regional average, and weighting a lone
        survivor gives it a weight of 1.
    """
    def col(df, *names):
        low = {c.lower(): c for c in df.columns}
        for n in names:
            if n in low:
                return df[low[n]]
        return None

    yr = col(growth, "year")
    ctry = col(growth, "country")
    val = pd.to_numeric(col(growth, "value", "growth"), errors="coerce")
    if yr is None or val is None:
        return np.asarray([], dtype=float)
    g = pd.DataFrame({"year": yr, "country": ctry, "growth": val}).dropna()
    g["year"] = pd.to_numeric(g["year"], errors="coerce")
    g = g.dropna(subset=["year"])
    if first_year is not None:
        g = g[g["year"] >= first_year]

    if prices is None or ctry is None:
        return g.groupby("year")["growth"].mean().sort_index().to_numpy()

    pv = pd.to_numeric(col(prices, "value", "gdp"), errors="coerce")
    p = pd.DataFrame({"year": col(prices, "year"),
                      "country": col(prices, "country"),
                      "gdp": pv}).dropna()
    p["year"] = pd.to_numeric(p["year"], errors="coerce")
    p = p.dropna(subset=["year"])
    m = g.merge(p, on=["year", "country"], how="inner")
    if len(m) == 0:
        return g.groupby("year")["growth"].mean().sort_index().to_numpy()
    m["wx"] = m["growth"] * m["gdp"]
    agg = m.groupby("year").agg(wx=("wx", "sum"), w=("gdp", "sum"),
                                n=("gdp", "size"))
    agg = agg[agg["n"] >= min_countries]
    return (agg["wx"] / agg["w"]).sort_index().to_numpy()
