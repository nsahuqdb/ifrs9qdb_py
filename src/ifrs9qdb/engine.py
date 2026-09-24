"""
Core IFRS 9 ECL calculation.

A direct port of the R engine in `ifrs9qdb`, reproducing it figure for figure.
The four EY reconciliation contracts and twelve run-324 contracts are kept as
golden tests, so any divergence from the R engine -- or from LIC -- fails the
suite rather than reaching a report.

The formula, and every convention in it, matters:

    ECL = SUM_t  EAD(t) * LGD * [cumPD(t+1) - cumPD(t)] / (1 + EIR) ** (t/12)

  * cumPD is zero-prepended, so the first marginal PD is cumPD(1) - 0.
  * discounting is monthly and starts at t = 0, i.e. the first month is
    undiscounted.
  * the horizon is the length of the EAD curve, capped at 12 months for
    Stage 1.
  * Stage 3 is governed by config, not by this formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "compute_lgd",
    "sum_marginal_ecl",
    "fallback_ead_curve",
    "resolve_ead_shape",
    "EAD_FALLBACK_RULES",
    "resolve_ead_curve",
    "months_to_maturity",
    "EclConfig",
    "compute_ecl_one",
]


# --------------------------------------------------------------------- LGD --
def compute_lgd(
    on_balance: float,
    collateral_net: float,
    base: float = 0.45,
    unsecured_floor: float = 0.5,
    zero_exposure_lgd: float = 1.0,
) -> float:
    """Loss given default.

    ``base * max(unsecured_floor, (exposure - collateral) / exposure)``.

    The floor means at least half of any exposure is treated as unsecured
    however much collateral is held, so the LGD cannot fall below
    ``base * unsecured_floor`` -- 0.225 on the QDB calibration. A consequence
    worth knowing: collateral beyond roughly 50% coverage does not reduce the
    provision at all.
    """
    if on_balance is None or not np.isfinite(on_balance) or on_balance <= 0:
        return zero_exposure_lgd
    cn = 0.0 if collateral_net is None or not np.isfinite(collateral_net) else float(collateral_net)
    return base * max(unsecured_floor, (on_balance - cn) / on_balance)


# --------------------------------------------------------------------- EAD --
def months_to_maturity(maturity, extract, min_months: int = 3) -> int:
    """Whole months from the extract date to maturity, floored at ``min_months``.

    Counted on calendar months, matching the R engine. A matured facility still
    gets the floor rather than zero, because a run-off still carries loss.
    """
    if maturity is None or extract is None:
        return min_months
    # A NaT reads as a date object but has no usable year, so the attribute
    # check alone is not enough -- a missing maturity has to fall back to the
    # floor rather than raise partway through a book.
    try:
        if maturity != maturity or extract != extract:      # NaT
            return min_months
        m = (maturity.year - extract.year) * 12 + (maturity.month - extract.month)
        if m != m:
            return min_months
    except (AttributeError, TypeError, ValueError):
        return min_months
    return max(min_months, int(m))


MIN_HORIZON_MONTHS = 3

# Which shape a contract with no supplied schedule amortises on, by
# (portfolio, payment type). Empirical, and matched against the production LIC
# run rather than inferred from the payment type alone: an unrecognised
# combination is a BULLET, which is the conservative reading and the one LIC
# takes.
EAD_FALLBACK_DEFAULT = "bullet"
EAD_FALLBACK_RULES: tuple[tuple[str | None, str, str], ...] = (
    ("Business Finance", "4", "linear"),
    ("Business Finance", "3", "bullet"),
    ("Al Dhameen", "4", "bullet"),
    ("Tasdeer", "3", "bullet"),
    ("Off BS", "3", "bullet"),
    # No portfolio: the payment type decides. Type 4 amortises, which is the
    # point EY's prose answer got wrong and their own workbook settles -- a
    # flat curve gives roughly 1.8x the LIC figure on their contract 11.
    (None, "4", "linear"),
)


def resolve_ead_shape(payment_type, portfolio=None, rules=None,
                      default: str = EAD_FALLBACK_DEFAULT) -> str:
    """The amortisation shape for a contract with no supplied curve.

    A rule naming a portfolio wins over one that does not, and anything
    unmatched falls to ``default``. Resolving on the payment type alone was
    wrong for Al Dhameen, whose type-4 facilities LIC prices as bullets.
    """
    rules = EAD_FALLBACK_RULES if rules is None else rules
    pt = str(payment_type).strip() if payment_type is not None else ""
    if pt.endswith(".0"):
        pt = pt[:-2]
    pf = str(portfolio).strip() if portfolio is not None else ""
    for rule_pf, rule_pt, shape in rules:
        if rule_pf is not None and rule_pf == pf and rule_pt == pt:
            return shape
    for rule_pf, rule_pt, shape in rules:
        if rule_pf is None and rule_pt == pt:
            return shape
    return default


def fallback_ead_curve(
    on_balance: float,
    months_remaining: int,
    payment_type: int | str | None,
    payment_frequency: int | None = 1,
    deferral: int | None = 0,
    horizon: int | None = None,
    portfolio: str | None = None,
    min_horizon_months: int = MIN_HORIZON_MONTHS,
) -> np.ndarray:
    """Parametric EAD curve for a contract with no supplied schedule.

    The shape comes from ``resolve_ead_shape``. A linear contract amortises
    over the periods remaining after any deferral, stepping on the payment
    frequency; a bullet stays at the balance to maturity.

    One rule overrides the shape. A facility whose maturity is at or before
    the extract date has had its remaining term FLOORED to
    ``min_horizon_months``, and it has no remaining amortisation schedule to
    run: LIC prices it as a bullet over the floored horizon. Amortising it
    instead understates the provision by a third on a three-month floor, which
    is where 50 of the 54 remaining differences against the R report came from.

    About 24% of a typical QDB book has no supplied curve, and that includes
    every Off BS, Al Dhameen and Tasdeer facility, so this path is not an edge
    case.
    """
    N = max(1, int(months_remaining))
    H = int(horizon) if horizon is not None else N
    H = max(1, min(H, 600))

    shape = resolve_ead_shape(payment_type, portfolio)
    if N <= int(min_horizon_months):
        shape = "bullet"
    if shape == "bullet":
        return np.full(H, float(on_balance), dtype=float)

    f = max(1, int(payment_frequency or 1))
    d = max(0, min(int(deferral or 0), N - 1))
    amortising = max(1, N - d)
    n_steps = max(1, amortising // f)
    out = np.empty(H, dtype=float)
    for t in range(H):
        elapsed = max(0, t - d)
        step = min(elapsed // f, n_steps)
        out[t] = float(on_balance) * max(0.0, 1.0 - step / n_steps)
    return out


def resolve_ead_curve(
    contract_id: str,
    schedule_curves: dict[str, Sequence[float]] | None,
    stage: int,
    on_balance: float,
    months_remaining: int,
    payment_type=None,
    payment_frequency: int | None = 1,
    deferral: int | None = 0,
    portfolio: str | None = None,
) -> np.ndarray | None:
    """The EAD curve the engine prices against.

    The supplied monthly curve when there is one, otherwise the parametric
    fallback. Stage 1 is capped at twelve months either way.
    """
    curve = None
    if schedule_curves:
        got = schedule_curves.get(str(contract_id))
        if got is not None and len(got) > 0:
            curve = np.asarray(got, dtype=float)
    if curve is None:
        if on_balance is None or not np.isfinite(on_balance):
            return None
        curve = fallback_ead_curve(
            on_balance, months_remaining, payment_type, payment_frequency,
            deferral, portfolio=portfolio
        )
    if stage == 1:
        curve = curve[: min(12, len(curve))]
    return curve


# --------------------------------------------------------------------- ECL --
def sum_marginal_ecl(
    ead: Sequence[float],
    lgd: float,
    cum_pd: Sequence[float],
    eir: float,
    horizon: int | None = None,
) -> float:
    """Discounted sum of monthly marginal losses.

    ``cum_pd`` must be zero-prepended: ``cum_pd[0] == 0`` and ``cum_pd[m]`` is
    the cumulative PD at month ``m``. Returns NaN when either curve is missing,
    which is how a contract with no resolvable PD bucket reports itself.
    """
    if ead is None or cum_pd is None:
        return float("nan")
    ead = np.asarray(ead, dtype=float)
    cum = np.asarray(cum_pd, dtype=float)
    if ead.size == 0 or cum.size < 2:
        return float("nan")
    H = int(horizon) if horizon is not None else ead.size
    H = min(H, ead.size, cum.size - 1)
    if H <= 0:
        return float("nan")
    t = np.arange(H)
    marginal = cum[1 : H + 1] - cum[:H]
    disc = (1.0 + (eir if eir is not None and np.isfinite(eir) else 0.0)) ** (t / 12.0)
    return float(np.sum(ead[:H] * lgd * marginal / disc))


# ------------------------------------------------------------------ config --
@dataclass
class EclConfig:
    """The settings that change a reported number.

    ``stage3_method`` -- ``full_outstanding`` books Stage 3 at the on-balance
    amount, which is the QDB basis and deliberately differs from LIC; ``zero``
    reproduces LIC, which reports nothing for Stage 3.

    ``cap_ecl_at_exposure`` -- caps the provision at the balance, as LIC does.
    A supplied EAD curve is expected exposure at default and can exceed today's
    outstanding where a facility has undrawn commitments, so an uncapped loss
    can come out larger than the balance.
    """

    stage3_method: str = "full_outstanding"
    cap_ecl_at_exposure: bool = True
    lgd_base: float = 0.45
    lgd_unsecured_floor: float = 0.5
    zero_exposure_lgd: float = 1.0
    max_month: int = 600

    def __post_init__(self):
        if self.stage3_method not in ("full_outstanding", "zero"):
            self.stage3_method = "full_outstanding"


def compute_ecl_one(
    *,
    contract_id: str,
    stage: int,
    on_balance: float,
    collateral_net: float,
    cum_pd: Sequence[float] | None,
    eir: float,
    months_remaining: int,
    schedule_curves: dict[str, Sequence[float]] | None = None,
    payment_type=None,
    payment_frequency: int | None = 1,
    deferral: int | None = 0,
    cfg: EclConfig | None = None,
) -> dict:
    """Price one contract. Returns ecl, lgd, ead curve length and coverage."""
    cfg = cfg or EclConfig()
    lgd = compute_lgd(
        on_balance,
        collateral_net,
        base=cfg.lgd_base,
        unsecured_floor=cfg.lgd_unsecured_floor,
        zero_exposure_lgd=cfg.zero_exposure_lgd,
    )

    if stage == 3:
        if cfg.stage3_method == "zero":
            ecl = 0.0
        else:
            ecl = 0.0 if (on_balance is None or not np.isfinite(on_balance) or on_balance <= 0) \
                else float(on_balance)
        cov = (ecl / on_balance) if (on_balance and on_balance > 0) else float("nan")
        return {"contract": contract_id, "ecl": ecl, "lgd": lgd, "horizon": 0, "coverage": cov}

    curve = resolve_ead_curve(
        contract_id, schedule_curves, stage, on_balance, months_remaining,
        payment_type, payment_frequency, deferral,
    )
    if curve is None or cum_pd is None:
        return {"contract": contract_id, "ecl": float("nan"), "lgd": lgd,
                "horizon": 0, "coverage": float("nan")}

    ecl = sum_marginal_ecl(curve, lgd, cum_pd, eir, horizon=len(curve))
    if np.isfinite(ecl) and cfg.cap_ecl_at_exposure and on_balance and on_balance > 0:
        ecl = min(ecl, float(on_balance))
    cov = (ecl / on_balance) if (on_balance and on_balance > 0 and np.isfinite(ecl)) else float("nan")
    return {"contract": contract_id, "ecl": ecl, "lgd": lgd,
            "horizon": len(curve), "coverage": cov}
