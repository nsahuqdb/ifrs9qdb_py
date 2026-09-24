"""Analytics over a completed run: pure functions, no UI."""
from .profile import (  # noqa: F401
    classify_stage, collateral_bands, concentration, customer_view,
    data_quality, dpd_profile, exposure_bands, hhi, hhi_band,
    hhi_equivalent_n, lgd_distribution, lorenz_curve, maturity_profile,
    normalise, pd_by_rating, pd_distribution, run_profile,
    stage2_triggers, staging_distribution, top_contributors, vintage_profile,
)
from .walk import (  # noqa: F401
    ecl_walk, ecl_walk_detail, flow_profile, movement_by, stage_transitions,
)
