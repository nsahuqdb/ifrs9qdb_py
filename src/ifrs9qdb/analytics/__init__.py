"""Analytics over a completed run: pure functions, no UI."""
from .attribution import (  # noqa: F401
    coverage_bridge, ecl_factor_attribution, factor_attribution_diagnosis,
    factor_attribution_exact,
)
from .bridge import (  # noqa: F401
    BRIDGE_COMPONENTS, BRIDGE_LEVELS, bridge_by, bridge_members, bridge_view,
    config_changes, ecl_bridge, load_bridge_run,
)
from .model_view import (  # noqa: F401
    config_used, mev_forecast_table, mev_weights_table,
    read_scenario_stpd, scenario_severity, scenario_weights,
)
from .profile import (  # noqa: F401
    classify_stage, collateral_bands, concentration, customer_lookup,
    customer_view,
    data_quality, data_quality_detail, dpd_profile, exposure_bands, hhi,
    hhi_band, hhi_equivalent_n, lgd_distribution, lorenz_curve,
    maturity_profile,
    normalise, pd_by_rating, pd_distribution, run_profile,
    stage2_triggers, staging_distribution, top_contributors, vintage_profile,
)
from .risk import (  # noqa: F401
    collateral_analysis, ead_runoff, lgd_floor_stats, lgd_vs_collateral,
    pd_profile, pd_term_structure, report_total, segment_matrix,
)
from .scenarios import (  # noqa: F401
    scenario_comparison, scenario_ecl_from_outputs, scenario_files,
    scenario_reweight, scenario_sensitivity, scenario_stage_split,
)
from .staging import (  # noqa: F401
    customer_stage_migration, dpd_by_stage, migration_summary,
    rating_migration, stage2_trigger_overlap, stage3_drivers, stage_movers,
    staging_consistency,
)
from .walk import (  # noqa: F401
    ecl_walk, ecl_walk_detail, flow_profile, movement_by, stage_transitions,
)
