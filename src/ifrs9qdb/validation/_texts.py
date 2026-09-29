"""Why each TRANSFORM and DERIVED check exists, and what to do when it fails.

R's validators carry only a description. The port adds the reasoning a
reviewer needs to act on a finding, keyed by id so the checks themselves
stay a literal port of R's.
"""

STAGE_TEXTS: dict[str, tuple[str, str]] = {
    'DERIVED_LTPO_contracts_subset_of_trans': (
        'A curve for a contract that is not on the book is priced by LIC '
        'against nothing.',
        'Check the schedule filter against the account list.',
    ),
    'DERIVED_LTPO_ead_nonincreasing': (
        'For a term loan a rising curve is wrong. For a revolving or off-'
        'balance facility it is the schedule carrying committed but undrawn '
        'amounts, which is why the engine caps ECL at exposure. Informational, '
        "so that nobody later 'fixes' a rising curve.",
        'No action for revolving products.',
    ),
    'DERIVED_LTPO_ead_nonneg': (
        'A negative exposure at default produces a negative provision.',
        'Check the repayment schedule for negative balances.',
    ),
    'DERIVED_LTPO_extract_date_unique': (
        'Two dates in one file means curves from two runs were mixed.',
        'Re-run the ETL from a single input set.',
    ),
    'DERIVED_LTPO_month_starts_at_zero': (
        "Month 0 is today's outstanding. Without it LIC has no starting "
        'exposure and prices the contract from the first scheduled payment.',
        'Check the month-0 row is written from the account master.',
    ),
    'DERIVED_LTPO_months_contiguous': (
        'A gap makes LIC interpolate across it; a duplicate double-counts that '
        "month's loss.",
        'Check the month bound is EXCLUSIVE: months 0 .. end_month - 1.',
    ),
    'DERIVED_LTPO_schema': (
        'LIC reads this file by column position as well as by name.',
        'Check LPO_COLUMNS in etl/lifetime.py.',
    ),
    'DERIVED_LTPO_total_month0_ead_reconciles': (
        "Month 0 is today's outstanding from the account master, not the "
        "schedule's first figure. The two differ whenever a payment falls in "
        'the current month, and taking the schedule value understates it.',
        'Compare a contract with a payment this month.',
    ),
    'DERIVED_MEV_weights_nonneg': (
        "A negative weight inverts that variable's contribution.",
        'Check model.yml.',
    ),
    'DERIVED_MEV_weights_sum_to_one': (
        'Only Non-Oil GDP carries weight in the production model; real estate '
        'and domestic credit are weighted 0.0. That is the model, not a fault, '
        'but the three must still sum to one.',
        'Check the mev_components block in model.yml.',
    ),
    'DERIVED_SCEN_external_average_sum_to_one': (
        'The average row applies to 45 of the 50 years in every external curve.',
        'Check it is the column-wise mean of the per-year rows.',
    ),
    'DERIVED_SCEN_external_per_year_sum_to_one': (
        'The external scale uses a DIFFERENT weight vector for each year, '
        'because the regional forecast moves. Each row must still be a '
        'probability distribution.',
        'Check compute_external_scenario_weights_per_year().',
    ),
    'DERIVED_SCEN_external_weights_nonneg': (
        'A negative probability is not a probability.',
        'Check the band construction.',
    ),
    'DERIVED_SCEN_internal_weights_nonneg': (
        'A negative probability is not a probability.',
        'Check the band construction; the central scenario takes the residual.',
    ),
    'DERIVED_SCEN_internal_weights_sum_to_one': (
        'An equal split sums to one too, so this alone will not catch it - but '
        'a weighting that does NOT sum to one is always wrong.',
        'Check resolve_internal_scenario_weights().',
    ),
    'DERIVED_STPD_bucket_set_complete': (
        'A TTC PD of ZERO must still get a bucket. Filtering on > 0 instead of '
        '>= 0 loses the three highest external grades and 3,600 rows.',
        'Check the TTC filter.',
    ),
    'DERIVED_STPD_dim2_all_na': (
        'The output schema leaves this column empty; populating it changes how '
        'LIC keys the curve.',
        'Check the StPD writer.',
    ),
    'DERIVED_STPD_external_portfolios_identical': (
        'As above, for the external scale.',
        'Check the term structure is built once per scale.',
    ),
    'DERIVED_STPD_extract_date_unique': (
        'Two dates means curves from two runs were mixed.',
        'Re-run the ETL from a single input set.',
    ),
    'DERIVED_STPD_internal_portfolios_identical': (
        'A curve belongs to the rating SCALE, not the portfolio. A difference '
        'means the two scales were mixed.',
        'Check the term structure is built once per scale.',
    ),
    'DERIVED_STPD_month_set_complete': (
        'A short curve makes LIC extrapolate beyond its end.',
        'Check max_month in the monthly conversion.',
    ),
    'DERIVED_STPD_pd_increases_with_hierarchy': (
        'If the ordering inverts, the scale has been applied upside down - '
        'which is exactly what happens if the internal probit shift is used on '
        'the external book, where the factor is SUBTRACTED.',
        'Check which formula each rating type uses.',
    ),
    'DERIVED_STPD_pd_monotone_non_decreasing': (
        'A cumulative default probability cannot decrease.',
        'Check the cumulative step.',
    ),
    'DERIVED_STPD_pd_nonneg_finite': (
        'A NaN PD silently prices a whole bucket to zero.',
        'Check the probit chain for a TTC PD of 0 or 1.',
    ),
    'DERIVED_STPD_pd_within_workbook_bound': (
        'The monthly conversion is a running SUM rather than the survival '
        'formula, so it can pass 1 and is capped. A value above the cap means '
        'the cap did not apply.',
        'Check the cap in convert_to_monthly_stpd().',
    ),
    'DERIVED_STPD_portfolio_set_complete': (
        'A missing portfolio means every contract in it prices to zero.',
        'Check portfolios.csv and the rating_type column.',
    ),
    'DERIVED_STPD_row_count': (
        '6 portfolios x 21 buckets x 600 months. A short file means a bucket or'
        ' a portfolio was dropped.',
        'Check the TTC filter uses >= 0 rather than > 0.',
    ),
    'DERIVED_STPD_schema': (
        'LIC reads the file by position as well as by name.',
        'Check the StPD writer.',
    ),
    'DERIVED_STPD_zero_ttc_zero_curve': (
        'The engine short-circuits a zero TTC to a zero curve, but the rating '
        'still needs a bucket in the output.',
        'Check the zero short-circuit in the term structure.',
    ),
    'TRANS_INVPV_account_count_matches_trans': (
        'The file is keyed on the ACCOUNT, not the counterparty - 73 rows '
        'against 62 counterparties is correct and a shortfall is not.',
        'Check the surrogate id assignment.',
    ),
    'TRANS_INVPV_stage_in_set': (
        'Anything else is not a stage LIC will accept.',
        'Check the investment staging rule.',
    ),
    'TRANS_INVPV_top_tier_implies_stage1': (
        'An Aaa-to-Aa3 holding staged above 1 is almost always a rating that '
        'failed to resolve rather than a genuine deterioration.',
        'Check the counterparty rating join.',
    ),
    'TRANS_INV_account_id_unique': (
        'Investment ids are surrogate sequential numbers; a duplicate means the'
        ' sequence was generated twice.',
        'Check the surrogate id assignment.',
    ),
    'TRANS_INV_exposure_nonneg': (
        'A missing holding prices to zero.',
        'Trace back to ONBALANCE in the investment extract.',
    ),
    'TRANS_INV_hierarchy_in_range': (
        'Outside 1..21 there is no curve. Note the two scales reuse the same '
        'numbers, so the lookup must be on (rating_type, rating).',
        'Check master_rating_scale.csv.',
    ),
    'TRANS_INV_rating_in_external_scale': (
        'The external book uses the agency scale, not the QDB ladder.',
        'Add the grade to master_rating_scale.csv.',
    ),
    'TRANS_INV_rating_populated': (
        'Without a grade the holding has no bucket and prices to zero.',
        'Check the counterparty join.',
    ),
    'TRANS_INV_total_exposure_positive': (
        'A zero total means the investment extract did not join.',
        'Check AccountMasterInvestments was read.',
    ),
    'TRANS_LENDPV_clean_low_dpd_stage1': (
        'A customer with no trigger at all should not be provisioned at '
        'lifetime. Rows here usually mean a flag is being read from the wrong '
        'column.',
        'Check the watchlist and restructuring joins.',
    ),
    'TRANS_LENDPV_customer_count_matches_trans': (
        'The customer view is keyed on CustomerMaster. A contract whose '
        'customer is not there is missing from the view, so it gets no '
        'customer-level stage.',
        "Add the missing customers to CustomerMaster, or correct the contracts'"
        ' customer ids at source.',
    ),
    'TRANS_LENDPV_customer_id_unique': (
        'The view is one row per customer. A duplicate means the grouping '
        'failed and the staging flags are ambiguous.',
        'The view is keyed on CustomerMaster: remove the duplicate CUSTOMERID '
        'rows there.',
    ),
    'TRANS_LENDPV_dpd_gt_90_implies_stage3': (
        'This is the first rule in the staging order, and it is absolute: a '
        'defaulted customer cannot be pulled back to Stage 2 by also being '
        'watchlisted.',
        'Check the order of the conditions in apply_staging_rule().',
    ),
    'TRANS_LENDPV_exposure_reconciles': (
        'The two views describe the same book. A difference means one of them '
        'lost rows.',
        'Look for contracts whose customer is not in CustomerMaster '
        '(TRANS_LENDPV_customer_count_matches_trans).',
    ),
    'TRANS_LENDPV_restructured_implies_stage_2_or_3': (
        'Restructuring is a significant-increase trigger by policy.',
        'Check IsLocal1 in the staging extract is being read as restructuring.',
    ),
    'TRANS_LENDPV_stage_in_set': (
        'Anything else is not a stage LIC will accept.',
        'Check apply_staging_rule().',
    ),
    'TRANS_LENDPV_watchlist_implies_stage_2_or_3': (
        'Watchlist is a significant-increase trigger by policy.',
        'Check the watchlist flag reaches the staging rule.',
    ),
    'TRANS_LEND_contract_id_unique': (
        'A duplicate after the id transformation means two source contracts '
        'collapsed onto one id, which double-counts exposure.',
        'Check apply_id_substitutions against the source ids.',
    ),
    'TRANS_LEND_customer_id_populated': (
        'The rating and the staging both come from the customer. A contract '
        'with no customer gets neither.',
        'Check the customer join in transform_lending().',
    ),
    'TRANS_LEND_dpd_nonneg': (
        'Negative days past due is a data-entry artefact and distorts staging.',
        'Correct at source, or suppress with a reason if legacy.',
    ),
    'TRANS_LEND_exposure_nonneg': (
        'A missing exposure prices to zero; a negative one gives a negative '
        'provision.',
        'Trace the contract back to ONBALANCE in the extract.',
    ),
    'TRANS_LEND_hierarchy_in_range': (
        'The hierarchy is the PD bucket key. Outside 1..21 there is no curve.',
        'Check master_rating_scale.csv for a bad hierarchy value.',
    ),
    'TRANS_LEND_pass6_overrides_filled': (
        "Pass 6 copies each customer's final rating and default flag, after any"
        ' override, back onto its contracts. A contract whose customer is not '
        'in CustomerMaster gets neither, so its rating and default status are '
        'undefined.',
        "Add the customer to CustomerMaster, or correct the contract's customer"
        ' id at source.',
    ),
    'TRANS_LEND_rating_in_internal_scale': (
        'A grade off the scale resolves to no bucket.',
        'Add the grade to master_rating_scale.csv or correct it at source.',
    ),
    'TRANS_LEND_rating_populated': (
        'The lending rating is a CUSTOMER attribute and is not on the account '
        'rows. If the join fails every rating is blank, no bucket resolves, and'
        ' the run prices almost nothing while completing normally.',
        'Check that CustomerMaster was read and the ids join.',
    ),
    'TRANS_LEND_total_exposure_positive': (
        'A zero total is the signature of a failed join, not of an empty book.',
        'Check the account extract and the customer join.',
    ),
}
