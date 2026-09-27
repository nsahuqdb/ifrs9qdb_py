"""The Python ETL against the R package, from the RAW extracts.

The engine was proven against R long ago, but only from R's OUTPUT files. The
ETL -- raw extract to those files -- was never compared, and when it finally
was, on the 2026-06-09 extract, it disagreed on seven things. Four of them
changed the provision: EIR left in percent, MATURITYDAT never read, the V4
rating chain missing (949 facilities priced at zero), and the S&P grades
dropped from Ratings.csv.

The synthetic tests below pin each fix and run everywhere. The last class runs
the actual R package when it is installed and compares every output file and
the final report, contract by contract.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.etl.lending import (build_account_master, transform_investments,
                                  transform_lending)
from ifrs9qdb.etl.static_ref import load_static_reference
from ifrs9qdb.etl.transform import build_origination_rows, r_format_numeric

from conftest import src_inputs


@pytest.fixture(scope="module")
def static():
    return load_static_reference()


def accounts(**cols):
    n = len(next(iter(cols.values())))
    base = {"CONTRACTID": [f"{500000 + i}" for i in range(n)],
            "CUSTOMERID": [f"{1000 + i}" for i in range(n)],
            "ACCOUNTTYPE": ["Long Term"] * n, "ONBALANCE": [1000.0] * n,
            "EIR": [4.0] * n, "PAYMENTFREQUENCY": [1] * n,
            "PASTDUEDAYS": [0] * n}
    base.update(cols)
    return pd.DataFrame(base)


def customers(ids, ratings):
    return pd.DataFrame({"CUSTOMERID": ids, "RATING": ratings})


class TestTheRatingChain:
    """V4 Transformation!Z, as R/transform_lending.R derives it."""

    def test_an_unrated_customer_takes_the_segment_fallback(self, static):
        a = accounts(CUSTOMERID=["1000"])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              customers=customers(["1000"], ["Unrated"]),
                              static=static)
        assert v["rating_worst"].iloc[0] == "QDB 5", (
            "'Unrated' is not a grade. Left as-is it has no PD bucket and the "
            "facility is priced at zero, which is what 949 June facilities were.")

    def test_an_unrated_al_dhameen_customer_takes_its_own_fallback(self, static):
        a = accounts(CUSTOMERID=["1000"], ACCOUNTTYPE=["Al Dhameen"])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              customers=customers(["1000"], ["Unrated"]),
                              static=static)
        assert v["rating_worst"].iloc[0] == "QDB 6"

    def test_a_graded_customer_keeps_its_grade(self, static):
        a = accounts(CUSTOMERID=["1000"])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              customers=customers(["1000"], ["QDB 6-"]),
                              static=static)
        assert v["rating_worst"].iloc[0] == "QDB 6-", (
            "QDB 6- was missing from the old hard-coded ladder")

    def test_the_worst_grade_is_taken_on_the_hierarchy(self, static):
        a = accounts(CUSTOMERID=["1000", "1000"])
        cm = customers(["1000"], ["QDB 3"])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              customers=cm, static=static)
        assert set(v["rating_worst"]) == {"QDB 3"}

    def test_an_agriculture_customer_is_rated_collectively_by_dpd(self, static):
        ind = pd.DataFrame({"CUSTOMERID": ["1000", "1001"],
                            "INDUST": [113, 113],
                            "DESCRIPTION": ["0113 Growing of vegetables"] * 2})
        a = accounts(CUSTOMERID=["1000", "1001"], PASTDUEDAYS=[0, 45])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              customers=customers(["1000", "1001"],
                                                  ["Unrated", "Unrated"]),
                              industry=ind, static=static)
        rules = static["collective_assessment_rules"]
        ag = rules[rules["sector"] == "Agriculture Sector"].sort_values("dpd_threshold")
        at0 = ag[ag["dpd_threshold"] <= 0]["rating"].iloc[-1]
        at45 = ag[ag["dpd_threshold"] <= 45]["rating"].iloc[-1]
        assert list(v["rating_worst"]) == [at0, at45]


class TestMaturity:
    def test_the_truncated_alias_is_read(self, static):
        a = accounts(MATURITYDAT=[pd.Timestamp("2031-06-30")])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              static=static, reporting_date="2026-06-09")
        assert v["maturity_date"].iloc[0] == pd.Timestamp("2031-06-30"), (
            "the SQL export truncates MATURITYDATE to MATURITYDAT; missing it "
            "left every maturity blank")

    def test_a_passed_maturity_is_extended_by_a_year(self, static):
        a = accounts(MATURITYDAT=[pd.Timestamp("2026-01-31")])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              static=static, reporting_date="2026-06-09")
        assert v["maturity_date"].iloc[0] == pd.Timestamp("2027-06-09")

    def test_maturing_on_the_reporting_date_is_not_extended(self, static):
        """Strictly less than, as R and the V4 formula have it -- although
        staging_thresholds.csv describes the rule as <=."""
        a = accounts(MATURITYDAT=[pd.Timestamp("2026-06-09")])
        v = transform_lending(a, static["product_portfolio_mapping"],
                              static=static, reporting_date="2026-06-09")
        assert v["maturity_date"].iloc[0] == pd.Timestamp("2026-06-09")


class TestFrequency:
    def test_a_zero_frequency_is_written_blank_and_is_a_bullet(self, static):
        a = accounts(PAYMENTFREQUENCY=[0])
        v = transform_lending(a, static["product_portfolio_mapping"], static=static)
        am = build_account_master(v, "6/9/2026")
        assert am["PaymentFrequency"].iloc[0] in ("", None) or pd.isna(
            am["PaymentFrequency"].iloc[0])
        assert int(am["PaymentTypeId"].iloc[0]) == 3


class TestInvestments:
    def test_ids_are_the_extracts_and_names(self, static):
        a = pd.DataFrame({"CONTRACTID": ["1142", "1106"],
                          "CUSTOMERID": ["DUKHAN BANK", "DUKHAN BANK"],
                          "ACCOUNTTYPE": ["x", "x"], "RATING": ["Aa2", "Aa2"],
                          "ONBALANCE": [1.0, 2.0]})
        v = transform_investments(a, static=static)
        assert list(v["contract_id"]) == ["1142", "1106"]
        assert list(v["customer_id"]) == ["DUKHAN BANK", "DUKHAN BANK"]

    def test_an_sp_grade_is_kept_and_an_unknown_one_falls_back(self, static):
        a = pd.DataFrame({"CONTRACTID": ["1", "2"], "CUSTOMERID": ["A", "B"],
                          "ACCOUNTTYPE": ["x", "x"], "RATING": ["BBB+", "Unrated"],
                          "ONBALANCE": [1.0, 1.0]})
        v = transform_investments(a, static=static)
        assert list(v["rating_worst"]) == ["BBB+", "Baa3"]


class TestReferenceFiles:
    def test_every_grade_is_written(self, static):
        from ifrs9qdb.etl.reference import build_reference_files
        r = build_reference_files(static, "6/9/2026")["Ratings.csv"]
        assert len(r) == len(static["master_rating_scale"]) == 62
        assert {"BBB+", "AAA", "CC"} <= set(r["Rating"]), (
            "dropping the S&P grades left an investment rated BBB+ with no PD "
            "bucket, priced at zero")

    def test_origination_rows_use_account_master_ids_and_stay_blank(self):
        o = build_origination_rows("6/9/2026", ["1123000471", "504055"])
        assert list(o["ContractId"]) == ["1123000471", "504055"]
        assert (o.drop(columns=["EXTRACTDATE", "ContractId"]) == "").all().all()


class TestRNumberFormatting:
    def test_a_column_takes_common_decimals_at_seven_figures(self):
        got = list(r_format_numeric([7852.9968, 160012.5, 11570350.0]))
        assert got == ["7852.997", "160012.500", "11570350.000"]

    def test_fx_rates(self):
        assert list(r_format_numeric([1.0, 3.645])) == ["1.000", "3.645"]

    def test_blank_stays_blank(self):
        assert list(r_format_numeric([np.nan, 2.5])) == ["", "2.5"]


# ------------------------------------------------------------ against R ------
def _r_package_dir():
    if shutil.which("Rscript") is None:
        return None
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(system.file(package="ifrs9qdb"))'],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"})
    except Exception:
        return None
    p = out.stdout.strip()
    return Path(p) if p and Path(p).is_dir() else None


R_PKG = _r_package_dir()
needs_r = pytest.mark.skipif(
    R_PKG is None or src_inputs() is None,
    reason="needs Rscript with the ifrs9qdb R package installed, and "
           "IFRS9_SRC_INPUTS pointing at a raw extract")

R_RUN = r"""
suppressPackageStartupMessages(library(ifrs9qdb))
a <- commandArgs(trailingOnly = TRUE)
res <- run_etl(config_path = a[1], reconcile = FALSE, verbose = FALSE,
               run_validation = TRUE, keep_history = TRUE)
rd <- list.dirs(a[2], recursive = FALSE)[1]
utils::write.csv(build_final_ecl_report(rd), a[3], row.names = FALSE)
cat(rd)
"""


@needs_r
class TestTheEtlReproducesR:
    """Both ETLs on the same raw extract, every file and every contract."""

    @staticmethod
    @pytest.fixture(scope="class")
    def runs(tmp_path_factory):
        import yaml
        from ifrs9qdb.etl.pipeline import run_etl
        from ifrs9qdb.etl.report import build_final_ecl_report
        work = tmp_path_factory.mktemp("rparity")
        cfg_dir = work / "config"
        shutil.copytree(R_PKG / "config", cfg_dir)
        cfg = yaml.safe_load((cfg_dir / "config.yml").read_text(encoding="utf-8"))
        runs_dir = work / "r_runs"
        cfg["paths"].update({
            "input_dir": str(src_inputs()), "output_dir": str(work / "r_out"),
            "runs_dir": str(runs_dir), "static_dir": str(R_PKG / "static"),
            "variable_dictionary": str(cfg_dir / "model.yml"),
            "models": str(cfg_dir / "model.yml"),
            "model_config": str(cfg_dir / "model.yml"),
            "model_inputs": str(cfg_dir / "model_inputs.yml")})
        cfg["paths"].pop("reference_outputs", None)
        cfg["logging"]["log_file"] = str(work / "etl.log")
        cfg["run"]["on_validation_error"] = "warn"
        (cfg_dir / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                             encoding="utf-8")
        script = work / "run.R"
        script.write_text(R_RUN, encoding="utf-8")
        r_report = work / "r_report.csv"
        res = subprocess.run(
            ["Rscript", str(script), str(cfg_dir / "config.yml"), str(runs_dir),
             str(r_report)], capture_output=True, text=True, timeout=1800,
            # R's run_etl writes logs/etl_audit.jsonl under the working
            # directory, so it runs in the scratch folder, not the repository.
            cwd=str(work),
            # R reads the YAML config with the session locale; under POSIX it
            # truncates at the first non-ASCII character (an em-dash) and then
            # reports that model.yml has no `variables:` block.
            env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"})
        assert res.returncode == 0, res.stderr[-2000:]
        r_out = Path(res.stdout.strip().splitlines()[-1]) / "Output"

        py = run_etl(src_inputs(), work / "py_runs")
        assert py.ok, py.error
        py_rep = build_final_ecl_report(py.run_dir, write=False)
        return r_out, Path(py.run_dir) / "Output", pd.read_csv(r_report, low_memory=False), py_rep

    # Row ORDER differs in three files and is not compared: R writes each EAD
    # curve by ascending month and the port by descending (the order the
    # Excel tool's output uses), and StPD and CustomerStagingFlag_1 are
    # grouped differently. LIC keys all three.
    KEYS = {"LifeTimeParameterOther.csv": ["ContractId", "MonthLifetime"],
            "StPD.csv": ["PortfolioCode", "PDBucketDim1", "MonthLifetime"],
            "CustomerStagingFlag_1.csv": ["CustomerId"],
            "CustomerMaster_1.csv": ["CustomerId"],
            "AccountCollateralAllocation.csv": ["ContractId", "CollateralId"]}

    def test_every_output_file_matches(self, runs):
        r_out, py_out, _, _ = runs
        bad = []
        for f in sorted(r_out.glob("*.csv")):
            if f.name == "FinalEclReport.csv":
                continue
            r = pd.read_csv(f, dtype=str, keep_default_na=False)
            p = pd.read_csv(py_out / f.name, dtype=str, keep_default_na=False)
            if r.shape != p.shape or list(r.columns) != list(p.columns):
                bad.append(f"{f.name}: shape {r.shape} vs {p.shape}")
                continue
            key = self.KEYS.get(f.name)
            if key:
                r = r.sort_values(key, kind="stable").reset_index(drop=True)
                p = p.sort_values(key, kind="stable").reset_index(drop=True)
            for c in r.columns:
                a, b = pd.to_numeric(r[c], errors="coerce"), pd.to_numeric(p[c], errors="coerce")
                if a.notna().any() or b.notna().any():
                    same = np.isclose(a.fillna(-9e99), b.fillna(-9e99), rtol=1e-9, atol=1e-6)
                else:
                    same = (r[c].str.strip() == p[c].str.strip()).to_numpy()
                if not same.all():
                    bad.append(f"{f.name}.{c}: {int((~same).sum())} rows")
        assert not bad, "the Python ETL no longer reproduces R:\n  " + "\n  ".join(bad)

    def test_every_contract_is_priced_the_same(self, runs):
        _, _, r_rep, py_rep = runs
        r_rep.columns = [c.replace(".", " ") for c in r_rep.columns]
        a = pd.to_numeric(r_rep["Cla Amount Onbal"], errors="coerce").fillna(0).to_numpy()
        b = pd.to_numeric(py_rep["Cla Amount Onbal"], errors="coerce").fillna(0).to_numpy()
        assert len(a) == len(b)
        assert (r_rep["Contract Id"].astype(str).to_numpy()
                == py_rep["Contract Id"].astype(str).to_numpy()).all()
        assert (pd.to_numeric(r_rep["Ifrs Stage"]).to_numpy()
                == pd.to_numeric(py_rep["Ifrs Stage"]).to_numpy()).all()
        assert np.abs(a - b).max() < 0.01, "a contract is priced differently"
        assert abs(a.sum() - b.sum()) < 0.01
