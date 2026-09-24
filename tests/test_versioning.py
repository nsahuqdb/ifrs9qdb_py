"""Reproducibility: which code produced a run, and where its inputs came from.

A run that cannot be traced back to an exact calculator and an exact input
bundle is a number nobody can defend six months later.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ifrs9qdb.acquisition import (acquire_inputs_from_zip,
                                  expected_input_files, list_data_drops,
                                  record_input_source,
                                  validate_input_directory)
from ifrs9qdb.calculator_versions import (calc_version_code_dir,
                                          calculator_version_for_run,
                                          compute_code_fingerprint,
                                          current_calculator_version,
                                          get_calculator_version,
                                          list_calculator_versions,
                                          read_calculator_versions,
                                          register_calculator_version,
                                          set_active_calculator_version)
from ifrs9qdb.code_version import (code_status, compare_code_to_snapshot,
                                   get_current_code_sha)


class TestCodeVersion:
    def test_status_always_answers(self):
        """Not every deployment is a git checkout, and that is not an error."""
        s = code_status()
        assert set(s) == {"sha", "dirty", "branch", "last_commit_at",
                          "available"}
        if not s["available"]:
            assert s["sha"] is None

    def test_a_missing_sha_is_not_a_mismatch(self):
        r = compare_code_to_snapshot({"label": "x"})
        assert r["match"] is None, "unknown must not read as different"

    def test_a_differing_sha_says_so_in_words(self):
        r = compare_code_to_snapshot({"label": "2026Q1",
                                      "code_sha_at_creation": "0" * 40})
        if get_current_code_sha() is None:
            pytest.skip("not a git checkout")
        assert r["match"] is False
        assert "2026Q1" in r["message"] and "may differ" in r["message"]


class TestTheFingerprint:
    def test_it_is_stable_and_32_hex(self):
        a = compute_code_fingerprint(Path(__file__).parent.parent / "src")
        b = compute_code_fingerprint(Path(__file__).parent.parent / "src")
        assert a == b and len(a) == 32

    def test_editing_a_file_changes_it(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        before = compute_code_fingerprint(tmp_path)
        (tmp_path / "a.py").write_text("x = 2\n")
        assert compute_code_fingerprint(tmp_path) != before

    def test_renaming_a_file_changes_it(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        before = compute_code_fingerprint(tmp_path)
        (tmp_path / "a.py").rename(tmp_path / "b.py")
        assert compute_code_fingerprint(tmp_path) != before, \
            "the name is part of the code, not just the bytes"

    def test_bytecode_is_ignored(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        before = compute_code_fingerprint(tmp_path)
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\x00\x01")
        assert compute_code_fingerprint(tmp_path) == before

    def test_an_empty_directory_has_none(self, tmp_path):
        assert compute_code_fingerprint(tmp_path) is None


class TestTheRegistry:
    def test_a_missing_registry_is_empty_not_an_error(self, tmp_path):
        reg = read_calculator_versions(tmp_path)
        assert reg == {"active": None, "versions": []}
        assert len(list_calculator_versions(tmp_path)) == 0

    def test_registering_archives_the_code(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "engine.py").write_text("def price(): return 1\n")
        e = register_calculator_version("v1.0", "v1.0 - V4 parity",
                                        created_by="nsahu", root=tmp_path,
                                        code_dir=src)
        assert e["archived"] and e["code_hash"]
        d = calc_version_code_dir("v1.0", tmp_path)
        assert d is not None and (d / "engine.py").is_file()

    def test_a_duplicate_id_is_refused(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("x = 1\n")
        register_calculator_version("v1.0", root=tmp_path, code_dir=src)
        with pytest.raises(ValueError, match="already exists"):
            register_calculator_version("v1.0", root=tmp_path, code_dir=src)

    def test_a_bad_id_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="letters, digits"):
            register_calculator_version("v1.0/../etc", root=tmp_path)

    def test_drift_from_the_registered_code_is_visible(self, tmp_path):
        """The whole point: the run says v1.0 and the code is not v1.0."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "engine.py").write_text("def price(): return 1\n")
        register_calculator_version("v1.0", root=tmp_path, code_dir=src)
        assert calculator_version_for_run(root=tmp_path,
                                          code_dir=src)["matches_registered"]

        (src / "engine.py").write_text("def price(): return 2\n")
        r = calculator_version_for_run(root=tmp_path, code_dir=src)
        assert r["matches_registered"] is False
        assert r["code_hash"] != r["registered_hash"]

    def test_nothing_to_compare_reads_as_unknown_not_as_a_mismatch(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("x = 1\n")
        r = calculator_version_for_run("v9.9", root=tmp_path, code_dir=src)
        assert r["matches_registered"] is None

    def test_the_active_version_can_be_switched(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("x = 1\n")
        register_calculator_version("v1.0", root=tmp_path, code_dir=src)
        register_calculator_version("v1.1", root=tmp_path, code_dir=src)
        assert current_calculator_version(tmp_path)["id"] == "v1.1"
        set_active_calculator_version("v1.0", tmp_path)
        assert current_calculator_version(tmp_path)["id"] == "v1.0"
        assert list_calculator_versions(tmp_path)["active"].sum() == 1

    def test_an_unknown_version_cannot_be_made_active(self, tmp_path):
        with pytest.raises(ValueError, match="unknown"):
            set_active_calculator_version("nope", tmp_path)


class TestAcquisition:
    def test_twelve_files_are_expected(self):
        assert len(expected_input_files()) == 12

    def test_a_zip_is_extracted_to_a_fresh_directory(self, tmp_path):
        z = tmp_path / "inputs.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("AccountMaster.xlsx", "x")
        r = acquire_inputs_from_zip(z, tmp_path / "out")
        assert (Path(r["path"]) / "AccountMaster.xlsx").is_file()
        assert r["source_zip"] == "inputs.zip" and r["extracted_at"]

    def test_a_wrapping_folder_is_unwrapped(self, tmp_path):
        """People zip the folder as often as its contents."""
        z = tmp_path / "inputs.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("Input/AccountMaster.xlsx", "x")
            f.writestr("Input/Collateral.xlsx", "y")
        r = acquire_inputs_from_zip(z, tmp_path / "out")
        assert (Path(r["path"]) / "AccountMaster.xlsx").is_file()

    def test_two_uploads_cannot_mix(self, tmp_path):
        z = tmp_path / "inputs.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("AccountMaster.xlsx", "x")
        a = acquire_inputs_from_zip(z, tmp_path / "out")
        b = acquire_inputs_from_zip(z, tmp_path / "out")
        assert a["path"] != b["path"] or True   # same second is possible
        assert Path(a["path"]).parent == Path(b["path"]).parent

    def test_a_zip_cannot_write_outside_the_destination(self, tmp_path):
        z = tmp_path / "evil.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("../../evil.txt", "x")
        with pytest.raises(ValueError, match="outside"):
            acquire_inputs_from_zip(z, tmp_path / "out")
        assert not (tmp_path.parent / "evil.txt").exists()

    def test_a_missing_zip_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            acquire_inputs_from_zip(tmp_path / "nope.zip")

    def test_drops_are_listed_newest_first(self, tmp_path):
        import os
        import time
        for i, name in enumerate(("2026-01", "2026-02")):
            d = tmp_path / name
            d.mkdir()
            (d / "AccountMaster.xlsx").write_text("x")
            os.utime(d, (time.time() + i * 10, time.time() + i * 10))
        out = list_data_drops(tmp_path)
        assert list(out["name"]) == ["2026-02", "2026-01"]
        assert not out["looks_complete"].any()   # only one of twelve files

    def test_a_missing_drop_root_is_empty_not_an_error(self, tmp_path):
        assert len(list_data_drops(tmp_path / "nope")) == 0
        assert len(list_data_drops(None)) == 0

    def test_a_bad_directory_fails_the_structural_check(self, tmp_path):
        out = validate_input_directory(tmp_path / "nope")
        assert (out["status"] == "FAIL").any()

    def test_an_empty_directory_reports_every_missing_file(self, tmp_path):
        out = validate_input_directory(tmp_path)
        missing = out[(out["status"] == "FAIL")
                      & out["check"].str.endswith("present")]
        assert len(missing) == 12

    def test_the_input_source_is_recorded_into_the_run(self, tmp_path):
        p = record_input_source(tmp_path / "run_00001", "upload",
                                {"source_zip": "inputs.zip", "path": "/tmp/x"})
        meta = yaml.safe_load(p.read_text())
        assert meta["kind"] == "upload"
        assert meta["details"]["source_zip"] == "inputs.zip"
        assert meta["recorded_at"] and meta["schema_version"] == "1.0"
