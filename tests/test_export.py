"""A run packaged as one zip a reviewer can open six months later.

The property that matters is completeness with honesty: a section that is
missing has to be SAID to be missing, in the README, rather than leaving a
reader to notice an absence. A handover that silently omits the config is
worse than one that says it has none.
"""
from __future__ import annotations

import zipfile

import pytest

from ifrs9qdb.reconcile import build_export

from conftest import ref_output

OUT = ref_output()
needs_run = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")


def _run_dir():
    return OUT.parent if OUT.name.lower() == "output" else OUT


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if OUT is None:
        pytest.skip("set IFRS9_REF_RUN")
    dest = tmp_path_factory.mktemp("export") / "run.zip"
    return build_export(_run_dir(), dest), dest


class TestTheBundle:
    @needs_run
    def test_it_unzips_into_one_self_contained_folder(self, exported):
        """Not a scatter of files into whatever directory it was opened in."""
        result, dest = exported
        assert result["files"] > 0
        with zipfile.ZipFile(dest) as z:
            tops = {n.split("/")[0] for n in z.namelist()}
        assert tops == {_run_dir().name}

    @needs_run
    def test_the_outputs_are_all_there(self, exported):
        result, dest = exported
        with zipfile.ZipFile(dest) as z:
            packaged = {n for n in z.namelist() if "/Output/" in n}
        on_disk = {f.name for f in OUT.glob("*.csv")}
        assert len(packaged) >= len(on_disk)
        assert any("FinalEclReport.csv" in n for n in packaged)

    @needs_run
    def test_the_frozen_config_travels_with_it(self, exported):
        """Reading the repository's current config to explain this run
        explains a different run."""
        result, dest = exported
        if not (_run_dir() / "config_used").is_dir():
            pytest.skip("this run predates the config freeze")
        with zipfile.ZipFile(dest) as z:
            assert any("/config_used/" in n for n in z.namelist())

    @needs_run
    def test_it_carries_its_own_explanation(self, exported):
        result, dest = exported
        with zipfile.ZipFile(dest) as z:
            names = z.namelist()
            readme = next(n for n in names if n.endswith("README.txt"))
            text = z.read(readme).decode()
        assert "WHAT IS HERE" in text
        assert "config_used" in text
        for extra in ("code_version.txt", "approval_summary.txt"):
            assert any(n.endswith(extra) for n in names), extra

    @needs_run
    def test_a_missing_section_is_stated_not_silently_absent(self, exported):
        """A reader should not have to notice an absence."""
        result, dest = exported
        with zipfile.ZipFile(dest) as z:
            readme = next(n for n in z.namelist() if n.endswith("README.txt"))
            text = z.read(readme).decode()
        assert "NOT PRESENT" in text or result["skipped"] == []
        for missing in result["skipped"]:
            assert missing in text, missing

    @needs_run
    def test_the_source_extracts_are_off_by_default(self, exported):
        """They are large and hold customer data, so including them is a
        decision rather than an accident."""
        result, dest = exported
        with zipfile.ZipFile(dest) as z:
            assert not any("/input/" in n for n in z.namelist())

    @needs_run
    def test_the_manifest_of_contents_matches_the_zip(self, exported):
        result, dest = exported
        with zipfile.ZipFile(dest) as z:
            inside = {n.split("/", 1)[1] for n in z.namelist()
                      if "/" in n and not n.endswith("/")}
        # README.txt is written last and is not itself listed as content.
        assert set(result["contents"]) | {"README.txt"} == inside



class TestRefusals:
    def test_a_run_that_is_not_there_is_reported(self, tmp_path):
        with pytest.raises(Exception):
            build_export(tmp_path / "nope", tmp_path / "x.zip")

    def test_a_run_with_no_outputs_still_packages(self, tmp_path):
        """A broken run is exactly the one somebody needs to hand to support."""
        run = tmp_path / "run_empty"
        (run / "reports").mkdir(parents=True)
        (run / "reports" / "validation.csv").write_text("id\n", encoding="utf-8")
        (run / "Output").mkdir()
        dest = tmp_path / "empty.zip"
        r = build_export(run, dest)
        assert dest.is_file()
        with zipfile.ZipFile(dest) as z:
            readme = next(n for n in z.namelist() if n.endswith("README.txt"))
            text = z.read(readme).decode()
        assert "NOT PRESENT" in text
        assert "config_used" in text
