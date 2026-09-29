"""
Reading the twelve source extracts.

The upstream Oracle jobs deliver a mixture: some files are genuine Office Open
XML, others are SQL*Plus HTML saved with an ``.xls`` extension. Which is which
has changed between extract runs, so the extension is not trusted -- the format
is detected from the file's first bytes, and the canonical name is tried before
its alternatives.

Column names arrive as the SQL aliased them, which for the HTML exports means
cryptic single letters (``P``, ``CO``, ``COL``). They are kept exactly as found;
the transformation layer maps them by position, as the original Excel tool did.
Renaming them here would look tidier and silently break when an alias changes.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["INPUT_SPECS", "detect_format", "read_input", "read_all_inputs",
           "InputSet"]


@dataclass(frozen=True)
class InputSpec:
    name: str
    file: str
    alternatives: tuple[str, ...]
    description: str


def _spec(name, file, *alts, description=""):
    return InputSpec(name, file, tuple(alts), description)


INPUT_SPECS: list[InputSpec] = [
    _spec("AccountMaster", "AccountMaster.xlsx", "AccountMaster.xls",
          description="Lending account-level master"),
    _spec("AccountMasterInvestments", "AccountMasterInvestments.xlsx",
          "AccountMasterInvestments.xls",
          description="Investment account-level master"),
    _spec("CustomerMaster", "CustomerMaster.xlsx", "CustomerMaster.xls",
          description="Lending customer-level master"),
    _spec("CustomerMasterInvestments", "CustomerMasterInvestments.xls",
          "CustomerMasterInvestments.xlsx",
          description="Investment customer-level master"),
    _spec("CustomerStagingFlag", "CustomerStagingFlag.xlsx",
          "CustomerStagingFlag.xls",
          description="Lending staging flags: watchlist, restructuring, local flags"),
    _spec("CustomerStagingFlagInvestments", "CustomerStagingFlagInvestments.xls",
          "CustomerStagingFlagInvestments.xlsx",
          description="Investment staging flags"),
    _spec("Collateral", "Collateral.xlsx", "Collateral.xls",
          description="Collateral master"),
    _spec("AccountCollateralAllocation", "AccountCollateralAllocation.xlsx",
          "AccountCollateralAllocation.xls",
          description="Which collateral is allocated to which contract"),
    _spec("RepaymentSchedule", "RepaymentSchedule.xlsx", "RepaymentSchedule.xls",
          description="Contractual repayment schedule, drives the EAD curve"),
    _spec("Origination", "Origination.xls", "Origination.xlsx",
          description="Origination rating and date"),
    _spec("OriginationInvestments", "OriginationInvestments.xls",
          "OriginationInvestments.xlsx",
          description="Investment origination"),
    _spec("IndustryCode", "IndustryCode.xls", "IndustryCode.xlsx",
          description="Industry code lookup"),
]

_XLSX_SIG = b"PK\x03\x04"          # zip container
_XLS_SIG = b"\xd0\xcf\x11\xe0"     # OLE2 compound document


def detect_format(path: Path) -> str:
    """Identify a file by its first bytes, not its extension.

    Returns ``"xlsx"``, ``"xls"``, ``"html"`` or ``"csv"``.
    """
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if head.startswith(_XLSX_SIG):
        return "xlsx"
    if head.startswith(_XLS_SIG):
        return "xls"
    lowered = head[:1024].lower()
    if b"<html" in lowered or b"<table" in lowered or b"<!doctype html" in lowered:
        return "html"
    return "csv"


def _read_html_table(path: Path) -> pd.DataFrame:
    """Read a SQL*Plus HTML export.

    SQL*Plus writes one ``<table>`` with a header row, and repeats that header
    every page when PAGESIZE is set -- so repeated header rows appear as data
    and must be dropped. It also writes the Windows-1256 charset used by the
    Arabic-capable database, which is why the encoding is tried explicitly
    before falling back.
    """
    raw = path.read_bytes()
    text = None
    for enc in ("windows-1256", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1", errors="replace")

    # thousands=None: R's reader (rvest) does not read "1,234.5" as a number,
    # so neither does this -- both engines then report it as a value the
    # typing could not read (INPUT_values_typed) instead of one pricing it.
    tables = pd.read_html(io.StringIO(text), flavor="lxml", thousands=None)
    if not tables:
        return pd.DataFrame()
    df = max(tables, key=len)          # the data table, not a heading table

    df.columns = [str(c).strip() for c in df.columns]
    # drop the repeated header rows SQL*Plus emits per page
    if len(df.columns):
        first = df.columns[0]
        df = df[df[first].astype(str).str.strip() != first]
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def read_input(path: Path) -> pd.DataFrame:
    """Read one extract, whatever format it turns out to be."""
    path = Path(path)
    fmt = detect_format(path)
    if fmt == "xlsx":
        return pd.read_excel(path, dtype=object)
    if fmt == "xls":
        return pd.read_excel(path, dtype=object)
    if fmt == "html":
        return _read_html_table(path)
    return pd.read_csv(path, dtype=object, low_memory=False)


def resolve_input_path(input_dir: Path, spec: InputSpec) -> Path | None:
    """Canonical name first, then the alternatives."""
    for name in (spec.file, *spec.alternatives):
        p = input_dir / name
        if p.is_file():
            return p
    # last resort: same stem, any extension, so a renamed delivery still loads
    stem = Path(spec.file).stem.lower()
    for p in input_dir.iterdir():
        if p.is_file() and p.stem.lower() == stem:
            return p
    return None


@dataclass
class InputSet:
    """Everything read, with what was missing recorded rather than raised."""
    tables: dict[str, pd.DataFrame]
    paths: dict[str, Path]
    missing: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in INPUT_SPECS:
            p = self.paths.get(s.name)
            rows.append({
                "input": s.name,
                "file": p.name if p else "—",
                "format": detect_format(p) if p else "—",
                "rows": len(self.tables.get(s.name, [])) if p else 0,
                "found": p is not None,
                "description": s.description,
            })
        return pd.DataFrame(rows)


def read_all_inputs(input_dir, verbose: bool = False) -> InputSet:
    """Read every extract from a directory.

    A missing file is recorded, not raised: the validators report the whole
    picture at once, which is more useful than stopping at the first gap.
    """
    input_dir = Path(input_dir)
    tables, paths, missing = {}, {}, []
    for spec in INPUT_SPECS:
        p = resolve_input_path(input_dir, spec)
        if p is None:
            missing.append(spec.file)
            continue
        try:
            df = read_input(p)
        except Exception as exc:  # a corrupt file is a missing file, with a reason
            missing.append(f"{spec.file} ({type(exc).__name__}: {exc})")
            continue
        tables[spec.name] = df
        paths[spec.name] = p
        if verbose:
            print(f"  {spec.name:<32}{len(df):>8,} rows  ({detect_format(p)})")
    return InputSet(tables=tables, paths=paths, missing=missing)
