"""Stress scenarios: the main variables a scenario sets, year by year, and
the saved scenarios.

A scenario sets one or more *main* variables -- a deviation from the
baseline in each forecast year, in the variable's own unit -- and lets the
*linked effects* (``macro.system``) move the rest. ``linked`` names the
variables allowed to move that way (``None``: every one the main variables
reach); a variable left out stays at baseline and passes nothing on. Only
main variables are set by hand: a linked variable's path is whatever the
system makes of the main ones.

The builder draws a main variable's path from a shape:

    one_year   the move in year 1, then left to the linked effects
    recover    the move in year 1, shrinking to nothing over ``years`` years
    held       the move in every year
    custom     year by year; a blank year is left to the linked effects

``restructured`` is the share of Stage 1 borrowers a payment-deferral
programme moves to Stage 2 (the 2020 programme moved about 72% of QDB's).

Saved scenarios live in an SQLite database in the project folder
(``ScenarioStore``), every save kept as a version; the library ships with
the package (``library.py``).
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SHAPES", "Scenario", "shape_path", "ScenarioStore", "scenarios_db_path"]

SHAPES = ("one_year", "recover", "held", "custom")


def shape_path(shape: str, size: float, H: int = 5, years: int = 2) -> list:
    """A main variable's yearly path from a shape: numbers for the years it
    sets, ``None`` for the years it leaves to the linked effects."""
    size = float(size)
    if shape == "held":
        return [size] * H
    if shape == "recover":
        n = max(1, int(years))
        return [size * max(0.0, 1.0 - t / n) for t in range(min(n, H))] + [None] * max(0, H - n)
    return [size] + [None] * (H - 1)


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


@dataclass
class Scenario:
    name: str = "Scenario"
    main: dict = field(default_factory=dict)        # variable -> [deviation or None] by year
    linked: list | None = None                      # None: every variable the main ones reach
    restructured: float = 0.0
    description: str = ""
    category: str = "mine"                          # history | supervisory | hypothetical | current | mine
    source: str = ""
    id: str | None = None
    based_on: str | None = None
    shapes: dict = field(default_factory=dict)      # variable -> {"shape", "size", "years"}
    episode: list | None = None
    window: list | None = None
    translation: str = ""
    tail: bool = False
    refs: dict = field(default_factory=dict)        # variable -> how the library derived it
    region: str = "qatar"                           # global | gcc | qatar
    history: str = ""                               # what happened
    narrative: str = ""                             # the scenario as it unfolds from Y1
    why: dict = field(default_factory=dict)         # main variable -> why and how it is set
    book: str = ""                                  # how it reaches the provision
    limits: str = ""
    references: list = field(default_factory=list)  # [{"title", "url"}]

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = dict(d or {})
        main = {}
        for v, p in (d.get("main") or {}).items():
            if p is None:
                continue
            main[str(v)] = [_num(x) for x in p]
        linked = d.get("linked")
        return cls(
            name=str(d.get("name") or "Scenario"), main=main,
            linked=None if linked is None else [str(x) for x in linked],
            restructured=float(d.get("restructured") or 0.0),
            description=str(d.get("description") or ""),
            category=str(d.get("category") or "mine"), source=str(d.get("source") or ""),
            id=d.get("id"), based_on=d.get("based_on"),
            shapes={str(k): dict(v) for k, v in (d.get("shapes") or {}).items() if v},
            episode=d.get("episode"), window=d.get("window"),
            translation=str(d.get("translation") or ""), tail=bool(d.get("tail", False)),
            refs={str(k): str(v) for k, v in (d.get("refs") or {}).items()},
            region=str(d.get("region") or "qatar"), history=str(d.get("history") or ""),
            narrative=str(d.get("narrative") or ""),
            why={str(k): str(v) for k, v in (d.get("why") or {}).items()},
            book=str(d.get("book") or ""), limits=str(d.get("limits") or ""),
            references=[dict(r) for r in (d.get("references") or []) if isinstance(r, dict)])

    def to_dict(self) -> dict:
        return asdict(self)

    def given(self, H: int = 5) -> dict:
        """The main variables' paths, padded to H years (``None`` = left to
        the linked effects)."""
        out = {}
        for v, p in self.main.items():
            p = list(p or [])
            if not any(x is not None for x in p):
                continue
            out[v] = ([_num(x) for x in p] + [None] * H)[:H]
        return out

    def variables(self) -> list:
        return list(self.given())

    def is_library(self) -> bool:
        return self.category != "mine"

    def edited_from(self, other: "Scenario", H: int = 5) -> bool:
        """Whether this scenario's numbers differ from ``other``'s: a changed
        library scenario must be saved under a new name."""
        a, b = self.given(H), other.given(H)
        if set(a) != set(b):
            return True
        for v in a:
            for x, y in zip(a[v], b[v]):
                if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-9):
                    return True
        la = None if self.linked is None else sorted(self.linked)
        lb = None if other.linked is None else sorted(other.linked)
        return la != lb or abs(self.restructured - other.restructured) > 1e-12


# ---------------------------------------------------------------- saving --
def scenarios_db_path(project_root) -> Path:
    return Path(project_root) / "scenarios" / "scenarios.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenario (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    based_on TEXT,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scenario_version (
    id TEXT NOT NULL,
    version INTEGER NOT NULL,
    body TEXT NOT NULL,
    saved_at TEXT NOT NULL,
    saved_by TEXT NOT NULL,
    action TEXT NOT NULL,
    PRIMARY KEY (id, version)
);
"""


class ScenarioStore:
    """The project's saved scenarios: one row each, every save kept as a
    version (who, when, what), a deletion only hiding the row."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as c:
            c.executescript(_SCHEMA)

    def _db(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _now() -> str:
        return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    def list(self) -> list[dict]:
        with self._db() as c:
            rows = c.execute("SELECT * FROM scenario WHERE deleted = 0 ORDER BY updated_at DESC")
            out = []
            for r in rows.fetchall():
                d = json.loads(r["body"])
                # a saved scenario is the user's own, unless it was saved as a
                # supervisor's (imported from the publisher's tables, or entered)
                cat = "supervisory" if d.get("category") == "supervisory" else "mine"
                d.update({"id": r["id"], "category": cat, "saved": True,
                          "based_on": r["based_on"],
                          "created_at": r["created_at"], "created_by": r["created_by"],
                          "updated_at": r["updated_at"], "updated_by": r["updated_by"]})
                out.append(d)
            return out

    def scenarios(self) -> list[Scenario]:
        return [Scenario.from_dict(d) for d in self.list()]

    def get(self, sid: str) -> dict | None:
        return next((d for d in self.list() if d["id"] == sid), None)

    def _name_taken(self, c, name: str, but: str | None = None) -> bool:
        q = "SELECT id FROM scenario WHERE deleted = 0 AND lower(name) = lower(?)"
        return any(r["id"] != but for r in c.execute(q, (name.strip(),)).fetchall())

    def save(self, scn: Scenario, user: str = "", sid: str | None = None) -> str:
        """Save ``scn``: a new scenario, or -- with ``sid`` -- a new version
        of a saved one. Names are unique among the saved scenarios."""
        name = (scn.name or "").strip()
        if not name:
            raise ValueError("A scenario needs a name.")
        body = scn.to_dict()
        body["category"] = "supervisory" if scn.category == "supervisory" else "mine"
        now, user = self._now(), user or "unknown"
        with self._db() as c:
            if self._name_taken(c, name, sid):
                raise ValueError(f"A saved scenario is already called {name!r}.")
            if sid is None:
                sid = uuid.uuid4().hex[:12]
                body["id"] = sid
                c.execute("INSERT INTO scenario VALUES (?,?,?,?,?,?,?,?,0)",
                          (sid, name, scn.based_on, json.dumps(body), now, user, now, user))
                action = "created"
            else:
                if not c.execute("SELECT 1 FROM scenario WHERE id = ? AND deleted = 0",
                                 (sid,)).fetchone():
                    raise KeyError(f"No saved scenario {sid!r}.")
                body["id"] = sid
                c.execute("UPDATE scenario SET name = ?, body = ?, updated_at = ?, "
                          "updated_by = ? WHERE id = ?", (name, json.dumps(body), now, user, sid))
                action = "updated"
            v = c.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM scenario_version "
                          "WHERE id = ?", (sid,)).fetchone()[0]
            c.execute("INSERT INTO scenario_version VALUES (?,?,?,?,?,?)",
                      (sid, v, json.dumps(body), now, user, action))
        return sid

    def delete(self, sid: str, user: str = "") -> None:
        now = self._now()
        with self._db() as c:
            row = c.execute("SELECT body FROM scenario WHERE id = ? AND deleted = 0",
                            (sid,)).fetchone()
            if not row:
                raise KeyError(f"No saved scenario {sid!r}.")
            c.execute("UPDATE scenario SET deleted = 1, updated_at = ?, updated_by = ? "
                      "WHERE id = ?", (now, user or "unknown", sid))
            v = c.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM scenario_version "
                          "WHERE id = ?", (sid,)).fetchone()[0]
            c.execute("INSERT INTO scenario_version VALUES (?,?,?,?,?,?)",
                      (sid, v, row["body"], now, user or "unknown", "deleted"))

    def history(self, sid: str) -> list[dict]:
        with self._db() as c:
            rows = c.execute("SELECT version, saved_at, saved_by, action FROM scenario_version "
                             "WHERE id = ? ORDER BY version", (sid,)).fetchall()
            return [dict(r) for r in rows]
