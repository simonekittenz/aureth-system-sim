"""Constellation sets: several different ways of joining the same named stars.

A set is a list of figures; a figure is a list of lines; a line joins two named stars. Sets live in data/constellations.yaml and
refer to stars by name (the stars themselves are in data/aureth.yaml), so any number of sets can sit on the same stars.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

CON_FILE = Path(__file__).resolve().parent.parent / "data" / "constellations.yaml"

HEADER = """# Constellation sets: different ways of joining the SAME named stars (the stars are in data/aureth.yaml, under sky.stars).
# A set holds figures; a figure holds lines; each line joins two stars by NAME.
# Everything here is a placeholder to build on. The Night sky tab can edit this file; its Save button rewrites it
# (comments other than this header are not kept, so put notes in the `note` fields instead).
"""


def new_set(name: str, color: str = "#8ecae6", note: str = "") -> dict:
    return {"name": name, "color": color, "note": note, "figures": []}


def load_constellations(path: Path = CON_FILE) -> list[dict]:
    if not Path(path).exists():
        return []
    d = yaml.safe_load(Path(path).read_text()) or {}
    out = []
    for s in d.get("sets", []) or []:
        out.append({"name": str(s["name"]), "color": str(s.get("color", "#8ecae6")), "note": str(s.get("note", "")),
                    "figures": [{"name": str(f["name"]), "lines": [[str(a), str(b)] for a, b in f.get("lines", [])]}
                                for f in s.get("figures", []) or []]})
    return out


def save_constellations(sets: list[dict], path: Path = CON_FILE) -> None:
    q = json.dumps                       # JSON strings are valid YAML strings, and handle apostrophes and accents
    out = [HEADER, "sets:"]
    for s in sets:
        out += [f"  - name: {q(s['name'], ensure_ascii=False)}", f"    color: {q(s['color'])}", f"    note: {q(s['note'], ensure_ascii=False)}"]
        if not s["figures"]:
            out.append("    figures: []")
            continue
        out.append("    figures:")
        for f in s["figures"]:
            out.append(f"      - name: {q(f['name'], ensure_ascii=False)}")
            if not f["lines"]:
                out.append("        lines: []")
                continue
            out.append("        lines:")
            out += [f"          - [{q(a, ensure_ascii=False)}, {q(b, ensure_ascii=False)}]" for a, b in f["lines"]]
    Path(path).write_text("\n".join(out) + "\n")


def clone(sets: list[dict]) -> list[dict]:
    return copy.deepcopy(sets)


def missing_stars(cset: dict, known: set[str]) -> list[str]:
    """Star names used by a set that no longer exist in the star list (for example after a rename)."""
    seen = []
    for f in cset["figures"]:
        for a, b in f["lines"]:
            for n in (a, b):
                if n not in known and n not in seen:
                    seen.append(n)
    return seen


def figure_stars(fig: dict) -> list[str]:
    out = []
    for a, b in fig["lines"]:
        for n in (a, b):
            if n not in out:
                out.append(n)
    return out
