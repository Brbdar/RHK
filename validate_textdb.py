#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_textdb.py

Kleiner Validator für rhk_textdb YAML-Dateien.

Aufruf:
    python validate_textdb.py

Exit-Code:
    0 = ok
    1 = Fehler gefunden
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set
import sys
import yaml

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from textdb_store import SafeFormatDict, extract_placeholders  # noqa: E402


CORE = BASE / "textdb" / "core.yaml"
OVR = BASE / "textdb" / "overrides.yaml"


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    core = load(CORE)
    blocks = core.get("blocks", {}) or {}
    bundles = core.get("bundles", {}) or {}
    quick = core.get("quickfinder", {}) or {}

    errors: List[str] = []

    # 1) id consistency
    for bid, b in blocks.items():
        if str(b.get("id", "")) != bid:
            errors.append(f"[blocks] Key '{bid}' != block.id '{b.get('id')}'")

    # 2) placeholder consistency + formatting sanity
    for bid, b in blocks.items():
        tpl = str(b.get("template", ""))
        variants = b.get("variants", {}) or {}
        inputs_used = set(b.get("inputs_used", []) or [])

        ph = set(extract_placeholders(tpl))
        for v in variants.values():
            ph |= set(extract_placeholders(str(v)))

        if inputs_used != ph:
            errors.append(f"[placeholders] {bid}: inputs_used != extracted placeholders (diff={sorted((inputs_used^ph))})")

        try:
            _ = tpl.format_map(SafeFormatDict())
        except Exception as e:
            errors.append(f"[format] {bid}: template format error: {e}")

        for vk, vv in variants.items():
            try:
                _ = str(vv).format_map(SafeFormatDict())
            except Exception as e:
                errors.append(f"[format] {bid}: variant '{vk}' format error: {e}")

    # 3) bundle references
    existing_block_ids: Set[str] = set(blocks.keys())
    for bundle_id, bundle in bundles.items():
        if not isinstance(bundle, dict):
            errors.append(f"[bundles] {bundle_id}: expected dict, got {type(bundle)}")
            continue
        for section, ids in bundle.items():
            if section == "P_suggestions":
                continue
            for bid in ids:
                if bid not in existing_block_ids:
                    errors.append(f"[bundles] {bundle_id}.{section}: unknown block id '{bid}'")

    # 4) quickfinder references -> bundle ids
    existing_bundle_ids: Set[str] = set(bundles.keys())
    for k, v in quick.items():
        if not isinstance(v, list):
            errors.append(f"[quickfinder] {k}: expected list of bundle ids")
            continue
        for bundle_id in v:
            if bundle_id not in existing_bundle_ids:
                errors.append(f"[quickfinder] {k}: unknown bundle '{bundle_id}'")

    if errors:
        print("\n".join(errors))
        return 1

    print("OK: core.yaml validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
