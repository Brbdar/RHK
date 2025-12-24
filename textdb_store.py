#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
textdb_store.py

YAML-basierter Textbaustein-Store für den RHK/PH-Befundassistenten.

Designziele
-----------
- core.yaml ist read-only (Release/Versionierung).
- overrides.yaml enthält lokale Anpassungen (draft/approved) + neue Bausteine.
- Kompatibilität: liefert TextBlock-Objekte ähnlich dem alten rhk_textdb.py.
- Robustes Rendering: fehlende Platzhalter führen nicht zu Exceptions (werden sichtbar markiert).

Hinweis: Medizinische Inhalte müssen immer klinisch validiert werden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import re
import yaml
import string
import datetime


SCHEMA_VERSION = 2


# ---------------------------
# Utilities
# ---------------------------

class SafeFormatDict(dict):
    """dict, der fehlende Keys als '{key}' stehen lässt statt KeyError."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def extract_placeholders(text: str) -> List[str]:
    """Extrahiert Python-format Platzhalter {name} aus einem Template."""
    formatter = string.Formatter()
    fields: List[str] = []
    for _, field_name, _, _ in formatter.parse(text):
        if not field_name:
            continue
        # Field name kann auch "a[0]" oder "a.b" sein -> so lassen.
        fields.append(field_name)
    # unique, sorted
    return sorted(set(fields))


def deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rekursives Merge:
    - dict + dict -> merge
    - sonst: patch überschreibt base
    """
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=120)


# ---------------------------
# Data classes
# ---------------------------

@dataclass
class TextBlock:
    """
    Kompatibles Objekt wie im alten rhk_textdb.py, erweitert um optionale Meta-Felder.
    """
    id: str
    title: str
    applies_to: str
    template: str
    category: str
    variants: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # v2 extras
    kind: str = ""
    tags: List[str] = field(default_factory=list)
    priority: int = 100
    inputs_used: List[str] = field(default_factory=list)

    def render(self, data: Dict[str, Any], variant: Optional[str] = None) -> str:
        """
        Rendert template oder variants[variant] mit SafeFormatDict.
        """
        mapping = SafeFormatDict(**(data or {}))
        if variant:
            tpl = self.variants.get(variant, self.template)
        else:
            tpl = self.template
        return tpl.format_map(mapping)


@dataclass
class TextDB:
    core_path: Path
    overrides_path: Path
    core: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    merged: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> "TextDB":
        self.core = load_yaml(self.core_path)
        self.overrides = load_yaml(self.overrides_path)
        self._validate_schema(self.core, "core.yaml")
        if self.overrides:
            self._validate_schema(self.overrides, "overrides.yaml", allow_empty=True)
        self.merged = self._merge(self.core, self.overrides)
        return self

    # ---------- Schema / Merge ----------

    def _validate_schema(self, obj: Dict[str, Any], name: str, allow_empty: bool = False) -> None:
        if allow_empty and not obj:
            return
        ver = obj.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise ValueError(f"{name}: schema_version erwartet {SCHEMA_VERSION}, gefunden {ver}")

    def _merge(self, core: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge-Regeln:
        - blocks: core.blocks + approved overrides.blocks
        - bundles, quickfinder: patchbar via overrides
        """
        merged = copy.deepcopy(core)

        ovr_root = (overrides or {}).get("overrides", {})
        ovr_blocks = ovr_root.get("blocks", {}) or {}
        ovr_bundles = ovr_root.get("bundles", {}) or {}
        ovr_quick = ovr_root.get("quickfinder", {}) or {}

        # blocks
        merged_blocks = merged.get("blocks", {}) or {}
        for bid, patch in ovr_blocks.items():
            status = (patch or {}).get("status", "approved")
            if status != "approved":
                continue
            if bid in merged_blocks:
                merged_blocks[bid] = deep_merge_dict(merged_blocks[bid], patch.get("data", patch))
            else:
                # neuer Block
                data = patch.get("data", patch)
                merged_blocks[bid] = data
        merged["blocks"] = merged_blocks

        # bundles / quickfinder: einfache deep_merge
        if ovr_bundles:
            merged["bundles"] = deep_merge_dict(merged.get("bundles", {}) or {}, ovr_bundles)
        if ovr_quick:
            merged["quickfinder"] = deep_merge_dict(merged.get("quickfinder", {}) or {}, ovr_quick)

        return merged

    # ---------- Public API ----------

    def get_block(self, block_id: str) -> Optional[TextBlock]:
        b = (self.merged.get("blocks", {}) or {}).get(block_id)
        if not b:
            return None
        return self._as_textblock(b)

    def list_blocks(self, prefix: Optional[str] = None) -> List[TextBlock]:
        blocks = self.merged.get("blocks", {}) or {}
        items = sorted(blocks.items(), key=lambda kv: kv[0])
        out: List[TextBlock] = []
        for bid, b in items:
            if prefix and not bid.upper().startswith(prefix.upper()):
                continue
            out.append(self._as_textblock(b))
        return out

    def list_blocks_by_category(self, category: str) -> List[TextBlock]:
        c = (category or "").upper()
        return [b for b in self.list_blocks() if (b.category or "").upper() == c]

    def bundles(self) -> Dict[str, Any]:
        return self.merged.get("bundles", {}) or {}

    def quickfinder(self) -> Dict[str, Any]:
        return self.merged.get("quickfinder", {}) or {}

    def rules(self) -> Dict[str, Any]:
        return self.merged.get("rules", {}) or {}

    def references(self) -> Dict[str, Any]:
        return self.merged.get("references", {}) or {}

    # ---------- Overrides workflow ----------

    def upsert_override_block(self, block_id: str, data_patch: Dict[str, Any], status: str = "draft") -> None:
        """
        Erzeugt/aktualisiert einen Block-Override in overrides.yaml.
        - status: 'draft' oder 'approved'
        - data_patch: Felder wie template/variants/title/notes/...
        """
        overrides = self.overrides or {"schema_version": SCHEMA_VERSION, "meta": {}, "overrides": {"blocks": {}, "bundles": {}, "quickfinder": {}}}
        overrides.setdefault("schema_version", SCHEMA_VERSION)
        overrides.setdefault("meta", {})
        overrides.setdefault("overrides", {}).setdefault("blocks", {})
        overrides["meta"]["updated_at"] = datetime.date.today().isoformat()

        overrides["overrides"]["blocks"][block_id] = {
            "status": status,
            "data": data_patch,
        }
        self.overrides = overrides
        dump_yaml(self.overrides, self.overrides_path)
        # reload merged
        self.load()

    def approve_override_block(self, block_id: str) -> None:
        overrides = self.overrides or {}
        blocks = (overrides.get("overrides", {}) or {}).get("blocks", {}) or {}
        if block_id not in blocks:
            raise KeyError(f"Override für {block_id} existiert nicht.")
        blocks[block_id]["status"] = "approved"
        overrides["meta"]["updated_at"] = datetime.date.today().isoformat()
        dump_yaml(overrides, self.overrides_path)
        self.overrides = overrides
        self.load()

    def discard_override_block(self, block_id: str) -> None:
        overrides = self.overrides or {}
        root = overrides.get("overrides", {}) or {}
        blocks = root.get("blocks", {}) or {}
        if block_id in blocks:
            del blocks[block_id]
        overrides["meta"]["updated_at"] = datetime.date.today().isoformat()
        dump_yaml(overrides, self.overrides_path)
        self.overrides = overrides
        self.load()

    # ---------- Internal ----------

    def _as_textblock(self, b: Dict[str, Any]) -> TextBlock:
        return TextBlock(
            id=str(b.get("id", "")),
            title=str(b.get("title", "")),
            applies_to=str(b.get("applies_to", "")),
            template=str(b.get("template", "")),
            category=str(b.get("category", "")),
            variants=dict(b.get("variants", {}) or {}),
            notes=str(b.get("notes", "") or ""),
            kind=str(b.get("kind", "") or ""),
            tags=list(b.get("tags", []) or []),
            priority=int(b.get("priority", 100) or 100),
            inputs_used=list(b.get("inputs_used", []) or []),
        )


def load_textdb(core_path: Path, overrides_path: Path) -> TextDB:
    return TextDB(core_path=core_path, overrides_path=overrides_path).load()
