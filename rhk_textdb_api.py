#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rhk_textdb_api.py

YAML‑backed Textbaustein‑Datenbank (core.yaml + overrides.yaml) mit
Kompatibilitäts‑API für bestehende Notebooks/GUI.

Highlights
----------
- Lädt die Textbausteine aus `textdb/core.yaml` (Release, read‑only) und
  `textdb/overrides.yaml` (lokale Anpassungen; draft/approved).
- Liefert **TextBlock‑Objekte** ähnlich dem alten `rhk_textdb.py`.
- Stellt **Kompatibilitäts‑Konstanten** bereit, damit ältere GUIs ohne große
  Umbauten weiterlaufen können:
  - `P_BLOCKS` (für Procedere‑Auswahl)
  - `DEFAULT_RULES` (Legacy‑Schema: rest/exercise/severity/stepox/volume)

Zusätzlich enthält das Modul Helper‑Funktionen für den Overrides‑Workflow
(Upsert/Approve/Discard) – gedacht für eine GUI‑Admin‑Maske.

Hinweis
-------
Medizinische Inhalte müssen klinisch validiert werden. Dieses Modul ist
"nur" Datenhaltung + Rendering‑Support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Any
import copy
import sys


# ---------------------------------------------------------------------------
# Locate package folder
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parent
# Damit Imports funktionieren, auch wenn das Verzeichnis nicht im sys.path ist (typisch in Jupyter).
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

from textdb_store import TextBlock, TextDB, load_textdb  # noqa: E402


CORE_PATH = _BASE / "textdb" / "core.yaml"
OVERRIDES_PATH = _BASE / "textdb" / "overrides.yaml"

_DB: Optional[TextDB] = None

ALL_BLOCKS: Dict[str, TextBlock] = {}
BUNDLES: Dict[str, Any] = {}
QUICKFINDER: Dict[str, Any] = {}
RULES: Dict[str, Any] = {}
REFERENCES: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Legacy/Compatibility exports (used by older GUIs)
# ---------------------------------------------------------------------------

# Procedere blocks (category == "P")
P_BLOCKS: Dict[str, TextBlock] = {}

# Legacy rules schema expected by some older GUI scripts:
# rules.rest.* / rules.exercise.* / rules.severity.* / rules.stepox.* / rules.volume.*
DEFAULT_RULES: Dict[str, Any] = {}


def _safe_get(d: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _legacy_rules_from_rules(rules: Dict[str, Any]) -> Dict[str, Any]:
    """Erzeugt ein "Legacy"-Rules-Objekt aus dem neuen Rules-Schema.

    Motivation: einige GUI-Skripte erwarten das alte Schema (rest/exercise/...).
    """

    # Ruhe
    mpap_ph = _safe_get(rules, "hemodynamic_definitions", "ph", "mPAP_gt_mmHg", default=20)
    pawp_postcap = _safe_get(rules, "hemodynamic_definitions", "postcapillary_ph", "PAWP_gt_mmHg", default=15)
    pvr_precap = _safe_get(rules, "hemodynamic_definitions", "precapillary_ph", "PVR_gt_WU", default=2)

    # Belastung
    mpap_slope = _safe_get(rules, "exercise_definitions", "exercise_ph", "mPAP_CO_slope_gt_mmHg_per_L_min", default=3)
    pawp_slope = _safe_get(
        rules, "exercise_definitions", "postcapillary_cause_suspected", "PAWP_CO_slope_gt_mmHg_per_L_min", default=2
    )

    # Stepwise Oximetry thresholds
    thr_atrial = _safe_get(rules, "stepwise_oximetry", "stepup_thresholds_percent_points", "atrial_ge", default=7)
    thr_vent = _safe_get(rules, "stepwise_oximetry", "stepup_thresholds_percent_points", "ventricular_or_pa_ge", default=5)

    # Fluid challenge
    pawp_post_thr = _safe_get(rules, "fluid_challenge", "positive_cutoff", "PAWP_post_gt_mmHg", default=18)
    # ΔPAWP wird im neuen Schema nicht als harter Cutoff geführt -> konservativer Default
    delta_pawp_thr = 5

    # Pragmatic severity bins
    mild_ge = 2
    moderate_ge = float(_safe_get(rules, "severity_bins_pragmatic", "pvr_moderate_ge", default=5.0))
    severe_ge = float(_safe_get(rules, "severity_bins_pragmatic", "pvr_severe_ge", default=10.0))
    ci_low = float(_safe_get(rules, "severity_bins_pragmatic", "ci_low_lt", default=2.0))

    return {
        "rest": {
            "mPAP_ph_mmHg": float(mpap_ph),
            "PAWP_postcap_mmHg": float(pawp_postcap),
            "PVR_precap_WU": float(pvr_precap),
        },
        "exercise": {
            "mPAP_CO_slope_mmHg_per_L_min": float(mpap_slope),
            "PAWP_CO_slope_mmHg_per_L_min": float(pawp_slope),
        },
        "severity": {
            "PVR_WU": {"mild_ge": float(mild_ge), "moderate_ge": float(moderate_ge), "severe_ge": float(severe_ge)},
            "CI_L_min_m2": {"severely_reduced_lt": float(ci_low)},
            # Backward-compat keys used by some scripts
            "PVR_mild_from_WU": float(mild_ge),
            "PVR_moderate_from_WU": float(moderate_ge),
            "PVR_severe_from_WU": float(severe_ge),
            "CI_low_lt_L_min_m2": float(ci_low),
        },
        "stepox": {"thr_ra_pct": float(thr_atrial), "thr_rv_pct": float(thr_vent), "thr_pa_pct": float(thr_vent)},
        "volume": {"pawp_post_thr_mmHg": float(pawp_post_thr), "delta_pawp_thr_mmHg": float(delta_pawp_thr)},
    }


def reload() -> None:
    """Re-load database from YAML (core + overrides) and refresh module-level exports."""

    global _DB, ALL_BLOCKS, BUNDLES, QUICKFINDER, RULES, REFERENCES
    global P_BLOCKS, DEFAULT_RULES

    _DB = load_textdb(CORE_PATH, OVERRIDES_PATH)

    ALL_BLOCKS = {b.id: b for b in _DB.list_blocks()}
    BUNDLES = _DB.bundles()
    QUICKFINDER = _DB.quickfinder()
    RULES = _DB.rules()
    REFERENCES = _DB.references()

    # Compatibility exports
    P_BLOCKS = {b.id: b for b in _DB.list_blocks_by_category("P")}
    DEFAULT_RULES = _legacy_rules_from_rules(RULES)


# initial load
reload()


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------


def get_db() -> TextDB:
    if _DB is None:
        raise RuntimeError("TextDB ist nicht geladen.")
    return _DB


def get_block(block_id: str) -> Optional[TextBlock]:
    return ALL_BLOCKS.get(block_id)


def list_blocks(prefix: str) -> List[TextBlock]:
    p = (prefix or "").upper()
    return [b for k, b in sorted(ALL_BLOCKS.items()) if k.upper().startswith(p)]


def list_blocks_by_category(category: str) -> List[TextBlock]:
    c = (category or "").upper()
    return [b for _, b in sorted(ALL_BLOCKS.items()) if (b.category or "").upper() == c]


# ---------------------------------------------------------------------------
# Overrides workflow helpers (used by GUI/Admin)
# ---------------------------------------------------------------------------


def upsert_override_block(block_id: str, data_patch: Dict[str, Any], status: str = "draft") -> None:
    """Create/update an override entry for a block.

    - `status`: "draft" or "approved"
    - `data_patch`: dict with any block fields (template/title/notes/variants/tags/...)
    """

    db = get_db()
    db.upsert_override_block(block_id=block_id, data_patch=data_patch, status=status)
    reload()


def approve_override_block(block_id: str) -> None:
    db = get_db()
    db.approve_override_block(block_id)
    reload()


def discard_override_block(block_id: str) -> None:
    db = get_db()
    db.discard_override_block(block_id)
    reload()


def get_override_entry(block_id: str) -> Optional[Dict[str, Any]]:
    """Returns the raw override entry for a block (or None)."""
    db = get_db()
    root = (db.overrides or {}).get("overrides", {}) or {}
    blocks = root.get("blocks", {}) or {}
    entry = blocks.get(block_id)
    if entry is None:
        return None
    return copy.deepcopy(entry)


def list_override_block_ids(status: Optional[str] = None) -> List[str]:
    """Lists override block IDs. Optionally filtered by status (draft/approved)."""
    db = get_db()
    root = (db.overrides or {}).get("overrides", {}) or {}
    blocks = root.get("blocks", {}) or {}
    out: List[str] = []
    for bid, entry in sorted(blocks.items()):
        if status and (entry or {}).get("status") != status:
            continue
        out.append(bid)
    return out


# ---------------------------------------------------------------------------
# Convenience: Engine re-exports (für GUI/Notebook-Integration)
# ---------------------------------------------------------------------------

from rhk_textdb_engine import (  # noqa: E402
    derive_metrics,
    classify_rest_hemodynamics,
    classify_exercise,
    vasoreactivity,
    fluid_challenge,
    detect_step_up,
    pah_hemodynamic_risk,
    evaluate_pawp_uncertainty,
    build_context,
    suggest_k_bundles,
    suggest_addon_blocks,
    suggest_plan,
)
