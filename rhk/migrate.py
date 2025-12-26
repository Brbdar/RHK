# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .version import SCHEMA_VERSION, APP_VERSION


def is_saved_case(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("schema") == "rhk_case"


def migrate_payload_to_ui(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Accepts:
      - New format: {"schema":"rhk_case","schema_version":N,"ui":{...}}
      - Old format: a dict of UI-like keys (flat) or structured keys.
    Returns (ui_dict, info_message).
    """
    if not isinstance(payload, dict):
        return {}, "Ungültige Datei (kein JSON-Objekt)."

    # New format
    if payload.get("schema") == "rhk_case" and isinstance(payload.get("ui"), dict):
        ui = payload["ui"]
        ver = payload.get("schema_version", "?")
        return ui, f"Fall geladen (Schema v{ver})."

    # v0: assume flat dict already
    if any(k in payload for k in ("last_name", "mpap", "pawp", "story", "who_fc")):
        return payload, "Fall geladen (Legacy-Format)."

    # Attempt to map structured legacy exports
    ui: Dict[str, Any] = {}
    patient = payload.get("patient") or {}
    rhc = payload.get("rhc") or payload.get("hemodynamics") or {}
    ui.update({
        "first_name": patient.get("first_name"),
        "last_name": patient.get("last_name"),
        "birthdate": patient.get("birthdate") or patient.get("dob"),
        "story": patient.get("story"),
        "mpap": rhc.get("mpap"),
        "pawp": rhc.get("pawp"),
        "rap": rhc.get("rap"),
        "co_td": rhc.get("co_td"),
        "co_fick": rhc.get("co_fick"),
    })
    return ui, "Fall geladen (Struktur-Import, unvollständig gemappt)."


def build_saved_case(ui: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "rhk_case",
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ui": ui,
    }
