#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RHK Befundassistent (Web) – v19

- Ultra-interaktiver Gradio-Assistenzbogen für RHK-/PH-Befunde
- Deklaratives Regelwerk (YAML) zur Guideline-nahen Klassifikation, Modulen & Empfehlungen
- Separater Patientenbericht in einfacher, erklärender Sprache (ohne Abkürzungen/Zahlen)

Hinweis: Dieses Tool ist als Assistenzsystem gedacht und ersetzt keine ärztliche Beurteilung.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
import datetime as _dt
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# =============================================================================
# App Meta
# =============================================================================

APP_NAME = "RHK Befundassistent"
APP_VERSION = "v22.0"
APP_TITLE = f"{APP_NAME} – {APP_VERSION}"

APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RULEBOOK_PATH = os.environ.get("RHK_RULEBOOK", os.path.join(APP_DIR, "rhk_rules_v22.yaml"))

# Clinical constants
TAPSE_SPAP_CUTOFF = 0.31  # TAPSE/sPAP cut-off for RV-PA uncoupling (Tello et al.)


# =============================================================================
# Formatting helpers
# =============================================================================

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            if math.isnan(v):
                return None
            return v
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        if not s:
            return None
        try:
            v = float(s)
            if math.isnan(v):
                return None
            return v
        except Exception:
            return None
    return None


def _fmt(x: Any, nd: int = 1) -> str:
    v = _safe_float(x)
    if v is None:
        return "–"
    # Avoid showing "-0.0"
    if abs(v) < 10 ** (-(nd + 1)):
        v = 0.0
    return f"{v:.{nd}f}".replace(".", ",")


def _fmt_int(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "–"
    return str(int(round(v)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# Simple physiologic calculations
# =============================================================================

def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if not height_cm or not weight_kg:
        return None
    h_m = height_cm / 100.0
    if h_m <= 0:
        return None
    return weight_kg / (h_m * h_m)


def calc_bsa(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    # Mosteller
    if not height_cm or not weight_kg:
        return None
    if height_cm <= 0 or weight_kg <= 0:
        return None
    return math.sqrt((height_cm * weight_kg) / 3600.0)


def calc_mpap_from_spap_dpap(spap: Optional[float], dpap: Optional[float]) -> Optional[float]:
    if spap is None or dpap is None:
        return None
    return (spap + 2.0 * dpap) / 3.0


# =============================================================================
# H2FPEF – continuous model (AHA Circulation 2018) (user provided)
# =============================================================================

@dataclass
class H2FPEFResult:
    percent: Optional[float]
    category: Optional[str]
    y: Optional[float] = None
    z: Optional[float] = None
    inputs_used: Dict[str, Any] = field(default_factory=dict)


def calc_h2fpef_probability(age: Optional[float],
                           bmi: Optional[float],
                           ee: Optional[float],
                           pasp: Optional[float],
                           af: Optional[bool]) -> H2FPEFResult:
    """
    Probability of heart failure with preserved EF (H2FPEF) using the continuous model:

    Probability = (Z / (1 + Z)) * 100, where Z = e^y and
    y = -9.1917 + 0.0451*age + 0.1307*BMI + 0.0859*(E/e') + 0.0520*PASP + 1.6997*AF
    AF: 1 if Yes else 0
    BMI capped at 50 to avoid extrapolation (user provided note).
    """
    inputs_used = {
        "age": age,
        "bmi": bmi,
        "e_over_eprime": ee,
        "pasp": pasp,
        "af": bool(af) if af is not None else None,
    }

    if age is None or bmi is None or ee is None or pasp is None or af is None:
        return H2FPEFResult(percent=None, category=None, inputs_used=inputs_used)

    bmi_c = _clamp(float(bmi), 10.0, 50.0)

    y = -9.1917 + 0.0451 * float(age) + 0.1307 * bmi_c + 0.0859 * float(ee) + 0.0520 * float(pasp) + 1.6997 * (1.0 if af else 0.0)
    z = math.exp(y)
    prob = (z / (1.0 + z)) * 100.0

    if prob < 20:
        cat = "unlikely"
    elif prob < 60:
        cat = "possible"
    else:
        cat = "likely"

    return H2FPEFResult(percent=prob, category=cat, y=y, z=z, inputs_used={**inputs_used, "bmi_capped": bmi_c})


# =============================================================================
# ESC/ERS risk (simple heuristic; not a full guideline calculator)
# =============================================================================

def _risk_bucket_from_points(points: List[Optional[int]], mode: str) -> Optional[str]:
    # mode: "3" or "4"
    pts = [p for p in points if isinstance(p, int)]
    if not pts:
        return None
    m = sum(pts) / len(pts)

    if mode == "3":
        # low, intermediate, high
        if m <= 1.5:
            return "low"
        if m <= 2.5:
            return "intermediate"
        return "high"

    # 4 strata
    if m <= 1.5:
        return "low"
    if m <= 2.0:
        return "intermediate-low"
    if m <= 2.5:
        return "intermediate-high"
    return "high"


def calc_esc_ers_4_strata(who_fc: Optional[str],
                          six_mwd_m: Optional[float],
                          bnp_pg_ml: Optional[float],
                          ntprobnp_pg_ml: Optional[float]) -> Optional[str]:
    """
    Extremely simplified point-based approximation:
    WHO-FC: I/II=1, III=2, IV=3
    6MWD: >440=1, 165-440=2, <165=3 (rough)
    BNP/NTproBNP: low=1, mid=2, high=3 (very rough)
    Then map to 4-strata via mean.
    """
    pts: List[Optional[int]] = []

    if who_fc:
        if who_fc.strip().upper() in ("I", "1"):
            pts.append(1)
        elif who_fc.strip().upper() in ("II", "2"):
            pts.append(1)
        elif who_fc.strip().upper() in ("III", "3"):
            pts.append(2)
        elif who_fc.strip().upper() in ("IV", "4"):
            pts.append(3)

    if six_mwd_m is not None:
        if six_mwd_m > 440:
            pts.append(1)
        elif six_mwd_m >= 165:
            pts.append(2)
        else:
            pts.append(3)

    # Use NT-proBNP if available, else BNP
    biom = ntprobnp_pg_ml if ntprobnp_pg_ml is not None else bnp_pg_ml
    if biom is not None:
        if biom < 300:
            pts.append(1)
        elif biom < 1400:
            pts.append(2)
        else:
            pts.append(3)

    return _risk_bucket_from_points(pts, mode="4")


def calc_esc_ers_3_strata(who_fc: Optional[str],
                          six_mwd_m: Optional[float],
                          bnp_pg_ml: Optional[float],
                          ntprobnp_pg_ml: Optional[float]) -> Optional[str]:
    pts: List[Optional[int]] = []

    if who_fc:
        if who_fc.strip().upper() in ("I", "1", "II", "2"):
            pts.append(1)
        elif who_fc.strip().upper() in ("III", "3"):
            pts.append(2)
        elif who_fc.strip().upper() in ("IV", "4"):
            pts.append(3)

    if six_mwd_m is not None:
        if six_mwd_m > 440:
            pts.append(1)
        elif six_mwd_m >= 165:
            pts.append(2)
        else:
            pts.append(3)

    biom = ntprobnp_pg_ml if ntprobnp_pg_ml is not None else bnp_pg_ml
    if biom is not None:
        if biom < 300:
            pts.append(1)
        elif biom < 1400:
            pts.append(2)
        else:
            pts.append(3)

    return _risk_bucket_from_points(pts, mode="3")


# =============================================================================
# Exercise pattern heuristics
# =============================================================================

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Additional risk scores (ESC/ERS comprehensive 3-strata; REVEAL Lite 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EscErsComprehensiveResult:
    category: str
    mean_grade: float
    n_params: int
    grades: Dict[str, int]
    missing: List[str]


def calc_esc_ers_comprehensive_3_strata(ui: Dict[str, Any], derived: Dict[str, Any]) -> Optional[EscErsComprehensiveResult]:
    """ESC/ERS 2022 comprehensive risk assessment (Table 16) – vereinfachte Umsetzung.

    Jede verfügbare Variable wird in drei Kategorien eingestuft:
      1 = niedriges Risiko
      2 = intermediäres Risiko
      3 = hohes Risiko

    Anschließend wird der Mittelwert gebildet und in (niedrig / intermediär / hoch) gemappt.

    Hinweis: Es werden nur Parameter bewertet, die tatsächlich vorliegen.
    """
    grades: Dict[str, int] = {}
    missing: List[str] = []

    # WHO-FC
    who = (ui.get("who_fc") or "").strip()
    if who in ("I", "II"):
        grades["WHO-FC"] = 1
    elif who == "III":
        grades["WHO-FC"] = 2
    elif who == "IV":
        grades["WHO-FC"] = 3
    else:
        missing.append("WHO-FC")

    # 6MWD
    six = _safe_float(ui.get("six_mwd_m"))
    if isinstance(six, (int, float)):
        if six > 440:
            grades["6MWD"] = 1
        elif six >= 165:
            grades["6MWD"] = 2
        else:
            grades["6MWD"] = 3
    else:
        missing.append("6MWD")

    # BNP / NT-proBNP (aus bnp_kind + bnp_value)
    bnp_kind = (ui.get("bnp_kind") or "").strip()
    bnp_val = _safe_float(ui.get("bnp_value"))
    bnp = None
    ntp = None
    if isinstance(bnp_val, (int, float)) and bnp_val > 0:
        if bnp_kind.upper().startswith("BNP"):
            bnp = bnp_val
        elif "NT" in bnp_kind.upper():
            ntp = bnp_val

    if bnp is not None:
        if bnp < 50:
            grades["BNP"] = 1
        elif bnp <= 300:
            grades["BNP"] = 2
        else:
            grades["BNP"] = 3
    elif ntp is not None:
        if ntp < 300:
            grades["NT-proBNP"] = 1
        elif ntp <= 1400:
            grades["NT-proBNP"] = 2
        else:
            grades["NT-proBNP"] = 3
    else:
        missing.append("BNP/NT-proBNP")

    # Synkope (keine / gelegentlich / wiederholt)
    syn = ui.get("syncope")
    if isinstance(syn, bool):
        grades["Synkope"] = 3 if syn else 1
    else:
        syn_s = (syn or "").strip().lower()
        if syn_s in ("", "keine", "nein", "0", "none"):
            grades["Synkope"] = 1
        elif syn_s in ("gelegentlich", "selten", "occasional"):
            grades["Synkope"] = 2
        elif syn_s in ("wiederholt", "häufig", "repeated"):
            grades["Synkope"] = 3
        else:
            missing.append("Synkope")

    # Echokardiographie: RA-ESA & Perikarderguss
    ra = _safe_float(ui.get("ra_esa_cm2"))
    if isinstance(ra, (int, float)) and ra > 0:
        if ra < 18:
            grades["RA-ESA"] = 1
        elif ra <= 26:
            grades["RA-ESA"] = 2
        else:
            grades["RA-ESA"] = 3
    else:
        missing.append("RA-ESA")

    pe = (ui.get("pericardial_effusion") or "").strip().lower()
    if pe in ("kein", "nein", "no", "none"):
        grades["Perikarderguss"] = 1
    elif pe in ("minimal", "klein"):
        grades["Perikarderguss"] = 2
    elif pe in ("relevant", "moderat", "gross", "groß", "großzügig"):
        grades["Perikarderguss"] = 3
    else:
        missing.append("Perikarderguss")

    # Hämodynamik: RAP, CI, SvO2
    rap = _safe_float(ui.get("rap_rest"))
    if isinstance(rap, (int, float)):
        if rap < 8:
            grades["RAP"] = 1
        elif rap <= 14:
            grades["RAP"] = 2
        else:
            grades["RAP"] = 3
    else:
        missing.append("RAP")

    ci = derived.get("ci")
    if not isinstance(ci, (int, float)) or ci <= 0:
        ci = _safe_float(ui.get("ci_rest"))
    if isinstance(ci, (int, float)) and ci > 0:
        if ci >= 2.5:
            grades["CI"] = 1
        elif ci >= 2.0:
            grades["CI"] = 2
        else:
            grades["CI"] = 3
    else:
        missing.append("CI")

    svo2 = _safe_float(ui.get("sat_pa"))
    if isinstance(svo2, (int, float)) and svo2 > 0:
        if svo2 > 65:
            grades["SvO2"] = 1
        elif svo2 >= 60:
            grades["SvO2"] = 2
        else:
            grades["SvO2"] = 3
    else:
        missing.append("SvO2")

    # TAPSE/sPAP ratio (aus derived)
    tsp = derived.get("tapse_spap")
    if isinstance(tsp, (int, float)) and tsp > 0:
        if tsp > 0.32:
            grades["TAPSE/sPAP"] = 1
        elif tsp >= 0.19:
            grades["TAPSE/sPAP"] = 2
        else:
            grades["TAPSE/sPAP"] = 3
    else:
        missing.append("TAPSE/sPAP")

    # CMR: RVEF
    rvef = _safe_float(ui.get("cmr_rvef"))
    if isinstance(rvef, (int, float)) and rvef > 0:
        if rvef > 54:
            grades["CMR-RVEF"] = 1
        elif rvef >= 37:
            grades["CMR-RVEF"] = 2
        else:
            grades["CMR-RVEF"] = 3
    else:
        missing.append("CMR-RVEF")

    if not grades:
        return None

    mean_grade = sum(grades.values()) / max(len(grades), 1)
    if mean_grade <= 1.5:
        cat = "niedrig"
    elif mean_grade <= 2.5:
        cat = "intermediär"
    else:
        cat = "hoch"

    return EscErsComprehensiveResult(
        category=cat,
        mean_grade=mean_grade,
        n_params=len(grades),
        grades=grades,
        missing=missing,
    )


@dataclass(frozen=True)
class RevealLite2Result:
    points: int
    category: str
    details: Dict[str, int]
    missing: List[str]


def calc_reveal_lite2(ui: Dict[str, Any]) -> Optional[RevealLite2Result]:
    """REVEAL Lite 2 – vereinfachte Implementierung (6 Parameter).

    Benötigt:
      - WHO-FC
      - 6MWD
      - BNP oder NT-proBNP (Angabe über 'bnp_kind' + 'bnp_value')
      - systolischer Blutdruck (bp_sys)
      - Herzfrequenz (hr)
      - eGFR (egfr)

    Rückgabe:
      - Punkte (Summe)
      - Risikokategorie (niedrig / intermediär / hoch)
    """
    missing: List[str] = []
    details: Dict[str, int] = {}

    who = (ui.get("who_fc") or "").strip()
    if who in ("I", "II"):
        details["WHO-FC"] = 0
    elif who == "III":
        details["WHO-FC"] = 1
    elif who == "IV":
        details["WHO-FC"] = 2
    else:
        missing.append("WHO-FC")

    six = _safe_float(ui.get("six_mwd_m"))
    if isinstance(six, (int, float)) and six > 0:
        if six >= 440:
            details["6MWD"] = 0
        elif six >= 320:
            details["6MWD"] = 1
        elif six >= 165:
            details["6MWD"] = 2
        else:
            details["6MWD"] = 3
    else:
        missing.append("6MWD")

    bnp_kind = (ui.get("bnp_kind") or "").strip()
    bnp_val = _safe_float(ui.get("bnp_value"))
    if not (isinstance(bnp_val, (int, float)) and bnp_val > 0 and bnp_kind):
        missing.append("BNP/NT-proBNP")
    else:
        if bnp_kind.upper().startswith("BNP"):
            if bnp_val < 50:
                details["BNP/NT-proBNP"] = 0
            elif bnp_val <= 199:
                details["BNP/NT-proBNP"] = 1
            elif bnp_val <= 800:
                details["BNP/NT-proBNP"] = 2
            else:
                details["BNP/NT-proBNP"] = 3
        else:
            # treat all non-BNP as NT-proBNP
            if bnp_val < 300:
                details["BNP/NT-proBNP"] = 0
            elif bnp_val <= 649:
                details["BNP/NT-proBNP"] = 1
            elif bnp_val <= 1100:
                details["BNP/NT-proBNP"] = 2
            else:
                details["BNP/NT-proBNP"] = 3

    sbp = _safe_float(ui.get("bp_sys"))
    if isinstance(sbp, (int, float)) and sbp > 0:
        if sbp >= 110:
            details["RRsys"] = 0
        elif sbp >= 100:
            details["RRsys"] = 1
        elif sbp >= 90:
            details["RRsys"] = 2
        else:
            details["RRsys"] = 3
    else:
        missing.append("RRsys")

    hr = _safe_float(ui.get("hr"))
    if isinstance(hr, (int, float)) and hr > 0:
        if hr < 96:
            details["HF"] = 0
        elif hr <= 105:
            details["HF"] = 1
        elif hr <= 115:
            details["HF"] = 2
        else:
            details["HF"] = 3
    else:
        missing.append("HF")

    egfr = _safe_float(ui.get("egfr"))
    if isinstance(egfr, (int, float)) and egfr > 0:
        if egfr >= 60:
            details["eGFR"] = 0
        elif egfr >= 45:
            details["eGFR"] = 1
        elif egfr >= 30:
            details["eGFR"] = 2
        else:
            details["eGFR"] = 3
    else:
        missing.append("eGFR")

    if missing:
        return None

    points = int(sum(details.values()))
    if points <= 5:
        cat = "niedrig"
    elif points <= 7:
        cat = "intermediär"
    else:
        cat = "hoch"

    return RevealLite2Result(points=points, category=cat, details=details, missing=[])

def classify_exercise_pattern(mpap_co_slope: Optional[float], pawp_co_slope: Optional[float]) -> Optional[str]:
    """
    Heuristic:
    - mpap/CO slope > 3 WU suggests abnormal pulmonary pressure response
    - PAWP/CO slope > 2 WU suggests left-heart filling pressure component
    """
    if mpap_co_slope is None or pawp_co_slope is None:
        return None
    if mpap_co_slope > 3 and pawp_co_slope <= 2:
        return "precap_pattern"
    if mpap_co_slope > 3 and pawp_co_slope > 2:
        return "postcap_pattern"
    if mpap_co_slope <= 3 and pawp_co_slope > 2:
        return "left_pressure_pattern"
    return "normal_pattern"


# =============================================================================
# Step oximetry – step-up detection
# =============================================================================

@dataclass
class StepUpResult:
    present: bool
    from_to: Optional[str] = None
    location: Optional[str] = None  # "atrial"|"ventricular"|"pulmonary"
    delta: Optional[float] = None
    sentence: Optional[str] = None


def detect_step_up(sat_svc: Optional[float],
                   sat_ivc: Optional[float],
                   sat_ra: Optional[float],
                   sat_rv: Optional[float],
                   sat_pa: Optional[float],
                   sat_ao: Optional[float],
                   thr_atrial: float = 7.0,
                   thr_ventricular: float = 5.0,
                   thr_pulmonary: float = 5.0) -> StepUpResult:
    """
    Very practical step-up detection using typical thresholds:
    - Atrial: RA - mean(SVC/IVC) >= ~7%
    - Ventricular: RV - RA >= ~5%
    - Pulmonary artery: PA - RV >= ~5%

    Returns the most likely location of a relevant step-up, if present.
    """
    # Normalize
    def _v(x):
        x = _safe_float(x)
        return None if x is None else float(x)

    svc = _v(sat_svc)
    ivc = _v(sat_ivc)
    ra = _v(sat_ra)
    rv = _v(sat_rv)
    pa = _v(sat_pa)
    ao = _v(sat_ao)

    venous_vals = [v for v in (svc, ivc) if v is not None]
    venous_ref = None
    if len(venous_vals) == 2:
        venous_ref = sum(venous_vals) / 2.0
    elif len(venous_vals) == 1:
        venous_ref = venous_vals[0]

    candidates: List[Tuple[str, float, str]] = []  # (from_to, delta, location)

    if venous_ref is not None and ra is not None:
        d = ra - venous_ref
        if d >= thr_atrial:
            candidates.append(("SVC/IVC → RA", d, "atrial"))

    if ra is not None and rv is not None:
        d = rv - ra
        if d >= thr_ventricular:
            candidates.append(("RA → RV", d, "ventricular"))

    if rv is not None and pa is not None:
        d = pa - rv
        if d >= thr_pulmonary:
            candidates.append(("RV → PA", d, "pulmonary"))

    if not candidates:
        return StepUpResult(
            present=False,
            sentence="Kein relevanter Sättigungssprung in der Stufenoxymetrie."
        )

    # pick the largest delta
    best = sorted(candidates, key=lambda t: t[1], reverse=True)[0]
    from_to, delta, loc = best

    loc_desc = {
        "atrial": "auf Vorhofebene",
        "ventricular": "auf Ventrikelebene",
        "pulmonary": "auf Pulmonalarterienebene",
    }.get(loc, "unklar")

    sentence = f"Relevanter Sättigungssprung {loc_desc} ({from_to}, Δ≈{_fmt(delta,1)}%). Shuntverdacht (Links-Rechts)."
    return StepUpResult(present=True, from_to=from_to, location=loc, delta=delta, sentence=sentence)


# =============================================================================
# TextDB loading (rhk_textdb.py)
# =============================================================================

@dataclass
class TextBlock:
    id: str
    title: str
    template: str
    kind: str  # "bundle"|"module"


class SafeDict(dict):
    """dict for str.format_map that returns an empty string for missing keys."""

    def __missing__(self, key: str) -> str:
        return ""


def load_textdb_blocks() -> Dict[str, TextBlock]:
    """
    Loads rhk_textdb.py if present.

    Expected in rhk_textdb: BLOCKS (dict[str, TextBlock-like]).
    The upstream rhk_textdb ships a TextBlock dataclass with attributes .id/.title/.template.
    We normalize into this app's local TextBlock to keep the rest of the code stable.

    Fallback: minimal built-in blocks.
    """
    import sys

    blocks: Dict[str, TextBlock] = {}

    # Ensure the directory of this script is importable (matches the v18 deployment pattern)
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    try:
        import rhk_textdb  # type: ignore

        src = getattr(rhk_textdb, "ALL_BLOCKS", None)
        if isinstance(src, dict):
            for bid, b in src.items():
                bid_s = str(bid)

                # title/template may be attributes (dataclass) or dict keys
                title = getattr(b, "title", None)
                template = getattr(b, "template", None)
                if title is None and isinstance(b, dict):
                    title = b.get("title")
                if template is None and isinstance(b, dict):
                    template = b.get("template")

                kind = "module" if (bid_s.startswith("P") and "_" not in bid_s) else "bundle"
                blocks[bid_s] = TextBlock(
                    id=bid_s,
                    title=str(title or bid_s),
                    template=str(template or ""),
                    kind=kind,
                )

        # Safety: ensure at least minimal core blocks exist
        if "K00_B" not in blocks:
            blocks["K00_B"] = TextBlock(
                id="K00_B", title="Kein Hinweis auf PH", kind="bundle",
                template="Kein Hinweis auf eine pulmonale Hypertonie in den vorliegenden Daten."
            )
        if "K00_E" not in blocks:
            blocks["K00_E"] = TextBlock(
                id="K00_E", title="Empfehlung", kind="bundle",
                template="Kontrolle nach klinischer Indikation. Bei persistierendem Verdacht weitere Diagnostik."
            )
        if "P01" not in blocks:
            blocks["P01"] = TextBlock(
                id="P01", title="Basisdiagnostik komplettieren", kind="module",
                template="• Echokardiographie\n• Lungenfunktion\n• Bildgebung/V/Q\n• Labor inkl. BNP/NT-proBNP"
            )

        return blocks

    except Exception:
        # Minimal fallback
        blocks["K00_B"] = TextBlock(
            id="K00_B", title="Kein Hinweis auf PH", kind="bundle",
            template="Kein Hinweis auf eine pulmonale Hypertonie in den vorliegenden Daten."
        )
        blocks["K00_E"] = TextBlock(
            id="K00_E", title="Empfehlung", kind="bundle",
            template="Kontrolle nach klinischer Indikation. Bei persistierendem Verdacht weitere Diagnostik."
        )
        blocks["P01"] = TextBlock(
            id="P01", title="Basisdiagnostik komplettieren", kind="module",
            template="• Echokardiographie\n• Lungenfunktion\n• Bildgebung/V/Q\n• Labor inkl. BNP/NT-proBNP"
        )
        return blocks


# =============================================================================
# Rendering helpers
# =============================================================================

def render_block(block: TextBlock, ctx: Dict[str, Any]) -> str:
    try:
        return block.template.format_map(SafeDict(ctx))
    except Exception as e:
        return f"[Template-Fehler in {block.id}: {e}]"


# =============================================================================
# Declarative Rule Engine
# =============================================================================

@dataclass
class Rule:
    id: str
    when: str
    then: Dict[str, Any]
    priority: int = 100


@dataclass
class Decision:
    bundle: str = "K00"
    primary_dx: str = "—"
    modules: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    require_fields: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    leading_cause: Optional[str] = None
    leading_action: Optional[str] = None


# --- Safe boolean expression evaluator (no builtins, no calls) ----------------

class SafeExprError(Exception):
    pass


_ALLOWED_NODES = (
    "Expression", "BoolOp", "BinOp", "UnaryOp", "Compare", "Name", "Load",
    "Constant", "And", "Or", "Not", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
    "Is", "IsNot", "In", "NotIn",
)


def safe_eval_bool(expr: str, env: Dict[str, Any]) -> bool:
    import ast

    if not expr or not expr.strip():
        return False

    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if node.__class__.__name__ not in _ALLOWED_NODES:
            raise SafeExprError(f"Node not allowed: {node.__class__.__name__}")

    class _Eval(ast.NodeVisitor):
        def visit_Expression(self, node):  # type: ignore
            return self.visit(node.body)

        def visit_Name(self, node):  # type: ignore
            return env.get(node.id)

        def visit_Constant(self, node):  # type: ignore
            return node.value

        def visit_BoolOp(self, node):  # type: ignore
            if isinstance(node.op, ast.And):
                return all(self.visit(v) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self.visit(v) for v in node.values)
            raise SafeExprError("Unsupported BoolOp")

        def visit_UnaryOp(self, node):  # type: ignore
            if isinstance(node.op, ast.Not):
                return not bool(self.visit(node.operand))
            raise SafeExprError("Unsupported UnaryOp")

        def visit_Compare(self, node):  # type: ignore
            left = self.visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self.visit(comp)
                ok = None
                try:
                    if isinstance(op, ast.Eq):
                        ok = (left == right)
                    elif isinstance(op, ast.NotEq):
                        ok = (left != right)
                    elif isinstance(op, ast.Lt):
                        ok = (left is not None and right is not None and left < right)
                    elif isinstance(op, ast.LtE):
                        ok = (left is not None and right is not None and left <= right)
                    elif isinstance(op, ast.Gt):
                        ok = (left is not None and right is not None and left > right)
                    elif isinstance(op, ast.GtE):
                        ok = (left is not None and right is not None and left >= right)
                    elif isinstance(op, ast.Is):
                        ok = (left is right)
                    elif isinstance(op, ast.IsNot):
                        ok = (left is not right)
                    elif isinstance(op, ast.In):
                        ok = (left in right) if right is not None else False
                    elif isinstance(op, ast.NotIn):
                        ok = (left not in right) if right is not None else False
                    else:
                        raise SafeExprError("Unsupported Compare op")
                except Exception:
                    ok = False
                if not ok:
                    return False
                left = right
            return True

    return bool(_Eval().visit(tree))


def load_rulebook(path: str) -> List[Rule]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please add pyyaml to requirements.")
    if not os.path.exists(path):
        # Empty rulebook fallback
        return []

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    rules: List[Rule] = []
    for r in doc.get("rules", []):
        rules.append(
            Rule(
                id=str(r.get("id")),
                when=str(r.get("when")),
                then=dict(r.get("then") or {}),
                priority=int(r.get("priority", 100)),
            )
        )

    rules.sort(key=lambda rr: rr.priority)
    return rules


def apply_rule_engine(env: Dict[str, Any], rules: List[Rule]) -> Decision:
    d = Decision(bundle="K00", primary_dx="Kein Hinweis auf PH")
    for rule in rules:
        try:
            if safe_eval_bool(rule.when, env):
                then = rule.then or {}

                if "set_bundle" in then:
                    d.bundle = str(then["set_bundle"])
                if "set_primary_dx" in then:
                    d.primary_dx = str(then["set_primary_dx"])
                if "set_leading_cause" in then:
                    d.leading_cause = str(then["set_leading_cause"])
                if "set_leading_action" in then:
                    d.leading_action = str(then["set_leading_action"])

                if "add_modules" in then:
                    for m in then.get("add_modules") or []:
                        if m not in d.modules:
                            d.modules.append(m)

                if "add_recommendations" in then:
                    for rec in then.get("add_recommendations") or []:
                        if rec and rec not in d.recommendations:
                            d.recommendations.append(str(rec))

                if "require_fields" in then:
                    for fld in then.get("require_fields") or []:
                        if fld not in d.require_fields:
                            d.require_fields.append(str(fld))

                if "add_tags" in then:
                    for t in then.get("add_tags") or []:
                        if t and t not in d.tags:
                            d.tags.append(str(t))

        except Exception:
            # Never crash on rule evaluation.
            continue
    return d


# =============================================================================
# Case building (derived values + env for rule engine)
# =============================================================================

def _infer_anemia(sex: Optional[str], hb_g_dl: Optional[float]) -> bool:
    if hb_g_dl is None:
        return False
    s = (sex or "").lower()
    # pragmatic thresholds
    if "männ" in s:
        return hb_g_dl < 13.0
    if "weib" in s:
        return hb_g_dl < 12.0
    return hb_g_dl < 12.5


def _hemo_category(mpap: Optional[float], pawp: Optional[float], pvr: Optional[float]) -> str:
    if mpap is None:
        return "unknown"
    if mpap <= 20:
        return "no_ph"
    if pawp is None or pvr is None:
        return "ph_unclassified"
    if pawp <= 15 and pvr > 2:
        return "precap"
    if pawp > 15 and pvr <= 2:
        return "ipcph"
    if pawp > 15 and pvr > 2:
        return "cpcph"
    # unusual combos
    if pawp <= 15 and pvr <= 2:
        return "high_flow_or_borderline"
    return "ph_unclassified"


def build_case(ui: Dict[str, Any], rules: List[Rule]) -> Dict[str, Any]:
    # ---- basic anthropometrics ----
    height_cm = _safe_float(ui.get("height_cm"))
    weight_kg = _safe_float(ui.get("weight_kg"))
    age = _safe_float(ui.get("age"))
    sex = ui.get("sex")

    bsa = calc_bsa(height_cm, weight_kg)
    bmi = calc_bmi(height_cm, weight_kg)

    # ---- Rest hemodynamics (allow optional direct entries) ----
    spap = _safe_float(ui.get("spap_rest"))
    dpap = _safe_float(ui.get("dpap_rest"))
    mpap_in = _safe_float(ui.get("mpap_rest"))  # optional
    pawp = _safe_float(ui.get("pawp_rest"))
    rap = _safe_float(ui.get("rap_rest"))

    co_in = _safe_float(ui.get("co_rest"))
    ci_in = _safe_float(ui.get("ci_rest"))  # optional

    mpap_calc = calc_mpap_from_spap_dpap(spap, dpap)
    mpap = mpap_in if mpap_in is not None else mpap_calc

    # CO/CI consistency: prefer CO; else infer CO from CI
    co = co_in
    if co is None and ci_in is not None and bsa is not None:
        co = ci_in * bsa

    ci = ci_in
    if ci is None and co is not None and bsa is not None:
        ci = co / bsa

    # PVR optional input
    pvr_in = _safe_float(ui.get("pvr_rest"))
    pvr_calc = None
    if mpap is not None and pawp is not None and co is not None and co > 0:
        pvr_calc = (mpap - pawp) / co
    pvr = pvr_in if pvr_in is not None else pvr_calc

    # Indexed PVR
    pvri = None
    if mpap is not None and pawp is not None and ci is not None and ci > 0:
        pvri = (mpap - pawp) / ci

    tpg = (mpap - pawp) if (mpap is not None and pawp is not None) else None
    dpg = (dpap - pawp) if (dpap is not None and pawp is not None) else None

    # ---- Exercise hemodynamics (optional) ----
    spap_pk = _safe_float(ui.get("spap_peak"))
    dpap_pk = _safe_float(ui.get("dpap_peak"))
    mpap_pk_in = _safe_float(ui.get("mpap_peak"))
    mpap_pk_calc = calc_mpap_from_spap_dpap(spap_pk, dpap_pk)
    mpap_peak = mpap_pk_in if mpap_pk_in is not None else mpap_pk_calc

    pawp_peak = _safe_float(ui.get("pawp_peak"))
    co_peak = _safe_float(ui.get("co_peak"))
    ci_peak_in = _safe_float(ui.get("ci_peak"))
    # Optional: allow entering CI directly (if CO not documented).
    if co_peak is None and ci_peak_in is not None and bsa is not None:
        co_peak = ci_peak_in * bsa
    ci_peak = None
    if ci_peak_in is not None:
        ci_peak = ci_peak_in
    elif co_peak is not None and bsa is not None:
        ci_peak = co_peak / bsa

    exercise_done = bool(ui.get("exercise_done")) or (co_peak is not None or mpap_peak is not None or pawp_peak is not None)

    mpap_co_slope = None
    pawp_co_slope = None
    if exercise_done and co is not None and co_peak is not None and co_peak != co:
        if mpap is not None and mpap_peak is not None:
            mpap_co_slope = (mpap_peak - mpap) / (co_peak - co)
        if pawp is not None and pawp_peak is not None:
            pawp_co_slope = (pawp_peak - pawp) / (co_peak - co)

    exercise_pattern = classify_exercise_pattern(mpap_co_slope, pawp_co_slope) if exercise_done else None

    delta_spap = (spap_pk - spap) if (spap is not None and spap_pk is not None) else None

    # Adaptation type (user rule)
    adaptation_type = None
    if exercise_done and delta_spap is not None and ci is not None and ci_peak is not None:
        if delta_spap > 30 and ci_peak >= ci:
            adaptation_type = "homeometric"
        elif delta_spap > 30 and ci_peak < ci:
            adaptation_type = "heterometric"

    # ---- Volume challenge ----
    volume_done = bool(ui.get("volume_challenge_done"))
    pawp_pre = _safe_float(ui.get("pawp_pre"))
    pawp_post = _safe_float(ui.get("pawp_post"))
    mpap_pre = _safe_float(ui.get("mpap_pre"))
    mpap_post = _safe_float(ui.get("mpap_post"))
    pawp_delta = (pawp_post - pawp_pre) if (pawp_pre is not None and pawp_post is not None) else None
    mpap_delta = (mpap_post - mpap_pre) if (mpap_pre is not None and mpap_post is not None) else None

    # ---- Vasoreactivity ----
    vaso_done = bool(ui.get("vaso_test_done"))
    vaso_response = ui.get("vaso_response_desc") or None

    # ---- Echo / imaging context ----
    lvef = _safe_float(ui.get("lvef"))
    la_enlarged = bool(ui.get("la_enlarged"))
    ee_ratio = _safe_float(ui.get("ee_ratio"))
    pasp_echo = _safe_float(ui.get("pasp_echo"))
    af = bool(ui.get("atrial_fib")) if ui.get("atrial_fib") is not None else None

    # IVC congestion proxy – categorical collapse yes/no
    ivc_diam = _safe_float(ui.get("ivc_diam_mm"))
    ivc_collapse = ui.get("ivc_collapse")  # "ja"/"nein"/None
    ivc_collapse_yes = True if (isinstance(ivc_collapse, str) and ivc_collapse.lower().startswith("ja")) else False if (isinstance(ivc_collapse, str) and ivc_collapse.lower().startswith("nein")) else None

    congestion_likely = False
    # Practical congestion heuristics:
    # - Elevated RAP supports congestion
    # - Absent IVC collapse (categorical "nein") is treated as a sign of venous congestion
    if rap is not None and rap >= 12:
        congestion_likely = True
    if ivc_collapse_yes is False:
        congestion_likely = True
    if ivc_diam is not None and ivc_diam >= 21 and ivc_collapse_yes is False:
        congestion_likely = True

    # ---- Step oximetry ----
    sat_svc = _safe_float(ui.get("sat_svc"))
    sat_ivc = _safe_float(ui.get("sat_ivc"))
    sat_ra = _safe_float(ui.get("sat_ra"))
    sat_rv = _safe_float(ui.get("sat_rv"))
    sat_pa = _safe_float(ui.get("sat_pa"))
    sat_ao = _safe_float(ui.get("sat_ao"))
    stepup = detect_step_up(sat_svc, sat_ivc, sat_ra, sat_rv, sat_pa, sat_ao)

    # ---- Curve morphology flags ----
    wedge_v_wave = bool(ui.get("wedge_v_wave"))
    wedge_a_wave = bool(ui.get("wedge_a_wave"))
    rap_a_wave = bool(ui.get("rap_a_wave"))
    rap_v_wave = bool(ui.get("rap_v_wave"))
    rv_pseudo_dip = bool(ui.get("rv_pseudo_dip"))
    rv_dip_plateau = bool(ui.get("rv_dip_plateau"))

    # ---- S'/RAAI ----
    s_prime = _safe_float(ui.get("s_prime_cm_s"))
    ra_esa_cm2 = _safe_float(ui.get("ra_esa_cm2"))
    raai = None
    sprime_raai = None
    if ra_esa_cm2 is not None and bsa is not None and bsa > 0:
        raai = ra_esa_cm2 / bsa
    if s_prime is not None and raai is not None and raai > 0:
        sprime_raai = s_prime / raai

    # Cut-off nach Yogeswaran et al.: S'/RAAi < 0.81 m²/(s·cm) als Warnsignal
    SPRIME_RAAI_CUTOFF = 0.81
    sprime_raai_low = None
    if sprime_raai is not None:
        sprime_raai_low = sprime_raai < SPRIME_RAAI_CUTOFF

    # ---- TAPSE/sPAP (Tello et al.) ----
    tapse = _safe_float(ui.get("tapse_mm"))
    tapse_spap = None
    tapse_spap_uncoupled = None  # Tello et al. (vereinfachter Cut-off)
    tapse_spap_risk = None       # ESC/ERS 2022 (Table 16) – 3-Strata Einordnung
    if tapse is not None and pasp_echo is not None and pasp_echo > 0:
        tapse_spap = tapse / pasp_echo
        tapse_spap_uncoupled = tapse_spap < TAPSE_SPAP_CUTOFF
        # ESC/ERS comprehensive risk table uses >0.32 (low), 0.19–0.32 (intermediate), <0.19 (high)
        if tapse_spap > 0.32:
            tapse_spap_risk = "niedrig"
        elif tapse_spap >= 0.19:
            tapse_spap_risk = "intermediär"
        else:
            tapse_spap_risk = "hoch"

    # ---- Labs ----
    hb = _safe_float(ui.get("hb_g_dl"))
    anemia = _infer_anemia(sex, hb)

    # BNP / NT-proBNP
    bnp_kind = ui.get("bnp_kind") or None
    bnp_val = _safe_float(ui.get("bnp_value"))
    bnp_pg_ml = None
    ntprobnp_pg_ml = None
    if bnp_val is not None:
        if isinstance(bnp_kind, str) and "NT" in bnp_kind.upper():
            ntprobnp_pg_ml = bnp_val
        else:
            bnp_pg_ml = bnp_val

    # ---- HFpEF probability (H2FPEF) ----
    hfpef_res = calc_h2fpef_probability(
        age=age,
        bmi=bmi,
        ee=ee_ratio,
        pasp=pasp_echo,
        af=af,
    )

    hemo_cat = _hemo_category(mpap, pawp, pvr)

    derived: Dict[str, Any] = {
        "bsa_m2": bsa,
        "bmi": bmi,
        "mpap": mpap,
        "mpap_calc": mpap_calc,
        "tpg": tpg,
        "dpg": dpg,
        "co": co,
        "ci": ci,
        "pvr": pvr,
        "pvr_calc": pvr_calc,
        "pvri": pvri,
        "hemo_category": hemo_cat,
        "exercise_done": exercise_done,
        "mpap_peak": mpap_peak,
        "co_peak": co_peak,
        "ci_peak": ci_peak,
        "mpap_co_slope": mpap_co_slope,
        "pawp_co_slope": pawp_co_slope,
        "exercise_pattern": exercise_pattern,
        "delta_spap": delta_spap,
        "adaptation_type": adaptation_type,
        "volume_challenge_done": volume_done,
        "pawp_delta": pawp_delta,
        "mpap_delta": mpap_delta,
        "vaso_test_done": vaso_done,
        "congestion_likely": congestion_likely,
        "ivc_collapse_yes": ivc_collapse_yes,
        "step_up_present": stepup.present,
        "step_up_from_to": stepup.from_to,
        "step_up_location": stepup.location,
        "step_up_delta": stepup.delta,
        "step_up_sentence": stepup.sentence,
        "v_wave": wedge_v_wave,
        "a_wave": wedge_a_wave,
        "rap_a_wave_flag": rap_a_wave,
        "rap_v_wave_flag": rap_v_wave,
        "rv_pseudo_dip_flag": rv_pseudo_dip,
        "rv_dip_plateau_flag": rv_dip_plateau,
        # Echo-Add-ons
        "s_prime_raai": sprime_raai,
        "sprime_raai": sprime_raai,  # Alias für Regelwerk
        "s_prime_raai_cutoff": SPRIME_RAAI_CUTOFF,
        "s_prime_raai_low": sprime_raai_low,
        "raai": raai,
        "raai_cm2_m2": raai,
        "tapse_spap": tapse_spap,
        "tapse_spap_cutoff": TAPSE_SPAP_CUTOFF,
        "tapse_spap_uncoupled": tapse_spap_uncoupled,
        "tapse_spap_risk": tapse_spap_risk,
        "anemia": anemia,
        "hfpef_percent": hfpef_res.percent,
        "hfpef_category": hfpef_res.category,
    }

    # ---- Scores ----
    who_fc = ui.get("who_fc") or None
    sixmwd = _safe_float(ui.get("six_mwd_m"))
    esc4 = calc_esc_ers_4_strata(who_fc, sixmwd, bnp_pg_ml, ntprobnp_pg_ml)
    esc3 = calc_esc_ers_3_strata(who_fc, sixmwd, bnp_pg_ml, ntprobnp_pg_ml)
    esc_comp = calc_esc_ers_comprehensive_3_strata(ui, derived)
    reveal_lite2 = calc_reveal_lite2(ui)

    scores: Dict[str, Any] = {
        "esc_ers_4s": esc4,
        "esc_ers_3s": esc3,
        "esc_ers_comp": esc_comp.category if esc_comp else None,
        "esc_ers_comp_mean": round(esc_comp.mean_grade, 2) if esc_comp else None,
        "reveal_lite2": reveal_lite2.category if reveal_lite2 else None,
        "reveal_lite2_points": reveal_lite2.points if reveal_lite2 else None,
        "hfpef": hfpef_res.category if hfpef_res else None,
        "hfpef_prob": round(hfpef_res.percent, 1) if (hfpef_res and hfpef_res.percent is not None) else None,
    }

    # ---- Env for rules ----
    env: Dict[str, Any] = {}
    env.update(ui)
    env.update(derived)
    env.update(scores)

    # convenience booleans for rules
    env["has_ph"] = (mpap is not None and mpap > 20)
    env["precap"] = (hemo_cat == "precap")
    env["ipcph"] = (hemo_cat == "ipcph")
    env["cpcph"] = (hemo_cat == "cpcph")

    # PH-guideline aligned PVR threshold (precap/cpc uses PVR>2)
    env["pvr_gt2"] = (pvr is not None and pvr > 2)
    env["pawp_gt15"] = (pawp is not None and pawp > 15)
    env["lvef_ge50"] = (lvef is not None and lvef >= 50)

    # apply rules
    decision = apply_rule_engine(env, rules)

    # missing fields required
    missing: List[str] = []
    for fld in decision.require_fields:
        v = env.get(fld)
        if v is None or v == "" or v is False:
            missing.append(fld)
    decision.missing_fields = missing

    # infer leading cause/action if rulebook didn't set them
    if not decision.leading_cause or not decision.leading_action:
        lc, la = infer_leading_conclusion(env, decision)
        decision.leading_cause = decision.leading_cause or lc
        decision.leading_action = decision.leading_action or la

    case: Dict[str, Any] = {
        "ui": ui,
        "derived": derived,
        "scores": scores,
        "decision": asdict(decision),
        "hfpef": asdict(hfpef_res),
        "env": env,
    }
    return case


# =============================================================================
# Leading cause/action inference (fallback)
# =============================================================================

def infer_leading_conclusion(env: Dict[str, Any], decision: Decision) -> Tuple[str, str]:
    """
    Produces a short "führende ..." cause and a matching main action for the concluding sentence.
    """
    # Shunt
    if env.get("step_up_present"):
        return ("kongenitalen Links-Rechts-Shunt", "eine gezielte Abklärung des Shunts (Echokardiographie inkl. Kontrast/TEE und ggf. kardiale Bildgebung)")

    # CTEPH/CTEPD
    if env.get("vq_defect") or env.get("ct_embolie") or env.get("ct_mosaic"):
        if env.get("precap") or env.get("has_ph"):
            return ("chronisch thromboembolischen Genese (CTEPD/CTEPH, Gruppe 4)", "die Vorstellung im CTEPH-/PH-Board und die weitere spezifische Abklärung (V/Q, CT-/Angio-Review, ggf. Pulmonalisangiographie)")

    # Group 3 (ILD/COPD)
    if env.get("ct_ild") or env.get("ct_emphysema") or env.get("lufu_obstructive") or env.get("lufu_restrictive") or env.get("lufu_diffusion"):
        if env.get("precap") or env.get("has_ph"):
            return ("Lungenerkrankung/Hypoxie (Gruppe 3)", "die konsequente pneumologische Therapie inkl. Optimierung der Oxygenierung und ILD-/COPD-spezifischer Mitbehandlung")

    # HFpEF / left-heart
    if env.get("pawp_gt15") or (env.get("hfpef_category") in ("possible", "likely")) or env.get("la_enlarged"):
        if env.get("lvef_ge50"):
            return ("linkskardialen Ursache im Sinne einer diastolischen Funktionsstörung/HFpEF (Gruppe 2)", "die kardiologische Therapieoptimierung (Volumenmanagement, Rhythmus-/RR-Kontrolle und HFpEF-spezifische Therapie nach Leitlinie)")
        return ("linkskardialen Ursache (Gruppe 2)", "die kardiologische Therapieoptimierung und Behandlung der linksventrikulären Dysfunktion")

    # Default: PAH / pulmonary vascular
    if env.get("precap"):
        return ("pulmonalvaskulären Ursache (PAH/Gruppe 1, DD andere präkapilläre Ursachen)", "die weiterführende Abklärung präkapillärer Ursachen (u.a. Autoimmunität, HIV/Leber, ggf. Genetik) und die PH-spezifische Therapie nach Risikostratifizierung")

    return ("unklaren Genese", "eine strukturierte Komplettierung der Diagnostik und interdisziplinäre Einordnung")


# =============================================================================
# Dashboard HTML
# =============================================================================

def build_dashboard_html(case: Optional[Dict[str, Any]]) -> str:
    if not case:
        return f"""
        <div class="card">
          <div class="card-title">{APP_TITLE}</div>
          <div class="muted">Noch kein Befund generiert. Bitte „Beispiel laden“ oder Daten eingeben und „Befund erstellen/aktualisieren“ klicken.</div>
        </div>
        """

    d = case["decision"]
    der = case["derived"]
    sc = case["scores"]

    def badge(text: str, cls: str = "badge") -> str:
        return f'<span class="{cls}">{text}</span>'

    risk_badges = []
    if sc.get("esc_ers_4s"):
        risk_badges.append(badge(f"ESC/ERS 4-Strata: {sc['esc_ers_4s']}", "badge badge-blue"))
    if sc.get("esc_ers_3s"):
        risk_badges.append(badge(f"ESC/ERS 3-Strata: {sc['esc_ers_3s']}", "badge badge-blue"))
    if sc.get("esc_ers_comp"):
        m = sc.get("esc_ers_comp_mean")
        extra = f" (Ø {m})" if m is not None else ""
        risk_badges.append(badge(f"ESC/ERS umfassend: {sc['esc_ers_comp']}{extra}", "badge badge-blue"))
    if sc.get("reveal_lite2"):
        pts = sc.get("reveal_lite2_points")
        extra = f" ({pts} Pkt.)" if pts is not None else ""
        risk_badges.append(badge(f"REVEAL Lite 2: {sc['reveal_lite2']}{extra}", "badge badge-purple"))

    if der.get("hfpef_percent") is not None and der.get("hfpef_category"):
        risk_badges.append(badge(f"HFpEF (H2FPEF): {der['hfpef_category']} ({_fmt(der['hfpef_percent'],0)}%)", "badge badge-purple"))

    if der.get("congestion_likely"):
        risk_badges.append(badge("Hinweis venöse Kongestion", "badge badge-orange"))

    if der.get("exercise_pattern"):
        risk_badges.append(badge(f"Belastungsmuster: {der['exercise_pattern']}", "badge badge-teal"))

    if der.get("adaptation_type"):
        risk_badges.append(badge(f"Ventrikeladaption: {der['adaptation_type']}", "badge badge-teal"))

    if der.get("step_up_present"):
        risk_badges.append(badge("Shuntverdacht (Sättigungssprung)", "badge badge-red"))

    tags = d.get("tags") or []
    missing = d.get("missing_fields") or []

    return f"""
    <div class="card">
      <div class="card-title">{APP_TITLE}</div>
      <div class="row">
        <div><b>Bundle:</b> {d.get('bundle','–')}</div>
        <div><b>Primäre Einordnung:</b> {d.get('primary_dx','–')}</div>
      </div>
      <div class="badges">{''.join(risk_badges) if risk_badges else '<span class="muted">Keine Scores verfügbar.</span>'}</div>
      <div class="row">
        <div><b>Tags:</b> {', '.join(tags) if tags else '<span class="muted">–</span>'}</div>
      </div>
      <div class="row">
        <div><b>Fehlende Angaben (für Regelwerk):</b> {', '.join(missing) if missing else '<span class="muted">keine</span>'}</div>
      </div>
    </div>
    """


# =============================================================================
# Render context for template blocks
# =============================================================================

def build_render_ctx(case: Dict[str, Any]) -> Dict[str, Any]:
    ui = case["ui"]
    der = case["derived"]
    env = case["env"]

    mpap = der.get("mpap")
    pawp = _safe_float(ui.get("pawp_rest"))
    rap = _safe_float(ui.get("rap_rest"))
    pvr = der.get("pvr")
    ci = der.get("ci")
    tpg = der.get("tpg")
    dpg = der.get("dpg")

    # prior RHK comparison (requested format)
    prev_date = ui.get("prev_rhk_date")
    prev_mpap = _safe_float(ui.get("prev_mpap"))
    prev_pawp = _safe_float(ui.get("prev_pawp"))
    prev_ci = _safe_float(ui.get("prev_ci"))
    prev_pvr = _safe_float(ui.get("prev_pvr"))
    prev_label = ui.get("prev_label") or "stabiler Verlauf"

    comparison_sentence = ""
    if prev_date and any(v is not None for v in (prev_mpap, prev_pawp, prev_ci, prev_pvr)):
        bits = []
        if prev_mpap is not None:
            bits.append(f"mPAP {str(_fmt(prev_mpap,0)) } mmHg")
        if prev_pawp is not None:
            bits.append(f"PAWP {str(_fmt(prev_pawp,0)) } mmHg")
        if prev_ci is not None:
            bits.append(f"CI {str(_fmt(prev_ci,2)) } l/min/m²")
        if prev_pvr is not None:
            bits.append(f"PVR {str(_fmt(prev_pvr,1)) } WU")
        joined = ", ".join(bits)
        comparison_sentence = f"Im Vergleich zu RHK {prev_date} {prev_label} ({joined})."

    # Step-up sentence
    step_up_sentence = der.get("step_up_sentence") or ""

    # V-wave short
    V_wave_short = "Prominente V-Welle in PAWP-Kurve." if der.get("v_wave") else "Keine prominente V-Welle in PAWP-Kurve."
    A_wave_short = "Prominente A-Welle in PAWP-Kurve." if der.get("a_wave") else "Keine prominente A-Welle in PAWP-Kurve."
    RA_A_wave_short = "Prominente A-Welle in RAP-Kurve." if der.get("rap_a_wave_flag") else "Keine prominente A-Welle in RAP-Kurve."
    RA_V_wave_short = "Prominente V-Welle in RAP-Kurve." if der.get("rap_v_wave_flag") else "Keine prominente V-Welle in RAP-Kurve."
    RV_pseudo_dip_short = "Pseudo-Dip in RV-Kurve." if der.get("rv_pseudo_dip_flag") else "Kein Pseudo-Dip in RV-Kurve."
    RV_dip_plateau_short = "Dip-Plateau in RV-Kurve." if der.get("rv_dip_plateau_flag") else "Kein Dip-Plateau in RV-Kurve."

    # Phrases used by rhk_textdb templates
    mpap_phrase = f"mPAP {str(_fmt(mpap,0)) } mmHg" if mpap is not None else "mPAP nicht angegeben"
    pawp_phrase = f"PAWP {str(_fmt(pawp,0)) } mmHg" if pawp is not None else "PAWP nicht angegeben"
    pvr_phrase = f"PVR {str(_fmt(pvr,1)) } WU" if pvr is not None else "PVR nicht angegeben"
    ci_phrase = f"CI {str(_fmt(ci,2)) } l/min/m²" if ci is not None else "CI nicht angegeben"

    hfpef_hint = ""
    if der.get("hfpef_category") in ("possible", "likely"):
        hfpef_hint = f"HFpEF-Wahrscheinlichkeit (H2FPEF) {der.get('hfpef_category')} (~{_fmt(der.get('hfpef_percent'),0)}%)."

    # Slopes hint only if exercise done
    slope_hint = ""
    if der.get("exercise_done") and der.get("mpap_co_slope") is not None and der.get("pawp_co_slope") is not None:
        slope_hint = f"mPAP/CO-Slope {str(_fmt(der.get('mpap_co_slope'),1)) } WU, PAWP/CO-Slope {str(_fmt(der.get('pawp_co_slope'),1)) } WU."

    tpg_hint = f"TPG {str(_fmt(tpg,0)) } mmHg" if tpg is not None else ""
    dpg_hint = f"DPG {str(_fmt(dpg,0)) } mmHg" if dpg is not None else ""

    pressure_resistance_short = ", ".join([x for x in [mpap_phrase, pawp_phrase, pvr_phrase, ci_phrase, tpg_hint, dpg_hint, slope_hint] if x])

    # Additional helper sentences for rhk_textdb templates (missing keys are filled with '' via SafeDict)
    co_method_desc = "unbekannter Methode"
    co_method = ui.get("co_method")
    if co_method == "thermodilution":
        co_method_desc = "Thermodilution"
    elif co_method == "fick":
        co_method_desc = "Fick-Prinzip"

    cv_stauung_phrase = "Hinweise auf venöse Kongestion." if der.get("congestion_likely") else "Keine Hinweise auf venöse Kongestion."
    pv_stauung_phrase = "Hinweise auf pulmonalvenöse Stauung." if env.get("pawp_gt15") else "Keine Hinweise auf pulmonalvenöse Stauung."

    sbp = _safe_float(ui.get("bp_sys"))
    hr = _safe_float(ui.get("hr"))
    systemic_sentence = ""
    if sbp is not None or hr is not None:
        parts = []
        if sbp is not None:
            parts.append(f"RR {_fmt(sbp,0)} mmHg")
        if hr is not None:
            parts.append(f"HF {_fmt(hr,0)}/min")
        systemic_sentence = "Systemische Hämodynamik: " + ", ".join(parts) + "."

    oxygen_sentence = ""
    om = ui.get("oxygen_mode")
    if isinstance(om, str) and om:
        if om == "keine":
            oxygen_sentence = "Messung in Raumluft."
        elif om == "O2":
            flow = _safe_float(ui.get("oxygen_flow_l_min"))
            oxygen_sentence = "Messung unter Sauerstoffgabe" + (f" {_fmt(flow,1)} l/min" if flow is not None else "") + "."
        elif om == "LTOT":
            flow = _safe_float(ui.get("ltot_flow_l_min")) or _safe_float(ui.get("oxygen_flow_l_min"))
            oxygen_sentence = "Messung unter Langzeitsauerstoff" + (f" {_fmt(flow,1)} l/min" if flow is not None else "") + "."
        elif om == "NIV":
            oxygen_sentence = "Messung unter NIV."

    exam_type_desc = "RHK in Ruhe"
    _parts = []
    if der.get("exercise_done"):
        _parts.append("Belastung")
    if der.get("volume_challenge_done"):
        _parts.append("Volumenchallenge")
    if der.get("vasoreactivity_done"):
        _parts.append("Vasoreaktivitätstest")
    if _parts:
        exam_type_desc = "RHK in Ruhe + " + " + ".join(_parts)

    provocation_sentence = ""
    provocation_type_desc = ""
    provocation_result_sentence = ""
    if der.get("volume_challenge_done"):
        provocation_type_desc = "Volumenchallenge"
        delta = der.get("vol_challenge_delta_pawp")
        resp = der.get("vol_challenge_resp")
        if delta is not None:
            provocation_sentence = f"Nach Volumenchallenge ΔPAWP {_fmt(delta,0)} mmHg"
            if resp:
                provocation_sentence += f" ({resp})"
            provocation_sentence += "."
            provocation_result_sentence = provocation_sentence
    elif der.get("vasoreactivity_done"):
        provocation_type_desc = "Vasoreaktivitätstest"
        resp = der.get("vasoreactivity_resp") or ""
        provocation_sentence = f"Vasoreaktivitätstest: {resp}."
        provocation_result_sentence = provocation_sentence
    elif der.get("exercise_done"):
        provocation_type_desc = "Belastung"

    # Follow-up placeholders (used in P11)
    risk = der.get("risk_category")
    followup_timing_desc = "einem geeigneten Intervall"
    if risk == "high":
        followup_timing_desc = "4–12 Wochen"
    elif risk == "intermediate":
        followup_timing_desc = "3–6 Monaten"
    elif risk == "low":
        followup_timing_desc = "6–12 Monaten"

    invasive_followup_desc = "einem passenden Intervall"
    if risk == "high" and der.get("has_ph"):
        invasive_followup_desc = "3–6 Monaten"
    elif risk == "intermediate" and der.get("has_ph"):
        invasive_followup_desc = "6–12 Monaten"

    # Valve focus placeholder (used in P09)
    valve_focus_desc = "Klappenvitien"
    if der.get("v_wave"):
        valve_focus_desc = "Mitralinsuffizienz/linksatriale Druckspitzen"
    elif der.get("rap_v_wave_flag"):
        valve_focus_desc = "Trikuspidalinsuffizienz"
    elif env.get("pawp_gt15"):
        valve_focus_desc = "diastolischer Funktion und Mitralklappe"
    elif der.get("hemo_category") == "precap":
        valve_focus_desc = "Rechtsherz und Trikuspidalklappe"
    elif der.get("a_wave"):
        valve_focus_desc = "diastolischer Funktion"

    # Anemia placeholder (used in P13)
    anemia_context_desc = None
    if der.get("anemia"):
        at = (ui.get("anemia_type") or "").strip().lower()
        if at.startswith("mikro"):
            anemia_context_desc = "mikrozytär (z.B. Eisenmangel/chron. Blutverlust)"
        elif at.startswith("makro"):
            anemia_context_desc = "makrozytär (z.B. Vitamin-B12-/Folat-Mangel, Leber, Alkohol)"
        elif at.startswith("normo"):
            anemia_context_desc = "normozytär (z.B. Entzündung/chron. Erkrankung, Niere)"
        elif "hämol" in at:
            anemia_context_desc = "hämolytisch (Hinweis auf Hämolyse)"
        elif "blut" in at:
            anemia_context_desc = "akute Blutung/Blutverlust" 
        elif at:
            anemia_context_desc = f"Anämie ({at})"
        else:
            anemia_context_desc = "Anämie (Typ unklar)"
    return {
        **env,
        "comparison_sentence": comparison_sentence,
        "step_up_sentence": step_up_sentence,
        "step_up_from_to": der.get("step_up_from_to") or "",
        "V_wave_short": V_wave_short,
        "A_wave_short": A_wave_short,
        "RA_A_wave_short": RA_A_wave_short,
        "RA_V_wave_short": RA_V_wave_short,
        "RV_pseudo_dip_short": RV_pseudo_dip_short,
        "RV_dip_plateau_short": RV_dip_plateau_short,
        "mpap_phrase": mpap_phrase,
        "pawp_phrase": pawp_phrase,
        "pvr_phrase": pvr_phrase,
        "ci_phrase": ci_phrase,
        "pressure_resistance_short": pressure_resistance_short,
        "hfpef_hint": hfpef_hint,
        "co_method_desc": co_method_desc,
        "cv_stauung_phrase": cv_stauung_phrase,
        "pv_stauung_phrase": pv_stauung_phrase,
        "systemic_sentence": systemic_sentence,
        "oxygen_sentence": oxygen_sentence,
        "exam_type_desc": exam_type_desc,
        "provocation_sentence": provocation_sentence,
        "provocation_type_desc": provocation_type_desc,
        "provocation_result_sentence": provocation_result_sentence,
        "followup_timing_desc": followup_timing_desc,
        "invasive_followup_desc": invasive_followup_desc,
        "valve_focus_desc": valve_focus_desc,
        "anemia_context_desc": anemia_context_desc,
    }


# =============================================================================
# Procedere/module rendering with "already done" filtering
# =============================================================================

def _done_flags(env: Dict[str, Any]) -> Dict[str, bool]:
    # These flags are used to remove "already done" items from module texts
    return {
        "vq": bool(env.get("vq_done")),
        "ct": bool(env.get("ct_done")),
        "echo": bool(env.get("echo_done")),
        "cmr": bool(env.get("cmr_done")),
        "lufu": bool(env.get("lufu_done")),
        "cpet": bool(env.get("cpet_done")),
        "ccta": False,
    }


def render_p01_dynamic(env: Dict[str, Any]) -> str:
    """
    P01 is often a broad checklist; we make it dynamic so 'done' items are not repeated.
    """
    done = _done_flags(env)
    lines = []

    # echo
    if not done["echo"]:
        lines.append("• Echokardiographie (inkl. diastolischer Parameter, IVC, ggf. Kontrast)")
    # lufu
    if not done["lufu"]:
        lines.append("• Lungenfunktion inkl. DLCO und ggf. BGA")
    # CT / imaging
    if not done["ct"]:
        lines.append("• CT Thorax / CT-Angio (je nach DD, inkl. Parenchymbeurteilung)")
    # V/Q
    if not done["vq"]:
        lines.append("• V/Q-Szintigraphie zum Ausschluss einer chronisch thromboembolischen Genese")
    # labs
    lines.append("• Laborbasis (BB, Nierenwerte, Entzündung, BNP/NT-proBNP, Gerinnung je nach Kontext)")
    # functional
    lines.append("• Funktionelle Einordnung (WHO-FC, 6MWD ± CPET)")

    if not lines:
        lines = ["• Basisdiagnostik ist weitgehend komplettiert – Verlauf/Follow-up nach Klinik."]

    return "\n".join(lines)


def filter_module_text(text: str, env: Dict[str, Any]) -> str:
    """
    Removes obvious 'already done' bullet lines from module text.
    Keeps the rest unchanged.
    """
    done = _done_flags(env)

    def _skip(line: str) -> bool:
        l = line.lower()
        if done["vq"] and ("v/q" in l or "vq" in l or "ventilations" in l):
            return True
        if done["ct"] and ("ct" in l or "computertom" in l or "angio" in l):
            # careful: don't remove "CCTA" or "CTEPH" words incorrectly
            if "cteph" in l:
                return False
            return True
        if done["echo"] and ("echo" in l or "echokard" in l):
            return True
        if done["lufu"] and ("lungenfun" in l or "dlco" in l):
            return True
        if done["cmr"] and ("mrt" in l or "cmr" in l or "kardio-mrt" in l):
            return True
        return False

    out_lines = []
    for ln in text.splitlines():
        if _skip(ln.strip()):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


# =============================================================================
# Befund – input summary block
# =============================================================================

def _md_kv(label: str, value: str) -> str:
    return f"- **{label}:** {value}"


def summarize_inputs(case: Dict[str, Any]) -> str:
    ui = case["ui"]
    env = case["env"]
    der = case["derived"]

    parts: List[str] = []

    # Klinik
    klinik_lines: List[str] = []
    story = (ui.get("story") or "").strip()
    if story:
        klinik_lines.append(_md_kv("Kurz-Anamnese", story))
    if ui.get("ph_known") is True:
        klinik_lines.append(_md_kv("PH-Diagnose", "bekannt"))
    elif ui.get("ph_suspected") is True:
        klinik_lines.append(_md_kv("PH-Verdachtsdiagnose", "ja"))

    # symptoms
    if ui.get("exertional_dyspnea") is True:
        klinik_lines.append(_md_kv("Belastungsdyspnoe", "ja"))
    syn = ui.get("syncope")
    syn_s: Optional[str] = None
    if isinstance(syn, bool):
        syn_s = "ja" if syn else None
    else:
        tmp = (syn or "").strip()
        if tmp and tmp.lower() not in ("keine", "nein"):
            syn_s = tmp
    if syn_s:
        klinik_lines.append(_md_kv("Synkope", syn_s))
    if ui.get("hemoptysis") is True:
        klinik_lines.append(_md_kv("Hämoptyse", "ja"))
    if ui.get("dizziness") is True:
        klinik_lines.append(_md_kv("Schwindel", "ja"))
    stairs = ui.get("stairs_flights")
    if stairs not in (None, "", 0):
        klinik_lines.append(_md_kv("Treppenstufen/Etagen (Alltag)", str(stairs)))
    walk = (ui.get("daily_walk") or "").strip()
    if walk:
        klinik_lines.append(_md_kv("Alltags-Gehstrecke", walk))

    who_fc = ui.get("who_fc")
    if who_fc:
        klinik_lines.append(_md_kv("WHO-FC", str(who_fc)))
    six = _safe_float(ui.get("six_mwd_m"))
    if six is not None:
        klinik_lines.append(_md_kv("6MWD", f"{_fmt(six,0)} m"))

    if not klinik_lines:
        klinik_lines.append("_Keine klinischen Angaben erfasst._")
    parts.append("### Klinik\n" + "\n".join(klinik_lines))

    # Labor
    lab_lines: List[str] = []
    hb = _safe_float(ui.get("hb_g_dl"))
    if hb is not None:
        lab_lines.append(_md_kv("Hb", f"{_fmt(hb,1)} g/dl" + (" (Anämie)" if der.get("anemia") else "")))
    crp = _safe_float(ui.get("crp_mg_l"))
    if crp is not None:
        lab_lines.append(_md_kv("CRP", f"{_fmt(crp,1)} mg/l"))
    crea = _safe_float(ui.get("creatinine_mg_dl"))
    if crea is not None:
        lab_lines.append(_md_kv("Kreatinin", f"{_fmt(crea,2)} mg/dl"))
    inr = _safe_float(ui.get("inr"))
    if inr is not None:
        lab_lines.append(_md_kv("INR", _fmt(inr,2)))
    ptt = _safe_float(ui.get("ptt_s"))
    if ptt is not None:
        lab_lines.append(_md_kv("PTT", f"{_fmt(ptt,0)} s"))
    thr = _safe_float(ui.get("platelets_g_l"))
    if thr is not None:
        lab_lines.append(_md_kv("Thrombozyten", f"{_fmt(thr,0)} G/l"))
    leuk = _safe_float(ui.get("leukocytes_g_l"))
    if leuk is not None:
        lab_lines.append(_md_kv("Leukozyten", f"{_fmt(leuk,1)} G/l"))

    bnp_kind = ui.get("bnp_kind")
    bnp_val = _safe_float(ui.get("bnp_value"))
    if bnp_val is not None:
        extra = ""
        if ui.get("entresto") is True and isinstance(bnp_kind, str) and "BNP" in bnp_kind.upper() and "NT" not in bnp_kind.upper():
            extra = " (Hinweis: unter ARNI ist NT-proBNP typischerweise besser verwertbar)"
        lab_lines.append(_md_kv(str(bnp_kind or "BNP/NT-proBNP"), f"{_fmt(bnp_val,0)} pg/ml{extra}"))

    cong_org = ui.get("congestive_organopathy")
    if isinstance(cong_org, str) and cong_org.lower().startswith("ja"):
        lab_lines.append(_md_kv("Hinweis auf congestive Organopathie", "ja"))
    elif isinstance(cong_org, str) and cong_org.lower().startswith("nein"):
        lab_lines.append(_md_kv("Hinweis auf congestive Organopathie", "nein"))

    if not lab_lines:
        lab_lines.append("_Keine Laborwerte erfasst._")
    parts.append("### Labor\n" + "\n".join(lab_lines))

    # Bildgebung & Echo/CMR
    img_lines: List[str] = []

    if ui.get("ct_done"):
        findings = []
        for key, lab in [
            ("ct_ild", "ILD"),
            ("ct_emphysema", "Emphysem"),
            ("ct_embolie", "Embolie"),
            ("ct_mosaic", "Mosaikperfusion"),
            ("ct_koronarkalk", "Koronarkalk"),
        ]:
            if ui.get(key):
                findings.append(lab)
        if findings:
            img_lines.append(_md_kv("CT Thorax/Angio", ", ".join(findings)))
        else:
            img_lines.append(_md_kv("CT Thorax/Angio", "durchgeführt (keine pathologischen Befunde angegeben)"))
    else:
        img_lines.append(_md_kv("CT Thorax/Angio", "nicht angegeben"))

    if ui.get("vq_done"):
        vq_abn = "pathologisch" if ui.get("vq_defect") else "unauffällig/keine Defekte angegeben"
        img_lines.append(_md_kv("V/Q", vq_abn))
        vq_desc = (ui.get("vq_desc") or "").strip()
        if vq_desc:
            img_lines.append(_md_kv("V/Q Details", vq_desc))

    if ui.get("echo_done") or any(ui.get(k) not in (None, "", False) for k in ["lvef", "la_enlarged", "ee_ratio", "pasp_echo"]):
        echo_bits = []
        lvef = _safe_float(ui.get("lvef"))
        if lvef is not None:
            echo_bits.append(f"LVEF {_fmt(lvef,0)}%")
        if ui.get("la_enlarged"):
            echo_bits.append("LA erweitert")
        ee = _safe_float(ui.get("ee_ratio"))
        if ee is not None:
            echo_bits.append(f"E/e' {_fmt(ee,1)}")
        pasp = _safe_float(ui.get("pasp_echo"))
        if pasp is not None:
            echo_bits.append(f"sPAP {_fmt(pasp,0)} mmHg")
        tapse = _safe_float(ui.get("tapse_mm"))
        if tapse is not None:
            echo_bits.append(f"TAPSE {_fmt(tapse,0)} mm")
        if der.get("tapse_spap") is not None:
            echo_bits.append(f"TAPSE/sPAP {_fmt(der.get('tapse_spap'),2)}")
        sprime = _safe_float(ui.get("s_prime_cm_s"))
        if sprime is not None:
            echo_bits.append(f"S' {_fmt(sprime,1)} cm/s")
        raesa = _safe_float(ui.get("ra_esa_cm2"))
        if raesa is not None:
            echo_bits.append(f"RA ESA {_fmt(raesa,0)} cm²")
        if der.get("raai") is not None:
            echo_bits.append(f"RAAI {_fmt(der.get('raai'),1)} cm²/m²")
        if der.get("s_prime_raai") is not None:
            echo_bits.append(f"S'/RAAI {_fmt(der.get('s_prime_raai'),2)}")
        ivcd = _safe_float(ui.get("ivc_diam_mm"))
        if ivcd is not None:
            echo_bits.append(f"IVC {_fmt(ivcd,0)} mm")
        ivcc = ui.get("ivc_collapse")
        if isinstance(ivcc, str) and ivcc:
            echo_bits.append(f"IVC Kollaps: {ivcc}")
        if echo_bits:
            img_lines.append(_md_kv("Echo", ", ".join(echo_bits)))
        else:
            img_lines.append(_md_kv("Echo", "durchgeführt (keine Details angegeben)"))

        # Echo-Add-ons / Cut-offs (nur wenn auffällig)
        echo_flags: List[str] = []
        if der.get("s_prime_raai_low") is True:
            echo_flags.append("S'/RAAi erniedrigt (<0,81; Yogeswaran et al.)")
        if der.get("tapse_spap_uncoupled") is True:
            echo_flags.append("TAPSE/sPAP erniedrigt (<0,31; Tello et al.)")
        if echo_flags:
            img_lines.append(_md_kv("Echo Zusatz", "; ".join(echo_flags)))

    if ui.get("cmr_done") or any(ui.get(k) not in (None, "", False) for k in ["rvef", "rvesvi"]):
        cmr_bits = []
        rvef = _safe_float(ui.get("rvef"))
        if rvef is not None:
            cmr_bits.append(f"RVEF {_fmt(rvef,0)}%")
        rvesvi = _safe_float(ui.get("rvesvi"))
        if rvesvi is not None:
            cmr_bits.append(f"RVESVi {_fmt(rvesvi,0)} ml/m²")
        if cmr_bits:
            img_lines.append(_md_kv("CMR", ", ".join(cmr_bits)))
        else:
            img_lines.append(_md_kv("CMR", "durchgeführt (keine Details angegeben)"))

    if not img_lines:
        img_lines.append("_Keine Bildgebung/Echo/CMR-Angaben._")
    parts.append("### Bildgebung / Echo / CMR\n" + "\n".join(img_lines))

    # Lungenfunktion
    lufu_lines: List[str] = []
    if ui.get("lufu_done"):
        phen = []
        if ui.get("lufu_obstructive"):
            phen.append("obstruktiv")
        if ui.get("lufu_restrictive"):
            phen.append("restriktiv")
        if ui.get("lufu_diffusion"):
            phen.append("Diffusionsstörung")
        if phen:
            lufu_lines.append(_md_kv("Phänotyp", ", ".join(phen)))
        fev1 = _safe_float(ui.get("fev1_l"))
        if fev1 is not None:
            lufu_lines.append(_md_kv("FEV1", f"{_fmt(fev1,2)} l"))
        fvc = _safe_float(ui.get("fvc_l"))
        if fvc is not None:
            lufu_lines.append(_md_kv("FVC", f"{_fmt(fvc,2)} l"))
        dlco = _safe_float(ui.get("dlco_sb"))
        if dlco is not None:
            lufu_lines.append(_md_kv("DLCO", f"{_fmt(dlco,1)}"))
        summ = (ui.get("lufu_summary") or "").strip()
        if summ:
            lufu_lines.append(_md_kv("Kommentar", summ))
        if not lufu_lines:
            lufu_lines.append("_Lungenfunktion durchgeführt (Details nicht angegeben)._")
    else:
        lufu_lines.append("_Keine Lungenfunktion erfasst._")
    parts.append("### Lungenfunktion\n" + "\n".join(lufu_lines))

    return "\n\n".join(parts)


# =============================================================================
# Doctor report (Markdown)
# =============================================================================

def build_doctor_report(case: Dict[str, Any], blocks: Dict[str, TextBlock]) -> str:
    ui = case["ui"]
    der = case["derived"]
    sc = case["scores"]
    dec = case["decision"]
    env = case["env"]

    ctx = build_render_ctx(case)

    # Bundle blocks
    b_id = f"{dec['bundle']}_B"
    e_id = f"{dec['bundle']}_E"
    beurteilung = render_block(blocks[b_id], ctx) if b_id in blocks else f"[Fehlender Textblock: {b_id}]"
    empfehlung = render_block(blocks[e_id], ctx) if e_id in blocks else f"[Fehlender Textblock: {e_id}]"

    # RHK structured section
    rest_line = f"- sPAP {_fmt(ui.get('spap_rest'),0)} / dPAP {_fmt(ui.get('dpap_rest'),0)} / mPAP {_fmt(der.get('mpap'),0)} mmHg\n" \
                f"- PAWP {_fmt(ui.get('pawp_rest'),0)} mmHg, RAP {_fmt(ui.get('rap_rest'),0)} mmHg\n" \
                f"- CO {_fmt(der.get('co'),2)} l/min, CI {_fmt(der.get('ci'),2)} l/min/m²\n" \
                f"- PVR {_fmt(der.get('pvr'),1)} WU (PVRi {_fmt(der.get('pvri'),1)} WU·m²), TPG {_fmt(der.get('tpg'),0)} mmHg, DPG {_fmt(der.get('dpg'),0)} mmHg"

    exercise_block = ""
    if der.get("exercise_done"):
        ex_lines = []
        ex_lines.append(_md_kv("mPAP/CO-Slope", f"{_fmt(der.get('mpap_co_slope'),1)} WU"))
        ex_lines.append(_md_kv("PAWP/CO-Slope", f"{_fmt(der.get('pawp_co_slope'),1)} WU"))
        ex_lines.append(_md_kv("ΔsPAP (Peak–Ruhe)", f"{_fmt(der.get('delta_spap'),0)} mmHg"))
        ex_lines.append(_md_kv("peak CI", f"{_fmt(der.get('ci_peak'),2)} l/min/m²"))
        if der.get("adaptation_type"):
            ex_lines.append(_md_kv("Adaptionstyp", "homeometrisch" if der["adaptation_type"] == "homeometric" else "heterometrisch"))
        if der.get("exercise_pattern"):
            ex_lines.append(_md_kv("Belastungsmuster", str(der.get("exercise_pattern"))))
        exercise_block = "#### Belastungshämodynamik\n" + "\n".join(ex_lines)

    volume_block = ""
    if der.get("volume_challenge_done"):
        vol_lines = []
        vol_lines.append(_md_kv("PAWP (Δ)", f"{_fmt(der.get('pawp_delta'),0)} mmHg"))
        vol_lines.append(_md_kv("mPAP (Δ)", f"{_fmt(der.get('mpap_delta'),0)} mmHg"))
        volume_block = "#### Volumenchallenge\n" + "\n".join(vol_lines)

    vaso_block = ""
    if der.get("vaso_test_done"):
        vaso_lines = []
        vaso_lines.append(_md_kv("Agent", str(ui.get("vaso_agent") or "—")))
        if ui.get("vaso_response_desc"):
            vaso_lines.append(_md_kv("Antwort", str(ui.get("vaso_response_desc"))))
        vaso_block = "#### Vasoreaktivität\n" + "\n".join(vaso_lines)

    stepox_block = ""
    if any(_safe_float(ui.get(k)) is not None for k in ["sat_svc", "sat_ivc", "sat_ra", "sat_rv", "sat_pa", "sat_ao"]):
        sat_lines = []
        for k, lab in [("sat_svc", "SVC"), ("sat_ivc", "IVC"), ("sat_ra", "RA"), ("sat_rv", "RV"), ("sat_pa", "PA"), ("sat_ao", "AO")]:
            v = _safe_float(ui.get(k))
            if v is not None:
                sat_lines.append(_md_kv(lab, f"{_fmt(v,0)}%"))
        sat_lines.append(_md_kv("Interpretation", der.get("step_up_sentence") or "—"))
        stepox_block = "#### Stufenoxymetrie\n" + "\n".join(sat_lines)

    curve_block = ""
    curve_flags = []
    if der.get("v_wave"):
        curve_flags.append("V-Welle (PAWP)")
    if der.get("a_wave"):
        curve_flags.append("A-Welle (PAWP)")
    if der.get("rap_a_wave_flag"):
        curve_flags.append("A-Welle (RAP)")
    if der.get("rap_v_wave_flag"):
        curve_flags.append("V-Welle (RAP)")
    if der.get("rv_pseudo_dip_flag"):
        curve_flags.append("Pseudo-Dip (RV)")
    if der.get("rv_dip_plateau_flag"):
        curve_flags.append("Dip-Plateau (RV)")
    if curve_flags:
        curve_block = "#### Kurvenmorphologie\n" + "\n".join([_md_kv("Befund", ", ".join(curve_flags))])

    # Risk lines (prominent, directly after dx)
    risk_lines = []
    if sc.get("esc_ers_4s"):
        risk_lines.append(_md_kv("ESC/ERS 4-Strata", sc["esc_ers_4s"]))
    if sc.get("esc_ers_3s"):
        risk_lines.append(_md_kv("ESC/ERS 3-Strata", sc["esc_ers_3s"]))
    if sc.get("esc_ers_comp"):
        m = sc.get("esc_ers_comp_mean")
        extra = f" (Ø {m})" if m is not None else ""
        risk_lines.append(_md_kv("ESC/ERS umfassend", f"{sc['esc_ers_comp']}{extra}"))
    if sc.get("reveal_lite2"):
        pts = sc.get("reveal_lite2_points")
        extra = f" ({pts} Punkte)" if pts is not None else ""
        risk_lines.append(_md_kv("REVEAL Lite 2", f"{sc['reveal_lite2']}{extra}"))
    if der.get("hfpef_category"):
        risk_lines.append(_md_kv("HFpEF (H2FPEF)", f"{der['hfpef_category']} (~{_fmt(der.get('hfpef_percent'),0)}%)"))
    risk_block = "\n".join(risk_lines) if risk_lines else "_Keine Risikostratifizierung möglich (Daten fehlen)._"

    # Modules (engine + user selected)
    selected = ui.get("modules") or []
    all_mods = list(dict.fromkeys((dec.get("modules") or []) + selected))

    modules_txts: List[str] = []
    for mid in all_mods:
        if mid == "P01":
            txt = render_p01_dynamic(env)
            modules_txts.append(f"**{mid} – {blocks.get(mid, TextBlock(mid, mid, '', 'module')).title}**\n{txt}")
            continue
        if mid in blocks:
            txt = render_block(blocks[mid], ctx)
            txt = filter_module_text(txt, env)
            if txt:
                modules_txts.append(f"**{mid} – {blocks[mid].title}**\n{txt}")

    recs = dec.get("recommendations") or []

    # Concluding sentence
    leading_cause = dec.get("leading_cause") or "unklaren Genese"
    leading_action = dec.get("leading_action") or "eine strukturierte Komplettierung der Diagnostik"
    concluding = f"In der Zusammenschau der Befunde gehen wir von einer führenden **{leading_cause}** aus. Entsprechend empfehlen wir **{leading_action}**."

    # Build final report
    header = f"# RHK-Befund (Assistenzbericht)\n\n**Datum:** {_dt.date.today().strftime('%d.%m.%Y')}\n\n"
    patient_line = ""
    if ui.get("name") or ui.get("firstname"):
        patient_line = f"**Patient:** {ui.get('firstname','')} {ui.get('name','')}".strip() + "\n\n"

    summary_block = summarize_inputs(case)

    report = [
        header,
        patient_line,
        "## Befundübersicht\n",
        summary_block,
        "\n## Rechtsherzkatheter\n",
        "#### Ruhehämodynamik\n",
        rest_line,
    ]
    if exercise_block:
        report.append("\n" + exercise_block)
    if volume_block:
        report.append("\n" + volume_block)
    if vaso_block:
        report.append("\n" + vaso_block)
    if stepox_block:
        report.append("\n" + stepox_block)
    if curve_block:
        report.append("\n" + curve_block)

    # Diagnosis + risk + assessment
    report.append("\n## Beurteilung\n")
    report.append(beurteilung.strip() + "\n")

    report.append("\n## Empfehlung\n")
    report.append(_md_kv("Diagnose/Einordnung", dec.get("primary_dx", "—")))
    report.append("\n" + risk_block + "\n")
    report.append(empfehlung.strip() + "\n")

    if recs:
        report.append("\n**Zusätzliche Hinweise (Regelwerk):**\n")
        report.extend([f"- {r}" for r in recs])
        report.append("")

    report.append(concluding + "\n")

    if modules_txts or ui.get("procedere_free"):
        report.append("\n## Procedere:\n")
        if modules_txts:
            report.append("\n\n".join(modules_txts))
        free = (ui.get("procedere_free") or "").strip()
        if free:
            report.append("\n**Freitext:**\n" + free)

    # Comparison sentence
    if ctx.get("comparison_sentence"):
        report.append("\n---\n" + ctx["comparison_sentence"])

    return "\n".join(report).strip()


# =============================================================================
# Patient report (plain language, no abbreviations/numbers)
# =============================================================================

def _patient_hemo_story(der: Dict[str, Any], env: Dict[str, Any]) -> List[str]:
    cat = der.get("hemo_category")
    lines: List[str] = []
    if cat == "no_ph":
        lines.append("Die Druckwerte in den Blutgefäßen der Lunge sind in Ruhe nicht erhöht.")
        return lines

    lines.append("Die Untersuchung zeigt einen erhöhten Druck in den Blutgefäßen der Lunge (Lungenhochdruck).")

    if cat == "precap":
        lines.append("Die Messwerte sprechen dafür, dass die Ursache vor allem in den Blutgefäßen der Lunge oder in der Lunge selbst liegt.")
    elif cat == "ipcph":
        lines.append("Die Messwerte sprechen dafür, dass der Druckanstieg vor allem durch die linke Herzseite entsteht. Das passt zu einer Herzschwäche, bei der das Herz sich nicht gut füllt, obwohl es noch gut pumpt.")
    elif cat == "cpcph":
        lines.append("Die Messwerte sprechen für eine Mischung: Es gibt einen Druckanstieg durch die linke Herzseite und zusätzlich eine Enge in den Blutgefäßen der Lunge.")
    else:
        lines.append("Die Messwerte lassen sich nicht eindeutig einer Ursache zuordnen. Dafür fehlen einzelne Angaben oder es gibt eine Mischkonstellation.")

    if env.get("congestion_likely"):
        lines.append("Zusätzlich gibt es Hinweise darauf, dass sich Flüssigkeit im Körper staut (zum Beispiel an den großen Venen).")

    # Zusatzzeichen im Herzultraschall (ohne Zahlen/Abkürzungen)
    if der.get("s_prime_raai_low") is True or der.get("tapse_spap_uncoupled") is True:
        lines.append("Im Herzultraschall gibt es zusätzlich Hinweise, dass die rechte Herzkammer im Moment stärker belastet ist. Das ist ein Warnsignal und sollte im Verlauf mitbeobachtet werden.")

    if env.get("exercise_done") and der.get("exercise_pattern") in ("precap_pattern", "postcap_pattern", "left_pressure_pattern"):
        lines.append("Unter Belastung zeigen sich auffällige Druckanstiege. Das hilft uns, die Ursache besser einzuordnen.")

    if env.get("step_up_present"):
        lines.append("Die Sauerstoffmessungen im Herzen sprechen für eine mögliche zusätzliche Verbindung zwischen Herzhöhlen, die zu einem Links-Rechts-Fluss führen kann. Das sollte gezielt abgeklärt werden.")

    return lines


def build_patient_report(case: Dict[str, Any]) -> str:
    ui = case["ui"]
    der = case["derived"]
    env = case["env"]
    dec = case["decision"]

    name = (ui.get("firstname") or "").strip()
    if ui.get("name"):
        name = (name + " " + str(ui.get("name")).strip()).strip()

    header = f"# Patientenbericht\n\n"
    if name:
        header += f"**Für:** {name}\n\n"

    intro = [
        "## Was wurde untersucht?",
        "Wir haben eine Untersuchung des rechten Herzens und der Lungengefäße durchgeführt (Rechtsherzkatheter).",
        "Damit kann man messen, wie hoch der Druck in den Lungengefäßen ist, wie gut das Herz Blut nach vorne pumpen kann und ob es Hinweise auf eine Stauung gibt.",
    ]

    findings = ["## Was haben wir gefunden?"]
    findings.extend(_patient_hemo_story(der, env))

    # Imaging context
    context_lines: List[str] = []
    if env.get("ct_ild") or env.get("ct_emphysema"):
        context_lines.append("In der Bildgebung gibt es Hinweise auf eine Lungenerkrankung, die den Lungenhochdruck mit erklären kann.")
    if env.get("vq_defect") or env.get("ct_embolie") or env.get("ct_mosaic"):
        context_lines.append("Es gibt Hinweise, die zu älteren oder chronischen Blutgerinnseln in der Lunge passen können. Das sollte weiter abgeklärt werden.")
    if env.get("hfpef_category") in ("possible", "likely") or env.get("la_enlarged"):
        context_lines.append("Ein Teil der Befunde passt zu einer Belastung der linken Herzseite. Das kann bei einer Herzschwäche mit erhaltener Pumpleistung auftreten.")

    if context_lines:
        findings.append("")
        findings.append("Zusätzlicher Kontext:")
        findings.extend([f"- {c}" for c in context_lines])

    meaning = [
        "## Was bedeutet das?",
        "Lungenhochdruck ist kein einzelnes Krankheitsbild, sondern kann verschiedene Ursachen haben.",
        "Darum ist wichtig, die führende Ursache zu finden: zum Beispiel eine Erkrankung der Lunge, eine Störung auf der linken Herzseite oder (seltener) alte Blutgerinnsel in den Lungengefäßen.",
        "Die Einordnung bestimmt, welche Behandlung am meisten hilft.",
    ]

    next_steps = ["## Wie geht es weiter?"]
    # Translate recommendations to patient language (very high level)
    def _patient_lead_action() -> Tuple[str, str]:
        # Keep this intentionally free of abbreviations and numbers.
        if env.get("step_up_present"):
            return (
                "eine mögliche zusätzliche Verbindung zwischen Herzhöhlen",
                "eine gezielte Ultraschall-Untersuchung des Herzens, gegebenenfalls ergänzt durch weitere Bildgebung"
            )
        if env.get("vq_defect") or env.get("ct_embolie") or env.get("ct_mosaic"):
            return (
                "mögliche ältere oder chronische Blutgerinnsel in den Lungengefäßen",
                "eine weitere Abklärung in einem spezialisierten Zentrum und eine passende Behandlung je nach Ergebnis"
            )
        if env.get("ct_ild") or env.get("ct_emphysema") or env.get("lufu_restrictive") or env.get("lufu_obstructive") or env.get("lufu_diffusion"):
            return (
                "eine Lungenerkrankung oder ein niedriger Sauerstoffgehalt als Mitursache",
                "eine pneumologische Mitbehandlung und Optimierung der Sauerstoffversorgung"
            )
        if env.get("pawp_gt15") or env.get("la_enlarged") or env.get("hfpef_category") in ("possible", "likely"):
            return (
                "eine Belastung der linken Herzseite",
                "eine kardiologische Therapieoptimierung mit Anpassung der Behandlung und Kontrolle von Blutdruck, Herzrhythmus und Flüssigkeitshaushalt"
            )
        if der.get("hemo_category") == "precap":
            return (
                "eine Erkrankung der Blutgefäße in der Lunge",
                "eine weiterführende Abklärung der Ursachen und, wenn sinnvoll, eine spezielle medikamentöse Behandlung"
            )
        return (
            "eine derzeit noch nicht eindeutige Ursache",
            "weitere Untersuchungen und eine gemeinsame Therapieplanung"
        )

    lead, action = _patient_lead_action()
    next_steps.append(f"Aus Sicht unseres Teams steht als wichtigste Ursache {lead} im Vordergrund.")
    next_steps.append(f"Als nächster Schritt empfehlen wir vor allem: {action}.")

    if env.get("anemia"):
        next_steps.append("Im Blutbild gibt es Hinweise auf eine Blutarmut. Das sollte weiter abgeklärt und behandelt werden, weil es die Belastbarkeit beeinflussen kann.")
    if env.get("congestion_likely"):
        next_steps.append("Wenn Sie Wassereinlagerungen oder Gewichtszunahme bemerken, sprechen Sie bitte Ihr Behandlungsteam an. Oft hilft eine Anpassung der Entwässerung.")
    if env.get("ltot"):
        next_steps.append("Wenn eine Langzeit-Sauerstofftherapie besteht, sollte geprüft werden, ob die Einstellung passt und ob Sie ausreichend versorgt sind.")

    closing = [
        "## Wichtig",
        "Dieser Text ist eine verständliche Zusammenfassung. Er ersetzt nicht das Gespräch mit Ihrem Behandlungsteam.",
        "Bitte nehmen Sie den Bericht zu Ihrem nächsten Termin mit, damit wir die nächsten Schritte gemeinsam planen können.",
    ]

    # Keep it coherent and not chopped: paragraphs rather than many short bullets
    def para(lines: List[str]) -> str:
        out: List[str] = []
        for ln in lines:
            if ln.startswith("##"):
                out.append(ln)
            else:
                out.append(ln)
        return "\n".join(out)

    sections = [
        header,
        para(intro),
        "",
        para(findings),
        "",
        para(meaning),
        "",
        para(next_steps),
        "",
        para(closing),
    ]
    return "\n".join(sections).strip()


# =============================================================================
# Internal report
# =============================================================================

def build_internal_report(case: Dict[str, Any]) -> str:
    env = case.get("env") or {}
    dec = case.get("decision") or {}
    lines = [
        "## Internal Debug",
        f"- Bundle: {dec.get('bundle')}",
        f"- Primary DX: {dec.get('primary_dx')}",
        f"- Tags: {', '.join(dec.get('tags') or [])}",
        f"- Missing: {', '.join(dec.get('missing_fields') or [])}",
        "",
        "### Env (Auszug)",
    ]
    keys = [
        "mpap", "pawp_rest", "pvr", "ci", "tpg", "dpg",
        "hemo_category", "precap", "ipcph", "cpcph",
        "hfpef_category", "hfpef_percent",
        "congestion_likely", "step_up_present", "step_up_from_to",
        "mpap_co_slope", "pawp_co_slope", "exercise_pattern",
        "adaptation_type",
        "s_prime_raai",
    ]
    for k in keys:
        lines.append(f"- {k}: {env.get(k)}")
    return "\n".join(lines)


# =============================================================================
# Random example generation (now with lab constellations)
# =============================================================================

def random_example() -> Dict[str, Any]:
    """
    Returns a random but coherent example case.
    Randomizes labs and patterns (normal, inflammation, anemia, renal).
    """
    today = _dt.date.today()

    # Base demographics
    age = random.choice([34, 52, 68, 74])
    sex = random.choice(["weiblich", "männlich"])
    height = random.choice([160, 168, 175, 182])
    weight = random.choice([58, 72, 86, 98])

    # Patterns
    pattern = random.choice(["precap_cteph", "precap_ild", "postcap_hfpef", "normal", "cpcph"])

    ui: Dict[str, Any] = {}

    ui["name"] = random.choice(["Muster", "Beispiel", "Patient"])
    ui["firstname"] = random.choice(["Anna", "Max", "Sofia", "Leon"])
    ui["age"] = age
    ui["sex"] = sex
    ui["height_cm"] = height
    ui["weight_kg"] = weight
    ui["story"] = random.choice([
        "Belastungsdyspnoe seit Monaten.",
        "Zunehmende Luftnot und reduzierte Belastbarkeit.",
        "Dyspnoe, gelegentlich Schwindel.",
        "Kontrolle nach PH-Verdacht."
    ])
    ui["ph_known"] = random.choice([False, False, True])
    ui["ph_suspected"] = True

    # Functional
    ui["who_fc"] = random.choice(["II", "III"])
    ui["six_mwd_m"] = random.choice([220, 320, 420])
    ui["syncope"] = random.choices(["keine", "gelegentlich", "wiederholt"], weights=[0.85, 0.12, 0.03], k=1)[0]
    ui["hemoptysis"] = False
    ui["dizziness"] = random.choice([False, True])
    ui["stairs_flights"] = random.choice([1, 2, 3, 0])
    ui["daily_walk"] = random.choice(["ca. 200 m", "ca. 500 m", "10–15 Minuten", ""])

    # Labs – random constellation
    lab_mode = random.choice(["normal", "inflammation", "anemia_micro", "anemia_macro", "renal"])
    ui["crp_mg_l"] = 3 if lab_mode != "inflammation" else random.choice([25, 60])
    ui["leukocytes_g_l"] = 7 if lab_mode != "inflammation" else random.choice([12, 16])
    ui["creatinine_mg_dl"] = random.choice([0.8, 1.0, 1.2]) if lab_mode != "renal" else random.choice([1.8, 2.2])
    ui["inr"] = random.choice([1.0, 1.1, 1.2])
    ui["ptt_s"] = random.choice([28, 31, 34])
    ui["platelets_g_l"] = random.choice([190, 240, 320])

    if "anemia" in lab_mode:
        ui["hb_g_dl"] = random.choice([9.8, 10.6, 11.4])
        ui["anemia_type"] = "mikrozytär" if lab_mode == "anemia_micro" else "makrozytär"
    else:
        ui["hb_g_dl"] = random.choice([12.6, 13.8, 15.1])
        ui["anemia_type"] = None

    ui["bnp_kind"] = random.choice(["NT-proBNP", "BNP"])
    ui["bnp_value"] = random.choice([120, 380, 1200, 2400]) if pattern != "normal" else random.choice([40, 80, 120])
    ui["entresto"] = random.choice([False, False, True])
    ui["congestive_organopathy"] = random.choice(["ja", "nein"]) if pattern in ("cpcph", "postcap_hfpef") else None

    # Imaging/Echo/CMR
    ui["ct_done"] = True
    ui["ct_koronarkalk"] = random.choice([False, True])

    ui["ct_ild"] = (pattern == "precap_ild")
    ui["ct_emphysema"] = random.choice([False, True]) if pattern == "precap_ild" else False
    ui["ct_embolie"] = (pattern == "precap_cteph")
    ui["ct_mosaic"] = (pattern == "precap_cteph")

    ui["ild_type"] = "Fibrosierende ILD" if ui["ct_ild"] else ""
    ui["ild_histology"] = random.choice([False, True]) if ui["ct_ild"] else False
    ui["ild_fibrosis_clinic"] = random.choice([False, True]) if ui["ct_ild"] else False
    ui["ild_extent"] = random.choice(["gering", "mittel", "ausgedehnt"]) if ui["ct_ild"] else ""

    ui["vq_done"] = random.choice([True, False]) if pattern == "precap_cteph" else random.choice([False, True])
    ui["vq_defect"] = True if (pattern == "precap_cteph" and ui["vq_done"]) else False
    ui["vq_desc"] = "Mehrsegmentale Perfusionsdefekte." if ui["vq_defect"] else ""

    ui["echo_done"] = True
    ui["lvef"] = 60 if pattern != "cpcph" else 55
    ui["la_enlarged"] = True if pattern in ("postcap_hfpef", "cpcph") else random.choice([False, True])
    ui["ee_ratio"] = 16 if pattern in ("postcap_hfpef", "cpcph") else random.choice([9, 11, 13])
    ui["pasp_echo"] = random.choice([35, 45, 60]) if pattern != "normal" else 28
    ui["tapse_mm"] = 22 if pattern == "normal" else random.choice([14, 16, 18, 20])
    ui["atrial_fib"] = True if pattern in ("postcap_hfpef", "cpcph") else False
    ui["ivc_diam_mm"] = random.choice([18, 22, 26])
    ui["ivc_collapse"] = random.choice(["ja", "nein"]) if pattern in ("postcap_hfpef", "cpcph") else "ja"

    # S'/RAAI
    ui["s_prime_cm_s"] = random.choice([9.0, 11.0, 13.0])
    ui["ra_esa_cm2"] = random.choice([16.0, 20.0, 26.0])

    ui["cmr_done"] = random.choice([False, True])
    ui["rvef"] = random.choice([35, 45, 55]) if ui["cmr_done"] else None
    ui["rvesvi"] = random.choice([55, 75, 95]) if ui["cmr_done"] else None

    # Lufu
    ui["lufu_done"] = True
    ui["lufu_obstructive"] = True if ui["ct_emphysema"] else False
    ui["lufu_restrictive"] = True if ui["ct_ild"] else False
    ui["lufu_diffusion"] = True if ui["ct_ild"] else random.choice([False, True])
    ui["fev1_l"] = random.choice([1.4, 2.1, 2.8])
    ui["fvc_l"] = random.choice([2.0, 2.8, 3.6])
    ui["dlco_sb"] = random.choice([35, 52, 68])
    ui["lufu_summary"] = random.choice(["", "Leichte Diffusionsstörung.", "Obstruktives Muster."])

    # Rest hemo
    if pattern == "normal":
        spap = 28
        dpap = 10
        pawp = 10
        co = 5.2
        rap = 6
    elif pattern == "postcap_hfpef":
        spap = 55
        dpap = 25
        pawp = 20
        co = 4.5
        rap = 10
    elif pattern == "cpcph":
        spap = 70
        dpap = 35
        pawp = 22
        co = 3.6
        rap = 14
    else:  # precap
        spap = 72
        dpap = 30
        pawp = 10
        co = 4.0
        rap = 9

    ui["spap_rest"] = spap
    ui["dpap_rest"] = dpap
    ui["mpap_rest"] = None  # let it be computed
    ui["pawp_rest"] = pawp
    ui["rap_rest"] = rap
    ui["co_rest"] = co
    ui["ci_rest"] = None
    ui["pvr_rest"] = None

    # Exercise
    ui["exercise_done"] = random.choice([False, True]) if pattern != "normal" else False
    if ui["exercise_done"]:
        ui["spap_peak"] = spap + random.choice([25, 35])
        ui["dpap_peak"] = dpap + random.choice([10, 15])
        ui["mpap_peak"] = None
        ui["pawp_peak"] = pawp + random.choice([3, 10, 15])  # stronger rise in HFpEF
        ui["co_peak"] = co + random.choice([1.0, 1.8, 2.5])
    else:
        ui["spap_peak"] = None
        ui["dpap_peak"] = None
        ui["mpap_peak"] = None
        ui["pawp_peak"] = None
        ui["co_peak"] = None

    # Step oximetry: only in shunt example sometimes
    if random.random() < 0.25:
        # likely no shunt
        ui["sat_svc"] = 65
        ui["sat_ivc"] = 70
        ui["sat_ra"] = 67
        ui["sat_rv"] = 67
        ui["sat_pa"] = 67
        ui["sat_ao"] = 96
    elif random.random() < 0.15:
        # atrial step-up
        ui["sat_svc"] = 65
        ui["sat_ivc"] = 70
        ui["sat_ra"] = 80
        ui["sat_rv"] = 80
        ui["sat_pa"] = 80
        ui["sat_ao"] = 96
    else:
        ui["sat_svc"] = None
        ui["sat_ivc"] = None
        ui["sat_ra"] = None
        ui["sat_rv"] = None
        ui["sat_pa"] = None
        ui["sat_ao"] = None

    # Curves
    ui["wedge_v_wave"] = True if pattern in ("postcap_hfpef", "cpcph") and random.random() < 0.4 else False
    ui["wedge_a_wave"] = True if pattern in ("postcap_hfpef", "cpcph") and random.random() < 0.3 else False
    ui["rap_a_wave"] = random.random() < 0.2
    ui["rap_v_wave"] = random.random() < 0.15
    ui["rv_pseudo_dip"] = random.random() < 0.1
    ui["rv_dip_plateau"] = random.random() < 0.05

    # Other sections
    ui["volume_challenge_done"] = False
    ui["vaso_test_done"] = False
    ui["ltot"] = random.choice([False, True]) if pattern in ("precap_ild", "cpcph") else False
    ui["ltot_flow_l_min"] = random.choice([1.0, 2.0, 3.0]) if ui["ltot"] else None

    # Infectio/Immunology
    ui["virology_pos"] = random.choice([False, False, True])
    ui["virology_desc"] = "HIV positiv." if ui["virology_pos"] else ""
    ui["immunology_pos"] = random.choice([False, True]) if pattern == "precap_ild" else False
    ui["immunology_desc"] = "ANA/ENA auffällig." if ui["immunology_pos"] else ""

    # Abdomen
    ui["abd_sono_done"] = random.choice([False, True])
    ui["abd_sono_desc"] = random.choice(["", "Stauungsleber möglich.", "Keine Auffälligkeiten."]) if ui["abd_sono_done"] else ""

    # Modules / procedere
    ui["modules"] = []
    ui["procedere_free"] = ""

    # Prior RHK
    if random.random() < 0.3:
        ui["prev_rhk_date"] = random.choice(["03/21", "11/22", "06/23"])
        ui["prev_label"] = random.choice(["stabiler Verlauf", "leicht progredient", "gebessert"])
        ui["prev_mpap"] = random.choice([18, 24, 30])
        ui["prev_pawp"] = random.choice([7, 12, 18])
        ui["prev_ci"] = random.choice([2.1, 2.8, 3.2])
        ui["prev_pvr"] = random.choice([1.5, 2.6, 4.2])
    else:
        ui["prev_rhk_date"] = ""
        ui["prev_label"] = ""
        ui["prev_mpap"] = None
        ui["prev_pawp"] = None
        ui["prev_ci"] = None
        ui["prev_pvr"] = None

    return ui


# =============================================================================
# JSON export/import helpers
# =============================================================================

def export_json(case: Dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False, indent=2)
    return path


def load_case_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Gradio UI
# =============================================================================

CSS = """
:root { --card-bg: rgba(255,255,255,0.96); --border: rgba(0,0,0,0.08); }
.gradio-container { max-width: 1450px !important; margin: 0 auto !important; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 2px 14px rgba(0,0,0,0.04);
}
.card-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
.row { display:flex; gap:16px; flex-wrap: wrap; margin: 6px 0; }
.badges { display:flex; gap:8px; flex-wrap:wrap; margin: 10px 0 0; }
.badge { padding: 5px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); background: rgba(0,0,0,0.03); }
.badge-blue { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.25); }
.badge-purple { background: rgba(168,85,247,0.12); border-color: rgba(168,85,247,0.25); }
.badge-orange { background: rgba(249,115,22,0.12); border-color: rgba(249,115,22,0.25); }
.badge-teal { background: rgba(20,184,166,0.12); border-color: rgba(20,184,166,0.25); }
.badge-red { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.25); }
.muted { color: rgba(0,0,0,0.55); }
.small { font-size: 12px; color: rgba(0,0,0,0.55); }
"""

def build_demo() -> Tuple[gr.Blocks, str, gr.Theme]:
    blocks = load_textdb_blocks()
    rules = load_rulebook(DEFAULT_RULEBOOK_PATH)

    theme = gr.themes.Soft()

    with gr.Blocks(title=APP_TITLE) as demo:
        # Header
        gr.Markdown(f"## {APP_TITLE}\n<span class='small'>Guideline-nahe Regel-Engine (YAML) + strukturierter Arztbefund + Patientenbericht in einfacher Sprache</span>")

        # Buttons top
        with gr.Row():
            btn_example_top = gr.Button("Beispiel laden (random)", variant="secondary")
            btn_generate_top = gr.Button("Befund erstellen/aktualisieren", variant="primary")
            btn_clear_top = gr.Button("Befund leeren", variant="secondary")
            save_btn_top = gr.Button("Fall speichern (.json)", variant="secondary")
            load_btn_top = gr.UploadButton("Fall laden (.json)", file_types=[".json"], variant="secondary")

        # Layout: left inputs, right outputs
        with gr.Row():
            with gr.Column(scale=7):
                tabs = gr.Tabs()

                field_components: Dict[str, gr.components.Component] = {}

                def add(name: str, comp: gr.components.Component):
                    field_components[name] = comp
                    return comp

                # ---- Tab 1: Klinik & Labor ----
                with gr.TabItem("Klinik & Labor", id=0):
                    with gr.Row():
                        add("firstname", gr.Textbox(label="Vorname"))
                        add("name", gr.Textbox(label="Name"))
                    with gr.Row():
                        add("age", gr.Number(label="Alter (Jahre)"))
                        add("sex", gr.Dropdown(label="Geschlecht", choices=["weiblich", "männlich", "divers"], value=None))
                    with gr.Row():
                        add("height_cm", gr.Number(label="Größe (cm)"))
                        add("weight_kg", gr.Number(label="Gewicht (kg)"))
                    add("story", gr.Textbox(label="Story / Kurz-Anamnese", lines=3))
                    with gr.Row():
                        add("ph_known", gr.Checkbox(label="PH-Diagnose bekannt"))
                        add("ph_suspected", gr.Checkbox(label="PH-Verdachtsdiagnose"))

                    gr.Markdown("### Symptome / Funktion")
                    with gr.Row():
                        add("who_fc", gr.Dropdown(label="WHO-FC", choices=["I", "II", "III", "IV"], value=None))
                        add("six_mwd_m", gr.Number(label="6MWD (m)"))
                        add("syncope", gr.Dropdown(label="Synkope", choices=["keine", "gelegentlich", "wiederholt"], value=None))
                    with gr.Row():
                        add("hemoptysis", gr.Checkbox(label="Hämoptyse"))
                        add("dizziness", gr.Checkbox(label="Schwindel"))
                        add("stairs_flights", gr.Number(label="Treppen (Etagen) bis Pause", precision=0))
                    add("daily_walk", gr.Textbox(label="Alltags-Gehstrecke (Freitext)", lines=1))

                    gr.Markdown("### Labor")
                    with gr.Row():
                        add("hb_g_dl", gr.Number(label="Hb (g/dl)"))
                        anemia_type = add("anemia_type", gr.Dropdown(
                            label="Anämie-Typ (falls Anämie vorliegt)",
                            choices=["mikrozytär", "normozytär", "makrozytär", "hämolytisch", "akute Blutung/Blutverlust", "unklar"],
                            value=None,
                            visible=False,
                        ))
                    with gr.Row():
                        add("crp_mg_l", gr.Number(label="CRP (mg/l)"))
                        add("leukocytes_g_l", gr.Number(label="Leukozyten (G/l)"))
                        add("platelets_g_l", gr.Number(label="Thrombozyten (G/l)"))
                    with gr.Row():
                        add("creatinine_mg_dl", gr.Number(label="Kreatinin (mg/dl)"))
                        add("inr", gr.Number(label="INR"))
                        add("ptt_s", gr.Number(label="PTT (s)"))
                    with gr.Row():
                        add("bnp_kind", gr.Dropdown(label="BNP/NT-proBNP", choices=["BNP", "NT-proBNP"], value="NT-proBNP"))
                        add("bnp_value", gr.Number(label="Wert (pg/ml)"))
                        add("entresto", gr.Checkbox(label="Entresto/ARNI? (BNP eingeschränkt)"))
                    with gr.Row():
                        add("congestive_organopathy", gr.Radio(label="Hinweis auf congestive Organopathie?", choices=["ja", "nein"], value=None))

                # ---- Tab 2: Bildgebung & Echo/CMR (merged) ----
                with gr.TabItem("Bildgebung & Echo/CMR", id=1):
                    gr.Markdown("### Thorax-Bildgebung")
                    with gr.Row():
                        add("ct_done", gr.Checkbox(label="CT Thorax/CT-Angio durchgeführt"))
                        add("vq_done", gr.Checkbox(label="V/Q durchgeführt"))
                    with gr.Row():
                        add("ct_ild", gr.Checkbox(label="ILD"))
                        add("ct_emphysema", gr.Checkbox(label="Emphysem"))
                        add("ct_embolie", gr.Checkbox(label="Embolie"))
                        add("ct_mosaic", gr.Checkbox(label="Mosaikperfusion"))
                        add("ct_koronarkalk", gr.Checkbox(label="Koronarkalk"))

                    with gr.Accordion("ILD – Details (nur bei ILD)", open=False) as acc_ild:
                        add("ild_type", gr.Textbox(label="Welche ILD?", lines=1))
                        with gr.Row():
                            add("ild_histology", gr.Checkbox(label="Histologisch gesichert?"))
                            add("ild_fibrosis_clinic", gr.Checkbox(label="An Fibroseambulanz angebunden?"))
                        add("ild_extent", gr.Dropdown(label="Ausmaß der ILD", choices=["gering", "mittel", "ausgedehnt"], value=None))

                    with gr.Accordion("V/Q – Details (nur bei V/Q)", open=False) as acc_vq:
                        with gr.Row():
                            add("vq_defect", gr.Checkbox(label="V/Q pathologisch (Perfusionsdefekte)"))
                            add("vq_desc", gr.Textbox(label="V/Q – Kurzbeschreibung", lines=2))

                    gr.Markdown("### Echokardiographie")
                    with gr.Row():
                        add("echo_done", gr.Checkbox(label="Echo durchgeführt"))
                        add("lvef", gr.Number(label="LV-EF (%)"))
                        add("la_enlarged", gr.Checkbox(label="Linksatrium erweitert"))
                    with gr.Row():
                        add("ee_ratio", gr.Number(label="E/e'"))
                        add("pasp_echo", gr.Number(label="sPAP Echo (mmHg)"))
                        add("tapse_mm", gr.Number(label="TAPSE (mm)"))
                        add("atrial_fib", gr.Checkbox(label="Vorhofflimmern"))
                    with gr.Row():
                        add("s_prime_cm_s", gr.Number(label="Trikuspidales S' (cm/s)"))
                        add("ra_esa_cm2", gr.Number(label="RA ESA (cm²)"))
                        add("rv_edd_mm", gr.Number(label="RV EDD (mm)", precision=0))
                    with gr.Row():
                        add("ivc_diam_mm", gr.Number(label="V. cava inferior Durchmesser (mm)"))
                        add("ivc_collapse", gr.Radio(label="VCI Kollaps >50%?", choices=["ja", "nein"], value=None))

                    gr.Markdown("### MRT / CMR (optional)")
                    with gr.Row():
                        add("cmr_done", gr.Checkbox(label="CMR durchgeführt"))
                        add("rvef", gr.Number(label="RV-EF (%)"))
                        add("rvesvi", gr.Number(label="RVESVi (ml/m²)"))

                # ---- Tab 3: Lungenfunktion ----
                with gr.TabItem("Lungenfunktion", id=2):
                    with gr.Row():
                        add("lufu_done", gr.Checkbox(label="Lufu durchgeführt"))
                        add("lufu_obstructive", gr.Checkbox(label="Obstruktiv"))
                        add("lufu_restrictive", gr.Checkbox(label="Restriktiv"))
                        add("lufu_diffusion", gr.Checkbox(label="Diffusionsstörung"))
                    with gr.Row():
                        add("fev1_l", gr.Number(label="FEV1 (l)"))
                        add("fvc_l", gr.Number(label="FVC (l)"))
                        add("dlco_sb", gr.Number(label="DLCO SB (optional)"))
                    add("lufu_summary", gr.Textbox(label="Lufu Summary (Freitext)", lines=3))

                # ---- Tab 4: RHK ----
                with gr.TabItem("RHK", id=3):
                    gr.Markdown("### Ruhehämodynamik")
                    with gr.Row():
                        add("spap_rest", gr.Number(label="sPAP (mmHg)"))
                        add("dpap_rest", gr.Number(label="dPAP (mmHg)"))
                        add("mpap_rest", gr.Number(label="mPAP (optional)"))
                    with gr.Row():
                        add("pawp_rest", gr.Number(label="PAWP (mmHg)"))
                        add("rap_rest", gr.Number(label="RAP (mmHg)"))
                    with gr.Row():
                        add("co_rest", gr.Number(label="CO (l/min)"))
                        add("ci_rest", gr.Number(label="CI (optional)"))
                        add("pvr_rest", gr.Number(label="PVR (optional, WU)"))

                    gr.Markdown("#### Auto-Berechnung (wird nach „Befund erstellen“ gefüllt)")
                    with gr.Row():
                        auto_mpap = gr.Number(label="mPAP (berechnet)", interactive=False)
                        auto_ci = gr.Number(label="CI (berechnet)", interactive=False)
                        auto_pvr = gr.Number(label="PVR (berechnet)", interactive=False)
                    with gr.Row():
                        auto_pvri = gr.Number(label="PVRi (berechnet)", interactive=False)
                        auto_tpg = gr.Number(label="TPG (berechnet)", interactive=False)
                        auto_dpg = gr.Number(label="DPG (berechnet)", interactive=False)

                    gr.Markdown("### Belastungshämodynamik (optional)")
                    with gr.Row():
                        add("exercise_done", gr.Checkbox(label="Belastung durchgeführt"))
                        add("spap_peak", gr.Number(label="sPAP Peak (mmHg)"))
                        add("dpap_peak", gr.Number(label="dPAP Peak (mmHg)"))
                        add("mpap_peak", gr.Number(label="mPAP Peak (optional)"))
                    with gr.Row():
                        add("pawp_peak", gr.Number(label="PAWP Peak (mmHg)"))
                        add("co_peak", gr.Number(label="CO Peak (l/min)"))
                        add("ci_peak", gr.Number(label="CI Peak (l/min/m²) (optional)"))

                    gr.Markdown("### Volumenchallenge (optional)")
                    with gr.Row():
                        add("volume_challenge_done", gr.Checkbox(label="Volumenchallenge durchgeführt"))
                        add("pawp_pre", gr.Number(label="PAWP pre (mmHg)"))
                        add("pawp_post", gr.Number(label="PAWP post (mmHg)"))
                    with gr.Row():
                        add("mpap_pre", gr.Number(label="mPAP pre (mmHg)"))
                        add("mpap_post", gr.Number(label="mPAP post (mmHg)"))

                    gr.Markdown("### Vasoreaktivität (optional)")
                    with gr.Row():
                        add("vaso_test_done", gr.Checkbox(label="Vasoreaktivität getestet"))
                        add("vaso_agent", gr.Textbox(label="Agent (z.B. iNO)", lines=1))
                    add("vaso_response_desc", gr.Textbox(label="Antwort / Kommentar", lines=2))

                    gr.Markdown("### Stufenoxymetrie (optional)")
                    with gr.Row():
                        add("sat_svc", gr.Number(label="SVC O2-Sättigung (%)"))
                        add("sat_ivc", gr.Number(label="IVC O2-Sättigung (%)"))
                        add("sat_ra", gr.Number(label="RA O2-Sättigung (%)"))
                    with gr.Row():
                        add("sat_rv", gr.Number(label="RV O2-Sättigung (%)"))
                        add("sat_pa", gr.Number(label="PA O2-Sättigung (%)"))
                        add("sat_ao", gr.Number(label="Aorta O2-Sättigung (%)"))

                    gr.Markdown("### Kurvenmorphologie (optional)")
                    with gr.Row():
                        add("wedge_v_wave", gr.Checkbox(label="Prominente V-Welle (PAWP)"))
                        add("wedge_a_wave", gr.Checkbox(label="Prominente A-Welle (PAWP)"))
                        add("rap_a_wave", gr.Checkbox(label="Prominente A-Welle (RAP)"))
                        add("rap_v_wave", gr.Checkbox(label="Prominente V-Welle (RAP)"))
                    with gr.Row():
                        add("rv_pseudo_dip", gr.Checkbox(label="Pseudo-Dip (RV-Kurve)"))
                        add("rv_dip_plateau", gr.Checkbox(label="Dip-Plateau (RV-Kurve)"))

                    gr.Markdown("### Vergleich (Vor-RHK, optional)")
                    with gr.Row():
                        add("prev_rhk_date", gr.Textbox(label="Vor-RHK (z.B. 03/21)"))
                        add("prev_label", gr.Textbox(label="Verlauf (z.B. stabiler Verlauf)"))
                    with gr.Row():
                        add("prev_mpap", gr.Number(label="mPAP vor (mmHg)"))
                        add("prev_pawp", gr.Number(label="PAWP vor (mmHg)"))
                        add("prev_ci", gr.Number(label="CI vor (l/min/m²)"))
                        add("prev_pvr", gr.Number(label="PVR vor (WU)"))

                # ---- Tab 5: Weitere Bereiche ----
                with gr.TabItem("Weitere Befunde", id=4):
                    gr.Markdown("### Blutgase / LTOT")
                    with gr.Row():
                        add("ltot", gr.Checkbox(label="LTOT vorhanden"))
                        ltot_flow = add("ltot_flow_l_min", gr.Number(label="LTOT (l/min)", visible=False))
                    gr.Markdown("### Infektiologie / Immunologie")
                    with gr.Row():
                        add("virology_pos", gr.Checkbox(label="Virologie positiv"))
                        viro_desc = add("virology_desc", gr.Textbox(label="Virologie – Details", lines=2, visible=False))
                    with gr.Row():
                        add("immunology_pos", gr.Checkbox(label="Immunologie positiv"))
                        immun_desc = add("immunology_desc", gr.Textbox(label="Immunologie – Details", lines=2, visible=False))

                    gr.Markdown("### Abdomen / Leber")
                    with gr.Row():
                        add("abd_sono_done", gr.Checkbox(label="Abdomen-Sono durchgeführt"))
                        abd_desc = add("abd_sono_desc", gr.Textbox(label="Besondere Befunde?", lines=2, visible=False))

                # ---- Tab 6: Procedere & Module ----
                with gr.TabItem("Procedere & Module", id=5):
                    gr.Markdown("### P-Module (optional)")
                    p_ids = sorted([bid for bid, b in blocks.items() if b.kind == "module" and bid.startswith("P")])

                    # Nutzerfreundliche Labels statt nur IDs
                    module_label_by_id = {pid: f"{pid} – {blocks[pid].title}" for pid in p_ids if pid in blocks}
                    module_id_by_label = {label: pid for pid, label in module_label_by_id.items()}
                    module_choices = [module_label_by_id[pid] for pid in p_ids if pid in module_label_by_id]

                    add("modules", gr.CheckboxGroup(label="Zusatzmodule auswählen", choices=module_choices))
                    add("procedere_free", gr.Textbox(label="Procedere – Freitext", lines=3))
                    gr.Markdown("Hinweis: Bereits durchgeführte Untersuchungen werden in den Modulen möglichst ausgefiltert (z.B. V/Q, CT, Echo, Lufu).")

            with gr.Column(scale=5):
                dashboard = gr.HTML(value=build_dashboard_html(None))
                with gr.Tabs():
                    with gr.TabItem("Arztbericht"):
                        out_doc = gr.Markdown()
                    with gr.TabItem("Patientenbericht"):
                        out_pat = gr.Markdown()
                    with gr.TabItem("Intern"):
                        out_int = gr.Markdown()
                    with gr.TabItem("Debug"):
                        out_json = gr.Code(language="json")

        # Buttons bottom (mirrored)
        with gr.Row():
            btn_example_bottom = gr.Button("Beispiel laden (random)", variant="secondary")
            btn_generate_bottom = gr.Button("Befund erstellen/aktualisieren", variant="primary")
            btn_clear_bottom = gr.Button("Befund leeren", variant="secondary")
            save_btn_bottom = gr.Button("Fall speichern (.json)", variant="secondary")
            load_btn_bottom = gr.UploadButton("Fall laden (.json)", file_types=[".json"], variant="secondary")

        file_out = gr.File(label="Download: gespeicherter Fall", visible=False)
        state_case = gr.State(value=None)

        # --- Conditional visibility bindings ---
        def _update_visibility_ild(ct_ild: bool):
            return (
                gr.update(visible=bool(ct_ild)),  # accordion open state cannot be updated; but content visible
            )

        # We can't update Accordion directly; instead show/hide inside by leaving content always, but ok.

        def _toggle_desc(flag: bool):
            return gr.update(visible=bool(flag))

        def _toggle_desc_text(flag: bool):
            return gr.update(visible=bool(flag))

        def _toggle_ltot(flag: bool):
            return gr.update(visible=bool(flag))

        def _toggle_anemia(hb_val, sex_val):
            hb = _safe_float(hb_val)
            anemia = _infer_anemia(sex_val, hb)
            return gr.update(visible=bool(anemia))
        field_components["virology_pos"].change(lambda x: _toggle_desc_text(x), inputs=[field_components["virology_pos"]], outputs=[viro_desc])
        field_components["immunology_pos"].change(lambda x: _toggle_desc_text(x), inputs=[field_components["immunology_pos"]], outputs=[immun_desc])
        field_components["abd_sono_done"].change(lambda x: _toggle_desc_text(x), inputs=[field_components["abd_sono_done"]], outputs=[abd_desc])
        field_components["ltot"].change(lambda x: _toggle_ltot(x), inputs=[field_components["ltot"]], outputs=[ltot_flow])

        # Anemia type show/hide when Hb or sex changes
        field_components["hb_g_dl"].change(_toggle_anemia, inputs=[field_components["hb_g_dl"], field_components["sex"]], outputs=[anemia_type])
        field_components["sex"].change(_toggle_anemia, inputs=[field_components["hb_g_dl"], field_components["sex"]], outputs=[anemia_type])

        # ILD details show/hide: show accordion content if ILD checked by just toggling its children? simplest: keep visible always; but we can open/close via markdown? We'll leave.

        # --- Helpers to map UI dict to component list ---
        input_components = [field_components[k] for k in field_components.keys()]
        input_keys = list(field_components.keys())

        def ui_get_raw(*vals):
            return {k: v for k, v in zip(input_keys, vals)}

        def apply_ui_to_components(ui_dict: Dict[str, Any]) -> List[Any]:
            out: List[Any] = []
            for k in input_keys:
                v = ui_dict.get(k)
                if k == "modules" and isinstance(v, list):
                    # Case-Files speichern Module als IDs; in der UI zeigen wir sprechende Labels.
                    v = [module_label_by_id.get(str(x), str(x)) for x in v]
                # Backward compatible mappings (alte Case-Files)
                if k == "syncope":
                    if isinstance(v, bool):
                        v = "gelegentlich" if v else "keine"
                if k == "anemia_type" and isinstance(v, str):
                    v_map = {
                        "microcytic": "mikrozytär",
                        "normocytic": "normozytär",
                        "macrocytic": "makrozytär",
                        "iron_deficiency": "mikrozytär",
                        "hemolytic": "hämolytisch",
                        "other": "unklar",
                    }
                    v = v_map.get(v, v)
                out.append(v)
            return out

        # --- Generate function ---
        def _generate(*vals):
            raw = ui_get_raw(*vals)
            # Module kommen aus der UI als Labels ("P01 – ..."); intern arbeiten wir mit IDs.
            mods = raw.get("modules") or []
            if isinstance(mods, list):
                raw["modules"] = [module_id_by_label.get(str(m), str(m)) for m in mods]
            case = build_case(raw, rules)

            doc = build_doctor_report(case, blocks)
            pat = build_patient_report(case)
            internal = build_internal_report(case)
            dash = build_dashboard_html(case)

            # computed outputs
            der = case["derived"]
            ci_calc = None
            if der.get("co") is not None and der.get("bsa_m2") is not None and der.get("bsa_m2"):
                try:
                    ci_calc = float(der.get("co")) / float(der.get("bsa_m2"))
                except Exception:
                    ci_calc = None

            return (
                der.get("mpap_calc"), ci_calc, der.get("pvr_calc"), der.get("pvri"), der.get("tpg"), der.get("dpg"),
                dash, doc, pat, internal,
                json.dumps(case, ensure_ascii=False, indent=2),
                case,
            )

        generate_outputs = [auto_mpap, auto_ci, auto_pvr, auto_pvri, auto_tpg, auto_dpg, dashboard, out_doc, out_pat, out_int, out_json, state_case]

        btn_generate_top.click(_generate, inputs=input_components, outputs=generate_outputs)
        btn_generate_bottom.click(_generate, inputs=input_components, outputs=generate_outputs)

        # --- Example loader ---
        def _load_example():
            ui = random_example()
            vals = apply_ui_to_components(ui)
            gen = _generate(*vals)
            return (*vals, *gen)

        load_outputs = input_components + generate_outputs

        btn_example_top.click(_load_example, inputs=[], outputs=load_outputs)
        btn_example_bottom.click(_load_example, inputs=[], outputs=load_outputs)

        # --- Clear / reset ---
        def _clear():
            empty_ui: Dict[str, Any] = {k: (False if isinstance(field_components[k], gr.Checkbox) else None) for k in input_keys}
            # Textboxes should be ""
            for k, comp in field_components.items():
                if isinstance(comp, gr.Textbox):
                    empty_ui[k] = ""
                if isinstance(comp, gr.CheckboxGroup):
                    empty_ui[k] = []
                if isinstance(comp, gr.Radio) or isinstance(comp, gr.Dropdown):
                    empty_ui[k] = None
            vals = apply_ui_to_components(empty_ui)
            dash = build_dashboard_html(None)
            return (*vals, None, None, None, None, None, None, dash, "", "", "", "{}", {})

        btn_clear_top.click(_clear, inputs=[], outputs=load_outputs)
        btn_clear_bottom.click(_clear, inputs=[], outputs=load_outputs)

        # --- Save case ---
        def _save_case(case_state):
            if not case_state:
                return gr.update(visible=False, value=None)
            # Store full case with ui/derived/decision; but saving 'ui' is enough; keep full for debugging
            path = os.path.join("/tmp", f"rhk_case_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            export_json(case_state, path)
            return gr.update(visible=True, value=path)

        save_btn_top.click(_save_case, inputs=[state_case], outputs=[file_out])
        save_btn_bottom.click(_save_case, inputs=[state_case], outputs=[file_out])

        # --- Load case ---
        def _load_case(file):
            if file is None:
                return
            fp = file.name if hasattr(file, "name") else str(file)
            data = load_case_json(fp)
            # Accept either full case or ui-only
            ui_dict = data.get("ui") if isinstance(data, dict) and "ui" in data else data
            if not isinstance(ui_dict, dict):
                ui_dict = {}
            vals = apply_ui_to_components(ui_dict)
            gen = _generate(*vals)
            return (*vals, *gen)

        load_btn_top.upload(_load_case, inputs=[load_btn_top], outputs=load_outputs)
        load_btn_bottom.upload(_load_case, inputs=[load_btn_bottom], outputs=load_outputs)

    return demo, CSS, theme


# =============================================================================
# Main
# =============================================================================

def _find_free_port(preferred: int) -> int:
    import socket
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return preferred


def main():
    demo, css, theme = build_demo()
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    port = _find_free_port(port)

    # Gradio 6+: css/theme are passed to launch; older versions accept in Blocks too (we use launch for forward-compat).
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
