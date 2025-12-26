#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_app_web_master_v6.py

RHK Befundassistent (Web-GUI, Gradio) – "Master" Version

Features (Auszug):
- Befund (intern/ärztlich), Patienten-Info (sehr einfache Sprache) + interner Log.
- Strukturierte Eingabe (Klinik/Labor zuerst), RHK (Ruhe/Belastung/Manöver), Lufu, Echo/CMR.
- Automatische Berechnungen (mPAP aus sPAP/dPAP, CI aus CO/BSA, PVR, TPG/DPG, SVI, PVRI).
- Belastungslogik inkl. mPAP/CO- und PAWP/CO-Slope; Homeometrisch vs. heterometrisch (wenn Belastung).
- Risiko prominent: ESC/ERS 3‑Strata (erweitert, inkl. Hämodynamik/Echo/CMR wenn vorhanden),
  ESC/ERS 4‑Strata (follow-up, nichtinvasiv), REVEAL Lite 2.
- HFpEF‑Wahrscheinlichkeit: H2FPEF‑Score (optional) mit Hinweis in Empfehlungen.
- Yogeswaran et al.: S'/RAAI (S' + RA‑Fläche + BSA) inkl. Cutoff.

Wichtiger Hinweis:
Dieses Tool ist eine Dokumentations‑/Formulierungshilfe. Es ersetzt keine ärztliche Entscheidung.
"""

from __future__ import annotations

APP_VERSION = "v14.1"

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import sys as _sys
import pathlib as _pathlib

# In notebooks/ipykernel there is no __file__. Fall back to CWD so the app can
# also be executed from a Jupyter cell (useful for debugging).
try:
    _HERE = _pathlib.Path(__file__).resolve().parent  # type: ignore[name-defined]
except Exception:
    _HERE = _pathlib.Path.cwd()

if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

# ---------------------------
# Textdatenbank (ärztlich)
# ---------------------------
try:
    import rhk_textdb as textdb  # type: ignore
except Exception:
    # Fallback: lokale Datei im selben Ordner (wenn als Skript verteilt)
    import importlib.util as _importlib_util

    _candidate = _HERE / "rhk_textdb.py"
    spec = _importlib_util.spec_from_file_location("rhk_textdb", str(_candidate))
    if spec and spec.loader:
        textdb = _importlib_util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(textdb)  # type: ignore
    else:
        raise

DEFAULT_RULES = getattr(textdb, "DEFAULT_RULES", {})


# ---------------------------
# UI-Feldreihenfolge
# (MUSS exakt zur Reihenfolge der Input-Komponenten in der Blocks-GUI passen)
# ---------------------------
UI_FIELDS = [
    # Klinik/Labor
    "last_name","first_name","birthdate","height_cm","weight_kg","bsa_m2","story",
    "ph_known","ph_suspected","la_enlarged",
    "inr","quick","crea","hst","ptt","plt","hb","crp","leuko",
    "bnp_kind","bnp_value","congestive_organopathy",
    "ltot_present","bga_rest_pO2","bga_rest_pCO2",
    "virology_positive","immunology_positive",
    "abdo_sono","portal_hypertension",
    "ct_angio","ct_lae","ct_ild","ct_emphysema","ct_embolie","ct_mosaic","ct_coronarycalc",
    "comorbidities","comorbidities_relevance",
    "ph_meds_yesno","ph_meds_past_yesno","diuretics_yesno",
    "ph_meds_which","ph_meds_since","other_meds",
    "who_fc","syncope","sixmwd_m","ve_vco2","vo2max","sbp","egfr",
    "hfpef_af","hfpef_htn_meds","hfpef_e_eprime","hfpef_pasp",
    # RHK Ruhe
    "mpap","pa_sys","pa_dia","pawp","rap","co","ci","pvr","svi","hr","svo2",
    "svc_sat","ivc_sat","ra_sat","rv_sat","pa_sat",
    # Belastung/Manöver
    "exercise_done","exercise_ph","ex_mpap","ex_pa_sys","ex_pa_dia","ex_pawp","ex_co","ex_ci","ex_pvr","ex_hr","mpap_co_slope","pawp_co_slope",
    "volume_done","volume_positive","volume_ml","volume_pre_pawp","volume_post_pawp",
    "vaso_done","vaso_responder","ino_ppm","vaso_pre_mpap","vaso_post_mpap","vaso_pre_pvr","vaso_post_pvr",
    # Lufu
    "lufu_done","lufu_obstr","lufu_restr","lufu_diff","lufu_fev1","lufu_fvc","lufu_fev1_fvc","lufu_tlc","lufu_rv","lufu_dlco","lufu_summary",
    # Echo/CMR
    "echo_sprime","echo_ra_area","pericard_eff","pericard_eff_grade",
    "cmr_rvesvi","cmr_svi","cmr_rvef",
    # Verlauf/Abschluss
    "prev_rhk_label","prev_rhk_course","prev_mpap","prev_pawp","prev_ci","prev_pvr",
    "therapy_plan_sentence","anticoag_plan_sentence","followup_timing_desc","declined_item","study_sentence",
    "modules",
]


def _labels_to_ids(labels: List[str]) -> List[str]:
    """Converts checkbox labels like 'P01 – Titel' to bare IDs like 'P01'."""
    ids: List[str] = []
    for lab in labels or []:
        s = str(lab or "").strip()
        if not s:
            continue
        for sep in ("–", "—", "-"):
            if sep in s:
                s = s.split(sep, 1)[0].strip()
                break
        if " " in s:
            s = s.split(" ", 1)[0].strip()
        if s and s not in ids:
            ids.append(s)
    return ids




def _rule_get(path: str, default=None):
    """Liest Regeln aus textdb.DEFAULT_RULES (unterstützt verschachtelte Dict-Struktur)."""
    cur = DEFAULT_RULES
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _rule_float(path: str, default: Optional[float] = None) -> Optional[float]:
    """Wie _rule_get, aber als float (oder default)."""
    val = _rule_get(path, default)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


# ---------------------------
# Patienten‑Textdatenbank
# ---------------------------
patientdb = None
for _mod in ("rhk_textdb_patient_v6", "rhk_textdb_patient_v5", "rhk_textdb_patient_v4", "rhk_textdb_patient_v3", "rhk_textdb_patient_v2", "rhk_textdb_patient"):
    try:
        patientdb = __import__(_mod, fromlist=["get_patient_block"])
        break
    except Exception:
        continue


# ---------------------------
# Helpers: parsing/formatting
# ---------------------------
def to_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # deutsche Dezimal-Kommas akzeptieren
    s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def to_int(x: Any) -> Optional[int]:
    v = to_num(x)
    if v is None:
        return None
    try:
        return int(round(v))
    except Exception:
        return None


def parse_date_any(x: Any) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    if not s:
        return None
    # akzeptiere: YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def calc_age_years(born: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if born is None:
        return None
    today = today or date.today()
    try:
        years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return int(years)
    except Exception:
        return None


def calc_bsa_m2(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    # Du Bois
    if height_cm is None or weight_kg is None:
        return None
    try:
        return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
    except Exception:
        return None


def calc_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if weight_kg is None or height_cm is None:
        return None
    try:
        h_m = height_cm / 100.0
        if h_m <= 0:
            return None
        return weight_kg / (h_m * h_m)
    except Exception:
        return None


def fmt_num(v: Any, decimals: int = 2) -> str:
    """
    Zahlformat in deutscher Darstellung (Dezimal-Komma).
    Gibt bei None einen leeren String zurück (damit Textbausteine nicht "kaputt" aussehen).
    """
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return ""
    if decimals <= 0:
        return str(int(round(f)))
    # Integer? dann ohne Nachkommastellen
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    s = f"{f:.{decimals}f}"
    return s.replace(".", ",")


def fmt_unit(v: Any, unit: str, decimals: int = 2) -> str:
    s = fmt_num(v, decimals)
    if not s:
        return ""
    return f"{s} {unit}"


def join_nonempty(parts: List[str], sep: str = " ") -> str:
    return sep.join([p for p in parts if (p or "").strip()])


# ---------------------------
# Physiologie / Berechnungen
# ---------------------------
def calc_mpap_from_spap_dpap(spap: Optional[float], dpap: Optional[float]) -> Optional[float]:
    # Näherungsformel: MAP ≈ (SYS + 2*DIA)/3
    if spap is None or dpap is None:
        return None
    try:
        return (spap + 2.0 * dpap) / 3.0
    except Exception:
        return None


def calc_tpg(mpap: Optional[float], pawp: Optional[float]) -> Optional[float]:
    if mpap is None or pawp is None:
        return None
    return mpap - pawp


def calc_dpg(dpap: Optional[float], pawp: Optional[float]) -> Optional[float]:
    if dpap is None or pawp is None:
        return None
    return dpap - pawp


def calc_ci_from_co_bsa(co: Optional[float], bsa: Optional[float]) -> Optional[float]:
    if co is None or bsa is None or bsa <= 0:
        return None
    return co / bsa


def calc_co_from_ci_bsa(ci: Optional[float], bsa: Optional[float]) -> Optional[float]:
    if ci is None or bsa is None or bsa <= 0:
        return None
    return ci * bsa


def calc_pvr_wu(mpap: Optional[float], pawp: Optional[float], co: Optional[float]) -> Optional[float]:
    if mpap is None or pawp is None or co is None or co <= 0:
        return None
    return (mpap - pawp) / co


def calc_pvri_wu_m2(mpap: Optional[float], pawp: Optional[float], ci: Optional[float]) -> Optional[float]:
    if mpap is None or pawp is None or ci is None or ci <= 0:
        return None
    return (mpap - pawp) / ci


def calc_svi_ml_m2(ci: Optional[float], hr: Optional[float]) -> Optional[float]:
    # SVI = CI*1000/HR   (mL/beat/m²)
    if ci is None or hr is None or hr <= 0:
        return None
    return ci * 1000.0 / hr


def calc_sprime_raai(
    sprime_cm_s: Optional[float],
    ra_area_cm2: Optional[float],
    bsa_m2: Optional[float],
) -> Optional[float]:
    """
    S'/RAAI = S' / (RA area / BSA) = S' * BSA / RA area
    Einheit: m²/(s·cm) wenn S' in cm/s und RA area in cm².
    """
    if sprime_cm_s is None or ra_area_cm2 is None or bsa_m2 is None:
        return None
    if ra_area_cm2 <= 0:
        return None
    return sprime_cm_s * bsa_m2 / ra_area_cm2


def detect_step_up(sats: Dict[str, Optional[float]]) -> Tuple[bool, str]:
    """
    Einfache Step-up Heuristik.
    Returns: (step_up_bool, sentence)
    """
    # Keys expected: svc, ivc, ra, rv, pa
    svc = sats.get("svc")
    ivc = sats.get("ivc")
    ra = sats.get("ra")
    rv = sats.get("rv")
    pa = sats.get("pa")

    # step-up: RA - SVC >= 7% oder RV - RA >= 5% oder PA - RV >= 5%
    msgs = []
    flag = False
    if ra is not None and svc is not None and (ra - svc) >= 7:
        flag = True
        msgs.append("O₂‑Sättigung steigt im rechten Vorhof deutlich an.")
    if rv is not None and ra is not None and (rv - ra) >= 5:
        flag = True
        msgs.append("O₂‑Sättigung steigt im rechten Ventrikel deutlich an.")
    if pa is not None and rv is not None and (pa - rv) >= 5:
        flag = True
        msgs.append("O₂‑Sättigung steigt in der Pulmonalarterie deutlich an.")
    if not msgs:
        msgs = ["Kein eindeutiger O₂‑Step‑up."]
    return flag, " ".join(msgs)


# ---------------------------
# HFpEF: H2FPEF‑Score (Reddy et al.)
# ---------------------------
@dataclass(frozen=True)
class H2FPEFResult:
    score: int
    category: str
    details: List[str]


def h2fpef_score(
    age_years: Optional[int],
    bmi: Optional[float],
    af: Optional[bool],
    antihypertensives_n: Optional[int],
    pasp_mmHg: Optional[float],
    e_over_eprime: Optional[float],
) -> Optional[H2FPEFResult]:
    """
    H2FPEF Score (0–9):
    - Heavy (BMI >30): 2
    - Hypertensive (≥2 Antihypertensiva): 1
    - Atrial Fibrillation: 3
    - Pulmonary hypertension (PASP >35): 1
    - Elder (Age >60): 1
    - Filling pressure (E/e' >9): 1

    Kategorien (vereinfachte Interpretation):
    - 0–1: niedrige Wahrscheinlichkeit
    - 2–5: mittlere Wahrscheinlichkeit
    - 6–9: hohe Wahrscheinlichkeit
    """
    # mindestens 2 Parameter nötig, sonst macht es wenig Sinn
    available = sum(v is not None for v in [age_years, bmi, af, antihypertensives_n, pasp_mmHg, e_over_eprime])
    if available < 2:
        return None

    score = 0
    details: List[str] = []

    if bmi is not None and bmi > 30:
        score += 2
        details.append("BMI > 30 → +2")
    if antihypertensives_n is not None and antihypertensives_n >= 2:
        score += 1
        details.append("≥2 Blutdruck‑Medikamente → +1")
    if af is True:
        score += 3
        details.append("Vorhofflimmern → +3")
    if pasp_mmHg is not None and pasp_mmHg > 35:
        score += 1
        details.append("PASP/sPAP > 35 mmHg → +1")
    if age_years is not None and age_years > 60:
        score += 1
        details.append("Alter > 60 → +1")
    if e_over_eprime is not None and e_over_eprime > 9:
        score += 1
        details.append("E/e' > 9 → +1")

    if score <= 1:
        cat = "niedrige Wahrscheinlichkeit"
    elif score <= 5:
        cat = "mittlere Wahrscheinlichkeit"
    else:
        cat = "hohe Wahrscheinlichkeit"

    return H2FPEFResult(score=score, category=cat, details=details)


# ---------------------------
# Risiko‑Scores
# ---------------------------
@dataclass(frozen=True)
class RiskResult:
    label: str
    value: Any
    details: List[str]


def esc3_grade_who_fc(who_fc: Optional[str]) -> Optional[int]:
    if not who_fc:
        return None
    s = str(who_fc).strip().upper()
    if s in ("I", "1", "I–II", "I-II", "II", "2"):
        return 1
    if s in ("III", "3"):
        return 2
    if s in ("IV", "4"):
        return 3
    return None


def esc3_grade_6mwd(m: Optional[float]) -> Optional[int]:
    if m is None:
        return None
    if m > 440:
        return 1
    if m >= 165:
        return 2
    return 3


def esc3_grade_bnp(bnp: Optional[float]) -> Optional[int]:
    if bnp is None:
        return None
    if bnp < 50:
        return 1
    if bnp <= 800:
        return 2
    return 3


def esc3_grade_ntprobnp(nt: Optional[float]) -> Optional[int]:
    if nt is None:
        return None
    if nt < 300:
        return 1
    if nt <= 1100:
        return 2
    return 3


def esc3_grade_rap(rap: Optional[float]) -> Optional[int]:
    if rap is None:
        return None
    if rap < 8:
        return 1
    if rap <= 14:
        return 2
    return 3


def esc3_grade_ci(ci: Optional[float]) -> Optional[int]:
    if ci is None:
        return None
    if ci >= 2.5:
        return 1
    if ci >= 2.0:
        return 2
    return 3


def esc3_grade_svi(svi: Optional[float]) -> Optional[int]:
    if svi is None:
        return None
    if svi > 38:
        return 1
    if svi >= 31:
        return 2
    return 3


def esc3_grade_svo2(svo2: Optional[float]) -> Optional[int]:
    if svo2 is None:
        return None
    if svo2 > 65:
        return 1
    if svo2 >= 60:
        return 2
    return 3


def esc3_grade_ra_area(ra_cm2: Optional[float]) -> Optional[int]:
    if ra_cm2 is None:
        return None
    if ra_cm2 < 18:
        return 1
    if ra_cm2 <= 26:
        return 2
    return 3


def esc3_grade_pericard_eff(pe_grade: Optional[str], pe_bool: Optional[bool] = None) -> Optional[int]:
    """
    none / minimal / moderate-large.
    Wenn nur bool vorhanden: True → 3, False → 1.
    """
    if pe_grade:
        s = str(pe_grade).strip().lower()
        if s in ("none", "kein", "nein", "0"):
            return 1
        if s in ("minimal", "klein", "gering"):
            return 2
        if s in ("moderate", "large", "mittel", "groß", "moderat-groß", "moderat", "mittel-groß"):
            return 3
    if pe_bool is None:
        return None
    return 3 if pe_bool else 1


def esc3_grade_cmr_rvef(rvef: Optional[float]) -> Optional[int]:
    # Cutoffs aus ESC/ERS Tabelle (RVEF >54 / 37–54 / <37)
    if rvef is None:
        return None
    if rvef > 54:
        return 1
    if rvef >= 37:
        return 2
    return 3


def esc3_grade_cmr_rvesvi(rv_esvi: Optional[float]) -> Optional[int]:
    # <42 / 42–54 / >54
    if rv_esvi is None:
        return None
    if rv_esvi < 42:
        return 1
    if rv_esvi <= 54:
        return 2
    return 3


def esc3_grade_cmr_svi(svi: Optional[float]) -> Optional[int]:
    # >40 / 26–40 / <26
    if svi is None:
        return None
    if svi > 40:
        return 1
    if svi >= 26:
        return 2
    return 3


def esc3_overall_extended(
    who_fc: Optional[str],
    sixmwd_m: Optional[float],
    bnp_kind: Optional[str],
    bnp_value: Optional[float],
    rap: Optional[float],
    ci: Optional[float],
    hr: Optional[float],
    svo2: Optional[float],
    ra_area_cm2: Optional[float],
    pericard_eff_grade: Optional[str],
    pericard_eff_bool: Optional[bool],
    cmr_rvef: Optional[float],
    cmr_rvesvi: Optional[float],
    cmr_svi: Optional[float],
) -> Optional[RiskResult]:
    """
    ESC/ERS 3‑Strata (erweitert): Durchschnitt der verfügbaren Einzel‑Grades (1–3), Ceiling → Gesamtrisiko.
    """
    grades: List[Tuple[str, int]] = []

    g = esc3_grade_who_fc(who_fc)
    if g:
        grades.append(("WHO‑FC", g))
    g = esc3_grade_6mwd(sixmwd_m)
    if g:
        grades.append(("6MWD", g))

    # BNP/NT-proBNP
    if bnp_value is not None:
        if (bnp_kind or "").strip().lower().startswith("nt"):
            g = esc3_grade_ntprobnp(bnp_value)
            if g:
                grades.append(("NT‑proBNP", g))
        else:
            g = esc3_grade_bnp(bnp_value)
            if g:
                grades.append(("BNP", g))

    g = esc3_grade_rap(rap)
    if g:
        grades.append(("RAP", g))
    g = esc3_grade_ci(ci)
    if g:
        grades.append(("CI", g))

    svi = calc_svi_ml_m2(ci, hr) if (ci is not None and hr is not None) else None
    g = esc3_grade_svi(svi)
    if g:
        grades.append(("SVI (aus CI/HR)", g))

    g = esc3_grade_svo2(svo2)
    if g:
        grades.append(("SvO₂", g))

    g = esc3_grade_ra_area(ra_area_cm2)
    if g:
        grades.append(("RA‑Fläche", g))

    g = esc3_grade_pericard_eff(pericard_eff_grade, pericard_eff_bool)
    if g:
        grades.append(("Perikarderguss", g))

    # CMR‑Marker (nur wenn vorhanden)
    g = esc3_grade_cmr_rvef(cmr_rvef)
    if g:
        grades.append(("CMR RVEF", g))
    g = esc3_grade_cmr_rvesvi(cmr_rvesvi)
    if g:
        grades.append(("CMR RVESVi", g))
    g = esc3_grade_cmr_svi(cmr_svi)
    if g:
        grades.append(("CMR SVi", g))

    if len(grades) < 2:
        return None

    mean_grade = sum(g for _, g in grades) / len(grades)
    overall = int(math.ceil(mean_grade))
    overall = max(1, min(3, overall))
    label = {1: "Niedrig", 2: "Intermediär", 3: "Hoch"}[overall]
    details = [f"{name}: {g}" for name, g in grades] + [f"Ø‑Grade: {fmt_num(mean_grade,2)} → {overall}"]
    return RiskResult(label=label, value=overall, details=details)


# ESC/ERS 4‑Strata (follow-up; non-invasive)
def esc4_grade_who_fc(who_fc: Optional[str]) -> Optional[int]:
    if not who_fc:
        return None
    s = str(who_fc).strip().upper()
    if s in ("I", "1", "II", "2"):
        return 1
    if s in ("III", "3"):
        return 3
    if s in ("IV", "4"):
        return 4
    return None


def esc4_grade_6mwd(m: Optional[float]) -> Optional[int]:
    if m is None:
        return None
    if m > 440:
        return 1
    if m >= 320:
        return 2
    if m >= 165:
        return 3
    return 4


def esc4_grade_bnp(bnp: Optional[float]) -> Optional[int]:
    if bnp is None:
        return None
    if bnp < 50:
        return 1
    if bnp <= 199:
        return 2
    if bnp <= 800:
        return 3
    return 4


def esc4_grade_ntprobnp(nt: Optional[float]) -> Optional[int]:
    if nt is None:
        return None
    if nt < 300:
        return 1
    if nt <= 649:
        return 2
    if nt <= 1100:
        return 3
    return 4


def esc4_overall(
    who_fc: Optional[str],
    sixmwd_m: Optional[float],
    bnp_kind: Optional[str],
    bnp_value: Optional[float],
) -> Optional[RiskResult]:
    grades: List[Tuple[str, int]] = []
    g = esc4_grade_who_fc(who_fc)
    if g:
        grades.append(("WHO‑FC", g))
    g = esc4_grade_6mwd(sixmwd_m)
    if g:
        grades.append(("6MWD", g))
    if bnp_value is not None:
        if (bnp_kind or "").strip().lower().startswith("nt"):
            g = esc4_grade_ntprobnp(bnp_value)
            if g:
                grades.append(("NT‑proBNP", g))
        else:
            g = esc4_grade_bnp(bnp_value)
            if g:
                grades.append(("BNP", g))

    if len(grades) < 2:
        return None

    mean_grade = sum(g for _, g in grades) / len(grades)
    overall = int(math.ceil(mean_grade))
    overall = max(1, min(4, overall))
    label_map = {1: "Niedrig", 2: "Intermediär‑niedrig", 3: "Intermediär‑hoch", 4: "Hoch"}
    label = label_map[overall]
    details = [f"{name}: {g}" for name, g in grades] + [f"Ø‑Grade: {fmt_num(mean_grade,2)} → {overall}"]
    return RiskResult(label=label, value=overall, details=details)


# REVEAL Lite 2 (wie in v5)
def reveal_lite2_score(
    who_fc: Optional[str],
    sixmwd_m: Optional[float],
    bnp_kind: Optional[str],
    bnp_value: Optional[float],
    sbp: Optional[float],
    hr: Optional[float],
    egfr: Optional[float],
) -> Optional[RiskResult]:
    # Minimal: mindestens 3 Parameter
    available = sum(v is not None for v in [who_fc, sixmwd_m, bnp_value, sbp, hr, egfr])
    if available < 3:
        return None

    score = 0
    details = []

    # WHO-FC
    s = (who_fc or "").strip().upper()
    if s in ("I", "1", "II", "2"):
        details.append("WHO‑FC I/II: +0")
    elif s in ("III", "3"):
        score += 1
        details.append("WHO‑FC III: +1")
    elif s in ("IV", "4"):
        score += 2
        details.append("WHO‑FC IV: +2")

    # 6MWD
    if sixmwd_m is not None:
        if sixmwd_m >= 440:
            details.append("6MWD ≥440 m: +0")
        elif sixmwd_m >= 165:
            score += 1
            details.append("6MWD 165–439 m: +1")
        else:
            score += 2
            details.append("6MWD <165 m: +2")

    # BNP/NT-proBNP
    if bnp_value is not None:
        if (bnp_kind or "").strip().lower().startswith("nt"):
            if bnp_value < 300:
                details.append("NT‑proBNP <300: +0")
            elif bnp_value <= 1100:
                score += 1
                details.append("NT‑proBNP 300–1100: +1")
            else:
                score += 2
                details.append("NT‑proBNP >1100: +2")
        else:
            if bnp_value < 50:
                details.append("BNP <50: +0")
            elif bnp_value <= 200:
                score += 1
                details.append("BNP 50–200: +1")
            else:
                score += 2
                details.append("BNP >200: +2")

    # SBP
    if sbp is not None:
        if sbp >= 110:
            details.append("RRsys ≥110: +0")
        elif sbp >= 95:
            score += 1
            details.append("RRsys 95–109: +1")
        else:
            score += 2
            details.append("RRsys <95: +2")

    # HR
    if hr is not None:
        if hr < 96:
            details.append("HF <96/min: +0")
        elif hr <= 110:
            score += 1
            details.append("HF 96–110/min: +1")
        else:
            score += 2
            details.append("HF >110/min: +2")

    # eGFR
    if egfr is not None:
        if egfr >= 60:
            details.append("eGFR ≥60: +0")
        elif egfr >= 30:
            score += 1
            details.append("eGFR 30–59: +1")
        else:
            score += 2
            details.append("eGFR <30: +2")

    if score <= 3:
        label = "Niedrig"
    elif score <= 6:
        label = "Intermediär"
    else:
        label = "Hoch"

    details.append(f"Summe: {score}")
    return RiskResult(label=label, value=score, details=details)


def render_risk_html(
    esc3_ext: Optional[RiskResult],
    esc4: Optional[RiskResult],
    reveal: Optional[RiskResult],
    hfpef: Optional[H2FPEFResult],
) -> str:
    def pill(text: str) -> str:
        return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #ccc;margin-left:6px;font-size:12px;'>{text}</span>"

    def box(title: str, body: str) -> str:
        return (
            "<div style='border:1px solid #e5e7eb;border-radius:14px;padding:12px 14px;margin:10px 0;background:#fff;'>"
            f"<div style='font-weight:700;font-size:14px;margin-bottom:6px;'>{title}</div>"
            f"<div style='font-size:13px;line-height:1.35;'>{body}</div>"
            "</div>"
        )

    blocks: List[str] = []

    if esc3_ext:
        body = f"{pill(esc3_ext.label)}<br>"
        body += " / ".join(esc3_ext.details[: min(6, len(esc3_ext.details))])
        if len(esc3_ext.details) > 6:
            body += "<br><span style='color:#6b7280;'>… weitere Parameter berücksichtigt.</span>"
        blocks.append(box("ESC/ERS 3‑Strata (erweitert)", body))
    else:
        blocks.append(box("ESC/ERS 3‑Strata (erweitert)", "Nicht berechenbar (zu wenig Daten)."))

    if esc4:
        body = f"{pill(esc4.label)}<br>" + " / ".join(esc4.details)
        blocks.append(box("ESC/ERS 4‑Strata (Follow‑up, nichtinvasiv)", body))
    else:
        blocks.append(box("ESC/ERS 4‑Strata (Follow‑up, nichtinvasiv)", "Nicht berechenbar (zu wenig Daten)."))

    if reveal:
        body = f"{pill(reveal.label)}<br>REVEAL Lite 2: {reveal.value} Punkte<br>"
        body += "<span style='color:#6b7280;'>" + " / ".join(reveal.details[: min(6, len(reveal.details))]) + "</span>"
        blocks.append(box("REVEAL Lite 2", body))
    else:
        blocks.append(box("REVEAL Lite 2", "Nicht berechenbar (zu wenig Daten)."))

    if hfpef:
        body = f"{pill(hfpef.category)}<br>H2FPEF: {hfpef.score}/9"
        if hfpef.details:
            body += "<br><span style='color:#6b7280;'>" + " / ".join(hfpef.details) + "</span>"
        blocks.append(box("HFpEF‑Wahrscheinlichkeit (H2FPEF)", body))
    else:
        blocks.append(box("HFpEF‑Wahrscheinlichkeit (H2FPEF)", "Nicht berechenbar / nicht eingegeben."))

    header = (
        "<div style='border-left:4px solid #111827;padding:8px 12px;margin:6px 0 12px 0;'>"
        "<div style='font-weight:800;font-size:16px;'>Risiko‑Scores</div>"
        "<div style='color:#6b7280;font-size:12px;'>Prominent dargestellt; Details im internen Log.</div>"
        "</div>"
    )
    return "<div style='background:#f9fafb;padding:10px 12px;border-radius:16px;'>" + header + "".join(blocks) + "</div>"


# ---------------------------
# Patientensprache: Fallback‑Vereinfachung
# ---------------------------
def simplify_text_for_patient_fallback(text: str) -> str:
    if not text:
        return ""

    t = text

    # Abkürzungen/Begriffe möglichst ersetzen
    replacements = {
        "pulmonale Hypertonie": "Lungenhochdruck",
        "Pulmonale Hypertonie": "Lungenhochdruck",
        "Rechtsherzkatheter": "Herzkatheter (Rechtsherzkatheter)",
        "mPAP": "Druck in der Lunge (mPAP)",
        "PAWP": "Druck am linken Herzen (PAWP)",
        "PVR": "Widerstand in den Lungengefäßen (PVR)",
        "CI": "Pumpleistung (CI)",
        "präkapillär": "in den Lungengefäßen",
        "postkapillär": "vom linken Herzen her",
        "kombiniert post-/präkapillär": "vom linken Herzen her und zusätzlich in den Lungengefäßen",
        "Vasoreagibilität": "Reaktion auf ein Test‑Gas",
        "Volumenchallenge": "Flüssigkeits‑Test",
        "Belastung": "körperliche Belastung",
        "Echokardiographie": "Herz‑Ultraschall",
        "Katheter": "Katheter",
        "WU": "Widerstands‑Einheit",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)

    # Klammer‑Kaskaden reduzieren
    t = re.sub(r"\s*\([^)]{40,}\)", "", t)  # sehr lange Klammern raus
    # Kürzen / Sätze trennen
    t = t.replace(";", ". ")
    t = re.sub(r"\s+", " ", t).strip()

    # Sehr lange Sätze grob teilen
    t = t.replace(" und ", ". Und ")

    return t.strip()


# ---------------------------
# Text‑Rendering
# ---------------------------
class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""

def render_sticky_bar_html(heading: str, subheading: str, chips: List[Tuple[str, str, str]], note: str = "") -> str:
    """Compact, always-visible summary bar (HTML) shown at top of the UI."""

    def level_style(level: str) -> str:
        s = (level or "").lower()

        # Allow passing either a normalized category ("low") or the human label ("Intermediär‑niedrig").
        if any(x in s for x in ("low", "nied")):
            return "background:#ecfdf5;border:1px solid #10b981;color:#065f46;"
        if any(x in s for x in ("high", "hoch")):
            return "background:#fef2f2;border:1px solid #ef4444;color:#991b1b;"
        if any(x in s for x in ("inter", "mittel")):
            return "background:#fffbeb;border:1px solid #f59e0b;color:#92400e;"

        return "background:#f3f4f6;border:1px solid #d1d5db;color:#374151;"

    chip_html = "".join(
        [
            f"<span class='rhk-chip' style='{level_style(level)}'>{label}: <b>{value}</b></span>"
            for (label, value, level) in (chips or [])
            if (label and value)
        ]
    )

    note_html = f"<div class='rhk-sticky-note'>{note}</div>" if note else ""

    return (
        "<div class='rhk-sticky-inner'>"
        f"<div class='rhk-sticky-head'><div class='rhk-sticky-h'>{heading}</div><div class='rhk-sticky-sub'>{subheading}</div></div>"
        f"<div class='rhk-sticky-chips'>{chip_html}</div>"
        f"{note_html}"
        "</div>"
    )




def get_block(block_id: str):
    try:
        return textdb.get_block(block_id)  # type: ignore
    except Exception:
        # fallback: direkte Dicts, falls get_block fehlt
        return getattr(textdb, "ALL_BLOCKS", {}).get(block_id)


def safe_render(block_id: str, ctx: Dict[str, Any]) -> str:
    b = get_block(block_id)
    if not b:
        return ""
    template = getattr(b, "template", "")
    try:
        return template.format_map(SafeDict(ctx)).strip()
    except Exception:
        return template.strip()


def safe_render_patient(block_id: str, ctx: Dict[str, Any]) -> str:
    """Rendert einen Textbaustein in Patientensprache (robust).

    - nutzt, wenn vorhanden, die Patient:innen-Datenbank
    - fällt sonst auf eine sehr einfache Zusammenfassung zurück
    - räumt Abkürzungen/Zahlen zusätzlich auf
    """
    text = ""
    if patientdb is not None and hasattr(patientdb, "get_patient_block"):
        pb = patientdb.get_patient_block(block_id)
        if pb is not None:
            try:
                text = pb.render(ctx)
            except Exception:
                text = pb.template
    if not text:
        # Fallback: (kurz) vereinfachen und dann bereinigen
        doc = TEXTDB.get(block_id, None)
        if doc is not None:
            text = simplify_text_for_patient_fallback(doc.get("text", ""))
        else:
            text = "Wir besprechen das weitere Vorgehen persönlich."
    return patient_clean_text(text).strip()

# ---------------------------
# PH‑Klassifikation / Haupt‑Bundle Auswahl
# ---------------------------

# ------------------------
# Patienten-Text: Aufräumen (keine Zahlen / weniger Abkürzungen)
# ------------------------
_PATIENT_REPLACEMENTS = {
    "RHK": "Rechtsherzkatheter",
    "PH": "Lungenhochdruck",
    "HFpEF": "Störung der Herzentspannung",
    "CT": "Computertomographie",
    "MRT": "Magnetresonanztomographie",
    "CMR": "Herz-Magnetresonanztomographie",
    "Echo": "Herzultraschall",
    "V/Q": "Durchblutungsuntersuchung der Lunge",
    "VQ": "Durchblutungsuntersuchung der Lunge",
    "CTEPH": "Folgen von Blutgerinnseln in der Lunge",
    "CTEPD": "Folgen von Blutgerinnseln in der Lunge",
    "PAH": "eine spezielle Form von Lungenhochdruck",
    "BNP": "Herzbelastungswert im Blut",
    "NT-proBNP": "Herzbelastungswert im Blut",
    "pro-NT-BNP": "Herzbelastungswert im Blut",
    "COPD": "chronische Lungenerkrankung",
    "ILD": "Erkrankung des Lungengewebes",
    "OSA": "schlafbezogene Atmungsstörung",
    "OHS": "starkes Übergewicht mit flacher Atmung",
    "LTOT": "Langzeit-Sauerstofftherapie",
    "CPAP": "Atemtherapie mit Maske",
    "NIV": "Atemunterstützung mit Maske",
    "DOAK": "Blutverdünnung als Tablette",
    "VKA": "Blutverdünnung mit Marcumar oder ähnlich",
    "i.v.": "über die Vene",
    "p.o.": "als Tablette",
    "ggf.": "wenn nötig",
    "z.B.": "zum Beispiel",
    "bzw.": "oder",
    "NYHA/WHO-FC": "Belastungsklasse",
    "NYHA/WHO‑FC": "Belastungsklasse",
    "NYHA": "Belastungsklasse",
    "WHO-FC": "Belastungsklasse",
    "WHO‑FC": "Belastungsklasse",
}

def patient_clean_text(text: str) -> str:
    """Macht Patiententexte einfacher: ersetzt häufige Abkürzungen und entfernt Zahlen."""
    if text is None:
        return ""
    t = str(text)
    # Standard-Replacements
    for src, dst in _PATIENT_REPLACEMENTS.items():
        t = t.replace(src, dst)
    # häufige Kurzformen ohne Punkt-Variante
    t = t.replace("ggf", "wenn nötig")
    t = t.replace("z. B.", "zum Beispiel")
    # Zahlen entfernen (Messwerte etc.)
    t = re.sub(r"\d", "", t)
    t = t.replace("%", "").replace("‰", "")
    # Einheiten/Abkürzungen, falls noch vorhanden
    t = re.sub(r"\bmmHg\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bWU\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bL\/min\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bL\/min\/m\s*\^?\s*\d+\b", "", t, flags=re.IGNORECASE)
    # Leerzeichen/Punktuation aufräumen
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    t = re.sub(r"\(\s*\)", "", t)
    return t.strip()
def classify_ph(mpap: Optional[float], pawp: Optional[float], pvr: Optional[float]) -> str:
    """
    Leitlinien-basierte PH-Klassifikation (ESC/ERS 2022):
    - PH in Ruhe: mPAP > 20 mmHg
    - Präkapillär: PAWP ≤ 15 mmHg UND PVR > 2 WU
    - Isoliert postkapillär (IpcPH): PAWP > 15 mmHg UND PVR ≤ 2 WU
    - Kombiniert post-/präkapillär (CpcPH): PAWP > 15 mmHg UND PVR > 2 WU

    Rückgabe-Keys:
    - keine_ph | precap | precap_pvr_niedrig | ipcph | cpcph | postcap | ph_unbekannt | unbekannt
    """
    if mpap is None:
        return "unbekannt"

    mpap_thr = _rule_float("rest.mPAP_ph_mmHg", 20.0) or 20.0
    pawp_thr = _rule_float("rest.PAWP_postcap_mmHg", 15.0) or 15.0
    pvr_thr = _rule_float("rest.PVR_precap_WU", 2.0) or 2.0

    try:
        mp = float(mpap)
    except Exception:
        return "unbekannt"

    if mp <= mpap_thr:
        return "keine_ph"

    # mPAP spricht für PH → PAWP erforderlich
    if pawp is None:
        return "ph_unbekannt"
    try:
        pw = float(pawp)
    except Exception:
        return "ph_unbekannt"

    # PAWP nicht erhöht → eher präkapillär
    if pw <= pawp_thr:
        if pvr is None:
            return "precap"
        try:
            pv = float(pvr)
        except Exception:
            return "precap"
        if pv > pvr_thr:
            return "precap"
        return "precap_pvr_niedrig"

    # PAWP erhöht → postkapillär; PVR entscheidet IpcPH vs CpcPH
    if pvr is None:
        return "postcap"
    try:
        pv = float(pvr)
    except Exception:
        return "postcap"
    if pv > pvr_thr:
        return "cpcph"
    return "ipcph"


    if mpap is None:
        return "unbekannt"
    if mpap <= mpap_thr:
        return "keine_ph"

    pawp = pawp if pawp is not None else None
    pvr = pvr if pvr is not None else None

    if pawp is None:
        return "ph_unbekannt"
    if pawp <= pawp_thr:
        # präkapillär, sofern PVR erfüllt
        if pvr is None:
            return "precap"
        return "precap" if pvr > pvr_thr else "precap_pvr_niedrig"
    else:
        # postkapillär
        if pvr is None:
            return "postcap"
        return "cpcph" if pvr > pvr_thr else "ipcph"


def pvr_severity_label(pvr: Optional[float]) -> Optional[str]:
    """Grobe PVR-Einordnung (Schwellen über textdb.DEFAULT_RULES konfigurierbar)."""
    if pvr is None:
        return None

    mild_ge = _rule_float("severity.PVR_WU.mild_ge", 2.0)
    mod_ge = _rule_float("severity.PVR_WU.moderate_ge", 5.0)
    sev_ge = _rule_float("severity.PVR_WU.severe_ge", 10.0)

    try:
        p = float(pvr)
    except Exception:
        return None

    if p < mild_ge:
        return "nicht erhöht"
    if p < mod_ge:
        return "mild erhöht"
    if p < sev_ge:
        return "moderat erhöht"
    return "deutlich erhöht"



def classify_exercise_pattern(mpap_co_slope: Optional[float], pawp_co_slope: Optional[float]) -> Optional[str]:
    """Heuristik: Einordnung der Belastungs-Slopes (falls vorhanden)."""
    if mpap_co_slope is None or pawp_co_slope is None:
        return None

    mpap_thr = _rule_float("exercise.mPAP_CO_slope_mmHg_per_L_min", 3.0) or 3.0
    pawp_thr = _rule_float("exercise.PAWP_CO_slope_mmHg_per_L_min", 2.0) or 2.0

    try:
        m = float(mpap_co_slope)
        p = float(pawp_co_slope)
    except Exception:
        return None

    mpap_path = m > mpap_thr
    pawp_path = p > pawp_thr

    if (not mpap_path) and (not pawp_path):
        return "normal"
    if mpap_path and pawp_path:
        return "linkskardial"
    if mpap_path and (not pawp_path):
        return "pulmvasc"
    return "isoliert_pawp"


def pick_main_bundle(
    *,
    ph_type: str,
    pvr: Optional[float],
    ci: Optional[float],
    cteph_suspected: bool,
    has_stepup: bool,
    vaso_done: bool,
    vaso_responder: bool,
    volume_done: bool,
    volume_positive: bool,
    exercise_done: bool,
    exercise_ph: bool,
    exercise_pattern: Optional[str] = None,
) -> str:
    """
    Wählt ein Kxx-Paket (rhk_textdb.K_BLOCKS) passend zur Leitlinien-Logik.

    Priorität:
    1) Shunt (Stufenoxymetrie) → K16
    2) Vasoreaktivitätstest → K17/K18
    3) PH-Typ in Ruhe (ESC/ERS) → K01/K05/K06/K07/K14/K15/K04
    4) Belastung/Volumen (nur als Zusatz, außer: keine PH in Ruhe + Provokation auffällig → K04)
    """

    # Overrides
    if has_stepup:
        return "K16"
    if vaso_done:
        return "K17" if vaso_responder else "K18"

    # Kein PH in Ruhe
    if ph_type == "keine_ph":
        if volume_done and volume_positive:
            return "K04"
        if exercise_done:
            pat = exercise_pattern
            if pat is None and exercise_ph:
                # Wenn nur manuell gesetzt, nehmen wir konservativ "linkskardial" als häufigste Konstellation.
                pat = "linkskardial"
            if pat in ("linkskardial", "isoliert_pawp"):
                return "K02"
            if pat == "pulmvasc":
                return "K03"
        return "K01"

    # Postkapillär / CpcPH
    if ph_type in ("ipcph", "postcap"):
        return "K14"
    if ph_type == "cpcph":
        return "K15"

    # Unklar / Borderline
    if ph_type in ("ph_unbekannt", "unbekannt", "precap_pvr_niedrig"):
        return "K04"

    # Präkapillär
    if ph_type.startswith("precap"):
        if cteph_suspected:
            return "K11"

        pvr_lbl = pvr_severity_label(pvr) or ""
        ci_thr = _rule_float("severity.CI_L_min_m2.severely_reduced_lt", 2.0) or 2.0

        ci_val: Optional[float] = None
        if ci is not None:
            try:
                ci_val = float(ci)
            except Exception:
                ci_val = None

        if (ci_val is not None and ci_val < ci_thr) or (pvr_lbl == "deutlich erhöht"):
            return "K07"
        if pvr_lbl == "moderat erhöht":
            return "K06"
        return "K05"

    return "K04"



# ---------------------------
# Befund‑Logik / DD‑Hinweise
# ---------------------------
def infer_ph_group_hints(ph_type: str, clinical: Dict[str, Any], add: Dict[str, Any], hfpef: Optional[H2FPEFResult]) -> List[str]:
    """
    Gibt kurze, klinisch sinnvolle Hinweise als Liste (ärztliche Sprache) zurück.
    """
    hints: List[str] = []

    # Marker
    ild = bool(clinical.get("ct_ild"))
    emph = bool(clinical.get("ct_emphysema"))
    mosaic = bool(clinical.get("ct_mosaic"))
    embolie = bool(clinical.get("ct_embolie") or clinical.get("ct_lae"))
    ltot = bool(clinical.get("ltot_present"))
    lufu_obst = bool(add.get("lufu_obstr"))
    lufu_rest = bool(add.get("lufu_restr"))
    lufu_diff = bool(add.get("lufu_diff"))
    virol = bool(clinical.get("virology_positive"))
    immun = bool(clinical.get("immunology_positive"))
    portal = bool(clinical.get("portal_hypertension"))
    la_enlarged = bool(clinical.get("la_enlarged"))

    group3 = ild or emph or ltot or lufu_obst or lufu_rest or lufu_diff
    group4 = embolie or mosaic
    group2 = ph_type in ("ipcph", "postcap", "cpcph") or la_enlarged or (hfpef is not None and hfpef.score >= 6)
    group1_ctd = immun  # sehr grob
    group1_porto = portal

    if ph_type.startswith("precap"):
        if group3 and group4:
            hints.append("Kontext: Hinweise auf Lungenerkrankung (Gruppe III) und thromboembolische DD (Gruppe IV).")
        elif group3:
            hints.append("Kontext: Hinweise auf Lungenerkrankung (Gruppe III) als mögliche Ursache/Komponente.")
        elif group4:
            hints.append("Kontext: Hinweis auf mögliche thromboembolische Genese (Gruppe IV, CTEPH‑Abklärung).")
        if group1_ctd:
            hints.append("Kontext: Positive Immunologie – CTD‑assoziierte PAH (Gruppe I) mitdenken.")
        if group1_porto:
            hints.append("Kontext: Portale Hypertension – portopulmonale PH (Gruppe I) mitdenken.")
        if virol:
            hints.append("Kontext: Positive Virologie – ggf. differenzialdiagnostisch relevant (Gruppe V).")

    if group2:
        hints.append("Kontext: Hinweise auf Linksherzbeteiligung (Gruppe II) möglich.")

    return hints


def infer_recommendation_addons(ph_type: str, clinical: Dict[str, Any], add: Dict[str, Any], hfpef: Optional[H2FPEFResult]) -> List[str]:
    """
    Zusätzliche Empfehlungssätze (ärztlich), abhängig von Logik.
    """
    recs: List[str] = []

    ild = bool(clinical.get("ct_ild"))
    emph = bool(clinical.get("ct_emphysema"))
    mosaic = bool(clinical.get("ct_mosaic"))
    embolie = bool(clinical.get("ct_embolie") or clinical.get("ct_lae"))
    ltot = bool(clinical.get("ltot_present"))
    la_enlarged = bool(clinical.get("la_enlarged"))

    group3 = ild or emph or ltot or bool(add.get("lufu_obstr")) or bool(add.get("lufu_restr")) or bool(add.get("lufu_diff"))
    group4 = embolie or mosaic

    if ph_type.startswith("precap"):
        if group4:
            recs.append("Bei DD CTEPH: V/Q‑Szintigrafie und Vorstellung im CTEPH‑Board erwägen.")
        if group3:
            recs.append("Bei relevanter Lungenerkrankung: Optimierung der pulmonologischen Therapie inkl. O₂‑Bedarf prüfen.")
    if ph_type in ("ipcph", "postcap", "cpcph") or la_enlarged:
        recs.append("Bei Verdacht auf diastolische Dysfunktion: kardiologische Mitbeurteilung (HFpEF/Diastolik) erwägen.")
    if hfpef is not None and hfpef.score >= 2:
        recs.append(f"H2FPEF‑Score: {hfpef.score}/9 ({hfpef.category}) → Hinweis auf mögliche HFpEF/diastolische Funktionsstörung.")

    return recs


# ---------------------------
# Report‑Generator
# ---------------------------
class RHKReportGenerator:
    def __init__(self) -> None:
        pass

    def generate_all(self, data: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        """
        Returns:
        - main_report (ärztlich)
        - patient_report (einfache Sprache)
        - internal_log (Debug/Transparenz)
        - risk_html (prominent)
        """
        patient = data.get("patient", {})
        rest = data.get("hemodynamics_rest", {})
        ex = data.get("hemodynamics_exercise", {})
        man = data.get("manoeuvres", {})
        clinical = data.get("clinical_context", {})
        add = data.get("additional_measurements", {})
        planned = data.get("planned_actions", {})

        # -------------------------
        # Patient basics
        # -------------------------
        born = parse_date_any(patient.get("birthdate"))
        age = calc_age_years(born)
        height_cm = to_num(patient.get("height_cm"))
        weight_kg = to_num(patient.get("weight_kg"))
        bsa = to_num(patient.get("bsa_m2"))
        if bsa is None:
            bsa = calc_bsa_m2(height_cm, weight_kg)

        bmi = calc_bmi(weight_kg, height_cm)

        # -------------------------
        # Hämodynamik Ruhe
        # -------------------------
        spap = to_num(rest.get("pa_sys"))
        dpap = to_num(rest.get("pa_dia"))
        mpap = to_num(rest.get("mpap"))
        if mpap is None:
            mpap = calc_mpap_from_spap_dpap(spap, dpap)

        pawp = to_num(rest.get("pawp"))
        rap = to_num(rest.get("rap"))

        hr = to_num(rest.get("hr")) or to_num(clinical.get("hr"))
        co = to_num(rest.get("co"))
        ci = to_num(rest.get("ci"))
        if ci is None:
            ci = calc_ci_from_co_bsa(co, bsa)
        if co is None:
            co = calc_co_from_ci_bsa(ci, bsa)

        pvr = to_num(rest.get("pvr"))
        if pvr is None:
            pvr = calc_pvr_wu(mpap, pawp, co)

        pvri = calc_pvri_wu_m2(mpap, pawp, ci)

        tpg = calc_tpg(mpap, pawp)
        dpg = calc_dpg(dpap, pawp)

        svo2 = to_num(rest.get("svo2"))  # pulmonary artery sat
        if svo2 is None:
            svo2 = to_num(rest.get("pa_sat"))

        svi = to_num(rest.get("svi"))
        if svi is None:
            svi = calc_svi_ml_m2(ci, hr)

        ph_type = classify_ph(mpap, pawp, pvr)
        cteph_suspected = bool(
            ("Gruppe 4" in str(clinical.get("ph_group") or ""))
            or clinical.get("ct_mosaic")
            or clinical.get("ct_embolie")
            or clinical.get("ct_lae")
        )


        # -------------------------
        # Step‑up
        # -------------------------
        sats = {
            "svc": to_num(rest.get("svc_sat")),
            "ivc": to_num(rest.get("ivc_sat")),
            "ra": to_num(rest.get("ra_sat")),
            "rv": to_num(rest.get("rv_sat")),
            "pa": to_num(rest.get("pa_sat")),
        }
        stepup_flag, stepup_sentence = detect_step_up(sats)
        has_stepup = bool(stepup_flag)

        # -------------------------
        # Belastung
        # -------------------------
        exercise_done = bool(ex.get("exercise_done"))

        ex_spap = to_num(ex.get("pa_sys"))
        ex_dpap = to_num(ex.get("pa_dia"))
        ex_mpap = to_num(ex.get("mpap"))
        if ex_mpap is None:
            ex_mpap = calc_mpap_from_spap_dpap(ex_spap, ex_dpap)
        ex_pawp = to_num(ex.get("pawp"))

        ex_hr = to_num(ex.get("hr")) or hr
        ex_co = to_num(ex.get("co"))
        ex_ci = to_num(ex.get("ci"))
        if ex_ci is None:
            ex_ci = calc_ci_from_co_bsa(ex_co, bsa)
        if ex_co is None:
            ex_co = calc_co_from_ci_bsa(ex_ci, bsa)

        ex_pvr = to_num(ex.get("pvr"))
        if ex_pvr is None:
            ex_pvr = calc_pvr_wu(ex_mpap, ex_pawp, ex_co)

        mpap_co_slope = to_num(ex.get("mpap_co_slope"))
        pawp_co_slope = to_num(ex.get("pawp_co_slope"))

        exercise_pattern = classify_exercise_pattern(mpap_co_slope, pawp_co_slope) if exercise_done else None

        # Belastungs‑PH Heuristik (nur wenn angegeben oder ableitbar)
        exercise_ph = bool(ex.get("exercise_ph"))
        if exercise_done and not exercise_ph:
            # sehr grobe Heuristik: mPAP > 30 UND mPAP/CO Slope > 3
            if ex_mpap is not None and ex_mpap > 30 and (mpap_co_slope is None or mpap_co_slope > 3):
                exercise_ph = True

        # Homeometrisch vs heterometrisch (nach Vorgabe)
        adaptation_sentence = ""
        if exercise_done and spap is not None and ex_spap is not None and ci is not None and ex_ci is not None:
            delta_spap = ex_spap - spap
            if delta_spap > 30:
                if ex_ci >= ci:
                    adaptation_sentence = "Belastungsreaktion spricht für einen homeometrischen Adaptionstyp (ΔsPAP >30 mmHg bei nicht verschlechtertem CI)."
                else:
                    adaptation_sentence = "Belastungsreaktion spricht eher für einen heterometrischen Adaptionstyp (ΔsPAP >30 mmHg bei schlechterem CI)."

        # -------------------------
        # Volumenchallenge
        # -------------------------
        volume = man.get("volume", {})
        volume_done = bool(volume.get("done"))
        volume_positive = bool(volume.get("positive"))

        # -------------------------
        # Vasoreaktivität
        # -------------------------
        vaso = man.get("vasoreactivity", {})
        vaso_done = bool(vaso.get("done"))
        vaso_responder = bool(vaso.get("responder"))

        # -------------------------
        # HFpEF Score
        # -------------------------
        hf_af = bool(add.get("hfpef_af")) if add.get("hfpef_af") is not None else None
        hf_htn = to_int(add.get("hfpef_htn_meds"))
        hf_ee = to_num(add.get("hfpef_e_eprime"))
        hf_pasp = to_num(add.get("hfpef_pasp"))
        if hf_pasp is None:
            # wenn nicht eingegeben: sPAP aus RHK als Surrogat verwenden
            hf_pasp = spap

        hfpef_res = h2fpef_score(age, bmi, hf_af, hf_htn, hf_pasp, hf_ee)

        # -------------------------
        # Echo: S'/RAAI
        # -------------------------
        sprime = to_num(add.get("echo_sprime"))
        ra_area = to_num(add.get("echo_ra_area"))
        sprime_raai = calc_sprime_raai(sprime, ra_area, bsa)
        sprime_raai_cutoff = float(DEFAULT_RULES.get("Sprime_RAAI_cutoff_m2_s_cm", 0.81))
        sprime_raai_label = ""
        if sprime_raai is not None:
            sprime_raai_label = "erniedrigt" if sprime_raai < sprime_raai_cutoff else "nicht erniedrigt"

        # -------------------------
        # CMR
        # -------------------------
        cmr_rvef = to_num(add.get("cmr_rvef"))
        cmr_rvesvi = to_num(add.get("cmr_rvesvi"))
        cmr_svi = to_num(add.get("cmr_svi"))

        # -------------------------
        # Risiko (prominent)
        # -------------------------
        who_fc = add.get("who_fc") or clinical.get("who_fc")
        sixmwd = to_num(add.get("sixmwd_m") or clinical.get("sixmwd_m"))
        bnp_kind = add.get("bnp_kind") or clinical.get("bnp_kind")
        bnp_value = to_num(add.get("bnp_value") or clinical.get("bnp_value"))
        sbp = to_num(add.get("sbp") or clinical.get("sbp"))
        egfr = to_num(add.get("egfr") or clinical.get("egfr"))
        # Perikarderguss: nehme echo/ct wenn vorhanden
        pe_grade = add.get("pericard_eff_grade")
        pe_bool = add.get("pericard_eff")  # bool

        esc3_ext = esc3_overall_extended(
            who_fc=who_fc,
            sixmwd_m=sixmwd,
            bnp_kind=bnp_kind,
            bnp_value=bnp_value,
            rap=rap,
            ci=ci,
            hr=hr,
            svo2=svo2,
            ra_area_cm2=ra_area,
            pericard_eff_grade=pe_grade,
            pericard_eff_bool=pe_bool,
            cmr_rvef=cmr_rvef,
            cmr_rvesvi=cmr_rvesvi,
            cmr_svi=cmr_svi,
        )

        esc4 = esc4_overall(who_fc=who_fc, sixmwd_m=sixmwd, bnp_kind=bnp_kind, bnp_value=bnp_value)
        reveal = reveal_lite2_score(who_fc=who_fc, sixmwd_m=sixmwd, bnp_kind=bnp_kind, bnp_value=bnp_value, sbp=sbp, hr=hr, egfr=egfr)
        risk_html = render_risk_html(esc3_ext, esc4, reveal, hfpef_res)

                # Plain risk summary for report
        risk_plain = ""
        if esc3_ext:
            risk_plain += f"ESC/ERS 3‑Strata (erweitert): {esc3_ext.label}. "
        if esc4:
            risk_plain += f"ESC/ERS 4‑Strata: {esc4.label}. "
        if reveal:
            risk_plain += f"REVEAL Lite 2: {reveal.value} Punkte ({reveal.label})."
        risk_plain = risk_plain.strip()

        # -------------------------
        # Haupt‑Bundle / Textblöcke
        # -------------------------
        main_bundle_id = pick_main_bundle(
            ph_type=ph_type,
            pvr=pvr,
            ci=ci,
            cteph_suspected=cteph_suspected,
            has_stepup=stepup_flag,
            vaso_done=vaso_done,
            vaso_responder=vaso_responder,
            volume_done=volume_done,
            volume_positive=volume_positive,
            exercise_done=exercise_done,
            exercise_ph=exercise_ph,
            exercise_pattern=exercise_pattern,
        )

        main_B_id = f"{main_bundle_id}_B"
        main_E_id = f"{main_bundle_id}_E"

        # -------------------------
        # Kontext für Textbausteine
        # -------------------------
        mpap_value = fmt_num(mpap, 0)
        pawp_value = fmt_num(pawp, 0)
        pvr_value = fmt_num(pvr, 1)
        ci_value = fmt_num(ci, 2)
        tpg_value = fmt_num(tpg, 0)
        dpg_value = fmt_num(dpg, 0)

        mpap_phrase = f"mPAP {mpap_value} mmHg" if mpap_value else ""
        pawp_phrase = f"PAWP {pawp_value} mmHg" if pawp_value else ""
        pvr_phrase = f"PVR {pvr_value} WU" if pvr_value else ""
        ci_phrase = f"CI {ci_value} l/min/m²" if ci_value else ""
        tpg_phrase = f"TPG {tpg_value} mmHg" if tpg_value else ""
        dpg_phrase = f"DPG {dpg_value} mmHg" if dpg_value else ""

        pvr_label = pvr_severity_label(pvr) or ""
        # Dashboard (Kurzfazit) für die UI: Diagnose + Kernwerte auf einen Blick
        dash_lines: List[str] = []
        if ph_type == "keine_ph":
            dash_lines.append("Kein Hinweis auf pulmonale Hypertonie in Ruhe.")
        elif ph_type == "ipcph":
            dash_lines.append("Postkapilläre pulmonale Hypertonie: isoliert (IpcPH).")
        elif ph_type == "cpcph":
            dash_lines.append("Kombinierte post- und präkapilläre pulmonale Hypertonie (CpcPH).")
        elif ph_type in ("ipcph", "postcap"):
            dash_lines.append("Postkapilläre pulmonale Hypertonie (PVR nicht angegeben).")
        elif ph_type == "ph_unbekannt":
            dash_lines.append("PH in Ruhe, aber PAWP/PVR unvollständig – Einordnung derzeit nicht sicher.")
        elif ph_type == "precap_pvr_niedrig":
            dash_lines.append("PH-Konstellation: mPAP erhöht bei nicht erhöhtem PAWP, PVR jedoch nicht sicher über Schwelle.")
        elif ph_type.startswith("precap"):
            dash_lines.append("Präkapilläre pulmonale Hypertonie.")
        else:
            dash_lines.append("Einordnung der pulmonalen Hypertonie: unklar.")

        key_bits = [b for b in [mpap_phrase, pawp_phrase, pvr_phrase, ci_phrase, tpg_phrase, dpg_phrase] if b]
        if key_bits:
            dash_lines.append("Werte: " + ", ".join(key_bits) + ".")
        if pvr_label:
            dash_lines.append(f"Schweregrad (PVR): {pvr_label}.")
        if exercise_done:
            dash_lines.append("Belastungsmessung: vorhanden.")
        if has_stepup:
            dash_lines.append("Step-up/Shunt-Hinweis: ja.")

        dash_html = (
            "<div style='margin:0 0 12px 0; padding:12px; border:1px solid rgba(0,0,0,0.12); "
            "border-radius:14px;'>"
            "<div style='font-weight:700; margin-bottom:6px;'>Kurzfazit</div>"
            "<ul style='margin:0; padding-left:18px;'>"
            + "".join([f"<li>{x}</li>" for x in dash_lines])
            + "</ul></div>"
        )
        risk_html = dash_html + risk_html


        # Vergleich RHK
        prev = add.get("prev_rhk", {}) or {}
        comparison_sentence = ""
        prev_label = (prev.get("label") or "").strip()
        prev_course = (prev.get("course") or "").strip()
        prev_mpap = to_num(prev.get("mpap"))
        prev_pawp = to_num(prev.get("pawp"))
        prev_ci = to_num(prev.get("ci"))
        prev_pvr = to_num(prev.get("pvr"))
        if prev_label and any(v is not None for v in [prev_mpap, prev_pawp, prev_ci, prev_pvr]):
            # gewünschtes Format
            comparison_sentence = (
                f"Im Vergleich zu RHK {prev_label} {prev_course} "
                f"(mPAP{fmt_num(prev_mpap,0)} mmHg, "
                f"PAWP {fmt_num(prev_pawp,0)} mmHg, "
                f"CI {fmt_num(prev_ci,2)} l/min/m2, "
                f"PVR {fmt_num(prev_pvr,1)} WU)."
            ).strip()

        # Slopes
        mpap_co_slope_str = fmt_num(mpap_co_slope, 1)
        pawp_co_slope_str = fmt_num(pawp_co_slope, 1)

        # --- Zusatz‑Phrasen für Textbausteine (vermeidet leere Fragmente)
        co_method_desc = "CO‑Messung"
        if rest.get("co") not in (None, "") or rest.get("ci") not in (None, ""):
            co_method_desc = "CO‑Messung"

        cv_stauung_phrase = ""
        if rap is not None:
            if rap < 8:
                cv_stauung_phrase = "Keine wesentliche systemvenöse Stauung."
            elif rap <= 14:
                cv_stauung_phrase = "Geringe systemvenöse Stauung möglich."
            else:
                cv_stauung_phrase = "Ausgeprägte systemvenöse Stauung."

        pv_stauung_phrase = ""
        pawp_thr = float(DEFAULT_RULES.get("PAWP_postcap_mmHg", 15.0))
        if pawp is not None:
            if pawp <= pawp_thr:
                pv_stauung_phrase = "Keine wesentliche pulmonalvenöse Stauung."
            else:
                pv_stauung_phrase = "Pulmonalvenöse Stauung."

        systemic_sentence = ""
        if sbp is not None:
            if sbp < 95:
                systemic_sentence = "Systemisch hypoton."
            elif sbp < 110:
                systemic_sentence = "Systemisch grenzwertig."
            else:
                systemic_sentence = "Systemisch normotensiv."

        oxygen_sentence = ""
        po2 = to_num(clinical.get("bga_rest_pO2"))
        if clinical.get("ltot_present") is True:
            oxygen_sentence = "Unter LTOT."
        elif po2 is not None:
            if po2 < 60:
                oxygen_sentence = "Hypoxämie."
            elif po2 >= 80:
                oxygen_sentence = "Oxygenierung unauffällig."

        exam_type_desc = "RHK in Ruhe"
        if exercise_done or volume_done or vaso_done:
            exam_type_desc = "RHK inkl. Provokationsmanöver"

        ctx: Dict[str, Any] = {
            # Patient
            "name": (patient.get("last_name") or "").strip(),
            "firstname": (patient.get("first_name") or "").strip(),
            "birthdate": born.strftime("%d.%m.%Y") if born else "",
            "age_years": str(age) if age is not None else "",
            "story": (patient.get("story") or "").strip(),
            # Hämodynamik
            "mpap_value": mpap_value,
            "pawp_value": pawp_value,
            "pvr_value": pvr_value,
            "ci_value": ci_value,
            "tpg_value": tpg_value,
            "dpg_value": dpg_value,
            "mpap_phrase": mpap_phrase,
            "pawp_phrase": pawp_phrase,
            "pvr_phrase": pvr_phrase,
            "ci_phrase": ci_phrase,
            "tpg_phrase": tpg_phrase,
            "dpg_phrase": dpg_phrase,
            "pvr_label": pvr_label,
            "co_method_desc": co_method_desc,
            "cv_stauung_phrase": cv_stauung_phrase,
            "pv_stauung_phrase": pv_stauung_phrase,
            "systemic_sentence": systemic_sentence,
            "oxygen_sentence": oxygen_sentence,
            "exam_type_desc": exam_type_desc,
            # Belastung
            "mPAP_CO_slope": mpap_co_slope_str,
            "PAWP_CO_slope": pawp_co_slope_str,
            "exercise_end_mpap": fmt_num(ex_mpap, 0),
            "exercise_end_pawp": fmt_num(ex_pawp, 0),
            "exercise_end_pvr": fmt_num(ex_pvr, 1),
            "exercise_end_ci": fmt_num(ex_ci, 2),
            # Step-up / Shunt
            "step_up_sentence": stepup_sentence,
            # Vergleich
            "comparison_sentence": comparison_sentence,
            # Planung (optional)
            "therapy_plan_sentence": (add.get("therapy_plan_sentence") or "").strip(),
            "anticoagulation_plan_sentence": (add.get("anticoag_plan_sentence") or "").strip(),
            "study_sentence": (add.get("study_sentence") or "").strip(),
            "declined_item": (add.get("declined_item") or "").strip(),
            "followup_timing_desc": (add.get("followup_timing_desc") or "3–6 Monaten").strip(),
        }

        # -------------------------
        # Text: Beurteilung
        # -------------------------
        beurteilung = safe_render(main_B_id, ctx)
        # Standard-Hämodynamikzeile immer ergänzen (TPG + Slopes wenn Belastung)
        hemo_parts = [p for p in [mpap_phrase, pawp_phrase, tpg_phrase, dpg_phrase, pvr_phrase, ci_phrase] if p]
        hemo_line = ""
        if hemo_parts:
            hemo_line = "Kennwerte (Ruhe): " + ", ".join(hemo_parts) + "."

        slope_line = ""
        if exercise_done and (mpap_co_slope_str or pawp_co_slope_str):
            bits = []
            if mpap_co_slope_str:
                bits.append(f"mPAP/CO‑Slope {mpap_co_slope_str} mmHg/(l/min)")
            if pawp_co_slope_str:
                bits.append(f"PAWP/CO‑Slope {pawp_co_slope_str} mmHg/(l/min)")
            slope_line = "Belastung: " + ", ".join(bits) + "."

        extra_lines = []
        if hemo_line and ("Kennwerte" not in beurteilung):
            extra_lines.append(hemo_line)
        if slope_line and ("Slope" not in beurteilung):
            extra_lines.append(slope_line)
        if adaptation_sentence:
            extra_lines.append(adaptation_sentence)
        if comparison_sentence and (comparison_sentence not in beurteilung):
            extra_lines.append(comparison_sentence)

        if extra_lines:
            beurteilung = (beurteilung + "\n\n" + "\n".join(extra_lines)).strip()

        # -------------------------
        # Text: Empfehlung
        # -------------------------
        empfehlung_main = safe_render(main_E_id, ctx).strip()
        # Risiko direkt nach Diagnose / Hauptempfehlung
        empfehlung_add: List[str] = []
        if risk_plain:
            empfehlung_add.append(f"Risikostratifizierung: {risk_plain}")
        # HFpEF‑Hinweis, wenn Score anschlägt
        if hfpef_res and hfpef_res.score >= 2:
            empfehlung_add.append(f"Hinweis HFpEF: H2FPEF {hfpef_res.score}/9 ({hfpef_res.category}).")

        # Logik‑Addons
        empfehlung_add += infer_recommendation_addons(ph_type, clinical, add, hfpef_res)

        empfehlung = empfehlung_main
        if empfehlung_add:
            empfehlung = (empfehlung + "\n\n" + "\n".join([f"- {s}" for s in empfehlung_add if s.strip()])).strip()

        # -------------------------
        # Procedere/Module
        # -------------------------
        chosen_ids: List[str] = planned.get("modules", []) or []
        # automatisch vorgeschlagene P‑Module aus Bundle
        # (rhk_textdb.BUNDLES ist typischerweise ein dict wie {"K05": {"P_suggestions": ["P01", ...]}})
        auto_ids: List[str] = []
        try:
            bundles = getattr(textdb, "BUNDLES", {}) or {}
            bundle = bundles.get(main_bundle_id)
            if isinstance(bundle, dict):
                auto_ids = list(bundle.get("P_suggestions") or [])
            else:
                auto_ids = list(getattr(bundle, "P_suggestions", []) or getattr(bundle, "p_modules", []) or [])
        except Exception:
            auto_ids = []

        # kombinieren (unique, Reihenfolge: auto → chosen)
        all_ids = []
        for bid in auto_ids + chosen_ids:
            if bid and bid not in all_ids:
                all_ids.append(bid)

        proc_lines: List[str] = []
        for bid in all_ids:
            txt = safe_render(bid, ctx)
            txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
            if not txt:
                continue
            # bullets sauber
            if "\n" in txt:
                proc_lines.append(f"- {txt.replace(chr(10), chr(10)+'  ')}")
            else:
                proc_lines.append(f"- {txt}")

        proc_text = "\n".join(proc_lines).strip()

        # -------------------------
        # Zusatzdaten (für Hauptbefund)
        # -------------------------
        name_line = join_nonempty([ctx.get("firstname", ""), ctx.get("name", "")]).strip()
        header = "RECHTSHERZKATHETER-BEFUND"
        if name_line:
            header += f" – {name_line}"

        # Klinik/Labor Zusammenfassung (aus Eingaben)
        clinic_lines: List[str] = []
        if (patient.get("story") or "").strip():
            clinic_lines.append(f"Kurz-Anamnese: {patient.get('story').strip()}")
        if clinical.get("ph_known") is True:
            clinic_lines.append("PH-Diagnose bekannt.")
        elif clinical.get("ph_suspected") is True:
            clinic_lines.append("PH-Verdachtsdiagnose.")

        # Labor
        lab_parts = []
        for k, label in [
            ("inr", "INR"),
            ("quick", "Quick"),
            ("crea", "Krea"),
            ("hst", "Hst"),
            ("ptt", "PTT"),
            ("plt", "Thrombos"),
            ("hb", "Hb"),
            ("crp", "CRP"),
            ("leuko", "Leukos"),
        ]:
            v = clinical.get(k)
            if v not in (None, ""):
                lab_parts.append(f"{label} {v}")
        if bnp_value is not None:
            lab_parts.append(f"{bnp_kind or 'BNP'} {fmt_num(bnp_value)}")
        if lab_parts:
            clinic_lines.append("Labor: " + ", ".join(lab_parts))

        if clinical.get("congestive_organopathy") is True:
            clinic_lines.append("Hinweis auf congestive Organopathie: ja.")
        elif clinical.get("congestive_organopathy") is False:
            clinic_lines.append("Hinweis auf congestive Organopathie: nein.")

        # BGA/LTOT (nur wenn eingetragen)
        if clinical.get("ltot_present") is True:
            clinic_lines.append("LTOT vorhanden.")
        if clinical.get("bga_rest_pO2") or clinical.get("bga_rest_pCO2"):
            clinic_lines.append(
                "BGA Ruhe: " + ", ".join([p for p in [
                    f"pO₂ {clinical.get('bga_rest_pO2')}" if clinical.get("bga_rest_pO2") else "",
                    f"pCO₂ {clinical.get('bga_rest_pCO2')}" if clinical.get("bga_rest_pCO2") else "",
                ] if p])
            )

        # Lungenfunktion (wenn vorhanden)
        if add.get("lufu_done") is True:
            lufu_bits = []
            phenos = []
            if add.get("lufu_obstr"):
                phenos.append("obstruktiv")
            if add.get("lufu_restr"):
                phenos.append("restriktiv")
            if add.get("lufu_diff"):
                phenos.append("Diffusionsstörung")
            if phenos:
                lufu_bits.append("Phänotyp: " + ", ".join(phenos))
            for k, lab in [
                ("lufu_fev1", "FEV₁"),
                ("lufu_fvc", "FVC"),
                ("lufu_fev1_fvc", "FEV₁/FVC"),
                ("lufu_tlc", "TLC"),
                ("lufu_rv", "RV"),
                ("lufu_dlco", "DLCO SB"),
            ]:
                v = add.get(k)
                if v not in (None, ""):
                    num = to_num(v)
                    if num is not None:
                        lufu_bits.append(f"{lab} {fmt_num(num,2)}")
                    else:
                        lufu_bits.append(f"{lab} {v}")
            if lufu_bits:
                clinic_lines.append("Lufu: " + ", ".join(lufu_bits))
            if (add.get("lufu_summary") or "").strip():
                clinic_lines.append("Lufu Summary: " + add.get("lufu_summary").strip())

        # Bildgebung (CT)
        img_flags = []
        for k, label in [
            ("ct_lae", "Lungenarterienembolie (LAE)"),
            ("ct_ild", "ILD"),
            ("ct_emphysema", "Emphysem"),
            ("ct_embolie", "Embolie"),
            ("ct_mosaic", "Mosaikperfusion"),
            ("ct_coronarycalc", "Koronarkalk"),
        ]:
            if clinical.get(k):
                img_flags.append(label)
        if img_flags:
            clinic_lines.append("CT/Bildgebung: " + ", ".join(img_flags))

        # Echo S'/RAAI
        if sprime_raai is not None:
            clinic_lines.append(
                f"Echo: S'/RAAI {fmt_num(sprime_raai,2)} (Cutoff {fmt_num(sprime_raai_cutoff,2)}), {sprime_raai_label}."
            )

        # CMR
        if any(v is not None for v in [cmr_rvef, cmr_rvesvi, cmr_svi]):
            cmr_parts = []
            if cmr_rvesvi is not None:
                cmr_parts.append(f"RVESVi {fmt_num(cmr_rvesvi,0)} ml/m²")
            if cmr_svi is not None:
                cmr_parts.append(f"SVi {fmt_num(cmr_svi,0)} ml/m²")
            if cmr_rvef is not None:
                cmr_parts.append(f"RVEF {fmt_num(cmr_rvef,0)} %")
            clinic_lines.append("CMR: " + ", ".join(cmr_parts))

        # Gruppen‑Hinweise
        group_hints = infer_ph_group_hints(ph_type, clinical, add, hfpef_res)
        if group_hints:
            clinic_lines.append("DD/Einordnung: " + " ".join(group_hints))

        clinic_block = "\n".join([f"- {l}" for l in clinic_lines if l.strip()]).strip()

        # -------------------------
        # Hauptbefund zusammenbauen
        # -------------------------
        main_parts: List[str] = []
        main_parts.append(header)
        main_parts.append("")
        main_parts.append("BEURTEILUNG")
        main_parts.append(beurteilung.strip())
        main_parts.append("")
        main_parts.append("EMPFEHLUNG")
        main_parts.append(empfehlung.strip())
        if proc_text:
            main_parts.append("")
            main_parts.append("PROCEDERE / MASSNAHMEN")
            main_parts.append(proc_text)
        if clinic_block:
            main_parts.append("")
            main_parts.append("KLINIK / LABOR / BILDGEBUNG (Auszug)")
            main_parts.append(clinic_block)

        main_report = "\n".join(main_parts).strip()

        # -------------------------
        
        # -------------------------
        # Patienten‑Info (sehr einfache Sprache, ausführlicher)
        # -------------------------
        patient_paras: List[str] = []

        ph_present = (ph_type not in ("keine_ph", "unbekannt"))

        # Optional: Name/Anamnese (nur wenn vorhanden)
        patient_name = join_nonempty([patient.get("first_name"), patient.get("last_name")], " ").strip()
        story = patient_clean_text(patient.get("story") or "").strip()

        if patient_name:
            patient_paras.append(f"Patientenbericht für {patient_name}.")
        if story:
            patient_paras.append(f"Kurz zur Vorgeschichte: {story}")

        # Was wurde gemacht?
        exam_parts = ["in Ruhe"]
        if exercise_done:
            exam_parts.append("unter Belastung")
        if volume_done:
            exam_parts.append("nach Gabe von Flüssigkeit")

        patient_paras.append(
            "Was wurde untersucht: "
            "Bei Ihnen wurde ein Rechtsherzkatheter durchgeführt. Dabei werden mit einem dünnen Schlauch "
            "Druck- und Flusswerte am Herzen und in den Lungengefäßen gemessen. "
            f"Die Messungen wurden {join_nonempty(exam_parts, ', ')} durchgeführt."
        )

        # Ergebnis in einfacher Sprache (ohne Zahlen/Abkürzungen)
        if not ph_present and not exercise_done:
            result_para = (
                "Ergebnis: Die Messwerte sprechen aktuell nicht für einen Lungenhochdruck."
            )
        elif not ph_present and exercise_done:
            if exercise_ph and exercise_pattern == "pulmonary_vascular":
                result_para = (
                    "Ergebnis: In Ruhe waren die Werte unauffällig. "
                    "Unter Belastung zeigte sich aber eine auffällige Reaktion der Lungengefäße. "
                    "Das kann ein frühes Zeichen für eine Belastungs‑Problematik sein."
                )
            elif exercise_ph and exercise_pattern == "left_heart":
                result_para = (
                    "Ergebnis: In Ruhe waren die Werte unauffällig. "
                    "Unter Belastung zeigte sich aber ein Rückstau, der eher vom linken Herzen ausgehen kann. "
                    "Das passt zum Beispiel zu einer Herzschwäche mit erhaltener Pumpfunktion."
                )
            else:
                result_para = (
                    "Ergebnis: In Ruhe waren die Werte unauffällig. "
                    "Unter Belastung gab es Hinweise auf eine grenzwertige oder auffällige Reaktion. "
                    "Das wird zusammen mit anderen Untersuchungen eingeordnet."
                )
        else:
            # PH in Ruhe vorhanden
            if ph_type in ("precap", "precap_pvr_niedrig"):
                result_para = (
                    "Ergebnis: Es liegt ein Lungenhochdruck vor. "
                    "Die Messung spricht eher dafür, dass die Ursache in den Lungengefäßen selbst liegt."
                )
            elif ph_type == "postcap":
                result_para = (
                    "Ergebnis: Es liegt ein Lungenhochdruck vor. "
                    "Die Messung spricht eher für einen Rückstau, der vom linken Herzen ausgehen kann."
                )
            elif ph_type == "cpcph":
                result_para = (
                    "Ergebnis: Es liegt ein Lungenhochdruck vor. "
                    "Dabei sprechen die Werte sowohl für einen Rückstau vom linken Herzen als auch für eine zusätzliche Belastung der Lungengefäße."
                )
            else:
                result_para = (
                    "Ergebnis: Es liegt ein Lungenhochdruck vor. "
                    "Die genaue Einordnung hängt von der Gesamtsituation und weiteren Befunden ab."
                )

        # Schweregrad (patientenverständlich)
        if ph_present and pvr_label:
            sev_text = (pvr_label or "").lower()
            sev_plain = ""
            if "mild" in sev_text:
                sev_plain = "leicht"
            elif "moderat" in sev_text:
                sev_plain = "mittel"
            elif "deutlich" in sev_text:
                sev_plain = "ausgeprägt"
            elif "nicht" in sev_text:
                sev_plain = "nur gering"

            if sev_plain:
                result_para += f" Insgesamt wirkt der Befund {sev_plain} ausgeprägt."

        patient_paras.append(result_para)

        # Einordnung / was bedeutet das?
        meaning_lines: List[str] = []
        meaning_lines.append(
            "Was bedeutet das: Lungenhochdruck kann das rechte Herz belasten. "
            "Je nachdem, wo die Ursache liegt, unterscheiden sich Diagnostik und Behandlung."
        )

        group_hint = ""
        if group_hints:
            joined = " ".join(group_hints)
            if "Group II" in joined:
                group_hint = "Group II"
            elif "Group III" in joined:
                group_hint = "Group III"
            elif "Group IV" in joined:
                group_hint = "Group IV"

        if group_hint:
            if group_hint == "Group II":
                meaning_lines.append(
                    "In Ihrem Befund gibt es Hinweise, dass das linke Herz eine wichtige Rolle spielt. "
                    "Dann stehen oft Blutdruck, Rhythmus, Klappen und das Flüssigkeits‑Management im Vordergrund."
                )
            elif group_hint == "Group III":
                meaning_lines.append(
                    "In Ihrem Befund gibt es Hinweise, dass eine Lungenerkrankung bzw. Sauerstoff‑Probleme mitbeteiligt sein können. "
                    "Dann sind Lungenfunktion, Bildgebung und ggf. eine Sauerstoff‑Therapie besonders wichtig."
                )
            elif group_hint == "Group IV":
                meaning_lines.append(
                    "In Ihrem Befund gibt es Hinweise, dass frühere oder aktuelle Blutgerinnsel in der Lunge eine Rolle spielen könnten. "
                    "Dann sind spezielle Untersuchungen der Lungendurchblutung wichtig."
                )

        if hfpef_res is not None:
            # hfpef_res kann Dict oder Objekt sein – wir greifen defensiv zu
            hf_cat = getattr(hfpef_res, "category", None)
            if not hf_cat and isinstance(hfpef_res, dict):
                hf_cat = hfpef_res.get("category")
            if hf_cat in ("possible", "likely"):
                meaning_lines.append(
                    "Zusatzhinweis: Aus Ihren Angaben ergeben sich Hinweise, die zu einer sogenannten diastolischen Funktionsstörung passen können "
                    "(eine Form der Herzschwäche, bei der das Herz sich schlechter entspannt). "
                    "Das sollte kardiologisch mitbeurteilt werden."
                )

        patient_paras.append(" ".join(meaning_lines))

        # Risiko (sehr einfach, ohne Score‑Namen)
        def _worst_risk_category_from_labels(labels: List[str]) -> str:
            labels = [(l or "").lower() for l in labels if l]
            if any(("hoch" in l) or ("high" in l) for l in labels):
                return "high"
            if any(("inter" in l) or ("mittel" in l) for l in labels):
                return "intermediate"
            if any(("nied" in l) or ("low" in l) for l in labels):
                return "low"
            return ""

        risk_labels: List[str] = []
        for rr in (esc3_ext, esc4, reveal):
            if rr is not None:
                risk_labels.append(getattr(rr, "label", "") or "")
        worst = _worst_risk_category_from_labels(risk_labels)

        if worst:
            risk_plain = {"low": "niedrig", "intermediate": "mittel", "high": "hoch"}.get(worst, "")
            if risk_plain:
                patient_paras.append(
                    "Risikoeinschätzung: Ärztinnen und Ärzte ordnen den Befund oft in ein Risiko ein, "
                    "um die Behandlung zu planen. Nach den vorliegenden Angaben liegt Ihr Risiko aktuell eher im Bereich "
                    f"{risk_plain}. "
                    "Das wird immer zusammen mit Ihrem Befinden und weiteren Untersuchungen beurteilt."
                )

        # Nächste Schritte / Empfehlungen (patientenfreundlich)
        rec_items: List[str] = []
        ctx_patient = {
            "patient": patient,
            "rhk": {"rest": rest, "exercise": ex, "manoeuvres": man, "additional": add},
            "rest": rest,
            "exercise": ex,
            "add": add,
        }
        for pid in all_ids:
            t = safe_render_patient(pid, ctx_patient)
            t = patient_clean_text(t).strip()
            if t:
                rec_items.append(t)

        if rec_items:
            patient_paras.append(
                "Wie geht es weiter: "
                "Je nach Gesamtsituation empfehlen wir als nächste Schritte unter anderem:\n"
                + "\n".join([f"• {x}" for x in rec_items])
            )
        else:
            patient_paras.append(
                "Wie geht es weiter: Das weitere Vorgehen richtet sich nach Ihren Beschwerden und den Begleitbefunden. "
                "Oft sind Kontrollen beim Herz‑ und/oder Lungen‑Facharzt sinnvoll."
            )

        # Was Sie selbst tun können
        patient_paras.append(
            "Was Sie selbst tun können: "
            "Achten Sie auf regelmäßige Bewegung im Rahmen Ihrer Möglichkeiten, "
            "nehmen Sie Medikamente wie verordnet ein und sprechen Sie mit Ihrem Behandlungsteam über Salz‑/Flüssigkeits‑Themen, "
            "wenn Wasseransammlungen oder Luftnot bestehen. "
            "Bei Rauchen: Aufhören hilft dem Herzen und der Lunge."
        )

        # Warnzeichen
        patient_paras.append(
            "Wichtig: Bitte suchen Sie zeitnah ärztliche Hilfe, wenn sich Luftnot rasch verschlechtert, "
            "wenn Sie Ohnmachtsanfälle haben, starke Brustschmerzen auftreten oder wenn Beine/Bauch deutlich anschwellen."
        )

        patient_report = "\n\n".join([p.strip() for p in patient_paras if p.strip()])

        # -------------------------
        # Internal log
        # -------------------------
        internal_lines: List[str] = []
        internal_lines.append("INTERNER LOG (Transparenz)")
        internal_lines.append(f"Bundle: {main_bundle_id} (B={main_B_id}, E={main_E_id})")
        internal_lines.append(f"PH-Typ: {ph_type}")
        internal_lines.append(f"Ruhe: mPAP={mpap}, PAWP={pawp}, RAP={rap}, CO={co}, CI={ci}, PVR={pvr}, TPG={tpg}, DPG={dpg}, SVI={svi}, SvO2={svo2}")
        if exercise_done:
            internal_lines.append(f"Belastung: mPAP={ex_mpap}, PAWP={ex_pawp}, CO={ex_co}, CI={ex_ci}, PVR={ex_pvr}, sPAP={ex_spap}, Slopes: mPAP/CO={mpap_co_slope}, PAWP/CO={pawp_co_slope}, exercise_ph={exercise_ph}")
        internal_lines.append(f"Step-up: {stepup_flag} ({stepup_sentence})")
        if adaptation_sentence:
            internal_lines.append(f"Adaptation: {adaptation_sentence}")
        if hfpef_res:
            internal_lines.append(f"H2FPEF: {hfpef_res.score}/9 ({hfpef_res.category})")
        if sprime_raai is not None:
            internal_lines.append(f"S'/RAAI: {sprime_raai} (cutoff {sprime_raai_cutoff}) -> {sprime_raai_label}")
        if esc3_ext:
            internal_lines.append("ESC/ERS 3‑Strata erweitert: " + "; ".join(esc3_ext.details))
        if esc4:
            internal_lines.append("ESC/ERS 4‑Strata: " + "; ".join(esc4.details))
        if reveal:
            internal_lines.append("REVEAL Lite 2: " + "; ".join(reveal.details))
        if comparison_sentence:
            internal_lines.append(f"Prev RHK: {comparison_sentence}")

        internal_report = "\n".join(internal_lines).strip()


        # Sticky Summary (für die immer sichtbare Kopfzeile)
        ph_present = (ph_type not in ("keine_ph", "unbekannt"))
        setting_desc = "Ruhe"
        if exercise_done:
            setting_desc = "Ruhe + Belastung"
        if volume_done:
            setting_desc = "Ruhe + Flüssigkeit"

        ph_type_plain = {
            "keine_ph": "keine PH",
            "precap": "präkapillär",
            "precap_pvr_niedrig": "präkapillär (Widerstand nicht erhöht)",
            "ipcph": "postkapillär (linkes Herz)",
            "postcap": "postkapillär (linkes Herz)",
            "cpcph": "kombiniert (linkes Herz + Lungengefäße)",
            "ph_unbekannt": "PH – Typ unklar",
            "unbekannt": "unklar",
        }.get(ph_type or "unbekannt", "unklar")

        chips: List[Tuple[str, str, str]] = []
        chips.append(("Setting", setting_desc, "neutral"))

        if ph_present:
            chips.append(("PH‑Typ", ph_type_plain, "neutral"))
            pvr_label_local = (pvr_severity_label(pvr) or "").strip()
            if pvr_label_local:
                chips.append(("Gefäßwiderstand", pvr_label_local, "neutral"))
        else:
            chips.append(("PH‑Typ", ph_type_plain, "neutral"))

        if esc3_ext:
            chips.append(("ESC‑3‑Strata", esc3_ext.label, esc3_ext.label))
        if esc4:
            chips.append(("ESC‑4‑Strata", esc4.label, esc4.label))
        if reveal:
            chips.append(("REVEAL", reveal.label, reveal.label))

        hf_cat = getattr(hfpef_res, "category", None) if hfpef_res is not None else None
        if not hf_cat and isinstance(hfpef_res, dict):
            hf_cat = hfpef_res.get("category")
        if hf_cat in ("possible", "likely"):
            chips.append(("HFpEF‑Hinweis", "möglich" if hf_cat == "possible" else "wahrscheinlich", "intermediate"))

        note = ""

        sticky_summary = " | ".join([x for x in (dash_lines[:2] if isinstance(dash_lines, list) else []) if x])

        sticky_html = render_sticky_bar_html("Befundübersicht", sticky_summary, chips, note=note)

        return main_report, patient_report, internal_report, risk_html, sticky_html


# ---------------------------
# UI glue
# ---------------------------
def build_data_from_ui(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    raw: flaches Dict aus UI-Werten
    """
    # Patient
    patient = {
        "last_name": raw.get("last_name", ""),
        "first_name": raw.get("first_name", ""),
        "birthdate": raw.get("birthdate"),
        "height_cm": raw.get("height_cm"),
        "weight_kg": raw.get("weight_kg"),
        "bsa_m2": raw.get("bsa_m2"),
        "story": raw.get("story", ""),
    }

    # Hämodynamik Ruhe
    rest = {
        "mpap": raw.get("mpap"),
        "pa_sys": raw.get("pa_sys"),
        "pa_dia": raw.get("pa_dia"),
        "pawp": raw.get("pawp"),
        "rap": raw.get("rap"),
        "co": raw.get("co"),
        "ci": raw.get("ci"),
        "pvr": raw.get("pvr"),
        "svi": raw.get("svi"),
        "hr": raw.get("hr"),
        "svo2": raw.get("svo2"),
        # step-up sats
        "svc_sat": raw.get("svc_sat"),
        "ivc_sat": raw.get("ivc_sat"),
        "ra_sat": raw.get("ra_sat"),
        "rv_sat": raw.get("rv_sat"),
        "pa_sat": raw.get("pa_sat"),
    }

    # Belastung
    ex = {
        "exercise_done": raw.get("exercise_done"),
        "exercise_ph": raw.get("exercise_ph"),
        "mpap": raw.get("ex_mpap"),
        "pa_sys": raw.get("ex_pa_sys"),
        "pa_dia": raw.get("ex_pa_dia"),
        "pawp": raw.get("ex_pawp"),
        "co": raw.get("ex_co"),
        "ci": raw.get("ex_ci"),
        "pvr": raw.get("ex_pvr"),
        "hr": raw.get("ex_hr"),
        "mpap_co_slope": raw.get("mpap_co_slope"),
        "pawp_co_slope": raw.get("pawp_co_slope"),
    }

    # Manöver
    volume = {
        "done": raw.get("volume_done"),
        "positive": raw.get("volume_positive"),
        "volume_ml": raw.get("volume_ml"),
        "pre_pawp": raw.get("volume_pre_pawp"),
        "post_pawp": raw.get("volume_post_pawp"),
    }
    vaso = {
        "done": raw.get("vaso_done"),
        "responder": raw.get("vaso_responder"),
        "ino_ppm": raw.get("ino_ppm"),
        "pre_mpap": raw.get("vaso_pre_mpap"),
        "post_mpap": raw.get("vaso_post_mpap"),
        "pre_pvr": raw.get("vaso_pre_pvr"),
        "post_pvr": raw.get("vaso_post_pvr"),
    }
    manoeuvres = {"volume": volume, "vasoreactivity": vaso}

    # Klinik/Labor etc
    clinical = {
        "ph_known": raw.get("ph_known"),
        "ph_suspected": raw.get("ph_suspected"),
        # labs
        "inr": raw.get("inr"),
        "quick": raw.get("quick"),
        "crea": raw.get("crea"),
        "hst": raw.get("hst"),
        "ptt": raw.get("ptt"),
        "plt": raw.get("plt"),
        "hb": raw.get("hb"),
        "crp": raw.get("crp"),
        "leuko": raw.get("leuko"),
        "congestive_organopathy": raw.get("congestive_organopathy"),
        # blood gases / ltot
        "ltot_present": raw.get("ltot_present"),
        "bga_rest_pO2": raw.get("bga_rest_pO2"),
        "bga_rest_pCO2": raw.get("bga_rest_pCO2"),
        # immunology / virology
        "virology_positive": raw.get("virology_positive"),
        "immunology_positive": raw.get("immunology_positive"),
        # abdomen
        "abdo_sono": raw.get("abdo_sono"),
        "portal_hypertension": raw.get("portal_hypertension"),
        # CT / imaging
        "ct_angio": raw.get("ct_angio"),
        "ct_lae": raw.get("ct_lae"),
        "ct_ild": raw.get("ct_ild"),
        "ct_emphysema": raw.get("ct_emphysema"),
        "ct_embolie": raw.get("ct_embolie"),
        "ct_mosaic": raw.get("ct_mosaic"),
        "ct_coronarycalc": raw.get("ct_coronarycalc"),
        "la_enlarged": raw.get("la_enlarged"),
        # meds/comorb
        "comorbidities": raw.get("comorbidities"),
        "comorbidities_relevance": raw.get("comorbidities_relevance"),
        "ph_meds_yesno": raw.get("ph_meds_yesno"),
        "ph_meds_which": raw.get("ph_meds_which"),
        "ph_meds_since": raw.get("ph_meds_since"),
        "ph_meds_past_yesno": raw.get("ph_meds_past_yesno"),
        "other_meds": raw.get("other_meds"),
        "diuretics_yesno": raw.get("diuretics_yesno"),
        # risk inputs
        "bnp_kind": raw.get("bnp_kind"),
        "bnp_value": raw.get("bnp_value"),
        "sbp": raw.get("sbp"),
        "hr": raw.get("hr"),
        "egfr": raw.get("egfr"),
        # functional tests
        "who_fc": raw.get("who_fc"),
        "syncope": raw.get("syncope"),
        "sixmwd_m": raw.get("sixmwd_m"),
        "ve_vco2": raw.get("ve_vco2"),
        "vo2max": raw.get("vo2max"),
    }

    add: Dict[str, Any] = {
        # lufu
        "lufu_done": raw.get("lufu_done"),
        "lufu_obstr": raw.get("lufu_obstr"),
        "lufu_restr": raw.get("lufu_restr"),
        "lufu_diff": raw.get("lufu_diff"),
        "lufu_fev1": raw.get("lufu_fev1"),
        "lufu_fvc": raw.get("lufu_fvc"),
        "lufu_fev1_fvc": raw.get("lufu_fev1_fvc"),
        "lufu_tlc": raw.get("lufu_tlc"),
        "lufu_rv": raw.get("lufu_rv"),
        "lufu_dlco": raw.get("lufu_dlco"),
        "lufu_summary": raw.get("lufu_summary"),
        # echo
        "echo_sprime": raw.get("echo_sprime"),
        "echo_ra_area": raw.get("echo_ra_area"),
        "pericard_eff": raw.get("pericard_eff"),
        "pericard_eff_grade": raw.get("pericard_eff_grade"),
        # CMR
        "cmr_rvesvi": raw.get("cmr_rvesvi"),
        "cmr_svi": raw.get("cmr_svi"),
        "cmr_rvef": raw.get("cmr_rvef"),
        # HFpEF score inputs
        "hfpef_af": raw.get("hfpef_af"),
        "hfpef_htn_meds": raw.get("hfpef_htn_meds"),
        "hfpef_e_eprime": raw.get("hfpef_e_eprime"),
        "hfpef_pasp": raw.get("hfpef_pasp"),
        # planned sentences
        "therapy_plan_sentence": raw.get("therapy_plan_sentence"),
        "anticoag_plan_sentence": raw.get("anticoag_plan_sentence"),
        "study_sentence": raw.get("study_sentence"),
        "declined_item": raw.get("declined_item"),
        "followup_timing_desc": raw.get("followup_timing_desc"),
        # previous RHK
        "prev_rhk": {
            "label": raw.get("prev_rhk_label"),
            "course": raw.get("prev_rhk_course"),
            "mpap": raw.get("prev_mpap"),
            "pawp": raw.get("prev_pawp"),
            "ci": raw.get("prev_ci"),
            "pvr": raw.get("prev_pvr"),
        },
    }

    planned_actions = {"modules": raw.get("modules") or []}

    return {
        "patient": patient,
        "hemodynamics_rest": rest,
        "hemodynamics_exercise": ex,
        "manoeuvres": manoeuvres,
        "clinical_context": clinical,
        "additional_measurements": add,
        "planned_actions": planned_actions,
    }


def _load_example_values() -> Dict[str, Any]:
    # "Apple‑like" Example: sinnvolle Demo‑Daten
    return {
        "last_name": "Muster",
        "first_name": "Erika",
        "birthdate": "1976-05-11",
        "height_cm": 168,
        "weight_kg": 74,
        "bsa_m2": None,
        "story": "Zunehmende Belastungsdyspnoe, NYHA/WHO‑FC III, Verdacht auf PH.",
        "ph_known": False,
        "ph_suspected": True,
        # RHK
        "mpap": None,
        "pa_sys": 42,
        "pa_dia": 18,
        "pawp": 10,
        "rap": 6,
        "co": 5.0,
        "ci": None,
        "pvr": None,
        "svi": None,
        "hr": 78,
        "svo2": 66,
        "svc_sat": 70,
        "ivc_sat": 68,
        "ra_sat": 69,
        "rv_sat": 69,
        "pa_sat": 66,
        # Belastung
        "exercise_done": True,
        "exercise_ph": True,
        "ex_mpap": 34,
        "ex_pa_sys": 80,
        "ex_pa_dia": 32,
        "ex_pawp": 14,
        "ex_co": 8.0,
        "ex_ci": None,
        "ex_pvr": None,
        "ex_hr": 110,
        "mpap_co_slope": 3.6,
        "pawp_co_slope": 1.2,
        # Volume / Vaso
        "volume_done": False,
        "volume_positive": False,
        "volume_ml": 500,
        "volume_pre_pawp": None,
        "volume_post_pawp": None,
        "vaso_done": False,
        "vaso_responder": False,
        "ino_ppm": 20,
        "vaso_pre_mpap": None,
        "vaso_post_mpap": None,
        "vaso_pre_pvr": None,
        "vaso_post_pvr": None,
        # Klinik/Labor
        "inr": 1.0,
        "quick": 95,
        "crea": 0.9,
        "hst": None,
        "ptt": 30,
        "plt": 250,
        "hb": 13.2,
        "crp": 2.0,
        "leuko": 6.5,
        "bnp_kind": "NT-proBNP",
        "bnp_value": 420,
        "congestive_organopathy": False,
        "ltot_present": False,
        "bga_rest_pO2": 72,
        "bga_rest_pCO2": 36,
        "virology_positive": False,
        "immunology_positive": False,
        "abdo_sono": False,
        "portal_hypertension": False,
        "ct_angio": True,
        "ct_lae": False,
        "ct_ild": False,
        "ct_emphysema": False,
        "ct_embolie": False,
        "ct_mosaic": False,
        "ct_coronarycalc": True,
        "la_enlarged": False,
        "comorbidities": "Arterielle Hypertonie.",
        "comorbidities_relevance": "Hypertonie: möglich relevant.",
        "ph_meds_yesno": False,
        "ph_meds_which": "",
        "ph_meds_since": "",
        "ph_meds_past_yesno": False,
        "other_meds": "Ramipril 5 mg 1-0-0.",
        "diuretics_yesno": False,
        "sbp": 118,
        "egfr": 78,
        "who_fc": "III",
        "syncope": False,
        "sixmwd_m": 320,
        "ve_vco2": 38,
        "vo2max": 15,
        # Lufu
        "lufu_done": True,
        "lufu_obstr": False,
        "lufu_restr": False,
        "lufu_diff": True,
        "lufu_fev1": 2.2,
        "lufu_fvc": 2.8,
        "lufu_fev1_fvc": 0.78,
        "lufu_tlc": 4.6,
        "lufu_rv": 1.6,
        "lufu_dlco": 55,
        "lufu_summary": "DLCO vermindert, sonst keine klare Obstruktion/Restriktion.",
        # Echo/CMR
        "echo_sprime": 9.5,
        "echo_ra_area": 18,
        "pericard_eff": False,
        "pericard_eff_grade": "none",
        "cmr_rvesvi": None,
        "cmr_svi": None,
        "cmr_rvef": None,
        # HFpEF Score
        "hfpef_af": False,
        "hfpef_htn_meds": 1,
        "hfpef_e_eprime": 8,
        "hfpef_pasp": None,
        # Zusatz
        "therapy_plan_sentence": "Falls passend, werden wir eine spezifische PH‑Therapie im Zentrum prüfen.",
        "anticoag_plan_sentence": "",
        "study_sentence": "",
        "declined_item": "",
        "followup_timing_desc": "3–6 Monaten",
        # Prev RHK
        "prev_rhk_label": "03/21",
        "prev_rhk_course": "stabiler Verlauf",
        "prev_mpap": 19,
        "prev_pawp": 7,
        "prev_ci": 3.24,
        "prev_pvr": 1.5,
        # Module
        "modules": ["P01", "P11"],
    }


def build_blocks_app():
    generator = RHKReportGenerator()

    # Module choices: aus TextDB (P/BE/C/G) – früh berechnen, damit Dropdown korrekt befüllt ist
    module_choices: List[str] = []
    try:
        all_blocks = getattr(textdb, "ALL_BLOCKS", {})
        for bid, b in all_blocks.items():
            if not isinstance(bid, str):
                continue
            if bid.startswith(("P", "BE", "C", "G")):
                title = getattr(b, "title", "").strip()
                label = f"{bid} – {title}" if title else bid
                module_choices.append(label)
        module_choices = sorted(module_choices, key=lambda s: s.split(" ")[0])
    except Exception:
        module_choices = ["P01 – Basisdiagnostik", "P11 – Verlauf"]

    css = """
    .gradio-container {max-width: 1400px !important; margin: 0 auto;}
    /* Überschriften: etwas kompakter */
    .gradio-container .prose h1 {margin-bottom: 0.25rem;}
    .gradio-container .prose h2 {margin-top: 1.0rem; margin-bottom: 0.5rem;}
    .gradio-container .prose h3 {margin-top: 0.8rem; margin-bottom: 0.4rem;}
    /* Kartenoptik für ausgewählte Bereiche */
    .rhk-card {
        border: 1px solid var(--border-color-primary);
        border-radius: 16px;
        padding: 16px;
        background: var(--block-background-fill, rgba(255,255,255,0.95));
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    .rhk-card .prose {max-width: none;}
    .rhk-muted {opacity: 0.85;}
    
    /* Sticky top bar */
    .rhk-sticky {
        position: sticky;
        top: 0;
        z-index: 50;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(10px);
        padding: 10px 0 6px 0;
        border-bottom: 1px solid rgba(0,0,0,0.08);
    }
    .rhk-sticky-inner {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .rhk-sticky-head {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .rhk-sticky-h {
        font-weight: 700;
        font-size: 0.95rem;
        line-height: 1.1;
    }
    .rhk-sticky-sub {
        font-size: 0.9rem;
        opacity: 0.9;
        line-height: 1.25;
    }
    .rhk-sticky-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .rhk-chip {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 0.82rem;
        line-height: 1.1;
        white-space: nowrap;
    }
    .rhk-sticky-note {
        font-size: 0.82rem;
        opacity: 0.75;
    }

    /* Keep dashboard visible while scrolling results */
    .rhk-dashboard-sticky {
        position: sticky;
        top: 110px;
    }
"""

    with gr.Blocks(css=css, title="RHK Befundassistent") as demo:
        gr.Markdown(
            """
# RHK Befundassistent
Strukturierte Eingabe → **Befund**, **Patienten‑Info (einfache Sprache)** und **Risiko‑Scores**.

**Hinweis:** Dieses Tool ist eine Formulierungshilfe. Inhalte müssen ärztlich geprüft werden.
"""
        )

        initial_sticky = render_sticky_bar_html(
            "Befundübersicht",
            "Noch kein Befund erstellt. Bitte Werte eingeben und dann 'Befund erstellen' klicken.",
            [],
            note="",
        )
        with gr.Group(elem_classes="rhk-sticky"):
            with gr.Row():
                with gr.Column(scale=6):
                    sticky_out = gr.HTML(value=initial_sticky)
                with gr.Column(scale=2, min_width=260):
                    with gr.Row():
                        example_btn_top = gr.Button("Beispiel laden", variant="secondary")
                        generate_btn_top = gr.Button("Befund erstellen", variant="primary")
        gr.Markdown("---")

        # -------------------------
        # Eingaben (oben)
        # -------------------------
        state = gr.State(_load_example_values())

        def ui_set_values(vals: Dict[str, Any]) -> Tuple[Any, ...]:
            # Rückgabe in genau der Reihenfolge von UI_FIELDS
            return tuple(vals.get(k) for k in UI_FIELDS)

        def ui_get_raw(*vals) -> Dict[str, Any]:
            return {k: v for k, v in zip(UI_FIELDS, vals)}

        # --- Tab layout (Klinik/Labor zuerst)
        with gr.Tabs():
            with gr.Tab("1) Klinik & Labor"):
                with gr.Accordion("Stammdaten & Anamnese", open=True):
                    with gr.Row():
                        last_name = gr.Textbox(label="Name", placeholder="Nachname")
                        first_name = gr.Textbox(label="Vorname", placeholder="Vorname")
                    with gr.Row():
                        birthdate = gr.Textbox(label="Geburtsdatum", placeholder="YYYY-MM-DD oder DD.MM.YYYY")
                        height_cm = gr.Number(label="Größe (cm)", value=None)
                        weight_kg = gr.Number(label="Gewicht (kg)", value=None)
                        bsa_m2 = gr.Number(label="BSA (m², optional – sonst automatisch)", value=None)
                    story = gr.Textbox(label="Story / Kurz‑Anamnese", lines=3)

                    with gr.Row():
                        ph_known = gr.Checkbox(label="PH‑Diagnose bekannt", value=False)
                        ph_suspected = gr.Checkbox(label="PH‑Verdachtsdiagnose", value=True)


                with gr.Accordion("Labor", open=False):
                    with gr.Row():
                        inr = gr.Number(label="INR", value=None)
                        quick = gr.Number(label="Quick (%)", value=None)
                        crea = gr.Number(label="Krea", value=None)
                        hst = gr.Number(label="Hst", value=None)
                    with gr.Row():
                        ptt = gr.Number(label="PTT", value=None)
                        plt = gr.Number(label="Thrombos", value=None)
                        hb = gr.Number(label="Hb", value=None)
                        crp = gr.Number(label="CRP", value=None)
                        leuko = gr.Number(label="Leukos", value=None)

                    with gr.Row():
                        bnp_kind = gr.Dropdown(label="BNP‑Typ", choices=["BNP", "NT-proBNP"], value="NT-proBNP")
                        bnp_value = gr.Number(label="BNP/NT‑proBNP", value=None)
                        congestive_organopathy = gr.Dropdown(label="Hinweis auf congestive Organopathie?", choices=["", "ja", "nein"], value="")


                with gr.Accordion("Blutgase & Sauerstoff", open=False):
                    with gr.Row():
                        ltot_present = gr.Checkbox(label="LTOT vorhanden", value=False)
                        bga_rest_pO2 = gr.Number(label="BGA Ruhe pO₂", value=None)
                        bga_rest_pCO2 = gr.Number(label="BGA Ruhe pCO₂", value=None)


                with gr.Accordion("Infektiologie & Immunologie", open=False):
                    with gr.Row():
                        virology_positive = gr.Checkbox(label="Virologie positiv?", value=False)
                        immunology_positive = gr.Checkbox(label="Immunologie positiv?", value=False)


                with gr.Accordion("Abdomen & Leber", open=False):
                    with gr.Row():
                        abdo_sono = gr.Checkbox(label="Abdomen‑Sono durchgeführt?", value=False)
                        portal_hypertension = gr.Checkbox(label="Hinweis auf portale Hypertension?", value=False)


                with gr.Accordion("Bildgebung Thorax", open=False):
                    with gr.Row():
                        ct_angio = gr.Checkbox(label="CT‑Angio vorhanden", value=False)
                        ct_lae = gr.Checkbox(label="Lungenarterienembolie", value=False)
                        ct_ild = gr.Checkbox(label="ILD", value=False)
                        ct_emphysema = gr.Checkbox(label="Emphysem", value=False)
                        ct_embolie = gr.Checkbox(label="Embolie", value=False)
                        ct_mosaic = gr.Checkbox(label="Mosaikperfusion", value=False)
                        ct_coronarycalc = gr.Checkbox(label="Koronarkalk", value=False)


                with gr.Accordion("Vorerkrankungen & Medikamente", open=False):
                    with gr.Row():
                        comorbidities = gr.Textbox(label="Relevante Vorerkrankungen (Freitext)", lines=2)
                        comorbidities_relevance = gr.Textbox(label="Relevant für PH? (Freitext)", lines=2)
                    with gr.Row():
                        ph_meds_yesno = gr.Checkbox(label="PH‑Medikation aktuell?", value=False)
                        ph_meds_past_yesno = gr.Checkbox(label="PH‑Medikation in der Vergangenheit?", value=False)
                        diuretics_yesno = gr.Checkbox(label="Diuretika?", value=False)
                    with gr.Row():
                        ph_meds_which = gr.Textbox(label="Welche PH‑Medikation?", lines=1)
                        ph_meds_since = gr.Textbox(label="Seit wann?", lines=1)
                        other_meds = gr.Textbox(label="Sonstige Medikation (Freitext)", lines=2)


                with gr.Accordion("Funktionelle Tests", open=False):
                    with gr.Row():
                        who_fc = gr.Dropdown(label="WHO‑FC", choices=["", "I", "II", "III", "IV"], value="")
                        syncope = gr.Checkbox(label="Synkope", value=False)
                        sixmwd_m = gr.Number(label="6MWD (m)", value=None)
                    with gr.Row():
                        ve_vco2 = gr.Number(label="CPET VE/VCO₂", value=None)
                        vo2max = gr.Number(label="CPET VO₂max", value=None)
                    with gr.Row():
                        sbp = gr.Number(label="RRsys (mmHg) – für REVEAL", value=None)
                        egfr = gr.Number(label="eGFR (ml/min/1,73m²) – für REVEAL", value=None)


                with gr.Accordion("HFpEF Hinweise", open=False):
                    with gr.Row():
                        hfpef_af = gr.Checkbox(label="Vorhofflimmern?", value=False)
                        hfpef_htn_meds = gr.Number(label="Anzahl Blutdruck‑Medikamente", value=None)
                        hfpef_e_eprime = gr.Number(label="E/e' (Echo)", value=None)
                        hfpef_pasp = gr.Number(label="PASP/sPAP (mmHg) – optional", value=None)


            with gr.Tab("2) RHK – Ruhe"):
                gr.Markdown("### Hämodynamik Ruhe")
                with gr.Row():
                    mpap = gr.Number(label="mPAP (mmHg) – optional (sonst aus s/dPAP)", value=None)
                    pa_sys = gr.Number(label="sPAP (mmHg)", value=None)
                    pa_dia = gr.Number(label="dPAP (mmHg)", value=None)
                    pawp = gr.Number(label="PAWP (mmHg)", value=None)
                    rap = gr.Number(label="RAP (mmHg)", value=None)

                with gr.Row():
                    co = gr.Number(label="CO (L/min)", value=None)
                    ci = gr.Number(label="CI (L/min/m²) – optional", value=None)
                    pvr = gr.Number(label="PVR (WU) – optional", value=None)
                    svi = gr.Number(label="SVI (ml/m²) – optional", value=None)
                    hr = gr.Number(label="HF (1/min)", value=None)
                    svo2 = gr.Number(label="SvO₂ / PA‑Sättigung (%)", value=None)

                gr.Markdown("### Stufenoximetrie (optional)")
                with gr.Row():
                    svc_sat = gr.Number(label="SVC‑Sat (%)", value=None)
                    ivc_sat = gr.Number(label="IVC‑Sat (%)", value=None)
                    ra_sat = gr.Number(label="RA‑Sat (%)", value=None)
                    rv_sat = gr.Number(label="RV‑Sat (%)", value=None)
                    pa_sat = gr.Number(label="PA‑Sat (%)", value=None)

            with gr.Tab("3) RHK – Belastung / Manöver"):
                gr.Markdown("### Belastung")
                with gr.Row():
                    exercise_done = gr.Checkbox(label="Belastung durchgeführt?", value=False)
                    exercise_ph = gr.Checkbox(label="Belastungs‑PH (manuell)", value=False)
                with gr.Row():
                    ex_mpap = gr.Number(label="mPAP Belastung", value=None)
                    ex_pa_sys = gr.Number(label="sPAP Belastung", value=None)
                    ex_pa_dia = gr.Number(label="dPAP Belastung", value=None)
                    ex_pawp = gr.Number(label="PAWP Belastung", value=None)
                with gr.Row():
                    ex_co = gr.Number(label="CO Belastung (L/min)", value=None)
                    ex_ci = gr.Number(label="CI Belastung (L/min/m²) – optional", value=None)
                    ex_pvr = gr.Number(label="PVR Belastung (WU) – optional", value=None)
                    ex_hr = gr.Number(label="HF Belastung (1/min)", value=None)
                with gr.Row():
                    mpap_co_slope = gr.Number(label="mPAP/CO‑Slope (mmHg/(L/min))", value=None)
                    pawp_co_slope = gr.Number(label="PAWP/CO‑Slope (mmHg/(L/min))", value=None)

                gr.Markdown("### Volumenchallenge")
                with gr.Row():
                    volume_done = gr.Checkbox(label="Volumenchallenge durchgeführt?", value=False)
                    volume_positive = gr.Checkbox(label="Volumenchallenge positiv?", value=False)
                with gr.Row():
                    volume_ml = gr.Number(label="Volumen (ml)", value=500)
                    volume_pre_pawp = gr.Number(label="PAWP vor Volumen (mmHg)", value=None)
                    volume_post_pawp = gr.Number(label="PAWP nach Volumen (mmHg)", value=None)

                gr.Markdown("### Vasoreaktivität")
                with gr.Row():
                    vaso_done = gr.Checkbox(label="Vasoreaktivität durchgeführt?", value=False)
                    vaso_responder = gr.Checkbox(label="Responder?", value=False)
                    ino_ppm = gr.Number(label="iNO (ppm)", value=20)
                with gr.Row():
                    vaso_pre_mpap = gr.Number(label="mPAP vor iNO", value=None)
                    vaso_post_mpap = gr.Number(label="mPAP nach iNO", value=None)
                    vaso_pre_pvr = gr.Number(label="PVR vor iNO", value=None)
                    vaso_post_pvr = gr.Number(label="PVR nach iNO", value=None)

            with gr.Tab("4) Lungenfunktion"):
                with gr.Row():
                    lufu_done = gr.Checkbox(label="Lufu durchgeführt?", value=False)
                    lufu_obstr = gr.Checkbox(label="Obstruktiv", value=False)
                    lufu_restr = gr.Checkbox(label="Restriktiv", value=False)
                    lufu_diff = gr.Checkbox(label="Diffusionsstörung", value=False)
                with gr.Row():
                    lufu_fev1 = gr.Number(label="FEV₁", value=None)
                    lufu_fvc = gr.Number(label="FVC", value=None)
                    lufu_fev1_fvc = gr.Number(label="FEV₁/FVC", value=None)
                    lufu_tlc = gr.Number(label="TLC", value=None)
                    lufu_rv = gr.Number(label="RV", value=None)
                with gr.Row():
                    lufu_dlco = gr.Number(label="DLCO SB (%)", value=None)
                lufu_summary = gr.Textbox(label="Lufu Summary (Freitext, optional)", lines=3)

            with gr.Tab("5) Echo & CMR"):
                gr.Markdown("### Echo – S'/RAAI (Yogeswaran et al.)")
                with gr.Row():
                    echo_sprime = gr.Number(label="S' (cm/s)", value=None)
                    echo_ra_area = gr.Number(label="RA‑Fläche / RA‑ESA (cm²)", value=None)
                with gr.Row():
                    pericard_eff = gr.Checkbox(label="Perikarderguss vorhanden?", value=False)
                    pericard_eff_grade = gr.Dropdown(
                        label="Perikarderguss – Größe (optional)",
                        choices=["none", "minimal", "moderate-large"],
                        value="none",
                    )
                    la_enlarged = gr.Checkbox(label="Linker Vorhof vergrößert?", value=False)

                gr.Markdown("### CMR")
                with gr.Row():
                    cmr_rvesvi = gr.Number(label="RVESVi (ml/m²)", value=None)
                    cmr_svi = gr.Number(label="SVi (ml/m²)", value=None)
                    cmr_rvef = gr.Number(label="RVEF (%)", value=None)

            with gr.Tab("6) Verlauf & Abschluss"):
                gr.Markdown("### Vorheriger RHK (optional)")
                with gr.Row():
                    prev_rhk_label = gr.Textbox(label="RHK‑Datum/Label (z.B. 03/21)", value="")
                    prev_rhk_course = gr.Dropdown(
                        label="Verlauf",
                        choices=["", "stabiler Verlauf", "gebessert", "progredient"],
                        value="",
                    )
                with gr.Row():
                    prev_mpap = gr.Number(label="Vor-RHK mPAP", value=None)
                    prev_pawp = gr.Number(label="Vor-RHK PAWP", value=None)
                    prev_ci = gr.Number(label="Vor-RHK CI", value=None)
                    prev_pvr = gr.Number(label="Vor-RHK PVR", value=None)

                gr.Markdown("### Zusätzliche Sätze (optional, für Procedere)")
                therapy_plan_sentence = gr.Textbox(label="Therapieplan‑Satz (optional)", lines=2, placeholder="z.B. Start/Umstellung einer PH‑Therapie …")
                anticoag_plan_sentence = gr.Textbox(label="Antikoagulations‑Plan (optional)", lines=2, placeholder="z.B. DOAK fortführen, INR Ziel …")
                followup_timing_desc = gr.Textbox(label="Follow‑up Timing (z.B. 3 Monaten)", value="3–6 Monaten")
                declined_item = gr.Textbox(label="Patient lehnt ab (optional)", lines=1, placeholder="z.B. erneute Belastungsmessung")
                study_sentence = gr.Textbox(label="Studien‑Satz (optional)", lines=1)

                gr.Markdown("### Module (zusätzliche Empfehlungen)")
                modules = gr.Dropdown(
                    label="Zusatz‑Module auswählen (Mehrfachauswahl möglich)",
                    choices=module_choices,
                    multiselect=True,
                    value=[],
                )

        # -------------------------
        # Fall speichern / laden (JSON)
        # -------------------------
        with gr.Accordion("Fall speichern / laden", open=False):
            gr.Markdown(
                "Speichert **alle Eingaben** als JSON und lädt sie später wieder in die Maske. "
                "(Für den Transport zwischen Rechnern/Stationsplätzen geeignet.)"
            )
            with gr.Row():
                case_file_in = gr.File(label="Fall laden (.json)", file_types=[".json"], type="filepath")
                load_case_btn = gr.Button("Fall laden")
            with gr.Row():
                save_case_btn = gr.Button("Fall speichern (.json)")
                case_file_out = gr.File(label="Download Fall (.json)")

        # Buttons
        with gr.Row():
            generate_btn = gr.Button("Befund generieren", variant="primary")
            example_btn = gr.Button("Beispiel laden")
            clear_btn = gr.Button("Alles leeren")

        # -------------------------
        # Ergebnisse (unten): erst Scores, dann Befunde
        # -------------------------
        gr.Markdown("---")

        # -------------------------
        # Ergebnisse (Befund + Dashboard)
        # -------------------------
        gr.Markdown("### Ergebnisse")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Befund (ärztlich)"):
                        main_out = gr.Textbox(label="Ärztlicher Befund", lines=28, interactive=False)
                    with gr.Tab("Patientenbericht (sehr einfache Sprache)"):
                        patient_out = gr.Textbox(label="Patientenbericht", lines=28, interactive=False)
                    with gr.Tab("Intern (Logik / Debug)"):
                        internal_out = gr.Textbox(label="Intern", lines=28, interactive=False)

            with gr.Column(scale=1, min_width=380):
                with gr.Group(elem_classes="rhk-card rhk-dashboard-sticky"):
                    gr.Markdown("### Dashboard / Scores (immer sichtbar)")
                    risk_out = gr.HTML(
                        value="<div class='muted'>Noch kein Befund erstellt. Bitte oben auf <b>Befund erstellen</b> klicken.</div>"
                    )

        # Inputs list in same order as UI_FIELDS (components)
        input_components = [
            last_name, first_name, birthdate, height_cm, weight_kg, bsa_m2, story,
            ph_known, ph_suspected, la_enlarged,
            inr, quick, crea, hst, ptt, plt, hb, crp, leuko,
            bnp_kind, bnp_value, congestive_organopathy,
            ltot_present, bga_rest_pO2, bga_rest_pCO2,
            virology_positive, immunology_positive,
            abdo_sono, portal_hypertension,
            ct_angio, ct_lae, ct_ild, ct_emphysema, ct_embolie, ct_mosaic, ct_coronarycalc,
            comorbidities, comorbidities_relevance,
            ph_meds_yesno, ph_meds_past_yesno, diuretics_yesno,
            ph_meds_which, ph_meds_since, other_meds,
            who_fc, syncope, sixmwd_m, ve_vco2, vo2max, sbp, egfr,
            hfpef_af, hfpef_htn_meds, hfpef_e_eprime, hfpef_pasp,
            mpap, pa_sys, pa_dia, pawp, rap, co, ci, pvr, svi, hr, svo2,
            svc_sat, ivc_sat, ra_sat, rv_sat, pa_sat,
            exercise_done, exercise_ph, ex_mpap, ex_pa_sys, ex_pa_dia, ex_pawp, ex_co, ex_ci, ex_pvr, ex_hr, mpap_co_slope, pawp_co_slope,
            volume_done, volume_positive, volume_ml, volume_pre_pawp, volume_post_pawp,
            vaso_done, vaso_responder, ino_ppm, vaso_pre_mpap, vaso_post_mpap, vaso_pre_pvr, vaso_post_pvr,
            lufu_done, lufu_obstr, lufu_restr, lufu_diff, lufu_fev1, lufu_fvc, lufu_fev1_fvc, lufu_tlc, lufu_rv, lufu_dlco, lufu_summary,
            echo_sprime, echo_ra_area, pericard_eff, pericard_eff_grade,
            cmr_rvesvi, cmr_svi, cmr_rvef,
            prev_rhk_label, prev_rhk_course, prev_mpap, prev_pawp, prev_ci, prev_pvr,
            therapy_plan_sentence, anticoag_plan_sentence, followup_timing_desc, declined_item, study_sentence,
            modules,
        ]

        # Generate handler
        def _generate(*vals):
            raw = ui_get_raw(*vals)
            raw["modules"] = _labels_to_ids(raw.get("modules") or [])
            # Organopathy dropdown -> bool/None
            org = (raw.get("congestive_organopathy") or "").strip().lower()
            if org == "ja":
                raw["congestive_organopathy"] = True
            elif org == "nein":
                raw["congestive_organopathy"] = False
            else:
                raw["congestive_organopathy"] = None

            data = build_data_from_ui(raw)
            main, patient_txt, internal, risk_html, sticky_html = generator.generate_all(data)
            return main, patient_txt, internal, risk_html, sticky_html

        generate_btn.click(
            fn=_generate,
            inputs=input_components,
            outputs=[main_out, patient_out, internal_out, risk_out, sticky_out],
        )
        generate_btn_top.click(
            fn=_generate,
            inputs=input_components,
            outputs=[main_out, patient_out, internal_out, risk_out, sticky_out],
        )


        # Example handler
        def _load_example():
            exv = _load_example_values()

            # Dropdown erwartet String-Werte: "", "ja", "nein"
            org = exv.get("congestive_organopathy")
            if org is True:
                exv["congestive_organopathy"] = "ja"
            elif org is False:
                exv["congestive_organopathy"] = "nein"
            elif isinstance(org, str) and org.strip().lower() in ("", "ja", "nein"):
                exv["congestive_organopathy"] = org.strip().lower()
            else:
                exv["congestive_organopathy"] = ""

            # Number-Komponenten: "" -> None (sonst Typfehler beim Befüllen)
            for field, comp in zip(UI_FIELDS, input_components):
                try:
                    if isinstance(comp, gr.Number) and exv.get(field) == "":
                        exv[field] = None
                except Exception:
                    pass

            # map module ids to labels
            ex_ids = exv.get("modules") or []
            labels = []
            for lab in module_choices:
                bid = str(lab).split("–")[0].strip()
                if bid in ex_ids:
                    labels.append(lab)
            exv["modules"] = labels
            return ui_set_values(exv)

        example_btn.click(fn=_load_example, inputs=None, outputs=input_components)
        example_btn_top.click(fn=_load_example, inputs=None, outputs=input_components)


        # Clear handler
        def _clear():
            empty = {k: "" for k in UI_FIELDS}
            # set booleans to default
            for k in ["ph_known","ph_suspected","la_enlarged","ltot_present","virology_positive","immunology_positive","abdo_sono","portal_hypertension",
                      "ct_angio","ct_lae","ct_ild","ct_emphysema","ct_embolie","ct_mosaic","ct_coronarycalc",
                      "ph_meds_yesno","ph_meds_past_yesno","diuretics_yesno","syncope","hfpef_af",
                      "exercise_done","exercise_ph","volume_done","volume_positive","vaso_done","vaso_responder",
                      "lufu_done","lufu_obstr","lufu_restr","lufu_diff","pericard_eff"]:
                empty[k] = False
            empty["bnp_kind"] = "NT-proBNP"
            empty["pericard_eff_grade"] = "none"
            empty["volume_ml"] = 500
            empty["ino_ppm"] = 20
            empty["followup_timing_desc"] = "3–6 Monaten"
            empty["modules"] = []
            return ui_set_values(empty) + ["", "", "", "<div class='muted'>Noch kein Befund erstellt. Bitte oben auf <b>Befund erstellen</b> klicken.</div>", initial_sticky]

        clear_btn.click(fn=_clear, inputs=None, outputs=input_components + [main_out, patient_out, internal_out, risk_out, sticky_out])

        # -------------------------
        # Save / Load (JSON)
        # -------------------------
        def _write_json_tmp(obj: Dict[str, Any]) -> str:
            fd, path = tempfile.mkstemp(prefix="rhk_case_", suffix=".json")
            os.close(fd)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return path

        def _save_case(*vals):
            raw = ui_get_raw(*vals)
            # Modules: UI speichert Labels, Datei speichert IDs (stabiler)
            labels = raw.get("modules") or []
            raw["_modules_labels"] = labels
            raw["modules"] = _labels_to_ids(labels)

            payload = {
                "schema": "rhk_case_v2",
                "app": "rhk_befundassistent",
                "app_version": APP_VERSION,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "ui": raw,
            }
            return _write_json_tmp(payload)

        def _file_to_path(fobj) -> Optional[str]:
            if fobj is None:
                return None
            if isinstance(fobj, str):
                return fobj
            if isinstance(fobj, dict):
                return fobj.get("name") or fobj.get("path") or fobj.get("filepath")
            for attr in ("name", "path", "filepath"):
                p = getattr(fobj, attr, None)
                if isinstance(p, str):
                    return p
            return None

        def _load_case(file_obj):
            path = _file_to_path(file_obj)
            if not path or not os.path.exists(path):
                return tuple(gr.update() for _ in input_components)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except Exception:
                return tuple(gr.update() for _ in input_components)

            if isinstance(loaded, dict) and isinstance(loaded.get("ui"), dict):
                ui = dict(loaded.get("ui") or {})
            elif isinstance(loaded, dict):
                ui = dict(loaded)
            else:
                return tuple(gr.update() for _ in input_components)

            # Dropdown erwartet String: "", "ja", "nein"
            if "congestive_organopathy" in ui:
                org = ui.get("congestive_organopathy")
                if org is True:
                    ui["congestive_organopathy"] = "ja"
                elif org is False:
                    ui["congestive_organopathy"] = "nein"
                elif isinstance(org, str) and org.strip().lower() in ("", "ja", "nein"):
                    ui["congestive_organopathy"] = org.strip().lower()
                else:
                    ui["congestive_organopathy"] = ""

            # Module können als IDs oder als Labels gespeichert sein
            if "modules" in ui:
                mod_val = ui.get("modules")
                labels_to_set: List[str] = []
                if isinstance(mod_val, list):
                    # bereits Labels?
                    if any(isinstance(x, str) and "–" in x for x in mod_val):
                        labels_to_set = [x for x in mod_val if x in module_choices]
                    else:
                        ids = [x for x in mod_val if isinstance(x, str)]
                        for lab in module_choices:
                            bid = str(lab).split("–")[0].strip()
                            if bid in ids:
                                labels_to_set.append(lab)
                ui["modules"] = labels_to_set

            # Build outputs: fehlende Keys -> gr.update() (ändert nichts)
            out: List[Any] = []
            for field, comp in zip(UI_FIELDS, input_components):
                if field not in ui:
                    out.append(gr.update())
                    continue

                v = ui.get(field)

                # modules ist Multiselect-Dropdown
                if field == "modules":
                    out.append(v if isinstance(v, list) else [])
                    continue

                # Numeric
                if isinstance(comp, gr.Number):
                    if v in ("", None):
                        out.append(None)
                    else:
                        out.append(to_num(v))
                    continue

                # Bool
                if isinstance(comp, gr.Checkbox):
                    out.append(bool(v) if v is not None else False)
                    continue

                # Dropdown
                if isinstance(comp, gr.Dropdown):
                    try:
                        choices = comp.choices  # type: ignore[attr-defined]
                    except Exception:
                        choices = None
                    if v is None:
                        out.append("")
                    elif choices and v not in choices:
                        out.append("" if "" in choices else (choices[0] if choices else v))
                    else:
                        out.append(v)
                    continue

                # Text
                out.append("" if v is None else v)

            return tuple(out)

        save_case_btn.click(fn=_save_case, inputs=input_components, outputs=[case_file_out])
        load_case_btn.click(fn=_load_case, inputs=[case_file_in], outputs=input_components)

    return demo



def build_interface_app():
    """
    Fallback‑UI für sehr alte Gradio‑Versionen (ohne Blocks/Tabs).

    Einschränkungen:
    - Keine Tabs/Spalten. Inputs sind der Reihe nach sortiert (Klinik/Labor zuerst).
    - Mehrfachauswahl für Module via CheckboxGroup.
    """
    generator = RHKReportGenerator()

    # Module choices (IDs + Titel)
    module_choices: List[str] = []
    try:
        all_blocks = getattr(textdb, "ALL_BLOCKS", {})
        for bid, b in all_blocks.items():
            if not isinstance(bid, str):
                continue
            if bid.startswith(("P", "BE", "C", "G")):
                title = getattr(b, "title", "").strip()
                label = f"{bid} – {title}" if title else bid
                module_choices.append(label)
        module_choices = sorted(module_choices, key=lambda s: s.split(" ")[0])
    except Exception:
        module_choices = ["P01 – Basisdiagnostik", "P11 – Verlauf"]

    id_to_label = {str(lab).split("–")[0].strip(): lab for lab in module_choices}

    def _labels_to_ids(labels: List[str]) -> List[str]:
        ids: List[str] = []
        for lab in labels or []:
            bid = str(lab).split("–")[0].strip()
            if bid and bid not in ids:
                ids.append(bid)
        return ids

    UI_FIELDS = [
        # Klinik/Labor
        "last_name","first_name","birthdate","height_cm","weight_kg","bsa_m2","story",
        "ph_known","ph_suspected","la_enlarged",
        "inr","quick","crea","hst","ptt","plt","hb","crp","leuko",
        "bnp_kind","bnp_value","congestive_organopathy",
        "ltot_present","bga_rest_pO2","bga_rest_pCO2",
        "virology_positive","immunology_positive",
        "abdo_sono","portal_hypertension",
        "ct_angio","ct_lae","ct_ild","ct_emphysema","ct_embolie","ct_mosaic","ct_coronarycalc",
        "comorbidities","comorbidities_relevance",
        "ph_meds_yesno","ph_meds_past_yesno","diuretics_yesno",
        "ph_meds_which","ph_meds_since","other_meds",
        "who_fc","syncope","sixmwd_m","ve_vco2","vo2max","sbp","egfr",
        "hfpef_af","hfpef_htn_meds","hfpef_e_eprime","hfpef_pasp",
        # RHK Ruhe
        "mpap","pa_sys","pa_dia","pawp","rap","co","ci","pvr","svi","hr","svo2",
        "svc_sat","ivc_sat","ra_sat","rv_sat","pa_sat",
        # Belastung/Manöver
        "exercise_done","exercise_ph","ex_mpap","ex_pa_sys","ex_pa_dia","ex_pawp","ex_co","ex_ci","ex_pvr","ex_hr","mpap_co_slope","pawp_co_slope",
        "volume_done","volume_positive","volume_ml","volume_pre_pawp","volume_post_pawp",
        "vaso_done","vaso_responder","ino_ppm","vaso_pre_mpap","vaso_post_mpap","vaso_pre_pvr","vaso_post_pvr",
        # Lufu
        "lufu_done","lufu_obstr","lufu_restr","lufu_diff","lufu_fev1","lufu_fvc","lufu_fev1_fvc","lufu_tlc","lufu_rv","lufu_dlco","lufu_summary",
        # Echo/CMR
        "echo_sprime","echo_ra_area","pericard_eff","pericard_eff_grade",
        "cmr_rvesvi","cmr_svi","cmr_rvef",
        # Verlauf/Abschluss
        "prev_rhk_label","prev_rhk_course","prev_mpap","prev_pawp","prev_ci","prev_pvr",
        "therapy_plan_sentence","anticoag_plan_sentence","followup_timing_desc","declined_item","study_sentence",
        "modules",
    ]

    # Spezifikation: Typen
    bool_fields = {
        "ph_known","ph_suspected","la_enlarged","ltot_present","virology_positive","immunology_positive",
        "abdo_sono","portal_hypertension","ct_angio","ct_lae","ct_ild","ct_emphysema","ct_embolie","ct_mosaic","ct_coronarycalc",
        "ph_meds_yesno","ph_meds_past_yesno","diuretics_yesno","syncope","hfpef_af",
        "exercise_done","exercise_ph","volume_done","volume_positive","vaso_done","vaso_responder",
        "lufu_done","lufu_obstr","lufu_restr","lufu_diff","pericard_eff",
    }

    number_fields = {
        "height_cm","weight_kg","bsa_m2","inr","quick","crea","hst","ptt","plt","hb","crp","leuko","bnp_value",
        "bga_rest_pO2","bga_rest_pCO2","sixmwd_m","ve_vco2","vo2max","sbp","egfr",
        "hfpef_htn_meds","hfpef_e_eprime","hfpef_pasp",
        "mpap","pa_sys","pa_dia","pawp","rap","co","ci","pvr","svi","hr","svo2",
        "svc_sat","ivc_sat","ra_sat","rv_sat","pa_sat",
        "ex_mpap","ex_pa_sys","ex_pa_dia","ex_pawp","ex_co","ex_ci","ex_pvr","ex_hr","mpap_co_slope","pawp_co_slope",
        "volume_ml","volume_pre_pawp","volume_post_pawp",
        "ino_ppm","vaso_pre_mpap","vaso_post_mpap","vaso_pre_pvr","vaso_post_pvr",
        "lufu_fev1","lufu_fvc","lufu_fev1_fvc","lufu_tlc","lufu_rv","lufu_dlco",
        "echo_sprime","echo_ra_area",
        "cmr_rvesvi","cmr_svi","cmr_rvef",
        "prev_mpap","prev_pawp","prev_ci","prev_pvr",
    }

    dropdown_specs = {
        "bnp_kind": (["BNP","NT-proBNP"], "NT-proBNP"),
        "congestive_organopathy": (["", "ja", "nein"], ""),
        "who_fc": (["", "I", "II", "III", "IV"], ""),
        "pericard_eff_grade": (["none", "minimal", "moderate-large"], "none"),
        "prev_rhk_course": (["", "stabiler Verlauf", "gebessert", "progredient"], ""),
    }

    multiline_fields = {
        "story": 3,
        "comorbidities": 2,
        "comorbidities_relevance": 2,
        "other_meds": 2,
        "lufu_summary": 3,
        "therapy_plan_sentence": 2,
        "anticoag_plan_sentence": 2,
        "declined_item": 1,
        "study_sentence": 1,
    }

    label_map = {
        "last_name": "Name",
        "first_name": "Vorname",
        "birthdate": "Geburtsdatum (YYYY-MM-DD oder DD.MM.YYYY)",
        "story": "Story / Kurz‑Anamnese",
        "ph_known": "PH‑Diagnose bekannt",
        "ph_suspected": "PH‑Verdachtsdiagnose",
        "la_enlarged": "Linkes Atrium vergrößert",
        "ct_lae": "Lungenarterienembolie (LAE)",
        "bnp_kind": "BNP‑Typ",
        "bnp_value": "BNP/NT‑proBNP",
        "congestive_organopathy": "Hinweis auf congestive Organopathie?",
        "modules": "Zusatz‑Module (Mehrfachauswahl)",
    }

    def _make_input(field: str):
        label = label_map.get(field, field)
        if field == "modules":
            return gr.inputs.CheckboxGroup(choices=module_choices, default=[], label=label)
        if field in dropdown_specs:
            choices, default = dropdown_specs[field]
            return gr.inputs.Dropdown(choices=choices, default=default, label=label)
        if field in bool_fields:
            return gr.inputs.Checkbox(default=False, label=label)
        if field in number_fields:
            # numeric default for a few
            default = None
            if field == "volume_ml":
                default = 500
            if field == "ino_ppm":
                default = 20
            if field == "followup_timing_desc":
                default = "3–6 Monaten"
            return gr.inputs.Number(default=default, label=label)
        # text
        lines = multiline_fields.get(field, 1)
        default = ""
        if field == "followup_timing_desc":
            default = "3–6 Monaten"
        return gr.inputs.Textbox(default=default, label=label, lines=lines)

    inputs = [_make_input(f) for f in UI_FIELDS]

    outputs = [
        gr.outputs.HTML(label="Risiko‑Scores"),
        gr.outputs.Textbox(label="Ärztlicher Befund"),
        gr.outputs.Textbox(label="Patienten‑Information (einfache Sprache)"),
        gr.outputs.Textbox(label="Interner Log"),
    ]

    def _generate(*vals):
        raw = {k: v for k, v in zip(UI_FIELDS, vals)}
        # modules: labels -> ids
        raw["modules"] = _labels_to_ids(raw.get("modules") or [])

        # Organopathy dropdown -> bool/None
        org = (raw.get("congestive_organopathy") or "").strip().lower()
        if org == "ja":
            raw["congestive_organopathy"] = True
        elif org == "nein":
            raw["congestive_organopathy"] = False
        else:
            raw["congestive_organopathy"] = None

        data = build_data_from_ui(raw)
        main, patient_txt, internal, risk_html, sticky_html = generator.generate_all(data)
        return risk_html, main, patient_txt, internal

    # Examples (ein Beispiel)
    exv = _load_example_values()
    # modules ids -> labels
    ex_ids = exv.get("modules") or []
    exv["modules"] = [id_to_label[i] for i in ex_ids if i in id_to_label]
    # organopathy bool -> dropdown string
    if exv.get("congestive_organopathy") is True:
        exv["congestive_organopathy"] = "ja"
    elif exv.get("congestive_organopathy") is False:
        exv["congestive_organopathy"] = "nein"
    else:
        exv["congestive_organopathy"] = ""

    examples = [[exv.get(k) for k in UI_FIELDS]]

    title = "RHK Befundassistent"
    description = "Strukturierte Eingabe → Befund + Patienten‑Info + Risiko‑Scores. (Fallback‑UI ohne Tabs.)"

    iface = gr.Interface(
        fn=_generate,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
        examples=examples,
        allow_flagging=False,
    )
    return iface


def build_app():
    # Moderne Gradio-Version? Dann Blocks‑UI, sonst Fallback‑Interface.
    if hasattr(gr, "Blocks"):
        return build_blocks_app()
    return build_interface_app()


# ============================================================
# v16 OVERRIDES / UI & LOGIC IMPROVEMENTS (2025-12-26)
# ============================================================

# Bump visible version (shown in UI header)
APP_VERSION = "v16.0.0 (2025-12-26)"
SCHEMA_VERSION = 6  # for saved JSON cases

def _compute_bsa_mosteller(height_cm: float | None, weight_kg: float | None) -> float | None:
    """Mosteller BSA [m²]."""
    try:
        if height_cm is None or weight_kg is None:
            return None
        if height_cm <= 0 or weight_kg <= 0:
            return None
        return ((height_cm * weight_kg) / 3600.0) ** 0.5
    except Exception:
        return None

def _compute_mpap_from_spap_dpap(spap: float | None, dpap: float | None) -> float | None:
    try:
        if spap is None or dpap is None:
            return None
        return (spap + 2.0 * dpap) / 3.0
    except Exception:
        return None

def _infer_exercise_done(ex: dict) -> bool:
    """Exercise is considered 'done' if the checkbox is set OR any key exercise value was entered."""
    if bool(ex.get("exercise_done")):
        return True
    numeric_keys = [
        "ex_spap","ex_dpap","ex_mpap","ex_pawp","ex_rap","ex_hr",
        "ex_co","ex_ci","ex_pvr","mpap_co_slope","pawp_co_slope",
        "ex_svo2","ex_art_sat"
    ]
    return any(to_num(ex.get(k)) is not None for k in numeric_keys)

def _preprocess_exercise_for_interactivity(data: dict) -> None:
    """
    Make the tool more 'ultra-interaktiv':
    - Auto-detect exercise_done when the user entered peak/exercise values.
    - Auto-calculate mPAP, CO and slopes (mPAP/CO, PAWP/CO) from rest vs peak values.
    - If exercise pattern is clearly abnormal, set exercise_ph=True so that the Befund
      reacts (K02/K03 selection etc.) even if the checkbox wasn't set.
    """
    try:
        rhk = data.get("rhk") or {}
        ex = data.get("exercise") or {}
        meta = data.get("meta") or {}

        # Determine BSA for CO<->CI conversions
        height_cm = to_num(meta.get("height_cm"))
        weight_kg = to_num(meta.get("weight_kg"))
        bsa = to_num(meta.get("bsa")) or _compute_bsa_mosteller(height_cm, weight_kg)

        # Rest values
        rest_spap = to_num(rhk.get("pap_syst"))
        rest_dpap = to_num(rhk.get("pap_diast"))
        rest_mpap = to_num(rhk.get("mpap")) or _compute_mpap_from_spap_dpap(rest_spap, rest_dpap)
        rest_pawp = to_num(rhk.get("pawp"))

        rest_co = to_num(rhk.get("co"))
        rest_ci = to_num(rhk.get("ci"))
        if rest_co is None and rest_ci is not None and bsa is not None:
            rest_co = rest_ci * bsa

        # Exercise/peak values
        ex_spap = to_num(ex.get("ex_spap"))
        ex_dpap = to_num(ex.get("ex_dpap"))
        ex_mpap = to_num(ex.get("ex_mpap")) or _compute_mpap_from_spap_dpap(ex_spap, ex_dpap)
        if ex.get("ex_mpap") is None and ex_mpap is not None:
            ex["ex_mpap"] = ex_mpap

        ex_pawp = to_num(ex.get("ex_pawp"))

        ex_co = to_num(ex.get("ex_co"))
        ex_ci = to_num(ex.get("ex_ci"))
        if ex_co is None and ex_ci is not None and bsa is not None:
            ex_co = ex_ci * bsa

        # Auto exercise_done
        if _infer_exercise_done(ex):
            ex["exercise_done"] = True

        # Auto slopes if possible
        def _safe_slope(rest_val, peak_val, rest_flow, peak_flow):
            try:
                if rest_val is None or peak_val is None or rest_flow is None or peak_flow is None:
                    return None
                dflow = peak_flow - rest_flow
                if abs(dflow) < 1e-6:
                    return None
                return (peak_val - rest_val) / dflow
            except Exception:
                return None

        mpap_co_slope = to_num(ex.get("mpap_co_slope"))
        pawp_co_slope = to_num(ex.get("pawp_co_slope"))

        if mpap_co_slope is None:
            mpap_co_slope = _safe_slope(rest_mpap, ex_mpap, rest_co, ex_co)
            if mpap_co_slope is not None:
                ex["mpap_co_slope"] = mpap_co_slope

        if pawp_co_slope is None:
            pawp_co_slope = _safe_slope(rest_pawp, ex_pawp, rest_co, ex_co)
            if pawp_co_slope is not None:
                ex["pawp_co_slope"] = pawp_co_slope

        # If we cannot compute a slope but PAWP rose a lot, treat as left-heart pattern
        # so it has consequences in Befund + Patientenbericht
        delta_pawp = None
        try:
            if rest_pawp is not None and ex_pawp is not None:
                delta_pawp = ex_pawp - rest_pawp
        except Exception:
            delta_pawp = None

        if pawp_co_slope is None and delta_pawp is not None and delta_pawp >= 10:
            ex["pawp_co_slope"] = 2.5  # heuristic to trigger linkskardial pattern

        # If exercise data strongly suggests an abnormal pattern, mark as relevant.
        # This makes the Befund react even if the user didn't tick "exercise_ph".
        # (Label in UI is adjusted accordingly.)
        if bool(ex.get("exercise_done")):
            mpap_s = to_num(ex.get("mpap_co_slope"))
            pawp_s = to_num(ex.get("pawp_co_slope"))
            pattern = classify_exercise_pattern(mpap_s, pawp_s)
            if pattern in ("pulmvasc", "linkskardial", "isoliert_pawp"):
                ex["exercise_ph"] = True

        data["exercise"] = ex
    except Exception:
        # Never crash report generation because of preprocessing
        return

def _interactive_details_doctor(data: dict) -> str:
    """Extra details that should appear in the physician report (Befund)."""
    try:
        clinical = data.get("clinical_context") or {}
        labs = data.get("labs") or {}
        gases = data.get("gases") or {}
        lufu = data.get("lufu") or {}

        lines = []

        # ILD details
        if clinical.get("ct_ild"):
            ild_type = (clinical.get("ct_ild_type") or "").strip()
            ild_extent = (clinical.get("ct_ild_extent") or "").strip()
            ild_hist = bool(clinical.get("ct_ild_histology"))
            ild_fib_clinic = bool(clinical.get("ct_ild_fibrosis_clinic"))
            parts = []
            if ild_type:
                parts.append(f"Typ: {ild_type}")
            if ild_extent:
                parts.append(f"Ausmaß: {ild_extent}")
            if ild_hist:
                parts.append("histologisch gesichert")
            if ild_fib_clinic:
                parts.append("Fibrose-/ILD-Sprechstunde (angebunden)")
            if parts:
                lines.append("ILD: " + ", ".join(parts))

        # Virology / immunology details
        if clinical.get("virology_positive"):
            det = (clinical.get("virology_details") or "").strip()
            lines.append("Virologie positiv" + (f" ({det})" if det else ""))

        if clinical.get("immunology_positive"):
            det = (clinical.get("immunology_details") or "").strip()
            lines.append("Immunologie/Autoantikörper positiv" + (f" ({det})" if det else ""))

        # Abdomen sono details
        if clinical.get("abdo_sono"):
            det = (clinical.get("abdo_findings") or "").strip()
            portal = clinical.get("portal_hypertension")
            extra = []
            if det:
                extra.append(det)
            if portal is True:
                extra.append("Hinweis auf portale Hypertension")
            elif portal is False:
                extra.append("keine portale Hypertension")
            if extra:
                lines.append("Abdomen-Sono: " + "; ".join(extra))

        # LTOT flow
        if gases.get("ltot_present"):
            flow = to_num(gases.get("ltot_flow_l_min"))
            if flow is not None:
                lines.append(f"LTOT: {flow:g} L/min")
            else:
                lines.append("LTOT: ja")

        # BNP + Entresto hint
        if labs.get("bnp_kind") == "BNP" and bool(labs.get("entresto")):
            lines.append("Hinweis: BNP unter Sacubitril/Valsartan (Entresto) eingeschränkt interpretierbar – NT-proBNP bevorzugen.")

        # Functional symptom quantifiers
        stairs = to_num(clinical.get("stairs"))
        walk_m = to_num(clinical.get("walk_distance_m"))
        if stairs is not None:
            lines.append(f"Belastbarkeit: ca. {stairs:g} Stockwerke")
        if walk_m is not None:
            lines.append(f"Belastbarkeit: ca. {walk_m:g} m Gehstrecke")

        if not lines:
            return ""
        return "\n\n### Zusätzliche, interaktiv erfasste Details\n" + "\n".join([f"- {l}" for l in lines])
    except Exception:
        return ""

def _interactive_details_patient(data: dict) -> str:
    """Extra details for patient report (without numbers/abbreviations)."""
    try:
        clinical = data.get("clinical_context") or {}
        labs = data.get("labs") or {}
        gases = data.get("gases") or {}

        paras = []

        if gases.get("ltot_present"):
            paras.append("Sie erhalten eine Langzeitsauerstofftherapie. Bitte verwenden Sie den Sauerstoff so, wie es mit Ihnen vereinbart wurde.")

        if clinical.get("ct_ild"):
            paras.append("In der Bildgebung gibt es Hinweise auf eine chronische Lungenerkrankung (zum Beispiel Vernarbungen). Das kann die Belastbarkeit beeinflussen und kann bei der Ursache Ihrer Beschwerden eine Rolle spielen.")

        if clinical.get("virology_positive") or clinical.get("immunology_positive"):
            paras.append("Einige Blutuntersuchungen waren auffällig. Wir besprechen mit Ihnen, welche weiteren Abklärungen sinnvoll sind und ob eine Mitbehandlung durch Spezialambulanzen hilfreich ist.")

        if labs.get("bnp_kind") == "BNP" and bool(labs.get("entresto")):
            paras.append("Ein bestimmter Laborwert zur Herzbelastung ist unter einer speziellen Herzmedikation weniger aussagekräftig. Deshalb werden wir – wenn nötig – andere Blutwerte bzw. Verlaufskontrollen nutzen.")

        if not paras:
            return ""
        return "\n\n**Zusatzinformationen**\n\n" + "\n\n".join(paras)
    except Exception:
        return ""

# --- Override: build_data_from_ui (adds new interactive fields) ---
def build_data_from_ui(raw: dict) -> dict:
    data = {
        "meta": {
            "last_name": raw.get("last_name") or "",
            "first_name": raw.get("first_name") or "",
            "dob": raw.get("dob") or "",
            "sex": raw.get("sex") or "",
            "height_cm": to_num(raw.get("height_cm")),
            "weight_kg": to_num(raw.get("weight_kg")),
            "bsa": to_num(raw.get("bsa")),
            "center": raw.get("center") or "",
            "date": raw.get("date") or "",
        },
        "history": {
            "indication": raw.get("indication") or "",
            "dyspnea": raw.get("dyspnea") or "",
            "syncope": bool(raw.get("syncope")),
            "angina": bool(raw.get("angina")),
            "edema": bool(raw.get("edema")),
            "covid_history": bool(raw.get("covid_history")),
            "smoking": raw.get("smoking") or "",
            "comorbidities": raw.get("comorbidities") or "",
        },
        "labs": {
            "hb": to_num(raw.get("hb")),
            "creatinine": to_num(raw.get("creatinine")),
            "bnp_kind": raw.get("bnp_kind") or "",
            "bnp": to_num(raw.get("bnp")),
            "ntprobnp": to_num(raw.get("ntprobnp")),
            "entresto": bool(raw.get("entresto")),
            "crp": to_num(raw.get("crp")),
            "tsh": to_num(raw.get("tsh")),
            "troponin": to_num(raw.get("troponin")),
            "d_dimer": to_num(raw.get("d_dimer")),
            "autoimmune_screen": bool(raw.get("autoimmune_screen")),
        },
        "gases": {
            "pH": to_num(raw.get("pH")),
            "pCO2": to_num(raw.get("pCO2")),
            "pO2": to_num(raw.get("pO2")),
            "hco3": to_num(raw.get("hco3")),
            "sao2": to_num(raw.get("sao2")),
            "ltot_present": bool(raw.get("ltot_present")),
            "ltot_flow_l_min": to_num(raw.get("ltot_flow_l_min")),
        },
        "rhk": {
            "rap": to_num(raw.get("rap")),
            "pap_syst": to_num(raw.get("pap_syst")),
            "pap_diast": to_num(raw.get("pap_diast")),
            "mpap": to_num(raw.get("mpap")),
            "pawp": to_num(raw.get("pawp")),
            "co": to_num(raw.get("co")),
            "ci": to_num(raw.get("ci")),
            "pvr": to_num(raw.get("pvr")),
            "svr": to_num(raw.get("svr")),
            "hr": to_num(raw.get("hr")),
            "svo2": to_num(raw.get("svo2")),
            "art_sat": to_num(raw.get("art_sat")),
            "notes": raw.get("rhk_notes") or "",
        },
        "exercise": {
            "exercise_done": bool(raw.get("exercise_done")),
            "exercise_ph": bool(raw.get("exercise_ph")),
            "ex_rap": to_num(raw.get("ex_rap")),
            "ex_spap": to_num(raw.get("ex_spap")),
            "ex_dpap": to_num(raw.get("ex_dpap")),
            "ex_mpap": to_num(raw.get("ex_mpap")),
            "ex_pawp": to_num(raw.get("ex_pawp")),
            "ex_co": to_num(raw.get("ex_co")),
            "ex_ci": to_num(raw.get("ex_ci")),
            "ex_pvr": to_num(raw.get("ex_pvr")),
            "ex_hr": to_num(raw.get("ex_hr")),
            "ex_svo2": to_num(raw.get("ex_svo2")),
            "ex_art_sat": to_num(raw.get("ex_art_sat")),
            "mpap_co_slope": to_num(raw.get("mpap_co_slope")),
            "pawp_co_slope": to_num(raw.get("pawp_co_slope")),
            "exercise_notes": raw.get("exercise_notes") or "",
        },
        "maneuvers": {
            "volume_challenge_done": bool(raw.get("volume_challenge_done")),
            "volume_positive": bool(raw.get("volume_positive")),
            "volume_pawp": to_num(raw.get("volume_pawp")),
            "volume_notes": raw.get("volume_notes") or "",
            "vasoreactivity_done": bool(raw.get("vasoreactivity_done")),
            "vasoreactivity_positive": bool(raw.get("vasoreactivity_positive")),
            "vasoreactivity_agent": raw.get("vasoreactivity_agent") or "",
            "vasoreactivity_notes": raw.get("vasoreactivity_notes") or "",
            "stepup_done": bool(raw.get("stepup_done")),
            "stepup_positive": bool(raw.get("stepup_positive")),
            "stepup_site": raw.get("stepup_site") or "",
            "stepup_notes": raw.get("stepup_notes") or "",
        },
        "previous_rhk": {
            "prev_date": raw.get("prev_date") or "",
            "prev_mpap": to_num(raw.get("prev_mpap")),
            "prev_pawp": to_num(raw.get("prev_pawp")),
            "prev_ci": to_num(raw.get("prev_ci")),
            "prev_pvr": to_num(raw.get("prev_pvr")),
            "prev_notes": raw.get("prev_notes") or "",
        },
        "lufu": {
            "fvc": to_num(raw.get("fvc")),
            "fvc_pct": to_num(raw.get("fvc_pct")),
            "fev1": to_num(raw.get("fev1")),
            "fev1_pct": to_num(raw.get("fev1_pct")),
            "fev1_fvc": to_num(raw.get("fev1_fvc")),
            "dlco": to_num(raw.get("dlco")),
            "dlco_pct": to_num(raw.get("dlco_pct")),
            "tlc": to_num(raw.get("tlc")),
            "tlc_pct": to_num(raw.get("tlc_pct")),
            "lufu_summary": raw.get("lufu_summary") or "",
        },
        "clinical_context": {
            # Imaging (CT)
            "ct_angio_done": bool(raw.get("ct_angio_done")),
            "ct_lae": bool(raw.get("ct_lae")),
            "ct_ild": bool(raw.get("ct_ild")),
            "ct_ild_type": raw.get("ct_ild_type") or "",
            "ct_ild_extent": raw.get("ct_ild_extent") or "",
            "ct_ild_histology": bool(raw.get("ct_ild_histology")),
            "ct_ild_fibrosis_clinic": bool(raw.get("ct_ild_fibrosis_clinic")),
            "ct_emphysema": bool(raw.get("ct_emphysema")),
            "ct_embolie": bool(raw.get("ct_embolie")),
            "ct_mosaic": bool(raw.get("ct_mosaic")),
            "ct_corcal": bool(raw.get("ct_corcal")),

            # Echo
            "echo_done": bool(raw.get("echo_done")),
            "echo_rv_dilation": raw.get("echo_rv_dilation") or "",
            "echo_rv_function": raw.get("echo_rv_function") or "",
            "echo_tapse": to_num(raw.get("echo_tapse")),
            "echo_sprime": to_num(raw.get("echo_sprime")),
            "echo_raa": to_num(raw.get("echo_raa")),
            "echo_raai": to_num(raw.get("echo_raai")),
            "echo_pericardial_effusion": raw.get("echo_pericardial_effusion") or "",
            "echo_tr_vmax": to_num(raw.get("echo_tr_vmax")),
            "echo_rvsp": to_num(raw.get("echo_rvsp")),
            "echo_la_enlarged": bool(raw.get("echo_la_enlarged")),

            # CMR
            "cmr_done": bool(raw.get("cmr_done")),
            "cmr_rvef": to_num(raw.get("cmr_rvef")),
            "cmr_rv_edvi": to_num(raw.get("cmr_rv_edvi")),
            "cmr_lv_ef": to_num(raw.get("cmr_lv_ef")),
            "cmr_lge": bool(raw.get("cmr_lge")),

            # Infection/Immunology
            "virology_positive": bool(raw.get("virology_positive")),
            "virology_details": raw.get("virology_details") or "",
            "immunology_positive": bool(raw.get("immunology_positive")),
            "immunology_details": raw.get("immunology_details") or "",

            # Abdomen
            "abdo_sono": bool(raw.get("abdo_sono")),
            "abdo_findings": raw.get("abdo_findings") or "",
            "portal_hypertension": None if raw.get("portal_hypertension") in (None, "") else bool(raw.get("portal_hypertension")),

            # Meds / Therapy
            "ph_meds_current": bool(raw.get("ph_meds_current")),
            "ph_meds_current_what": raw.get("ph_meds_current_what") or "",
            "ph_meds_current_since": raw.get("ph_meds_current_since") or "",
            "ph_meds_past": bool(raw.get("ph_meds_past")),
            "ph_meds_past_what": raw.get("ph_meds_past_what") or "",
            "ph_meds_past_since": raw.get("ph_meds_past_since") or "",

            # Functional symptoms (new)
            "hemoptysis": bool(raw.get("hemoptysis")),
            "dizziness": bool(raw.get("dizziness")),
            "stairs": to_num(raw.get("stairs")),
            "walk_distance_m": to_num(raw.get("walk_distance_m")),

            # Other
            "la_enlarged": bool(raw.get("la_enlarged")),
        },
        "additional_measurements": {
            # Risk / clinical
            "who_fc": raw.get("who_fc") or "",
            "six_mwd": to_num(raw.get("six_mwd")),
            "reveal_renal": bool(raw.get("reveal_renal")),
            "reveal_hosp6": bool(raw.get("reveal_hosp6")),
            "reveal_hr": to_num(raw.get("reveal_hr")),
            "reveal_sysbp": to_num(raw.get("reveal_sysbp")),
            "reveal_bnp": to_num(raw.get("reveal_bnp")),
            "reveal_ntprobnp": to_num(raw.get("reveal_ntprobnp")),
            "reveal_dlco_pct": to_num(raw.get("reveal_dlco_pct")),
            "reveal_pericardial_effusion": bool(raw.get("reveal_pericardial_effusion")),
            "reveal_functional_class": raw.get("reveal_functional_class") or "",
            "hfpef_age": to_num(raw.get("hfpef_age")),
            "hfpef_bmi": to_num(raw.get("hfpef_bmi")),
            "hfpef_af": bool(raw.get("hfpef_af")),
            "hfpef_htn_meds": to_num(raw.get("hfpef_htn_meds")),
            "hfpef_e_eprime": to_num(raw.get("hfpef_e_eprime")),
            "hfpef_pasp": to_num(raw.get("hfpef_pasp")),
        },
        "planned_actions": {
            "force_package": raw.get("force_package") or "",
            "modules": raw.get("modules") or [],
            "recommendation_free_text": raw.get("recommendation_free_text") or "",
        }
    }
    return data

# --- Override build_blocks_app with merged imaging + echo/CMR, conditional fields, cleaner top bar ---
def build_blocks_app():
    import gradio as gr
    generator = RHKReportGenerator()

    # Keep the nice v14 CSS, but pass it to launch (Blocks(css=...) triggers warnings in newer Gradio)
    css = r"""
    .rhk-title {font-weight:800; font-size:20px;}
    .rhk-sub   {color: #666; font-size: 13px;}
    .card {border: 1px solid rgba(0,0,0,.08); border-radius: 12px; padding: 14px 16px; background: white;}
    .card h3 {margin: 0 0 8px 0; font-size: 14px;}
    .card .big {font-size: 20px; font-weight: 800;}
    .muted {color: #666;}
    .btnrow {gap: 8px;}
    """

    try:
        theme = gr.themes.Soft()
    except Exception:
        theme = None

    # Helper to create robust mapping (prevents "Befund updates not working" due to order mismatches)
    field_pairs = []
    def reg(key, comp):
        field_pairs.append((key, comp))
        return comp

    # Module options
    def make_module_option(block_id: str) -> str:
        b = get_block(block_id)
        title = b.title if b and getattr(b, "title", None) else ""
        return f"{block_id} — {title}" if title else block_id

    def label_to_id(label: str) -> str:
        if not label:
            return ""
        if " — " in label:
            return label.split(" — ", 1)[0].strip()
        # fallback: first token
        return label.split()[0].strip()

    def labels_to_ids(labels):
        return [label_to_id(x) for x in (labels or []) if label_to_id(x)]

    def ids_to_labels(ids, options):
        # ids are like P01; options are labels like "P01 — ..."
        ids = set(ids or [])
        out = []
        for opt in options:
            bid = label_to_id(opt)
            if bid in ids:
                out.append(opt)
        return out

    # Build categorized module lists
    all_ids = sorted([bid for bid in ALL_BLOCKS.keys() if isinstance(bid, str)])
    proc_ids = [bid for bid in all_ids if bid.startswith("P")]
    other_ids = [bid for bid in all_ids if bid.startswith(("BE", "C", "G", "ALT"))]

    proc_opts = [make_module_option(bid) for bid in proc_ids]
    other_opts = [make_module_option(bid) for bid in other_ids]

    # Force packages
    force_opts = [""] + sorted(BUNDLES.keys())

    with gr.Blocks(theme=theme, title=f"RHK Befundassistent {APP_VERSION}") as demo:
        gr.Markdown(f"<div class='rhk-title'>RHK Befundassistent</div><div class='rhk-sub'>Version {APP_VERSION} • interaktiv (Gradio)</div>")

        # --- Top action bar (buttons only) ---
        with gr.Row(elem_classes=["btnrow"]):
            example_btn_top = gr.Button("Beispiel laden")
            generate_btn_top = gr.Button("Befund erstellen/aktualisieren", variant="primary")
            clear_btn_top = gr.Button("Leeren")
            save_btn_top = gr.Button("Fall speichern")
            UploadButton = getattr(gr, "UploadButton", None)
            if UploadButton is not None:
                load_upload_top = UploadButton("Fall laden", file_types=[".json"], file_count="single")
            else:
                load_upload_top = None
                load_file_top = gr.File(label="Fall laden (.json)", file_types=[".json"])

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Tabs():
                    # ---------------- Tab 1: Klinik & Labor ----------------
                    with gr.Tab("Klinik & Labor"):
                        with gr.Accordion("Stammdaten", open=True):
                            with gr.Row():
                                last_name = reg("last_name", gr.Textbox(label="Name", placeholder="Muster"))
                                first_name = reg("first_name", gr.Textbox(label="Vorname", placeholder="Max"))
                            with gr.Row():
                                dob = reg("dob", gr.Textbox(label="Geburtsdatum", placeholder="TT.MM.JJJJ"))
                                sex = reg("sex", gr.Dropdown(label="Geschlecht", choices=["", "w", "m", "div"], value=""))
                            with gr.Row():
                                height_cm = reg("height_cm", gr.Number(label="Größe (cm)", value=None))
                                weight_kg = reg("weight_kg", gr.Number(label="Gewicht (kg)", value=None))
                                bsa = reg("bsa", gr.Number(label="KOF/BSA (m², optional)", value=None))
                            with gr.Row():
                                center = reg("center", gr.Textbox(label="Zentrum/Klinik (optional)", value=""))
                                date = reg("date", gr.Textbox(label="Datum (optional)", placeholder="TT.MM.JJJJ"))

                        with gr.Accordion("Anamnese / Indikation", open=True):
                            indication = reg("indication", gr.Textbox(label="Indikation (Freitext)", lines=2))
                            dyspnea = reg("dyspnea", gr.Textbox(label="Dyspnoe / Belastungsangaben (Freitext)", lines=2))
                            with gr.Row():
                                syncope = reg("syncope", gr.Checkbox(label="Synkopen", value=False))
                                angina = reg("angina", gr.Checkbox(label="Angina/Brustschmerz", value=False))
                                edema = reg("edema", gr.Checkbox(label="Beinödeme", value=False))
                            with gr.Row():
                                hemoptysis = reg("hemoptysis", gr.Checkbox(label="Hämoptysen", value=False))
                                dizziness = reg("dizziness", gr.Checkbox(label="Schwindel", value=False))
                            with gr.Row():
                                stairs = reg("stairs", gr.Number(label="Belastbarkeit: Stockwerke (optional)", value=None))
                                walk_distance_m = reg("walk_distance_m", gr.Number(label="Belastbarkeit: Gehstrecke (m, optional)", value=None))
                            with gr.Row():
                                covid_history = reg("covid_history", gr.Checkbox(label="COVID in der Vorgeschichte", value=False))
                                smoking = reg("smoking", gr.Dropdown(label="Rauchen", choices=["", "nie", "ex", "aktuell"], value=""))
                            comorbidities = reg("comorbidities", gr.Textbox(label="Komorbiditäten (Freitext)", lines=2))

                        with gr.Accordion("Labor", open=False):
                            with gr.Row():
                                hb = reg("hb", gr.Number(label="Hb (g/dl)", value=None))
                                creatinine = reg("creatinine", gr.Number(label="Kreatinin (mg/dl)", value=None))
                                crp = reg("crp", gr.Number(label="CRP (mg/l)", value=None))
                                tsh = reg("tsh", gr.Number(label="TSH (mU/l)", value=None))
                            with gr.Row():
                                troponin = reg("troponin", gr.Number(label="Troponin (ng/l)", value=None))
                                d_dimer = reg("d_dimer", gr.Number(label="D-Dimer (µg/l)", value=None))
                                autoimmune_screen = reg("autoimmune_screen", gr.Checkbox(label="Autoimmun-Screening durchgeführt", value=False))
                            with gr.Row():
                                bnp_kind = reg("bnp_kind", gr.Dropdown(label="Herzmarker", choices=["", "BNP", "NT-proBNP"], value=""))
                                bnp = reg("bnp", gr.Number(label="BNP (pg/ml)", value=None))
                                ntprobnp = reg("ntprobnp", gr.Number(label="NT-proBNP (pg/ml)", value=None))
                            entresto = reg("entresto", gr.Checkbox(label="Sacubitril/Valsartan (Entresto) wird eingenommen", value=False))

                        with gr.Accordion("Blutgase / LTOT", open=False):
                            with gr.Row():
                                pH = reg("pH", gr.Number(label="pH", value=None))
                                pCO2 = reg("pCO2", gr.Number(label="pCO2 (mmHg)", value=None))
                                pO2 = reg("pO2", gr.Number(label="pO2 (mmHg)", value=None))
                                sao2 = reg("sao2", gr.Number(label="sO2 (%)", value=None))
                            hco3 = reg("hco3", gr.Number(label="HCO3- (mmol/l)", value=None))
                            ltot_present = reg("ltot_present", gr.Checkbox(label="LTOT vorhanden", value=False))
                            ltot_flow_l_min = reg("ltot_flow_l_min", gr.Number(label="LTOT: Flow (L/min)", value=None, visible=False))

                        with gr.Accordion("Infektiologie / Immunologie", open=False):
                            virology_positive = reg("virology_positive", gr.Checkbox(label="Virologie positiv", value=False))
                            virology_details = reg("virology_details", gr.Textbox(label="Welche Befunde? (Freitext)", lines=2, visible=False))
                            immunology_positive = reg("immunology_positive", gr.Checkbox(label="Immunologie/Autoantikörper positiv", value=False))
                            immunology_details = reg("immunology_details", gr.Textbox(label="Welche Befunde? (Freitext)", lines=2, visible=False))

                        with gr.Accordion("Abdomen / Leber", open=False):
                            abdo_sono = reg("abdo_sono", gr.Checkbox(label="Abdomen-Sonographie durchgeführt", value=False))
                            abdo_findings = reg("abdo_findings", gr.Textbox(label="Besondere Befunde (Freitext)", lines=2, visible=False))
                            portal_hypertension = reg("portal_hypertension", gr.Dropdown(label="Portale Hypertension?", choices=["", "ja", "nein"], value="", visible=False))

                        with gr.Accordion("Therapie (PH-spezifisch)", open=False):
                            ph_meds_current = reg("ph_meds_current", gr.Checkbox(label="Aktuell PH-spezifische Medikation", value=False))
                            ph_meds_current_what = reg("ph_meds_current_what", gr.Textbox(label="Welche? (Freitext)", visible=False))
                            ph_meds_current_since = reg("ph_meds_current_since", gr.Textbox(label="Seit wann? (Freitext)", visible=False))
                            ph_meds_past = reg("ph_meds_past", gr.Checkbox(label="Zuvor PH-spezifische Medikation", value=False))
                            ph_meds_past_what = reg("ph_meds_past_what", gr.Textbox(label="Welche? (Freitext)", visible=False))
                            ph_meds_past_since = reg("ph_meds_past_since", gr.Textbox(label="Bis wann? / Verlauf (Freitext)", visible=False))

                        la_enlarged = reg("la_enlarged", gr.Checkbox(label="Linksatrium vergrößert (klinischer Kontext)", value=False))

                    # ---------------- Tab 2: RHK ----------------
                    with gr.Tab("RHK"):
                        with gr.Accordion("RHK in Ruhe", open=True):
                            with gr.Row():
                                rap = reg("rap", gr.Number(label="RAP (mmHg)", value=None))
                                pap_syst = reg("pap_syst", gr.Number(label="sPAP (mmHg)", value=None))
                                pap_diast = reg("pap_diast", gr.Number(label="dPAP (mmHg)", value=None))
                                mpap = reg("mpap", gr.Number(label="mPAP (mmHg)", value=None))
                                pawp = reg("pawp", gr.Number(label="PAWP (mmHg)", value=None))
                            with gr.Row():
                                co = reg("co", gr.Number(label="CO (L/min)", value=None))
                                ci = reg("ci", gr.Number(label="CI (L/min/m²)", value=None))
                                pvr = reg("pvr", gr.Number(label="PVR (WU)", value=None))
                                hr = reg("hr", gr.Number(label="HF (bpm)", value=None))
                            with gr.Row():
                                svo2 = reg("svo2", gr.Number(label="SvO2 (%)", value=None))
                                art_sat = reg("art_sat", gr.Number(label="SaO2 (%)", value=None))
                                svr = reg("svr", gr.Number(label="SVR (dyn·s·cm⁻5, optional)", value=None))
                            rhk_notes = reg("rhk_notes", gr.Textbox(label="RHK Notizen (optional)", lines=2))

                        with gr.Accordion("Belastung (optional)", open=False):
                            exercise_done = reg("exercise_done", gr.Checkbox(label="Belastung durchgeführt", value=False))
                            # IMPORTANT: label changed to avoid misunderstanding ("PH criteria") and to reflect interactivity
                            exercise_ph = reg("exercise_ph", gr.Checkbox(label="Belastungsreaktion auffällig / relevant", value=False))
                            with gr.Row():
                                ex_rap = reg("ex_rap", gr.Number(label="RAP peak (mmHg)", value=None))
                                ex_spap = reg("ex_spap", gr.Number(label="sPAP peak (mmHg)", value=None))
                                ex_dpap = reg("ex_dpap", gr.Number(label="dPAP peak (mmHg)", value=None))
                                ex_mpap = reg("ex_mpap", gr.Number(label="mPAP peak (mmHg)", value=None))
                                ex_pawp = reg("ex_pawp", gr.Number(label="PAWP peak (mmHg)", value=None))
                            with gr.Row():
                                ex_co = reg("ex_co", gr.Number(label="CO peak (L/min)", value=None))
                                ex_ci = reg("ex_ci", gr.Number(label="CI peak (L/min/m²)", value=None))
                                ex_pvr = reg("ex_pvr", gr.Number(label="PVR peak (WU)", value=None))
                                ex_hr = reg("ex_hr", gr.Number(label="HF peak (bpm)", value=None))
                            with gr.Row():
                                mpap_co_slope = reg("mpap_co_slope", gr.Number(label="mPAP/CO-Slope (mmHg/L/min, optional)", value=None))
                                pawp_co_slope = reg("pawp_co_slope", gr.Number(label="PAWP/CO-Slope (mmHg/L/min, optional)", value=None))
                            exercise_notes = reg("exercise_notes", gr.Textbox(label="Belastungsnotizen (optional)", lines=2))

                        with gr.Accordion("Manöver / Zusatz", open=False):
                            volume_challenge_done = reg("volume_challenge_done", gr.Checkbox(label="Volumenbelastung durchgeführt", value=False))
                            volume_positive = reg("volume_positive", gr.Checkbox(label="Volumenbelastung positiv", value=False))
                            volume_pawp = reg("volume_pawp", gr.Number(label="PAWP nach Volumen (mmHg, optional)", value=None))
                            volume_notes = reg("volume_notes", gr.Textbox(label="Volumen-Notizen (optional)", lines=2))

                            vasoreactivity_done = reg("vasoreactivity_done", gr.Checkbox(label="Vasoreagibilitätstest durchgeführt", value=False))
                            vasoreactivity_positive = reg("vasoreactivity_positive", gr.Checkbox(label="Vasoreagibilität positiv", value=False))
                            vasoreactivity_agent = reg("vasoreactivity_agent", gr.Textbox(label="Testsubstanz (optional)", value=""))
                            vasoreactivity_notes = reg("vasoreactivity_notes", gr.Textbox(label="Vasoreagibilität Notizen (optional)", lines=2))

                            stepup_done = reg("stepup_done", gr.Checkbox(label="O2-Step-up Messung durchgeführt", value=False))
                            stepup_positive = reg("stepup_positive", gr.Checkbox(label="O2-Step-up positiv", value=False))
                            stepup_site = reg("stepup_site", gr.Textbox(label="Step-up Lokalisation (optional)", value=""))
                            stepup_notes = reg("stepup_notes", gr.Textbox(label="Step-up Notizen (optional)", lines=2))

                        with gr.Accordion("Vorheriger RHK (optional)", open=False):
                            with gr.Row():
                                prev_date = reg("prev_date", gr.Textbox(label="Datum (z.B. 03/2021)", value=""))
                                prev_mpap = reg("prev_mpap", gr.Number(label="mPAP (mmHg)", value=None))
                                prev_pawp = reg("prev_pawp", gr.Number(label="PAWP (mmHg)", value=None))
                                prev_ci = reg("prev_ci", gr.Number(label="CI (L/min/m²)", value=None))
                                prev_pvr = reg("prev_pvr", gr.Number(label="PVR (WU)", value=None))
                            prev_notes = reg("prev_notes", gr.Textbox(label="Kommentar (optional)", lines=2))

                    # ---------------- Tab 3: Lufu ----------------
                    with gr.Tab("Lufu"):
                        with gr.Accordion("Lungenfunktion (Werte)", open=True):
                            with gr.Row():
                                fvc = reg("fvc", gr.Number(label="FVC (l)", value=None))
                                fvc_pct = reg("fvc_pct", gr.Number(label="FVC (%)", value=None))
                                fev1 = reg("fev1", gr.Number(label="FEV1 (l)", value=None))
                                fev1_pct = reg("fev1_pct", gr.Number(label="FEV1 (%)", value=None))
                            with gr.Row():
                                fev1_fvc = reg("fev1_fvc", gr.Number(label="FEV1/FVC", value=None))
                                tlc = reg("tlc", gr.Number(label="TLC (l)", value=None))
                                tlc_pct = reg("tlc_pct", gr.Number(label="TLC (%)", value=None))
                            with gr.Row():
                                dlco = reg("dlco", gr.Number(label="DLCO (ml/min/mmHg)", value=None))
                                dlco_pct = reg("dlco_pct", gr.Number(label="DLCO (%)", value=None))
                        with gr.Accordion("Lufu-Zusammenfassung (Freitext)", open=False):
                            lufu_summary = reg("lufu_summary", gr.Textbox(label="Zusammenfassung", lines=4))

                    # ---------------- Tab 4: Bildgebung & Echo/CMR ----------------
                    with gr.Tab("Bildgebung & Echo/CMR"):
                        with gr.Accordion("CT / Thoraxbildgebung", open=True):
                            ct_angio_done = reg("ct_angio_done", gr.Checkbox(label="CT-Angio vorhanden", value=False))
                            with gr.Row():
                                ct_lae = reg("ct_lae", gr.Checkbox(label="Lungenarterienembolie (CT)", value=False))
                                ct_embolie = reg("ct_embolie", gr.Checkbox(label="Thromboembolische Residuen / chronische Emboliezeichen", value=False))
                                ct_mosaic = reg("ct_mosaic", gr.Checkbox(label="Mosaikperfusion", value=False))
                            with gr.Row():
                                ct_ild = reg("ct_ild", gr.Checkbox(label="ILD / Fibrosezeichen", value=False))
                                ct_emphysema = reg("ct_emphysema", gr.Checkbox(label="Emphysem", value=False))
                                ct_corcal = reg("ct_corcal", gr.Checkbox(label="Koronarkalk", value=False))

                            # ILD details (conditional)
                            ct_ild_type = reg("ct_ild_type", gr.Textbox(label="ILD: Typ (Freitext)", visible=False))
                            with gr.Row():
                                ct_ild_extent = reg("ct_ild_extent", gr.Dropdown(label="ILD: Ausmaß", choices=["", "mild", "moderat", "ausgeprägt"], value="", visible=False))
                                ct_ild_histology = reg("ct_ild_histology", gr.Checkbox(label="Histologisch gesichert", value=False, visible=False))
                                ct_ild_fibrosis_clinic = reg("ct_ild_fibrosis_clinic", gr.Checkbox(label="Fibrose-/ILD-Sprechstunde", value=False, visible=False))

                        with gr.Accordion("Echo", open=False):
                            echo_done = reg("echo_done", gr.Checkbox(label="Echo vorhanden", value=False))
                            with gr.Row():
                                echo_rv_dilation = reg("echo_rv_dilation", gr.Dropdown(label="RV-Dilatation", choices=["", "nein", "leicht", "moderat", "schwer"], value=""))
                                echo_rv_function = reg("echo_rv_function", gr.Dropdown(label="RV-Funktion", choices=["", "normal", "leicht reduziert", "moderat reduziert", "schwer reduziert"], value=""))
                            with gr.Row():
                                echo_tapse = reg("echo_tapse", gr.Number(label="TAPSE (mm)", value=None))
                                echo_sprime = reg("echo_sprime", gr.Number(label="S' (cm/s)", value=None))
                                echo_raa = reg("echo_raa", gr.Number(label="RAA (cm²)", value=None))
                                echo_raai = reg("echo_raai", gr.Number(label="RAAI (cm²/m²)", value=None))
                            with gr.Row():
                                echo_tr_vmax = reg("echo_tr_vmax", gr.Number(label="TR Vmax (m/s)", value=None))
                                echo_rvsp = reg("echo_rvsp", gr.Number(label="geschätzter RVSP/PASP (mmHg)", value=None))
                                echo_la_enlarged = reg("echo_la_enlarged", gr.Checkbox(label="Linksatrium vergrößert", value=False))
                            echo_pericardial_effusion = reg("echo_pericardial_effusion", gr.Dropdown(label="Perikarderguss", choices=["", "nein", "klein", "moderat", "groß"], value=""))

                        with gr.Accordion("Kardio-MRT / CMR", open=False):
                            cmr_done = reg("cmr_done", gr.Checkbox(label="CMR vorhanden", value=False))
                            with gr.Row():
                                cmr_rvef = reg("cmr_rvef", gr.Number(label="RVEF (%)", value=None))
                                cmr_rv_edvi = reg("cmr_rv_edvi", gr.Number(label="RV EDVi (ml/m²)", value=None))
                                cmr_lv_ef = reg("cmr_lv_ef", gr.Number(label="LVEF (%)", value=None))
                            cmr_lge = reg("cmr_lge", gr.Checkbox(label="LGE vorhanden", value=False))

                    # ---------------- Tab 5: Scores ----------------
                    with gr.Tab("Scores"):
                        with gr.Accordion("Risikostratifizierung (PAH-orientiert)", open=True):
                            with gr.Row():
                                who_fc = reg("who_fc", gr.Dropdown(label="WHO-Funktionsklasse", choices=["", "I", "II", "III", "IV"], value=""))
                                six_mwd = reg("six_mwd", gr.Number(label="6MWD (m)", value=None))
                            with gr.Row():
                                reveal_functional_class = reg("reveal_functional_class", gr.Dropdown(label="REVEAL: Functional Class", choices=["", "I", "II", "III", "IV"], value=""))
                                reveal_renal = reg("reveal_renal", gr.Checkbox(label="REVEAL: Niereninsuffizienz", value=False))
                                reveal_hosp6 = reg("reveal_hosp6", gr.Checkbox(label="REVEAL: Hospitalisation <6 Monate", value=False))
                            with gr.Row():
                                reveal_hr = reg("reveal_hr", gr.Number(label="REVEAL: Herzfrequenz (bpm)", value=None))
                                reveal_sysbp = reg("reveal_sysbp", gr.Number(label="REVEAL: syst. RR (mmHg)", value=None))
                            with gr.Row():
                                reveal_bnp = reg("reveal_bnp", gr.Number(label="REVEAL: BNP (pg/ml)", value=None))
                                reveal_ntprobnp = reg("reveal_ntprobnp", gr.Number(label="REVEAL: NT-proBNP (pg/ml)", value=None))
                            with gr.Row():
                                reveal_dlco_pct = reg("reveal_dlco_pct", gr.Number(label="REVEAL: DLCO (%)", value=None))
                                reveal_pericardial_effusion = reg("reveal_pericardial_effusion", gr.Checkbox(label="REVEAL: Perikarderguss", value=False))

                        with gr.Accordion("HFpEF-Score (H2FPEF)", open=False):
                            gr.Markdown("Referenz/Erklärung: siehe AHA-Tool / Reddy et al. (2018).")
                            with gr.Row():
                                hfpef_age = reg("hfpef_age", gr.Number(label="Alter (Jahre, optional – sonst aus Stammdaten)", value=None))
                                hfpef_bmi = reg("hfpef_bmi", gr.Number(label="BMI (optional – sonst aus Größe/Gewicht)", value=None))
                            with gr.Row():
                                hfpef_af = reg("hfpef_af", gr.Checkbox(label="Vorhofflimmern", value=False))
                                hfpef_htn_meds = reg("hfpef_htn_meds", gr.Number(label="Antihypertensiva (Anzahl)", value=None))
                            with gr.Row():
                                hfpef_e_eprime = reg("hfpef_e_eprime", gr.Number(label="E/e' (optional)", value=None))
                                hfpef_pasp = reg("hfpef_pasp", gr.Number(label="PASP (Echo, mmHg – optional)", value=None))

                    # ---------------- Tab 6: Module ----------------
                    with gr.Tab("Module / Procedere"):
                        force_package = reg("force_package", gr.Dropdown(label="Befund-Paket erzwingen (optional)", choices=force_opts, value=""))
                        with gr.Row():
                            modules_proc = reg("modules_proc", gr.CheckboxGroup(label="Procedere (P-Module)", choices=proc_opts, value=[]))
                            modules_other = reg("modules_other", gr.CheckboxGroup(label="Zusatzmodule (BE/C/G/ALT)", choices=other_opts, value=[]))
                        recommendation_free_text = reg("recommendation_free_text", gr.Textbox(label="Empfehlung – Freitext (optional)", lines=4))

                        # Store combined module list (hidden) to keep old generator API compatible
                        modules = reg("modules", gr.State(value=[]))

            with gr.Column(scale=5):
                sticky_out = gr.HTML(label="Dashboard")
                risk_out = gr.HTML(label="Risiko/Score Übersicht")
                with gr.Tabs():
                    with gr.Tab("Arztbefund"):
                        main_out = gr.Markdown()
                    with gr.Tab("Patientenbericht"):
                        patient_out = gr.Markdown()
                    with gr.Tab("Intern"):
                        internal_out = gr.Markdown()

        # --- Bottom action bar (same as top) ---
        with gr.Row(elem_classes=["btnrow"]):
            example_btn_bot = gr.Button("Beispiel laden")
            generate_btn_bot = gr.Button("Befund erstellen/aktualisieren", variant="primary")
            clear_btn_bot = gr.Button("Leeren")
            save_btn_bot = gr.Button("Fall speichern")
            if UploadButton is not None:
                load_upload_bot = UploadButton("Fall laden", file_types=[".json"], file_count="single")
            else:
                load_upload_bot = None

        save_file = gr.File(label="Download", visible=False)

        # --- Conditional visibility wiring ---
        def _toggle_visible(flag: bool, *comps):
            return [gr.update(visible=bool(flag)) for _ in comps]

        ltot_present.change(lambda x: gr.update(visible=bool(x)), inputs=ltot_present, outputs=ltot_flow_l_min)
        virology_positive.change(lambda x: gr.update(visible=bool(x)), inputs=virology_positive, outputs=virology_details)
        immunology_positive.change(lambda x: gr.update(visible=bool(x)), inputs=immunology_positive, outputs=immunology_details)
        abdo_sono.change(lambda x: [gr.update(visible=bool(x)), gr.update(visible=bool(x))], inputs=abdo_sono, outputs=[abdo_findings, portal_hypertension])
        ph_meds_current.change(lambda x: [gr.update(visible=bool(x)), gr.update(visible=bool(x))], inputs=ph_meds_current, outputs=[ph_meds_current_what, ph_meds_current_since])
        ph_meds_past.change(lambda x: [gr.update(visible=bool(x)), gr.update(visible=bool(x))], inputs=ph_meds_past, outputs=[ph_meds_past_what, ph_meds_past_since])
        ct_ild.change(lambda x: [gr.update(visible=bool(x))]*4, inputs=ct_ild, outputs=[ct_ild_type, ct_ild_extent, ct_ild_histology, ct_ild_fibrosis_clinic])

        # --- Mapping helpers ---
        field_keys = [k for k,_ in field_pairs]
        field_comps = [c for _,c in field_pairs]

        bool_keys = {
            "syncope","angina","edema","covid_history","hemoptysis","dizziness",
            "autoimmune_screen","entresto","ltot_present","virology_positive","immunology_positive",
            "abdo_sono","ph_meds_current","ph_meds_past","la_enlarged",
            "exercise_done","exercise_ph","volume_challenge_done","volume_positive",
            "vasoreactivity_done","vasoreactivity_positive","stepup_done","stepup_positive",
            "ct_angio_done","ct_lae","ct_ild","ct_ild_histology","ct_ild_fibrosis_clinic",
            "ct_emphysema","ct_embolie","ct_mosaic","ct_corcal",
            "echo_done","echo_la_enlarged","cmr_done","cmr_lge",
            "reveal_renal","reveal_hosp6","reveal_pericardial_effusion",
            "hfpef_af"
        }

        def _empty_value(key: str):
            if key in bool_keys:
                return False
            # Dropdown/Textbox default
            if key in ("sex","smoking","bnp_kind","echo_rv_dilation","echo_rv_function","echo_pericardial_effusion","who_fc","reveal_functional_class","force_package","ct_ild_extent","portal_hypertension"):
                return ""
            if key in ("modules_proc","modules_other"):
                return []
            if key in ("modules",):
                return []
            # Numbers
            return None if key not in ("last_name","first_name","dob","center","date","indication","dyspnea","comorbidities","lufu_summary","rhk_notes","exercise_notes","volume_notes","vasoreactivity_agent","vasoreactivity_notes","stepup_site","stepup_notes","prev_date","prev_notes","ct_ild_type","virology_details","immunology_details","abdo_findings","ph_meds_current_what","ph_meds_current_since","ph_meds_past_what","ph_meds_past_since","recommendation_free_text") else ""

        def _vals_to_raw(*vals):
            raw = dict(zip(field_keys, vals))
            # portal_hypertension normalization (dropdown)
            if raw.get("portal_hypertension") == "ja":
                raw["portal_hypertension"] = True
            elif raw.get("portal_hypertension") == "nein":
                raw["portal_hypertension"] = False
            else:
                raw["portal_hypertension"] = ""
            # Combine module groups into raw["modules"] as IDs
            ids = labels_to_ids(raw.get("modules_proc") or []) + labels_to_ids(raw.get("modules_other") or [])
            raw["modules"] = ids
            return raw

        def _load_example_values() -> dict:
            # Start from v14 example and extend with new keys
            example = {
                "last_name": "Muster",
                "first_name": "Max",
                "dob": "01.01.1970",
                "sex": "m",
                "height_cm": 178,
                "weight_kg": 82,
                "bsa": None,
                "center": "PH-Zentrum",
                "date": "26.12.2025",
                "indication": "Abklärung Belastungsdyspnoe, Verdacht auf PH.",
                "dyspnea": "NYHA III, Belastungsdyspnoe seit 12 Monaten.",
                "syncope": False,
                "angina": False,
                "edema": True,
                "hemoptysis": False,
                "dizziness": False,
                "stairs": 1,
                "walk_distance_m": 150,
                "covid_history": True,
                "smoking": "ex",
                "comorbidities": "COPD, arterielle Hypertonie.",
                "hb": 13.8,
                "creatinine": 1.0,
                "bnp_kind": "NT-proBNP",
                "bnp": None,
                "ntprobnp": 820,
                "entresto": False,
                "crp": 6,
                "tsh": 1.8,
                "troponin": None,
                "d_dimer": None,
                "autoimmune_screen": True,
                "pH": 7.43,
                "pCO2": 36,
                "pO2": 62,
                "hco3": 24,
                "sao2": 91,
                "ltot_present": False,
                "ltot_flow_l_min": None,
                # RHK rest
                "rap": 8,
                "pap_syst": 62,
                "pap_diast": 28,
                "mpap": 39,
                "pawp": 11,
                "co": 4.2,
                "ci": None,
                "pvr": 6.7,
                "svr": None,
                "hr": 88,
                "svo2": 62,
                "art_sat": 92,
                "rhk_notes": "",
                # exercise
                "exercise_done": True,
                "exercise_ph": True,
                "ex_rap": 12,
                "ex_spap": 92,
                "ex_dpap": 40,
                "ex_mpap": 57,
                "ex_pawp": 18,
                "ex_co": 6.0,
                "ex_ci": None,
                "ex_pvr": None,
                "ex_hr": 118,
                "mpap_co_slope": None,
                "pawp_co_slope": None,
                "exercise_notes": "",
                # maneuvers
                "volume_challenge_done": False,
                "volume_positive": False,
                "volume_pawp": None,
                "volume_notes": "",
                "vasoreactivity_done": False,
                "vasoreactivity_positive": False,
                "vasoreactivity_agent": "",
                "vasoreactivity_notes": "",
                "stepup_done": False,
                "stepup_positive": False,
                "stepup_site": "",
                "stepup_notes": "",
                # prev RHK
                "prev_date": "",
                "prev_mpap": None,
                "prev_pawp": None,
                "prev_ci": None,
                "prev_pvr": None,
                "prev_notes": "",
                # lufu
                "fvc": 3.1,
                "fvc_pct": 72,
                "fev1": 1.9,
                "fev1_pct": 58,
                "fev1_fvc": 0.61,
                "dlco": 12.0,
                "dlco_pct": 41,
                "tlc": 5.7,
                "tlc_pct": 86,
                "lufu_summary": "Obstruktiv, DLCO deutlich reduziert.",
                # imaging + echo/cmr
                "ct_angio_done": True,
                "ct_lae": False,
                "ct_embolie": True,
                "ct_mosaic": True,
                "ct_ild": False,
                "ct_ild_type": "",
                "ct_ild_extent": "",
                "ct_ild_histology": False,
                "ct_ild_fibrosis_clinic": False,
                "ct_emphysema": True,
                "ct_corcal": True,
                "echo_done": True,
                "echo_rv_dilation": "moderat",
                "echo_rv_function": "moderat reduziert",
                "echo_tapse": 15,
                "echo_sprime": 10,
                "echo_raa": 22,
                "echo_raai": 12,
                "echo_tr_vmax": 3.9,
                "echo_rvsp": 70,
                "echo_la_enlarged": False,
                "echo_pericardial_effusion": "nein",
                "cmr_done": False,
                "cmr_rvef": None,
                "cmr_rv_edvi": None,
                "cmr_lv_ef": None,
                "cmr_lge": False,
                # infection/immunology
                "virology_positive": False,
                "virology_details": "",
                "immunology_positive": False,
                "immunology_details": "",
                # abdomen
                "abdo_sono": False,
                "abdo_findings": "",
                "portal_hypertension": "",
                # meds
                "ph_meds_current": False,
                "ph_meds_current_what": "",
                "ph_meds_current_since": "",
                "ph_meds_past": False,
                "ph_meds_past_what": "",
                "ph_meds_past_since": "",
                "la_enlarged": False,
                # scores
                "who_fc": "III",
                "six_mwd": 280,
                "reveal_renal": False,
                "reveal_hosp6": False,
                "reveal_hr": 88,
                "reveal_sysbp": 118,
                "reveal_bnp": None,
                "reveal_ntprobnp": 820,
                "reveal_dlco_pct": 41,
                "reveal_pericardial_effusion": False,
                "reveal_functional_class": "III",
                "hfpef_age": None,
                "hfpef_bmi": None,
                "hfpef_af": False,
                "hfpef_htn_meds": 2,
                "hfpef_e_eprime": None,
                "hfpef_pasp": None,
                # modules/procedere
                "force_package": "",
                "modules_proc": [],
                "modules_other": [],
                "recommendation_free_text": "",
                "modules": [],
            }
            return example

        def _example_click():
            ex = _load_example_values()
            return [ex.get(k, _empty_value(k)) for k in field_keys]

        # --- Save/load helpers ---
        def _save_case(*vals):
            raw = _vals_to_raw(*vals)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "app_version": APP_VERSION,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "data": raw,
            }
            fn = f"rhk_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = Path(tempfile.gettempdir()) / fn
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return gr.update(value=str(path), visible=True)

        def _load_case_from_dict(raw: dict):
            # Backward compatible: accept either full wrapper or raw dict
            if "data" in raw and isinstance(raw.get("data"), dict):
                raw = raw["data"]

            # Normalize portal_hypertension dropdown values
            if raw.get("portal_hypertension") is True:
                raw["portal_hypertension"] = "ja"
            elif raw.get("portal_hypertension") is False:
                raw["portal_hypertension"] = "nein"
            elif raw.get("portal_hypertension") in (None, ""):
                raw["portal_hypertension"] = ""

            # Convert stored module IDs to labels for each group
            mod_ids = raw.get("modules") or []
            raw["modules_proc"] = ids_to_labels([m for m in mod_ids if str(m).startswith("P")], proc_opts)
            raw["modules_other"] = ids_to_labels([m for m in mod_ids if str(m).startswith(("BE","C","G","ALT"))], other_opts)

            return [raw.get(k, _empty_value(k)) for k in field_keys]

        def _load_case_file(file_obj):
            if file_obj is None:
                return [gr.update() for _ in field_keys]
            # file_obj may be a path or an UploadedFile-like
            try:
                path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                raw = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                try:
                    raw = json.loads(file_obj.decode("utf-8"))
                except Exception:
                    raw = {}
            return _load_case_from_dict(raw)

        def _generate(*vals):
            raw = _vals_to_raw(*vals)
            data = build_data_from_ui(raw)
            _preprocess_exercise_for_interactivity(data)
            main, patient_txt, internal, risk_html, sticky_html = generator.generate_all(data)

            # Append extra interactive details (so every input has consequences)
            main += _interactive_details_doctor(data)
            patient_txt += _interactive_details_patient(data)

            return main, patient_txt, internal, risk_html, sticky_html

        def _clear_all():
            return [_empty_value(k) for k in field_keys]

        # --- Button wiring ---
        example_btn_top.click(_example_click, inputs=[], outputs=field_comps)
        example_btn_bot.click(_example_click, inputs=[], outputs=field_comps)

        generate_btn_top.click(_generate, inputs=field_comps, outputs=[main_out, patient_out, internal_out, risk_out, sticky_out])
        generate_btn_bot.click(_generate, inputs=field_comps, outputs=[main_out, patient_out, internal_out, risk_out, sticky_out])

        clear_btn_top.click(_clear_all, inputs=[], outputs=field_comps)
        clear_btn_bot.click(_clear_all, inputs=[], outputs=field_comps)

        save_btn_top.click(_save_case, inputs=field_comps, outputs=save_file)
        save_btn_bot.click(_save_case, inputs=field_comps, outputs=save_file)

        if UploadButton is not None and load_upload_top is not None:
            load_upload_top.upload(_load_case_file, inputs=load_upload_top, outputs=field_comps)
            load_upload_bot.upload(_load_case_file, inputs=load_upload_bot, outputs=field_comps)
        else:
            # Fallback: file input + manual button
            def _load_case_btn(file_obj):
                return _load_case_file(file_obj)
            load_btn = gr.Button("Fall laden (Datei auswählen oben)")
            load_btn.click(_load_case_btn, inputs=load_file_top, outputs=field_comps)

    # Return css separately to avoid Blocks(css=...) deprecation
    return demo, theme, css

def build_interface_auto():
    # Keep API used by main(): return app, theme, css
    try:
        app, theme, css = build_blocks_app()
        return app, theme, css
    except Exception:
        # Fallback to old auto-builder if something goes wrong
        try:
            app = build_interface_app()
            return app, None, None
        except Exception as e:
            raise e

def main():
    app, theme, css = build_interface_auto()
    port = int(os.getenv("PORT", "7860"))
    # Render requires 0.0.0.0
    launch_kwargs = dict(server_name="0.0.0.0", server_port=port, show_api=False)
    try:
        app.queue(default_concurrency_limit=32).launch(**launch_kwargs, css=css)
    except TypeError:
        # Older Gradio versions might not accept css in launch()
        app.queue(default_concurrency_limit=32).launch(**launch_kwargs)


# --- v16 bugfix: patient text rendering must not depend on undefined TEXTDB ---
def safe_render_patient(block_id: str, ctx: Dict[str, Any]) -> str:
    """
    Render a patient-friendly text snippet for a given module/block.
    Prefers dedicated patient blocks (rhk_textdb_patient*). Falls back to simplifying
    the physician template.
    """
    try:
        pblock = get_patient_block(block_id)
        if pblock is not None:
            return pblock.render(ctx)

        b = get_block(block_id)
        if b is None:
            return ""

        # Render template safely, then simplify language and strip jargon/numbers
        rendered = (b.template or "").format_map(SafeDict(ctx))
        return patient_simplify_text(rendered)
    except Exception:
        # last resort
        try:
            b = get_block(block_id)
            return patient_simplify_text(getattr(b, "template", "") or "")
        except Exception:
            return ""

if __name__ == "__main__":
    main()
