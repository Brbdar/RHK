#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_core.py
Core-Logik (ohne GUI) für den RHK-Befundassistenten.

Ziele:
- robuste Ableitung fehlender hämodynamischer Parameter (z.B. mPAP aus sPAP/dPAP, CI aus CO & BSA, PVR aus mPAP/PAWP/CO)
- guideline-nahe Klassifikation der PH (Ruhe + optional Belastung/Volumen) und konsistente "Konsequenzen" aus allen Eingaben
- zentrale Rule-Engine: Auswahl Hauptpaket (Kxx), Zusatz-Hinweise, Workup- und Therapie-Add-ons, automatische Modulvorschläge
- Generierung von Arztbefund + Patiententext + internem Debug-Protokoll + Risiko-Dashboard HTML
- PDF-Export (reportlab, optional)

Hinweis:
Dieses Tool unterstützt Ärzt:innen beim Formulieren/Strukturieren. Es ersetzt nicht die klinische Beurteilung.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Import Textdatenbanken
# -----------------------------
TEXTDB_AVAILABLE = False
PATIENT_DB_AVAILABLE = False

try:
    import rhk_textdb as _textdb  # type: ignore

    TEXTDB_AVAILABLE = True
except Exception:
    _textdb = None  # type: ignore

try:
    import rhk_textdb_patient_v7 as _pdb  # type: ignore

    PATIENT_DB_AVAILABLE = True
except Exception:
    _pdb = None  # type: ignore


# -----------------------------
# Version
# -----------------------------
APP_VERSION = "v17.0.0-master"


# -----------------------------
# Utilities
# -----------------------------
class SafeDict(dict):
    """format_map helper: fehlende Keys -> leere Zeichenkette."""
    def __missing__(self, key: str) -> str:
        return ""



def dedup_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def normalize_module_id(x: Any) -> Optional[str]:
    """
    Accepts strings like 'P02', 'P02 – Diuretika', etc. Returns 'P02' or None.
    """
    if x is None:
        return None
    s = _clean_str(x)
    if not s:
        return None
    m = re.match(r"^(P\d{2})", s.strip())
    return m.group(1) if m else None

def _clean_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if math.isnan(float(x)):
            return None
        return float(x)
    s = _clean_str(x)
    if not s:
        return None
    # Komma-Dezimal, Tausenderpunkte
    s = s.replace(" ", "")
    s = s.replace(".", "").replace(",", ".") if re.search(r"\d+,\d+", s) else s.replace(",", ".")
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    v = to_float(x)
    if v is None:
        return None
    try:
        return int(round(v))
    except Exception:
        return None


def fmt_num(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return "—"
    try:
        return f"{x:.{nd}f}".replace(".", ",")
    except Exception:
        return "—"


def fmt_mmHg(x: Optional[float]) -> str:
    return f"{fmt_num(x, 0)} mmHg" if x is not None else "—"


def fmt_l_min_m2(x: Optional[float]) -> str:
    return f"{fmt_num(x, 2)} L/min/m²" if x is not None else "—"


def fmt_wu(x: Optional[float]) -> str:
    return f"{fmt_num(x, 2)} WU" if x is not None else "—"


def parse_date(s: Any) -> Optional[_dt.date]:
    if not s:
        return None
    if isinstance(s, _dt.date) and not isinstance(s, _dt.datetime):
        return s
    if isinstance(s, _dt.datetime):
        return s.date()
    ss = _clean_str(s)
    # allow DD.MM.YYYY, YYYY-MM-DD, MM/YYYY
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d.%m.%y"):
        try:
            return _dt.datetime.strptime(ss, fmt).date()
        except Exception:
            pass
    # MM/YYYY -> take 01.MM.YYYY
    m = re.match(r"^\s*(\d{1,2})\s*/\s*(\d{2,4})\s*$", ss)
    if m:
        mm = int(m.group(1))
        yy = int(m.group(2))
        if yy < 100:
            yy += 2000
        try:
            return _dt.date(yy, mm, 1)
        except Exception:
            return None
    return None


def calc_age(dob: Optional[_dt.date], ref: Optional[_dt.date] = None) -> Optional[int]:
    if not dob:
        return None
    ref = ref or _dt.date.today()
    try:
        years = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
        return max(0, years)
    except Exception:
        return None


def calc_bsa_m2(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    """Mosteller: sqrt(height(cm)*weight(kg)/3600)."""
    if not height_cm or not weight_kg:
        return None
    if height_cm <= 0 or weight_kg <= 0:
        return None
    return math.sqrt((height_cm * weight_kg) / 3600.0)


def calc_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if not weight_kg or not height_cm:
        return None
    h_m = height_cm / 100.0
    if h_m <= 0:
        return None
    return weight_kg / (h_m * h_m)


# -----------------------------
# H2FPEF (continuous model, CirculationAHA.118.034646)
# user-provided coefficients
# -----------------------------
@dataclass
class H2FPEFResult:
    probability_pct: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    category: str = "unknown"  # unlikely / possible / likely / unknown
    inputs_used: Dict[str, Optional[float]] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def calc_h2fpef_probability(age: Optional[float], bmi: Optional[float], ee: Optional[float], pasp: Optional[float], af_yes: Optional[bool]) -> H2FPEFResult:
    # Cap BMI per user instruction
    bmi_cap = None if bmi is None else min(bmi, 50.0)
    af = 1.0 if af_yes else 0.0
    res = H2FPEFResult(inputs_used={"age": age, "bmi": bmi_cap, "e_eprime": ee, "pasp": pasp, "af": 1.0 if af_yes else 0.0})

    if age is None or bmi_cap is None or ee is None or pasp is None:
        res.category = "unknown"
        return res

    y = -9.1917 + 0.0451 * age + 0.1307 * bmi_cap + 0.0859 * ee + 0.0520 * pasp + 1.6997 * af
    z = math.exp(y)
    p = (z / (1.0 + z)) * 100.0

    res.y = y
    res.z = z
    res.probability_pct = p

    # pragmatische Kategorien (klinisch interpretieren!)
    if p < 25:
        res.category = "unlikely"
    elif p < 70:
        res.category = "possible"
    else:
        res.category = "likely"
    return res


# -----------------------------
# PH classification (ESC/ERS 2022 definitions as in DEFAULT_RULES)
# -----------------------------
def classify_ph_rest(mpap: Optional[float], pawp: Optional[float], pvr: Optional[float]) -> Dict[str, Any]:
    """Return dict with: ph_present, ph_type (none/borderline/precap/ipcph/cpcph/unknown)."""
    # Defaults
    mPAP_PH = 20.0
    PAWP_POST = 15.0
    PVR_PRE = 2.0
    if TEXTDB_AVAILABLE and hasattr(_textdb, "DEFAULT_RULES"):
        try:
            rest = _textdb.DEFAULT_RULES.get("rest", {})
            mPAP_PH = float(rest.get("mPAP_ph_mmHg", mPAP_PH))
            PAWP_POST = float(rest.get("PAWP_postcap_mmHg", PAWP_POST))
            PVR_PRE = float(rest.get("PVR_precap_WU", PVR_PRE))
        except Exception:
            pass

    out: Dict[str, Any] = {"ph_present": None, "ph_type": "unknown"}

    if mpap is None:
        return out

    out["ph_present"] = mpap > mPAP_PH

    if mpap <= mPAP_PH:
        out["ph_type"] = "none"
        return out

    # mpap elevated
    if pawp is None or pvr is None:
        out["ph_type"] = "borderline"
        return out

    if pawp <= PAWP_POST and pvr > PVR_PRE:
        out["ph_type"] = "precap"
    elif pawp > PAWP_POST and pvr <= PVR_PRE:
        out["ph_type"] = "ipcph"
    elif pawp > PAWP_POST and pvr > PVR_PRE:
        out["ph_type"] = "cpcph"
    else:
        out["ph_type"] = "borderline"
    return out


def classify_exercise_pattern(mpap_co_slope: Optional[float], pawp_co_slope: Optional[float]) -> str:
    """
    Returns: 'normal' / 'left' / 'pulmvasc' / 'mixed' / 'unknown'
    Based on common cutoffs: mPAP/CO slope >3 = abnormal; PAWP/CO slope >2 = left component.
    """
    if mpap_co_slope is None or pawp_co_slope is None:
        return "unknown"
    mpap_abn = mpap_co_slope > 3.0
    pawp_abn = pawp_co_slope > 2.0
    if not mpap_abn and not pawp_abn:
        return "normal"
    if mpap_abn and pawp_abn:
        return "mixed"  # often left-limited + pulmonary vascular contribution
    if mpap_abn and not pawp_abn:
        return "pulmvasc"
    if pawp_abn and not mpap_abn:
        return "left"
    return "unknown"


# -----------------------------
# Derived values + "facts"
# -----------------------------
@dataclass
class Derived:
    # anthropometrics
    age: Optional[int] = None
    bsa_m2: Optional[float] = None
    bmi: Optional[float] = None

    # rest hemodynamics
    spap: Optional[float] = None
    dpap: Optional[float] = None
    mpap: Optional[float] = None
    pawp: Optional[float] = None
    rap: Optional[float] = None
    co_td: Optional[float] = None
    co_fick: Optional[float] = None
    co_used: Optional[float] = None
    co_method: str = ""
    ci: Optional[float] = None
    pvr: Optional[float] = None
    pvri: Optional[float] = None
    tpg: Optional[float] = None
    dpg: Optional[float] = None

    # exercise
    exercise_done: bool = False
    mpap_co_slope: Optional[float] = None
    pawp_co_slope: Optional[float] = None
    delta_spap: Optional[float] = None
    ci_rest_ex: Optional[float] = None
    ci_peak_ex: Optional[float] = None
    adaptation_type: Optional[str] = None  # homeometric / heterometric

    # volume challenge
    volume_done: bool = False
    volume_pawp_baseline: Optional[float] = None
    volume_pawp_post: Optional[float] = None
    volume_delta: Optional[float] = None
    volume_positive: Optional[bool] = None

    # echo add-ons
    sprime_raai: Optional[float] = None
    raai: Optional[float] = None

    # hfpef
    hfpef: Optional[H2FPEFResult] = None

    # anemia
    anemia: bool = False
    anemia_threshold: Optional[float] = None

    # trace / warnings
    warnings: List[str] = dataclasses.field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        if self.hfpef:
            d["hfpef"] = self.hfpef.as_dict()
        return d


def derive_from_raw(raw: Dict[str, Any]) -> Derived:
    d = Derived()

    dob = parse_date(raw.get("dob"))
    d.age = to_int(raw.get("age")) or calc_age(dob)

    height = to_float(raw.get("height_cm"))
    weight = to_float(raw.get("weight_kg"))
    bsa = to_float(raw.get("bsa_m2")) or calc_bsa_m2(height, weight)
    d.bsa_m2 = bsa

    d.bmi = to_float(raw.get("bmi")) or calc_bmi(weight, height)

    # Rest pressures
    d.spap = to_float(raw.get("spap_mmHg"))
    d.dpap = to_float(raw.get("dpap_mmHg"))
    d.mpap = to_float(raw.get("mpap_mmHg"))

    if d.mpap is None and d.spap is not None and d.dpap is not None:
        d.mpap = (d.spap + 2.0 * d.dpap) / 3.0

    d.pawp = to_float(raw.get("pawp_mmHg"))
    d.rap = to_float(raw.get("rap_mmHg"))

    d.co_td = to_float(raw.get("td_co_L_min"))
    d.co_fick = to_float(raw.get("fick_co_L_min"))

    # choose CO
    prefer = _clean_str(raw.get("co_preference")).lower()
    co = None
    method = ""
    if prefer == "fick" and d.co_fick is not None:
        co = d.co_fick
        method = "Fick"
    elif d.co_td is not None:
        co = d.co_td
        method = "Thermodilution"
    elif d.co_fick is not None:
        co = d.co_fick
        method = "Fick"
    d.co_used = co
    d.co_method = method

    d.ci = to_float(raw.get("ci_L_min_m2"))
    if d.ci is None and co is not None and bsa is not None and bsa > 0:
        d.ci = co / bsa

    d.pvr = to_float(raw.get("pvr_WU"))
    if d.pvr is None and d.mpap is not None and d.pawp is not None and co is not None and co > 0:
        d.pvr = (d.mpap - d.pawp) / co

    if d.pvr is not None and bsa is not None:
        d.pvri = d.pvr * bsa

    if d.mpap is not None and d.pawp is not None:
        d.tpg = d.mpap - d.pawp

    if d.dpap is not None and d.pawp is not None:
        d.dpg = d.dpap - d.pawp

    # Exercise
    d.exercise_done = bool(raw.get("exercise_done"))

    if d.exercise_done:
        mpap_r = to_float(raw.get("ex_mpap_rest"))
        mpap_p = to_float(raw.get("ex_mpap_peak"))
        pawp_r = to_float(raw.get("ex_pawp_rest"))
        pawp_p = to_float(raw.get("ex_pawp_peak"))

        # CO rest/peak: prefer explicit; else via CI & BSA
        co_r = to_float(raw.get("ex_co_rest"))
        co_p = to_float(raw.get("ex_co_peak"))
        ci_r = to_float(raw.get("ex_ci_rest"))
        ci_p = to_float(raw.get("ex_ci_peak"))
        if co_r is None and ci_r is not None and bsa is not None:
            co_r = ci_r * bsa
        if co_p is None and ci_p is not None and bsa is not None:
            co_p = ci_p * bsa

        d.ci_rest_ex = ci_r
        d.ci_peak_ex = ci_p

        if mpap_r is not None and mpap_p is not None and co_r is not None and co_p is not None and (co_p - co_r) != 0:
            d.mpap_co_slope = (mpap_p - mpap_r) / (co_p - co_r)

        if pawp_r is not None and pawp_p is not None and co_r is not None and co_p is not None and (co_p - co_r) != 0:
            d.pawp_co_slope = (pawp_p - pawp_r) / (co_p - co_r)

        # ΔsPAP
        spap_r = to_float(raw.get("ex_spap_rest")) or d.spap
        spap_p = to_float(raw.get("ex_spap_peak"))
        if spap_r is not None and spap_p is not None:
            d.delta_spap = spap_p - spap_r

        # adaptation type
        if d.delta_spap is not None and d.ci_rest_ex is not None and d.ci_peak_ex is not None:
            if d.delta_spap > 30 and d.ci_peak_ex >= d.ci_rest_ex:
                d.adaptation_type = "homeometrisch"
            else:
                d.adaptation_type = "heterometrisch"

    # Volume challenge (hemodynamic unmasking)
    d.volume_done = bool(raw.get("volume_done"))
    if d.volume_done:
        d.volume_pawp_baseline = to_float(raw.get("volume_pawp_baseline"))
        d.volume_pawp_post = to_float(raw.get("volume_pawp_post"))
        if d.volume_pawp_baseline is not None and d.volume_pawp_post is not None:
            d.volume_delta = d.volume_pawp_post - d.volume_pawp_baseline
            # pragmatic positivity definition (site can adapt):
            # - post >= 18 mmHg OR
            # - delta >= 5 mmHg with post >= 18 mmHg
            d.volume_positive = (d.volume_pawp_post >= 18.0) and (
                (d.volume_delta is None) or (d.volume_delta >= 5.0) or (d.volume_pawp_baseline <= 15.0)
            )
        else:
            d.volume_positive = None

        # Echo add-on S'/RAAI
    sprime = to_float(raw.get("echo_sprime_cm_s"))
    ra_esa = to_float(raw.get("echo_ra_esa_cm2"))
    if sprime is not None and ra_esa is not None and bsa is not None and bsa > 0:
        d.raai = ra_esa / bsa
        if d.raai and d.raai > 0:
            d.sprime_raai = sprime / d.raai

    # HFpEF probability
    af_yes = bool(raw.get("hfpef_af")) or bool(raw.get("af_known"))
    ee = to_float(raw.get("hfpef_e_eprime"))
    pasp = to_float(raw.get("hfpef_pasp"))
    d.hfpef = calc_h2fpef_probability(float(d.age) if d.age is not None else None, d.bmi, ee, pasp, af_yes)

    # anemia
    sex = _clean_str(raw.get("sex")).lower()
    hb = to_float(raw.get("hb_g_dl"))
    thr = None
    if sex.startswith("m"):
        thr = 13.0
    elif sex.startswith("w") or sex.startswith("f"):
        thr = 12.0
    elif sex:
        # unknown/divers: conservative threshold
        thr = 12.5
    if hb is not None and thr is not None:
        d.anemia_threshold = thr
        d.anemia = hb < thr

    # BNP + Entresto hint
    if bool(raw.get("entresto")) and (to_float(raw.get("bnp_pg_ml")) is not None) and (to_float(raw.get("ntprobnp_pg_ml")) is None):
        d.warnings.append("Unter Sacubitril/Valsartan (Entresto®) ist BNP häufig weniger aussagekräftig – NT-proBNP wird bevorzugt interpretiert.")

    return d


# -----------------------------
# Rule engine
# -----------------------------
@dataclass
class Decision:
    main_bundle_id: str
    main_bundle_reason: str
    dx_tags: List[str] = dataclasses.field(default_factory=list)      # kurze Diagnose-Tags (intern)
    dd_hints: List[str] = dataclasses.field(default_factory=list)     # Differentialdiagnosen/Hinweise
    rec_addons: List[str] = dataclasses.field(default_factory=list)   # Zusatz in Empfehlungen
    workup_needed: List[str] = dataclasses.field(default_factory=list)
    auto_modules: List[str] = dataclasses.field(default_factory=list)  # Modul-IDs
    fired_rules: List[str] = dataclasses.field(default_factory=list)


def _bool(x: Any) -> bool:
    return bool(x) and str(x).lower() not in ("0", "false", "nein", "no", "off")


def rule_engine(raw: Dict[str, Any], der: Derived) -> Decision:
    # Convenience
    mpap, pawp, pvr, ci = der.mpap, der.pawp, der.pvr, der.ci
    rest = classify_ph_rest(mpap, pawp, pvr)
    ph_type = rest.get("ph_type", "unknown")
    ph_present = bool(rest.get("ph_present")) if rest.get("ph_present") is not None else False

    # Markers
    ild = _bool(raw.get("ct_ild")) or _bool(raw.get("lufu_restrictive")) or _bool(raw.get("lufu_diffusion"))
    emph = _bool(raw.get("ct_emphysem")) or _bool(raw.get("lufu_obstructive"))
    mosaic = _bool(raw.get("ct_mosaik"))
    chronic_ctep = _bool(raw.get("ct_chronic_thrombo"))  # renamed label; key stays stable
    vq_pos = _bool(raw.get("vq_positive"))
    vq_done = _bool(raw.get("vq_done"))
    # Acute embolism: relevant, but not equal CTEPH
    acute_pe = _bool(raw.get("ct_embolie"))
    portal_htn = _bool(raw.get("portal_htn"))
    immun_pos = _bool(raw.get("immunology_positive"))
    virol_pos = _bool(raw.get("virology_positive"))
    la_enl = _bool(raw.get("cardiac_la_enlarged")) or _bool(raw.get("echo_la_enlarged"))
    af = _bool(raw.get("hfpef_af")) or _bool(raw.get("af_known"))
    lvef = to_float(raw.get("lvef_percent")) or to_float(raw.get("cmr_lvef_percent"))
    lvef_preserved = (lvef is None) or (lvef >= 50)

    # Congestion markers
    congestive_organopathy = _bool(raw.get("congestive_organopathy"))
    rap_high = (der.rap is not None and der.rap >= 12)
    pawp_high = (pawp is not None and pawp > 15)
    ivc_diam = to_float(raw.get("echo_ivc_diam_mm"))
    ivc_collapse = to_float(raw.get("echo_ivc_collapse_pct"))
    ivc_high = (ivc_diam is not None and ivc_diam > 21) and (ivc_collapse is not None and ivc_collapse < 50)

    volume_pos = bool(der.volume_positive) if der.volume_positive is not None else False

    # Exercise pattern
    exercise_done = der.exercise_done
    exercise_pattern = classify_exercise_pattern(der.mpap_co_slope, der.pawp_co_slope) if exercise_done else "unknown"

    # HFpEF
    hfpef = der.hfpef
    hfpef_cat = hfpef.category if hfpef else "unknown"
    hfpef_prob = hfpef.probability_pct if hfpef else None

    # Group likelihood (heuristisch; Workup abhängig vom Kontext)
    group2 = pawp_high or volume_pos or (exercise_pattern in ("left", "mixed")) or la_enl or (hfpef_cat in ("possible", "likely"))
    group3 = ild or emph or _bool(raw.get("ltot_present"))
    group4 = chronic_ctep or mosaic or vq_pos  # avoid acute_pe as sole driver

    # Bundle selection
    reason = ""
    bundle = "K01"  # default: normal
    fired: List[str] = []

    # Shunt has priority
    if _bool(raw.get("step_up_present")):
        bundle, reason = "K16", "Sättigungssprung/Links-Rechts-Shunt Hinweis"
        fired.append("R_SHUNT")

    # Vasoreactivity
    elif _bool(raw.get("vasoreactivity_done")):
        if _bool(raw.get("vasoreactivity_positive")):
            bundle, reason = "K17", "Vasoreagibilitätstest positiv"
            fired.append("R_VASO_POS")
        else:
            bundle, reason = "K18", "Vasoreagibilitätstest negativ"
            fired.append("R_VASO_NEG")

    # No PH in rest: exercise-driven bundles if exercised
    elif ph_type == "none":
        if exercise_done and exercise_pattern in ("left", "mixed"):
            bundle, reason = "K02", "Keine PH in Ruhe, Belastungsreaktion eher linkskardial"
            fired.append("R_EX_LEFT")
        elif exercise_done and exercise_pattern == "pulmvasc":
            bundle, reason = "K03", "Keine PH in Ruhe, Belastungsreaktion eher pulmonalvaskulär"
            fired.append("R_EX_PVASC")
        else:
            bundle, reason = "K01", "Keine PH in Ruhe"
            fired.append("R_NO_PH")

    # Borderline
    elif ph_type == "borderline":
        bundle, reason = "K04", "Grenzbefund / nicht eindeutig klassifizierbar"
        fired.append("R_BORDERLINE")

    # Postcap / CpcPH -> HFpEF / left-heart path
    elif ph_type in ("ipcph", "cpcph"):
        if ph_type == "ipcph":
            bundle, reason = "K14", "Postkapilläre PH (Gruppe 2-Konstellation)"
            fired.append("R_POSTCAP")
        else:
            bundle, reason = "K15", "Kombinierte prä- und postkapilläre PH"
            fired.append("R_CPCPH")

    # Precap
    elif ph_type == "precap":
        # CTEPH path if strong markers
        if group4:
            bundle, reason = "K11", "Präkapilläre PH mit Hinweis auf thromboembolische Genese"
            fired.append("R_GROUP4")
        # Porto/high flow
        elif portal_htn and (ci is not None and ci >= 4.0):
            bundle, reason = "K12", "Hyperzirkulation/portopulmonal (hoher CI) im Kontext portaler Hypertension"
            fired.append("R_PORTO_FLOW")
        # CTD/ILD constellation
        elif immun_pos and group3:
            bundle, reason = "K13", "CTD-Kontext mit ILD-Thema"
            fired.append("R_CTD_ILD")
        else:
            # severity by PVR/CI (pragmatisch; anpassbar)
            pvr_v = pvr or 0.0
            ci_v = ci if ci is not None else 99.0
            if (pvr is not None and pvr_v >= 10.0) or (ci is not None and ci_v < 2.0):
                bundle, reason = "K07", "Schwergradige präkapilläre PH (PVR/CI)"
                fired.append("R_PRECAP_SEVERE")
            elif pvr is not None and pvr_v >= 5.0:
                bundle, reason = "K06", "Mittelgradige präkapilläre PH"
                fired.append("R_PRECAP_MOD")
            else:
                bundle, reason = "K05", "Leichtgradige präkapilläre PH"
                fired.append("R_PRECAP_MILD")
    else:
        bundle, reason = "K04", "Unklare Konstellation"
        fired.append("R_FALLBACK")

    decision = Decision(main_bundle_id=bundle, main_bundle_reason=reason, fired_rules=fired)

    # Differentialdiagnostische Hinweise / Workup
    if ph_present or exercise_done:
        if group4:
            decision.dd_hints.append("Differenzialdiagnostisch Hinweis auf chronisch thromboembolische Genese (CTEPD/CTEPH) – V/Q-Szintigraphie/Board-Path prüfen.")
            if not vq_done:
                decision.workup_needed.append("V/Q-Szintigraphie (CTEPD/CTEPH-Ausschluss)")
            if vq_pos:
                decision.workup_needed.append("Vorstellung im CTEPH-Board (PEA/BPA/medikamentöse Optionen)")
        if group3:
            decision.dd_hints.append("Begleitende/führende Lungenerkrankung möglich (Gruppe 3) – Pneumologie/ILD-Assessment je nach Kontext.")
            if not _bool(raw.get("lufu_done")):
                decision.workup_needed.append("Lungenfunktion inkl. DLCO")
            if _bool(raw.get("ct_ild")) and not _bool(raw.get("ild_fibrosis_clinic")):
                decision.workup_needed.append("Anbindung Fibroseambulanz / ILD-Board")
        if group2 and lvef_preserved:
            decision.dd_hints.append("Hinweise auf linksatriale/diastolische Komponente (Gruppe 2/HFpEF) – kardiologische Mitbeurteilung/Diastolik.")
            if volume_pos:
                decision.dd_hints.append("Volumenchallenge mit deutlicher PAWP-Erhöhung spricht für eine diastolische/linksatriale Komponente (HFpEF-Konstellation) – klinisch einordnen.")
                decision.fired_rules.append("R_VOLUME_POS")
            # Provide HFpEF score based suggestion
            if hfpef_cat in ("possible", "likely"):
                decision.workup_needed.append("HFpEF-Abklärung/Diastolik (Echo inkl. E/e', LA, ggf. Belastung/Volumen-Kontext)")
        if immun_pos:
            decision.dd_hints.append("Immunologischer Hinweis: CTD-assoziierte PH/DD (Gruppe 1) erwägen.")
            decision.workup_needed.append("Rheumatologische/Autoimmun-Diagnostik (falls nicht komplett)")
        if virol_pos:
            decision.dd_hints.append("Virologischer Kontext: sekundäre Ursachen (z.B. HIV/Hepatitis) in DD berücksichtigen.")
        if portal_htn:
            decision.dd_hints.append("Portale Hypertension: portopulmonale Konstellation in DD berücksichtigen.")

    # Congestion recommendations
    if congestive_organopathy or rap_high or pawp_high or ivc_high:
        decision.rec_addons.append("Bei Stauungszeichen: konsequentes Volumenmanagement/Dekongestion und Kontrolle von Nierenfunktion/Elekrolyten.")
        decision.auto_modules.append("P02")  # diuretics module from textdb, if present
        decision.fired_rules.append("R_CONGESTION")

    # anemia recommendations
    if der.anemia:
        morph = _clean_str(raw.get("anemia_morphology")).lower()
        if morph == "mikrozytär":
            decision.rec_addons.append("Anämie (mikrozytär): Eisenmangel/chronische Blutung differenzialdiagnostisch führend – Ferritin, Transferrinsättigung, CRP/Entzündung, ggf. gastrointestinale Blutungsquelle prüfen.")
        elif morph == "makrozytär":
            decision.rec_addons.append("Anämie (makrozytär): Vitamin-B12-/Folat-Mangel, Leber-/Schilddrüsenfunktion, Alkohol/Medikamente differenzialdiagnostisch prüfen.")
        elif morph == "normozytär":
            decision.rec_addons.append("Anämie (normozytär): Entzündung/chronische Erkrankung, Nierenfunktion, ggf. Hämolyse/Blutung differenzialdiagnostisch prüfen.")
        elif morph:
            decision.rec_addons.append(f"Anämie ({morph}): gezielte Abklärung je nach Muster (Eisenstatus/B12/Folat/Entzündung/Niere) empfohlen.")
        else:
            decision.rec_addons.append("Anämie-Konstellation: Abklärung der Ursache (z.B. Eisenstatus, B12/Folat, Entzündung, Nierenfunktion) empfohlen.")
        decision.auto_modules.append("P13")  # iron/anemia module if present
        decision.fired_rules.append("R_ANEMIA")

    # HFpEF therapy hint (clinician-facing; optional medication naming)
    if lvef_preserved and hfpef_cat == "likely":
        decision.rec_addons.append("Bei hoher HFpEF-Wahrscheinlichkeit: SGLT2-Inhibitor-Therapie kann erwogen werden (z.B. Empagliflozin 10 mg 1×/d), unter Beachtung Kontraindikationen, eGFR und typischer Nebenwirkungen (u.a. genitale Infektionen, Volumenmangel).")
        decision.fired_rules.append("R_HFPEF_SGLT2")

    # Volume challenge consequences
    if volume_pos:
        decision.rec_addons.append("Bei positiver Volumenchallenge: es besteht ein starker Hinweis auf eine (okkulte) linksatriale/diastolische Komponente; Therapiekonzept entsprechend (Volumenstatus, Blutdruck, HFpEF-Strategien) prüfen.")
        decision.fired_rules.append("R_VOLUME_CONSEQUENCE")

        # Entresto/BNP note
    for w in der.warnings:
        decision.rec_addons.append(w)

    # Exercise adaptation type sentence -> in Beurteilung, not recommendations; handled in context builder

    return decision


# -----------------------------
# Risk scoring (pragmatisch, robust; nicht vollständiger Leitlinienersatz)
# -----------------------------
@dataclass
class RiskResult:
    name: str
    category: str
    details: str = ""
    score: Optional[float] = None


def _risk_cat_esc3(who_fc: Optional[str], six_mwd: Optional[float], bnp: Optional[float], ntprobnp: Optional[float]) -> RiskResult:
    """
    Very pragmatic ESC3 using the common cutoffs.
    If multiple parameters present -> worst category wins.
    """
    # Categories: low / intermediate / high / unknown
    cats = []

    if who_fc:
        fc = who_fc.strip().upper().replace("WHO", "").replace("FC", "").strip()
        if fc in ("I", "1"):
            cats.append("low")
        elif fc in ("II", "2"):
            cats.append("low")
        elif fc in ("III", "3"):
            cats.append("intermediate")
        elif fc in ("IV", "4"):
            cats.append("high")

    if six_mwd is not None:
        if six_mwd > 440:
            cats.append("low")
        elif six_mwd >= 165:
            cats.append("intermediate")
        else:
            cats.append("high")

    # BNP/NT-proBNP (typical ESC cutoffs; interpret in context)
    if ntprobnp is not None:
        if ntprobnp < 300:
            cats.append("low")
        elif ntprobnp <= 1400:
            cats.append("intermediate")
        else:
            cats.append("high")
    elif bnp is not None:
        if bnp < 50:
            cats.append("low")
        elif bnp <= 300:
            cats.append("intermediate")
        else:
            cats.append("high")

    if not cats:
        return RiskResult(name="ESC/ERS (3-Strata)", category="unknown", details="Unzureichende Eingaben für Risikoeinschätzung.")

    # worst wins
    if "high" in cats:
        cat = "high"
    elif "intermediate" in cats:
        cat = "intermediate"
    else:
        cat = "low"
    return RiskResult(name="ESC/ERS (3-Strata)", category=cat, details="(pragmatisch aus WHO-FC/6MWD/BNP bzw. NT-proBNP)")


def _risk_cat_esc4(who_fc: Optional[str], six_mwd: Optional[float], bnp: Optional[float], ntprobnp: Optional[float]) -> RiskResult:
    """
    Pragmatic ESC4: low / intermediate-low / intermediate-high / high.
    """
    cats = []

    # WHO FC mapping (coarse)
    if who_fc:
        fc = who_fc.strip().upper().replace("WHO", "").replace("FC", "").strip()
        if fc in ("I", "1"):
            cats.append("low")
        elif fc in ("II", "2"):
            cats.append("intermediate-low")
        elif fc in ("III", "3"):
            cats.append("intermediate-high")
        elif fc in ("IV", "4"):
            cats.append("high")

    if six_mwd is not None:
        if six_mwd > 440:
            cats.append("low")
        elif six_mwd >= 320:
            cats.append("intermediate-low")
        elif six_mwd >= 165:
            cats.append("intermediate-high")
        else:
            cats.append("high")

    # Biomarkers (approximate; site can adapt)
    if ntprobnp is not None:
        if ntprobnp < 300:
            cats.append("low")
        elif ntprobnp < 650:
            cats.append("intermediate-low")
        elif ntprobnp <= 1400:
            cats.append("intermediate-high")
        else:
            cats.append("high")
    elif bnp is not None:
        if bnp < 50:
            cats.append("low")
        elif bnp < 200:
            cats.append("intermediate-low")
        elif bnp <= 300:
            cats.append("intermediate-high")
        else:
            cats.append("high")

    if not cats:
        return RiskResult(name="ESC/ERS (4-Strata)", category="unknown", details="Unzureichende Eingaben für Risikoeinschätzung.")

    # worst wins order
    order = ["low", "intermediate-low", "intermediate-high", "high"]
    idx = max(order.index(c) for c in cats if c in order)
    return RiskResult(name="ESC/ERS (4-Strata)", category=order[idx], details="(pragmatisch aus WHO-FC/6MWD/BNP bzw. NT-proBNP)")


def _risk_reveal_lite2(sys_bp: Optional[float], hr: Optional[float], egfr: Optional[float], ntprobnp: Optional[float], bnp: Optional[float], six_mwd: Optional[float], who_fc: Optional[str]) -> RiskResult:
    """
    Very simplified REVEAL Lite 2 proxy. Not the official calculator.
    We mainly want a consistent "traffic light" display.
    """
    # points style (0..??)
    pts = 0
    details = []
    if sys_bp is not None:
        if sys_bp < 110:
            pts += 2
            details.append("RRsys niedrig")
        elif sys_bp < 120:
            pts += 1
    if hr is not None:
        if hr > 96:
            pts += 2
            details.append("HF hoch")
        elif hr > 84:
            pts += 1
    if egfr is not None:
        if egfr < 30:
            pts += 2
            details.append("eGFR niedrig")
        elif egfr < 60:
            pts += 1
    biom = ntprobnp if ntprobnp is not None else bnp
    if biom is not None:
        if (ntprobnp is not None and biom > 1400) or (ntprobnp is None and biom > 300):
            pts += 2
            details.append("Biomarker hoch")
        elif (ntprobnp is not None and biom >= 300) or (ntprobnp is None and biom >= 50):
            pts += 1
    if six_mwd is not None:
        if six_mwd < 165:
            pts += 2
            details.append("6MWD niedrig")
        elif six_mwd < 320:
            pts += 1
    if who_fc:
        fc = who_fc.strip().upper().replace("WHO", "").replace("FC", "").strip()
        if fc in ("IV", "4"):
            pts += 2
        elif fc in ("III", "3"):
            pts += 1

    # map pts
    if pts <= 2:
        cat = "low"
    elif pts <= 5:
        cat = "intermediate"
    else:
        cat = "high"
    return RiskResult(name="REVEAL Lite 2 (Proxy)", category=cat, score=float(pts), details="; ".join(details) if details else "")


def compute_risks(raw: Dict[str, Any], der: Derived) -> List[RiskResult]:
    who_fc = _clean_str(raw.get("who_fc")) or None
    six_mwd = to_float(raw.get("six_mwd_m"))
    bnp = to_float(raw.get("bnp_pg_ml"))
    ntprobnp = to_float(raw.get("ntprobnp_pg_ml"))
    esc3 = _risk_cat_esc3(who_fc, six_mwd, bnp, ntprobnp)
    esc4 = _risk_cat_esc4(who_fc, six_mwd, bnp, ntprobnp)

    sys_bp = to_float(raw.get("reveal_rbsys"))
    hr = to_float(raw.get("reveal_hr"))
    egfr = to_float(raw.get("reveal_egfr"))
    reveal = _risk_reveal_lite2(sys_bp, hr, egfr, ntprobnp, bnp, six_mwd, who_fc)

    out = [esc3, esc4, reveal]

    # HFpEF result as "risk-like" card
    if der.hfpef and der.hfpef.probability_pct is not None:
        out.append(RiskResult(name="HFpEF-Wahrscheinlichkeit (H2FPEF)", category=der.hfpef.category, score=der.hfpef.probability_pct, details="Kontinuierliches Modell (CirculationAHA.118.034646)"))
    return out


def _risk_badge_html(rr: RiskResult) -> str:
    cat = rr.category
    cls = {
        "low": "badge badge-low",
        "intermediate": "badge badge-mid",
        "intermediate-low": "badge badge-midlow",
        "intermediate-high": "badge badge-midhigh",
        "high": "badge badge-high",
        "unlikely": "badge badge-low",
        "possible": "badge badge-mid",
        "likely": "badge badge-high",
        "unknown": "badge badge-unk",
    }.get(cat, "badge badge-unk")
    score = ""
    if rr.score is not None:
        if rr.name.startswith("HFpEF"):
            score = f"{fmt_num(rr.score, 0)} %"
        else:
            score = fmt_num(rr.score, 0)
    return f"<div class='riskcard'><div class='riskname'>{rr.name}</div><div class='{cls}'>{cat}</div><div class='riskscore'>{score}</div><div class='riskdetail'>{rr.details}</div></div>"


def build_risk_dashboard_html(risks: List[RiskResult]) -> str:
    cards = "\n".join(_risk_badge_html(r) for r in risks)
    return f"""
    <div class='riskwrap'>
      {cards}
    </div>
    """


# -----------------------------
# Report rendering helpers (textdb)
# -----------------------------
def _get_block(block_id: str):
    if not TEXTDB_AVAILABLE:
        return None
    try:
        return _textdb.get_block(block_id)
    except Exception:
        return None


def render_block(block_id: str, ctx: Dict[str, Any]) -> str:
    blk = _get_block(block_id)
    if not blk:
        return ""
    tpl = getattr(blk, "template", "")
    try:
        return tpl.format_map(SafeDict(ctx)).strip()
    except Exception:
        # fallback: remove braces
        return re.sub(r"\{[^}]+\}", "", tpl).strip()


def render_bundle(bundle_id: str, ctx: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Returns beurteilung_text, empfehlung_text, p_suggestions(ids)
    """
    if TEXTDB_AVAILABLE and hasattr(_textdb, "BUNDLES"):
        b = _textdb.BUNDLES.get(bundle_id, {})
        b_ids = b.get("B", [])
        e_ids = b.get("E", [])
        p_ids = b.get("P_suggestions", []) or []
        beur = "\n".join([render_block(i, ctx) for i in b_ids if i]).strip()
        empf = "\n".join([render_block(i, ctx) for i in e_ids if i]).strip()
        return beur, empf, list(p_ids)
    # fallback
    return "", "", []


def _severity_label(pvr: Optional[float], ci: Optional[float]) -> str:
    if pvr is None:
        return ""
    if pvr >= 10 or (ci is not None and ci < 2.0):
        return "schwer"
    if pvr >= 5:
        return "mittelgradig"
    if pvr >= 2:
        return "leicht"
    return ""


def build_ctx(raw: Dict[str, Any], der: Derived, decision: Decision, risks: List[RiskResult]) -> Dict[str, Any]:
    """
    Baut die Platzhalter-Map für Textbausteine (rhk_textdb.py).
    Missing keys werden via SafeDict leer.
    """
    mpap = der.mpap
    pawp = der.pawp
    pvr = der.pvr
    ci = der.ci
    co_method = der.co_method or "CO"
    tpg = der.tpg

    # Basic phrases
    ctx: Dict[str, Any] = {}
    ctx["mpap_phrase"] = f"mPAP {fmt_num(mpap, 0)} mmHg" if mpap is not None else ""
    ctx["pawp_phrase"] = f"PAWP {fmt_num(pawp, 0)} mmHg" if pawp is not None else ""
    ctx["pvr_phrase"] = f"PVR {fmt_num(pvr, 2)} WU" if pvr is not None else ""
    ctx["ci_phrase"] = f"(CI {fmt_num(ci, 2)} L/min/m²)" if ci is not None else ""
    ctx["CI_value"] = fmt_num(ci, 2) if ci is not None else ""
    ctx["co_method_desc"] = co_method
    ctx["TPG_value"] = fmt_num(tpg, 0) if tpg is not None else ""
    ctx["step_up_sentence"] = "Kein relevanter Sättigungssprung." if not _bool(raw.get("step_up_present")) else "Sättigungssprung in der Stufenoxymetrie."
    ctx["systemic_sentence"] = ""
    ctx["oxygen_sentence"] = ""
    ctx["comparison_sentence"] = ""

    # Rest PH sentence for K03
    rest = classify_ph_rest(mpap, pawp, pvr)
    if rest.get("ph_type") == "none":
        ctx["rest_ph_sentence"] = "keine pulmonale Hypertonie in Ruhe"
    elif rest.get("ph_type") == "borderline":
        ctx["rest_ph_sentence"] = "grenzwertige Druckkonstellation in Ruhe"
    else:
        ctx["rest_ph_sentence"] = "pulmonale Hypertonie in Ruhe"

    # Borderline sentence
    if rest.get("ph_type") == "borderline":
        ctx["borderline_ph_sentence"] = "Grenzwertige pulmonale Druckwerte"
    else:
        ctx["borderline_ph_sentence"] = ""

    # Congestion phrases
    cv = []
    if der.rap is not None:
        if der.rap < 8:
            cv.append("keine zentrale Stauung")
        elif der.rap < 12:
            cv.append("leichtgradige zentrale Stauung")
        else:
            cv.append("ausgeprägte zentrale Stauung")
    ctx["cv_stauung_phrase"] = (cv[0] + ".") if cv else ""

    pv = []
    if pawp is not None:
        if pawp <= 15:
            pv.append("keine pulmonalvenöse Stauung")
        else:
            pv.append("Hinweis auf pulmonalvenöse Stauung")
    ctx["pv_stauung_phrase"] = (pv[0] + ".") if pv else ""

    # V-wave
    ctx["V_wave_short"] = ""
    if _bool(raw.get("v_wave_present")):
        ctx["V_wave_short"] = " (prominente v-Welle)"

    # Exercise slopes placeholders
    if der.exercise_done:
        ctx["mPAP_CO_slope"] = fmt_num(der.mpap_co_slope, 2) if der.mpap_co_slope is not None else "—"
        ctx["PAWP_CO_slope"] = fmt_num(der.pawp_co_slope, 2) if der.pawp_co_slope is not None else "—"
        ctx["PAWP_CO_slope_phrase"] = "nicht führend erhöhter PAWP/CO-Slope" if classify_exercise_pattern(der.mpap_co_slope, der.pawp_co_slope) == "pulmvasc" else "PAWP/CO-Slope"
    else:
        ctx["mPAP_CO_slope"] = ""
        ctx["PAWP_CO_slope"] = ""
        ctx["PAWP_CO_slope_phrase"] = ""

    # Additional adaptation sentence
    if der.exercise_done and der.adaptation_type:
        ctx["exercise_adaptation_sentence"] = f"Die Belastungsreaktion spricht für einen {der.adaptation_type} Adaptationstyp des Ventrikels."
    else:
        ctx["exercise_adaptation_sentence"] = ""

    # Echo S'/RAAI
    if der.sprime_raai is not None:
        cutoff = 0.81
        if TEXTDB_AVAILABLE and hasattr(_textdb, "DEFAULT_RULES"):
            try:
                cutoff = float(_textdb.DEFAULT_RULES.get("echo", {}).get("Sprime_RAAI_cutoff_m2_per_s_cm", cutoff))
            except Exception:
                pass
        interp = "nicht erniedrigt"
        if der.sprime_raai < cutoff:
            interp = "erniedrigt"
        ctx["Sprime_RAAI_value"] = fmt_num(der.sprime_raai, 2)
        ctx["Sprime_RAAI_cutoff"] = fmt_num(cutoff, 2)
        ctx["Sprime_RAAI_interpretation_sentence"] = interp
    else:
        ctx["Sprime_RAAI_value"] = ""
        ctx["Sprime_RAAI_cutoff"] = ""
        ctx["Sprime_RAAI_interpretation_sentence"] = ""

    # Risk short text
    # pick most relevant: ESC4 if available else ESC3
    esc4 = next((r for r in risks if r.name.startswith("ESC/ERS (4")), None)
    if esc4 and esc4.category != "unknown":
        ctx["risk_profile_desc"] = f"{esc4.category}"
    else:
        ctx["risk_profile_desc"] = ""

    # exam type
    ctx["exam_type_desc"] = "RHK"  # can be overridden
    ctx["therapy_neutral_sentence"] = ""
    ctx["therapy_plan_sentence"] = ""
    ctx["therapy_escalation_sentence"] = ""
    ctx["anticoagulation_plan_sentence"] = ""

    # Comparison sentence
    prev_date = _clean_str(raw.get("prev_rhk_date"))
    if prev_date:
        prev = {
            "mpap": to_float(raw.get("prev_mpap")),
            "pawp": to_float(raw.get("prev_pawp")),
            "ci": to_float(raw.get("prev_ci")),
            "pvr": to_float(raw.get("prev_pvr")),
        }
        if any(v is not None for v in prev.values()):
            stable = _clean_str(raw.get("prev_course")) or "Verlauf"
            parts = []
            if prev["mpap"] is not None:
                parts.append(f"mPAP {fmt_num(prev['mpap'], 0)} mmHg")
            if prev["pawp"] is not None:
                parts.append(f"PAWP {fmt_num(prev['pawp'], 0)} mmHg")
            if prev["ci"] is not None:
                parts.append(f"CI {fmt_num(prev['ci'], 2)} l/min/m²")
            if prev["pvr"] is not None:
                parts.append(f"PVR {fmt_num(prev['pvr'], 2)} WU")
            ctx["comparison_sentence"] = f"Im Vergleich zu RHK {prev_date} {stable} ({', '.join(parts)})."

    return ctx


# -----------------------------
# Report generation
# -----------------------------
def generate_doctor_report(raw: Dict[str, Any]) -> Tuple[str, Derived, Decision, List[RiskResult]]:
    der = derive_from_raw(raw)
    risks = compute_risks(raw, der)
    decision = rule_engine(raw, der)

    ctx = build_ctx(raw, der, decision, risks)
    beur, empf, p_suggestions = render_bundle(decision.main_bundle_id, ctx)

    # Insert hemo block (structured)
    severity = _severity_label(der.pvr, der.ci)
    rest = classify_ph_rest(der.mpap, der.pawp, der.pvr)
    ph_type = rest.get("ph_type", "unknown")

    hemo_lines = []
    hemo_lines.append("Hämodynamik (Ruhe):")
    hemo_lines.append(f"• mPAP: {fmt_mmHg(der.mpap)}; PAWP: {fmt_mmHg(der.pawp)}; RAP: {fmt_mmHg(der.rap)}")
    hemo_lines.append(f"• CI: {fmt_l_min_m2(der.ci)}; PVR: {fmt_wu(der.pvr)}; TPG: {fmt_mmHg(der.tpg)}")
    if der.dpg is not None:
        hemo_lines.append(f"• DPG: {fmt_mmHg(der.dpg)}")
    if der.pvri is not None:
        hemo_lines.append(f"• PVRi: {fmt_num(der.pvri,2)} WU·m²")
    if der.co_used is not None:
        hemo_lines.append(f"• HZV: {fmt_num(der.co_used,2)} L/min ({der.co_method})")
    hemo_lines.append(f"• Einordnung: {ph_type}{' ('+severity+')' if severity else ''}")

    # exercise block
    if der.exercise_done:
        ex_lines = []
        ex_lines.append("Belastung:")
        if der.mpap_co_slope is not None:
            ex_lines.append(f"• mPAP/CO-Slope: {fmt_num(der.mpap_co_slope,2)} mmHg/(L/min)")
        if der.pawp_co_slope is not None:
            ex_lines.append(f"• PAWP/CO-Slope: {fmt_num(der.pawp_co_slope,2)} mmHg/(L/min)")
        if der.delta_spap is not None:
            ex_lines.append(f"• ΔsPAP (Peak–Ruhe): {fmt_num(der.delta_spap,0)} mmHg")
        if der.ci_rest_ex is not None and der.ci_peak_ex is not None:
            ex_lines.append(f"• CI Ruhe/Peak: {fmt_num(der.ci_rest_ex,2)} → {fmt_num(der.ci_peak_ex,2)} L/min/m²")
        if der.adaptation_type:
            ex_lines.append(f"• Ventrikelreaktion: {der.adaptation_type} (heuristisch)")
        hemo_lines.append("")
        hemo_lines.extend(ex_lines)

    # echo add-ons
    if der.sprime_raai is not None:
        hemo_lines.append("")
        hemo_lines.append("Echo-Add-on:")
        hemo_lines.append(f"• S'/RAAi: {fmt_num(der.sprime_raai,2)} m²/(s·cm)")

    # Risk block (prominent)
    risk_lines = []
    risk_lines.append("Risikoeinschätzung:")
    for r in risks:
        if r.name.startswith("HFpEF"):
            if r.score is not None:
                risk_lines.append(f"• {r.name}: {fmt_num(r.score,0)} % ({r.category})")
        else:
            risk_lines.append(f"• {r.name}: {r.category}" + (f" (Score {fmt_num(r.score,0)})" if r.score is not None and r.name.startswith("REVEAL") else ""))

    # Recommendations block
    rec_lines = []
    rec_lines.append("Empfehlungen:")
    # Addons from rules
    for line in decision.rec_addons:
        rec_lines.append(f"• {line}")
    # Workup list
    if decision.workup_needed:
        rec_lines.append("• Diagnostik/Workup (aus den Eingaben abgeleitet):")
        for w in decision.workup_needed:
            rec_lines.append(f"  – {w}")

    # Module integration: automatische Vorschläge + Nutzer-Auswahl + Paketvorschläge
    selected_modules_raw = raw.get("modules") or []
    if isinstance(selected_modules_raw, str):
        selected_modules_raw = [selected_modules_raw]
    selected_modules = [normalize_module_id(x) for x in selected_modules_raw]
    selected_modules = [m for m in selected_modules if m]

    module_ids = dedup_preserve((decision.auto_modules or []) + (p_suggestions or []) + selected_modules)

    if module_ids:
        rec_lines.append("• Procedere/Module:")
        for mid in module_ids:
            mt = render_block(mid, ctx)
            if mt:
                # module text can be multi-line
                for line in mt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec_lines.append(f"  – {line}")
            else:
                rec_lines.append(f"  – {mid}")

    # Compose final report
    parts = []
    # header
    header = f"RHK Befundassistent {APP_VERSION}"
    parts.append(header)
    parts.append("=" * len(header))
    # patient identification
    name = " ".join([_clean_str(raw.get("firstname")), _clean_str(raw.get("lastname"))]).strip()
    if name:
        parts.append(f"Patient: {name}")
    if der.age is not None:
        parts.append(f"Alter: {der.age} Jahre")
    story = _clean_str(raw.get("story"))
    if story:
        parts.append(f"Anamnese/Kontext: {story}")

    comorb = _clean_str(raw.get("comorbidities_text"))
    if comorb:
        parts.append(f"Vorerkrankungen: {comorb}")

    comorb_ph = _clean_str(raw.get("comorbidities_ph_relevant"))
    if comorb_ph:
        parts.append(f"PH-relevante Hinweise: {comorb_ph}")

    # Medikation (Kurz)
    med_lines = []
    if bool(raw.get("ph_meds_current")):
        med_lines.append("PH-Therapie aktuell: " + (_clean_str(raw.get("ph_meds_current_which")) or "ja"))
        since = _clean_str(raw.get("ph_meds_current_since"))
        if since:
            med_lines.append(f"  (seit: {since})")
    if bool(raw.get("ph_meds_past")):
        med_lines.append("PH-Therapie früher: " + (_clean_str(raw.get("ph_meds_past_which")) or "ja"))
    if bool(raw.get("diuretics")):
        med_lines.append("Diuretika: ja")
    other = _clean_str(raw.get("other_meds_text"))
    if other:
        med_lines.append("Weitere Medikation: " + other)
    if med_lines:
        parts.append("Medikation: " + " ".join(med_lines).strip())

    parts.append("")
    parts.append("\n".join(hemo_lines).strip())
    parts.append("")
    if beur:
        parts.append("Beurteilung:")
        parts.append(beur.strip())
        if ctx.get("exercise_adaptation_sentence"):
            parts.append(ctx["exercise_adaptation_sentence"])
        parts.append("")
    parts.append("\n".join(risk_lines).strip())
    parts.append("")
    if empf:
        # Ensure scores appear directly after diagnosis in recommendation part:
        parts.append("Empfehlung (Paket):")
        parts.append(empf.strip())
        parts.append("")
    parts.append("\n".join(rec_lines).strip())

    # DD hints (optional)
    if decision.dd_hints:
        parts.append("")
        parts.append("Differenzialdiagnostische Hinweise:")
        for h in decision.dd_hints:
            parts.append(f"• {h}")

    # Free text (optional)
    free = _clean_str(raw.get("final_free_text"))
    if free:
        parts.append("")
        parts.append("Zusatz (Freitext):")
        parts.append(free)

    report = "\n".join([p for p in parts if p is not None]).strip()
    return report, der, decision, risks


def generate_patient_report(raw: Dict[str, Any], der: Derived, decision: Decision, risks: List[RiskResult]) -> str:
    """
    Patiententext: keine Abkürzungen, keine Zahlenwerte; Fokus auf Bedeutung + nächste Schritte.
    """
    # Basic context
    name = _clean_str(raw.get("firstname"))
    story = _clean_str(raw.get("story"))
    exercise = der.exercise_done

    rest = classify_ph_rest(der.mpap, der.pawp, der.pvr)
    ph_type = rest.get("ph_type", "unknown")

    # Determine severity wording
    sev = _severity_label(der.pvr, der.ci)
    if sev == "schwer":
        sev_word = "ausgeprägt"
    elif sev == "mittelgradig":
        sev_word = "deutlich"
    elif sev == "leicht":
        sev_word = "leicht"
    else:
        sev_word = ""

    # group wording
    # Use decision hints + markers
    group_sentences = []
    if any("Gruppe 2" in h or "HFpEF" in h for h in decision.dd_hints) or ph_type in ("ipcph", "cpcph"):
        group_sentences.append("Die Messwerte passen zu einer Belastung durch das linke Herz (z.B. eine Steifigkeit des Herzmuskels).")
    if any("Lungenerkrankung" in h or "Gruppe 3" in h for h in decision.dd_hints):
        group_sentences.append("Außerdem können Veränderungen der Lunge oder eine niedrige Sauerstoffversorgung eine Rolle spielen.")
    if any("CTEP" in h for h in decision.dd_hints):
        group_sentences.append("Es gibt Hinweise, dass frühere Blutgerinnsel in den Lungengefäßen beteiligt sein könnten.")

    # Start writing
    paras: List[str] = []

    if name:
        paras.append(f"Hallo {name},")
    paras.append("wir haben eine Herzkatheter-Untersuchung durchgeführt. Dabei messen wir Druck und Blutfluss im Herzen und in den Lungengefäßen.")

    if story:
        paras.append(f"Grund für die Untersuchung war: {story}")

    # Main result
    if ph_type == "none" and not exercise:
        paras.append("Die Druckwerte in den Lungengefäßen sind in Ruhe unauffällig. Das spricht gegen eine Lungenhochdruck-Erkrankung in Ruhe.")
    elif ph_type == "none" and exercise:
        pat = classify_exercise_pattern(der.mpap_co_slope, der.pawp_co_slope)
        if pat in ("left", "mixed"):
            paras.append("In Ruhe waren die Werte unauffällig. Unter Belastung zeigte sich jedoch eine Reaktion, die eher zu einer Belastung durch das linke Herz passt.")
        elif pat == "pulmvasc":
            paras.append("In Ruhe waren die Werte unauffällig. Unter Belastung zeigte sich jedoch eine auffällige Reaktion in den Lungengefäßen.")
        else:
            paras.append("In Ruhe waren die Werte unauffällig. Unter Belastung gab es Hinweise auf eine auffällige Reaktion, die wir im Gesamtbild einordnen.")
    elif ph_type == "ipcph":
        paras.append("Es zeigt sich eine Lungenhochdruck-Erkrankung, die vor allem durch einen Rückstau vom linken Herzen mitbedingt ist.")
    elif ph_type == "cpcph":
        paras.append("Es zeigt sich eine Lungenhochdruck-Erkrankung mit einer Kombination aus Rückstau vom linken Herzen und zusätzlich erhöhtem Widerstand in den Lungengefäßen.")
    elif ph_type == "precap":
        if sev_word:
            paras.append(f"Es zeigt sich eine {sev_word} Lungenhochdruck-Erkrankung, die vor allem durch einen erhöhten Widerstand in den Lungengefäßen entsteht.")
        else:
            paras.append("Es zeigt sich eine Lungenhochdruck-Erkrankung, die vor allem durch einen erhöhten Widerstand in den Lungengefäßen entsteht.")
    else:
        paras.append("Die Messwerte sind grenzwertig oder nicht eindeutig. Wir interpretieren das zusammen mit Ihren Beschwerden und den weiteren Untersuchungen.")

    if group_sentences:
        paras.extend(group_sentences)

    # Symptoms / risk
    esc4 = next((r for r in risks if r.name.startswith("ESC/ERS (4")), None)
    if esc4 and esc4.category != "unknown":
        if esc4.category in ("low", "unlikely"):
            paras.append("In der Gesamteinschätzung wirken die aktuellen Risikomerkmale eher günstig.")
        elif "high" in esc4.category:
            paras.append("In der Gesamteinschätzung gibt es Risikomerkmale, die eine engmaschige Betreuung und zügige nächste Schritte wichtig machen.")
        else:
            paras.append("In der Gesamteinschätzung liegen Risikomerkmale im mittleren Bereich vor. Das bedeutet: Behandlung und Kontrollen sollten gut geplant werden.")

    # Next steps
    next_steps: List[str] = []
    if decision.workup_needed:
        next_steps.append("Als nächste Schritte empfehlen wir zusätzliche Untersuchungen, um die Ursache besser zu verstehen.")
    if any("Volumenmanagement" in x or "Dekongestion" in x for x in decision.rec_addons):
        next_steps.append("Wichtig ist außerdem, dass Zeichen von Wassereinlagerung oder Rückstau konsequent behandelt werden.")
    if der.anemia:
        next_steps.append("Im Blutbild zeigt sich eine Blutarmut. Das sollte gezielt abgeklärt werden, weil es die Belastbarkeit beeinflussen kann.")
    if der.hfpef and der.hfpef.category in ("possible", "likely"):
        next_steps.append("Es gibt Hinweise, dass eine Form der Herzschwäche mit erhaltener Pumpfunktion eine Rolle spielen könnte. Das wird kardiologisch weiter eingeordnet.")

    if next_steps:
        paras.append("Wie geht es weiter?")
        paras.extend([f"• {s}" for s in next_steps])

    paras.append("Bitte besprechen Sie die Ergebnisse und das weitere Vorgehen mit Ihrem behandelnden Team. Wenn neue oder starke Beschwerden auftreten (z.B. Ohnmacht, starke Luftnot, Brustschmerz), sollten Sie zeitnah ärztliche Hilfe suchen.")

    return "\n\n".join(paras).strip()


def generate_internal_debug(raw: Dict[str, Any], der: Derived, decision: Decision) -> str:
    lines: List[str] = []
    lines.append(f"DEBUG {APP_VERSION}")
    lines.append(f"main_bundle_id: {decision.main_bundle_id} ({decision.main_bundle_reason})")
    lines.append(f"fired_rules: {', '.join(decision.fired_rules) if decision.fired_rules else '—'}")
    lines.append("")
    lines.append("Derived:")
    for k, v in der.as_dict().items():
        if k == "hfpef" and isinstance(v, dict):
            lines.append(f"- hfpef: {json.dumps(v, ensure_ascii=False)}")
        else:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines).strip()


# -----------------------------
# PDF Export
# -----------------------------
def _pdf_available() -> bool:
    try:
        import reportlab  # noqa
        return True
    except Exception:
        return False


def make_pdf_from_text(title: str, text: str, filename_prefix: str = "rhk_report") -> Optional[str]:
    """
    Creates a simple PDF and returns the file path.
    Needs reportlab.
    """
    if not _pdf_available():
        return None

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    # Write to temp file
    fd, path = tempfile.mkstemp(prefix=filename_prefix + "_", suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm
    line_height = 12

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 18

    c.setFont("Helvetica", 10)

    # Wrap lines
    max_width = width - 4 * cm
    for para in text.split("\n"):
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 10)
        if para.strip() == "":
            y -= line_height
            continue

        # naive wrap by character count; good enough for monospaced-ish
        # estimate chars per line
        approx_chars = int(max_width / 5.5)
        chunks = textwrap_wrap(para, approx_chars)
        for ch in chunks:
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 10)
            c.drawString(x, y, ch)
            y -= line_height

    c.save()
    return path


def textwrap_wrap(s: str, width: int) -> List[str]:
    # simple safe wrap without importing textwrap (keep minimal)
    if width <= 10:
        return [s]
    words = s.split(" ")
    out = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


# -----------------------------
# Public API for GUI
# -----------------------------
def generate_all(raw: Dict[str, Any]) -> Tuple[str, str, str, str, Derived, Decision, List[RiskResult]]:
    """
    Returns:
      doctor_txt, patient_txt, internal_txt, risk_html, derived, decision, risks
    """
    doctor_txt, der, decision, risks = generate_doctor_report(raw)
    patient_txt = generate_patient_report(raw, der, decision, risks)
    internal = generate_internal_debug(raw, der, decision)
    risk_html = build_risk_dashboard_html(risks)
    return doctor_txt, patient_txt, internal, risk_html, der, decision, risks


def serialize_case(raw: Dict[str, Any]) -> str:
    # Ensure JSON serializable
    return json.dumps(raw, ensure_ascii=False, indent=2, default=str)


def deserialize_case(s: str) -> Dict[str, Any]:
    return json.loads(s)


def save_case_to_file(raw: Dict[str, Any], filename: str = "rhk_case.json") -> str:
    content = serialize_case(raw)
    fd, path = tempfile.mkstemp(prefix="rhk_case_", suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
