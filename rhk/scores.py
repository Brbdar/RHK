# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .util import calc_age_years, fmt_num
from .calcs import calc_bmi


@dataclass
class RiskResult:
    name: str
    category: str  # "low"|"intermediate"|"high" or 4-strata
    score: Optional[float] = None
    details: List[str] = None  # type: ignore[assignment]


def _grade_3(value: Optional[float], low_hi: Tuple[float, float], invert: bool = False) -> Optional[int]:
    """
    Generic 3-tier grading:
      - low: >= high_cut if invert False else <= low_cut
      - intermediate: between
      - high: <= low_cut if invert False else >= high_cut
    low_hi = (low_cut, high_cut) boundary for intermediate.
    """
    if value is None:
        return None
    low_cut, high_cut = low_hi
    if not invert:
        if value >= high_cut:
            return 1
        if value <= low_cut:
            return 3
        return 2
    else:
        # lower is better
        if value <= low_cut:
            return 1
        if value >= high_cut:
            return 3
        return 2


def esc_ers_3_strata(params: Dict[str, Any]) -> RiskResult:
    """
    ESC/ERS follow-up 3-strata (simplified but guideline-aligned) using available:
    WHO-FC, 6MWD, BNP/NT-proBNP, RAP, CI, SvO2.
    Computes mean grade across available items.
    """
    details: List[str] = []
    grades: List[int] = []

    # WHO-FC
    fc = params.get("who_fc")
    if isinstance(fc, str):
        fc_norm = fc.strip().upper().replace("WHO", "").replace("FC", "").strip()
        mapping = {"I": 1, "II": 1, "III": 2, "IV": 3}
        g = mapping.get(fc_norm)
        if g:
            grades.append(g)
            details.append(f"WHO-FC {fc_norm}: Grad {g}")
    # 6MWD
    mwd = params.get("mwd")
    if mwd is not None:
        # >440 low, 165-440 intermediate, <165 high
        if mwd > 440:
            g = 1
        elif mwd < 165:
            g = 3
        else:
            g = 2
        grades.append(g)
        details.append(f"Gehtest: Grad {g}")
    # BNP / NT-proBNP
    bnp = params.get("bnp")
    nt = params.get("ntprobnp")
    if nt is not None:
        # <300 low, 300-1400 intermediate, >1400 high (rough)
        if nt < 300:
            g = 1
        elif nt > 1400:
            g = 3
        else:
            g = 2
        grades.append(g)
        details.append(f"NT-proBNP: Grad {g}")
    elif bnp is not None:
        # <50 low, 50-300 intermediate, >300 high (rough)
        if bnp < 50:
            g = 1
        elif bnp > 300:
            g = 3
        else:
            g = 2
        grades.append(g)
        details.append(f"BNP: Grad {g}")

    rap = params.get("rap")
    if rap is not None:
        if rap < 8:
            g = 1
        elif rap > 14:
            g = 3
        else:
            g = 2
        grades.append(g)
        details.append(f"RAP: Grad {g}")

    ci = params.get("ci")
    if ci is not None:
        if ci >= 2.5:
            g = 1
        elif ci < 2.0:
            g = 3
        else:
            g = 2
        grades.append(g)
        details.append(f"CI: Grad {g}")

    svo2 = params.get("svo2")
    if svo2 is not None:
        if svo2 > 65:
            g = 1
        elif svo2 < 60:
            g = 3
        else:
            g = 2
        grades.append(g)
        details.append(f"SvO₂: Grad {g}")

    if not grades:
        return RiskResult(name="ESC/ERS 3-Strata", category="unknown", score=None, details=["Zu wenig Daten für Risikostratifizierung."])

    mean_grade = sum(grades) / len(grades)
    # Round to nearest integer (1..3)
    overall = int(round(mean_grade))
    overall = max(1, min(3, overall))
    cat = {1: "low", 2: "intermediate", 3: "high"}[overall]
    return RiskResult(name="ESC/ERS 3-Strata", category=cat, score=mean_grade, details=details)


def esc_ers_4_strata(params: Dict[str, Any]) -> RiskResult:
    """
    ESC/ERS follow-up 4-strata (FC + 6MWD + BNP/NT-proBNP).
    """
    details: List[str] = []
    grades: List[int] = []

    # WHO-FC (I/II = 1, III = 3, IV = 4)
    fc = params.get("who_fc")
    if isinstance(fc, str):
        fc_norm = fc.strip().upper().replace("WHO", "").replace("FC", "").strip()
        mapping = {"I": 1, "II": 1, "III": 3, "IV": 4}
        g = mapping.get(fc_norm)
        if g:
            grades.append(g)
            details.append(f"WHO-FC {fc_norm}: Grad {g}")

    mwd = params.get("mwd")
    if mwd is not None:
        if mwd > 440:
            g = 1
        elif mwd >= 320:
            g = 2
        elif mwd >= 165:
            g = 3
        else:
            g = 4
        grades.append(g)
        details.append(f"Gehtest: Grad {g}")

    bnp = params.get("bnp")
    nt = params.get("ntprobnp")
    if nt is not None:
        if nt < 300:
            g = 1
        elif nt < 650:
            g = 2
        elif nt < 1100:
            g = 3
        else:
            g = 4
        grades.append(g)
        details.append(f"NT-proBNP: Grad {g}")
    elif bnp is not None:
        if bnp < 50:
            g = 1
        elif bnp < 200:
            g = 2
        elif bnp < 800:
            g = 3
        else:
            g = 4
        grades.append(g)
        details.append(f"BNP: Grad {g}")

    if not grades:
        return RiskResult(name="ESC/ERS 4-Strata", category="unknown", score=None, details=["Zu wenig Daten für Risikostratifizierung."])

    mean_grade = sum(grades) / len(grades)
    overall = int(round(mean_grade))
    overall = max(1, min(4, overall))
    cat = {1: "low", 2: "intermediate-low", 3: "intermediate-high", 4: "high"}[overall]
    return RiskResult(name="ESC/ERS 4-Strata", category=cat, score=mean_grade, details=details)


def reveal_lite2(params: Dict[str, Any]) -> RiskResult:
    """
    REVEAL 2.0 Lite (rough implementation):
      - WHO-FC
      - 6MWD
      - BNP/NT-proBNP
      - SBP
      - HR
      - eGFR/Creatinine
    This implementation is an approximation and should be used as decision-support only.
    """
    details: List[str] = []
    score = 0

    fc = (params.get("who_fc") or "").strip().upper()
    if "IV" in fc:
        score += 2; details.append("WHO-FC IV (+2)")
    elif "III" in fc:
        score += 1; details.append("WHO-FC III (+1)")

    mwd = params.get("mwd")
    if mwd is not None:
        if mwd < 165:
            score += 2; details.append("Gehtest <165 (+2)")
        elif mwd < 320:
            score += 1; details.append("Gehtest 165–319 (+1)")

    nt = params.get("ntprobnp")
    bnp = params.get("bnp")
    marker = nt if nt is not None else bnp
    if marker is not None:
        if nt is not None:
            if nt > 1100:
                score += 2; details.append("NT-proBNP hoch (+2)")
            elif nt > 650:
                score += 1; details.append("NT-proBNP mittel (+1)")
        else:
            if bnp > 800:
                score += 2; details.append("BNP hoch (+2)")
            elif bnp > 200:
                score += 1; details.append("BNP mittel (+1)")

    sbp = params.get("sbp")
    if sbp is not None and sbp < 110:
        score += 1; details.append("SBP <110 (+1)")

    hr = params.get("hr")
    if hr is not None and hr > 96:
        score += 1; details.append("HR >96 (+1)")

    egfr = params.get("egfr")
    if egfr is not None and egfr < 60:
        score += 1; details.append("eGFR <60 (+1)")

    # Category (very rough)
    if score <= 1:
        cat = "low"
    elif score <= 3:
        cat = "intermediate"
    else:
        cat = "high"
    return RiskResult(name="REVEAL Lite 2 (approx)", category=cat, score=float(score), details=details)


@dataclass
class H2FPEFResult:
    score: int
    category: str  # "unlikely"|"possible"|"likely"
    details: List[str]


def h2fpef(params: Dict[str, Any]) -> Optional[H2FPEFResult]:
    """
    H2FPEF score (Reddy et al.): Heavy, Hypertensive, AF, PH, Elder, Filling pressure.
    This requires echo data (PASP, E/e') + clinical inputs.
    """
    details: List[str] = []
    score = 0

    # Heavy (BMI > 30): +2
    bmi = params.get("bmi")
    if bmi is None:
        # try compute
        h = params.get("height_cm")
        w = params.get("weight_kg")
        bmi = calc_bmi(h, w)
    if bmi is not None and bmi > 30:
        score += 2; details.append("BMI >30 (+2)")

    # Hypertensive (≥2 antihypertensive meds): +1
    antihyp = params.get("antihypertensive_count")
    if antihyp is not None and antihyp >= 2:
        score += 1; details.append("≥2 Blutdruckmedikamente (+1)")

    # AF: +3
    rhythm = (params.get("rhythm") or "").lower()
    if "vorhofflim" in rhythm or "af" in rhythm:
        score += 3; details.append("Vorhofflimmern (+3)")

    # PH (echo PASP > 35): +1
    pasp = params.get("pasp")
    if pasp is not None and pasp > 35:
        score += 1; details.append("Echo: erhöhter syst. PAP (+1)")

    # Elder (age > 60): +1
    age = params.get("age_years")
    if age is None:
        age = calc_age_years(params.get("dob"))
    if age is not None and age > 60:
        score += 1; details.append("Alter >60 (+1)")

    # Filling pressure (E/e' > 9): +1
    ee = params.get("e_over_eprime")
    if ee is not None and ee > 9:
        score += 1; details.append("E/e' >9 (+1)")

    if not details:
        return None

    if score <= 1:
        cat = "unlikely"
    elif score <= 5:
        cat = "possible"
    else:
        cat = "likely"
    return H2FPEFResult(score=score, category=cat, details=details)


def cmr_rvef_risk(rvef: Optional[float]) -> Optional[RiskResult]:
    if rvef is None:
        return None
    # Heuristic categorisation
    if rvef >= 45:
        cat = "low"
    elif rvef >= 35:
        cat = "intermediate"
    else:
        cat = "high"
    return RiskResult(name="RV-Funktion (CMR RVEF)", category=cat, score=rvef, details=[f"RVEF {fmt_num(rvef,0)}%"])
