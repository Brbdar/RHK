# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .util import ValidationReport, fmt_num


@dataclass
class PHClassification:
    ph_present: Optional[bool]
    ph_type: str  # "none" | "precap" | "ipcph" | "cpcph" | "unclassifiable"
    label: str
    severity: Optional[str] = None  # "mild"|"moderate"|"severe"
    notes: List[str] = None  # type: ignore[assignment]


def classify_ph_rest(mpap: Optional[float], pawp: Optional[float], pvr: Optional[float], rules: Dict[str, Any]) -> PHClassification:
    """
    ESC/ERS 2022 definitions:
    - PH: mPAP > 20 mmHg
    - Pre-capillary PH: mPAP > 20, PAWP ≤ 15, PVR > 2 WU
    - IpcPH: mPAP > 20, PAWP > 15, PVR ≤ 2
    - CpcPH: mPAP > 20, PAWP > 15, PVR > 2
    """
    rest = (rules or {}).get("rest", {})
    mpap_cut = float(rest.get("mPAP_ph_mmHg", 20))
    pawp_post = float(rest.get("PAWP_postcap_mmHg", 15))
    pvr_precap = float(rest.get("PVR_precap_WU", 2.0))

    notes: List[str] = []
    if mpap is None:
        return PHClassification(None, "unclassifiable", "mPAP nicht erhoben", notes=notes)
    if mpap <= mpap_cut:
        return PHClassification(False, "none", "Keine PH in Ruhe (mPAP ≤ 20 mmHg)", notes=notes)

    # PH present
    if pawp is None or pvr is None:
        return PHClassification(True, "unclassifiable", "PH (mPAP > 20 mmHg), aber PAWP/PVR unvollständig", notes=notes)

    if pawp <= pawp_post and pvr > pvr_precap:
        return PHClassification(True, "precap", "Präkapilläre PH", notes=notes)
    if pawp > pawp_post and pvr <= pvr_precap:
        return PHClassification(True, "ipcph", "Isoliert postkapilläre PH (IpcPH)", notes=notes)
    if pawp > pawp_post and pvr > pvr_precap:
        return PHClassification(True, "cpcph", "Kombinierte prä-/postkapilläre PH (CpcPH)", notes=notes)

    # Edge cases: PH by mPAP, but doesn't meet full pre/post definitions
    notes.append("Grenzkonstellation: PH nach mPAP, aber Zuordnung prä/post nicht eindeutig.")
    return PHClassification(True, "unclassifiable", "PH (unklassifiziert)", notes=notes)


def pvr_severity(pvr: Optional[float], rules: Dict[str, Any]) -> Optional[str]:
    if pvr is None:
        return None
    sev = (rules or {}).get("severity", {}).get("PVR_WU", {})
    mild_ge = float(sev.get("mild_ge", 2.0))
    mod_ge = float(sev.get("moderate_ge", 5.0))
    sev_ge = float(sev.get("severe_ge", 10.0))
    if pvr >= sev_ge:
        return "severe"
    if pvr >= mod_ge:
        return "moderate"
    if pvr >= mild_ge:
        return "mild"
    return None


def ci_severity(ci: Optional[float], rules: Dict[str, Any]) -> Optional[str]:
    if ci is None:
        return None
    sev = (rules or {}).get("severity", {}).get("CI_L_min_m2", {})
    normal_ge = float(sev.get("normal_ge", 2.5))
    severely_reduced_lt = float(sev.get("severely_reduced_lt", 2.0))
    if ci < severely_reduced_lt:
        return "severely_reduced"
    if ci < normal_ge:
        return "reduced"
    return "normal"


def classify_exercise_pattern(
    mpap_co_slope: Optional[float],
    pawp_co_slope: Optional[float],
    rules: Dict[str, Any],
) -> Optional[str]:
    """
    Exercise-RHC heuristics (ESC/ERS / exercise PH concepts):
    - mPAP/CO slope > 3 mmHg/L/min suggests abnormal pulmonary pressure-flow relationship
    - PAWP/CO slope > 2 suggests a left-heart contribution
    Returns:
      "normal" | "left_heart" | "pulmonary_vascular" | "indeterminate"
    """
    if mpap_co_slope is None or pawp_co_slope is None:
        return None

    ex = (rules or {}).get("exercise", {})
    mpap_cut = float(ex.get("mPAP_CO_slope_mmHg_per_L_min", 3.0))
    pawp_cut = float(ex.get("PAWP_CO_slope_mmHg_per_L_min", 2.0))

    mpap_abn = mpap_co_slope > mpap_cut
    pawp_abn = pawp_co_slope > pawp_cut

    if not mpap_abn and not pawp_abn:
        return "normal"
    if mpap_abn and pawp_abn:
        return "left_heart"
    if mpap_abn and not pawp_abn:
        return "pulmonary_vascular"
    return "indeterminate"


def detect_step_up(sats: Dict[str, Optional[float]], threshold: float = 7.0) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Heuristic step-up detection. Expects keys among: 'SVC', 'IVC', 'RA', 'RV', 'PA'.
    Returns (present, location_str, delta_max).
    """
    order = ["SVC", "IVC", "RA", "RV", "PA"]
    vals = {k: sats.get(k) for k in order}
    # Use the max of SVC/IVC as "venous baseline"
    baseline = max([v for v in [vals.get("SVC"), vals.get("IVC")] if v is not None], default=None)
    if baseline is None:
        baseline = vals.get("SVC") or vals.get("IVC")

    best = (0.0, None)  # delta, location
    def consider(frm: str, to: str, v_from: Optional[float], v_to: Optional[float]):
        nonlocal best
        if v_from is None or v_to is None:
            return
        d = v_to - v_from
        if d > best[0]:
            best = (d, f"{frm} → {to}")

    # Baseline->RA, RA->RV, RV->PA
    consider("SVC/IVC", "RA", baseline, vals.get("RA"))
    consider("RA", "RV", vals.get("RA"), vals.get("RV"))
    consider("RV", "PA", vals.get("RV"), vals.get("PA"))

    present = best[0] >= threshold
    return present, best[1] if present else None, best[0] if present else None


@dataclass
class EtiologyHints:
    group2_possible: bool = False
    group3_possible: bool = False
    group4_possible: bool = False
    porto_possible: bool = False
    shunt_possible: bool = False
    notes: List[str] = None  # type: ignore[assignment]

    def summary_sentence(self) -> str:
        parts: List[str] = []
        if self.group2_possible:
            parts.append("Linksherz-Komponente (Gruppe 2) möglich")
        if self.group3_possible:
            parts.append("Lungen-/Hypoxie-Komponente (Gruppe 3) möglich")
        if self.group4_possible:
            parts.append("Thromboembolische Genese (Gruppe 4) möglich")
        if self.porto_possible:
            parts.append("Portopulmonaler Kontext möglich")
        if self.shunt_possible:
            parts.append("Shunt-Konstellation möglich")
        return "; ".join(parts) if parts else "Keine eindeutigen ätiologischen Hinweise aus den Zusatzangaben."


def infer_etiology_hints(flags: Dict[str, Any]) -> EtiologyHints:
    """
    flags expects booleans like:
      lae, ild, emphysema, embolism, mosaic_perfusion, portal_htn, step_up
    """
    notes: List[str] = []
    lae = bool(flags.get("lae"))
    ild = bool(flags.get("ild"))
    emph = bool(flags.get("emphysema"))
    embol = bool(flags.get("embolism"))
    mosaic = bool(flags.get("mosaic_perfusion"))
    portal = bool(flags.get("portal_htn"))
    step_up = bool(flags.get("step_up"))

    g2 = lae
    g3 = ild or emph or bool(flags.get("lufu_restrictive")) or bool(flags.get("lufu_obstructive")) or bool(flags.get("hypoxemia"))
    g4 = embol or mosaic or bool(flags.get("vq_defect"))
    porto = portal
    shunt = step_up or bool(flags.get("known_shunt"))

    # Add some combined-logic notes
    if lae and (ild or emph) and bool(flags.get("precap_hemo")):
        notes.append("Präkapilläre Hämodynamik trotz LAE + ILD/Emphysem: DD Gruppe 3 vs. Gruppe 4 (CTEPD/CTEPH) besonders beachten.")
    if g4 and not embol and mosaic:
        notes.append("Mosaikperfusion kann auf CTEPD/CTEPH hinweisen – V/Q-Szintigraphie wichtig.")
    return EtiologyHints(
        group2_possible=g2,
        group3_possible=g3,
        group4_possible=g4,
        porto_possible=porto,
        shunt_possible=shunt,
        notes=notes,
    )


def validate_key_values(data: Dict[str, Any]) -> ValidationReport:
    missing: List[str] = []
    warnings: List[str] = []

    def need(label: str, v: Optional[float]):
        if v is None:
            missing.append(label)

    need("mPAP", data.get("mpap"))
    need("PAWP", data.get("pawp"))
    # CO/CI: at least one
    if data.get("co") is None and data.get("ci") is None:
        missing.append("CO oder CI")

    # Plausibility checks
    mpap = data.get("mpap")
    pawp = data.get("pawp")
    rap = data.get("rap")
    if mpap is not None and (mpap < 5 or mpap > 80):
        warnings.append("mPAP wirkt unplausibel (außerhalb 5–80 mmHg).")
    if pawp is not None and (pawp < 1 or pawp > 40):
        warnings.append("PAWP wirkt unplausibel (außerhalb 1–40 mmHg).")
    if rap is not None and (rap < 0 or rap > 30):
        warnings.append("RAP wirkt unplausibel (außerhalb 0–30 mmHg).")

    return ValidationReport(missing=missing, warnings=warnings)
