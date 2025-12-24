#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rhk_textdb_engine.py

Engine-Funktionen für den RHK/PH-Befundassistenten:

- Ableitung zusätzlicher hämodynamischer Kenngrößen (z.B. PVR, CI, TPG, DPG, SV/SVI, PAC, PAPi)
- Hämodynamische Klassifikation (Ruhe + Belastung) nach ESC/ERS 2022 Definitionen
- Erweiterungen nach RHK-Kompendium (u.a. PAWP-Grauzone 13–18, transmural/Atmung, Wedge-Sättigung QC,
  Stepwise Oximetry Trigger/Schwellen, hämodynamische Risikomarker RAP/CI/SvO₂/SVI)
- Vorschlagslogik für K-Katalog-Bundles (K01..K20) + optionale Add-on-Block-Vorschläge (BZ..)

WICHTIG
- Diese Engine ist bewusst konservativ und transparent.
- Klinische Interpretation bleibt ärztliche Verantwortung.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", ".")
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "wahr", "yes", "ja", "y"}:
        return True
    if s in {"0", "false", "falsch", "no", "nein", "n"}:
        return False
    return None


def _get_first(data: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in data:
            v = _to_float(data.get(k))
            if v is not None:
                return v
    return None


def _get_first_bool(data: Dict[str, Any], keys: List[str]) -> Optional[bool]:
    for k in keys:
        if k in data:
            v = _to_bool(data.get(k))
            if v is not None:
                return v
    return None


def _fmt_num(x: Optional[float], decimals: int = 1) -> str:
    if x is None:
        return "n/a"
    try:
        s = f"{float(x):.{decimals}f}"
        if decimals > 0:
            s = s.rstrip("0").rstrip(".")
        return s.replace(".", ",")
    except Exception:
        return str(x)


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------

def derive_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Gibt eine *Kopie* von data zurück, ergänzt um abgeleitete Kennzahlen, wenn möglich.

    Erwartete Keys (Beispiele; Synonyme werden akzeptiert):
      - mPAP, PAWP, CO               -> PVR
      - CO, BSA                      -> CI
      - CO, HR                       -> SV (mL)
      - SV, BSA                      -> SVI (mL/m²)
      - mPAP, PAWP                   -> TPG
      - dPAP, PAWP                   -> DPG
      - sPAP, dPAP, SV               -> PAC (mL/mmHg)
      - sPAP, dPAP, RAP              -> PAPi
      - MAP, RAP, CO                 -> SVR (WU) + dyn*s/cm^5

    Es werden nur Werte ergänzt, die nicht bereits im Input vorhanden sind.
    """
    d = dict(data or {})

    # Pressures
    mPAP = _get_first(d, ["mPAP", "mPAP_rest", "mpap"])
    sPAP = _get_first(d, ["sPAP", "PAP_s", "PAP_sys", "PASP", "pap_s"])
    dPAP = _get_first(d, ["dPAP", "PAP_d", "PAP_dia", "padp", "pap_d"])
    PAWP = _get_first(d, ["PAWP", "PAWP_rest", "pawp", "PCWP", "pcwp"])
    RAP = _get_first(d, ["RAP", "RA_mean", "CVP", "zvd", "rap"])
    MAP = _get_first(d, ["MAP", "map", "mean_bp", "RR_mean"])

    # Flow / performance
    CO = _get_first(d, ["CO", "CO_rest", "co"])
    CI = _get_first(d, ["CI", "CI_rest", "ci"])
    HR = _get_first(d, ["HR", "heart_rate", "hf", "puls"])
    BSA = _get_first(d, ["BSA", "bsa", "KOE", "kof"])

    # Existing derived
    PVR = _get_first(d, ["PVR", "PVR_rest", "pvr"])
    TPG = _get_first(d, ["TPG", "tpg"])
    DPG = _get_first(d, ["DPG", "dpg"])
    SV = _get_first(d, ["SV_mL", "SV", "sv"])
    SVI = _get_first(d, ["SVI", "svi"])
    PAC = _get_first(d, ["PAC", "pac"])
    PAPi = _get_first(d, ["PAPi", "papi"])

    # PVR
    if PVR is None and mPAP is not None and PAWP is not None and CO is not None and CO > 0:
        d["PVR"] = (mPAP - PAWP) / CO

    # CI
    if CI is None and CO is not None and BSA is not None and BSA > 0:
        d["CI"] = CO / BSA

    # SV (mL)
    if SV is None and CO is not None and HR is not None and HR > 0:
        d["SV_mL"] = (CO * 1000.0) / HR

    # SVI (mL/m²)
    SV_eff = _get_first(d, ["SV_mL", "SV", "sv"])  # maybe just computed
    if SVI is None and SV_eff is not None and BSA is not None and BSA > 0:
        d["SVI"] = SV_eff / BSA

    # Gradients
    if TPG is None and mPAP is not None and PAWP is not None:
        d["TPG"] = (mPAP - PAWP)
    if DPG is None and dPAP is not None and PAWP is not None:
        d["DPG"] = (dPAP - PAWP)

    # Pulse pressure (PA)
    if sPAP is not None and dPAP is not None:
        d.setdefault("PA_pulse_pressure", (sPAP - dPAP))

    # PAC (mL/mmHg) ≈ SV / (sPAP - dPAP)
    pp = _get_first(d, ["PA_pulse_pressure"])  # if computed
    if PAC is None and SV_eff is not None and pp is not None and pp > 0:
        d["PAC"] = SV_eff / pp

    # PAPi = (sPAP-dPAP)/RAP
    if PAPi is None and pp is not None and RAP is not None and RAP != 0:
        d["PAPi"] = pp / RAP

    # SVR
    svr = _get_first(d, ["SVR_WU", "SVR", "svr"])
    if svr is None and MAP is not None and RAP is not None and CO is not None and CO > 0:
        d["SVR_WU"] = (MAP - RAP) / CO
        d["SVR_dyn_s_cm5"] = ((MAP - RAP) / CO) * 80.0

    # Simple ratios
    if RAP is not None and PAWP is not None and PAWP != 0:
        d.setdefault("RAP_PAWP_ratio", RAP / PAWP)

    return d


# ---------------------------------------------------------------------------
# QC / Uncertainty
# ---------------------------------------------------------------------------

def evaluate_wedge_saturation_qc(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Bewertet (optional) die Wedge-Sättigung als QC-Marker.

    Erwartete Keys (Synonyme):
      - wedge_sat / wedge_SaO2 / SaO2_wedge
      - SaO2 (systemisch)

    Rückgabe enthält u.a.:
      - available: bool
      - wedge_sat, sao2, delta_pp
      - confirmed_ok: bool|None
    """
    d = dict(data or {})
    w = _get_first(d, ["wedge_sat", "wedge_SaO2", "SaO2_wedge", "PCWP_SaO2", "PAWP_SaO2"])
    sao2 = _get_first(d, ["SaO2", "sao2", "SaO2_art", "art_SaO2"])

    vr = (rules or {}).get("wedge_saturation_validation", {}) or {}
    thr = float(vr.get("confirm_if_within_abs_percent_points_le", 5))
    alt_thr = float(vr.get("alternative_confirm_if_wedge_sat_ge_percent", 90))

    available = w is not None
    confirmed_ok: Optional[bool] = None
    delta: Optional[float] = None

    if w is not None and sao2 is not None:
        delta = abs(w - sao2)
        confirmed_ok = bool(delta <= thr)
    elif w is not None and sao2 is None:
        # without reference we can only give a weak heuristic
        confirmed_ok = True if w >= alt_thr else None

    return {
        "available": available,
        "wedge_sat": w,
        "SaO2": sao2,
        "delta_pp": delta,
        "confirmed_ok": confirmed_ok,
        "threshold_pp": thr,
        "alt_threshold_wedge_sat_ge": alt_thr,
    }


def evaluate_pawp_uncertainty(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Erkennt die PAWP-'Grauzone' und typische Confounder-Hinweise.

    - PAWP-Grauzone: typ. 13–18 mmHg (Kompendium)
    - optionale Confounder-Flags: respiratory_swings_large, obesity, copd, mechanical_ventilation
    - optional: Wedge-Sättigung QC
    """
    d = derive_metrics(data)
    pawp = _get_first(d, ["PAWP", "PAWP_rest", "pawp", "PCWP", "pcwp"])

    z = (rules or {}).get("pawp_uncertainty_zone", {}) or {}
    z_low = float(z.get("zone_low_mmHg", 13))
    z_high = float(z.get("zone_high_mmHg", 18))

    in_zone = None
    if pawp is not None:
        in_zone = bool(z_low <= pawp <= z_high)

    # optional confounder flags
    resp_swings = _get_first_bool(d, ["respiratory_swings_large", "large_respiratory_swings", "big_resp_variation"])
    obesity = _get_first_bool(d, ["obesity", "adipositas", "bmi_gt_30"])
    copd = _get_first_bool(d, ["copd", "COPD", "auto_peep", "autoPEEP"])
    vent = _get_first_bool(d, ["mechanical_ventilation", "ventilated", "beatmet", "PEEP_present"])

    qc = evaluate_wedge_saturation_qc(d, rules)

    reasons: List[str] = []
    if in_zone is True:
        reasons.append("PAWP in Grauzone 13–18 mmHg")
    if resp_swings is True:
        reasons.append("große respiratorische Druckschwankungen")
    if obesity is True:
        reasons.append("Adipositas/erhöhter pleuraler Grunddruck möglich")
    if copd is True:
        reasons.append("COPD/Auto-PEEP als Confounder möglich")
    if vent is True:
        reasons.append("Beatmung/PEEP als Confounder möglich")
    if qc.get("available") and qc.get("confirmed_ok") is False:
        reasons.append("Wedge-Sättigung deutlich unter SaO₂ (Under-wedge möglich)")

    uncertain = bool(reasons) if pawp is not None else False

    return {
        "PAWP": pawp,
        "in_uncertainty_zone": in_zone,
        "respiratory_swings_large": resp_swings,
        "obesity_flag": obesity,
        "copd_flag": copd,
        "ventilation_flag": vent,
        "wedge_sat_qc": qc,
        "uncertain": uncertain,
        "reasons": reasons,
        "zone_low": z_low,
        "zone_high": z_high,
    }


# ---------------------------------------------------------------------------
# Oximetry / Shunt
# ---------------------------------------------------------------------------

def detect_step_up(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Versucht aus Sättigungen einen Step-up (Links-Rechts-Shunt-Verdacht) zu erkennen.

    Unterstützte Keys (Synonyme):
      - Sat_SVC / SVC_Sat / SVC_SaO2
      - Sat_IVC / IVC_Sat / IVC_SaO2 (optional)
      - Sat_RA / RA_Sat / RA_SaO2
      - Sat_RV / RV_Sat / RV_SaO2
      - Sat_PA / PA_Sat / PA_SaO2  (oft identisch mit SvO2)
      - SvO2

    Regeln (Kompendium):
      - Trigger: SvO2 >75% -> stepwise oximetry erwägen
      - Signifikanter Step-up: ≥7 %-Punkte Vorhof-Ebene, ≥5 %-Punkte Ventrikel/PA-Ebene

    Wenn keine Sättigungen vorhanden sind, wird auf data.step_up_present/has_shunt zurückgegriffen.
    """
    d = dict(data or {})

    # explicit flag
    explicit = _get_first_bool(d, ["step_up_present", "has_shunt", "shunt_present"])

    svc = _get_first(d, ["Sat_SVC", "SVC_Sat", "SVC_SaO2", "sao2_svc"])
    ivc = _get_first(d, ["Sat_IVC", "IVC_Sat", "IVC_SaO2", "sao2_ivc"])
    ra = _get_first(d, ["Sat_RA", "RA_Sat", "RA_SaO2", "sao2_ra"])
    rv = _get_first(d, ["Sat_RV", "RV_Sat", "RV_SaO2", "sao2_rv"])
    pa = _get_first(d, ["Sat_PA", "PA_Sat", "PA_SaO2", "sao2_pa"])
    svo2 = _get_first(d, ["SvO2", "svo2"])

    # If SvO2 not explicitly provided, PA sat is commonly the SvO2.
    if svo2 is None and pa is not None:
        svo2 = pa

    st_rules = (rules or {}).get("stepwise_oximetry", {}) or {}
    thr = (st_rules.get("stepup_thresholds_percent_points") or {})
    thr_atrial = float(thr.get("atrial_ge", 7))
    thr_vent = float(thr.get("ventricular_or_pa_ge", 5))

    # Mixed venous reference (very pragmatic): if both SVC and IVC available, use weighted mix (3:1).
    mixed_venous = None
    if svc is not None and ivc is not None:
        mixed_venous = (3.0 * svc + 1.0 * ivc) / 4.0
    elif svc is not None:
        mixed_venous = svc
    elif ivc is not None:
        mixed_venous = ivc

    deltas: List[Tuple[str, Optional[float], float]] = []  # (label, delta, threshold)
    if mixed_venous is not None and ra is not None:
        deltas.append(("SVC/IVC→RA", ra - mixed_venous, thr_atrial))
    if ra is not None and rv is not None:
        deltas.append(("RA→RV", rv - ra, thr_vent))
    if rv is not None and pa is not None:
        deltas.append(("RV→PA", pa - rv, thr_vent))

    step_up_present: Optional[bool] = None
    step_up_from_to: Optional[str] = None
    step_up_max_pp: Optional[float] = None

    if deltas:
        # Earliest 'positive' in physiological sequence wins; still report max.
        max_abs = None
        for lbl, dv, _thr in deltas:
            if dv is None:
                continue
            if max_abs is None or abs(dv) > max_abs:
                max_abs = abs(dv)
        step_up_max_pp = max_abs

        positive_labels = []
        for lbl, dv, _thr in deltas:
            if dv is None:
                continue
            if dv >= _thr:
                positive_labels.append(lbl)

        if positive_labels:
            step_up_present = True
            step_up_from_to = positive_labels[0]
        else:
            step_up_present = False

    if step_up_present is None:
        # fall back to explicit flag
        step_up_present = explicit

    trigger = None
    if svo2 is not None:
        trig_thr = float((st_rules.get("trigger") or {}).get("SvO2_gt_percent", 75))
        trigger = bool(svo2 > trig_thr)

    return {
        "SvO2": svo2,
        "trigger_svo2_gt_75": trigger,
        "step_up_present": step_up_present,
        "step_up_from_to": step_up_from_to,
        "step_up_max_pp": step_up_max_pp,
        "thresholds_pp": {"atrial": thr_atrial, "ventricular_or_pa": thr_vent},
        "sats": {
            "SVC": svc,
            "IVC": ivc,
            "RA": ra,
            "RV": rv,
            "PA": pa,
        },
    }


# ---------------------------------------------------------------------------
# Risk (hemodynamic subset)
# ---------------------------------------------------------------------------

def _bin_3level(value: Optional[float], low_cond: Tuple[str, float], mid_range: Tuple[float, float], high_cond: Tuple[str, float]) -> Optional[str]:
    """Very small helper to classify into low/intermediate/high by thresholds."""
    if value is None:
        return None

    low_op, low_thr = low_cond
    high_op, high_thr = high_cond

    def _cmp(v: float, op: str, thr: float) -> bool:
        if op == "lt":
            return v < thr
        if op == "le":
            return v <= thr
        if op == "gt":
            return v > thr
        if op == "ge":
            return v >= thr
        return False

    v = float(value)

    if _cmp(v, low_op, low_thr):
        return "low"
    if _cmp(v, high_op, high_thr):
        return "high"

    lo, hi = mid_range
    if lo <= v <= hi:
        return "intermediate"

    # fallback
    return "intermediate"


def pah_hemodynamic_risk(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Hämodynamische Teil-Risikomarker (PAH-Kontext): RAP/CI/SvO₂/SVI.

    Quelle der Cut-offs: im System als rules.pah_hemodynamic_risk_bins hinterlegt.

    Rückgabe enthält Einzelbins + 'overall' (worst-of).
    """
    d = derive_metrics(data)

    bins = (rules or {}).get("pah_hemodynamic_risk_bins", {}) or {}

    rap = _get_first(d, ["RAP", "RA_mean", "CVP", "zvd", "rap"])
    ci = _get_first(d, ["CI", "CI_rest", "ci"])
    svo2 = _get_first(d, ["SvO2", "svo2", "Sat_PA", "PA_Sat", "PA_SaO2"])
    svi = _get_first(d, ["SVI", "svi"])

    rap_bins = bins.get("RAP_mmHg", {}) or {}
    ci_bins = bins.get("CI_L_min_m2", {}) or {}
    svo2_bins = bins.get("SvO2_percent", {}) or {}
    svi_bins = bins.get("SVI_mL_m2", {}) or {}

    rap_bin = _bin_3level(
        rap,
        ("lt", float(rap_bins.get("low_lt", 8))),
        tuple(map(float, rap_bins.get("intermediate_range", [8, 14]))),
        ("gt", float(rap_bins.get("high_gt", 14))),
    )

    ci_bin = _bin_3level(
        ci,
        ("ge", float(ci_bins.get("low_ge", 2.5))),
        tuple(map(float, ci_bins.get("intermediate_range", [2.0, 2.4]))),
        ("lt", float(ci_bins.get("high_lt", 2.0))),
    )

    svo2_bin = _bin_3level(
        svo2,
        ("gt", float(svo2_bins.get("low_gt", 65))),
        tuple(map(float, svo2_bins.get("intermediate_range", [60, 65]))),
        ("lt", float(svo2_bins.get("high_lt", 60))),
    )

    svi_bin = _bin_3level(
        svi,
        ("gt", float(svi_bins.get("low_gt", 38))),
        tuple(map(float, svi_bins.get("intermediate_range", [31, 38]))),
        ("lt", float(svi_bins.get("high_lt", 31))),
    )

    # overall = worst-of (high > intermediate > low)
    order = {"low": 0, "intermediate": 1, "high": 2}
    overall = None
    for b in [rap_bin, ci_bin, svo2_bin, svi_bin]:
        if b is None:
            continue
        if overall is None or order.get(b, 0) > order.get(overall, 0):
            overall = b

    return {
        "RAP": rap,
        "CI": ci,
        "SvO2": svo2,
        "SVI": svi,
        "bins": {"RAP": rap_bin, "CI": ci_bin, "SvO2": svo2_bin, "SVI": svi_bin},
        "overall": overall,
        "available": any(v is not None for v in [rap, ci, svo2, svi]),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_rest_hemodynamics(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Klassifikation nach ESC/ERS 2022 (Table 5).

    Rückgabe:
      {
        "has_ph": bool|None,
        "rest_class": "no_ph"|"precap_ph"|"ipcph"|"cpcph"|"unclassified_ph"|"unknown",
        "mPAP": float|None, "PAWP": float|None, "PVR": float|None, "CI": float|None,
      }

    Zusätzlich:
      - "pawp_uncertainty": dict (Grauzone/QC-Flags)
    """
    d = derive_metrics(data)
    mPAP = _get_first(d, ["mPAP", "mPAP_rest", "mpap"])
    PAWP = _get_first(d, ["PAWP", "PAWP_rest", "pawp", "PCWP", "pcwp"])
    PVR = _get_first(d, ["PVR", "PVR_rest", "pvr"])
    CI = _get_first(d, ["CI", "CI_rest", "ci"])
    CO = _get_first(d, ["CO", "CO_rest", "co"])

    defs = (rules or {}).get("hemodynamic_definitions", {})
    ph_thr = float(((defs.get("ph") or {}).get("mPAP_gt_mmHg")) or 20)
    pawp_thr = float(((defs.get("postcapillary_ph") or {}).get("PAWP_gt_mmHg")) or 15)
    pvr_thr = float(((defs.get("precapillary_ph") or {}).get("PVR_gt_WU")) or 2)

    has_ph = None
    if mPAP is not None:
        has_ph = bool(mPAP > ph_thr)

    rest_class = "unknown"
    if has_ph is False:
        rest_class = "no_ph"
    elif has_ph is True:
        pawp_high = (PAWP is not None and PAWP > pawp_thr)
        pvr_high = (PVR is not None and PVR > pvr_thr)

        if pawp_high:
            rest_class = "cpcph" if pvr_high else "ipcph"
        else:
            # PAWP ≤ 15
            if pvr_high:
                rest_class = "precap_ph"
            else:
                rest_class = "unclassified_ph"

    # hyperdynamic flag (optional)
    hyperdynamic = False
    if CI is not None and CI > 3.5:
        hyperdynamic = True
    if CO is not None and CO > 8:
        hyperdynamic = True

    return {
        "has_ph": has_ph,
        "rest_class": rest_class,
        "hyperdynamic": hyperdynamic,
        "mPAP": mPAP,
        "PAWP": PAWP,
        "PVR": PVR,
        "CI": CI,
        "CO": CO,
        "pawp_uncertainty": evaluate_pawp_uncertainty(d, rules),
    }


def classify_exercise(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Belastungs-Klassifikation anhand Slope-Kriterien (ESC/ERS 2022: mPAP/CO slope >3).

    Zusätzlich:
      - PAWP/CO slope >2 spricht eher postkapillär.
    """
    d = dict(data or {})
    mpap_slope = _get_first(d, ["mPAP_CO_slope", "mpap_co_slope"])
    pawp_slope = _get_first(d, ["PAWP_CO_slope", "pawp_co_slope"])

    ex_rules = (rules or {}).get("exercise_definitions", {})
    thr_mpap = float((ex_rules.get("exercise_ph") or {}).get("mPAP_CO_slope_gt_mmHg_per_L_min", 3))
    thr_pawp = float((ex_rules.get("postcapillary_cause_suspected") or {}).get("PAWP_CO_slope_gt_mmHg_per_L_min", 2))

    has_exercise = (mpap_slope is not None) or (pawp_slope is not None)
    exercise_ph = None
    postcap_suspected = None
    if mpap_slope is not None:
        exercise_ph = bool(mpap_slope > thr_mpap)
    if pawp_slope is not None:
        postcap_suspected = bool(pawp_slope > thr_pawp)

    classification = "unknown"
    if not has_exercise:
        classification = "no_exercise_data"
    else:
        if exercise_ph is False:
            classification = "exercise_normal"
        elif exercise_ph is True:
            if postcap_suspected is True:
                classification = "exercise_ph_postcap_suspected"
            elif postcap_suspected is False:
                classification = "exercise_ph_pulmvascular_suspected"
            else:
                classification = "exercise_ph_unclassified"

    return {
        "has_exercise": has_exercise,
        "mPAP_CO_slope": mpap_slope,
        "PAWP_CO_slope": pawp_slope,
        "exercise_ph": exercise_ph,
        "postcap_suspected": postcap_suspected,
        "exercise_class": classification,
    }


def vasoreactivity(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Akute Vasoreaktivität (typisch iNO): positive Antwort nach ESC/ERS:

      - mPAP Abfall ≥10 mmHg auf ≤40 mmHg
      - CO unverändert oder erhöht

    Erwartete (optionale) Keys:
      mPAP_iNO_pre, mPAP_iNO_post, CO_iNO_pre, CO_iNO_post
    """
    d = dict(data or {})
    m_pre = _get_first(d, ["mPAP_iNO_pre", "mPAP_pre_iNO", "mPAP_pre"])
    m_post = _get_first(d, ["mPAP_iNO_post", "mPAP_post_iNO", "mPAP_post"])
    co_pre = _get_first(d, ["CO_iNO_pre", "CO_pre_iNO", "CO_pre"])
    co_post = _get_first(d, ["CO_iNO_post", "CO_post_iNO", "CO_post"])

    vr = (rules or {}).get("vasoreactivity", {}).get("positive_response", {}) or {}
    drop_thr = float(vr.get("mPAP_drop_ge_mmHg", 10))
    target_thr = float(vr.get("mPAP_target_le_mmHg", 40))

    done = (m_pre is not None and m_post is not None)

    responder = None
    if done:
        drop_ok = (m_pre - m_post) >= drop_thr
        target_ok = m_post <= target_thr
        co_ok = True
        if co_pre is not None and co_post is not None:
            co_ok = (co_post >= co_pre)
        responder = bool(drop_ok and target_ok and co_ok)

    statement = f"mPAP-Abfall ≥{int(drop_thr)} mmHg auf ≤{int(target_thr)} mmHg bei unverändertem/steigendem HZV"

    return {
        "done": done,
        "mPAP_pre": m_pre,
        "mPAP_post": m_post,
        "CO_pre": co_pre,
        "CO_post": co_post,
        "responder": responder,
        "responder_statement": statement,
    }


def fluid_challenge(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Volumenbelastung: CHEST (D'Alto et al.) diskutiert PAWP >18 mmHg nach 500 mL als Cutoff.

    Erwartete Keys:
      PAWP_pre, PAWP_post
    """
    d = dict(data or {})
    pawp_pre = _get_first(d, ["PAWP_pre", "PAWP_fluid_pre", "PAWP_rest", "PAWP"])
    pawp_post = _get_first(d, ["PAWP_post", "PAWP_fluid_post"])

    fc = (rules or {}).get("fluid_challenge", {}).get("positive_cutoff", {}) or {}
    thr = float(fc.get("PAWP_post_gt_mmHg", 18))

    done = (pawp_pre is not None and pawp_post is not None)
    positive = None
    if done:
        positive = bool(pawp_post > thr)

    return {
        "done": done,
        "PAWP_pre": pawp_pre,
        "PAWP_post": pawp_post,
        "positive": positive,
        "cutoff_mmHg": thr,
    }


# ---------------------------------------------------------------------------
# Context building (for templates)
# ---------------------------------------------------------------------------

def build_context(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Erzeugt eine Mapping-Dict, das direkt in Templates gerendert werden kann.

    Fokus: neue v3-Platzhalter (TPG/DPG/PAC/SVI/PAPi + Wedge-Sättigung QC + hemodyn. Risiko).

    Die Funktion ist absichtlich "addon-orientiert" und ersetzt nicht die komplette GUI-Logik.
    """
    d = derive_metrics(data)

    # Derived values
    tpg = _get_first(d, ["TPG", "tpg"])
    dpg = _get_first(d, ["DPG", "dpg"])
    pac = _get_first(d, ["PAC", "pac"])
    svi = _get_first(d, ["SVI", "svi"])
    papi = _get_first(d, ["PAPi", "papi"])

    def _phrase(label: str, value: Optional[float], unit: str, dec: int = 1) -> str:
        if value is None:
            return f"{label} nicht berechenbar"
        return f"{label} {_fmt_num(value, dec)} {unit}".strip()

    context: Dict[str, Any] = dict(d)
    context["tpg_phrase"] = _phrase("TPG", tpg, "mmHg", 0)
    context["dpg_phrase"] = _phrase("DPG", dpg, "mmHg", 0)
    context["pac_phrase"] = _phrase("PAC", pac, "mL/mmHg", 1)
    context["svi_phrase"] = _phrase("SVI", svi, "mL/m²", 0)
    context["papi_phrase"] = _phrase("PAPi", papi, "", 2)

    # Wedge-sat QC sentence
    qc = evaluate_wedge_saturation_qc(d, rules)
    if qc.get("available"):
        w = qc.get("wedge_sat")
        sao2 = qc.get("SaO2")
        delta = qc.get("delta_pp")
        ok = qc.get("confirmed_ok")
        if sao2 is not None and delta is not None:
            if ok is True:
                context["wedge_sat_qc_sentence"] = (
                    f"Wedge-O₂-Sättigung {_fmt_num(w,0)}% nahe an SaO₂ {_fmt_num(sao2,0)}% (Δ {_fmt_num(delta,0)} %-Punkte) – "
                    "Okklusion plausibel, PAWP-Messung unterstützt."
                )
            else:
                context["wedge_sat_qc_sentence"] = (
                    f"Wedge-O₂-Sättigung {_fmt_num(w,0)}% deutlich unter SaO₂ {_fmt_num(sao2,0)}% (Δ {_fmt_num(delta,0)} %-Punkte) – "
                    "Hinweis auf Under-wedge/Leckage möglich; PAWP-Wert mit Vorsicht interpretieren."
                )
        else:
            # no reference SaO2
            if ok is True:
                context["wedge_sat_qc_sentence"] = (
                    f"Wedge-O₂-Sättigung {_fmt_num(w,0)}% – arterialisiertes Muster spricht für okklusive Wedge-Position; "
                    "Interpretation idealerweise relativ zu parallel gemessener SaO₂."
                )
            else:
                context["wedge_sat_qc_sentence"] = (
                    f"Wedge-O₂-Sättigung {_fmt_num(w,0)}% – ohne parallele SaO₂ nur eingeschränkt beurteilbar; "
                    "bei Zweifel Wedge-Position/Okklusion erneut prüfen."
                )
    else:
        context["wedge_sat_qc_sentence"] = "Keine Wedge-Sättigung dokumentiert."

    # Hemodynamic risk sentence
    risk = pah_hemodynamic_risk(d, rules)
    if risk.get("available"):
        bins = risk.get("bins", {})
        def _bin_de(b: Optional[str]) -> str:
            return {"low": "niedrig", "intermediate": "intermediär", "high": "hoch"}.get(b or "", "n/a")

        parts = []
        if risk.get("RAP") is not None:
            parts.append(f"RAP {_fmt_num(risk['RAP'],0)} mmHg ({_bin_de(bins.get('RAP'))})")
        if risk.get("CI") is not None:
            parts.append(f"CI {_fmt_num(risk['CI'],1)} L/min/m² ({_bin_de(bins.get('CI'))})")
        if risk.get("SvO2") is not None:
            parts.append(f"SvO₂ {_fmt_num(risk['SvO2'],0)}% ({_bin_de(bins.get('SvO2'))})")
        if risk.get("SVI") is not None:
            parts.append(f"SVI {_fmt_num(risk['SVI'],0)} mL/m² ({_bin_de(bins.get('SVI'))})")

        overall = _bin_de(risk.get("overall"))
        context["hemo_risk_sentence"] = (
            "Hämodynamische Risikomarker (PAH-Kontext, Teilparameter): "
            + ", ".join(parts)
            + f". Gesamt (worst-of): {overall}."
        )
    else:
        context["hemo_risk_sentence"] = "Hämodynamische Risikomarker (PAH-Kontext): nicht ausreichend Daten für RAP/CI/SvO₂/SVI."

    return context


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def suggest_k_bundles(data: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
    """Gibt eine Liste von K-Katalog-IDs zurück (z.B. ["K01"]).

    Die Logik ist bewusst "baseline" gedacht: sie soll in der GUI als Vorschlag erscheinen.
    v3-Erweiterungen:
      - Step-up-Erkennung aus Sättigungen (falls vorhanden)
      - PAWP-Grauzone/QC: bei Unsicherheit zusätzlich K04 als Alternativvorschlag
      - Präkapillär: optional stärkere Gewichtung von RAP/CI/SvO₂/SVI (hemodyn. Risiko) für den Schweregrad
    """
    rest = classify_rest_hemodynamics(data, rules)
    ex = classify_exercise(data, rules)
    vr = vasoreactivity(data, rules)
    fc = fluid_challenge(data, rules)
    step = detect_step_up(data, rules)

    # 1) Shunt-Konstellation schlägt vieles
    has_step_up = bool(step.get("step_up_present"))
    if has_step_up:
        return ["K16"]

    # 2) Vasoreaktivitätstest vorhanden
    if vr.get("done") and vr.get("responder") is not None:
        return ["K17" if vr["responder"] else "K18"]

    # 3) Volumenbelastung als Hinweis auf HFpEF
    if fc.get("done") and fc.get("positive") is True:
        if rest.get("rest_class") == "no_ph":
            return ["K02"]
        if rest.get("rest_class") == "ipcph":
            return ["K14"]
        if rest.get("rest_class") == "cpcph":
            return ["K15"]

    # 4) Belastungsdaten
    if ex.get("has_exercise") and ex.get("exercise_ph") is not None:
        if rest.get("rest_class") == "no_ph":
            if ex["exercise_ph"] is False:
                return ["K02"] if fc.get("positive") else ["K01"]
            if ex["exercise_ph"] is True:
                if ex.get("postcap_suspected") is True:
                    return ["K02"]
                if ex.get("postcap_suspected") is False:
                    return ["K03"]
                return ["K03"]

    # 5) Ruheklassifikation
    rc = rest.get("rest_class")
    pawp_unc = (rest.get("pawp_uncertainty") or {})
    pawp_uncertain = bool(pawp_unc.get("uncertain"))

    if rc == "no_ph":
        return ["K01"]

    if rc == "unclassified_ph":
        return ["K12"] if rest.get("hyperdynamic") else ["K04"]

    if rc in {"ipcph", "cpcph"}:
        # In der PAWP-Grauzone/QC-Unklarheit: zusätzlich K04 als Alternative vorschlagen
        primary = "K14" if rc == "ipcph" else "K15"
        if pawp_uncertain:
            return ["K04", primary]
        return [primary]

    if rc == "precap_ph":
        # Schweregrad pragmatisch über PVR/CI, optional ergänzt um hemodyn. Risiko (RAP/CI/SvO2/SVI)
        PVR = rest.get("PVR")
        CI = rest.get("CI")

        # severity via classic rules
        sev = "K05"
        if CI is not None and CI < 2.0:
            sev = "K07"
        elif PVR is not None:
            if PVR >= 10:
                sev = "K07"
            elif PVR >= 5:
                sev = "K06"
            else:
                sev = "K05"

        # severity via risk (worst-of)
        risk = pah_hemodynamic_risk(data, rules)
        r_overall = risk.get("overall")
        if r_overall == "high":
            sev_r = "K07"
        elif r_overall == "intermediate":
            sev_r = "K06"
        elif r_overall == "low":
            sev_r = "K05"
        else:
            sev_r = None

        if sev_r is not None:
            # choose the more severe of the two
            order = {"K05": 0, "K06": 1, "K07": 2}
            sev = sev_r if order.get(sev_r, 0) > order.get(sev, 0) else sev

        return [sev]

    return ["K04"]


def suggest_addon_blocks(data: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
    """Vorschläge für zusätzliche (optionale) Add-on-Blöcke, v.a. Methodik/QC/Advanced.

    Rückgabe ist eine Liste von Block-IDs (z.B. ["BZ14_PAWP_GREYZONE_ALGO", ...]).

    Die GUI kann diese als "Empfohlene Zusatzbausteine" anzeigen.
    """
    d = derive_metrics(data)
    rest = classify_rest_hemodynamics(d, rules)
    pawp_unc = (rest.get("pawp_uncertainty") or {})
    step = detect_step_up(d, rules)
    risk = pah_hemodynamic_risk(d, rules)

    addons: List[str] = []

    # PAWP Grauzone / Confounder
    if pawp_unc.get("in_uncertainty_zone") is True or pawp_unc.get("uncertain") is True:
        addons.append("BZ14_PAWP_GREYZONE_ALGO")

    if any(
        pawp_unc.get(k) is True
        for k in ["respiratory_swings_large", "obesity_flag", "copd_flag", "ventilation_flag"]
    ):
        addons.append("BZ12_TRANSMURAL_THORAX")
        addons.append("BZ13_PAWP_RESP_MEAN")

    # Wedge saturation QC
    if evaluate_wedge_saturation_qc(d, rules).get("available"):
        addons.append("BZ15_WEDGE_SAT_QC")

    # Stepwise oximetry / triggers
    if step.get("trigger_svo2_gt_75") is True or step.get("step_up_present") is True:
        addons.append("BZ16_STEPWISE_OXIMETRY")

    # Advanced derivations
    if any(_get_first(d, [k]) is not None for k in ["TPG", "DPG", "PAC", "SVI", "PAPi"]):
        addons.append("BZ17_ADVANCED_DERIVATIONS")
        addons.append("BZ19_PULSATILE_LOAD_PAC")

    # Hemodynamic risk in precap/PAH context
    if rest.get("rest_class") == "precap_ph" and risk.get("available"):
        addons.append("BZ18_PAH_HEMO_RISK")

    # unique + stable order
    seen = set()
    out: List[str] = []
    for a in addons:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def suggest_plan(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Gibt ein strukturiertes Suggestion-Objekt zurück.

    Enthält:
      - k_bundles: Ergebnis aus suggest_k_bundles
      - addon_blocks: Ergebnis aus suggest_addon_blocks
      - rest/exercise/vasoreactivity/fluid_challenge: Engine-Outputs
      - step_up / risk / pawp_uncertainty
      - context: build_context() für v3-Add-ons

    Diese Funktion ist optional, aber für GUI-Integration sehr praktisch.
    """
    rest = classify_rest_hemodynamics(data, rules)
    ex = classify_exercise(data, rules)
    vr = vasoreactivity(data, rules)
    fc = fluid_challenge(data, rules)
    step = detect_step_up(data, rules)
    risk = pah_hemodynamic_risk(data, rules)

    return {
        "k_bundles": suggest_k_bundles(data, rules),
        "addon_blocks": suggest_addon_blocks(data, rules),
        "rest": rest,
        "exercise": ex,
        "vasoreactivity": vr,
        "fluid_challenge": fc,
        "step_up": step,
        "risk": risk,
        "pawp_uncertainty": rest.get("pawp_uncertainty"),
        "context": build_context(data, rules),
    }
