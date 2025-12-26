# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import rhk_textdb as textdb

from .calcs import (
    calc_bsa_dubois,
    calc_ci,
    calc_dpg,
    calc_mean,
    calc_pvr,
    calc_pvri,
    calc_slope,
    calc_tpg,
    calc_delta_spap,
    calc_ci_peak,
    sprime_raai,
    exercise_rv_adaptation,
)
from .classify import (
    PHClassification,
    classify_exercise_pattern,
    classify_ph_rest,
    ci_severity,
    detect_step_up,
    infer_etiology_hints,
    pvr_severity,
    validate_key_values,
)
from .scores import (
    RiskResult,
    esc_ers_3_strata,
    esc_ers_4_strata,
    reveal_lite2,
    h2fpef,
    cmr_rvef_risk,
)
from .util import SafeDict, ValidationReport, calc_age_years, clean_sentence, fmt_num, fmt_unit, join_nonempty, parse_date_yyyy_mm_dd, to_float, to_int


@dataclass
class Computed:
    # Basic derived
    bsa: Optional[float]
    bmi: Optional[float]
    age: Optional[int]

    # Hemodynamics
    mpap: Optional[float]
    pawp: Optional[float]
    rap: Optional[float]
    spap: Optional[float]
    dpap: Optional[float]
    tpg: Optional[float]
    dpg: Optional[float]

    co: Optional[float]
    ci: Optional[float]
    pvr: Optional[float]
    pvri: Optional[float]

    # Exercise
    exercise_done: bool
    mpap_co_slope: Optional[float]
    pawp_co_slope: Optional[float]
    delta_spap: Optional[float]
    ci_peak: Optional[float]
    rv_adaptation: Optional[str]

    # Step-up
    step_up_present: bool
    step_up_location: Optional[str]
    step_up_delta: Optional[float]

    # Echo add-ons
    sprime_raai_value: Optional[float]
    sprime_raai_cutoff: Optional[float]
    sprime_raai_interpretation: Optional[str]

    # Classification
    ph: PHClassification
    pvr_sev: Optional[str]
    ci_sev: Optional[str]
    exercise_pattern: Optional[str]
    etiology_hints: Any

    # Scores
    esc3: RiskResult
    esc4: RiskResult
    reveal: RiskResult
    hfpef: Any
    cmr_rvef: Optional[RiskResult]

    # Validation
    validation: ValidationReport


def _pick_co(ui: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Choose a CO value based on availability. Preference:
      1) TD-CO, else 2) Fick-CO, else None.
    """
    co_td = to_float(ui.get("co_td"))
    co_fick = to_float(ui.get("co_fick"))
    if co_td is not None:
        return co_td, "Thermodilution"
    if co_fick is not None:
        return co_fick, "Fick"
    return None, "—"


def compute_all(ui: Dict[str, Any], rules: Optional[Dict[str, Any]] = None) -> Computed:
    rules = rules or textdb.DEFAULT_RULES

    # Patient basics
    height = to_float(ui.get("height_cm"))
    weight = to_float(ui.get("weight_kg"))
    bsa = calc_bsa_dubois(height, weight).value
    bmi = None
    if height and weight:
        bmi = weight / ((height/100.0)**2) if height > 0 else None

    dob = parse_date_yyyy_mm_dd(ui.get("birthdate"))
    age = calc_age_years(dob)

    # Hemodynamics: allow mPAP to be computed from sPAP/dPAP if missing
    spap = to_float(ui.get("spap"))
    dpap = to_float(ui.get("dpap"))
    mpap = to_float(ui.get("mpap"))
    if mpap is None:
        mpap = calc_mean(spap, dpap).value

    pawp = to_float(ui.get("pawp"))
    rap = to_float(ui.get("rap"))

    # CO/CI
    ci_direct = to_float(ui.get("ci"))
    co, co_method = _pick_co(ui)

    if ci_direct is None and co is not None and bsa is not None:
        ci_direct = calc_ci(co, bsa).value
    if co is None and ci_direct is not None and bsa is not None:
        co = ci_direct * bsa

    ci_val = ci_direct
    pvr_val = to_float(ui.get("pvr"))
    if pvr_val is None:
        pvr_val = calc_pvr(mpap, pawp, co).value

    pvri_val = to_float(ui.get("pvri"))
    if pvri_val is None:
        pvri_val = calc_pvri(mpap, pawp, ci_val).value

    tpg_val = calc_tpg(mpap, pawp).value
    dpg_val = calc_dpg(dpap, pawp).value

    # Exercise
    exercise_done = bool(ui.get("exercise_done"))
    mpap_co_slope = None
    pawp_co_slope = None
    delta_spap = None
    ci_peak = None
    rv_adapt = None
    if exercise_done:
        mpap_rest = mpap
        pawp_rest = pawp
        co_rest = co
        mpap_peak = to_float(ui.get("mpap_peak"))
        pawp_peak = to_float(ui.get("pawp_peak"))
        co_peak = to_float(ui.get("co_peak"))
        if mpap_rest is not None and co_rest is not None and mpap_peak is not None and co_peak is not None:
            mpap_co_slope = calc_slope(mpap_rest, co_rest, mpap_peak, co_peak).value
        if pawp_rest is not None and co_rest is not None and pawp_peak is not None and co_peak is not None:
            pawp_co_slope = calc_slope(pawp_rest, co_rest, pawp_peak, co_peak).value

        spap_rest = spap
        spap_peak = to_float(ui.get("spap_peak"))
        delta_spap = calc_delta_spap(spap_rest, spap_peak)

        ci_peak = calc_ci_peak(co_peak, bsa) if (co_peak is not None and bsa is not None) else None
        rv_adapt = exercise_rv_adaptation(delta_spap, ci_val, ci_peak)

    exercise_pattern = classify_exercise_pattern(mpap_co_slope, pawp_co_slope, rules) if exercise_done else None

    # Step-up
    sats = {
        "SVC": to_float(ui.get("sat_svc")),
        "IVC": to_float(ui.get("sat_ivc")),
        "RA": to_float(ui.get("sat_ra")),
        "RV": to_float(ui.get("sat_rv")),
        "PA": to_float(ui.get("sat_pa")),
    }
    step_up_present, step_up_loc, step_up_delta = detect_step_up(sats)

    # Classification
    ph = classify_ph_rest(mpap, pawp, pvr_val, rules)
    pvr_sev = pvr_severity(pvr_val, rules)
    ci_sev = ci_severity(ci_val, rules)

    # Etiology hints
    flags = {
        "lae": bool(ui.get("imaging_lae")),
        "ild": bool(ui.get("ct_ild")),
        "emphysema": bool(ui.get("ct_emphysema")),
        "embolism": bool(ui.get("ct_embolie")),
        "mosaic_perfusion": bool(ui.get("ct_mosaic")),
        "portal_htn": (ui.get("portal_htn") == "ja"),
        "vq_defect": bool(ui.get("vq_defect")),
        "lufu_restrictive": bool(ui.get("lufu_restrictive")),
        "lufu_obstructive": bool(ui.get("lufu_obstructive")),
        "hypoxemia": bool(ui.get("bga_hypoxemia")),
        "step_up": step_up_present,
        "precap_hemo": ph.ph_type == "precap",
    }
    etiology = infer_etiology_hints(flags)

    # Echo: S'/RAAI
    sprime = to_float(ui.get("echo_sprime"))
    ra_area = to_float(ui.get("echo_ra_area"))
    sprime_val = sprime_raai(sprime, ra_area, bsa)
    sprime_cut = float(rules.get("echo", {}).get("Sprime_RAAI_cutoff_m2_per_s_cm", 0.81))
    sprime_interp = None
    if sprime_val is not None:
        sprime_interp = "erniedrigt" if sprime_val < sprime_cut else "nicht erniedrigt"

    # Scores
    params = {
        "who_fc": ui.get("who_fc"),
        "mwd": to_float(ui.get("mwd")),
        "bnp": to_float(ui.get("bnp")),
        "ntprobnp": to_float(ui.get("ntprobnp")),
        "rap": rap,
        "ci": ci_val,
        "svo2": to_float(ui.get("svo2")),
        "sbp": to_float(ui.get("sbp")),
        "hr": to_float(ui.get("hr")),
        "egfr": to_float(ui.get("egfr")),
        "height_cm": height,
        "weight_kg": weight,
        "bmi": bmi,
        "dob": dob,
        "age_years": age,
        "rhythm": ui.get("rhythm"),
        "antihypertensive_count": to_int(ui.get("antihypertensive_count")),
        "pasp": to_float(ui.get("echo_pasp")),
        "e_over_eprime": to_float(ui.get("echo_e_over_eprime")),
    }
    esc3 = esc_ers_3_strata(params)
    esc4 = esc_ers_4_strata(params)
    rev = reveal_lite2(params)
    hf = h2fpef(params)
    cmr = cmr_rvef_risk(to_float(ui.get("cmr_rvef")))

    # Validation
    validation = validate_key_values({"mpap": mpap, "pawp": pawp, "co": co, "ci": ci_val, "rap": rap})

    return Computed(
        bsa=bsa,
        bmi=bmi,
        age=age,
        mpap=mpap,
        pawp=pawp,
        rap=rap,
        spap=spap,
        dpap=dpap,
        tpg=tpg_val,
        dpg=dpg_val,
        co=co,
        ci=ci_val,
        pvr=pvr_val,
        pvri=pvri_val,
        exercise_done=exercise_done,
        mpap_co_slope=mpap_co_slope,
        pawp_co_slope=pawp_co_slope,
        delta_spap=delta_spap,
        ci_peak=ci_peak,
        rv_adaptation=rv_adapt,
        step_up_present=step_up_present,
        step_up_location=step_up_loc,
        step_up_delta=step_up_delta,
        sprime_raai_value=sprime_val,
        sprime_raai_cutoff=sprime_cut,
        sprime_raai_interpretation=sprime_interp,
        ph=ph,
        pvr_sev=pvr_sev,
        ci_sev=ci_sev,
        exercise_pattern=exercise_pattern,
        etiology_hints=etiology,
        esc3=esc3,
        esc4=esc4,
        reveal=rev,
        hfpef=hf,
        cmr_rvef=cmr,
        validation=validation,
    )


def _severity_label(pvr_sev: Optional[str], ci_sev: Optional[str]) -> Optional[str]:
    if pvr_sev == "severe" or ci_sev == "severely_reduced":
        return "schwer"
    if pvr_sev == "moderate" or ci_sev == "reduced":
        return "mittelgradig"
    if pvr_sev == "mild":
        return "leicht"
    return None


def _choose_main_bundle(comp: Computed) -> str:
    """
    Select one main K-package (or legacy PKG) based on hemodynamics & context.
    """
    # Shunt has priority
    if comp.step_up_present:
        return "K16"

    # Exercise-only phenotypes (no rest PH)
    if (not comp.ph.ph_present) and comp.exercise_done and comp.exercise_pattern:
        if comp.exercise_pattern == "left_heart":
            return "K02"
        if comp.exercise_pattern == "pulmonary_vascular":
            return "K03"
        return "K01"

    # Rest phenotypes
    if comp.ph.ph_type == "none":
        return "K01"
    if comp.ph.ph_type == "ipcph":
        return "K14"
    if comp.ph.ph_type == "cpcph":
        return "K15"
    if comp.ph.ph_type == "precap":
        # CTEPH context?
        if comp.etiology_hints.group4_possible:
            return "K11"
        # severity by PVR/CI
        sev = _severity_label(comp.pvr_sev, comp.ci_sev)
        if sev == "schwer":
            return "K07"
        if sev == "mittelgradig":
            return "K06"
        return "K05"

    # fallback
    return "K04"


def _render_block(block_id: str, ctx: Dict[str, Any]) -> str:
    blk = textdb.get_block(block_id)
    if not blk:
        return ""
    try:
        return blk.template.format_map(SafeDict(ctx))
    except Exception:
        # last resort: return template raw
        return blk.template


def _mk_phrases(comp: Computed) -> Dict[str, str]:
    # phrases for templates
    mpap_phrase = f"mPAP {fmt_num(comp.mpap,0)} mmHg" if comp.mpap is not None else "mPAP nicht erhoben"
    pawp_phrase = f"PAWP {fmt_num(comp.pawp,0)} mmHg" if comp.pawp is not None else "PAWP nicht erhoben"
    pvr_phrase = f"PVR {fmt_num(comp.pvr,1)} WU" if comp.pvr is not None else "PVR nicht erhoben"
    ci_phrase = f"CI {fmt_num(comp.ci,2)} L/min/m²" if comp.ci is not None else "CI nicht erhoben"
    return {
        "mpap_phrase": mpap_phrase,
        "pawp_phrase": pawp_phrase,
        "pvr_phrase": pvr_phrase,
        "ci_phrase": ci_phrase,
    }


def _risk_badge_html(r: RiskResult) -> str:
    color = {
        "low": "#16a34a",
        "intermediate": "#f59e0b",
        "intermediate-low": "#84cc16",
        "intermediate-high": "#f97316",
        "high": "#dc2626",
        "unknown": "#64748b",
    }.get(r.category, "#64748b")
    label = r.category.replace("-", " ").upper()
    return f'<span style="display:inline-block;padding:6px 10px;border-radius:999px;background:{color};color:white;font-weight:700;font-size:12px;">{r.name}: {label}</span>'


def build_risk_dashboard(comp: Computed) -> str:
    badges = [
        _risk_badge_html(comp.esc3),
        _risk_badge_html(comp.esc4),
        _risk_badge_html(comp.reveal),
    ]
    if comp.hfpef is not None:
        cat = comp.hfpef.category
        hf_badge = RiskResult(name="HFpEF (H2FPEF)", category={"unlikely":"low","possible":"intermediate","likely":"high"}.get(cat,"unknown"))
        badges.append(_risk_badge_html(hf_badge))
    if comp.cmr_rvef is not None:
        badges.append(_risk_badge_html(comp.cmr_rvef))
    return "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center'>" + "".join(badges) + "</div>"


class RHKReportGenerator:
    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.rules = rules or textdb.DEFAULT_RULES

    def generate_all(self, ui: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        """
        Returns:
          doctor_md, patient_txt, internal_md, risk_html, validation_md
        """
        comp = compute_all(ui, self.rules)
        doctor = self._doctor_report(ui, comp)
        patient = self._patient_report(ui, comp)
        internal = self._internal_report(ui, comp)
        risk_html = build_risk_dashboard(comp)
        validation_md = comp.validation.to_markdown()
        return doctor, patient, internal, risk_html, validation_md

    def _doctor_report(self, ui: Dict[str, Any], comp: Computed) -> str:
        phrases = _mk_phrases(comp)
        bundle = _choose_main_bundle(comp)
        block_ids = textdb.BUNDLES.get(bundle, {}).get("B", []) + textdb.BUNDLES.get(bundle, {}).get("E", [])

        # Context sentences
        co_method = "Thermodilution" if to_float(ui.get("co_td")) is not None else ("Fick" if to_float(ui.get("co_fick")) is not None else "—")
        step_up_sentence = ""
        if comp.step_up_present and comp.step_up_location:
            step_up_sentence = f"In der Stufenoxymetrie zeigt sich ein relevanter Sättigungssprung ({comp.step_up_location}, Δ≈{fmt_num(comp.step_up_delta,1)}%)."
        else:
            step_up_sentence = "Kein relevanter Sättigungssprung in der Stufenoxymetrie."

        systemic_sentence = join_nonempty([
            f"RR {fmt_num(to_float(ui.get('sbp')),0)}/{fmt_num(to_float(ui.get('dbp')),0)} mmHg" if to_float(ui.get('sbp')) is not None else "",
            f"HF {fmt_num(to_float(ui.get('hr')),0)}/min" if to_float(ui.get('hr')) is not None else "",
            f"Rhythmus: {ui.get('rhythm')}" if ui.get('rhythm') else "",
        ], sep=", ")
        oxygen_sentence = ""
        if ui.get("ltot_present"):
            oxygen_sentence = f"Unter Sauerstofftherapie (LTOT) {fmt_num(to_float(ui.get('o2_flow')),0)} L/min."
        elif to_float(ui.get("bga_po2_rest")) is not None:
            oxygen_sentence = "Blutgase in Ruhe dokumentiert."

        exam_type_desc = ui.get("exam_type") or "RHK"
        if comp.exercise_done:
            exam_type_desc += " mit Belastung"

        # Previous RHK
        comparison_sentence = ""
        if ui.get("prev_rhk_label"):
            prev_label = str(ui.get("prev_rhk_label"))
            prev_summary = join_nonempty([
                f"mPAP {fmt_num(to_float(ui.get('prev_mpap')),0)} mmHg" if to_float(ui.get('prev_mpap')) is not None else "",
                f"PAWP {fmt_num(to_float(ui.get('prev_pawp')),0)} mmHg" if to_float(ui.get('prev_pawp')) is not None else "",
                f"CI {fmt_num(to_float(ui.get('prev_ci')),2)} L/min/m²" if to_float(ui.get('prev_ci')) is not None else "",
                f"PVR {fmt_num(to_float(ui.get('prev_pvr')),1)} WU" if to_float(ui.get('prev_pvr')) is not None else "",
            ], sep=", ")
            if prev_summary:
                comparison_sentence = f"Im Vergleich zu RHK {prev_label} {ui.get('prev_course') or 'stabiler Verlauf'} ({prev_summary})."

        # Extra: Slopes and TPG
        extras: List[str] = []
        if comp.tpg is not None:
            extras.append(f"TPG {fmt_num(comp.tpg,0)} mmHg")
        if comp.dpg is not None:
            extras.append(f"DPG {fmt_num(comp.dpg,0)} mmHg")
        if comp.exercise_done and comp.mpap_co_slope is not None and comp.pawp_co_slope is not None:
            extras.append(f"mPAP/CO-Slope {fmt_num(comp.mpap_co_slope,2)} mmHg/(L/min)")
            extras.append(f"PAWP/CO-Slope {fmt_num(comp.pawp_co_slope,2)} mmHg/(L/min)")
        if comp.exercise_done and comp.rv_adaptation is not None:
            if comp.rv_adaptation == "homeometrisch":
                extras.append("Belastungsreaktion vereinbar mit homeometrischem Adaptationstyp")
            else:
                extras.append("Belastungsreaktion mit Hinweis auf heterometrischen Adaptationstyp")

        extra_sentence = ""
        if extras:
            extra_sentence = "Zusatzparameter: " + "; ".join(extras) + "."

        # Echo addon sentence
        echo_sentence = ""
        if comp.sprime_raai_value is not None:
            echo_sentence = f"S'/RAAI {fmt_num(comp.sprime_raai_value,2)} (Cut-off {fmt_num(comp.sprime_raai_cutoff,2)}): {comp.sprime_raai_interpretation}."

        # Etiology hints
        etio_sentence = ""
        etio_notes = []
        if comp.etiology_hints and getattr(comp.etiology_hints, "notes", None):
            etio_notes.extend(comp.etiology_hints.notes)
        if etio_notes:
            etio_sentence = "Hinweise: " + " ".join(etio_notes)

        ctx: Dict[str, Any] = {}
        ctx.update(phrases)
        ctx.update({
            "ci_phrase": phrases["ci_phrase"],
            "co_method_desc": co_method,
            "step_up_sentence": step_up_sentence,
            "systemic_sentence": systemic_sentence,
            "oxygen_sentence": oxygen_sentence,
            "exam_type_desc": exam_type_desc,
            "comparison_sentence": comparison_sentence,
            "borderline_ph_sentence": comp.ph.label,
            "rest_ph_sentence": comp.ph.label,
            "pressure_resistance_short": join_nonempty([phrases.get("mpap_phrase",""), phrases.get("pvr_phrase","")], sep=", "),
            "step_up_from_to": comp.step_up_location or "—",
            "PAWP_CO_slope_phrase": "nicht führender PAWP/CO-Slope" if comp.pawp_co_slope is not None and comp.pawp_co_slope <= self.rules["exercise"]["PAWP_CO_slope_mmHg_per_L_min"] else "erhöhter PAWP/CO-Slope",
            "mPAP_CO_slope": fmt_num(comp.mpap_co_slope,2) if comp.mpap_co_slope is not None else "nicht erhoben",
            "PAWP_CO_slope": fmt_num(comp.pawp_co_slope,2) if comp.pawp_co_slope is not None else "nicht erhoben",
            "V_wave_short": "",
            "cv_stauung_phrase": "Keine zentralvenöse Stauung." if (comp.rap is None or comp.rap < 8) else ("Leichtgradige zentralvenöse Stauung." if comp.rap < 14 else "Ausgeprägte zentralvenöse Stauung."),
            "pv_stauung_phrase": "Keine pulmonalvenöse Stauung." if (comp.pawp is None or comp.pawp <= self.rules["rest"]["PAWP_postcap_mmHg"]) else "Hinweis auf pulmonalvenöse Stauung.",
            "provocation_sentence": "",
            "provocation_type_desc": "Belastung" if comp.exercise_done else "Provokation",
            "provocation_result_sentence": "",
            "therapy_plan_sentence": ui.get("therapy_plan") or "",
            "therapy_escalation_sentence": ui.get("therapy_escalation") or "",
            "therapy_neutral_sentence": "",
            "risk_profile_desc": ui.get("risk_profile_desc") or "",
            "patient_preference_sentence": ui.get("patient_preference") or "",
            "measurement_limitation_sentence": ui.get("measurement_limitations") or "",
        })

        beurteilung_parts: List[str] = []
        empfehlung_parts: List[str] = []
        for bid in block_ids:
            if bid.endswith("_B") or bid.startswith("B") or bid.startswith("K") and bid.endswith("_B"):
                beurteilung_parts.append(_render_block(bid, ctx))
            elif bid.endswith("_E") or bid.startswith("E") or bid.startswith("K") and bid.endswith("_E"):
                empfehlung_parts.append(_render_block(bid, ctx))
            else:
                # fallback: include
                beurteilung_parts.append(_render_block(bid, ctx))

        # Add risk block in recommendations directly after diagnosis sentence
        risk_lines: List[str] = []
        risk_lines.append(f"- {comp.esc3.name}: {comp.esc3.category}")
        risk_lines.append(f"- {comp.esc4.name}: {comp.esc4.category}")
        risk_lines.append(f"- {comp.reveal.name}: {comp.reveal.category}")
        if comp.cmr_rvef is not None:
            risk_lines.append(f"- {comp.cmr_rvef.name}: {comp.cmr_rvef.category}")
        if comp.hfpef is not None and comp.hfpef.category in ("possible","likely"):
            risk_lines.append(f"- HFpEF-Score (H2FPEF): {comp.hfpef.category} (Hinweis auf mögliche diastolische Dysfunktion)")
        risk_block = "#### Risikostratifizierung\n" + "\n".join(risk_lines)

        # P-modules: selected + bundle suggestions
        selected_modules: List[str] = ui.get("modules") or []
        suggestions = textdb.BUNDLES.get(bundle, {}).get("P_suggestions", [])
        all_modules = []
        # keep order: suggestions first, then selected
        for mid in suggestions + selected_modules:
            if mid and mid not in all_modules:
                all_modules.append(mid)

        procedere_parts: List[str] = []
        for mid in all_modules:
            blk = textdb.get_block(mid)
            if blk:
                procedere_parts.append(f"- {blk.template.format_map(SafeDict(ctx))}")

        # Compose markdown
        header = f"**RHK Befund**\n\n"
        header += join_nonempty([
            f"Name: {ui.get('last_name','')} {ui.get('first_name','')}".strip(),
            f"Geb.-Datum: {ui.get('birthdate','')}" if ui.get('birthdate') else "",
            f"Alter: {comp.age} J." if comp.age is not None else "",
        ], sep="  \n")
        header = header.strip()

        # ---- Additional structured sections (so the data is actually used) ----
        sections: List[str] = []

        # Klinik / Anamnese
        klinik_lines: List[str] = []
        if ui.get("story"):
            klinik_lines.append(f"- Kurz-Anamnese: {ui.get('story')}")
        if ui.get("ph_dx_known"):
            klinik_lines.append("- PH-Diagnose: bekannt")
        if ui.get("ph_suspected"):
            klinik_lines.append("- PH-Verdacht: ja")
        if ui.get("comorbidities"):
            klinik_lines.append(f"- Vorerkrankungen: {ui.get('comorbidities')}")
        if ui.get("ph_relevance"):
            klinik_lines.append(f"- Relevanz für PH: {ui.get('ph_relevance')}")
        if klinik_lines:
            sections.append("### Klinik / Anamnese\n" + "\n".join(klinik_lines))

        # Labor
        lab_map = [
            ("inr","INR"),("quick","Quick"),("krea","Krea"),("egfr","eGFR"),
            ("ptt","PTT"),("thrombos","Thrombos"),("hb","Hb"),("crp","CRP"),("leukos","Leukos"),
            ("bnp","BNP"),("ntprobnp","pro-NT-BNP"),
        ]
        lab_lines: List[str] = []
        for key, label in lab_map:
            v = to_float(ui.get(key))
            if v is not None:
                lab_lines.append(f"- {label}: {fmt_num(v,1)}")
        if ui.get("congestive_organopathy") == "ja":
            lab_lines.append("- Hinweis auf congestive Organopathie: ja")
        if lab_lines:
            sections.append("### Labor\n" + "\n".join(lab_lines))

        # Blutgase / LTOT
        bga_lines: List[str] = []
        if ui.get("ltot_present"):
            bga_lines.append(f"- LTOT: ja ({fmt_num(to_float(ui.get('o2_flow')),0)} L/min)" if to_float(ui.get('o2_flow')) is not None else "- LTOT: ja")
        if to_float(ui.get("bga_po2_rest")) is not None or to_float(ui.get("bga_pco2_rest")) is not None:
            bga_lines.append(f"- BGA Ruhe: pO₂ {fmt_num(to_float(ui.get('bga_po2_rest')),0)}, pCO₂ {fmt_num(to_float(ui.get('bga_pco2_rest')),0)}")
        if to_float(ui.get("bga_po2_ex")) is not None or to_float(ui.get("bga_pco2_ex")) is not None:
            bga_lines.append(f"- BGA Belastung: pO₂ {fmt_num(to_float(ui.get('bga_po2_ex')),0)}, pCO₂ {fmt_num(to_float(ui.get('bga_pco2_ex')),0)}")
        if to_float(ui.get("bga_ph_night")) is not None or to_float(ui.get("bga_be_night")) is not None:
            bga_lines.append(f"- BGA Nacht: pH {fmt_num(to_float(ui.get('bga_ph_night')),2)}, BE {fmt_num(to_float(ui.get('bga_be_night')),0)}")
        if bga_lines:
            sections.append("### Blutgase / Sauerstoff\n" + "\n".join(bga_lines))

        # Bildgebung
        img_lines: List[str] = []
        ct_flags = []
        if ui.get("ct_embolie"): ct_flags.append("Pulmonalembolie")
        if ui.get("ct_ild"): ct_flags.append("ILD")
        if ui.get("ct_emphysema"): ct_flags.append("Emphysem")
        if ui.get("ct_mosaic"): ct_flags.append("Mosaikperfusion")
        if ui.get("ct_coronary_calc"): ct_flags.append("Koronarkalk")
        if ct_flags:
            img_lines.append("- CT: " + ", ".join(ct_flags))
        card_flags = []
        if ui.get("card_ventricular_abn"): card_flags.append("ventrikulär auffällig")
        if ui.get("pericardial_effusion"): card_flags.append("Perikarderguss")
        if ui.get("imaging_lae"): card_flags.append("Linksatrium erweitert")
        if card_flags:
            img_lines.append("- Kardialer Phänotyp: " + ", ".join(card_flags))
        if ui.get("vq_defect"):
            img_lines.append("- V/Q: Perfusionsdefekt / CTEPH-Verdacht")
        if img_lines:
            sections.append("### Bildgebung\n" + "\n".join(img_lines))

        # Lungenfunktion
        lufu_lines: List[str] = []
        if ui.get("lufu_done") == "ja":
            lufu_lines.append("- Lufu: durchgeführt")
        phen = []
        if ui.get("lufu_obstructive"): phen.append("obstruktiv")
        if ui.get("lufu_restrictive"): phen.append("restriktiv")
        if ui.get("lufu_diffusion"): phen.append("Diffusionsstörung")
        if phen:
            lufu_lines.append("- Phänotyp: " + ", ".join(phen))
        for key, label in [("fev1","FEV1"),("fvc","FVC"),("fev1_fvc","FEV1/FVC"),("tlc","TLC"),("rv","RV"),("dlco_sb","DLCO"),("dlco_va","DLCO/VA")]:
            v = to_float(ui.get(key))
            if v is not None:
                lufu_lines.append(f"- {label}: {fmt_num(v,2)}")
        if ui.get("lufu_summary"):
            lufu_lines.append(f"- Zusammenfassung: {ui.get('lufu_summary')}")
        if lufu_lines:
            sections.append("### Lungenfunktion\n" + "\n".join(lufu_lines))

        # Echo / CMR
        echo_lines: List[str] = []
        if ui.get("echo_done") == "ja":
            echo_lines.append("- Echo: vorhanden")
        if ui.get("echo_free"):
            echo_lines.append(f"- Echo (Freitext): {ui.get('echo_free')}")
        if comp.sprime_raai_value is not None:
            echo_lines.append(f"- S'/RAAI: {fmt_num(comp.sprime_raai_value,2)} ({comp.sprime_raai_interpretation})")
        for key, label in [("echo_pasp","PASP"),("echo_e_over_eprime","E/e'")]:
            v = to_float(ui.get(key))
            if v is not None:
                echo_lines.append(f"- {label}: {fmt_num(v,1)}")
        cmr_lines: List[str] = []
        for key, label in [("cmr_rvesvi","RVESVi"),("cmr_svi","SVi"),("cmr_rvef","RVEF")]:
            v = to_float(ui.get(key))
            if v is not None:
                cmr_lines.append(f"- {label}: {fmt_num(v,1)}")
        if cmr_lines:
            echo_lines.append("- CMR: " + "; ".join([l.replace("- ","") for l in cmr_lines]))
        if echo_lines:
            sections.append("### Echokardiographie / CMR\n" + "\n".join(echo_lines))

        # Medikation
        med_lines: List[str] = []
        if ui.get("ph_meds_current") == "ja":
            med_lines.append("- PH-Medikation: ja")
            if ui.get("ph_meds_which"):
                med_lines.append(f"- Welche: {ui.get('ph_meds_which')}")
            if ui.get("ph_meds_since"):
                med_lines.append(f"- Seit wann: {ui.get('ph_meds_since')}")
        if ui.get("diuretics") == "ja":
            med_lines.append("- Diuretika: ja")
        if ui.get("other_meds"):
            med_lines.append(f"- Sonstige Medikation: {ui.get('other_meds')}")
        if med_lines:
            sections.append("### Medikation\n" + "\n".join(med_lines))

        extra_md = "\n\n".join(sections)

        beurteilung = clean_sentence(" ".join([p for p in beurteilung_parts if p]))
        empfehlung = clean_sentence(" ".join([p for p in empfehlung_parts if p]))

        # Append extras
        if comparison_sentence:
            beurteilung = clean_sentence(beurteilung + " " + comparison_sentence)
        if extra_sentence:
            beurteilung = clean_sentence(beurteilung + " " + extra_sentence)
        if echo_sentence:
            beurteilung = clean_sentence(beurteilung + " " + echo_sentence)
        if etio_sentence:
            empfehlung = clean_sentence(empfehlung + " " + etio_sentence)

        parts_md: List[str] = [header]
        if extra_md:
            parts_md.append(extra_md)

        parts_md.extend([
            "### Beurteilung",
            beurteilung or "—",
            "### Empfehlung",
            empfehlung or "—",
            risk_block,
            "### Procedere / Module",
            "\n".join(procedere_parts) if procedere_parts else "—",
        ])
        md = "\n\n".join([p for p in parts_md if p and str(p).strip()])

        return md

    def _patient_report(self, ui: Dict[str, Any], comp: Computed) -> str:
        from .reports_patient import build_patient_report
        return build_patient_report(ui, comp)

    def _internal_report(self, ui: Dict[str, Any], comp: Computed) -> str:
        # A compact internal JSON-like summary for debugging/board
        lines: List[str] = []
        lines.append("### Interner Kurzüberblick")
        lines.append(f"- PH-Klasse: {comp.ph.label} ({comp.ph.ph_type})")
        if comp.pvr is not None:
            lines.append(f"- PVR: {fmt_num(comp.pvr,1)} WU ({comp.pvr_sev})")
        if comp.ci is not None:
            lines.append(f"- CI: {fmt_num(comp.ci,2)} ({comp.ci_sev})")
        if comp.exercise_done:
            lines.append(f"- Belastung: mPAP/CO {fmt_num(comp.mpap_co_slope,2)} | PAWP/CO {fmt_num(comp.pawp_co_slope,2)} | ΔsPAP {fmt_num(comp.delta_spap,1)} | CI_peak {fmt_num(comp.ci_peak,2)} | RV-Adaption {comp.rv_adaptation}")
        if comp.step_up_present:
            lines.append(f"- Step-up: {comp.step_up_location} (Δ≈{fmt_num(comp.step_up_delta,1)}%)")
        lines.append(f"- Ätiologie-Hinweise: {comp.etiology_hints.summary_sentence() if comp.etiology_hints else '—'}")
        return "\n".join(lines)
