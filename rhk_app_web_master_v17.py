#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_app_web_master_v17.py
Gradio-Web-GUI (Gradio >=4/5) für den RHK Befundassistenten.

Fix für Render-Fehler:
- Gradio >=4 hat kein `gr.inputs.*` mehr. Diese Version nutzt die neuen Komponenten (gr.Textbox, gr.Number, ...).

Deployment (Render):
- Start Command: python rhk_app_web_master_v17.py
- PORT wird automatisch via ENV PORT gebunden.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import gradio as gr

from rhk_core import (
    APP_VERSION,
    generate_all,
    save_case_to_file,
    make_pdf_from_text,
)

# Optional: module choices from textdb (if present in repo)
try:
    import rhk_textdb as _textdb  # type: ignore
    _ALL_BLOCKS = getattr(_textdb, "ALL_BLOCKS", {})
except Exception:
    _textdb = None  # type: ignore
    _ALL_BLOCKS = {}

APP_TITLE = "RHK Befundassistent"


# -------------------------
# UI Schema (flat keys)
# -------------------------
UI_FIELDS: List[str] = [
    # Stammdaten
    "firstname", "lastname", "sex", "dob", "age", "height_cm", "weight_kg", "bsa_m2", "bmi",
    "story", "ph_known", "ph_suspected",

    # Labor / Klinik
    "inr", "quick", "krea_mg_dl", "hst", "ptt", "thrombos", "hb_g_dl", "crp", "leukos",
    "bnp_pg_ml", "ntprobnp_pg_ml", "entresto", "congestive_organopathy",
    "sym_syncope", "sym_dizziness", "sym_hemoptysis", "sym_stairs_flights", "sym_walk_distance",

    # BGA / LTOT
    "ltot_present", "ltot_flow_lpm",
    "bga_rest_po2", "bga_rest_pco2",
    "bga_ex_po2", "bga_ex_pco2",
    "bga_night_ph", "bga_night_be", "bga_paused",

    # Infektiologie / Immunologie
    "virology_positive", "virology_details",
    "immunology_positive", "immunology_details",

    # Abdomen / Leber
    "abdomen_sono_done", "portal_htn", "abdomen_findings",

    # Bildgebung Thorax + kardialer Phänotyp
    "ct_angio_done",
    "ct_ild", "ct_emphysem", "ct_embolie", "ct_mosaik", "ct_chronic_thrombo", "ct_koronarkalk",
    "vq_done", "vq_positive",
    "ild_type", "ild_histology", "ild_fibrosis_clinic", "ild_extent",
    "cardiac_ventricular", "cardiac_pericardial_effusion", "cardiac_la_enlarged",

    # Vorerkrankungen
    "comorbidities_text", "comorbidities_ph_relevant",

    # Medikamente
    "ph_meds_current", "ph_meds_current_which", "ph_meds_current_since",
    "ph_meds_past", "ph_meds_past_which",
    "other_meds_text", "diuretics",

    # Lungenfunktion
    "lufu_done", "lufu_summary_text",
    "lufu_obstructive", "lufu_restrictive", "lufu_diffusion",
    "fev1_l", "fvc_l", "fev1_fvc_pct", "tlc_l", "rv_l",
    "dlco_sb_pct", "dlco_va_pct",

    # Echokardiographie
    "echo_done",
    "echo_ra_area_cm2", "echo_pericardial_effusion", "echo_tapse_mm", "echo_spap_mmHg",
    "echo_sprime_cm_s", "echo_ra_esa_cm2", "echo_la_enlarged",
    "echo_ivc_diam_mm", "echo_ivc_collapse_pct",

    # HFpEF (H2FPEF continuous)
    "hfpef_af", "af_known", "hfpef_e_eprime", "hfpef_pasp", "lvef_percent",

    # CMR
    "cmr_rvef_percent", "cmr_lvef_percent",

    # RHK Meta
    "rhk_consent", "anticoagulation", "access_site", "co_preference",

    # RHK Ruhe
    "spap_mmHg", "dpap_mmHg", "mpap_mmHg", "pawp_mmHg", "rap_mmHg",
    "td_co_L_min", "fick_co_L_min", "ci_L_min_m2", "pvr_WU",
    "v_wave_present", "step_up_present",

    # Volumenchallenge (optional)
    "volume_done", "volume_pawp_baseline", "volume_pawp_post",

    # Vasoreagibilität
    "vasoreactivity_done", "vasoreactivity_positive",

    # Belastung
    "exercise_done",
    "ex_mpap_rest", "ex_mpap_peak",
    "ex_pawp_rest", "ex_pawp_peak",
    "ex_spap_rest", "ex_spap_peak",
    "ex_co_rest", "ex_co_peak",
    "ex_ci_rest", "ex_ci_peak",

    # Funktionelle Tests
    "who_fc", "six_mwd_m", "cpet_ve_vco2", "cpet_vo2max",

    # REVEAL lite params
    "reveal_rbsys", "reveal_hr", "reveal_egfr",

    # Anämie
    "anemia_morphology",

    # Vergleich RHK
    "prev_rhk_date", "prev_course", "prev_mpap", "prev_pawp", "prev_ci", "prev_pvr",

    # Module + Abschluss
    "modules", "final_free_text",
]


# Example case (minimal but representative)
EXAMPLE_CASE: Dict[str, Any] = {
    "firstname": "Max",
    "lastname": "Mustermann",
    "sex": "männlich",
    "dob": "1975-05-12",
    "height_cm": 178,
    "weight_kg": 86,
    "story": "Belastungsdyspnoe, Abklärung Lungenhochdruck",

    "hb_g_dl": 12.2,
    "ntprobnp_pg_ml": 900,
    "entresto": False,
    "congestive_organopathy": False,

    "ltot_present": False,
    "immunology_positive": False,
    "virology_positive": False,

    "ct_angio_done": True,
    "ct_ild": False,
    "ct_emphysem": True,
    "ct_mosaik": False,
    "ct_chronic_thrombo": False,
    "vq_done": False,
    "vq_positive": False,
    "cardiac_la_enlarged": True,

    "lufu_done": True,
    "lufu_obstructive": True,
    "lufu_restrictive": False,
    "lufu_diffusion": True,
    "lufu_summary_text": "Obstruktiv; DLCO reduziert",

    "echo_done": True,
    "echo_spap_mmHg": 45,
    "echo_sprime_cm_s": 10,
    "echo_ra_esa_cm2": 22,
    "echo_ivc_diam_mm": 22,
    "echo_ivc_collapse_pct": 30,

    "hfpef_af": False,
    "hfpef_e_eprime": 14,
    "hfpef_pasp": 45,
    "lvef_percent": 60,

    "spap_mmHg": 55,
    "dpap_mmHg": 20,
    "pawp_mmHg": 18,
    "rap_mmHg": 10,
    "td_co_L_min": 5.5,

    "who_fc": "III",
    "six_mwd_m": 320,

    "reveal_rbsys": 115,
    "reveal_hr": 88,
    "reveal_egfr": 62,
    "modules": [],
}


# -------------------------
# Module choices
# -------------------------
def _module_choices() -> List[Tuple[str, str]]:
    """
    Returns list of (label, value) for CheckboxGroup. value = Modul-ID (Pxx).
    """
    out: List[Tuple[str, str]] = []
    if _ALL_BLOCKS:
        for bid, blk in _ALL_BLOCKS.items():
            if isinstance(bid, str) and bid.startswith("P") and len(bid) <= 4:
                title = getattr(blk, "title", bid)
                out.append((f"{bid} – {title}", bid))
    if not out:
        out = [
            ("P01 – Allgemeines Procedere", "P01"),
            ("P02 – Diuretika/Volumenmanagement", "P02"),
            ("P03 – Antikoagulation", "P03"),
            ("P08 – Pneumologie/Lungenerkrankung", "P08"),
            ("P11 – HFpEF/Linksherz", "P11"),
            ("P13 – Eisen/Anämie", "P13"),
        ]
    out.sort(key=lambda x: x[1])
    return out

MODULE_CHOICES = _module_choices()


# -------------------------
# Helpers
# -------------------------
def ui_get_raw(*vals) -> Dict[str, Any]:
    return {k: v for k, v in zip(UI_FIELDS, vals)}


def _load_example() -> List[Any]:
    raw = {k: None for k in UI_FIELDS}
    raw.update(EXAMPLE_CASE)
    return [raw.get(k) for k in UI_FIELDS]


def _load_case_file(file_obj) -> List[Any]:
    if file_obj is None:
        raw = {k: None for k in UI_FIELDS}
        return [raw.get(k) for k in UI_FIELDS]
    try:
        path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or file_obj
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    raw = {k: None for k in UI_FIELDS}
    if isinstance(data, dict):
        raw.update(data)
    return [raw.get(k) for k in UI_FIELDS]


def _generate(*vals):
    raw = ui_get_raw(*vals)
    doctor_txt, patient_txt, internal_txt, risk_html, der, decision, risks = generate_all(raw)
    derived_small = {
        "main_bundle_id": decision.main_bundle_id,
        "main_bundle_reason": decision.main_bundle_reason,
        "mpap_mmHg": der.mpap,
        "pawp_mmHg": der.pawp,
        "pvr_WU": der.pvr,
        "ci_L_min_m2": der.ci,
        "tpg_mmHg": der.tpg,
        "hfpef_probability_pct": (der.hfpef.probability_pct if der.hfpef else None),
        "hfpef_category": (der.hfpef.category if der.hfpef else None),
    }
    derived_json = json.dumps(derived_small, ensure_ascii=False, indent=2)
    return doctor_txt, patient_txt, internal_txt, risk_html, derived_json


def _save_case(*vals):
    raw = ui_get_raw(*vals)
    path = save_case_to_file(raw)
    return gr.update(value=path, visible=True)


def _pdf_doctor(doctor_txt: str):
    path = make_pdf_from_text(f"{APP_TITLE} – Arztbefund ({APP_VERSION})", doctor_txt, filename_prefix="rhk_doctor")
    if not path:
        return gr.update(value=None, visible=True)
    return gr.update(value=path, visible=True)


def _pdf_patient(patient_txt: str):
    path = make_pdf_from_text(f"{APP_TITLE} – Patientenbefund ({APP_VERSION})", patient_txt, filename_prefix="rhk_patient")
    if not path:
        return gr.update(value=None, visible=True)
    return gr.update(value=path, visible=True)


def _vis_bool(flag: bool):
    return gr.update(visible=bool(flag))


def _anemia_visibility(sex: str, hb: Any):
    try:
        hb_v = float(str(hb).replace(",", "."))
    except Exception:
        hb_v = None
    s = (sex or "").lower()
    thr = None
    if s.startswith("m"):
        thr = 13.0
    elif s.startswith("w") or s.startswith("f"):
        thr = 12.0
    elif s:
        thr = 12.5
    show = (hb_v is not None and thr is not None and hb_v < thr)
    return gr.update(visible=show)


# -------------------------
# UI
# -------------------------
def build_demo() -> Tuple[gr.Blocks, str]:
    css = """
    .app-header {display:flex; align-items:center; justify-content:space-between; gap:12px; padding:10px 12px; border:1px solid rgba(0,0,0,0.08); border-radius:12px; margin-bottom:10px;}
    .app-title {font-size:18px; font-weight:700;}
    .app-version {font-size:12px; opacity:0.75;}
    .riskwrap {display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin: 8px 0 12px 0;}
    .riskcard {border:1px solid rgba(0,0,0,0.08); border-radius:12px; padding:10px;}
    .riskname {font-weight:700; margin-bottom:6px;}
    .riskscore {font-size:18px; font-weight:700; margin-top:6px;}
    .riskdetail {font-size:12px; opacity:0.8; margin-top:6px; line-height:1.2;}
    .badge {display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; font-weight:700;}
    .badge-low {background: rgba(0,200,0,0.12);}
    .badge-mid {background: rgba(255,165,0,0.18);}
    .badge-midlow {background: rgba(255,165,0,0.12);}
    .badge-midhigh {background: rgba(255,120,0,0.20);}
    .badge-high {background: rgba(255,0,0,0.14);}
    .badge-unk {background: rgba(0,0,0,0.06);}
    """

    theme = gr.themes.Soft() if hasattr(gr, "themes") else None

    with gr.Blocks(title=APP_TITLE, theme=theme, css=css) as demo:
        gr.HTML(f"""
        <div class="app-header">
          <div>
            <div class="app-title">{APP_TITLE}</div>
            <div class="app-version">Version: <b>{APP_VERSION}</b></div>
          </div>
        </div>
        """)

        # Top actions
        with gr.Row():
            btn_example_top = gr.Button("🧪 Beispiel laden", variant="secondary")
            btn_generate_top = gr.Button("🧾 Befund erstellen / aktualisieren", variant="primary")
            btn_save_top = gr.Button("💾 Fall speichern", variant="secondary")
            file_case_top = gr.File(label="Fall laden (JSON)", file_types=[".json"])
            btn_load_top = gr.Button("📂 Fall laden", variant="secondary")

        # Dashboard always visible
        risk_html = gr.HTML(value="<div class='riskwrap'></div>", label="Dashboard (Risiko/Score)")
        derived_json = gr.Code(label="Kurz-Übersicht (abgeleitet)", value="", language="json")

        # Outputs (placed at bottom tab, but also accessible here? keep in tab)
        with gr.Tabs():
            with gr.Tab("1) Klinik & Labor"):
                with gr.Row():
                    with gr.Column(scale=1):
                        firstname = gr.Textbox(label="Vorname")
                        lastname = gr.Textbox(label="Name")
                        sex = gr.Dropdown(label="Geschlecht", choices=["männlich", "weiblich", "divers/unklar"], value=None)
                        dob = gr.Textbox(label="Geburtsdatum (YYYY-MM-DD oder DD.MM.YYYY)")
                        age = gr.Number(label="Alter (optional, wird sonst berechnet)", precision=0)
                    with gr.Column(scale=2):
                        story = gr.Textbox(label="Story / Kurz-Anamnese", lines=4)
                        ph_known = gr.Checkbox(label="PH-Diagnose bekannt", value=False)
                        ph_suspected = gr.Checkbox(label="PH-Verdachtsdiagnose", value=False)

                with gr.Accordion("Anthropometrie", open=False):
                    with gr.Row():
                        height_cm = gr.Number(label="Größe (cm)")
                        weight_kg = gr.Number(label="Gewicht (kg)")
                        bsa_m2 = gr.Number(label="KOF/BSA (m²) (optional)")
                        bmi = gr.Number(label="BMI (optional)")

                with gr.Accordion("Labor", open=True):
                    with gr.Row():
                        inr = gr.Number(label="INR")
                        quick = gr.Number(label="Quick (%)")
                        krea_mg_dl = gr.Number(label="Kreatinin (mg/dl)")
                        hst = gr.Number(label="Hst/Hkt (%)")
                    with gr.Row():
                        ptt = gr.Number(label="PTT (s)")
                        thrombos = gr.Number(label="Thrombos (G/l)")
                        hb_g_dl = gr.Number(label="Hb (g/dl)")
                        leukos = gr.Number(label="Leukos (G/l)")
                    with gr.Row():
                        crp = gr.Number(label="CRP (mg/l)")
                        bnp_pg_ml = gr.Number(label="BNP (pg/ml)")
                        ntprobnp_pg_ml = gr.Number(label="NT-proBNP (pg/ml)")
                        entresto = gr.Checkbox(label="Sacubitril/Valsartan (Entresto®)?", value=False)
                    congestive_organopathy = gr.Checkbox(label="Hinweis auf congestive Organopathie?", value=False)

                    anemia_morphology = gr.Dropdown(
                        label="Wenn Anämie: Morphologie (optional)",
                        choices=["mikrozytär", "normozytär", "makrozytär"],
                        value=None,
                        visible=False,
                    )

                with gr.Accordion("Symptome (funktionell)", open=False):
                    with gr.Row():
                        sym_syncope = gr.Checkbox(label="Synkope/Ohnmacht", value=False)
                        sym_dizziness = gr.Checkbox(label="Schwindel", value=False)
                        sym_hemoptysis = gr.Checkbox(label="Hämoptysen", value=False)
                    with gr.Row():
                        sym_stairs_flights = gr.Number(label="Treppenstufen (Stockwerke) (optional)", precision=0)
                        sym_walk_distance = gr.Number(label="Gehstrecke (m) (optional)", precision=0)

                with gr.Accordion("Blutgase / LTOT", open=False):
                    ltot_present = gr.Checkbox(label="LTOT vorhanden", value=False)
                    ltot_flow_lpm = gr.Number(label="LTOT: Liter/Minute", visible=False)
                    with gr.Row():
                        bga_rest_po2 = gr.Number(label="BGA Ruhe pO₂")
                        bga_rest_pco2 = gr.Number(label="BGA Ruhe pCO₂")
                    with gr.Row():
                        bga_ex_po2 = gr.Number(label="BGA Belastung pO₂")
                        bga_ex_pco2 = gr.Number(label="BGA Belastung pCO₂")
                    with gr.Row():
                        bga_night_ph = gr.Number(label="BGA Nacht pH")
                        bga_night_be = gr.Number(label="BGA Nacht BE")
                        bga_paused = gr.Checkbox(label="BGA pausiert", value=False)

                with gr.Accordion("Infektiologie / Immunologie", open=False):
                    virology_positive = gr.Checkbox(label="Virologie positiv?", value=False)
                    virology_details = gr.Textbox(label="Wenn ja: welche Virologie?", visible=False)
                    immunology_positive = gr.Checkbox(label="Immunologie positiv?", value=False)
                    immunology_details = gr.Textbox(label="Wenn ja: welche Immunologie?", visible=False)

                with gr.Accordion("Abdomen / Leber", open=False):
                    abdomen_sono_done = gr.Checkbox(label="Abdomen-Sono durchgeführt?", value=False)
                    portal_htn = gr.Checkbox(label="Hinweis auf portale Hypertension?", value=False)
                    abdomen_findings = gr.Textbox(label="Wenn Sono: besondere Befunde?", visible=False, lines=2)

                with gr.Accordion("Vorerkrankungen", open=False):
                    comorbidities_text = gr.Textbox(label="Vorerkrankungen (Freitext)", lines=2)
                    comorbidities_ph_relevant = gr.Textbox(label="PH-relevante Vorerkrankungen / Hinweise (Freitext)", lines=2)

                with gr.Accordion("Medikation (Kurzüberblick)", open=False):
                    ph_meds_current = gr.Checkbox(label="PH-spezifische Therapie aktuell?", value=False)
                    ph_meds_current_which = gr.Textbox(label="Welche PH-Therapie? (Freitext)", visible=False)
                    ph_meds_current_since = gr.Textbox(label="Seit wann? (Freitext)", visible=False)
                    ph_meds_past = gr.Checkbox(label="PH-spezifische Therapie früher?", value=False)
                    ph_meds_past_which = gr.Textbox(label="Welche frühere PH-Therapie? (Freitext)", visible=False)
                    diuretics = gr.Checkbox(label="Diuretika aktuell?", value=False)
                    other_meds_text = gr.Textbox(label="Weitere Medikation (Freitext)", lines=2)

            with gr.Tab("2) Bildgebung, Lufu, Echo/MRT"):
                with gr.Accordion("CT / Thoraxbildgebung", open=True):
                    ct_angio_done = gr.Checkbox(label="CT-Angio durchgeführt", value=False)
                    with gr.Row():
                        ct_ild = gr.Checkbox(label="ILD vorhanden", value=False)
                        ct_emphysem = gr.Checkbox(label="Emphysem", value=False)
                        ct_embolie = gr.Checkbox(label="Akute Lungenarterienembolie", value=False)
                    with gr.Row():
                        ct_mosaik = gr.Checkbox(label="Mosaikperfusion", value=False)
                        ct_chronic_thrombo = gr.Checkbox(label="Chronisch thromboembolische Veränderungen (CTEPD/CTEPH)", value=False)
                        ct_koronarkalk = gr.Checkbox(label="Koronarkalk", value=False)

                    with gr.Row():
                        vq_done = gr.Checkbox(label="V/Q-Szinti vorhanden", value=False)
                        vq_positive = gr.Checkbox(label="V/Q positiv (Perfusionsdefekte)", value=False)

                    ild_type = gr.Textbox(label="ILD: Typ/Entität (z.B. UIP/IPF, NSIP, HP, Sarkoidose)", visible=False)
                    with gr.Row():
                        ild_histology = gr.Checkbox(label="ILD histologisch gesichert?", value=False, visible=False)
                        ild_fibrosis_clinic = gr.Checkbox(label="An Fibroseambulanz angebunden?", value=False, visible=False)
                    ild_extent = gr.Dropdown(label="Ausmaß der ILD", choices=["mild", "moderat", "ausgeprägt", "unbekannt"], value=None, visible=False)

                with gr.Accordion("Kardialer Phänotyp (Bildgebung)", open=False):
                    cardiac_ventricular = gr.Dropdown(label="Ventrikulär", choices=["normal", "auffällig", "unbekannt"], value="unbekannt")
                    cardiac_pericardial_effusion = gr.Checkbox(label="Perikarderguss", value=False)
                    cardiac_la_enlarged = gr.Checkbox(label="Linkes Atrium vergrößert", value=False)

                with gr.Accordion("Lungenfunktion", open=False):
                    lufu_done = gr.Checkbox(label="Lufu durchgeführt", value=False)
                    lufu_summary_text = gr.Textbox(label="Lufu Kurzbefund (Freitext)", lines=2, visible=False)
                    with gr.Row():
                        lufu_obstructive = gr.Checkbox(label="Obstruktiv", value=False)
                        lufu_restrictive = gr.Checkbox(label="Restriktiv", value=False)
                        lufu_diffusion = gr.Checkbox(label="Diffusionsstörung", value=False)
                    with gr.Accordion("Lufu Werte (optional)", open=False):
                        with gr.Row():
                            fev1_l = gr.Number(label="FEV1 (L)")
                            fvc_l = gr.Number(label="FVC (L)")
                            fev1_fvc_pct = gr.Number(label="FEV1/FVC (%)")
                        with gr.Row():
                            tlc_l = gr.Number(label="TLC (L)")
                            rv_l = gr.Number(label="RV (L)")
                        with gr.Row():
                            dlco_sb_pct = gr.Number(label="DLCO-SB (% Soll)")
                            dlco_va_pct = gr.Number(label="DLCO/VA (% Soll)")

                with gr.Accordion("Echokardiographie", open=True):
                    echo_done = gr.Checkbox(label="Echo durchgeführt", value=False)
                    with gr.Row():
                        echo_ra_area_cm2 = gr.Number(label="RA Fläche (cm²)")
                        echo_pericardial_effusion = gr.Checkbox(label="Perikarderguss (Echo)", value=False)
                        echo_tapse_mm = gr.Number(label="TAPSE (mm)")
                        echo_spap_mmHg = gr.Number(label="sPAP (Echo, mmHg)")
                    with gr.Row():
                        echo_sprime_cm_s = gr.Number(label="S' (cm/s) (Yogeswaran et al.)")
                        echo_ra_esa_cm2 = gr.Number(label="RA ESA (cm²) (für RAAI)")
                        echo_la_enlarged = gr.Checkbox(label="LA vergrößert (Echo)", value=False)
                    with gr.Row():
                        echo_ivc_diam_mm = gr.Number(label="V. cava inferior Durchmesser (mm)")
                        echo_ivc_collapse_pct = gr.Number(label="V. cava Kollaps (%)")

                with gr.Accordion("MRT / CMR", open=False):
                    cmr_rvef_percent = gr.Number(label="RVEF (%)")
                    cmr_lvef_percent = gr.Number(label="LVEF (%)")

            with gr.Tab("3) RHK"):
                with gr.Accordion("Meta", open=True):
                    with gr.Row():
                        rhk_consent = gr.Checkbox(label="RHK-Aufklärung erfolgt?", value=False)
                        anticoagulation = gr.Checkbox(label="Antikoagulation?", value=False)
                        access_site = gr.Dropdown(label="Zugang", choices=["V. jug. dextra", "V. jug. sinistra", "andere", "unbekannt"], value="unbekannt")
                        co_preference = gr.Dropdown(label="CO-Präferenz", choices=["Thermodilution", "Fick"], value="Thermodilution")

                with gr.Accordion("Ruhe-Messungen", open=True):
                    with gr.Row():
                        spap_mmHg = gr.Number(label="sPAP (mmHg)")
                        dpap_mmHg = gr.Number(label="dPAP (mmHg)")
                        mpap_mmHg = gr.Number(label="mPAP (mmHg) (optional – wird sonst berechnet)")
                        pawp_mmHg = gr.Number(label="PAWP (mmHg)")
                    with gr.Row():
                        rap_mmHg = gr.Number(label="RAP (mmHg)")
                        td_co_L_min = gr.Number(label="TD-CO (L/min)")
                        fick_co_L_min = gr.Number(label="Fick-CO (L/min)")
                        ci_L_min_m2 = gr.Number(label="CI (L/min/m²) (optional)")
                    with gr.Row():
                        pvr_WU = gr.Number(label="PVR (WU) (optional)")
                        v_wave_present = gr.Checkbox(label="prominente v-Welle?", value=False)
                        step_up_present = gr.Checkbox(label="Sättigungssprung/Step-up?", value=False)

                with gr.Accordion("Volumenchallenge", open=False):
                    volume_done = gr.Checkbox(label="Volumenchallenge durchgeführt?", value=False)
                    with gr.Row():
                        volume_pawp_baseline = gr.Number(label="PAWP vor Volumen (mmHg)", visible=False)
                        volume_pawp_post = gr.Number(label="PAWP nach Volumen (mmHg)", visible=False)

                with gr.Accordion("Vasoreagibilität", open=False):
                    vasoreactivity_done = gr.Checkbox(label="Vasoreagibilitätstest durchgeführt?", value=False)
                    vasoreactivity_positive = gr.Checkbox(label="Vasoreagibilität positiv?", value=False, visible=False)

                with gr.Accordion("Belastung", open=False):
                    exercise_done = gr.Checkbox(label="Belastung durchgeführt?", value=False)
                    with gr.Row():
                        ex_mpap_rest = gr.Number(label="mPAP Ruhe (Belastungsteil)", visible=False)
                        ex_mpap_peak = gr.Number(label="mPAP Peak", visible=False)
                        ex_pawp_rest = gr.Number(label="PAWP Ruhe (Belastungsteil)", visible=False)
                        ex_pawp_peak = gr.Number(label="PAWP Peak", visible=False)
                    with gr.Row():
                        ex_spap_rest = gr.Number(label="sPAP Ruhe (optional)", visible=False)
                        ex_spap_peak = gr.Number(label="sPAP Peak", visible=False)
                    with gr.Row():
                        ex_co_rest = gr.Number(label="CO Ruhe (optional)", visible=False)
                        ex_co_peak = gr.Number(label="CO Peak (optional)", visible=False)
                        ex_ci_rest = gr.Number(label="CI Ruhe (optional)", visible=False)
                        ex_ci_peak = gr.Number(label="CI Peak (optional)", visible=False)

                with gr.Accordion("Vor-RHK / Vergleich", open=False):
                    prev_rhk_date = gr.Textbox(label="Vor-RHK Datum (Freitext)")
                    prev_course = gr.Textbox(label="Verlauf/Kommentar (z.B. stabil, gebessert, verschlechtert)")
                    with gr.Row():
                        prev_mpap = gr.Number(label="Vor-RHK mPAP")
                        prev_pawp = gr.Number(label="Vor-RHK PAWP")
                        prev_ci = gr.Number(label="Vor-RHK CI")
                        prev_pvr = gr.Number(label="Vor-RHK PVR")

            with gr.Tab("4) Funktion & Scores"):
                with gr.Row():
                    who_fc = gr.Dropdown(label="WHO-FC", choices=["I", "II", "III", "IV"], value=None)
                    six_mwd_m = gr.Number(label="6MWD (m)")
                    cpet_ve_vco2 = gr.Number(label="CPET VE/VCO₂")
                    cpet_vo2max = gr.Number(label="CPET VO₂max")

                with gr.Accordion("HFpEF (H2FPEF, kontinuierlich)", open=False):
                    with gr.Row():
                        hfpef_af = gr.Checkbox(label="Vorhofflimmern", value=False)
                        af_known = gr.Checkbox(label="VHF bekannt (Anamnese)", value=False)
                        hfpef_e_eprime = gr.Number(label="E/e' Ratio")
                        hfpef_pasp = gr.Number(label="PASP (mmHg)")
                        lvef_percent = gr.Number(label="LVEF (%)")

                with gr.Accordion("REVEAL 2 lite (Eingaben)", open=False):
                    with gr.Row():
                        reveal_rbsys = gr.Number(label="RRsys (mmHg)")
                        reveal_hr = gr.Number(label="HF (bpm)")
                        reveal_egfr = gr.Number(label="eGFR (ml/min/1,73m²)")

            with gr.Tab("5) Module & Ausgabe"):
                with gr.Accordion("Zusatz-Module (P-Module) und Freitext", open=True):
                    modules = gr.CheckboxGroup(
                        label="Module auswählen (werden in die Empfehlungen integriert)",
                        choices=MODULE_CHOICES,
                        value=[],
                    )
                    final_free_text = gr.Textbox(label="Abschluss (Therapie/Procedere, Freitext)", lines=4)

                with gr.Row():
                    doctor_out = gr.Textbox(label="Arztbefund", lines=22)
                with gr.Row():
                    patient_out = gr.Textbox(label="Patientenbefund (Einfache Sprache)", lines=18)
                with gr.Row():
                    internal_out = gr.Textbox(label="Intern (Debug/Regeln)", lines=10)

                with gr.Row():
                    btn_pdf_doctor = gr.Button("⬇️ PDF Arztbefund")
                    file_pdf_doctor = gr.File(label="Download Arzt-PDF", visible=False)
                    btn_pdf_patient = gr.Button("⬇️ PDF Patientenbefund")
                    file_pdf_patient = gr.File(label="Download Patienten-PDF", visible=False)

        # Bottom actions
        with gr.Row():
            btn_example_bottom = gr.Button("🧪 Beispiel laden", variant="secondary")
            btn_generate_bottom = gr.Button("🧾 Befund erstellen / aktualisieren", variant="primary")
            btn_save_bottom = gr.Button("💾 Fall speichern", variant="secondary")
            file_case_bottom = gr.File(label="Fall laden (JSON)", file_types=[".json"])
            btn_load_bottom = gr.Button("📂 Fall laden", variant="secondary")

        file_save_top = gr.File(label="Download Fall-JSON", visible=False)
        file_save_bottom = gr.File(label="Download Fall-JSON", visible=False)

        # Input components in EXACT UI_FIELDS order
        input_components = [
            # Stammdaten
            firstname, lastname, sex, dob, age, height_cm, weight_kg, bsa_m2, bmi,
            story, ph_known, ph_suspected,

            # Labor / Klinik
            inr, quick, krea_mg_dl, hst, ptt, thrombos, hb_g_dl, crp, leukos,
            bnp_pg_ml, ntprobnp_pg_ml, entresto, congestive_organopathy,
            sym_syncope, sym_dizziness, sym_hemoptysis, sym_stairs_flights, sym_walk_distance,

            # BGA / LTOT
            ltot_present, ltot_flow_lpm,
            bga_rest_po2, bga_rest_pco2,
            bga_ex_po2, bga_ex_pco2,
            bga_night_ph, bga_night_be, bga_paused,

            # Infektiologie / Immunologie
            virology_positive, virology_details,
            immunology_positive, immunology_details,

            # Abdomen / Leber
            abdomen_sono_done, portal_htn, abdomen_findings,

            # Bildgebung
            ct_angio_done,
            ct_ild, ct_emphysem, ct_embolie, ct_mosaik, ct_chronic_thrombo, ct_koronarkalk,
            vq_done, vq_positive,
            ild_type, ild_histology, ild_fibrosis_clinic, ild_extent,
            cardiac_ventricular, cardiac_pericardial_effusion, cardiac_la_enlarged,

            # Vorerkrankungen
            comorbidities_text, comorbidities_ph_relevant,

            # Medikamente
            ph_meds_current, ph_meds_current_which, ph_meds_current_since,
            ph_meds_past, ph_meds_past_which,
            other_meds_text, diuretics,

            # Lufu
            lufu_done, lufu_summary_text,
            lufu_obstructive, lufu_restrictive, lufu_diffusion,
            fev1_l, fvc_l, fev1_fvc_pct, tlc_l, rv_l,
            dlco_sb_pct, dlco_va_pct,

            # Echo
            echo_done,
            echo_ra_area_cm2, echo_pericardial_effusion, echo_tapse_mm, echo_spap_mmHg,
            echo_sprime_cm_s, echo_ra_esa_cm2, echo_la_enlarged,
            echo_ivc_diam_mm, echo_ivc_collapse_pct,

            # HFpEF
            hfpef_af, af_known, hfpef_e_eprime, hfpef_pasp, lvef_percent,

            # CMR
            cmr_rvef_percent, cmr_lvef_percent,

            # RHK Meta
            rhk_consent, anticoagulation, access_site, co_preference,

            # Ruhe
            spap_mmHg, dpap_mmHg, mpap_mmHg, pawp_mmHg, rap_mmHg,
            td_co_L_min, fick_co_L_min, ci_L_min_m2, pvr_WU,
            v_wave_present, step_up_present,

            # Volumenchallenge
            volume_done, volume_pawp_baseline, volume_pawp_post,

            # Vaso
            vasoreactivity_done, vasoreactivity_positive,

            # Belastung
            exercise_done,
            ex_mpap_rest, ex_mpap_peak,
            ex_pawp_rest, ex_pawp_peak,
            ex_spap_rest, ex_spap_peak,
            ex_co_rest, ex_co_peak,
            ex_ci_rest, ex_ci_peak,

            # Funktion
            who_fc, six_mwd_m, cpet_ve_vco2, cpet_vo2max,

            # REVEAL
            reveal_rbsys, reveal_hr, reveal_egfr,

            # Anämie
            anemia_morphology,

            # Vergleich
            prev_rhk_date, prev_course, prev_mpap, prev_pawp, prev_ci, prev_pvr,

            # Module + Abschluss
            modules, final_free_text,
        ]
        assert len(input_components) == len(UI_FIELDS), f"UI mismatch: {len(input_components)} comps vs {len(UI_FIELDS)} fields"

        # Wiring
        outputs = [doctor_out, patient_out, internal_out, risk_html, derived_json]

        btn_example_top.click(fn=_load_example, inputs=[], outputs=input_components).then(fn=_generate, inputs=input_components, outputs=outputs)
        btn_example_bottom.click(fn=_load_example, inputs=[], outputs=input_components).then(fn=_generate, inputs=input_components, outputs=outputs)

        btn_load_top.click(fn=_load_case_file, inputs=[file_case_top], outputs=input_components).then(fn=_generate, inputs=input_components, outputs=outputs)
        btn_load_bottom.click(fn=_load_case_file, inputs=[file_case_bottom], outputs=input_components).then(fn=_generate, inputs=input_components, outputs=outputs)

        btn_generate_top.click(fn=_generate, inputs=input_components, outputs=outputs)
        btn_generate_bottom.click(fn=_generate, inputs=input_components, outputs=outputs)

        btn_save_top.click(fn=_save_case, inputs=input_components, outputs=[file_save_top])
        btn_save_bottom.click(fn=_save_case, inputs=input_components, outputs=[file_save_bottom])

        btn_pdf_doctor.click(fn=_pdf_doctor, inputs=[doctor_out], outputs=[file_pdf_doctor])
        btn_pdf_patient.click(fn=_pdf_patient, inputs=[patient_out], outputs=[file_pdf_patient])

        # Dynamic visibility
        # ILD details
        ct_ild.change(fn=lambda x: _vis_bool(x), inputs=[ct_ild], outputs=[ild_type])
        ct_ild.change(fn=lambda x: _vis_bool(x), inputs=[ct_ild], outputs=[ild_histology])
        ct_ild.change(fn=lambda x: _vis_bool(x), inputs=[ct_ild], outputs=[ild_fibrosis_clinic])
        ct_ild.change(fn=lambda x: _vis_bool(x), inputs=[ct_ild], outputs=[ild_extent])

        # V/Q positive only if V/Q done
        vq_done.change(fn=lambda x: _vis_bool(x), inputs=[vq_done], outputs=[vq_positive])

        # Virology / immunology details
        virology_positive.change(fn=lambda x: _vis_bool(x), inputs=[virology_positive], outputs=[virology_details])
        immunology_positive.change(fn=lambda x: _vis_bool(x), inputs=[immunology_positive], outputs=[immunology_details])

        # Abdomen findings
        abdomen_sono_done.change(fn=lambda x: _vis_bool(x), inputs=[abdomen_sono_done], outputs=[abdomen_findings])

        # LTOT flow
        ltot_present.change(fn=lambda x: _vis_bool(x), inputs=[ltot_present], outputs=[ltot_flow_lpm])

        # Lufu summary only if done
        lufu_done.change(fn=lambda x: _vis_bool(x), inputs=[lufu_done], outputs=[lufu_summary_text])

        # PH meds details
        ph_meds_current.change(fn=lambda x: _vis_bool(x), inputs=[ph_meds_current], outputs=[ph_meds_current_which])
        ph_meds_current.change(fn=lambda x: _vis_bool(x), inputs=[ph_meds_current], outputs=[ph_meds_current_since])
        ph_meds_past.change(fn=lambda x: _vis_bool(x), inputs=[ph_meds_past], outputs=[ph_meds_past_which])

        # Volume challenge values
        volume_done.change(fn=lambda x: _vis_bool(x), inputs=[volume_done], outputs=[volume_pawp_baseline])
        volume_done.change(fn=lambda x: _vis_bool(x), inputs=[volume_done], outputs=[volume_pawp_post])

        # Vasoreactivity positive only if test done
        vasoreactivity_done.change(fn=lambda x: _vis_bool(x), inputs=[vasoreactivity_done], outputs=[vasoreactivity_positive])

        # Exercise fields visible if exercise done
        for comp in [ex_mpap_rest, ex_mpap_peak, ex_pawp_rest, ex_pawp_peak, ex_spap_rest, ex_spap_peak, ex_co_rest, ex_co_peak, ex_ci_rest, ex_ci_peak]:
            exercise_done.change(fn=_vis_bool, inputs=[exercise_done], outputs=[comp])

        # Anemia morphology visibility based on sex + Hb
        sex.change(fn=_anemia_visibility, inputs=[sex, hb_g_dl], outputs=[anemia_morphology])
        hb_g_dl.change(fn=_anemia_visibility, inputs=[sex, hb_g_dl], outputs=[anemia_morphology])

        # Ultra-Interaktivität:
        # Jede Nutzer-Eingabe soll die Ausgabe aktualisieren (ohne extra Button).
        # Wir nutzen bevorzugt `.input()` (nur User-Input), sonst fallback `.change()`.
        for comp in input_components:
            if hasattr(comp, "input"):
                comp.input(fn=_generate, inputs=input_components, outputs=outputs)
            else:
                comp.change(fn=_generate, inputs=input_components, outputs=outputs)

    return demo, css


def main():
    demo, css = build_demo()
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False
    )


if __name__ == "__main__":
    main()
