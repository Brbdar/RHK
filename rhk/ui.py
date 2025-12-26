# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import gradio as gr

import rhk_textdb as textdb

from .generator import RHKReportGenerator
from .migrate import build_saved_case, migrate_payload_to_ui
from .version import APP_NAME, APP_VERSION


def build_demo() -> gr.Blocks:
    generator = RHKReportGenerator()

    # --- UI registry: ensures mapping is always consistent ---
    field_components: List[Tuple[str, Any]] = []

    def reg(field_id: str, comp: Any) -> Any:
        field_components.append((field_id, comp))
        return comp

    CSS = """
    /* Apple-ish clean layout */
    .rhk-container { max-width: 1200px; margin: 0 auto; }
    #toolbar_top, #toolbar_bottom {
        position: sticky;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(8px);
        z-index: 50;
        padding: 10px 10px;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
    }
    #toolbar_top { top: 8px; }
    #toolbar_bottom { bottom: 8px; }
    #dashboard {
        position: sticky;
        top: 86px;
        z-index: 20;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px;
    }
    .section-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 12px;
        background: white;
    }
    .small-note { font-size: 12px; opacity: 0.75; }
    """

    with gr.Blocks(css=CSS, title=f"{APP_NAME} v{APP_VERSION}") as demo:
        gr.HTML(f"<div class='rhk-container'><h2 style='margin-bottom:0'>{APP_NAME} <span style='opacity:0.6;font-size:14px'>v{APP_VERSION}</span></h2><div class='small-note'>Decision-support • Nicht als alleinige Entscheidungsgrundlage</div></div>")

        # TOOLBAR TOP
        with gr.Row(elem_id="toolbar_top"):
            with gr.Column(scale=3):
                expert_mode = gr.Checkbox(label="Expert Mode", value=False)
            with gr.Column(scale=7):
                with gr.Row():
                    btn_example = gr.Button("Beispiel laden", variant="secondary")
                    file_load = gr.File(label="Fall laden (JSON)", file_types=[".json"])
                    btn_generate_top = gr.Button("Befund erstellen", variant="primary")
                    btn_save = gr.Button("Fall speichern", variant="secondary")

        # ERROR + VALIDATION
        error_md = gr.Markdown("", visible=False)
        validation_md = gr.Markdown("—", elem_classes=["section-card"])

        # INPUT TABS (Klinik & Labor first)
        with gr.Tabs():
            with gr.Tab("Klinik & Labor"):
                with gr.Row():
                    with gr.Column():
                        reg("last_name", gr.Textbox(label="Name", placeholder="Nachname"))
                        reg("first_name", gr.Textbox(label="Vorname"))
                        reg("birthdate", gr.Textbox(label="Geburtsdatum (YYYY-MM-DD)"))
                    with gr.Column():
                        reg("height_cm", gr.Number(label="Größe (cm)", precision=0))
                        reg("weight_kg", gr.Number(label="Gewicht (kg)", precision=1))
                        reg("story", gr.Textbox(label="Story / Kurz-Anamnese", lines=4))

                with gr.Row():
                    reg("sbp", gr.Number(label="RR syst. (mmHg)", precision=0))
                    reg("dbp", gr.Number(label="RR diast. (mmHg)", precision=0))
                    reg("hr", gr.Number(label="Herzfrequenz (/min)", precision=0))
                    reg("rhythm", gr.Textbox(label="Rhythmus", placeholder="z.B. Sinusrhythmus"))

                with gr.Row():
                    reg("ph_dx_known", gr.Checkbox(label="PH-Diagnose bekannt"))
                    reg("ph_suspected", gr.Checkbox(label="PH-Verdachtsdiagnose"))

                gr.Markdown("### Labor")
                with gr.Row():
                    reg("inr", gr.Number(label="INR"))
                    reg("quick", gr.Number(label="Quick (%)"))
                    reg("krea", gr.Number(label="Kreatinin (mg/dl)"))
                    reg("egfr", gr.Number(label="eGFR (ml/min/1.73m²)"))
                with gr.Row():
                    reg("ptt", gr.Number(label="PTT (s)"))
                    reg("thrombos", gr.Number(label="Thrombozyten (G/l)"))
                    reg("hb", gr.Number(label="Hb (g/dl)"))
                    reg("crp", gr.Number(label="CRP (mg/l)"))
                with gr.Row():
                    reg("leukos", gr.Number(label="Leukozyten (G/l)"))
                    reg("bnp", gr.Number(label="BNP (pg/ml)"))
                    reg("ntprobnp", gr.Number(label="pro-NT-BNP (pg/ml)"))
                    reg("congestive_organopathy", gr.Radio(["nein", "ja"], label="Hinweis auf congestive Organopathie?", value="nein"))

                gr.Markdown("### Blutgase / LTOT")
                with gr.Row():
                    reg("ltot_present", gr.Checkbox(label="LTOT vorhanden"))
                    reg("o2_flow", gr.Number(label="O₂-Fluss (L/min)"))
                    reg("bga_pause", gr.Checkbox(label="BGA pausiert"))
                with gr.Row():
                    reg("bga_po2_rest", gr.Number(label="BGA Ruhe: pO₂"))
                    reg("bga_pco2_rest", gr.Number(label="BGA Ruhe: pCO₂"))
                    reg("bga_po2_ex", gr.Number(label="BGA Belastung: pO₂"))
                    reg("bga_pco2_ex", gr.Number(label="BGA Belastung: pCO₂"))
                with gr.Row():
                    reg("bga_ph_night", gr.Number(label="BGA Nacht: pH"))
                    reg("bga_be_night", gr.Number(label="BGA Nacht: BE"))
                    reg("bga_hypoxemia", gr.Checkbox(label="Hypoxämie-Hinweis (für Logik)"))

                gr.Markdown("### Infektiologie / Immunologie")
                with gr.Row():
                    reg("virology_pos", gr.Radio(["nein", "ja"], label="Virologie positiv?", value="nein"))
                    reg("immunology_pos", gr.Radio(["nein", "ja"], label="Immunologie positiv?", value="nein"))

                gr.Markdown("### Abdomen / Leber")
                with gr.Row():
                    reg("abd_sono_done", gr.Radio(["nein", "ja"], label="Abdomen-Sono vorhanden?", value="nein"))
                    reg("portal_htn", gr.Radio(["nein", "ja"], label="Hinweis auf portale Hypertension?", value="nein"))

                gr.Markdown("### Vorerkrankungen / Medikamente")
                reg("comorbidities", gr.Textbox(label="Relevante Vorerkrankungen (Freitext)", lines=3))
                reg("ph_relevance", gr.Textbox(label="Relevant für PH? (Freitext / ja-nein)", lines=2))

                with gr.Accordion("Medikation (Erweitert)", open=False, visible=False) as acc_meds:
                    with gr.Row():
                        reg("ph_meds_current", gr.Radio(["nein", "ja"], label="PH-Medikation aktuell?", value="nein"))
                        reg("ph_meds_past", gr.Radio(["nein", "ja"], label="PH-Medikation in der Vergangenheit?", value="nein"))
                        reg("diuretics", gr.Radio(["nein", "ja"], label="Diuretika?", value="nein"))
                    reg("ph_meds_which", gr.Textbox(label="Welche PH-Medikation?"))
                    reg("ph_meds_since", gr.Textbox(label="Seit wann?"))
                    reg("other_meds", gr.Textbox(label="Sonstige Medikation (Freitext)", lines=3))

            with gr.Tab("RHK – Ruhe"):
                with gr.Row():
                    reg("exam_type", gr.Dropdown(["Initial", "Verlaufskontrolle"], label="Untersuchungstyp", value="Initial"))
                    reg("rhk_consent", gr.Radio(["nein", "ja"], label="Aufklärung erfolgt?", value="ja"))
                    reg("anticoag", gr.Radio(["nein", "ja"], label="Antikoagulation?", value="nein"))
                with gr.Row():
                    reg("access", gr.Dropdown(["V. jug. dextra", "V. jug. sinistra", "sonstiges"], label="Zugang", value="V. jug. dextra"))
                    reg("measurement_limitations", gr.Textbox(label="Messqualität / Limitationen (Freitext)"))

                gr.Markdown("### Druckwerte / Widerstände")
                with gr.Row():
                    reg("spap", gr.Number(label="sPAP (mmHg)"))
                    reg("dpap", gr.Number(label="dPAP (mmHg)"))
                    reg("mpap", gr.Number(label="mPAP (mmHg)"))
                    reg("pawp", gr.Number(label="PAWP (mmHg)"))
                with gr.Row():
                    reg("rap", gr.Number(label="RAP (mmHg)"))
                    reg("co_td", gr.Number(label="TD-CO (L/min)"))
                    reg("co_fick", gr.Number(label="Fick-CO (L/min)"))
                    reg("ci", gr.Number(label="CI (L/min/m²)"))
                with gr.Row():
                    reg("pvr", gr.Number(label="PVR (WU)"))
                    reg("pvri", gr.Number(label="PVRI (WU·m²)"))
                    reg("svi", gr.Number(label="SVI (ml/m²)"))
                    reg("svo2", gr.Number(label="SvO₂ (%)"))

                gr.Markdown("### Stufenoxymetrie (falls vorhanden)")
                with gr.Row():
                    reg("sat_svc", gr.Number(label="SVC-Sättigung (%)"))
                    reg("sat_ivc", gr.Number(label="IVC-Sättigung (%)"))
                    reg("sat_ra", gr.Number(label="RA-Sättigung (%)"))
                    reg("sat_rv", gr.Number(label="RV-Sättigung (%)"))
                    reg("sat_pa", gr.Number(label="PA-Sättigung (%)"))

            with gr.Tab("Belastung / Manöver"):
                reg("exercise_done", gr.Checkbox(label="Belastung durchgeführt"))
                with gr.Accordion("Belastungswerte", open=False, visible=False) as acc_ex:
                    with gr.Row():
                        reg("spap_peak", gr.Number(label="Peak sPAP (mmHg)"))
                        reg("mpap_peak", gr.Number(label="Peak mPAP (mmHg)"))
                        reg("pawp_peak", gr.Number(label="Peak PAWP (mmHg)"))
                        reg("co_peak", gr.Number(label="Peak CO (L/min)"))

                with gr.Accordion("Vorheriger RHK (optional)", open=False, visible=False) as acc_prev:
                    reg("prev_rhk_label", gr.Textbox(label="Vor-RHK Label (z.B. 03/21)"))
                    with gr.Row():
                        reg("prev_mpap", gr.Number(label="Vor-RHK mPAP"))
                        reg("prev_pawp", gr.Number(label="Vor-RHK PAWP"))
                        reg("prev_ci", gr.Number(label="Vor-RHK CI"))
                        reg("prev_pvr", gr.Number(label="Vor-RHK PVR"))
                    reg("prev_course", gr.Dropdown(["stabiler Verlauf", "gebessert", "progredient"], label="Verlauf", value="stabiler Verlauf"))

            with gr.Tab("Bildgebung"):
                gr.Markdown("### CT / Bildgebung Thorax (CT-Angio)")
                with gr.Row():
                    reg("ct_embolie", gr.Checkbox(label="Pulmonalembolie (CT-Hinweis)"))
                    reg("ct_ild", gr.Checkbox(label="ILD"))
                    reg("ct_emphysema", gr.Checkbox(label="Emphysem"))
                    reg("ct_mosaic", gr.Checkbox(label="Mosaikperfusion"))
                    reg("ct_coronary_calc", gr.Checkbox(label="Koronarkalk"))
                gr.Markdown("### Kardialer Phänotyp (Bildgebung/Echo)")
                with gr.Row():
                    reg("card_ventricular_abn", gr.Checkbox(label="Ventrikulär auffällig"))
                    reg("pericardial_effusion", gr.Checkbox(label="Perikarderguss"))
                    reg("imaging_lae", gr.Checkbox(label="Linksatrium erweitert"))
                with gr.Row():
                    reg("vq_defect", gr.Checkbox(label="V/Q: Perfusionsdefekt / CTEPH-Verdacht"))

            with gr.Tab("Lungenfunktion"):
                reg("lufu_done", gr.Radio(["nein", "ja"], label="Lufu durchgeführt?", value="nein"))
                with gr.Row():
                    reg("lufu_obstructive", gr.Checkbox(label="Obstruktiv"))
                    reg("lufu_restrictive", gr.Checkbox(label="Restriktiv"))
                    reg("lufu_diffusion", gr.Checkbox(label="Diffusionsstörung"))
                gr.Markdown("### Einzelwerte")
                with gr.Row():
                    reg("fev1", gr.Number(label="FEV₁"))
                    reg("fvc", gr.Number(label="FVC"))
                    reg("fev1_fvc", gr.Number(label="FEV₁/FVC"))
                    reg("tlc", gr.Number(label="TLC"))
                with gr.Row():
                    reg("rv", gr.Number(label="RV"))
                    reg("dlco_sb", gr.Number(label="DLCO SB"))
                    reg("dlco_va", gr.Number(label="DLCO SB/VA"))
                reg("lufu_summary", gr.Textbox(label="Lufu Summary (Freitext)", lines=3))

            with gr.Tab("Echo / MRT"):
                gr.Markdown("### Echokardiographie")
                reg("echo_done", gr.Radio(["nein", "ja"], label="Echo-Phänotyp vorhanden?", value="nein"))
                reg("echo_free", gr.Textbox(label="Relevante Echo-Parameter (Freitext)", lines=3))
                with gr.Row():
                    reg("echo_sprime", gr.Number(label="S' (cm/s)"))
                    reg("echo_ra_area", gr.Number(label="RA ESA (cm²)"))
                    reg("echo_pasp", gr.Number(label="Echo: sPAP/PASP (mmHg)"))
                    reg("echo_e_over_eprime", gr.Number(label="E/e'"))

                gr.Markdown("### MRT / CMR")
                with gr.Row():
                    reg("cmr_rvesvi", gr.Number(label="RVESVi"))
                    reg("cmr_svi", gr.Number(label="SVi"))
                    reg("cmr_rvef", gr.Number(label="RVEF (%)"))

            with gr.Tab("Scores / Funktion"):
                with gr.Row():
                    reg("who_fc", gr.Dropdown(["I", "II", "III", "IV"], label="WHO-FC"))
                    reg("syncope", gr.Radio(["nein", "ja"], label="Synkope?", value="nein"))
                    reg("mwd", gr.Number(label="6MWD (m)"))
                with gr.Row():
                    reg("cpet_ve_vco2", gr.Number(label="CPET VE/VCO₂"))
                    reg("cpet_vo2max", gr.Number(label="CPET VO₂max"))
                with gr.Accordion("HFpEF-Score (H2FPEF) – Parameter", open=False, visible=False) as acc_hfpef:
                    reg("antihypertensive_count", gr.Number(label="Anzahl Blutdruck-Medikamente", precision=0))

            with gr.Tab("Procedere / Module"):
                gr.Markdown("### Procedere-Module (Pxx)")
                # Build module choices from textdb.P_BLOCKS
                module_choices = sorted(list(getattr(textdb, "P_BLOCKS", {}).keys()))
                module_labels = [f"{mid} – {textdb.P_BLOCKS[mid].title}" for mid in module_choices]
                # We'll store labels in UI, convert to ids
                modules_comp = reg("modules", gr.CheckboxGroup(choices=module_labels, label="Zusätzliche Module auswählen"))

                reg("therapy_plan", gr.Textbox(label="Therapieplan (Freitext / optional)", lines=2))
                reg("therapy_escalation", gr.Textbox(label="Therapie-Eskalation (Freitext / optional)", lines=2))
                reg("risk_profile_desc", gr.Textbox(label="Risikoprofil (Freitext / optional)", lines=2))
                reg("patient_preference", gr.Textbox(label="Patient:innenpräferenz (Freitext / optional)", lines=2))

        # DASHBOARD (always visible)
        with gr.Column(elem_id="dashboard"):
            risk_html = gr.HTML("<div>—</div>")
            gr.Markdown("**Live-Checks**")
            # validation_md already exists above; re-use there but keep dashboard compact
            live_keyvals = gr.Markdown("—")

        # OUTPUTS
        gr.Markdown("## Befunde")
        with gr.Tabs():
            with gr.Tab("Arztbefund (Markdown)"):
                out_doctor = gr.Markdown("—")
            with gr.Tab("Patientenbericht"):
                out_patient = gr.Textbox(value="—", lines=22)
            with gr.Tab("Intern"):
                out_internal = gr.Markdown("—")

        # TOOLBAR BOTTOM
        with gr.Row(elem_id="toolbar_bottom"):
            btn_generate_bottom = gr.Button("Befund erstellen", variant="primary")
            btn_example_bottom = gr.Button("Beispiel laden", variant="secondary")
            btn_save_bottom = gr.Button("Fall speichern", variant="secondary")

        file_download = gr.File(label="Download (JSON)")

        # --- helpers ---
        def _labels_to_ids(labels: List[str]) -> List[str]:
            if not labels:
                return []
            out: List[str] = []
            for lab in labels:
                if not isinstance(lab, str):
                    continue
                mid = lab.split("–")[0].strip()
                if mid:
                    out.append(mid)
            return out

        def _ui_get_raw(*vals):
            ui = {fid: v for (fid, _), v in zip(field_components, vals)}
            # convert module labels to ids
            ui["modules"] = _labels_to_ids(ui.get("modules") or [])
            return ui

        def _make_live_keyvals(ui: Dict[str, Any]) -> str:
            # lightweight display; avoid heavy calcs here (generator will compute anyway)
            parts = []
            for k, label in [("mpap","mPAP"),("pawp","PAWP"),("rap","RAP"),("ci","CI"),("pvr","PVR")]:
                v = ui.get(k)
                if v is None or v == "":
                    continue
                parts.append(f"- **{label}**: {v}")
            return "\n".join(parts) if parts else "—"

        def _generate(*vals):
            try:
                ui = _ui_get_raw(*vals)
                # live keyvals
                lk = _make_live_keyvals(ui)
                # run generator
                doctor_md, patient_txt, internal_md, risk_dash, val_md = generator.generate_all(ui)
                return doctor_md, patient_txt, internal_md, risk_dash, lk, gr.update(value=val_md), gr.update(visible=False, value="")
            except Exception:
                tb = traceback.format_exc()
                return "—", "—", "—", "<div>—</div>", "—", gr.update(value="—"), gr.update(visible=True, value=f"### Fehler\n```\n{tb}\n```")

        def _load_example():
            example = {
                "last_name":"Mustermann",
                "first_name":"Max",
                "birthdate":"1965-05-12",
                "height_cm":178,
                "weight_kg":86,
                "story":"Belastungsdyspnoe, Ausschluss/Einordnung einer PH.",
                "sbp":122,
                "dbp":74,
                "hr":78,
                "rhythm":"Sinusrhythmus",
                "mpap":32,
                "pawp":10,
                "rap":7,
                "co_td":4.9,
                "spap":52,
                "dpap":18,
                "svo2":63,
                "exercise_done":True,
                "mpap_peak":44,
                "pawp_peak":14,
                "co_peak":7.2,
                "spap_peak":82,
                "ct_mosaic":True,
                "vq_defect":True,
                "lufu_obstructive":False,
                "lufu_restrictive":False,
                "lufu_diffusion":True,
                "echo_sprime":11.0,
                "echo_ra_area":22.0,
                "echo_pasp":55,
                "echo_e_over_eprime":12,
                "who_fc":"III",
                "mwd":320,
                "ntprobnp":850,
                "egfr":68,
                "antihypertensive_count":2,
            }
            # modules: default empty
            values = []
            for fid, _ in field_components:
                if fid == "modules":
                    values.append([])  # label list
                else:
                    values.append(example.get(fid))
            return values

        def _load_case(file_obj):
            if not file_obj:
                return [None for _ in field_components]
            try:
                # gradio may pass a tempfile object, a dict-like FileData, or a path
                if isinstance(file_obj, str):
                    path = file_obj
                elif hasattr(file_obj, "name") and isinstance(getattr(file_obj, "name"), str):
                    path = file_obj.name
                elif isinstance(file_obj, dict) and isinstance(file_obj.get("name"), str):
                    path = file_obj["name"]
                else:
                    path = None
                content = file_obj.read() if hasattr(file_obj, "read") else (open(path, "rb").read() if path else b"")
                payload = json.loads(content.decode("utf-8"))
                ui, msg = migrate_payload_to_ui(payload)
            except Exception:
                ui, msg = ({}, "Fehler beim Laden der Datei.")
            # Map to values list
            values = []
            for fid, comp in field_components:
                if fid == "modules":
                    # ui stores ids; convert to labels
                    ids = ui.get("modules") or []
                    labels = []
                    for mid in ids:
                        if mid in textdb.P_BLOCKS:
                            labels.append(f"{mid} – {textdb.P_BLOCKS[mid].title}")
                    values.append(labels)
                else:
                    values.append(ui.get(fid))
            return values

        def _save_case(*vals):
            ui = _ui_get_raw(*vals)
            payload = build_saved_case(ui)
            fd, path = tempfile.mkstemp(prefix="rhk_case_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path

        # Bind actions
        input_components = [c for _, c in field_components]

        btn_generate_top.click(_generate, inputs=input_components, outputs=[out_doctor, out_patient, out_internal, risk_html, live_keyvals, validation_md, error_md])
        btn_generate_bottom.click(_generate, inputs=input_components, outputs=[out_doctor, out_patient, out_internal, risk_html, live_keyvals, validation_md, error_md])

        btn_example.click(_load_example, outputs=input_components)
        btn_example_bottom.click(_load_example, outputs=input_components)

        file_load.change(_load_case, inputs=[file_load], outputs=input_components)

        btn_save.click(_save_case, inputs=input_components, outputs=[file_download])
        btn_save_bottom.click(_save_case, inputs=input_components, outputs=[file_download])

        # Expert mode: show/hide advanced accordions (meds, hfpef, exercise, prev)
        def _toggle_expert(on: bool):
            return (
                gr.update(visible=on),
                gr.update(visible=on),
                gr.update(visible=on),
                gr.update(visible=on),
            )
        expert_mode.change(
            _toggle_expert,
            inputs=[expert_mode],
            outputs=[acc_meds, acc_ex, acc_prev, acc_hfpef],
        )

    return demo