#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_textdb.py  (Befunddatenbank)
Textbaustein-Datenbank für den RHK-Befundassistenten.

Enthält:
- "Katalog v1.0 (K01–K20)" als Paket-Bausteine (jeweils Beurteilung + Empfehlung)
- Zusatzmodule (BZ.. / BE.. / C.. / ALT..)
- Legacy-Blöcke (B../E../P../Z../G..) aus der bisherigen v1-Struktur (für Rückwärtskompatibilität)
- Default-Cut-offs (leitlinien-/literaturbasiert, aber bewusst konfigurierbar)

WICHTIG (Sicherheitslogik im Hauptprogramm):
- Medikamenten-/Interventionsnamen sollen nur ausgegeben werden, wenn diese im Input als geplant/gewünscht vorliegen.
  Deshalb sind pharmakologische Beispiele in diesem Katalog – wo möglich – als Variante hinterlegt
  oder über Platzhalter-Sätze ein-/ausblendbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Iterable, Tuple


# ---------------------------------------------------------------------------
# Default-Regeln / Cut-offs (vom Hauptprogramm importierbar)
# ---------------------------------------------------------------------------

DEFAULT_RULES: dict = {
    # Ruhe-Definitionen (ESC/ERS 2022): PH >20 mmHg; pre-cap: PAWP ≤15 & PVR >2
    "rest": {
        "mPAP_ph_mmHg": 20,
        "PAWP_postcap_mmHg": 15,
        "PVR_precap_WU": 2.0,
    },
    # Belastung (Exercise-RHC): mPAP/CO slope >3 mmHg/L/min; PAWP/CO slope >2 spricht eher für Linksherz-Komponente
    "exercise": {
        "mPAP_CO_slope_mmHg_per_L_min": 3.0,
        "PAWP_CO_slope_mmHg_per_L_min": 2.0,
    },
    # "Schweregrad" der Widerstände ist nicht einheitlich standardisiert.
    # Default hier: an deiner Vorgabe orientiert (PVR ≥10 WU = schwer), aber im Hauptprogramm konfigurierbar.
    "severity": {
        "PVR_WU": {
            "mild_ge": 2.0,
            "moderate_ge": 5.0,     # pragmatisch; passt u.a. zum "severe group-3 PH"-Konzept (PVR >5)
            "severe_ge": 10.0,      # User-Vorgabe (anpassbar)
        },
        "CI_L_min_m2": {
            "normal_ge": 2.5,
            "reduced_lt": 2.5,
            "severely_reduced_lt": 2.0,
        },
    },
    # Echo-Add-ons
    "echo": {
        # Yogeswaran et al.: S'/RAAi Cut-off ca. 0.81 m^2/(s·cm) (Interpretation im klinischen Kontext)
        "Sprime_RAAI_cutoff_m2_per_s_cm": 0.81,
    },
    # Thal et al.: ΔsPAP = sPAP_peak - sPAP_rest (keine feste harte Schwelle; als Zusatzparameter dokumentieren)
    "exercise_addons": {
        "delta_sPAP_definition": "sPAP_peak_mmHg - sPAP_rest_mmHg",
    },
}


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextBlock:
    """
    Ein Textbaustein (Template).
    template ist ein Python-Format-String (str.format_map).
    """
    id: str
    title: str
    applies_to: str
    template: str
    category: str = "MISC"
    variants: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Legacy-Blöcke (B.../E.../P.../Z.../G.../ALT...) – unverändert/leicht ergänzt
# ---------------------------------------------------------------------------

# -------------------------
# Kern-Bausteine (B...)
# -------------------------

B_BLOCKS: Dict[str, TextBlock] = {
    "B01": TextBlock(
        id="B01",
        title="Normalbefund (kein PH-Hinweis in Ruhe)",
        applies_to="mPAP/PAWP/PVR unauffällig, keine Stauung",
        template=(
            "Normale pulmonale Druck- und Widerstandswerte {ci_phrase} nach {co_method_desc}. "
            "Keine zentralvenöse oder pulmonalvenöse Stauung. {step_up_sentence}"
        ),
        category="B",
    ),
    "B02": TextBlock(
        id="B02",
        title="Kein PH in Ruhe + Belastung unauffällig",
        applies_to="Belastungsprotokoll vorhanden, Slopes unauffällig",
        template=(
            "In Ruhe normale pulmonale Druck- und Widerstandswerte {ci_phrase}. "
            "Unter ergometrischer Belastung keine pathologische Druck-/Fluss-Reaktion "
            "(mPAP/CO-Slope {mPAP_CO_slope}, PAWP/CO-Slope {PAWP_CO_slope}). "
            "{step_up_sentence} Keine Hinweise auf zentrale oder pulmonalvenöse Stauung."
        ),
        category="B",
    ),
    "B03": TextBlock(
        id="B03",
        title="Kein PH in Ruhe, aber pathologische Belastungsreaktion (eher linkskardial)",
        applies_to="Ruhe unauffällig; unter Belastung mPAP/CO und PAWP/CO erhöht (PAWP/CO spricht für Linksherz)",
        template=(
            "Keine pulmonale Druck- und Widerstandserhöhung in Ruhe {ci_phrase} nach {co_method_desc}. "
            "Keine zentralvenöse oder pulmonalvenöse Stauung. "
            "Unter Belastung zeigt sich eine pathologische mPAP/CO-Slope von {mPAP_CO_slope} "
            "bei pathologischer PAWP/CO-Slope von {PAWP_CO_slope}. "
            "{step_up_sentence}"
        ),
        category="B",
    ),
    "B04": TextBlock(
        id="B04",
        title="Grenzwertige/milde Druckerhöhung ohne klare Zuordnung",
        applies_to="mild/grenzwertig; PVR wenig erhöht oder uneindeutig; CI leicht reduziert",
        template=(
            "Leichtgradig erhöhte pulmonale Druckwerte bei {pvr_phrase} und {ci_phrase} nach {co_method_desc}. "
            "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence} "
            "Insgesamt grenzwertige Hämodynamik, die in Zusammenschau mit Klinik und Zusatzdiagnostik zu interpretieren ist."
        ),
        category="B",
    ),
    "B05": TextBlock(
        id="B05",
        title="Präkapilläre PH – leicht",
        applies_to="präkapilläre Konstellation (PAWP nicht erhöht), PVR erhöht, mPAP erhöht",
        template=(
            "Pulmonale Druck- und Widerstandserhöhung bei präkapillärer Konstellation "
            "({mpap_phrase}, {pawp_phrase}, {pvr_phrase}) {ci_phrase} nach {co_method_desc}. "
            "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence}"
        ),
        category="B",
    ),
    "B06": TextBlock(
        id="B06",
        title="Präkapilläre PH – mittelgradig",
        applies_to="mPAP/PVR deutlich erhöht, PAWP nicht erhöht",
        template=(
            "Deutlich erhöhte pulmonale Druck- und Widerstandswerte bei präkapillärer Konstellation "
            "({mpap_phrase}, {pawp_phrase}, {pvr_phrase}) {ci_phrase} nach {co_method_desc}. "
            "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence}"
        ),
        category="B",
    ),
    "B07": TextBlock(
        id="B07",
        title="Präkapilläre PH – schwer (mit RV-Belastung)",
        applies_to="PVR hoch, CI reduziert, ZVD häufig erhöht",
        template=(
            "Ausgeprägte pulmonale Druck- und Widerstandserhöhung bei präkapillärer Konstellation "
            "({mpap_phrase}, {pawp_phrase}, {pvr_phrase}) {ci_phrase} nach {co_method_desc}. "
            "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence}"
        ),
        category="B",
    ),
    "B08": TextBlock(
        id="B08",
        title="Präkapilläre PH – Diskrepanz CI (Thermodilution vs. Fick)",
        applies_to="CI_TD und CI_Fick deutlich verschieden; Verdacht Messlimit (TI/Arrhythmie/VO2-Schätzung)",
        template=(
            "Pulmonale Druck- und Widerstandserhöhung bei diskrepanten Cardiac-Index-Werten "
            "(Thermodilution: {CI_TD}; Fick: {CI_Fick}). "
            "Die Interpretation des Herzzeitvolumens ist aufgrund {co_discrepancy_reason} eingeschränkt. "
            "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence}"
        ),
        category="B",
    ),
    "B09": TextBlock(
        id="B09",
        title="Postkapilläre PH (isoliert) – linksatrial/linksventrikulär führend",
        applies_to="PAWP erhöht, PVR nicht überproportional",
        template=(
            "Pulmonale Druckerhöhung bei Hinweis auf pulmonalvenöse Stauung ({pawp_phrase}) und {pvr_phrase}. "
            "{ci_phrase} nach {co_method_desc}. {cv_stauung_phrase} {step_up_sentence} "
            "Hämodynamisch Hinweis auf eine führend linkskardiale Genese."
        ),
        category="B",
    ),
    "B10": TextBlock(
        id="B10",
        title="Kombinierte prä- und postkapilläre PH (cpcPH / gemischt)",
        applies_to="PAWP erhöht + PVR deutlich erhöht",
        template=(
            "Pulmonale Druckerhöhung bei Hinweis auf pulmonalvenöse Stauung ({pawp_phrase}) "
            "und gleichzeitig deutlich erhöhten pulmonalvaskulären Widerstandswerten ({pvr_phrase}). "
            "{ci_phrase} nach {co_method_desc}. {cv_stauung_phrase} {step_up_sentence} "
            "Insgesamt Konstellation einer kombinierten prä- und postkapillären pulmonalen Hypertonie."
        ),
        category="B",
    ),
    "B11": TextBlock(
        id="B11",
        title="Volumenbelastung: HFpEF-Unmasking (PAWP-Anstieg)",
        applies_to="Volumenbolus dokumentiert, PAWP steigt deutlich",
        template=(
            "Unter Volumenbelastung ({volume_challenge_desc}) zeigt sich ein Anstieg des PAWP von {PAWP_pre} auf {PAWP_post} "
            "bei gleichzeitigem Anstieg von mPAP {mPAP_pre} → {mPAP_post}. "
            "Dies spricht für eine relevante linkskardiale Komponente (Belastungs-/Volumen-assoziierte pulmonalvenöse Stauung)."
        ),
        category="B",
    ),
    "B12": TextBlock(
        id="B12",
        title="Volumenbelastung ohne relevante PAWP-Änderung",
        applies_to="Volumenbolus, PAWP bleibt stabil",
        template=(
            "Unter Volumenbelastung ({volume_challenge_desc}) kein relevanter Anstieg des PAWP ({PAWP_pre} → {PAWP_post}). "
            "Damit kein klarer Hinweis auf eine volumeninduzierbare pulmonalvenöse Stauung im Untersuchungssetting."
        ),
        category="B",
    ),
    "B13": TextBlock(
        id="B13",
        title="Vasoreaktivitätstest/iNO: Responderkriterien NICHT erfüllt",
        applies_to="iNO gegeben, keine/ungenügende Response",
        template=(
            "Unter iNO-Gabe ({iNO_ppm}) {iNO_o2_desc} keine/geringe Abnahme der pulmonalen Druck- und Widerstandswerte. "
            "Responderkriterien: {iNO_responder_statement}."
        ),
        category="B",
    ),
    "B14": TextBlock(
        id="B14",
        title="Vasoreaktivitätstest/iNO: Responderkriterien erfüllt",
        applies_to="deutliche Drucksenkung und adäquater CO-Erhalt (hausübliche Definition)",
        template=(
            "Unter iNO-Gabe ({iNO_ppm}) {iNO_o2_desc} zeigt sich eine deutliche Abnahme der pulmonalen Druck- und Widerstandswerte "
            "bei erhaltener/steigender Förderleistung. Responderkriterien: {iNO_responder_statement}."
        ),
        category="B",
    ),
    "B15": TextBlock(
        id="B15",
        title="Prominente V-Welle im PAWP (Hinweis auf MI)",
        applies_to="prominente V-Welle dokumentiert",
        template=(
            "Hinweis auf pulmonalvenöse Komponente mit prominenter V-Welle im PAWP-Signal ({V_wave_desc}). "
            "Konstellation vereinbar mit z.B. hämodynamisch relevanter Mitralinsuffizienz; echokardiographische Korrelation empfohlen."
        ),
        category="B",
    ),
    "B16": TextBlock(
        id="B16",
        title="Sättigungssprung vorhanden (Shuntverdacht)",
        applies_to="Stufenoxymetrie mit relevantem Sprung",
        template=(
            "In der Stufenoxymetrie zeigt sich ein relevanter Sättigungssprung {step_up_location_desc} "
            "(Anhalt für Links-Rechts-Shunt)."
        ),
        category="B",
    ),
    "B17": TextBlock(
        id="B17",
        title="Kein Sättigungssprung",
        applies_to="Stufenoxymetrie ohne Sprung",
        template="Kein relevanter Sättigungssprung in der Stufenoxymetrie.",
        category="B",
    ),
    "B18": TextBlock(
        id="B18",
        title="Schwere TI / Pseudo-Dip / Messartefakt",
        applies_to="massive TI, Pseudo-Dip, RV-Kurve auffällig",
        template=(
            "Die Druckkurven zeigen Hinweise auf Pseudo-Dip bei ausgeprägter Trikuspidalinsuffizienz. "
            "Die hämodynamische Interpretation (insb. Thermodilution und RV-Kurvenmorphologie) ist dadurch potenziell eingeschränkt. "
            "Echokardiographische Korrelation empfohlen."
        ),
        category="B",
    ),
    "B19": TextBlock(
        id="B19",
        title="Zentralvenöse Stauung führend (ohne pulmonalvenöse Stauung)",
        applies_to="RAP hoch, PAWP nicht hoch",
        template=(
            "Ausgeprägte zentralvenöse Stauung bei fehlender/geringer pulmonalvenöser Stauung. "
            "Dies spricht für eine führende rechtskardiale Volumen-/Druckbelastung."
        ),
        category="B",
    ),
    "B20": TextBlock(
        id="B20",
        title="Pulmonalvenöse Stauung führend",
        applies_to="PAWP hoch (± V-Wellen)",
        template=(
            "Hinweis auf pulmonalvenöse Stauung ({pawp_phrase}) {V_wave_short}. "
            "{cv_stauung_phrase} Dies spricht für eine relevante linkskardiale Komponente."
        ),
        category="B",
    ),
    "B21": TextBlock(
        id="B21",
        title="Keine PH in Ruhe – Belastungsreaktion eher pulmonalvaskulär",
        applies_to="Ruhewerte unauffällig; unter Belastung mPAP/CO-Slope pathologisch bei nicht führender PAWP/CO-Slope",
        template=(
            "Keine pulmonale Druck- und Widerstandserhöhung in Ruhe {ci_phrase}. "
            "Keine zentrale oder pulmonalvenöse Stauung in Ruhe. "
            "Unter Belastung zeigt sich eine pathologische mPAP/CO-Slope {mPAP_CO_slope} bei nicht führend erhöhter "
            "PAWP/CO-Slope {PAWP_CO_slope}, vereinbar mit einer auffälligen pulmonalvaskulären Belastungsreaktion."
        ),
        category="B",
    ),
}

# -------------------------
# Empfehlung / Einordnung (E...)
# -------------------------

E_BLOCKS: Dict[str, TextBlock] = {
    "E01": TextBlock(
        id="E01",
        title="Ausschluss einer PH in Ruhe (Standard)",
        applies_to="Normalbefund in Ruhe",
        template=(
            "Aktuell Ausschluss einer pulmonalen Hypertonie in Ruhe. "
            "Aus hämodynamischer Sicht ergeben sich derzeit keine Hinweise auf eine pulmonalvaskuläre Problematik. "
            "Weitere Abklärung richtet sich nach Symptomatik und Begleitbefunden (z.B. Lungenfunktion, Bildgebung, Echokardiographie)."
        ),
        category="E",
    ),
    "E02": TextBlock(
        id="E02",
        title="Keine PH in Ruhe, auffällige Belastungsreaktion – eher linkskardial",
        applies_to="B03 passend",
        template=(
            "In Ruhe kein Hinweis auf eine pulmonale Hypertonie. Unter Belastung pathologische Druck-/Fluss-Reaktion, "
            "die aufgrund der PAWP/CO-Slope eher durch eine linkskardiale Limitation (z.B. HFpEF/diastolische Dysfunktion/klappenvitiumassoziiert) "
            "als durch ein primär pulmonalvaskuläres Problem erklärt werden kann. "
            "Kardiologische Mitbeurteilung inkl. Echokardiographie (ggf. Belastungsecho) empfohlen."
        ),
        category="E",
    ),
    "E03": TextBlock(
        id="E03",
        title="Leichte präkapilläre PH – DD-Baustein (Gruppe I–IV)",
        applies_to="präkapilläre Konstellation, Ätiologie offen",
        template=(
            "Erstdiagnose/Bestätigung einer präkapillären pulmonalen Hypertonie. "
            "Differenzialdiagnostisch kommen je nach Gesamtkonstellation eine PAH (Gruppe I), "
            "eine PH bei Linksherzerkrankung (Gruppe II; v.a. bei grenzwertiger PAWP/Belastungs-/Volumenreaktion), "
            "eine PH bei Lungenerkrankung (Gruppe III) sowie eine chronisch thromboembolische Ursache (Gruppe IV/CTEPD/CTEPH) in Betracht. "
            "Strukturierte Komplettierung der Diagnostik (Lufu/DLCO, Bildgebung, V/Q, Echo, Labor inkl. Autoimmun/Infekt) empfohlen."
        ),
        category="E",
    ),
    "E04": TextBlock(
        id="E04",
        title="PAH (Gruppe I) bei Kollagenose/CTD – Verlauf/Progress",
        applies_to="CTD bekannt, präkapillär",
        template=(
            "Bekannte präkapilläre pulmonale Hypertonie am ehesten der Gruppe I bei {ctd_desc}. "
            "Unter bestehender spezifischer Therapie zeigt sich {hemodynamic_course_desc}. "
            "Risikoeinschätzung in Zusammenschau mit Klinik (z.B. NYHA, BNP/NT-proBNP) und Echo empfohlen."
        ),
        category="E",
    ),
    "E05": TextBlock(
        id="E05",
        title="PAH – Eskalation grundsätzlich indiziert, aber Kontra/Präferenz/Setting",
        applies_to="Risiko intermediär/hoch, aber z.B. Stauung führend oder Patientenwunsch",
        template=(
            "Bei {risk_profile_desc} besteht grundsätzlich eine Indikation zur Therapieeskalation. "
            "Vor dem Hintergrund von {limiting_factor_desc} empfehlen wir aktuell jedoch primär {primary_focus_desc} "
            "und eine zeitnahe Reevaluation."
        ),
        category="E",
    ),
    "E06": TextBlock(
        id="E06",
        title="PH bei Linksherzerkrankung (Gruppe II) / HFpEF-Baustein",
        applies_to="postkapillär oder Belastungs-/Volumen-PAWP-Anstieg",
        template=(
            "Die Hämodynamik spricht am ehesten für eine führend linkskardiale Genese "
            "(postkapilläre PH bzw. Belastungs-/Volumen-assoziierte pulmonalvenöse Stauung). "
            "Im Vordergrund stehen kardiologische Abklärung und Optimierung der Herzinsuffizienztherapie "
            "(Volumenmanagement, Blutdruck-/Rhythmuskontrolle, Behandlung von Klappenvitien, Risikofaktoren). "
            "PH-spezifische Therapie ist bei dieser Konstellation in der Regel nicht primär indiziert und sollte – "
            "falls diskutiert – nur im spezialisierten Setting nach strenger Indikationsprüfung erfolgen."
        ),
        category="E",
    ),
    "E07": TextBlock(
        id="E07",
        title="Kombinierte prä-/postkapilläre PH (cpcPH) – Zwei-Säulen-Strategie",
        applies_to="B10 passend",
        template=(
            "Es zeigt sich eine kombinierte prä- und postkapilläre pulmonale Hypertonie. "
            "Therapeutisch sollte neben konsequentem HFpEF-/Linksherz-Management "
            "(Volumenstatus, Blutdruck, Rhythmus, Klappenvitien) die präkapilläre Komponente im spezialisierten Setting beurteilt werden. "
            "Bei überproportionaler präkapillärer Komponente (z.B. deutlich erhöhtem PVR) kann – "
            "nach Ausschluss relevanter Kontraindikationen – eine PH-spezifische Therapie im Einzelfall erwogen werden; "
            "dies bedarf enger Verlaufskontrollen."
        ),
        category="E",
    ),
    "E08": TextBlock(
        id="E08",
        title="PH bei Lungenerkrankung (Gruppe III) – Fokus Lunge/O2",
        applies_to="COPD/ILD/Restriktion/DLCO-Abfall",
        template=(
            "Die Befundkonstellation ist vereinbar mit einer pulmonalen Hypertonie im Rahmen einer zugrundeliegenden Lungenerkrankung (Gruppe III) "
            "{lung_component_extra}. "
            "Im Vordergrund stehen Optimierung der pneumologischen Therapie, Abklärung/Behandlung von Hypoxämie (ggf. LTOT-Indikation), "
            "ggf. schlafassoziierte Atmungsstörung sowie pulmonale Rehabilitation. "
            "PH-spezifische Therapie ist bei Gruppe-III-PH nur im ausgewählten Einzelfall im spezialisierten Zentrum zu diskutieren."
        ),
        category="E",
    ),
    "E09": TextBlock(
        id="E09",
        title="CTEPH/CTEPD-Verdacht – Standard-Einordnung",
        applies_to="Perfusionsdefekte/VTE-Anamnese, V/Q ausstehend oder auffällig",
        template=(
            "Bei {vte_context_desc} besteht differenzialdiagnostisch der Verdacht auf eine chronisch thromboembolische Genese (CTEPD/CTEPH). "
            "Komplettierung der Diagnostik (V/Q-Szintigraphie bzw. Befundbeurteilung, ggf. CT-Pulmonalisangiographie/PA-Angiographie) "
            "sowie Vorstellung im spezialisierten Zentrum zur Therapieevaluation (operativ/interventionell/medikamentös) empfohlen. "
            "Antikoagulation gemäß Indikation/Vorgeschichte sicherstellen."
        ),
        category="E",
    ),
    "E10": TextBlock(
        id="E10",
        title="Kleiner Perfusionsdefekt – hämodynamisch nicht relevant",
        applies_to="V/Q kleiner Defekt, Hämodynamik unauffällig",
        template=(
            "Ein {perfusion_defect_desc} hat in Zusammenschau mit der unauffälligen Ruhe-Hämodynamik und fehlender pulmonalvaskulärer "
            "Druck-/Widerstandserhöhung wahrscheinlich keine hämodynamische Relevanz. "
            "Weitere Bewertung im Kontext der Gerinnungs-/Bildgebungsbefunde."
        ),
        category="E",
    ),
    "E11": TextBlock(
        id="E11",
        title="Shuntverdacht (Sättigungssprung) – Einordnung + nächste Schritte",
        applies_to="B16 passend",
        template=(
            "Der relevante Sättigungssprung in der Stufenoxymetrie spricht für einen Links-Rechts-Shunt. "
            "Weiterführende Abklärung zur Shuntlokalisation und Quantifizierung (z.B. kontrastverstärkte Echokardiographie/TEE, "
            "ggf. Kardio-MRT, Vorstellung in einem Zentrum für angeborene Herzfehler) empfohlen."
        ),
        category="E",
    ),
    "E12": TextBlock(
        id="E12",
        title="Keine PH in Ruhe, auffällige pulmonalvaskuläre Belastungsreaktion",
        applies_to="B21 passend",
        template=(
            "Aktuell keine PH in Ruhe, jedoch auffällige pulmonalvaskuläre Belastungsreaktion. "
            "Komplettierung der PH-Abklärung (inkl. Lungenfunktion/DLCO, Bildgebung, V/Q-Szintigraphie, Echo, Labor/Autoimmun- und Infektionsscreening "
            "je nach Kontext) empfohlen. Je nach Risikoprofil und Verlauf: engmaschige klinische und echokardiographische Verlaufskontrolle; "
            "erneute invasive Diagnostik bei Progress oder neuen Risikomerkmalen."
        ),
        category="E",
    ),
}

# -------------------------
# Procedere / Maßnahmen (P...)
# -------------------------

P_BLOCKS: Dict[str, TextBlock] = {
    "P01": TextBlock(
        id="P01",
        title="PH-Basisdiagnostik komplettieren (universell)",
        applies_to="unklare PH-Ätiologie, Erstdiagnose oder Reevaluation",
        template=(
            "Strukturierte Komplettierung/Überprüfung der PH-Basisdiagnostik:\n"
            "• Echokardiographie inkl. RV-Funktion, TR-Grad, Klappenvitien, ggf. Belastungsecho.\n"
            "• Lungenfunktion inkl. Bodyplethysmographie und Diffusionskapazität (DLCO) sowie BGA (Ruhe/Belastung nach Verfügbarkeit).\n"
            "• Bildgebung je nach Fragestellung: HR-CT/CT-Thorax, ggf. Beurteilung in radiologisch-pneumologischer Konferenz.\n"
            "• V/Q-Szintigraphie (CTEPD/CTEPH-Ausschluss).\n"
            "• Labor je nach Kontext: Autoimmunserologie (ANA/ENA etc.), Infektiologie (z.B. HIV/Hepatitis nach SOP), "
            "Blutbild/Eisenstatus, Schilddrüse, Leber/Niere, BNP/NT-proBNP."
        ),
        category="P",
    ),
    "P02": TextBlock(
        id="P02",
        title="Diuretische Therapie intensivieren (bei zentralvenöser Stauung/RV-Versagen)",
        applies_to="ausgeprägte ZVD/Ödeme/kardiorenale Dynamik",
        template=(
            "Bei führender zentralvenöser Stauung: Intensivierung der diuretischen Therapie, "
            "idealerweise initial intravenös im stationären Setting mit anschließender Anpassung der Erhaltungsmedikation. "
            "Engmaschige Kontrolle von Nierenfunktion/Retentionsparametern sowie Elektrolyten empfohlen."
        ),
        category="P",
    ),
    "P03": TextBlock(
        id="P03",
        title="PH-spezifische Therapie beginnen: PDE5-Inhibitor (SOP-neutral)",
        applies_to="Indikation gestellt, keine Kontraindikationen",
        template=(
            "{therapy_plan_sentence}"
        ),
        category="P",
        notes="Medikamenten-/Dosisdetails sollen aus planned_actions kommen; sonst Platzhalter leer lassen.",
        variants={
            "example_pde5i": (
                "Beginn einer PH-spezifischen Therapie mit einem PDE5-Inhibitor gemäß Fachinformation/hausinterner SOP "
                "(Kontraindikationen/Interaktionen beachten; Blutdruck und Verträglichkeit zeitnah kontrollieren)."
            )
        },
    ),
    "P04": TextBlock(
        id="P04",
        title="PH-spezifische Therapie beginnen/erweitern: ERA (SOP-neutral)",
        applies_to="PAH-Therapie, duale Strategie",
        template="{therapy_plan_sentence}",
        category="P",
        variants={
            "example_era": (
                "Ergänzung/Initiierung einer Endothelin-Rezeptorantagonisten-Therapie gemäß Fachinformation/hausinterner SOP "
                "(Kontrollen je nach Präparat/Standard)."
            )
        },
    ),
    "P05": TextBlock(
        id="P05",
        title="Riociguat (z.B. bei CTEPH oder als Wechselstrategie) – Sicherheitsformulierung",
        applies_to="Indikation gegeben, keine PDE5-Kombination",
        template="{therapy_plan_sentence}",
        category="P",
        variants={
            "example_riociguat": (
                "Therapie mit Riociguat gemäß Fachinformation/hausinterner SOP (einschleichend, strukturierte Aufdosierung nach Blutdruck/Verträglichkeit). "
                "Kombination mit PDE5-Inhibitoren kontraindiziert; erforderliche Washout-Zeiten einhalten."
            )
        },
    ),
    "P06": TextBlock(
        id="P06",
        title="Prostacyclin-Therapie/Parenteraloption ansprechen",
        applies_to="intermediär/hoch, RV-Versagen, Eskalationsindikation",
        template=(
            "{therapy_escalation_sentence}"
        ),
        category="P",
        variants={
            "generic": (
                "Aufklärung über Eskalationsoptionen im spezialisierten Setting (inkl. ggf. parenteraler Therapieoptionen) "
                "und Festlegung des weiteren Vorgehens nach klinischer Gesamtkonstellation/Patient:innenpräferenz."
            )
        },
    ),
    "P07": TextBlock(
        id="P07",
        title="Studienevaluation (neutral)",
        applies_to="wenn lokal verfügbar",
        template=(
            "Prüfung einer Studieneinschluss-Option im spezialisierten Zentrum gemäß aktueller Verfügbarkeit und Ein-/Ausschlusskriterien."
        ),
        category="P",
    ),
    "P08": TextBlock(
        id="P08",
        title="Radiologisch-pneumologische Konferenz / Fibrose-Ambulanz",
        applies_to="ILD-Hinweis, CT-Befund ausstehend oder progredient",
        template=(
            "Nach Vorliegen des CT-Thorax: Vorstellung/Beurteilung in der radiologisch-pneumologischen Konferenz empfohlen. "
            "Bei Hinweis auf ILD/Progredienz: Konsultation der Fibrose-Sprechstunde und Therapieplanung im interdisziplinären Setting."
        ),
        category="P",
    ),
    "P09": TextBlock(
        id="P09",
        title="Kardiologische Mitbeurteilung (Klappen/Rhythmus/Ischämie)",
        applies_to="V-Welle/MI-Verdacht, Rhythmusauffälligkeiten, TI etc.",
        template=(
            "Kardiologische Vorstellung/Verlaufskontrolle empfohlen (inkl. Echokardiographie zur Beurteilung von {valve_focus_desc} sowie ggf. Rhythmusdiagnostik). "
            "Bei Arrhythmiezeichen: EKG/Monitoring erwägen. Koronardiagnostik gemäß Klinik/Risikoprofil."
        ),
        category="P",
    ),
    "P10": TextBlock(
        id="P10",
        title="Antikoagulation & Gerinnungsambulanz",
        applies_to="VTE/Perfusionsdefekt/APS-Verdacht/Rezidiv unter DOAK",
        template=(
            "{anticoagulation_plan_sentence}"
        ),
        category="P",
        variants={
            "generic": (
                "Zeitnahe Vorstellung in der Gerinnungsambulanz zur Einordnung der thromboembolischen Konstellation und Optimierung der Antikoagulation. "
                "Antikoagulation gemäß Indikation konsequent fortführen."
            )
        },
    ),
    "P11": TextBlock(
        id="P11",
        title="Verlaufskontrolle (Standard-Timing)",
        applies_to="Erstdiagnose/Änderung/instabile Lage",
        template=(
            "Verlaufskontrolle im spezialisierten Setting in {followup_timing_desc} sowie Reevaluation der Risikokonstellation "
            "(Klinik, BNP/NT-proBNP, Echo, Belastbarkeit). "
            "Invasive Verlaufskontrolle (RHK) bei Therapieentscheidungen oder unklarem Verlauf in {invasive_followup_desc} erwägen."
        ),
        category="P",
    ),
    "P12": TextBlock(
        id="P12",
        title="Lungenfunktionelle Abklärung Restriktion/DLCO",
        applies_to="Restriktion/DLCO reduziert/hohe Atemarbeit",
        template=(
            "Weitere Abklärung der {pulm_workup_focus_desc} inkl. Komplettierung der pneumologischen Diagnostik "
            "und Korrelation mit Bildgebung (z.B. HR-CT) sowie ggf. Belastungsuntersuchungen."
        ),
        category="P",
    ),
    "P13": TextBlock(
        id="P13",
        title="Eisenmangel/Anämie-Baustein",
        applies_to="Anämie/Eisenmangelverdacht",
        template=(
            "Bei {anemia_context_desc}: Bestimmung von Eisenstatus (Ferritin, TSAT) und ggf. Substitution gemäß hausinternem Standard erwägen."
        ),
        category="P",
    ),
}

# -------------------------
# Zusatzbausteine (Z...)
# -------------------------

Z_BLOCKS: Dict[str, TextBlock] = {
    "Z01": TextBlock(
        id="Z01",
        title="Systemische Situation (RR/Herzfrequenz/Rhythmus)",
        applies_to="wenn Angaben vorhanden",
        template=(
            "Systemisch {bp_status} bei {hr_status_phrase} und {rhythm_status}."
        ),
        category="Z",
        variants={
            "extrasystolie": "Während der Untersuchung vermehrt Extrasystolie; EKG/Monitoring wird empfohlen.",
        },
    ),
    "Z02": TextBlock(
        id="Z02",
        title="Oxygenierung",
        applies_to="wenn Angaben vorhanden",
        template=(
            "{oxygenation_status} unter {oxygen_desc} in Ruhe."
        ),
        category="Z",
        variants={
            "exercise_abort": "Unter Belastung Abbruch aufgrund {exercise_abort_reason}.",
        },
    ),
    "Z03": TextBlock(
        id="Z03",
        title="Vergleich zum Vorbefund",
        applies_to="Voruntersuchung mit Datum und Vorwerten vorhanden",
        template=(
            "Im Vergleich zur Voruntersuchung vom {prev_date} zeigt sich {comparison_desc} "
            "({prev_values_desc} vs. aktuell {current_values_desc})."
        ),
        category="Z",
    ),
    "Z04": TextBlock(
        id="Z04",
        title="Messqualität eingeschränkt",
        applies_to="Messlimitationen vorhanden",
        template=(
            "Hinweis auf eingeschränkte Messqualität/Interpretierbarkeit aufgrund von {measurement_limitations}. "
            "Korrelation mit Echo und ggf. Wiederholung/Alternativmessung (z.B. direkte VO₂-Messung, LHK zur LVEDP-Korrelation) kann erwogen werden."
        ),
        category="Z",
    ),
    "Z05": TextBlock(
        id="Z05",
        title="Punktions-/Prozedurhinweis",
        applies_to="wenn gewünscht",
        template=(
            "Kontrolle der Punktionsstelle und klinische Überwachung gemäß hausinternem Standard. "
            "Antikoagulation (Pausierung/Wiederaufnahme) gemäß SOP und Blutungsrisiko."
        ),
        category="Z",
    ),
}

# -------------------------
# Allgemeine Empfehlungen (G...)
# -------------------------

G_BLOCKS: Dict[str, TextBlock] = {
    "G01": TextBlock(
        id="G01",
        title="Standard-Schlusssatz (neutral)",
        applies_to="optional",
        template=(
            "Die weitere Therapie und Diagnostik sollte interdisziplinär und unter Berücksichtigung von Klinik, "
            "Komorbiditäten und Patientenpräferenz erfolgen."
        ),
        category="G",
    ),
    "G02": TextBlock(
        id="G02",
        title="Keine spezifische Therapie aktuell",
        applies_to="wenn keine PH-spezifische Therapie geplant",
        template=(
            "Aus hämodynamischer Sicht ergibt sich aktuell keine Indikation für eine PH-spezifische Therapie. "
            "Verlaufskontrollen und weitere Abklärung entsprechend der führenden Differentialdiagnose empfohlen."
        ),
        category="G",
    ),
    "G03": TextBlock(
        id="G03",
        title="Volumenmanagement immer erwähnen (bei Stauung)",
        applies_to="bei Stauung",
        template=(
            "Volumenstatus als wesentlicher Mitfaktor der Symptomatik; konsequentes Volumenmanagement und Gewichtskontrolle empfohlen."
        ),
        category="G",
    ),
}

# -------------------------
# Alternativformulierungen (ALT...)
# -------------------------

ALT_BLOCKS: Dict[str, TextBlock] = {
    "ALT01": TextBlock(
        id="ALT01",
        title="statt 'Wir empfehlen folgendes Procedere'",
        applies_to="optional",
        template="Wir schlagen folgendes Vorgehen vor:",
        category="ALT",
    ),
    "ALT02": TextBlock(
        id="ALT02",
        title="statt 'Aktuell Initial-RHK'",
        applies_to="optional",
        template="Erstmalige invasive hämodynamische Evaluation im Rahmen dieser Untersuchung.",
        category="ALT",
    ),
    "ALT03": TextBlock(
        id="ALT03",
        title="statt 'Kein relevanter Sättigungssprung'",
        applies_to="optional",
        template="In der Stufenoxymetrie kein Hinweis auf einen hämodynamisch relevanten Shunt.",
        category="ALT",
    ),
}


# ---------------------------------------------------------------------------
# NEU: "Katalog v1.0 – K01..K20" als Pakettexte (Beurteilung + Empfehlung)
# Umsetzung: je Paket zwei Blöcke: Kxx_B und Kxx_E
# ---------------------------------------------------------------------------

K_BLOCKS: Dict[str, TextBlock] = {}

def _add_k_package(
    k_id: str,
    title: str,
    use_when: str,
    beurteilung: str,
    empfehlung: str,
    *,
    variants_b: Optional[Dict[str, str]] = None,
    variants_e: Optional[Dict[str, str]] = None,
    notes: str = "",
) -> None:
    """Interne Hilfe: registriert Kxx_B und Kxx_E."""
    K_BLOCKS[f"{k_id}_B"] = TextBlock(
        id=f"{k_id}_B",
        title=f"{k_id} – {title} (Beurteilung)",
        applies_to=use_when,
        template=beurteilung,
        category="K_B",
        variants=variants_b or {},
        notes=notes,
    )
    K_BLOCKS[f"{k_id}_E"] = TextBlock(
        id=f"{k_id}_E",
        title=f"{k_id} – {title} (Empfehlung)",
        applies_to=use_when,
        template=empfehlung_safe_wrap(empfehlung),
        category="K_E",
        variants=variants_e or {},
        notes=notes,
    )

def empfehlung_safe_wrap(text: str) -> str:
    """
    Hook: Im Hauptprogramm können Therapie-spezifische Sätze als Platzhalter gefüllt oder leer gelassen werden.
    Hier keine Logik, nur ein definierter Name.
    """
    return text


# --- K01 ---
_add_k_package(
    "K01",
    "Normalbefund in Ruhe, keine PH",
    "mPAP/PAWP/PVR unauffällig, keine Stauung, keine Shunt-Hinweise",
    beurteilung=(
        "Keine pulmonale Druck- und Widerstandserhöhung bei {ci_phrase} nach {co_method_desc}. "
        "Keine zentralvenöse oder pulmonalvenöse Stauung. {step_up_sentence} "
        "{systemic_sentence} {oxygen_sentence} "
        "Aktuell {exam_type_desc}."
    ),
    empfehlung=(
        "Aktuell kein Hinweis auf eine pulmonale Hypertonie in Ruhe. "
        "Bei persistierender Belastungsdyspnoe weitere Abklärung der Symptomatik gemäß klinischem Kontext "
        "(kardiologisch/pneumologisch, ggf. Spiroergometrie). "
        "{therapy_neutral_sentence} "
        "Verlaufskontrolle klinisch und echokardiographisch nach Bedarf."
    ),
)

# --- K02 ---
_add_k_package(
    "K02",
    "Keine PH in Ruhe, pathologische Belastungsreaktion eher linkskardial",
    "Ruhewerte normal; mPAP/CO-Slope pathologisch UND PAWP/CO-Slope ebenfalls pathologisch bzw. PAWP steigt deutlich unter Belastung",
    beurteilung=(
        "Keine pulmonale Druck- und Widerstandserhöhung in Ruhe {ci_phrase} nach {co_method_desc}. "
        "Keine zentralvenöse oder pulmonalvenöse Stauung in Ruhe. {step_up_sentence} "
        "{systemic_sentence} {oxygen_sentence} "
        "Unter ergometrischer Belastung pathologische mPAP/CO-Slope {mPAP_CO_slope} mmHg/(L/min) "
        "bei pathologischer/grenzwertiger PAWP/CO-Slope {PAWP_CO_slope} mmHg/(L/min) "
        "mit Hinweis auf eine linkskardiale Limitation."
    ),
    empfehlung=(
        "Ausschluss einer pulmonalen Hypertonie in Ruhe. Die Belastungsreaktion ist eher durch eine linkskardiale Limitation "
        "als durch ein primär pulmonalvaskuläres Problem erklärbar. "
        "Kardiologische Abklärung/Optimierung (HFpEF-/Diastolik, Echokardiographie inkl. Klappenvitien-Beurteilung, "
        "Rhythmusdiagnostik je nach Klinik) empfohlen. "
        "Pulmonal: Abklärung begleitender ventilatorischer Limitation (Lufu, ggf. CPET) je nach Kontext. "
        "{therapy_neutral_sentence} "
        "Verlauf entsprechend Befundkonstellation."
    ),
)

# --- K03 ---
_add_k_package(
    "K03",
    "Keine/geringe PH in Ruhe, pathologische Belastungsreaktion eher pulmonalvaskulär",
    "mPAP/CO-Slope pathologisch, PAWP/CO-Slope nicht führend erhöht (eher pulmonalvaskulär/early disease)",
    beurteilung=(
        "In Ruhe {rest_ph_sentence} bei {ci_phrase} nach {co_method_desc}. "
        "Keine zentrale oder pulmonalvenöse Stauung in Ruhe. "
        "Unter Belastung pathologische mPAP/CO-Slope {mPAP_CO_slope} mmHg/(L/min) bei {PAWP_CO_slope_phrase} "
        "({PAWP_CO_slope} mmHg/(L/min)) als Hinweis auf eine pulmonalvaskuläre Belastungslimitierung."
    ),
    empfehlung=(
        "Konstellation vereinbar mit Belastungs-PH / früher pulmonalvaskulärer Limitation. "
        "Komplettierung der PH-Diagnostik empfohlen (V/Q-Szintigraphie bzw. Ausschluss CTEPD/CTEPH, "
        "HRCT/CT-Thorax/ILD-Assessment, Autoimmun- und Infektionsscreening je nach Kontext, "
        "Echokardiographie mit Rechtsherzfokus). "
        "Therapeutisch stehen zunächst Trigger/Komorbiditäten im Vordergrund; "
        "PH-spezifische Therapieentscheidung nach Gesamtbild und im Verlauf."
    ),
)

# --- K04 ---
_add_k_package(
    "K04",
    "Unklassifizierte oder grenzwertige Konstellation",
    "mPAP leicht erhöht/PAWP grenzwertig, PVR nicht klar pathologisch, evtl. Demaskierung unter Volumen/Belastung ohne klare Definition",
    beurteilung=(
        "{borderline_ph_sentence} bei {pvr_phrase} und {ci_phrase} nach {co_method_desc}. "
        "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence} "
        "Unter {provocation_type_desc} zeigt sich {provocation_result_sentence}."
    ),
    empfehlung=(
        "Aktuell keine eindeutig klassifizierbare PH nach Definitionskriterien, jedoch Hinweise auf eine mögliche postkapilläre Komponente/Durchstauungstendenz "
        "je nach Provokation. "
        "Empfohlen: Optimierung kardiovaskulärer Risikofaktoren, regelmäßige kardiologische Verlaufskontrolle "
        "(Diastolik, Rhythmus, Blutdruck), konsequentes Volumenmanagement sowie Abklärung/Behandlung pulmonaler Komorbiditäten "
        "(Lufu/ILD/COPD/OSA/OHS) nach Klinik. "
        "{therapy_neutral_sentence}"
    ),
)

# --- K05 ---
_add_k_package(
    "K05",
    "Leichtgradige präkapilläre PH, niedriger Risikobereich, Diagnostik im Vordergrund",
    "präkapillär mild, keine ausgeprägte Stauung, eher niedriges Risiko",
    beurteilung=(
        "Leichtgradige pulmonale Druck- und Widerstandserhöhung bei präkapillärer Konstellation "
        "({mpap_phrase}, {pawp_phrase}, {pvr_phrase}) bei {ci_phrase} nach {co_method_desc}. "
        "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence} "
        "{systemic_sentence} {oxygen_sentence} "
        "Aktuell {exam_type_desc}."
    ),
    empfehlung=(
        "Konstellation einer milden präkapillären PH; Differenzialdiagnose je nach Kontext (Gruppe 1 vs. Gruppe 3 vs. andere). "
        "Komplettierung der PH-Diagnostik empfohlen (V/Q-Szintigraphie/Thromboembolie-Ausschluss, "
        "HRCT/ILD-Assessment bei restriktiver Ventilationsstörung/Diffusionsstörung, "
        "Autoimmun-/Infektionsscreening je nach Kontext, Echokardiographie mit Rechtsherzfokus, ggf. HFpEF-Abklärung). "
        "Therapieentscheidung nach Diagnostikabschluss, Symptomlast und Verlauf."
    ),
)

# --- K06 ---
_add_k_package(
    "K06",
    "Mittelgradige präkapilläre PH, Therapieeinleitung (therapieneutral)",
    "präkapilläre PH klarer, Symptomlast relevant; Therapieeinleitung wird diskutiert/geplant",
    beurteilung=(
        "Mittelgradige pulmonale Druck- und Widerstandserhöhung bei präkapillärer Konstellation "
        "({mpap_phrase}, {pawp_phrase}, {pvr_phrase}) bei {ci_phrase} nach {co_method_desc}. "
        "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence} "
        "{systemic_sentence} {oxygen_sentence} "
        "Aktuell {exam_type_desc}."
    ),
    empfehlung=(
        "Konstellation einer präkapillären PH (Differenzialdiagnose je nach Kontext, inkl. Gruppe 1 vs. Gruppe 3 vs. Gruppe 4). "
        "{therapy_plan_sentence} "
        "Komplettierung der PH-Diagnostik, sofern noch nicht vollständig. "
        "Verlaufskontrollen (klinisch/echo/labor) zeitnah nach Therapieentscheidung; erneute invasive Reevaluation bei geplanter Therapieeskalation oder klinischem Verlauf."
    ),
    variants_e={
        "with_examples": (
            "Konstellation einer präkapillären PH (Differenzialdiagnose je nach Kontext). "
            "{therapy_plan_sentence} "
            "Hinweis: Beispiele für spezifische Therapieklassen (nur falls geplant/gewünscht): {therapy_examples_sentence}. "
            "Komplettierung der PH-Diagnostik und Verlaufskontrollen wie geplant."
        )
    },
)

# --- K07 ---
_add_k_package(
    "K07",
    "Schwergradige präkapilläre PH mit reduziertem CI",
    "PVR deutlich erhöht und CI reduziert (hohes Risiko)",
    beurteilung=(
        "Schwergradige pulmonale Druck- und Widerstandserhöhung bei präkapillärer Konstellation "
        "({mpap_phrase}, {pawp_phrase}, {pvr_phrase}) bei {ci_phrase} nach {co_method_desc}. "
        "{cv_stauung_phrase} {pv_stauung_phrase} {step_up_sentence} "
        "{systemic_sentence} {oxygen_sentence} "
        "{comparison_sentence} "
        "Aktuell {exam_type_desc}."
    ),
    empfehlung=(
        "Hochgradige präkapilläre PH mit Zeichen der Rechtsherzbelastung und reduziertem Herzzeitvolumen. "
        "{therapy_escalation_sentence} "
        "Engmaschige klinische Kontrolle (Klinik, BNP/NT-proBNP, Echo, Belastbarkeit) empfohlen. "
        "Bei führender Stauung: Priorisierung der Dekongestion und Trigger-/Komorbiditätsdiagnostik (Infekt, Rhythmus, Anämie, Hypoxie/ILD-Progress) "
        "je nach klinischem Kontext. "
        "Diskussion im Expert:innenboard/Referenzzentrum bei persistierend hohem Risiko erwägen."
    ),
)

# --- K08 ---
_add_k_package(
    "K08",
    "Schwergradige PH, Dekongestion im Vordergrund, Patient:innenpräferenz",
    "hohe PH-Last, zentrale Stauung führend; Therapieerweiterung grundsätzlich indiziert, aber klinisch zunächst Dekongestion / Präferenz",
    beurteilung=(
        "Schwergradige pulmonale Druck- und Widerstandserhöhung bei {ci_phrase} nach {co_method_desc}. "
        "Ausgeprägte zentralvenöse, keine pulmonalvenöse Stauung. "
        "{systemic_sentence} {oxygen_sentence} "
        "{comparison_sentence} "
        "{measurement_limitation_sentence}"
    ),
    empfehlung=(
        "Bei {risk_profile_desc} besteht grundsätzlich eine Indikation zur Therapieerweiterung; "
        "aktuell steht jedoch die deutliche zentralvenöse Stauung im Vordergrund (Priorität: Dekongestion). "
        "{patient_preference_sentence} "
        "Nach Stabilisierung interdisziplinäre Reevaluation und Festlegung des weiteren Vorgehens im spezialisierten Setting."
    ),
)

# --- K09 ---
_add_k_package(
    "K09",
    "Verlaufskontrolle PAH, hämodynamisch gebessert",
    "bekannte PAH/PH unter Therapie; objektive Besserung; klinisch stabil",
    beurteilung=(
        "{comparison_sentence} "
        "Keine zentrale oder pulmonalvenöse Stauung. {systemic_sentence} {oxygen_sentence}"
    ),
    empfehlung=(
        "Unter bestehender pulmonalvaskulärer Therapie zeigt sich eine hämodynamische Verbesserung bei klinischer Stabilität. "
        "Fortführung der aktuellen Therapie und Verlaufskontrolle gemäß Risikoprofil empfohlen."
    ),
)

# --- K10 ---
_add_k_package(
    "K10",
    "Verlaufskontrolle PAH, progredient / nicht ausreichend kontrolliert",
    "trotz Therapie hohe Werte / Risiko bleibt hoch / CI sinkt",
    beurteilung=(
        "{comparison_sentence} "
        "{cv_stauung_phrase} {pv_stauung_phrase} "
        "{systemic_sentence} {oxygen_sentence}"
    ),
    empfehlung=(
        "Unter {therapy_current_desc} persistiert ein {risk_profile_desc}. "
        "{therapy_escalation_sentence} "
        "Zusätzlich: konsequentes Volumenmanagement, Triggerdiagnostik und engmaschige Verlaufskontrollen."
    ),
)

# --- K11 ---
_add_k_package(
    "K11",
    "CTEPH/CTEPD Pfad – Diagnostik + Konferenz",
    "CTEPH bekannt/verdächtig oder residuelle Perfusionsdefekte; BPA/PEA Evaluation",
    beurteilung=(
        "Präkapilläre pulmonale Hypertonie mit {pvr_phrase} bei {ci_phrase}. "
        "{step_up_sentence} {cv_stauung_phrase} {pv_stauung_phrase} "
        "Aktuell {exam_type_desc}."
    ),
    empfehlung=(
        "Konstellation vereinbar mit CTEPH bzw. CTEPD (je nach hämodynamischer Ausprägung) bei {cteph_context_desc}. "
        "Empfohlen: Komplettierung der CTEPH-Diagnostik (V/Q, CT-PA/PA-Angiographie je nach Vorbefund) "
        "und Vorstellung in einem CTEPH-Board (PEA/BPA/medikamentös) zur Festlegung des weiteren Procedere. "
        "{anticoagulation_plan_sentence}"
    ),
)

# --- K12 ---
_add_k_package(
    "K12",
    "Portopulmonale PH / Hyperzirkulation (z.B. TIPSS)",
    "hoher CO/CI, PVR moderat, portale Hypertension/TIPSS-Kontext",
    beurteilung=(
        "Pulmonale Druckerhöhung bei hyperzirkulatorischem Herzzeitvolumen (CI {CI_value}) und {pvr_phrase}. "
        "Keine zentrale oder pulmonalvenöse Stauung. {systemic_sentence} {oxygen_sentence} "
        "{comparison_sentence}"
    ),
    empfehlung=(
        "Konstellation einer präkapillären PH im Kontext {porto_context_desc} bei Hyperzirkulation. "
        "Therapieentscheidung und Verlaufskontrollen im interdisziplinären Setting (PH-/Leberzentrum) empfohlen; "
        "Fokus auf Trigger-/Komorbiditätsmanagement und Monitoring von Leber-/TIPSS-Setting."
    ),
)

# --- K13 ---
_add_k_package(
    "K13",
    "CTD-PAH (z.B. SSc/CREST) – ILD-Abklärung, diuretischer Fokus",
    "CTD bekannt; PAH; ILD Thema; Rechtsherzbelastung/Volumen",
    beurteilung=(
        "{severity_ph_sentence} bei {ci_phrase}. "
        "{cv_stauung_phrase} {pv_stauung_phrase} "
        "{systemic_sentence} {oxygen_sentence} "
        "{measurement_limitation_sentence} "
        "{comparison_sentence}"
    ),
    empfehlung=(
        "Bekannte/verdächtige präkapilläre PH im Kontext {ctd_desc}. "
        "{lufu_context_sentence} "
        "Empfohlen: ILD-Assessment (HRCT/CT-Thorax, ILD-Board/Fibroseambulanz je nach Befund), "
        "Dekongestion bei Stauung und kardiologische Mitbetreuung (Klappen/Rhythmus) nach Klinik. "
        "{therapy_escalation_sentence}"
    ),
)

# --- K14 ---
_add_k_package(
    "K14",
    "Isoliert postkapilläre PH (Gruppe 2) – linkskardiale Genese",
    "PAWP erhöht, PVR nicht oder nur gering erhöht; HF/Valvulär prominent",
    beurteilung=(
        "Pulmonale Druckerhöhung bei Hinweis auf pulmonalvenöse Stauung ({pawp_phrase}{V_wave_short}). "
        "{pvr_phrase}. {ci_phrase}. {cv_stauung_phrase} {step_up_sentence}"
    ),
    empfehlung=(
        "Konstellation einer postkapillären PH (Gruppe 2) bei {left_heart_context_desc}. "
        "Im Vordergrund: kardiologische Therapieoptimierung (Volumenmanagement, Blutdruck, Rhythmus, ggf. Klappentherapie) "
        "gemäß behandelndem Team. PH-spezifische Therapie ist in dieser Konstellation nicht indiziert."
    ),
)

# --- K15 ---
_add_k_package(
    "K15",
    "Kombiniert prä- und postkapillär (CpcPH)",
    "PAWP erhöht und PVR signifikant erhöht; Mischbild",
    beurteilung=(
        "Pulmonale Druckerhöhung mit Hinweis auf pulmonalvenöse Stauung ({pawp_phrase}) bei zugleich erhöhter "
        "pulmonalvaskulärer Widerstandskomponente ({pvr_phrase}). {ci_phrase}. "
        "{provocation_sentence}"
    ),
    empfehlung=(
        "Mischbild einer kombinierten prä- und postkapillären PH (CpcPH). "
        "Empfohlen: HF-/Linksherz-Optimierung (Dekongestion, Risikofaktoren, Rhythmus) und vollständige DD-Abklärung "
        "der präkapillären Komponente (V/Q, HRCT, Autoimmun etc.) im spezialisierten Setting. "
        "PH-spezifische Therapie nur bei klar relevanter präkapillärer Komponente und nach Expert:innen-Einordnung."
    ),
)

# --- K16 ---
_add_k_package(
    "K16",
    "Sättigungssprung / Shunt-Konstellation",
    "Sättigungssprung in Stufenoxymetrie, Shunt-Verdacht",
    beurteilung=(
        "Pulmonale Druck-/Widerstandswerte {pressure_resistance_short}. "
        "In der Stufenoxymetrie zeigt sich ein Sättigungssprung zwischen {step_up_from_to}. "
        "{cv_stauung_phrase} {systemic_sentence}"
    ),
    empfehlung=(
        "Hinweis auf möglichen intra- oder extrakardialen Shunt. "
        "Empfohlen: strukturelle Abklärung (Echo, ggf. Kontrast/Bubble, ggf. TEE, ggf. Kardio-MRT/CT je nach Fragestellung) "
        "und Vorstellung in einer EMAH/CHD-Sprechstunde bei strukturellem Verdacht. "
        "Therapeutische Konsequenzen abhängig von gesicherter Diagnose und hämodynamischer Relevanz."
    ),
)

# --- K17 ---
_add_k_package(
    "K17",
    "Vasoreagibilitätstest positiv – Leitlinienpfad (therapieneutral)",
    "iNO-Test/Pharmakotest positiv (Responderkriterien erfüllt)",
    beurteilung=(
        "Präkapilläre PH-Konstellation. Unter pharmakologischer Austestung ({vasoreactivity_agent_desc}) "
        "zeigt sich ein signifikanter Abfall der pulmonalen Druckwerte bei erhaltener Förderleistung; Responderkriterien erfüllt."
    ),
    empfehlung=(
        "Bei positiver Vasoreagibilität: Therapiepfad gemäß Leitlinie/zentraler SOP erwägen. "
        "{therapy_plan_sentence} "
        "Engmaschige Verlaufskontrolle (klinisch, BNP, Echo; ggf. erneute invasive Reevaluation nach Verlauf) empfohlen."
    ),
)

# --- K18 ---
_add_k_package(
    "K18",
    "Vasoreagibilitätstest negativ – kein Vasoreagibilitäts-Pfad",
    "iNO-Test/Pharmakotest negativ (Responderkriterien nicht erfüllt)",
    beurteilung=(
        "Präkapilläre PH-Konstellation. Unter iNO-Testung zeigt sich {iNO_response_desc}; Responderkriterien nicht erfüllt."
    ),
    empfehlung=(
        "Keine Indikation für einen Vasoreagibilitäts-spezifischen Therapiepfad. "
        "Weiteres Vorgehen gemäß PH-Ätiologie und Risikoprofil; Diagnostik vervollständigen und Verlaufskontrollen wie geplant."
    ),
)

# --- K19 ---
_add_k_package(
    "K19",
    "PVOD-Verdacht / DD",
    "präkapilläre PH bei normalem/grenzwertigem PAWP + PVOD-Hinweise (z.B. CT/DLCO)",
    beurteilung=(
        "Präkapilläre PH-Konstellation bei {pawp_phrase}. "
        "Klinisch/diagnostisch bestehen Hinweise auf eine mögliche PVOD-Konstellation ({pvod_hint_desc})."
    ),
    empfehlung=(
        "Differenzialdiagnose PVOD. Betreuung im spezialisierten Zentrum und vorsichtige Therapieplanung empfohlen "
        "(Risiko pulmonaler Ödeme unter Vasodilatation beachten). "
        "Engmaschiger Verlauf (Klinik, O2-Bedarf, BNP, Bildgebung) nach Kontext."
    ),
)

# --- K20 ---
_add_k_package(
    "K20",
    "Sekundäre Ursachen / Trigger-Module",
    "sekundäre Ursachen sollen aktualisiert/abgeklärt werden (HIV, Leber, Schisto, OSA/OHS, ILD/COPD, Thromboembolie)",
    beurteilung=(
        "Sekundäre Ursachen/Trigger der PH sollten – sofern noch nicht erfolgt – aktualisiert abgeklärt werden."
    ),
    empfehlung=(
        "Empfohlen je nach Kontext: Autoimmun-Diagnostik (CTD-Screen), HIV- und Hepatitis-Serologie, Leberdiagnostik "
        "(bei portaler Hypertension/portopulmonalem Verdacht), schlafmedizinische Diagnostik bei V. a. OSA/OHS, "
        "Parenchymdiagnostik (HRCT/ILD-Board) und Lungenfunktion inkl. Diffusion sowie Thromboembolie-Abklärung "
        "(V/Q-Szinti, CTEPH-Pfad)."
    ),
)


# ---------------------------------------------------------------------------
# NEU: Zusatzmodule aus dem erweiterten Katalog (BZ.. / BE.. / C..)
# ---------------------------------------------------------------------------

ADDON_BLOCKS: Dict[str, TextBlock] = {
    # B Z01 – zentralvenöse Stauung
    "BZ01_NONE": TextBlock(
        id="BZ01_NONE",
        title="BZ01 – Zentrale Stauung: keine",
        applies_to="RAP/ZVD unauffällig",
        template="Keine zentralvenöse Stauung.",
        category="BZ",
    ),
    "BZ01_MILD": TextBlock(
        id="BZ01_MILD",
        title="BZ01 – Zentrale Stauung: leicht",
        applies_to="RAP/ZVD leicht erhöht",
        template="Leichtgradige zentralvenöse Stauung.",
        category="BZ",
    ),
    "BZ01_SEVERE": TextBlock(
        id="BZ01_SEVERE",
        title="BZ01 – Zentrale Stauung: ausgeprägt",
        applies_to="RAP/ZVD deutlich erhöht",
        template="Ausgeprägte zentralvenöse Stauung.",
        category="BZ",
    ),

    # B Z02 – pulmonalvenöse Stauung / v-Welle
    "BZ02_PV_NONE": TextBlock(
        id="BZ02_PV_NONE",
        title="BZ02 – Pulmonalvenöse Stauung: keine",
        applies_to="PAWP/PCWP nicht erhöht",
        template="Keine pulmonalvenöse Stauung.",
        category="BZ",
    ),
    "BZ02_PV_PRESENT": TextBlock(
        id="BZ02_PV_PRESENT",
        title="BZ02 – Pulmonalvenöse Stauung (± v-Welle)",
        applies_to="PAWP/PCWP erhöht ± v-Welle",
        template="Hinweis auf pulmonalvenöse Stauung (PAWP/PCWP {PAWP_value} mmHg){V_wave_short}.",
        category="BZ",
    ),
    "BZ02_PV_EXERCISE": TextBlock(
        id="BZ02_PV_EXERCISE",
        title="BZ02 – Pulmonalvenöse Stauung unter Belastung/Volumen",
        applies_to="PAWP steigt unter Belastung/Volumen",
        template="Unter {provocation_type_desc} deutlicher PAWP-Anstieg mit {V_wave_desc}.",
        category="BZ",
    ),

    # B Z03 – Stufenoxymetrie
    "BZ03_NO_STEPUP": TextBlock(
        id="BZ03_NO_STEPUP",
        title="BZ03 – Stufenoxymetrie: kein Sättigungssprung",
        applies_to="kein Sprung",
        template="Kein relevanter Sättigungssprung in der Stufenoxymetrie.",
        category="BZ",
    ),
    "BZ03_STEPUP": TextBlock(
        id="BZ03_STEPUP",
        title="BZ03 – Stufenoxymetrie: Sättigungssprung",
        applies_to="Sprung vorhanden",
        template="Sättigungssprung in der Stufenoxymetrie zwischen {step_up_from_to} – Shunt-Abklärung empfohlen.",
        category="BZ",
    ),

    # B Z04 – Oxygenation
    "BZ04_O2_NORM_RA": TextBlock(
        id="BZ04_O2_NORM_RA",
        title="BZ04 – Oxygenation: Normoxämie unter Raumluft",
        applies_to="normoxäm",
        template="Normoxämie unter Raumluft in Ruhe.",
        category="BZ",
    ),
    "BZ04_O2_PARTIAL_RA": TextBlock(
        id="BZ04_O2_PARTIAL_RA",
        title="BZ04 – Oxygenation: respiratorische Partialinsuffizienz unter Raumluft",
        applies_to="Hypoxämie",
        template="Respiratorische Partialinsuffizienz unter Raumluft in Ruhe.",
        category="BZ",
    ),
    "BZ04_O2_LTOT": TextBlock(
        id="BZ04_O2_LTOT",
        title="BZ04 – Oxygenation: Normoxämie unter LTOT",
        applies_to="LTOT",
        template="Normoxämie unter LTOT {O2_flow} L/min in Ruhe.",
        category="BZ",
    ),

    # B Z05 – Rhythmus
    "BZ05_RHY_NORM": TextBlock(
        id="BZ05_RHY_NORM",
        title="BZ05 – Rhythmus: Normofrequenz",
        applies_to="HF normal",
        template="Normofrequenz.",
        category="BZ",
    ),
    "BZ05_RHY_TACHY": TextBlock(
        id="BZ05_RHY_TACHY",
        title="BZ05 – Rhythmus: Tachykardie",
        applies_to="HF erhöht",
        template="Tachykardie.",
        category="BZ",
    ),
    "BZ05_RHY_EXTRAS": TextBlock(
        id="BZ05_RHY_EXTRAS",
        title="BZ05 – Rhythmus: Extrasystolie",
        applies_to="Extrasystolen",
        template="Extrasystolie-Neigung während der Untersuchung; EKG/Holter je nach Klinik erwägen.",
        category="BZ",
    ),

    # B Z06 – Messmethodik / Plausibilität
    "BZ06_CO_METHODS": TextBlock(
        id="BZ06_CO_METHODS",
        title="BZ06 – Messmethodik: CI Thermodilution vs Fick",
        applies_to="CO/CI mehrfach bestimmt",
        template="CI nach Thermodilution und {fick_desc} {co_method_concordance_desc}; Interpretation im klinischen Kontext.",
        category="BZ",
    ),
    "BZ06_TI_LIMIT": TextBlock(
        id="BZ06_TI_LIMIT",
        title="BZ06 – Messlimitation: schwere TI / Pseudo-Dip",
        applies_to="massive TI/Pseudo-Dip",
        template="Bei massiver TI/Pseudo-Dip ist die Thermodilution potenziell limitiert; CI nach direkter Fick-Methode besonders berücksichtigen.",
        category="BZ",
    ),

    # B Z07 – Vergleich
    "BZ07_COMPARISON": TextBlock(
        id="BZ07_COMPARISON",
        title="BZ07 – Vergleich zum Vorbefund",
        applies_to="Vorbefund vorhanden",
        template="Im Vergleich zur Voruntersuchung vom {prev_date} {comparison_desc} (mPAP {mpap_prev}→{mpap_now} mmHg, PAWP {pawp_prev}→{pawp_now} mmHg, CI {ci_prev}→{ci_now}, PVR {pvr_prev}→{pvr_now}).",
        category="BZ",
    ),

    # B Z08 – Belastungs-Slope
    "BZ08_SLOPES": TextBlock(
        id="BZ08_SLOPES",
        title="BZ08 – Belastung: Slope-Dokumentation",
        applies_to="Belastungsdaten vorhanden",
        template="mPAP/CO-Slope {mPAP_CO_slope} mmHg/(L/min), PAWP/CO-Slope {PAWP_CO_slope} mmHg/(L/min).",
        category="BZ",
    ),

    # B Z09 – Volumenchallenge
    "BZ09_VOLUME": TextBlock(
        id="BZ09_VOLUME",
        title="BZ09 – Volumenchallenge",
        applies_to="Volumenchallenge durchgeführt",
        template="Unter Volumenchallenge mit {volume_ml} ml {infusion_type} zeigt sich {volume_response_sentence}.",
        category="BZ",
    ),

    # B E01 – Diuretika-Sicherheitsmodul
    "BE01_DIURETICS_SAFETY": TextBlock(
        id="BE01_DIURETICS_SAFETY",
        title="BE01 – Diuretika: Sicherheitsmodul",
        applies_to="bei diuretischer Anpassung",
        template="Bitte Kontrolle der Nierenretentionsparameter und der Blutelektrolyte unter diuretischer Therapieanpassung.",
        category="BE",
    ),

    # B E02 – Eisenmangel
    "BE02_IRON": TextBlock(
        id="BE02_IRON",
        title="BE02 – Eisenmangel/Anämie",
        applies_to="Anämie/Eisenmangelverdacht",
        template="Bei Anämie bzw. V. a. Eisenmangel: Bestimmung von Eisenstatus (Ferritin, Transferrinsättigung) und ggf. Substitution gemäß Standard.",
        category="BE",
    ),

    # B E03 – Studien
    "BE03_STUDY": TextBlock(
        id="BE03_STUDY",
        title="BE03 – Studienmodul",
        applies_to="Studienevaluation",
        template="{study_sentence}",
        category="BE",
        variants={
            "offer": "Studienevaluation über die Studienambulanz je nach Therapiesituation/Kriterien.",
            "none": "Aktuell keine Studienoptionen / kein Studieneinschluss möglich.",
        },
    ),

    # B E04 – Patient:innenpräferenz
    "BE04_PATIENT_PREF": TextBlock(
        id="BE04_PATIENT_PREF",
        title="BE04 – Patient:innenpräferenz",
        applies_to="Therapie/Studie abgelehnt",
        template="Nach Aufklärung über mögliche Therapieoptionen lehnt die/der Patient:in {declined_item} ab.",
        category="BE",
    ),

    # B E05 – ILD Board
    "BE05_ILD_BOARD": TextBlock(
        id="BE05_ILD_BOARD",
        title="BE05 – ILD Board / Fibroseambulanz",
        applies_to="ILD/Restriktion/DLCO↓/CT ausstehend",
        template="Vorstellung in der radiologisch-pneumologischen Konferenz/ILD-Board; bei ILD-Hinweis Anbindung an die Fibroseambulanz.",
        category="BE",
    ),

    # B E06 – CTEPH Konferenz
    "BE06_CTEPH_BOARD": TextBlock(
        id="BE06_CTEPH_BOARD",
        title="BE06 – CTEPH Konferenz",
        applies_to="CTEPH/CTEPD Pfad",
        template="PA-Angiographie zur operativen/interventionellen Planung und Vorstellung in der CTEPH-Konferenz (BPA/PEA Evaluation).",
        category="BE",
    ),

    # B E07 – Antikoag Review
    "BE07_ANTICOAG": TextBlock(
        id="BE07_ANTICOAG",
        title="BE07 – Antikoagulations-Review",
        applies_to="LAE/CTEPH/Rezidiv unter DOAK/VKA-Frage",
        template="Zeitnahe Vorstellung in der Gerinnungsambulanz zur Evaluation der Antikoagulationsstrategie (bei {anticoag_context}).",
        category="BE",
    ),

    # B E08 – Post-Prozedur Allgemein
    "BE08_POST_PROC": TextBlock(
        id="BE08_POST_PROC",
        title="BE08 – Post-Prozedur Allgemein",
        applies_to="immer wenn gewünscht",
        template="Kontrolle der Punktionsstelle, Vitalparameter und EKG-Kontrolle bei Arrhythmieereignissen/Palpitationen gemäß Standard.",
        category="BE",
    ),

    # C 01 OSA/OHS
    "C01_OSA_OHS": TextBlock(
        id="C01_OSA_OHS",
        title="C01 – OSA/OHS Zusatzblock",
        applies_to="Hinweis auf Schlaf-/Hypoventilation (Adipositas, Tagesschläfrigkeit, Hyperkapnie, OHS)",
        template="Bei Hinweis auf Schlaf-/Hypoventilationsproblematik empfehlen wir eine schlafmedizinische Diagnostik (Polygraphie/Polysomnographie) und Therapie (CPAP/NIV) sowie strukturierte Gewichtsreduktion.",
        category="C",
    ),

    # C 02 Schistosomiasis
    "C02_SCHISTO": TextBlock(
        id="C02_SCHISTO",
        title="C02 – Schistosomiasis/tropenassoziierte Ursachen",
        applies_to="Reise-/Expositionsanamnese passend",
        template="Bei entsprechender Reise-/Expositionsanamnese empfehlen wir die Abklärung auf schistosomiasisassoziierte PH bzw. andere tropenassoziierte Ursachen in Kooperation mit Tropenmedizin/Infektiologie.",
        category="C",
    ),

    # C 03 Hochfluss
    "C03_HIGH_FLOW_B": TextBlock(
        id="C03_HIGH_FLOW_B",
        title="C03 – Hochfluss-Konstellation (Beurteilung)",
        applies_to="hoher CO/CI ohne TIPSS",
        template="Pulmonale Druckerhöhung bei hyperzirkulatorischem Herzzeitvolumen; Widerstände {pvr_phrase}.",
        category="C_B",
    ),
    "C03_HIGH_FLOW_E": TextBlock(
        id="C03_HIGH_FLOW_E",
        title="C03 – Hochfluss-Konstellation (Empfehlung)",
        applies_to="hoher CO/CI ohne TIPSS",
        template="Abklärung und Behandlung möglicher Hochfluss-Ursachen (Anämie, Hyperthyreose, AV-Fistel, Sepsis/Entzündung, Lebererkrankung). Therapieentscheidung streng nach PVR-Komponente und Gesamtkontext.",
        category="C_E",
    ),

    # C 04 CTEPD ohne PH
    "C04_CTEPD_NO_PH": TextBlock(
        id="C04_CTEPD_NO_PH",
        title="C04 – CTEPD ohne PH in Ruhe",
        applies_to="Perfusionsdefekte ohne PH in Ruhe",
        template="Bei Perfusionsdefekten ohne PH in Ruhe: Beurteilung der Symptomatik, Ausschluss alternativer Dyspnoeursachen und Diskussion im CTEPH-Board (CTEPD-Pfad) bei relevanter Symptomlast; ggf. Belastungsdiagnostik.",
        category="C",
    ),

    # C 05 Klappenvitien / v-Wellen
    "C05_VALVE_VWAVE": TextBlock(
        id="C05_VALVE_VWAVE",
        title="C05 – Klappenvitien & v-Wellen",
        applies_to="prominente v-Welle/MI-Verdacht",
        template="Bei prominenter v-Welle/MI-Hinweis: kardiologische Mitbeurteilung der Klappensituation (Echo, ggf. TEE/Interventionsplanung). PH-Einordnung im Kontext der Klappenerkrankung.",
        category="C",
    ),

    # NEU: Echo-Add-on S'/RAAi (Yogeswaran et al.)
    "ECHO_SPRIME_RAAI": TextBlock(
        id="ECHO_SPRIME_RAAI",
        title="Echo-Add-on: S'/RAAi (Yogeswaran et al.)",
        applies_to="S'/RAAi gemessen",
        template=(
            "S'/RAAi: {Sprime_RAAI_value} m²/(s·cm) "
            "(Cut-off {Sprime_RAAI_cutoff} m²/(s·cm)); {Sprime_RAAI_interpretation_sentence}."
        ),
        category="ECHO",
        notes="Interpretation soll durch Hauptprogramm erfolgen (z.B. 'erniedrigt' vs 'nicht erniedrigt').",
    ),

    # NEU: Dip-Plateau / Restriktion
    "WAVEFORM_DIP_PLATEAU": TextBlock(
        id="WAVEFORM_DIP_PLATEAU",
        title="Kurvenmorphologie: Dip-Plateau (Restriktionshinweis)",
        applies_to="RV-Druckkurve mit Dip-Plateau",
        template=(
            "Hinweis auf Dip-Plateau-Konfiguration (\"square root sign\") in der RV-Druckkurve. "
            "Differenzialdiagnostisch vereinbar mit restriktiver Füllungsstörung/Perikardkonstriktion; "
            "klinische und echokardiographische Korrelation empfohlen."
        ),
        category="WAVEFORM",
    ),

    # NEU: Exercise-Add-on nach Thal et al. (ΔsPAP & peak-CI dokumentieren)
    "EX_THAL_DELTA_SPAP": TextBlock(
        id="EX_THAL_DELTA_SPAP",
        title="Belastungs-Add-on: ΔsPAP & peak-CI (Thal et al.)",
        applies_to="Exercise-RHK mit Ruhe- und Peak-Belastungswerten",
        template=(
            "Belastungs-Parameter: ΔsPAP (Peak–Ruhe) {delta_sPAP} mmHg; peak-CI {CI_peak} L/min/m²."
        ),
        category="EX",
        notes="ΔsPAP wird im Hauptprogramm als sPAP_peak - sPAP_rest berechnet, falls beide vorhanden.",
    ),

    # NEU: Lungenfunktion Summary (frei einsetzbar)
    "PULM_LUFU_SUMMARY": TextBlock(
        id="PULM_LUFU_SUMMARY",
        title="Pulmonologie: Lungenfunktion (Kurz)",
        applies_to="Lungenfunktion vorhanden",
        template="Lungenfunktion: {lufu_summary}.",
        category="PULM",
    ),
}


# ---------------------------------------------------------------------------
# Gesamtindex + Hilfsfunktionen
# ---------------------------------------------------------------------------

ALL_BLOCKS: Dict[str, TextBlock] = {}
for _d in (B_BLOCKS, E_BLOCKS, P_BLOCKS, Z_BLOCKS, G_BLOCKS, ALT_BLOCKS, K_BLOCKS, ADDON_BLOCKS):
    ALL_BLOCKS.update(_d)


def get_block(block_id: str) -> Optional[TextBlock]:
    return ALL_BLOCKS.get(block_id)


def list_blocks(prefix: str) -> List[TextBlock]:
    prefix = prefix.upper()
    return [b for k, b in sorted(ALL_BLOCKS.items()) if k.upper().startswith(prefix)]


def list_blocks_by_category(category: str) -> List[TextBlock]:
    category = category.upper()
    return [b for _, b in sorted(ALL_BLOCKS.items()) if b.category.upper() == category]


# ---------------------------------------------------------------------------
# Bundles / Pakete: Mapping "Paket" -> Block-IDs
# (Das Hauptprogramm kann hieraus Beurteilung+Empfehlung+Procedere-Vorschläge ableiten)
# ---------------------------------------------------------------------------

BUNDLES: Dict[str, Dict[str, List[str]]] = {
    # Legacy
    "PKG00": {"B": ["B01", "B17"], "E": ["E01"], "P_suggestions": []},
    "PKG01": {"B": ["B03", "B17"], "E": ["E02"], "P_suggestions": ["P09"]},
    "PKG02": {"B": ["B21", "B17"], "E": ["E12"], "P_suggestions": ["P01", "P11"]},
    "PKG20": {"B": ["B09"], "E": ["E06"], "P_suggestions": ["P09", "P11"]},
    "PKG21": {"B": ["B11"], "E": ["E06"], "P_suggestions": ["P09", "P11"]},
    "PKG22": {"B": ["B10"], "E": ["E07"], "P_suggestions": ["P01", "P09", "P11"]},
    "PKG50": {"B": ["B16"], "E": ["E11"], "P_suggestions": ["P09"]},

    # Neue K-Pakete (Beurteilung/Empfehlung getrennt)
    "K01": {"B": ["K01_B"], "E": ["K01_E"], "P_suggestions": []},
    "K02": {"B": ["K02_B"], "E": ["K02_E"], "P_suggestions": ["P09", "P12"]},
    "K03": {"B": ["K03_B"], "E": ["K03_E"], "P_suggestions": ["P01", "P11"]},
    "K04": {"B": ["K04_B"], "E": ["K04_E"], "P_suggestions": ["P09", "P12"]},
    "K05": {"B": ["K05_B"], "E": ["K05_E"], "P_suggestions": ["P01", "P11"]},
    "K06": {"B": ["K06_B"], "E": ["K06_E"], "P_suggestions": ["P01", "P11"]},
    "K07": {"B": ["K07_B"], "E": ["K07_E"], "P_suggestions": ["P02", "P11"]},
    "K08": {"B": ["K08_B"], "E": ["K08_E"], "P_suggestions": ["P02", "P11"]},
    "K09": {"B": ["K09_B"], "E": ["K09_E"], "P_suggestions": ["P11"]},
    "K10": {"B": ["K10_B"], "E": ["K10_E"], "P_suggestions": ["P11"]},
    "K11": {"B": ["K11_B"], "E": ["K11_E"], "P_suggestions": ["P10"]},
    "K12": {"B": ["K12_B"], "E": ["K12_E"], "P_suggestions": ["P11"]},
    "K13": {"B": ["K13_B"], "E": ["K13_E"], "P_suggestions": ["P08", "P02", "P11"]},
    "K14": {"B": ["K14_B"], "E": ["K14_E"], "P_suggestions": ["P09"]},
    "K15": {"B": ["K15_B"], "E": ["K15_E"], "P_suggestions": ["P01", "P09"]},
    "K16": {"B": ["K16_B"], "E": ["K16_E"], "P_suggestions": ["P09"]},
    "K17": {"B": ["K17_B"], "E": ["K17_E"], "P_suggestions": ["P11"]},
    "K18": {"B": ["K18_B"], "E": ["K18_E"], "P_suggestions": ["P11"]},
    "K19": {"B": ["K19_B"], "E": ["K19_E"], "P_suggestions": ["P11"]},
    "K20": {"B": ["K20_B"], "E": ["K20_E"], "P_suggestions": ["P01"]},
}


# Schnellfinder (für GUI)
QUICKFINDER: Dict[str, List[str]] = {
    "Normalbefund / keine PH": ["K01"],
    "Belastung: eher linkskardial": ["K02"],
    "Belastung: eher pulmonalvaskulär": ["K03"],
    "Grenzbefund / unklassifiziert": ["K04"],
    "Präkapillär: mild": ["K05"],
    "Präkapillär: mittel": ["K06"],
    "Präkapillär: schwer + CI↓": ["K07", "K08"],
    "Verlaufskontrolle PAH: gebessert": ["K09"],
    "Verlaufskontrolle PAH: progredient": ["K10"],
    "CTEPH/CTEPD Pfad": ["K11"],
    "Portopulmonal / Hyperzirkulation": ["K12"],
    "CTD-PAH / ILD-Thema": ["K13"],
    "Postkapillär (Gruppe 2)": ["K14"],
    "CpcPH (gemischt)": ["K15"],
    "Shunt": ["K16"],
    "Vasoreagibilität +": ["K17"],
    "Vasoreagibilität -": ["K18"],
    "PVOD-Verdacht": ["K19"],
    "Sekundäre Ursachen/Trigger": ["K20"],
}
