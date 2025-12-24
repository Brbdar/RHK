#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_textdb_patient_v4.py

Patienten-Textdatenbank (sehr einfache Sprache) für den RHK-Befundassistenten.

Ziel:
- Kurze Sätze.
- Möglichst wenig Fachwörter.
- Klare Handlungsschritte.
- Verständlich ohne medizinisches Vorwissen.

Hinweis:
Diese Datenbank deckt vor allem Maßnahmen-/Empfehlungs-Module ab (P/BE/C/G).
Wenn ein Block fehlt, soll das Hauptprogramm einen Fallback benutzen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PatientBlock:
    id: str
    title: str
    template: str


PATIENT_BLOCKS: Dict[str, PatientBlock] = {
    # ---------------------------------------------------------------------
    # P‑Module (Maßnahmen / Procedere)
    # ---------------------------------------------------------------------
    "P01": PatientBlock(
        id="P01",
        title="Weitere Untersuchungen (Basis)",
        template=(
            "Wir empfehlen weitere Untersuchungen, damit wir die Ursache besser verstehen:\n"
            "• Herz‑Ultraschall (Echokardiographie).\n"
            "• Lungenfunktion.\n"
            "• Bildgebung der Lunge (z.B. CT) – je nach Situation.\n"
            "• Test auf alte Lungenembolien (z.B. V/Q‑Szintigraphie).\n"
            "• Blutuntersuchungen – je nach Situation."
        ),
    ),
    "P02": PatientBlock(
        id="P02",
        title="Entwässerung (wenn Wasser im Körper)",
        template=(
            "Wenn Sie Wasser im Körper haben (z.B. dicke Beine, Gewichtszunahme): "
            "Dann müssen Entwässerungs‑Medikamente (Diuretika) oft angepasst werden. "
            "Dabei sollen Nierenwerte und Salze im Blut kontrolliert werden."
        ),
    ),
    "P03": PatientBlock(
        id="P03",
        title="Spezielle Behandlung (wenn geplant)",
        template=(
            "{therapy_plan_sentence}"
        ),
    ),
    "P04": PatientBlock(
        id="P04",
        title="Spezielle Behandlung erweitern (wenn geplant)",
        template=(
            "{therapy_plan_sentence}"
        ),
    ),
    "P05": PatientBlock(
        id="P05",
        title="Spezielle Behandlung (wenn geplant)",
        template=(
            "{therapy_plan_sentence}"
        ),
    ),
    "P06": PatientBlock(
        id="P06",
        title="Behandlung im Spezial‑Zentrum besprechen",
        template=(
            "Wir besprechen im Spezial‑Zentrum, ob eine stärkere Behandlung nötig ist. "
            "Wir erklären Ihnen Vor‑ und Nachteile."
        ),
    ),
    "P07": PatientBlock(
        id="P07",
        title="Studie prüfen",
        template=(
            "Wir prüfen, ob eine Teilnahme an einer Studie für Sie möglich ist."
        ),
    ),
    "P08": PatientBlock(
        id="P08",
        title="Besprechung der Lungen‑Bilder",
        template=(
            "Wenn das CT Hinweise auf eine Lungenerkrankung zeigt, "
            "soll der Befund in einem Team aus Lungenärzt:innen und Radiologie besprochen werden."
        ),
    ),
    "P09": PatientBlock(
        id="P09",
        title="Termin beim Herz‑Team",
        template=(
            "Wir empfehlen eine Kontrolle beim Herz‑Team (Kardiologie). "
            "Dabei geht es z.B. um Herzklappen, Herzrhythmus oder Herzschwäche."
        ),
    ),
    "P10": PatientBlock(
        id="P10",
        title="Blutverdünnung / Gerinnung",
        template=(
            "{anticoagulation_plan_sentence}"
        ),
    ),
    "P11": PatientBlock(
        id="P11",
        title="Nächste Kontrolle",
        template=(
            "Wir empfehlen die nächste Kontrolle in {followup_timing_desc}. "
            "Was wir genau kontrollieren, hängt von Beschwerden und Werten ab."
        ),
    ),
    "P12": PatientBlock(
        id="P12",
        title="Lungen‑Abklärung",
        template=(
            "Wir empfehlen weitere Lungen‑Untersuchungen (z.B. Lungenfunktion und ggf. CT), "
            "damit wir die Luftnot besser einordnen können."
        ),
    ),
    "P13": PatientBlock(
        id="P13",
        title="Eisen / Blutarmut",
        template=(
            "Wenn Blutarmut oder Eisenmangel möglich ist: "
            "Dann sollen Eisenwerte bestimmt und bei Bedarf behandelt werden."
        ),
    ),

    # Optional: freier Lungenfunktion‑Kurztext (falls als Modul auftaucht)
    "PULM_LUFU_SUMMARY": PatientBlock(
        id="PULM_LUFU_SUMMARY",
        title="Lungenfunktion (Kurz)",
        template=(
            "Lungenfunktion (Kurz): {lufu_summary}."
        ),
    ),

    # ---------------------------------------------------------------------
    # BE‑Module (Sicherheits‑/Zusatz‑Empfehlungen)
    # ---------------------------------------------------------------------
    "BE01_DIURETICS_SAFETY": PatientBlock(
        id="BE01_DIURETICS_SAFETY",
        title="Sicherheit bei Entwässerung",
        template=(
            "Wenn Entwässerungs‑Medikamente geändert werden: "
            "Bitte Nierenwerte und Salze im Blut kontrollieren."
        ),
    ),
    "BE02_IRON": PatientBlock(
        id="BE02_IRON",
        title="Eisenmangel",
        template=(
            "Bei Verdacht auf Eisenmangel: Eisenwerte bestimmen und bei Bedarf behandeln."
        ),
    ),
    "BE03_STUDY": PatientBlock(
        id="BE03_STUDY",
        title="Studien",
        template=(
            "{study_sentence}"
        ),
    ),
    "BE04_PATIENT_PREF": PatientBlock(
        id="BE04_PATIENT_PREF",
        title="Ihre Entscheidung",
        template=(
            "Nach dem Gespräch möchten Sie aktuell {declined_item} nicht. "
            "Wir besprechen Alternativen und das weitere Vorgehen."
        ),
    ),
    "BE05_ILD_BOARD": PatientBlock(
        id="BE05_ILD_BOARD",
        title="Lungen‑Konferenz",
        template=(
            "Wenn das CT Hinweise auf eine Lungenerkrankung (z.B. Narben/Entzündung) zeigt: "
            "Besprechung im ILD‑Team (Lunge + Radiologie) empfohlen."
        ),
    ),
    "BE06_CTEPH_BOARD": PatientBlock(
        id="BE06_CTEPH_BOARD",
        title="Spezial‑Konferenz bei alten Lungenembolien",
        template=(
            "Wenn der Verdacht auf alte Lungenembolien besteht: "
            "Vorstellung in einem Spezial‑Zentrum (CTEPH‑Board) empfohlen."
        ),
    ),
    "BE07_ANTICOAG": PatientBlock(
        id="BE07_ANTICOAG",
        title="Gerinnungs‑Sprechstunde",
        template=(
            "Wir empfehlen eine Vorstellung in der Gerinnungs‑Sprechstunde, "
            "damit die Blutverdünnung gut eingestellt ist (" 
            "{anticoag_context})."
        ),
    ),
    "BE08_POST_PROC": PatientBlock(
        id="BE08_POST_PROC",
        title="Nach dem Katheter",
        template=(
            "Bitte Punktionsstelle kontrollieren und körperliche Schonung wie besprochen. "
            "Bei starker Blutung, starker Schwellung, Schmerzen oder Kreislaufproblemen: sofort melden."
        ),
    ),

    # ---------------------------------------------------------------------
    # C‑Module (Zusatz‑Hinweise)
    # ---------------------------------------------------------------------
    "C01_OSA_OHS": PatientBlock(
        id="C01_OSA_OHS",
        title="Schlaf‑Atmung prüfen",
        template=(
            "Wir empfehlen eine Untersuchung auf Schlafapnoe oder nächtliche Atemschwäche. "
            "Wenn nötig: Behandlung mit Maske (CPAP/NIV) und Unterstützung beim Abnehmen."
        ),
    ),
    "C02_SCHISTO": PatientBlock(
        id="C02_SCHISTO",
        title="Reise‑/Tropen‑Ursachen",
        template=(
            "Wenn Reise‑ oder Tropen‑Kontakt dazu passt: "
            "Abklärung auf seltene Infektionen in Zusammenarbeit mit Infektiologie/Tropenmedizin."
        ),
    ),
    "C03_HIGH_FLOW_B": PatientBlock(
        id="C03_HIGH_FLOW_B",
        title="Sehr hoher Blutfluss (Beurteilung)",
        template=(
            "Es gibt Hinweise auf einen sehr hohen Blutfluss durch das Herz. "
            "Die Widerstände in den Lungengefäßen sind {pvr_phrase}."
        ),
    ),
    "C03_HIGH_FLOW_E": PatientBlock(
        id="C03_HIGH_FLOW_E",
        title="Sehr hoher Blutfluss (Empfehlung)",
        template=(
            "Wir empfehlen die Suche nach Ursachen für sehr hohen Blutfluss (z.B. Blutarmut, Schilddrüse, "
            "Gefäß‑Kurzschluss/AV‑Fistel, Entzündung)."
        ),
    ),
    "C04_CTEPD_NO_PH": PatientBlock(
        id="C04_CTEPD_NO_PH",
        title="Alte Lungenembolien ohne Lungenhochdruck in Ruhe",
        template=(
            "Es gibt Hinweise auf frühere Lungenembolien. "
            "In Ruhe zeigt sich aber kein Lungenhochdruck. "
            "Wenn Sie trotzdem stark Luftnot haben, besprechen wir weitere Tests (z.B. Belastung) und ggf. eine Vorstellung im Spezial‑Zentrum."
        ),
    ),
    "C05_VALVE_VWAVE": PatientBlock(
        id="C05_VALVE_VWAVE",
        title="Herzklappen prüfen",
        template=(
            "Es gibt Hinweise, dass eine Herzklappe mitbeteiligt sein könnte. "
            "Wir empfehlen eine genaue Ultraschall‑Untersuchung des Herzens (ggf. auch TEE)."
        ),
    ),

    # ---------------------------------------------------------------------
    # G‑Module (Allgemeine Sätze)
    # ---------------------------------------------------------------------
    "G01": PatientBlock(
        id="G01",
        title="Allgemeiner Hinweis",
        template=(
            "Wir planen die weiteren Schritte gemeinsam. "
            "Dabei berücksichtigen wir Ihre Beschwerden, Begleiterkrankungen und Ihre Wünsche."
        ),
    ),
    "G02": PatientBlock(
        id="G02",
        title="Keine Spezial‑Therapie aktuell",
        template=(
            "Im Moment sehen wir keine klare Notwendigkeit für eine spezielle Therapie gegen Lungenhochdruck. "
            "Wir empfehlen Kontrollen und die Abklärung der wahrscheinlichen Ursache."
        ),
    ),
    "G03": PatientBlock(
        id="G03",
        title="Volumen / Wasserhaushalt",
        template=(
            "Der Wasserhaushalt ist wichtig. "
            "Bitte regelmäßig wiegen und bei schneller Gewichtszunahme Rücksprache halten."
        ),
    ),
}


def get_patient_block(block_id: str) -> Optional[PatientBlock]:
    return PATIENT_BLOCKS.get(block_id)
