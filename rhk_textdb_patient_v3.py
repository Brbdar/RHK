#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_textdb_patient_v3.py

Sehr einfache Sprache ("Patienten-Version") für den RHK-Befundassistenten.

Ziel:
- Kurze Sätze.
- Wenig Fachwörter.
- Klare Handlungsanweisungen.
- Für nicht-medizinische Leser:innen verständlich.

Hinweis:
Diese Datenbank deckt vor allem Maßnahmen-/Empfehlungs-Module (P/BE/C/G) ab.
Für nicht vorhandene IDs soll das Hauptprogramm eine Fallback-Vereinfachung verwenden.
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
    # -------------------------
    # Procedere / Maßnahmen (P...)
    # -------------------------
    "P01": PatientBlock(
        id="P01",
        title="Weitere Basis-Untersuchungen",
        template=(
            "Wir empfehlen weitere Untersuchungen, um die Ursache besser zu finden:\n"
            "• Herz-Ultraschall (Echokardiographie).\n"
            "• Lungenfunktion.\n"
            "• Bildgebung der Lunge (z.B. CT) – je nach Situation.\n"
            "• Untersuchung auf frühere Lungenembolien (z.B. V/Q-Szintigrafie).\n"
            "• Blutuntersuchungen – je nach Situation."
        ),
    ),
    "P02": PatientBlock(
        id="P02",
        title="Entwässerung (bei Wasser/Schwellungen)",
        template=(
            "Wenn Sie Wasser in den Beinen oder Bauch haben: "
            "Die Entwässerungs-Tabletten müssen ggf. angepasst werden. "
            "Dabei sollen Blutwerte (Niere/Salze) eng kontrolliert werden."
        ),
    ),
    "P03": PatientBlock(
        id="P03",
        title="Spezielle Medikamente (wenn geplant)",
        template=(
            "{therapy_plan_sentence}"
        ),
    ),
    "P04": PatientBlock(
        id="P04",
        title="Spezielle Medikamente ergänzen (wenn geplant)",
        template=(
            "{therapy_plan_sentence}"
        ),
    ),
    "P05": PatientBlock(
        id="P05",
        title="Spezielle Medikamente (wenn geplant)",
        template=(
            "{therapy_plan_sentence}"
        ),
    ),
    "P06": PatientBlock(
        id="P06",
        title="Therapie-Ausbau im Zentrum",
        template=(
            "Wir besprechen im Spezial-Zentrum, ob eine stärkere Behandlung nötig ist. "
            "Das hängt von Ihrer Gesamtsituation ab."
        ),
    ),
    "P07": PatientBlock(
        id="P07",
        title="Studien",
        template="Wenn passend, prüfen wir eine Teilnahme an einer Studie.",
    ),
    "P08": PatientBlock(
        id="P08",
        title="Lungen-Konferenz",
        template=(
            "Wenn es Hinweise auf eine Lungen-Narbenkrankheit gibt, "
            "soll der CT-Befund in einer Fach-Konferenz besprochen werden."
        ),
    ),
    "P09": PatientBlock(
        id="P09",
        title="Herzärztliche Mitbeurteilung",
        template=(
            "Wir empfehlen eine Kontrolle beim Herz-Team (Kardiologie), "
            "zum Beispiel für Herzklappen, Herzrhythmus oder Herzschwäche."
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
        title="Kontrolle / Verlauf",
        template=(
            "Wir empfehlen eine Verlaufskontrolle in {followup_timing_desc}. "
            "Welche Kontrollen genau nötig sind, hängt von den Beschwerden und den Werten ab."
        ),
    ),
    "P12": PatientBlock(
        id="P12",
        title="Lungen-Abklärung",
        template=(
            "Wir empfehlen weitere Lungen-Untersuchungen (Lungenfunktion, ggf. Bildgebung), "
            "um die Ursache der Luftnot besser einzuordnen."
        ),
    ),
    "P13": PatientBlock(
        id="P13",
        title="Eisen / Blutarmut",
        template=(
            "Wenn eine Blutarmut oder Eisenmangel möglich ist, "
            "sollen Eisenwerte bestimmt und ggf. behandelt werden."
        ),
    ),

    # -------------------------
    # Zusatz-Empfehlungen / Safety (BE...)
    # -------------------------
    "BE01_DIURETICS_SAFETY": PatientBlock(
        id="BE01_DIURETICS_SAFETY",
        title="Sicherheit bei Entwässerung",
        template="Bei Entwässerungs-Medikamenten: Bitte Nierenwerte und Salze im Blut kontrollieren.",
    ),
    "BE02_IRON": PatientBlock(
        id="BE02_IRON",
        title="Eisenmangel",
        template="Bei Verdacht auf Eisenmangel: Eisenwerte bestimmen und ggf. behandeln.",
    ),
    "BE03_STUDY": PatientBlock(
        id="BE03_STUDY",
        title="Studien",
        template="{study_sentence}",
    ),
    "BE04_PATIENT_PREF": PatientBlock(
        id="BE04_PATIENT_PREF",
        title="Ihre Entscheidung",
        template="Nach Aufklärung möchten Sie aktuell {declined_item} nicht.",
    ),
    "BE05_ILD_BOARD": PatientBlock(
        id="BE05_ILD_BOARD",
        title="Lungen-Konferenz",
        template="Wir empfehlen die Besprechung der CT-Bilder in einer Lungen-Fach-Konferenz.",
    ),
    "BE06_CTEPH_BOARD": PatientBlock(
        id="BE06_CTEPH_BOARD",
        title="CTEPH-Konferenz",
        template="Wir empfehlen die Vorstellung in einer Spezial-Konferenz für chronische Lungenembolien (CTEPH).",
    ),
    "BE07_ANTICOAG": PatientBlock(
        id="BE07_ANTICOAG",
        title="Blutverdünnung prüfen",
        template="Wir empfehlen eine Kontrolle der Blutverdünnung in einer Gerinnungs-Sprechstunde.",
    ),
    "BE08_POST_PROC": PatientBlock(
        id="BE08_POST_PROC",
        title="Nach dem Eingriff",
        template="Bitte Punktionsstelle und Kreislauf wie üblich überwachen.",
    ),

    # -------------------------
    # Ursachen-Module (C...)
    # -------------------------
    "C01_OSA_OHS": PatientBlock(
        id="C01_OSA_OHS",
        title="Schlaf-Atemstörung",
        template=(
            "Bei Verdacht auf Schlaf-Atemstörung empfehlen wir eine Schlaf-Untersuchung "
            "(z.B. Polygraphie) und Behandlung (z.B. CPAP), wenn nötig."
        ),
    ),
    "C02_SCHISTO": PatientBlock(
        id="C02_SCHISTO",
        title="Tropenmedizin",
        template="Bei passender Reise-Anamnese empfehlen wir eine Abklärung in der Tropenmedizin/Infektiologie.",
    ),
    "C04_CTEPD_NO_PH": PatientBlock(
        id="C04_CTEPD_NO_PH",
        title="Perfusionsdefekte ohne Ruhe-Lungenhochdruck",
        template="Wenn es Hinweise auf ältere Lungenembolien gibt, besprechen wir das im Spezial-Zentrum.",
    ),
    "C05_VALVE_VWAVE": PatientBlock(
        id="C05_VALVE_VWAVE",
        title="Herzklappen",
        template="Bei Verdacht auf Herzklappen-Problem: Bitte Kardiologie-Kontrolle (Herz-Ultraschall).",
    ),

    # -------------------------
    # Allgemein (G...)
    # -------------------------
    "G01": PatientBlock(
        id="G01",
        title="Allgemein",
        template="Das weitere Vorgehen wird mit Ihnen gemeinsam und passend zur Gesamtsituation geplant.",
    ),
    "G03": PatientBlock(
        id="G03",
        title="Volumen / Gewicht",
        template="Bitte auf Gewicht und Wassereinlagerungen achten. Ein gutes Volumen-Management ist wichtig.",
    ),
}


def get_patient_block(block_id: str) -> Optional[PatientBlock]:
    return PATIENT_BLOCKS.get(block_id)
