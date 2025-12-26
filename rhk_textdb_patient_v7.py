#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_textdb_patient_v7.py
Patientenfreundliche Textbausteine (Einfache Sprache, zusammenhängende Absätze).
Ziel: Die Inhalte sollen ohne Abkürzungen und ohne Zahlenwerte verständlich sein.

Hinweis:
Diese Texte sind bewusst "soft" formuliert und sollen das ärztliche Gespräch nicht ersetzen.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PatientBlock:
    id: str
    title: str
    template: str


# -----------------------------
# Basisbausteine (Absätze)
# -----------------------------
PATIENT_BLOCKS: Dict[str, PatientBlock] = {}

def _add(id: str, title: str, template: str) -> None:
    PATIENT_BLOCKS[id] = PatientBlock(id=id, title=title, template=template.strip())


_add(
    "PX_INTRO",
    "Einleitung",
    """
Hallo {name},

wir haben eine Herzkatheter-Untersuchung durchgeführt. Dabei messen wir, wie das Blut durch Herz und Lunge fließt und welche Drücke dabei entstehen.
""",
)

_add(
    "PX_WHAT_IS_PH",
    "Was bedeutet Lungenhochdruck?",
    """
Wenn der Druck in den Blutgefäßen der Lunge zu hoch ist, muss die rechte Herzkammer stärker arbeiten. Das kann zu Luftnot, Müdigkeit und eingeschränkter Belastbarkeit führen.
""",
)

_add(
    "PX_NO_PH",
    "Kein Lungenhochdruck in Ruhe",
    """
In Ruhe sind die Werte unauffällig. Das spricht gegen eine Lungenhochdruck-Erkrankung in Ruhe.
Das bedeutet nicht automatisch, dass Ihre Beschwerden „eingebildet“ sind – es gibt mehrere mögliche Ursachen, die wir weiter einordnen.
""",
)

_add(
    "PX_EX_LEFT",
    "Auffällige Belastungsreaktion – eher linkes Herz",
    """
In Ruhe waren die Werte unauffällig. Unter Belastung zeigte sich jedoch eine Reaktion, die eher dazu passt, dass das linke Herz unter Belastung „zu steif“ ist.
Dadurch kann es zu einem Rückstau kommen, der auch die Lunge belastet.
""",
)

_add(
    "PX_EX_PVASC",
    "Auffällige Belastungsreaktion – eher Lungengefäße",
    """
In Ruhe waren die Werte unauffällig. Unter Belastung zeigte sich jedoch eine auffällige Reaktion in den Lungengefäßen.
Das kann erklären, warum Belastung (z.B. Treppen, längeres Gehen) schneller anstrengend wird.
""",
)

_add(
    "PX_PRECAP",
    "Lungenhochdruck – v. a. Lungengefäße",
    """
Die Messwerte sprechen für eine Lungenhochdruck-Erkrankung, bei der vor allem der Widerstand in den Lungengefäßen erhöht ist.
Wir ordnen das zusammen mit den anderen Untersuchungen ein, um die wahrscheinlichste Ursache zu finden.
""",
)

_add(
    "PX_POSTCAP",
    "Lungenhochdruck – v. a. Rückstau vom linken Herzen",
    """
Die Messwerte sprechen für eine Lungenhochdruck-Erkrankung, die vor allem durch einen Rückstau vom linken Herzen mitbedingt ist.
In solchen Fällen ist es wichtig, die Behandlung des linken Herzens und des Flüssigkeitshaushalts gut einzustellen.
""",
)

_add(
    "PX_CPCPH",
    "Lungenhochdruck – Kombination",
    """
Die Messwerte sprechen für eine Kombination: Es gibt einen Rückstau vom linken Herzen und zusätzlich einen erhöhten Widerstand in den Lungengefäßen.
Das braucht häufig eine besonders sorgfältige Planung der nächsten Schritte und Kontrollen.
""",
)

_add(
    "PX_BORDERLINE",
    "Grenzbefund",
    """
Die Messwerte liegen in einem Grenzbereich oder sind nicht eindeutig. Das kommt vor.
Dann schauen wir besonders auf Ihre Beschwerden, die Bildgebung und ggf. auf Tests unter Belastung oder nach Flüssigkeitsgabe.
""",
)

_add(
    "PX_GROUP2_HINT",
    "Hinweis auf HFpEF/diastolische Komponente",
    """
Es gibt Hinweise, dass eine Form der Herzschwäche mit erhaltener Pumpkraft eine Rolle spielen könnte.
Dabei ist nicht die „Kraft“ das Problem, sondern eher die Entspannung des Herzmuskels. Das wird kardiologisch weiter eingeordnet.
""",
)

_add(
    "PX_GROUP3_HINT",
    "Hinweis auf Lungenerkrankung",
    """
Es gibt Hinweise, dass eine Lungenerkrankung oder eine niedrige Sauerstoffversorgung mitbeteiligt sein könnte.
Dann sind Lungenfunktion, Bildgebung und ggf. eine spezialisierte Pneumologie-Mitbetreuung wichtig.
""",
)

_add(
    "PX_GROUP4_HINT",
    "Hinweis auf frühere Blutgerinnsel",
    """
Es gibt Hinweise, dass frühere Blutgerinnsel in den Lungengefäßen beteiligt sein könnten.
Dann empfehlen wir spezielle Untersuchungen, um das sicher zu klären und die besten Behandlungsmöglichkeiten zu prüfen.
""",
)

_add(
    "PX_ANEMIA",
    "Blutarmut",
    """
Im Blutbild gibt es Hinweise auf eine Blutarmut. Das kann die Belastbarkeit deutlich beeinflussen und sollte gezielt abgeklärt werden.
""",
)

_add(
    "PX_CONGESTION",
    "Wassereinlagerung/Rückstau",
    """
Es gibt Hinweise auf Wassereinlagerung oder Rückstau. Dann ist es wichtig, den Flüssigkeitshaushalt zu optimieren und Nierenwerte regelmäßig zu kontrollieren.
""",
)

_add(
    "PX_NEXT_STEPS",
    "Wie geht es weiter?",
    """
Wir besprechen die Ergebnisse mit Ihnen und planen die nächsten Schritte. Das kann zusätzliche Diagnostik, eine Anpassung der Medikamente und Verlaufskontrollen beinhalten.
Wenn neue oder starke Beschwerden auftreten (z.B. Ohnmacht, starke Luftnot, Brustschmerz), suchen Sie bitte zeitnah ärztliche Hilfe.
""",
)


# -----------------------------
# Bündel: welche Bausteine typischerweise zusammen ausgegeben werden
# (nicht zwingend – die App kann zusätzlich dynamisch ergänzen)
# -----------------------------
PATIENT_BUNDLES: Dict[str, List[str]] = {
    "K01": ["PX_INTRO", "PX_NO_PH", "PX_NEXT_STEPS"],
    "K02": ["PX_INTRO", "PX_EX_LEFT", "PX_NEXT_STEPS"],
    "K03": ["PX_INTRO", "PX_EX_PVASC", "PX_NEXT_STEPS"],
    "K04": ["PX_INTRO", "PX_BORDERLINE", "PX_NEXT_STEPS"],
    "K05": ["PX_INTRO", "PX_PRECAP", "PX_WHAT_IS_PH", "PX_NEXT_STEPS"],
    "K06": ["PX_INTRO", "PX_PRECAP", "PX_WHAT_IS_PH", "PX_NEXT_STEPS"],
    "K07": ["PX_INTRO", "PX_PRECAP", "PX_WHAT_IS_PH", "PX_NEXT_STEPS"],
    "K11": ["PX_INTRO", "PX_PRECAP", "PX_GROUP4_HINT", "PX_NEXT_STEPS"],
    "K12": ["PX_INTRO", "PX_PRECAP", "PX_NEXT_STEPS"],
    "K13": ["PX_INTRO", "PX_PRECAP", "PX_GROUP3_HINT", "PX_NEXT_STEPS"],
    "K14": ["PX_INTRO", "PX_POSTCAP", "PX_NEXT_STEPS"],
    "K15": ["PX_INTRO", "PX_CPCPH", "PX_NEXT_STEPS"],
    "K16": ["PX_INTRO", "PX_BORDERLINE", "PX_NEXT_STEPS"],
    "K17": ["PX_INTRO", "PX_PRECAP", "PX_NEXT_STEPS"],
    "K18": ["PX_INTRO", "PX_PRECAP", "PX_NEXT_STEPS"],
}

def get_patient_block(block_id: str) -> PatientBlock:
    return PATIENT_BLOCKS[block_id]
