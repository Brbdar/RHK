#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_textdb_patient.py  (Patient:innen-Textdatenbank)
====================================================

Dieses Modul liefert patientenfreundliche Textbausteine ("Einfache Sprache")
für den RHK‑Befundassistenten.

Design:
- Primär werden ausgewählte Kernblöcke (v.a. K01–K20) in gut verständlicher Sprache
  gepflegt (Overrides).
- Für alle übrigen Blöcke wird automatisch ein einfacher, regelbasierter Fallback aus
  der Ärzt:innen-Version (rhk_textdb.py) erzeugt. So ist sichergestellt, dass *jeder*
  Block renderbar bleibt – auch wenn noch nicht alles manuell "übersetzt" ist.

Hinweis:
- Die patientenfreundlichen Texte sind bewusst "neutral" formuliert.
- Sie ersetzen kein ärztliches Gespräch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import re

try:
    import rhk_textdb as _src
except Exception:
    _src = None


@dataclass(frozen=True)
class TextBlock:
    id: str
    title: str
    template: str


def _simplify(template: str) -> str:
    """Ein sehr einfacher, regelbasierter Simplifier für den Fallback.

    Wichtig: Platzhalter {like_this} bleiben unverändert.
    """
    if not template:
        return ""
    t = str(template)

    replacements = {
        "pulmonale Hypertonie": "Lungenhochdruck",
        "Pulmonale Hypertonie": "Lungenhochdruck",
        "präkapillär": "vor den Lungenkapillaren (präkapillär)",
        "postkapillär": "durch Rückstau vom linken Herzen (postkapillär)",
        "pulmonalvenöse Stauung": "Rückstau in die Lunge",
        "zentralvenöse Stauung": "Rückstau im venösen System",
        "Echokardiographie": "Herzultraschall",
        "Differenzialdiagnostisch": "Als mögliche Ursachen kommen u.a. in Frage",
        "Ätiologie": "Ursache",
        "Genese": "Ursache",
        "Dekongestion": "Entwässerung",
        "Volumenmanagement": "Entwässerung / Flüssigkeitshaushalt",
        "V/Q-Szintigraphie": "Szintigraphie der Lungen‑Durchblutung (V/Q)",
        "CT-Pulmonalisangiographie": "CT der Lungengefäße",
        "CTPA": "CT der Lungengefäße",
        "spezialisierten Setting": "Spezialzentrum",
        "spezialisierten Zentrum": "Spezialzentrum",
    }
    for a, b in replacements.items():
        t = t.replace(a, b)

    # Whitespace glätten
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Manuelle "Einfache Sprache" Overrides (Kernpakete & häufige Zusätze)
# ---------------------------------------------------------------------------

_OVERRIDES: Dict[str, str] = {
    # --- K01 ---
    "K01_B": (
        "Die Messwerte in Ruhe sind unauffällig. Das spricht gegen einen Lungenhochdruck im Ruhezustand. "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient} {step_up_sentence_patient}"
    ),
    "K01_E": (
        "Aktuell gibt es in Ruhe keinen Hinweis auf Lungenhochdruck. "
        "Wenn weiterhin Luftnot oder Leistungsminderung besteht, können – je nach Situation – weitere Untersuchungen sinnvoll sein "
        "(z.B. Herzultraschall, Lungenfunktion, Belastungstests). Bitte besprechen Sie die nächsten Schritte mit Ihrem Behandlungsteam."
    ),

    # --- K02 ---
    "K02_B": (
        "In Ruhe sind die Werte unauffällig. Unter Belastung steigen die Drücke aber deutlich an, vor allem der PAWP/PCWP "
        "(dieser Wert spiegelt häufig den Füllungsdruck auf der linken Herzseite wider). "
        "Das passt eher zu einer Belastung der linken Herzhälfte (z.B. steifer Herzmuskel, Klappenproblem oder Blutdruck unter Belastung)."
    ),
    "K02_E": (
        "Wir empfehlen eine kardiologische Abklärung/Optimierung (z.B. Herzultraschall, Blutdruck- und Rhythmusdiagnostik – je nach Situation). "
        "Zusätzlich ist eine pneumologische Abklärung sinnvoll, wenn es Hinweise auf eine Lungenerkrankung gibt."
    ),

    # --- K03 ---
    "K03_B": (
        "In Ruhe sind die Werte unauffällig. Unter Belastung steigt der Lungendruck stärker als erwartet, "
        "ohne dass der PAWP/PCWP entsprechend stark ansteigt. "
        "Das kann ein Hinweis auf eine frühe Störung der Lungengefäße unter Belastung sein."
    ),
    "K03_E": (
        "Wir empfehlen eine weiterführende Abklärung, um mögliche Ursachen zu klären (z.B. Lungenfunktion, Bildgebung der Lunge, "
        "ggf. Untersuchung der Lungen‑Durchblutung (V/Q) sowie Herzultraschall). "
        "Ob eine spezielle Behandlung nötig ist, hängt vom Gesamtbild und dem Verlauf ab."
    ),

    # --- K04 ---
    "K04_B": (
        "Die Messwerte sind grenzwertig oder nicht eindeutig zuzuordnen. "
        "Es können Hinweise auf eine Durchstauungstendenz oder eine Mischsituation bestehen. "
        "Für eine sichere Einordnung braucht es die Zusammenschau mit Beschwerden, Vorbefunden und ggf. Verlauf."
    ),
    "K04_E": (
        "Wir empfehlen eine Verlaufskontrolle und – je nach Situation – weitere Untersuchungen (Herz, Lunge, Durchblutung der Lunge). "
        "Wichtig sind außerdem Blutdruckkontrolle, Volumen-/Gewichtsmanagement und die Behandlung von Begleiterkrankungen."
    ),

    # --- K05 ---
    "K05_B": (
        "Die Werte sprechen für einen leichten Lungenhochdruck, der aus den Lungengefäßen kommt "
        "(vor den Lungenkapillaren / präkapillär). "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient} {step_up_sentence_patient}"
    ),
    "K05_E": (
        "Als nächster Schritt ist es wichtig, die Ursache zu klären (Herz, Lunge, frühere Blutgerinnsel in der Lunge usw.). "
        "Dazu können z.B. Lungenfunktion, Bildgebung, V/Q‑Untersuchung und Herzultraschall gehören."
    ),

    # --- K06 ---
    "K06_B": (
        "Die Werte sprechen für einen mittelgradigen Lungenhochdruck aus den Lungengefäßen "
        "(vor den Lungenkapillaren / präkapillär). "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient} {step_up_sentence_patient}"
    ),
    "K06_E": (
        "Wir empfehlen eine weiterführende Abklärung der Ursache und – je nach Situation – eine Therapieplanung im Spezialzentrum. "
        "Regelmäßige Kontrollen sind wichtig."
    ),

    # --- K07 ---
    "K07_B": (
        "Die Werte sprechen für einen ausgeprägten Lungenhochdruck aus den Lungengefäßen (präkapillär). "
        "Die Herzleistung ist dabei vermindert, was das rechte Herz stark belasten kann. "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient}"
    ),
    "K07_E": (
        "Bei dieser Ausprägung sind engmaschige Kontrollen und eine Behandlung im Spezialzentrum wichtig. "
        "Je nach Situation müssen Therapieoptionen angepasst oder erweitert werden. "
        "Wenn Wasseransammlungen/Rückstau bestehen, ist Entwässerung besonders wichtig."
    ),

    # --- K08 ---
    "K08_B": (
        "Es liegt ein schwerer Lungenhochdruck vor. Zusätzlich gibt es deutliche Zeichen eines Rückstaus im venösen System. "
        "In dieser Situation steht häufig zunächst die Stabilisierung/Entwässerung im Vordergrund."
    ),
    "K08_E": (
        "Wir empfehlen zunächst eine Behandlung des Rückstaus/der Flüssigkeitsüberlastung (Entwässerung) und danach eine erneute "
        "Bewertung im Spezialzentrum. Die weitere Therapieplanung erfolgt individuell – auch nach Ihren Wünschen/Präferenzen."
    ),

    # --- K11 ---
    "K11_B": (
        "Die Werte sprechen für einen Lungenhochdruck aus den Lungengefäßen. "
        "Im passenden Kontext kann eine chronische Form von Lungenembolie/Blutgerinnseln als Ursache in Frage kommen (CTEPH/CTEPD)."
    ),
    "K11_E": (
        "Wir empfehlen eine gezielte Abklärung auf chronische Lungenembolie (z.B. V/Q‑Untersuchung, CT der Lungengefäße) "
        "und die Vorstellung in einem Spezialzentrum, das über operative/interventionelle Behandlungen entscheiden kann. "
        "Eine Blutverdünnung muss – falls indiziert – konsequent fortgeführt werden."
    ),

    # --- K12 ---
    "K12_B": (
        "Die Drücke in der Lunge sind erhöht, gleichzeitig ist die Herzleistung sehr hoch (\"Hochfluss\"). "
        "Der Widerstand der Lungengefäße ist dabei nur mäßig erhöht."
    ),
    "K12_E": (
        "Wir empfehlen die Abklärung möglicher Ursachen für einen Hochfluss (z.B. Blutarmut, Schilddrüse, Entzündung, Lebererkrankung). "
        "Die weitere Therapie richtet sich nach dem Gesamtbild."
    ),

    # --- K14 ---
    "K14_B": (
        "Der PAWP/PCWP ist erhöht. Das spricht für einen Rückstau vom linken Herzen in Richtung Lunge "
        "(postkapilläre Ursache)."
    ),
    "K14_E": (
        "Im Vordergrund stehen eine kardiologische Behandlung/Optimierung (Entwässerung, Blutdruck, Rhythmus, ggf. Klappen). "
        "Eine spezielle Lungenhochdruck‑Therapie ist bei dieser Ursache in der Regel nicht die erste Wahl."
    ),

    # --- K15 ---
    "K15_B": (
        "Es gibt Hinweise auf eine Mischsituation: Rückstau vom linken Herzen (PAWP/PCWP erhöht) "
        "und zusätzlich einen erhöhten Widerstand in den Lungengefäßen."
    ),
    "K15_E": (
        "Wir empfehlen sowohl die Optimierung der linken Herzseite (z.B. Entwässerung, Blutdruck, Rhythmus) "
        "als auch eine vollständige Abklärung der Lungengefäße im Spezialzentrum. "
        "Eine spezielle Therapie wird individuell entschieden."
    ),

    # --- K16 ---
    "K16_B": (
        "Bei der Messung der Sauerstoffwerte zeigt sich ein Hinweis auf eine mögliche Kurzschlussverbindung (Shunt). "
        "Ob das behandlungsbedürftig ist, hängt von der Ursache und der Größe des Shunts ab."
    ),
    "K16_E": (
        "Wir empfehlen eine weiterführende Abklärung (Herzultraschall, ggf. transösophagealer Ultraschall/TEE oder Bildgebung), "
        "um die Ursache und Relevanz zu klären."
    ),

    # --- K17/K18 (Vasoreaktivität) ---
    "K17_B": (
        "Beim Vasoreaktivitätstest haben sich die Werte unter Testmedikation deutlich gebessert (positiver Test)."
    ),
    "K17_E": (
        "Ein positiver Vasoreaktivitätstest kann bedeuten, dass bestimmte Therapiepfade in Frage kommen. "
        "Die konkrete Behandlung wird im Spezialzentrum geplant und engmaschig kontrolliert."
    ),
    "K18_B": (
        "Beim Vasoreaktivitätstest zeigte sich keine ausreichende Besserung der Werte (negativer Test)."
    ),
    "K18_E": (
        "Bei negativem Vasoreaktivitätstest wird die weitere Behandlung nach Ursache und Gesamtrisiko geplant. "
        "Kontrollen und ggf. weitere Diagnostik bleiben wichtig."
    ),

    # --- B11/B12 (Volumenchallenge) ---
    "B11": (
        "Nach Flüssigkeitsgabe stieg der PAWP/PCWP deutlich an. Das kann auf eine versteckte Belastung/Schwäche der linken Herzhälfte hinweisen."
    ),
    "B12": (
        "Nach Flüssigkeitsgabe zeigte sich kein deutlicher Anstieg des PAWP/PCWP. Das spricht im Untersuchungssetting gegen eine volumen‑auslösbare Durchstauung."
    ),
}


# ---------------------------------------------------------------------------
# Aufbau ALL_BLOCKS (Fallback aus rhk_textdb + Overrides)
# ---------------------------------------------------------------------------

ALL_BLOCKS: Dict[str, TextBlock] = {}

if _src is not None and hasattr(_src, "ALL_BLOCKS"):
    for bid, b in _src.ALL_BLOCKS.items():
        title = getattr(b, "title", bid)
        tpl = getattr(b, "template", "")
        ALL_BLOCKS[bid] = TextBlock(id=bid, title=title, template=_simplify(tpl))

# Overrides anwenden (auch wenn _src fehlt)
for bid, tpl in _OVERRIDES.items():
    if bid in ALL_BLOCKS:
        ALL_BLOCKS[bid] = TextBlock(id=bid, title=ALL_BLOCKS[bid].title, template=tpl)
    else:
        ALL_BLOCKS[bid] = TextBlock(id=bid, title=bid, template=tpl)


def get_block(block_id: str) -> Optional[TextBlock]:
    return ALL_BLOCKS.get(block_id)
