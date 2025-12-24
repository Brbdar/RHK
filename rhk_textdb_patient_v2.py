#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rhk_textdb_patient.py  (Patient:innen-Textdatenbank)
====================================================

Dieses Modul liefert patientenfreundliche Textbausteine in *sehr einfacher Sprache*
für den RHK‑Befundassistenten.

Ziel
- Kurze Sätze.
- Möglichst wenig Fachwörter.
- Wenn Fachwörter nötig sind: kurz erklären.

Technik
- Für die häufigsten Befund-Pakete (K01–K20) und wichtige Procedere-Module (P01–P13)
  gibt es manuelle, gut lesbare Übersetzungen (Overrides).
- Für alle übrigen Blöcke wird ein stärkerer, regelbasierter Fallback aus der
  Ärzt:innen-Version (rhk_textdb.py) erzeugt.

Hinweis
- Diese Texte ersetzen kein Arztgespräch.
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


def _protect_placeholders(text: str) -> tuple[str, Dict[str, str]]:
    """Ersetzt {platzhalter} temporär durch Tokens, damit Replacements sie nicht verändern."""
    if not text:
        return "", {}
    mapping: Dict[str, str] = {}
    def repl(m: re.Match) -> str:
        key = f"__PH_{len(mapping)}__"
        mapping[key] = m.group(0)
        return key
    protected = re.sub(r"\{[^{}]+\}", repl, text)
    return protected, mapping


def _restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def _simplify(template: str) -> str:
    """Regelbasierter Fallback in sehr einfacher Sprache.

    Wichtig: Platzhalter {like_this} bleiben erhalten.
    """
    if not template:
        return ""

    t = str(template)

    # Platzhalter schützen
    t, ph = _protect_placeholders(t)

    # 1) Abkürzungen und typische "Arzt-Formulierungen" vereinfachen
    # (nicht perfekt – aber deutlich einfacher als das Original)
    replacements = {
        "pulmonale Hypertonie": "Lungenhochdruck",
        "Pulmonale Hypertonie": "Lungenhochdruck",
        "PH-spezifische": "spezielle",
        "PH-spezifisch": "speziell",
        "hämodynamisch": "bei den Messwerten",
        "Hämodynamik": "Messwerte",
        "Konstellation": "Situation",
        "Konstellationen": "Situationen",
        "Differenzialdiagnostisch": "Mögliche Ursachen sind",
        "Ätiologie": "Ursache",
        "Genese": "Ursache",
        "klinischer Kontext": "Gesamtsituation",
        "spezialisierten Setting": "Spezialzentrum",
        "spezialisierten Zentrum": "Spezialzentrum",
        "interdisziplinär": "gemeinsam im Team",
        "Reevaluation": "erneute Kontrolle",
        "zeitnah": "bald",
        "ggf.": "wenn nötig",
        "z.B.": "zum Beispiel",
        "inkl.": "",
        "insb.": "besonders",
        "indiziert": "sinnvoll",
        "Kontraindikationen": "Gegenanzeigen",
        "Dekongestion": "Entwässerung",
        "Volumenmanagement": "Entwässerung und Flüssigkeit",
        "pulmonalvenöse Stauung": "Rückstau in die Lunge",
        "zentralvenöse Stauung": "Rückstau im Körper",
        "Echokardiographie": "Herzultraschall",
        "V/Q-Szintigraphie": "Untersuchung der Lungen‑Durchblutung (V/Q)",
        "CT-Pulmonalisangiographie": "CT der Lungengefäße",
        "CTPA": "CT der Lungengefäße",
        "Komorbiditäten": "Begleiterkrankungen",
        "Risikoprofil": "Risiko",
        "Risikokonstellation": "Risiko",
        "Therapieeskalation": "Therapie-Ausbau",
        "Therapieeinleitung": "Start der Behandlung",
        "Therapieanpassung": "Anpassung der Behandlung",
        "pathologisch": "auffällig",
        "unauffällig": "normal",
        "präkapillär": "aus den Gefäßen der Lunge (präkapillär)",
        "postkapillär": "Rückstau vom linken Herzen (postkapillär)",
        "cpcPH": "Mischform",
    }
    for a, b in replacements.items():
        t = t.replace(a, b)

    # 2) Typische Sonderzeichen/Bullets vereinheitlichen
    t = t.replace("•", "-")

    # 3) Whitespace glätten
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)

    # 4) Sehr lange Klammern vereinfachen (grobe Regel)
    #    Beispiel: "(Kontraindikationen/Interaktionen beachten; ...)" → weglassen
    t = re.sub(r"\([^)]{60,}\)", "", t)

    # Platzhalter zurück
    t = _restore_placeholders(t, ph)
    return t.strip()


# ---------------------------------------------------------------------------
# Manuelle Overrides in sehr einfacher Sprache
# ---------------------------------------------------------------------------

_OVERRIDES: Dict[str, str] = {
    # ---------
    # K-Pakete
    # ---------
    "K01_B": (
        "Die Werte in Ruhe sind normal. Das spricht gegen Lungenhochdruck in Ruhe. "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient} {step_up_sentence_patient}"
    ),
    "K01_E": (
        "Im Moment sehen wir in Ruhe keinen Lungenhochdruck. "
        "Wenn Sie weiter Luftnot haben, prüfen wir andere Ursachen (Herz und Lunge)."
    ),

    "K02_B": (
        "In Ruhe sind die Werte normal. "
        "Bei Belastung steigt der Druck am linken Herzen deutlich an. "
        "Das passt eher zu einer Herz‑Ursache."
    ),
    "K02_E": (
        "Wir empfehlen: Herz prüfen (Herzultraschall, Blutdruck, Herzrhythmus – je nach Bedarf). "
        "Auch die Lunge prüfen, wenn es Hinweise auf eine Lungenerkrankung gibt."
    ),

    "K03_B": (
        "In Ruhe sind die Werte normal. "
        "Bei Belastung steigt der Lungendruck zu stark. "
        "Das kann auf ein Problem der Lungen‑Gefäße hinweisen."
    ),
    "K03_E": (
        "Wir empfehlen weitere Untersuchungen, um die Ursache zu finden (Herz und Lunge). "
        "Kontrollen sind wichtig."
    ),

    "K04_B": (
        "Die Messwerte sind grenzwertig. "
        "Im Moment ist die Ursache nicht eindeutig."
    ),
    "K04_E": (
        "Wir empfehlen Kontrollen und – je nach Beschwerden – weitere Untersuchungen von Herz und Lunge."
    ),

    "K05_B": (
        "Es gibt einen leichten Lungenhochdruck. "
        "Die Ursache liegt eher in den Gefäßen der Lunge. "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient} {step_up_sentence_patient}"
    ),
    "K05_E": (
        "Wichtig ist jetzt: die Ursache klären. "
        "Dazu gehören oft Untersuchungen von Herz und Lunge und Bluttests."
    ),

    "K06_B": (
        "Es gibt einen mittleren Lungenhochdruck. "
        "Die Ursache liegt eher in den Gefäßen der Lunge. "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient} {step_up_sentence_patient}"
    ),
    "K06_E": (
        "Wir empfehlen eine Behandlung und Kontrollen im Spezialzentrum. "
        "Dort wird das weitere Vorgehen geplant."
    ),

    "K07_B": (
        "Es gibt einen schweren Lungenhochdruck. "
        "Die Herzleistung ist dabei niedrig. "
        "Das kann das rechte Herz stark belasten. "
        "{cv_stauung_phrase_patient} {pv_stauung_phrase_patient}"
    ),
    "K07_E": (
        "Wir empfehlen eine enge Kontrolle und Behandlung im Spezialzentrum. "
        "Wenn ein Rückstau/Wasser im Körper besteht: Entwässerung ist besonders wichtig."
    ),

    "K08_B": (
        "Es gibt einen schweren Lungenhochdruck. "
        "Gleichzeitig gibt es einen starken Rückstau im Körper."
    ),
    "K08_E": (
        "Wir empfehlen zuerst: den Rückstau behandeln (Entwässerung). "
        "Danach: erneute Kontrolle und Planung im Spezialzentrum."
    ),

    "K09_B": (
        "Im Verlauf sehen die Messwerte besser aus als vorher (wenn ein Vorwert vorhanden ist)."
    ),
    "K09_E": (
        "Wir empfehlen: aktuelle Behandlung weiterführen und regelmäßig kontrollieren."
    ),

    "K10_B": (
        "Im Verlauf sind die Messwerte nicht ausreichend gebessert oder wieder schlechter."
    ),
    "K10_E": (
        "Wir empfehlen: Behandlung im Spezialzentrum prüfen und ggf. anpassen. "
        "Kontrollen sind wichtig."
    ),

    "K11_B": (
        "Die Werte sprechen für Lungenhochdruck. "
        "Eine mögliche Ursache sind alte/chronische Blutgerinnsel in der Lunge (CTEPH/CTEPD)."
    ),
    "K11_E": (
        "Wir empfehlen eine Abklärung auf chronische Lungenembolie (zum Beispiel V/Q und CT der Lungengefäße) "
        "und die Vorstellung im Spezialzentrum. "
        "Wenn eine Blutverdünnung nötig ist, soll sie zuverlässig eingenommen werden."
    ),

    "K12_B": (
        "Die Drücke in der Lunge sind erhöht. "
        "Gleichzeitig pumpt das Herz sehr viel Blut (Hochfluss)."
    ),
    "K12_E": (
        "Wir empfehlen: Ursachen für Hochfluss prüfen (zum Beispiel Blutarmut, Schilddrüse, Entzündung, Leber)."
    ),

    "K13_B": (
        "Die Werte sprechen für Lungenhochdruck. "
        "Es gibt (je nach Kontext) Hinweise auf eine Grunderkrankung (zum Beispiel eine Rheuma‑Erkrankung)."
    ),
    "K13_E": (
        "Wir empfehlen: Ursache weiter abklären und Behandlung im Spezialzentrum planen."
    ),

    "K14_B": (
        "Der Druck am linken Herzen ist erhöht. "
        "Das spricht für Rückstau in Richtung Lunge."
    ),
    "K14_E": (
        "Im Vordergrund steht die Behandlung des Herzens (zum Beispiel Entwässerung, Blutdruck, Rhythmus, Klappen – je nach Bedarf)."
    ),

    "K15_B": (
        "Es ist eine Misch‑Situation: "
        "Rückstau vom linken Herzen *und* zusätzlich ein erhöhter Widerstand in den Lungen‑Gefäßen."
    ),
    "K15_E": (
        "Wir empfehlen: Herz gut behandeln *und* die Lungen‑Gefäße weiter abklären. "
        "Das wird im Spezialzentrum entschieden."
    ),

    "K16_B": (
        "Bei den Sauerstoff‑Messungen gibt es einen Hinweis auf einen Shunt (eine Kurzschluss‑Verbindung im Herzen)."
    ),
    "K16_E": (
        "Wir empfehlen weitere Untersuchungen am Herzen (Herzultraschall, ggf. spezieller Ultraschall/TEE), um das genau zu klären."
    ),

    "K17_B": (
        "Der Test auf Gefäß‑Reaktion war *positiv*. Unter der Test‑Gabe wurden die Werte besser."
    ),
    "K17_E": (
        "Wir empfehlen: weiteres Vorgehen im Spezialzentrum planen. Dort wird entschieden, welche Behandlung passt."
    ),

    "K18_B": (
        "Der Test auf Gefäß‑Reaktion war *negativ*. Unter der Test‑Gabe wurden die Werte nicht genug besser."
    ),
    "K18_E": (
        "Wir empfehlen: Behandlung nach Ursache und Risiko planen. Kontrollen bleiben wichtig."
    ),

    "K19_B": (
        "Die Werte sprechen für Lungenhochdruck. "
        "Es gibt Hinweise auf eine seltene Erkrankung der Lungen‑Gefäße (PVOD)."
    ),
    "K19_E": (
        "Wir empfehlen Betreuung im Spezialzentrum. "
        "Manche Medikamente müssen bei PVOD sehr vorsichtig eingesetzt werden."
    ),

    "K20_B": (
        "Wir sollten mögliche Auslöser und Begleiterkrankungen überprüfen."
    ),
    "K20_E": (
        "Wir empfehlen je nach Situation weitere Untersuchungen (Bluttests, Herz‑ und Lungen‑Untersuchungen, ggf. Schlaf‑Untersuchung)."
    ),

    # ----------------
    # Volumenchallenge
    # ----------------
    "B11": (
        "Nach Flüssigkeitsgabe stieg der Druck am linken Herzen deutlich an. "
        "Das kann zu Luftnot führen und spricht eher für eine Herz‑Ursache."
    ),
    "B12": (
        "Nach Flüssigkeitsgabe stieg der Druck am linken Herzen nicht deutlich an. "
        "Das spricht in dieser Untersuchung gegen einen Flüssigkeits‑bedingten Rückstau."
    ),

    # -----------
    # Procedere P
    # -----------
    "P01": (
        "Weitere Untersuchungen können sinnvoll sein:\n"
        "- Herzultraschall.\n"
        "- Lungenfunktion.\n"
        "- Bilder der Lunge (CT/andere Bildgebung).\n"
        "- Wenn nötig: Untersuchung der Lungen‑Durchblutung (V/Q).\n"
        "- Bluttests."
    ),
    "P02": (
        "Wenn Wasser/Rückstau im Körper besteht: Entwässerung (Diuretika) anpassen. "
        "Dabei Nierenwerte und Salze im Blut kontrollieren."
    ),
    "P03": (
        "Es ist eine spezielle Behandlung gegen Lungenhochdruck geplant (Tabletten‑Therapie). "
        "Details besprechen wir im Termin."
    ),
    "P04": (
        "Es ist eine Erweiterung/Start einer weiteren Behandlung gegen Lungenhochdruck geplant. "
        "Details besprechen wir im Termin."
    ),
    "P05": (
        "Es ist eine Behandlung mit Riociguat geplant. "
        "Wichtig: Riociguat darf nicht zusammen mit PDE5‑Hemmern genommen werden."
    ),
    "P06": (
        "Wir sprechen im Spezialzentrum über stärkere Therapie‑Optionen (wenn nötig, z.B. Infusionen)."
    ),
    "P07": (
        "Wir prüfen, ob eine Teilnahme an einer Studie möglich und sinnvoll ist."
    ),
    "P08": (
        "Die CT‑Bilder werden in einer Spezial‑Konferenz besprochen. Danach planen wir das weitere Vorgehen."
    ),
    "P09": (
        "Herz‑Abklärung: Herzultraschall, Herzklappen und Herzrhythmus prüfen (je nach Bedarf)."
    ),
    "P10": (
        "Gerinnungs‑Abklärung: Blutverdünnung prüfen und ggf. anpassen (Gerinnungs‑Sprechstunde)."
    ),
    "P11": (
        "Wir planen einen Kontrolltermin. Dabei prüfen wir Beschwerden, Blutwerte und Herzultraschall (je nach Situation)."
    ),
    "P12": (
        "Lungen‑Abklärung: Lungenfunktion und ggf. weitere Tests (je nach Beschwerden und Vorbefunden)."
    ),
    "P13": (
        "Blutarmut/Eisenmangel prüfen und – wenn nötig – behandeln."
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
