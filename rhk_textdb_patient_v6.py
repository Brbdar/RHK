# -*- coding: utf-8 -*-
"""
Patienten‑Textdatenbank (sehr einfache Sprache)

Version: v6
- längere, zusammenhängende Sätze
- möglichst keine Zahlen / Abkürzungen
- Fokus: Bedeutung & nächste Schritte

Hinweis: Die App nutzt diese Bausteine u. a. für den Patientenbericht
und für patientenfreundliche „Nächste Schritte“.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class SafeDict(dict):
    """Dict, das fehlende Keys als leere Strings zurückgibt (für format_map)."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return ""


@dataclass(frozen=True)
class PatientBlock:
    key: str
    title: str
    template: str

    def render(self, ctx: Dict[str, Any]) -> str:
        ctx = dict(ctx or {})
        ctx.setdefault("title", self.title)
        ctx.setdefault("key", self.key)
        return self.template.format_map(SafeDict(ctx)).strip()


PATIENT_BLOCKS: Dict[str, PatientBlock] = {
    # P‑Module (Empfehlungen / Procedere)
    "P01": PatientBlock(
        key="P01",
        title="Weitere Untersuchungen",
        template=(
            "Wir empfehlen weitere Untersuchungen, damit wir die Ursache Ihrer Beschwerden besser verstehen. "
            "Diese Untersuchungen helfen uns, die passende Behandlung auszuwählen."
        ),
    ),
    "P02": PatientBlock(
        key="P02",
        title="Entwässernde Behandlung",
        template=(
            "Wenn sich Wasser im Körper einlagert (zum Beispiel an den Beinen oder am Bauch), "
            "kann eine entwässernde Behandlung sinnvoll sein. "
            "Dadurch wird das Herz entlastet und die Atmung kann leichter werden."
        ),
    ),
    "P03": PatientBlock(
        key="P03",
        title="Spezielle Behandlung gegen Lungenhochdruck",
        template=(
            "Es gibt Medikamente, die gezielt die Gefäße in der Lunge entlasten können. "
            "Ob diese Behandlung für Sie geeignet ist, hängt von der genauen Ursache und der Gesamtsituation ab."
        ),
    ),
    "P04": PatientBlock(
        key="P04",
        title="Blutverdünnung",
        template=(
            "In manchen Situationen ist eine Blutverdünnung wichtig, zum Beispiel wenn Blutgerinnsel eine Rolle spielen könnten. "
            "Welche Blutverdünnung sinnvoll ist, wird individuell entschieden."
        ),
    ),
    "P05": PatientBlock(
        key="P05",
        title="Sauerstoff",
        template=(
            "Wenn der Sauerstoffgehalt im Blut zu niedrig ist, kann eine Sauerstoff‑Therapie helfen. "
            "Das entlastet Herz und Lunge und kann die Belastbarkeit verbessern."
        ),
    ),
    "P06": PatientBlock(
        key="P06",
        title="Schlafbezogene Atmungsstörung",
        template=(
            "Wenn es Hinweise auf eine Atmungsstörung im Schlaf gibt, sollte das gezielt abgeklärt werden. "
            "Eine passende Behandlung kann Luftnot und Müdigkeit deutlich verbessern."
        ),
    ),
    "P07": PatientBlock(
        key="P07",
        title="Rehabilitation und Training",
        template=(
            "Ein angepasstes Training unter fachlicher Anleitung kann helfen, wieder belastbarer zu werden. "
            "Wichtig ist, dass das Training zu Ihrer Erkrankung passt."
        ),
    ),
    "P08": PatientBlock(
        key="P08",
        title="Kontrolle und Verlauf",
        template=(
            "Wir empfehlen regelmäßige Kontrollen. "
            "So können wir früh erkennen, ob sich etwas verändert und die Behandlung anpassen."
        ),
    ),
    "P09": PatientBlock(
        key="P09",
        title="Mitbeurteilung durch Herz‑ und Lungen‑Fachärzte",
        template=(
            "Je nach Befund ist es sinnvoll, dass sowohl ein Herz‑Facharzt als auch ein Lungen‑Facharzt mitbeurteilen. "
            "So stellen wir sicher, dass keine wichtige Ursache übersehen wird."
        ),
    ),
    "P10": PatientBlock(
        key="P10",
        title="Behandlung von Begleiterkrankungen",
        template=(
            "Begleiterkrankungen wie Bluthochdruck, Herzrhythmus‑Störungen, Lungenerkrankungen oder Übergewicht "
            "können Beschwerden verstärken. "
            "Eine gute Behandlung dieser Faktoren ist oft ein wichtiger Teil der Therapie."
        ),
    ),
    "P11": PatientBlock(
        key="P11",
        title="Erneute Katheter‑Kontrolle",
        template=(
            "Manchmal ist eine erneute Katheter‑Kontrolle sinnvoll, um den Verlauf zu beurteilen oder die Therapie zu überprüfen. "
            "Das wird nur gemacht, wenn es medizinisch notwendig ist."
        ),
    ),

    # Zusatz‑Hinweise / Erklärungen (PLUS_…)
    "PLUS_DIAG_HFPEF": PatientBlock(
        key="PLUS_DIAG_HFPEF",
        title="Hinweis auf Herzschwäche trotz erhaltener Pumpkraft",
        template=(
            "Ein Teil der Beschwerden kann zu einer Form der Herzschwäche passen, bei der das Herz sich schlechter entspannt. "
            "Das kann schneller zu Rückstau führen – vor allem bei Belastung. "
            "Wir empfehlen eine kardiologische Einordnung."
        ),
    ),
    "PLUS_DIAG_ILD": PatientBlock(
        key="PLUS_DIAG_ILD",
        title="Hinweis auf Lungenerkrankung",
        template=(
            "Es gibt Hinweise auf Veränderungen des Lungengewebes. "
            "Das kann die Atmung und den Sauerstoffgehalt beeinflussen und damit auch das Herz belasten. "
            "Wir empfehlen eine genaue Abklärung der Lunge."
        ),
    ),
    "PLUS_DIAG_CTEPH": PatientBlock(
        key="PLUS_DIAG_CTEPH",
        title="Hinweis auf Durchblutungs‑Probleme der Lunge",
        template=(
            "Es gibt Hinweise, dass die Durchblutung der Lunge beeinträchtigt sein könnte, zum Beispiel nach früheren Blutgerinnseln. "
            "Dafür gibt es spezielle Untersuchungen und in manchen Fällen auch spezielle Behandlungen."
        ),
    ),
    "PLUS_TEST_ECHO": PatientBlock(
        key="PLUS_TEST_ECHO",
        title="Herzultraschall",
        template=(
            "Ein Herzultraschall zeigt, wie gut das Herz arbeitet und ob es Hinweise auf Rückstau gibt. "
            "Das ist eine wichtige Ergänzung zur Katheter‑Messung."
        ),
    ),
    "PLUS_TEST_LUFU": PatientBlock(
        key="PLUS_TEST_LUFU",
        title="Lungenfunktion",
        template=(
            "Eine Lungenfunktions‑Messung zeigt, wie gut die Lunge Luft ein‑ und ausatmen kann und wie gut sie Sauerstoff aufnimmt. "
            "Das hilft bei der Suche nach der Ursache."
        ),
    ),
    "PLUS_TEST_CT": PatientBlock(
        key="PLUS_TEST_CT",
        title="Bildgebung der Lunge",
        template=(
            "Eine Bildgebung der Lunge kann zeigen, ob Engstellen, Narben oder andere Veränderungen vorliegen. "
            "Das ist wichtig, um die richtige Ursache zu finden."
        ),
    ),
    "PLUS_TEST_VQ": PatientBlock(
        key="PLUS_TEST_VQ",
        title="Untersuchung der Lungendurchblutung",
        template=(
            "Eine Untersuchung der Lungendurchblutung kann zeigen, ob alle Bereiche der Lunge gut durchblutet sind. "
            "Das ist besonders wichtig, wenn Blutgerinnsel eine Rolle spielen könnten."
        ),
    ),
    "PLUS_TEST_SLEEP": PatientBlock(
        key="PLUS_TEST_SLEEP",
        title="Schlaf‑Untersuchung",
        template=(
            "Eine Untersuchung im Schlaf hilft, Atemaussetzer oder zu flache Atmung zu erkennen. "
            "Das kann die Belastbarkeit und den Schlaf deutlich beeinflussen."
        ),
    ),
    "PLUS_TEST_LAB": PatientBlock(
        key="PLUS_TEST_LAB",
        title="Blutuntersuchungen",
        template=(
            "Blutuntersuchungen geben Hinweise auf Entzündung, Blutarmut, Nieren‑ und Leberfunktion sowie auf eine Belastung des Herzens. "
            "Das hilft, die Situation besser einzuordnen."
        ),
    ),
    "PLUS_TEST_CMR": PatientBlock(
        key="PLUS_TEST_CMR",
        title="Herz‑MRT",
        template=(
            "Ein Herz‑MRT kann sehr genau zeigen, wie das rechte Herz arbeitet und ob es Zeichen einer Überlastung gibt. "
            "Das kann bei der Planung der Behandlung helfen."
        ),
    ),
    "PLUS_TEST_6MWD": PatientBlock(
        key="PLUS_TEST_6MWD",
        title="Gehtest",
        template=(
            "Ein standardisierter Gehtest zeigt, wie belastbar Sie im Alltag sind. "
            "Er hilft auch, Veränderungen im Verlauf zu erkennen."
        ),
    ),
    "PLUS_TEST_CPET": PatientBlock(
        key="PLUS_TEST_CPET",
        title="Belastungstest",
        template=(
            "Ein Belastungstest kann zeigen, ob die Luftnot eher vom Herzen, von der Lunge oder von der allgemeinen Kondition kommt. "
            "Das ist hilfreich, wenn die Ursache nicht eindeutig ist."
        ),
    ),
    "PLUS_THERAPY_OXYGEN": PatientBlock(
        key="PLUS_THERAPY_OXYGEN",
        title="Sauerstoff‑Therapie",
        template=(
            "Wenn die Sauerstoffwerte zu niedrig sind, kann Sauerstoff die Organe schützen und Beschwerden lindern. "
            "Ob und wie lange Sauerstoff nötig ist, wird individuell geprüft."
        ),
    ),
    "PLUS_THERAPY_DIURETICS": PatientBlock(
        key="PLUS_THERAPY_DIURETICS",
        title="Entwässern",
        template=(
            "Entwässernde Medikamente können helfen, wenn der Körper Wasser einlagert. "
            "Bitte nehmen Sie diese nur wie verordnet ein."
        ),
    ),
    "PLUS_THERAPY_REHAB": PatientBlock(
        key="PLUS_THERAPY_REHAB",
        title="Rehabilitation",
        template=(
            "Eine Rehabilitation oder ein strukturiertes Trainingsprogramm kann die Belastbarkeit verbessern "
            "und Sicherheit im Umgang mit der Erkrankung geben."
        ),
    ),
    "PLUS_FOLLOWUP": PatientBlock(
        key="PLUS_FOLLOWUP",
        title="Nachkontrolle",
        template=(
            "Wir empfehlen eine Nachkontrolle, um zu prüfen, ob die Behandlung wirkt und ob sich Ihr Zustand verändert."
        ),
    ),
}


def get_patient_block(block_id: str) -> Optional[PatientBlock]:
    return PATIENT_BLOCKS.get((block_id or "").strip())
