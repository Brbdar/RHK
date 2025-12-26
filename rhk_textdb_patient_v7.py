# -*- coding: utf-8 -*-
"""
Patienten-Textbausteine (Einfache Sprache, ausführlicher).

Hinweis:
- Keine Zahlenwerte / Abkürzungen, damit Patient:innen den Text gut verstehen.
- Die Texte sind als Ergänzung gedacht und ersetzen keine ärztliche Aufklärung.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PatientBlock:
    key: str
    title: str
    text: str


PATIENT_BLOCKS: Dict[str, Dict[str, str]] = {
    "P01": {
        "title": "Weitere Abklärung (Basisdiagnostik)",
        "text": (
            "Als nächstes ist wichtig, die Ursache Ihrer Beschwerden möglichst genau einzugrenzen. "
            "Dazu werden – je nach Situation – zusätzliche Untersuchungen von Herz und Lunge geplant. "
            "Das kann zum Beispiel eine detaillierte Bildgebung, eine genaue Lungenfunktionsprüfung, "
            "spezielle Blutuntersuchungen oder eine Abklärung von Blutgerinnseln in der Lunge umfassen. "
            "Das Ziel ist, die Behandlung später wirklich passend auszuwählen."
        ),
    },
    "P02": {
        "title": "Entwässerung (bei Wasseransammlungen)",
        "text": (
            "Wenn sich im Körper Wasser ansammelt, kann das das Herz zusätzlich belasten und die Luftnot verstärken. "
            "Entwässernde Medikamente helfen, überschüssige Flüssigkeit auszuschwemmen. "
            "Wichtig sind dabei regelmäßige Gewichtskontrollen, eine sinnvolle Trinkmengen‑Abstimmung "
            "und Blutkontrollen, damit Salz- und Nierenwerte stabil bleiben. "
            "Bitte melden Sie sich frühzeitig, wenn das Gewicht schnell ansteigt oder Beine/Bauch deutlich anschwellen."
        ),
    },
    "P03": {
        "title": "Medikament zur Entspannung der Lungengefäße",
        "text": (
            "Es gibt Medikamente, die die Blutgefäße in der Lunge entspannen und dadurch die Durchblutung verbessern können. "
            "Ziel ist, dass das rechte Herz weniger gegen Widerstand arbeiten muss und Sie im Alltag besser belastbar sind. "
            "Solche Medikamente werden meist langsam eingeschlichen und die Wirkung wird in Verlaufskontrollen überprüft. "
            "Wenn Nebenwirkungen auftreten (zum Beispiel Kopfschmerzen, Schwindel oder Blutdruckabfall), "
            "sollten Sie sich bitte melden."
        ),
    },
    "P04": {
        "title": "Medikament mit Wirkung auf Botenstoffe der Gefäße",
        "text": (
            "Es gibt Medikamente, die bestimmte Botenstoffe im Körper beeinflussen, die die Lungengefäße verengen können. "
            "Dadurch kann sich der Druck im Lungenkreislauf langfristig verbessern. "
            "Vor und während der Behandlung sind regelmäßige Kontrollen wichtig, "
            "zum Beispiel Blutwerte und – je nach Präparat – weitere Sicherheitskontrollen. "
            "Ihr Behandlungsteam erklärt Ihnen genau, worauf Sie achten sollen."
        ),
    },
    "P05": {
        "title": "Alternative/Wechsel‑Therapie zur Verbesserung des Lungenkreislaufs",
        "text": (
            "In manchen Situationen ist ein anderes Medikament sinnvoll, das die Durchblutung im Lungenkreislauf verbessert "
            "und so das rechte Herz entlasten kann. "
            "Ob ein Wechsel oder eine Ergänzung passt, hängt von der vermuteten Ursache, der Verträglichkeit "
            "und dem bisherigen Verlauf ab. "
            "Auch hier sind regelmäßige Kontrollen wichtig, damit Nutzen und Sicherheit gut eingeschätzt werden können."
        ),
    },
    "P06": {
        "title": "Intensivere Therapieoptionen (bei ausgeprägter Belastung)",
        "text": (
            "Wenn die Erkrankung stärker ausgeprägt ist oder sich trotz Behandlung nicht ausreichend bessert, "
            "gibt es intensivere Therapieoptionen. "
            "Ein Teil dieser Behandlungen wirkt sehr stark und wird manchmal als Dauertherapie über eine Pumpe gegeben. "
            "Das klingt aufwendig, kann aber in passenden Fällen die Beschwerden deutlich verbessern. "
            "Ob das für Sie infrage kommt, wird in Ruhe besprochen und gemeinsam entschieden."
        ),
    },
    "P07": {
        "title": "Studien / neue Behandlungsmöglichkeiten",
        "text": (
            "Manchmal kann die Teilnahme an einer Studie sinnvoll sein – zum Beispiel, wenn Standardbehandlungen nicht ausreichen "
            "oder wenn neue Therapieansätze geprüft werden. "
            "Eine Studienteilnahme ist immer freiwillig. "
            "Sie bekommen ausführliche Informationen über Ziele, mögliche Vorteile und Risiken, "
            "und Sie können jederzeit ohne Nachteile wieder aussteigen."
        ),
    },
    "P08": {
        "title": "Besprechung im Lungen‑Spezialteam",
        "text": (
            "Wenn Hinweise auf Veränderungen des Lungengewebes bestehen, ist oft eine Besprechung im spezialisierten Team sinnvoll. "
            "Dabei werden Bildgebung, Lungenfunktion und Ihre Beschwerden gemeinsam bewertet. "
            "So lässt sich besser entscheiden, ob eine gezielte Behandlung der Lunge nötig ist "
            "und wie das mit der Behandlung des Lungenkreislaufs zusammenspielt."
        ),
    },
    "P09": {
        "title": "Mitbeurteilung durch Herz‑Spezialist:innen",
        "text": (
            "Da Herz und Lunge eng zusammenarbeiten, ist manchmal eine zusätzliche kardiologische Mitbeurteilung sinnvoll. "
            "Dabei wird zum Beispiel geprüft, ob Herzklappen, Herzrhythmus oder eine Durchblutungsstörung des Herzens "
            "zu den Beschwerden beitragen könnten. "
            "Wenn man solche Faktoren erkennt und behandelt, kann sich die Belastbarkeit deutlich verbessern."
        ),
    },
    "P10": {
        "title": "Blutgerinnung / Blutverdünnung",
        "text": (
            "Wenn der Verdacht besteht, dass Blutgerinnsel in der Lunge eine Rolle spielen, kann eine Blutverdünnung nötig sein. "
            "Welche Art der Blutverdünnung passend ist, hängt von Ihrer Situation und möglichen Blutungsrisiken ab. "
            "Oft ist eine Mitbetreuung durch eine Gerinnungs‑Sprechstunde hilfreich, "
            "damit die Behandlung sicher eingestellt und regelmäßig kontrolliert wird."
        ),
    },
    "P11": {
        "title": "Verlaufskontrolle",
        "text": (
            "Zur sicheren Beurteilung braucht es Verlaufskontrollen. "
            "Dabei wird geschaut, wie es Ihnen klinisch geht, wie belastbar Sie sind und ob sich Blutwerte oder Herz‑/Lungenwerte verändern. "
            "Je nach Verlauf kann die Therapie beibehalten, angepasst oder erweitert werden. "
            "Bitte kommen Sie zu den vereinbarten Terminen, auch wenn es Ihnen zwischenzeitlich besser geht."
        ),
    },
    "P12": {
        "title": "Genauere Lungenfunktions‑Abklärung",
        "text": (
            "Eine genaue Lungenfunktions‑Untersuchung hilft zu verstehen, wie gut Luft in die Lunge hinein‑ und herausströmt "
            "und wie gut der Sauerstoff aus der Lunge in das Blut übergeht. "
            "Das ist wichtig, weil Lungenerkrankungen Luftnot verursachen und gleichzeitig den Lungenkreislauf belasten können. "
            "Mit den Ergebnissen kann die Behandlung besser angepasst werden."
        ),
    },
    "P13": {
        "title": "Eisenmangel / Blutarmut",
        "text": (
            "Ein Eisenmangel oder eine Blutarmut kann Müdigkeit, Schwäche und Luftnot verstärken – unabhängig vom Lungenkreislauf. "
            "Darum werden entsprechende Blutwerte häufig gezielt geprüft. "
            "Wenn ein Mangel vorliegt, kann eine Behandlung (zum Beispiel Eisengabe) die Leistungsfähigkeit verbessern. "
            "Ihr Team bespricht mit Ihnen, welche Form der Therapie sinnvoll ist."
        ),
    },
}
