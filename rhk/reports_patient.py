# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

import rhk_textdb as textdb

from .generator import Computed
from .util import clean_sentence


def _load_patient_db():
    # Try canonical name first, then fallbacks
    for mod in ("rhk_textdb_patient_v7", "rhk_textdb_patient", "rhk_textdb_patient_v6", "rhk_textdb_patient_v5", "rhk_textdb_patient_v4"):
        try:
            return __import__(mod, fromlist=["*"])
        except Exception:
            continue
    return None


_PAT_DB = _load_patient_db()


def _risk_text(comp: Computed) -> str:
    # Prefer esc4 (follow-up) if available, else esc3
    cat = comp.esc4.category if comp.esc4 and comp.esc4.category != "unknown" else comp.esc3.category
    mapping = {
        "low": "eher niedrig",
        "intermediate-low": "eher niedrig bis mittel",
        "intermediate": "mittel",
        "intermediate-high": "mittel bis erhöht",
        "high": "erhöht",
        "unknown": "nicht sicher einzuordnen",
    }
    return mapping.get(cat, "nicht sicher einzuordnen")


def _ph_simple(comp: Computed) -> str:
    if comp.ph.ph_type == "none":
        return "Es gibt aktuell keinen Hinweis auf einen erhöhten Druck in den Lungengefäßen in Ruhe."
    if comp.ph.ph_type == "precap":
        return "Die Messwerte passen zu einem erhöhten Druck in den Lungengefäßen, der eher von den Lungengefäßen selbst ausgeht."
    if comp.ph.ph_type == "ipcph":
        return "Die Messwerte passen zu einem erhöhten Druck in den Lungengefäßen, der eher mit dem linken Herzen zusammenhängt."
    if comp.ph.ph_type == "cpcph":
        return "Die Messwerte sprechen für eine Mischung aus einem Problem des linken Herzens und einer zusätzlichen Belastung der Lungengefäße."
    return "Die Messwerte zeigen eine Druckerhöhung, die aber nicht eindeutig einzuordnen ist."


def _severity_simple(comp: Computed) -> Optional[str]:
    sev = None
    if comp.pvr_sev == "mild":
        sev = "leicht"
    elif comp.pvr_sev == "moderate":
        sev = "mittelgradig"
    elif comp.pvr_sev == "severe":
        sev = "ausgeprägt"
    # If CI strongly reduced, upgrade wording
    if comp.ci_sev == "severely_reduced":
        sev = "ausgeprägt"
    return sev


def _exercise_simple(comp: Computed) -> Optional[str]:
    if not comp.exercise_done:
        return None
    if comp.exercise_pattern == "normal":
        return "Auch unter Belastung zeigte sich keine auffällige Druck‑Reaktion."
    if comp.exercise_pattern == "left_heart":
        return "Unter Belastung stiegen die Drücke so an, dass eher eine Belastung durch das linke Herz wahrscheinlich ist."
    if comp.exercise_pattern == "pulmonary_vascular":
        return "Unter Belastung zeigte sich eine auffällige Reaktion der Lungengefäße."
    return "Unter Belastung zeigte sich eine auffällige Reaktion, die sich nicht eindeutig zuordnen lässt."


def _etiology_simple(comp: Computed) -> List[str]:
    if not comp.etiology_hints:
        return []
    parts: List[str] = []
    if comp.etiology_hints.group3_possible:
        parts.append("Es gibt Hinweise, dass eine Lungenerkrankung oder eine zu niedrige Sauerstoffversorgung eine Rolle spielen könnte.")
    if comp.etiology_hints.group4_possible:
        parts.append("Es gibt Hinweise, dass frühere oder kleine Blutgerinnsel in der Lunge eine Rolle spielen könnten.")
    if comp.etiology_hints.group2_possible and comp.ph.ph_type != "ipcph":
        parts.append("Ein Anteil durch das linke Herz ist möglich und sollte mitgeprüft werden.")
    if comp.etiology_hints.porto_possible:
        parts.append("Im Bauch/Leber‑Bereich gibt es Hinweise, die mit dem Lungenkreislauf zusammenhängen können.")
    if comp.etiology_hints.shunt_possible:
        parts.append("Die Sauerstoffmessungen sprechen für eine mögliche Kurzschlussverbindung im Kreislauf. Das sollte weiter abgeklärt werden.")
    return parts


def _patient_modules(ui: Dict[str, Any]) -> List[str]:
    # Map selected doctor modules (Pxx etc.) to patient blocks if present
    modules: List[str] = ui.get("modules") or []
    suggestions: List[str] = []
    # If main bundle suggests modules, include those too
    # (We don't know chosen bundle here; keep to user selection)
    out: List[str] = []
    for mid in modules:
        if not mid:
            continue
        out.append(mid)
    return out


def _patient_module_text(module_id: str) -> Optional[str]:
    if _PAT_DB is None:
        return None
    # Patient DB in v6 uses keys like P01, P02 etc inside PATIENT_BLOCKS
    blocks = getattr(_PAT_DB, "PATIENT_BLOCKS", None)
    if not isinstance(blocks, dict):
        return None
    b = blocks.get(module_id)
    if not b:
        return None
    # b might be dict with "text"
    if isinstance(b, dict):
        return b.get("text")
    return None


def build_patient_report(ui: Dict[str, Any], comp: Computed) -> str:
    """
    Patient-friendly report:
    - no numbers, no abbreviations
    - coherent paragraphs (not chopped sentences)
    """
    paragraphs: List[str] = []

    # Intro / what was done
    paragraphs.append(
        "Bei Ihnen wurde eine Untersuchung des rechten Herzens und des Lungenkreislaufs durchgeführt. "
        "Dabei werden Drücke und die Pumpleistung des Herzens gemessen. Das hilft, Ursachen von Luftnot besser einzuordnen."
    )

    # Main finding
    main = _ph_simple(comp)
    sev = _severity_simple(comp)
    if sev and comp.ph.ph_type in ("precap", "ipcph", "cpcph"):
        main = main + f" Insgesamt wirkt die Ausprägung eher {sev}."
    paragraphs.append(main)

    # Exercise
    ex = _exercise_simple(comp)
    if ex:
        paragraphs.append(ex)

    # Step-up / shunt
    if comp.step_up_present:
        paragraphs.append(
            "In einer zusätzlichen Sauerstoff‑Messreihe gab es Hinweise auf eine mögliche Kurzschlussverbindung im Kreislauf. "
            "Das bedeutet: Blut könnte an einer Stelle den normalen Weg teilweise abkürzen. Das sollte gezielt weiter untersucht werden."
        )

    # Meaning / risk
    risk = _risk_text(comp)
    paragraphs.append(
        "Aus den verfügbaren Angaben ergibt sich aktuell eine Einschätzung des Verlauf‑Risikos als "
        f"**{risk}**. Diese Einschätzung ist eine Orientierung und wird immer zusammen mit Ihren Beschwerden "
        "und weiteren Untersuchungen bewertet."
    )

    # Etiology hints
    eti_parts = _etiology_simple(comp)
    if eti_parts:
        paragraphs.append(" ".join(eti_parts))

    # HFpEF hint (avoid abbreviation)
    if comp.hfpef is not None and comp.hfpef.category in ("possible", "likely"):
        paragraphs.append(
            "Zusätzlich ergibt ein Punktesystem Hinweise darauf, dass die Entspannungs‑ und Füllungsphase des linken Herzens "
            "mitbeteiligt sein könnte. Das nennt man auch eine diastolische Funktionsstörung. "
            "Dazu passt häufig, dass die Beschwerden vor allem unter Belastung auftreten."
        )

    # Next steps / recommendations
    rec: List[str] = []
    rec.append(
        "Wichtig ist jetzt, die wahrscheinlichste Ursache weiter einzugrenzen und die Behandlung darauf auszurichten. "
        "Dazu gehören je nach Situation Herz‑ und Lungenuntersuchungen, Blutwerte und Bildgebung."
    )
    # Add patient module texts
    for mid in _patient_modules(ui):
        txt = _patient_module_text(mid)
        if txt:
            rec.append(txt)

    paragraphs.extend(rec)

    # Self-management
    paragraphs.append(
        "Wenn bei Ihnen Wasseransammlungen, schnelle Gewichtszunahme oder deutlich geschwollene Beine auftreten, "
        "ist ein konsequentes Volumen‑Management wichtig. Sprechen Sie in diesem Fall frühzeitig Ihr Behandlungsteam an."
    )

    paragraphs.append(
        "Bitte melden Sie sich zeitnah, wenn sich die Luftnot deutlich verschlechtert, wenn neue Brustschmerzen auftreten, "
        "bei Ohnmacht/Beinahe‑Ohnmacht oder wenn Sie sich insgesamt deutlich instabil fühlen."
    )

    # Join nicely
    text = "\n\n".join([clean_sentence(p) for p in paragraphs if p and p.strip()])
    return text
