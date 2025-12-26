# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def to_float(x: Any) -> Optional[float]:
    """Best-effort conversion. Returns None for empty/invalid."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def to_int(x: Any) -> Optional[int]:
    f = to_float(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def is_intish(v: float, tol: float = 1e-6) -> bool:
    return abs(v - round(v)) < tol


def fmt_num(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "nicht erhoben"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "nicht erhoben"
    if is_intish(v):
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


def fmt_unit(v: Optional[float], unit: str, decimals: int = 1) -> str:
    if v is None:
        return "nicht erhoben"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "nicht erhoben"
    return f"{fmt_num(v, decimals)} {unit}"


def join_nonempty(parts: Sequence[str], sep: str = " | ") -> str:
    return sep.join([p for p in parts if p and str(p).strip()])


def parse_date_yyyy_mm_dd(s: Any) -> Optional[date]:
    if s is None:
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    txt = str(s).strip()
    if not txt:
        return None
    try:
        return datetime.strptime(txt, "%Y-%m-%d").date()
    except Exception:
        return None


def calc_age_years(dob: Optional[date], ref: Optional[date] = None) -> Optional[int]:
    if dob is None:
        return None
    ref = ref or date.today()
    years = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    return max(0, years)


class SafeDict(dict):
    """Format-map helper that never raises KeyError."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "nicht erhoben"


def clean_sentence(text: str) -> str:
    """Tidies spaces/punctuation for concatenated sentences."""
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    t = re.sub(r"\(\s*\)", "", t)
    return t.strip()


def clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class ValidationReport:
    missing: List[str]
    warnings: List[str]

    def to_markdown(self) -> str:
        lines: List[str] = []
        if self.missing:
            lines.append("### Fehlende Schlüsselwerte")
            for m in self.missing:
                lines.append(f"- {m}")
        if self.warnings:
            lines.append("### Plausibilitäts-/Hinweis-Checks")
            for w in self.warnings:
                lines.append(f"- {w}")
        if not lines:
            return "—"
        return "\n".join(lines)
