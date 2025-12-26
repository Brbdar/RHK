
# rhk_app_web_master_v18.py
# -*- coding: utf-8 -*-
"""
RHK-Befundassistent v18 – Web GUI (Gradio) mit deklarativer Rule-Engine (YAML).

Highlights
- Regelwerk als YAML (rhk_rules_v18.yaml): Bedingungen -> Aktionen (Bundle/Module/Empfehlungen/required fields).
- Ultra-interaktiv: Jede Eingabe hat Konsequenzen (Auto-Recompute & Re-Render).
- Random-Beispiel: "Beispiel laden" erzeugt bei jedem Klick einen neuen, plausiblen Fall.
- Fall speichern/laden (JSON) ohne Dropdown: Download + UploadButton.

Start lokal:
    python rhk_app_web_master_v18.py

Start in Jupyter:
    import os
    os.environ["PORT"] = "7861"
    import rhk_app_web_master_v18 as app
    app.main()

Hinweis
- Dieses Tool richtet sich an medizinisch geschultes Personal.
- Empfehlungen sind kontextabhängig und ersetzen keine klinische Entscheidung / Leitlinienlektüre.
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_VERSION = "18.0.0"
APP_TITLE = f"RHK Befundassistent v{APP_VERSION}"
UI_SCHEMA = "rhk_case_ui_v18"
DEFAULT_PORT = 7860

# Robust base dir (Render/Jupyter)
try:
    _APP_DIR = Path(__file__).resolve().parent
except NameError:
    _APP_DIR = Path.cwd()

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# --- Textbausteine ---
try:
    import rhk_textdb as textdb
except Exception as e:
    raise ImportError(
        "Konnte rhk_textdb.py nicht importieren. Stelle sicher, dass rhk_textdb.py im selben Ordner liegt.\n"
        f"Originalfehler: {e}"
    )

# --- YAML loader ---
def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML ist nicht installiert. Bitte in requirements aufnehmen: pyyaml\n"
            f"Originalfehler: {e}"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

# --- Gradio ---
try:
    import gradio as gr
except Exception as e:
    raise RuntimeError(
        "Gradio ist nicht installiert. Installiere es z.B. mit: pip install gradio\n"
        f"Originalfehler: {e}"
    )

# -----------------------------
# Helpers: parsing/formatting
# -----------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return None
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _is_intish(v: float, tol: float = 1e-6) -> bool:
    return abs(v - round(v)) < tol

def fmt_num(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if _is_intish(v):
        return str(int(round(v)))
    return f"{v:.{decimals}f}"

def fmt_unit(v: Optional[float], unit: str, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{fmt_num(v, decimals)} {unit}"

def parse_date(s: Any) -> Optional[date]:
    if s is None:
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    txt = str(s).strip()
    if not txt:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(txt, fmt).date()
        except Exception:
            pass
    # allow "03/21" -> 2021-03-01
    try:
        if "/" in txt and len(txt) in (5, 7):
            mm, yy = txt.split("/", 1)
            mm = int(mm)
            yy = int(yy)
            if yy < 100:
                yy += 2000 if yy < 50 else 1900
            return date(yy, mm, 1)
    except Exception:
        pass
    return None

def calc_age_years(dob: Optional[date], ref: Optional[date]) -> Optional[int]:
    if dob is None or ref is None:
        return None
    years = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    return max(0, years)

class SafeDict(dict):
    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "—"

def join_nonempty(parts: List[str], sep: str = " | ") -> str:
    return sep.join([p for p in parts if p and str(p).strip()])

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# -----------------------------
# Hemodynamics calculations
# -----------------------------
def calc_mean(sys_: Optional[float], dia: Optional[float]) -> Optional[float]:
    if sys_ is None or dia is None:
        return None
    return (sys_ + 2.0 * dia) / 3.0

def calc_bsa_dubois(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if height_cm is None or weight_kg is None or height_cm <= 0 or weight_kg <= 0:
        return None
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)

def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if height_cm is None or weight_kg is None or height_cm <= 0 or weight_kg <= 0:
        return None
    return weight_kg / ((height_cm / 100.0) ** 2)

def calc_ci(co: Optional[float], bsa: Optional[float]) -> Optional[float]:
    if co is None or bsa is None or bsa <= 0:
        return None
    return co / bsa

def calc_pvr(mpap: Optional[float], pawp: Optional[float], co: Optional[float]) -> Optional[float]:
    if mpap is None or pawp is None or co is None or co == 0:
        return None
    return (mpap - pawp) / co

def calc_tpg(mpap: Optional[float], pawp: Optional[float]) -> Optional[float]:
    if mpap is None or pawp is None:
        return None
    return mpap - pawp

def calc_dpg(dpap: Optional[float], pawp: Optional[float]) -> Optional[float]:
    if dpap is None or pawp is None:
        return None
    return dpap - pawp

def calc_slope(p_rest: Optional[float], co_rest: Optional[float], p_peak: Optional[float], co_peak: Optional[float]) -> Optional[float]:
    if p_rest is None or co_rest is None or p_peak is None or co_peak is None:
        return None
    dco = co_peak - co_rest
    if dco == 0:
        return None
    return (p_peak - p_rest) / dco

# -----------------------------
# Scores (pragmatic)
# -----------------------------
def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))

def esc3_grade_who_fc(who_fc: Optional[str]) -> Optional[int]:
    if not who_fc:
        return None
    s = str(who_fc).strip().upper().replace("WHO", "").replace("FC", "").strip()
    if s in ("I", "1", "II", "2"):
        return 1
    if s in ("III", "3"):
        return 2
    if s in ("IV", "4"):
        return 3
    return None

def esc3_grade_6mwd(m: Optional[float]) -> Optional[int]:
    if m is None:
        return None
    if m > 440:
        return 1
    if m >= 165:
        return 2
    return 3

def esc3_grade_bnp(value: Optional[float], kind: str) -> Optional[int]:
    if value is None:
        return None
    k = (kind or "").strip().lower()
    if "nt" in k:
        if value < 300:
            return 1
        if value <= 1100:
            return 2
        return 3
    if value < 50:
        return 1
    if value <= 800:
        return 2
    return 3

def esc4_grade_who_fc(who_fc: Optional[str]) -> Optional[int]:
    if not who_fc:
        return None
    s = str(who_fc).strip().upper().replace("WHO", "").replace("FC", "").strip()
    if s in ("I", "1", "II", "2"):
        return 1
    if s in ("III", "3"):
        return 3
    if s in ("IV", "4"):
        return 4
    return None

def esc4_grade_6mwd(m: Optional[float]) -> Optional[int]:
    if m is None:
        return None
    if m > 440:
        return 1
    if m >= 320:
        return 2
    if m >= 165:
        return 3
    return 4

def esc4_grade_bnp(value: Optional[float], kind: str) -> Optional[int]:
    if value is None:
        return None
    k = (kind or "").strip().lower()
    if "nt" in k:
        if value < 300:
            return 1
        if value <= 649:
            return 2
        if value <= 1100:
            return 3
        return 4
    if value < 50:
        return 1
    if value <= 199:
        return 2
    if value <= 800:
        return 3
    return 4

def aggregate_grade(grades: List[int], max_grade: int) -> Tuple[Optional[int], Optional[float]]:
    if not grades:
        return None, None
    mean = sum(grades) / len(grades)
    g = _round_half_up(mean)
    g = max(1, min(max_grade, g))
    return g, mean

def esc3_overall(who_fc: Optional[str], sixmwd_m: Optional[float], bnp_kind: str, bnp_value: Optional[float]) -> Dict[str, Any]:
    g_fc = esc3_grade_who_fc(who_fc)
    g_6 = esc3_grade_6mwd(sixmwd_m)
    g_b = esc3_grade_bnp(bnp_value, bnp_kind)
    grades = [g for g in [g_fc, g_6, g_b] if isinstance(g, int)]
    overall, mean = aggregate_grade(grades, 3)
    cat = {1: "low", 2: "intermediate", 3: "high"}.get(overall)
    return {"overall": overall, "mean": mean, "category": cat, "grades": {"WHO_FC": g_fc, "6MWD": g_6, "BNP": g_b}}

def esc4_overall(who_fc: Optional[str], sixmwd_m: Optional[float], bnp_kind: str, bnp_value: Optional[float]) -> Dict[str, Any]:
    g_fc = esc4_grade_who_fc(who_fc)
    g_6 = esc4_grade_6mwd(sixmwd_m)
    g_b = esc4_grade_bnp(bnp_value, bnp_kind)
    grades = [g for g in [g_fc, g_6, g_b] if isinstance(g, int)]
    overall, mean = aggregate_grade(grades, 4)
    cat = {1: "low", 2: "intermediate-low", 3: "intermediate-high", 4: "high"}.get(overall)
    return {"overall": overall, "mean": mean, "category": cat, "grades": {"WHO_FC": g_fc, "6MWD": g_6, "BNP": g_b}}

def reveal_lite2_score(
    who_fc: Optional[str],
    sixmwd_m: Optional[float],
    bnp_kind: str,
    bnp_value: Optional[float],
    sbp_mmHg: Optional[float],
    hr_min: Optional[float],
    egfr_ml_min_1_73: Optional[float],
) -> Dict[str, Any]:
    pts: Dict[str, Optional[int]] = {"WHO_FC": None, "6MWD": None, "BNP": None, "SBP": None, "HR": None, "Renal": None}
    if who_fc:
        s = str(who_fc).strip().upper()
        pts["WHO_FC"] = {"I": -1, "II": 0, "III": 1, "IV": 2}.get(s)
    if sixmwd_m is not None:
        if sixmwd_m >= 440: pts["6MWD"] = -2
        elif sixmwd_m >= 320: pts["6MWD"] = -1
        elif sixmwd_m >= 165: pts["6MWD"] = 0
        else: pts["6MWD"] = 1
    if bnp_value is not None:
        k = (bnp_kind or "").strip().lower()
        if "nt" in k:
            if bnp_value < 300: pts["BNP"] = -2
            elif bnp_value < 1100: pts["BNP"] = 0
            else: pts["BNP"] = 2
        else:
            if bnp_value < 50: pts["BNP"] = -2
            elif bnp_value < 200: pts["BNP"] = 0
            elif bnp_value < 800: pts["BNP"] = 1
            else: pts["BNP"] = 2
    if sbp_mmHg is not None:
        pts["SBP"] = 0 if sbp_mmHg >= 110 else 1
    if hr_min is not None:
        pts["HR"] = 0 if hr_min <= 96 else 1
    if egfr_ml_min_1_73 is not None:
        pts["Renal"] = 0 if egfr_ml_min_1_73 >= 60 else 1

    available = [v for v in pts.values() if isinstance(v, int)]
    if len(available) < 3:
        return {"score": None, "risk": None, "points": pts, "note": "Zu wenige Parameter für REVEAL Lite 2."}
    score = sum(int(v) for v in available) + 6
    risk = "low" if score <= 6 else ("intermediate" if score <= 8 else "high")
    return {"score": score, "risk": risk, "points": pts, "note": None}

# -----------------------------
# HFpEF probability model (continuous) – per user formula
# -----------------------------
@dataclass
class HFpEFProbResult:
    prob: Optional[float]
    percent: Optional[float]
    y: Optional[float]
    note: Optional[str] = None

def hfpef_probability_model(age: Optional[int], bmi: Optional[float], e_over_eprime: Optional[float], pasp: Optional[float], af: Optional[bool]) -> HFpEFProbResult:
    if age is None or bmi is None or e_over_eprime is None or pasp is None or af is None:
        return HFpEFProbResult(prob=None, percent=None, y=None, note="Zu wenige Parameter für HFpEF-Wahrscheinlichkeitsmodell.")
    bmi_c = min(float(bmi), 50.0)
    y = -9.1917 + 0.0451 * float(age) + 0.1307 * bmi_c + 0.0859 * float(e_over_eprime) + 0.0520 * float(pasp) + 1.6997 * (1.0 if af else 0.0)
    z = math.exp(y)
    p = z / (1.0 + z)
    return HFpEFProbResult(prob=p, percent=100.0 * p, y=y, note=("BMI wurde auf 50 kg/m² begrenzt." if bmi is not None and bmi > 50 else None))

# -----------------------------
# Rule engine (safe expression eval)
# -----------------------------
_ALLOWED_CALLS = {"is_none", "not_none", "present", "missing", "contains"}

def is_none(x: Any) -> bool: return x is None
def not_none(x: Any) -> bool: return x is not None
def present(x: Any) -> bool:
    if x is None: return False
    if isinstance(x, str): return bool(x.strip())
    return True
def missing(x: Any) -> bool: return not present(x)
def contains(container: Any, item: Any) -> bool:
    try: return item in container
    except Exception: return False

class SafeExpr(ast.NodeVisitor):
    def __init__(self, env: Dict[str, Any]):
        self.env = env
    def visit(self, node):  # type: ignore[override]
        allowed = (
            ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp,
            ast.Compare, ast.Name, ast.Load, ast.Constant,
            ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.In, ast.NotIn, ast.Is, ast.IsNot,
            ast.Call, ast.Tuple, ast.List,
            ast.Add, ast.Sub, ast.Mult, ast.Div,
        )
        if not isinstance(node, allowed):
            raise ValueError(f"Unsafe node: {type(node).__name__}")
        return super().visit(node)
    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)
    def visit_Name(self, node: ast.Name):
        return self.env.get(node.id)
    def visit_Constant(self, node: ast.Constant):
        return node.value
    def visit_List(self, node: ast.List):
        return [self.visit(e) for e in node.elts]
    def visit_Tuple(self, node: ast.Tuple):
        return tuple(self.visit(e) for e in node.elts)
    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not bool(self.visit(node.operand))
        raise ValueError("Only 'not' is allowed.")
    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not bool(self.visit(v)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(self.visit(v)):
                    return True
            return False
        raise ValueError("Unsupported BoolOp")
    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add): return (left or 0) + (right or 0)
        if isinstance(node.op, ast.Sub): return (left or 0) - (right or 0)
        if isinstance(node.op, ast.Mult): return (left or 0) * (right or 0)
        if isinstance(node.op, ast.Div): return (left or 0) / (right or 1)
        raise ValueError("Unsupported BinOp")
    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            if isinstance(op, ast.Eq): ok = (left == right)
            elif isinstance(op, ast.NotEq): ok = (left != right)
            elif isinstance(op, ast.Lt): ok = (left is not None and right is not None and left < right)
            elif isinstance(op, ast.LtE): ok = (left is not None and right is not None and left <= right)
            elif isinstance(op, ast.Gt): ok = (left is not None and right is not None and left > right)
            elif isinstance(op, ast.GtE): ok = (left is not None and right is not None and left >= right)
            elif isinstance(op, ast.In):
                try: ok = left in right
                except Exception: ok = False
            elif isinstance(op, ast.NotIn):
                try: ok = left not in right
                except Exception: ok = False
            elif isinstance(op, ast.Is): ok = (left is right)
            elif isinstance(op, ast.IsNot): ok = (left is not right)
            else: raise ValueError("Unsupported comparison")
            if not ok: return False
            left = right
        return True
    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        fname = node.func.id
        if fname not in _ALLOWED_CALLS:
            raise ValueError(f"Call not allowed: {fname}")
        fn = {"is_none": is_none, "not_none": not_none, "present": present, "missing": missing, "contains": contains}[fname]
        args = [self.visit(a) for a in node.args]
        return fn(*args)

def safe_eval_bool(expr: str, env: Dict[str, Any]) -> bool:
    expr = (expr or "").strip()
    if not expr:
        return False
    tree = ast.parse(expr, mode="eval")
    return bool(SafeExpr(env).visit(tree))

@dataclass
class Decision:
    bundle: Optional[str] = None
    primary_dx: Optional[str] = None
    modules: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    require_fields: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    fired_rules: List[str] = field(default_factory=list)

def apply_rule_engine(rulebook: Dict[str, Any], env: Dict[str, Any]) -> Decision:
    rules = rulebook.get("rules") if isinstance(rulebook, dict) else None
    if not isinstance(rules, list):
        return Decision()

    def _prio(r: Dict[str, Any]) -> int:
        try: return int(r.get("priority", 0))
        except Exception: return 0

    rules_sorted = sorted([r for r in rules if isinstance(r, dict)], key=_prio, reverse=True)

    d = Decision()
    for r in rules_sorted:
        rid = str(r.get("id") or "").strip() or "RULE"
        cond = str(r.get("when") or "").strip()
        try:
            ok = safe_eval_bool(cond, env)
        except Exception:
            ok = False
        if not ok:
            continue

        d.fired_rules.append(rid)
        actions = r.get("actions") or {}
        if not isinstance(actions, dict):
            actions = {}

        if actions.get("set_bundle"):
            d.bundle = str(actions["set_bundle"]).strip()
        if actions.get("set_primary_dx"):
            d.primary_dx = str(actions["set_primary_dx"]).strip()

        for key, target in (("add_modules", d.modules), ("add_recommendations", d.recommendations), ("add_tags", d.tags), ("require_fields", d.require_fields)):
            vals = actions.get(key)
            if isinstance(vals, list):
                for v in vals:
                    s = str(v).strip()
                    if s and s not in target:
                        target.append(s)

        if actions.get("stop_processing") is True:
            break

    return d

# -----------------------------
# Text block rendering (from rhk_textdb)
# -----------------------------
def _render_block(block_id: str, ctx: SafeDict) -> str:
    try:
        b = textdb.get_block(block_id)
    except Exception:
        b = None
    if not b:
        return ""
    try:
        return str(b.template).format_map(ctx)
    except Exception:
        return str(b.template)

def _render_bundle(bundle_id: str, ctx: SafeDict) -> Tuple[str, str]:
    if not bundle_id:
        return "", ""
    bid = f"{bundle_id}_B"
    eid = f"{bundle_id}_E"
    return _render_block(bid, ctx).strip(), _render_block(eid, ctx).strip()

# -----------------------------
# Classification helpers
# -----------------------------
def classify_exercise_pattern(mpap_co_slope: Optional[float], pawp_co_slope: Optional[float], thr_mpap: float, thr_pawp: float) -> Optional[str]:
    if mpap_co_slope is None or pawp_co_slope is None:
        return None
    mpap_path = mpap_co_slope > thr_mpap
    pawp_path = pawp_co_slope > thr_pawp
    if (not mpap_path) and (not pawp_path):
        return "normal"
    if mpap_path and pawp_path:
        return "linkskardial"
    if mpap_path and (not pawp_path):
        return "pulmvasc"
    return "isoliert_pawp"

def pvr_severity(pvr: Optional[float], mild_ge: float, mod_ge: float, sev_ge: float) -> Optional[str]:
    if pvr is None:
        return None
    if pvr >= sev_ge: return "severe"
    if pvr >= mod_ge: return "moderate"
    if pvr >= mild_ge: return "mild"
    return "below"

def anemia_present(hb_g_dl: Optional[float], sex: Optional[str]) -> Optional[bool]:
    if hb_g_dl is None:
        return None
    s = (sex or "").strip().lower()
    if s.startswith("m"): return hb_g_dl < 13.0
    if s.startswith("w") or s.startswith("f"): return hb_g_dl < 12.0
    return hb_g_dl < 12.5

# -----------------------------
# Dashboard
# -----------------------------
def _badge(label: str, value: str, level: str = "neutral") -> str:
    colors = {
        "low": ("#0f7b0f", "#eaffea"),
        "intermediate": ("#8a6d00", "#fff6d5"),
        "intermediate-low": ("#8a6d00", "#fff6d5"),
        "intermediate-high": ("#a14400", "#ffe7d5"),
        "high": ("#b00020", "#ffe5e9"),
        "neutral": ("#333", "#f4f4f4"),
        "warn": ("#8a1c00", "#ffe7d5"),
        "info": ("#004c99", "#e8f2ff"),
    }
    fg, bg = colors.get(level, colors["neutral"])
    return (
        f"<div style='display:inline-flex;gap:8px;align-items:center;border-radius:12px;"
        f"padding:8px 10px;border:1px solid {fg};background:{bg};color:{fg};"
        f"margin:6px 6px 0 0;'><b>{label}</b><span>{value}</span></div>"
    )

def render_dashboard_html(summary: Dict[str, Any]) -> str:
    badges: List[str] = []
    if summary.get("bundle"):
        badges.append(_badge("Bundle", str(summary["bundle"]), "info"))
    if summary.get("primary_dx"):
        badges.append(_badge("Kernaussage", str(summary["primary_dx"]), "neutral"))

    esc3 = summary.get("esc3") or {}
    esc4 = summary.get("esc4") or {}
    rev = summary.get("reveal") or {}
    if esc3.get("overall"):
        badges.append(_badge("ESC/ERS 3-Strata", f"{esc3['overall']} ({esc3.get('category')})", esc3.get("category") or "neutral"))
    else:
        badges.append(_badge("ESC/ERS 3-Strata", "—", "neutral"))
    if esc4.get("overall"):
        badges.append(_badge("ESC/ERS 4-Strata", f"{esc4['overall']} ({esc4.get('category')})", esc4.get("category") or "neutral"))
    else:
        badges.append(_badge("ESC/ERS 4-Strata", "—", "neutral"))
    if rev.get("score") is not None:
        badges.append(_badge("REVEAL Lite 2", f"{rev['score']} ({rev.get('risk')})", rev.get("risk") or "neutral"))
    else:
        badges.append(_badge("REVEAL Lite 2", "—", "neutral"))

    hf_percent = summary.get("hfpef_percent")
    if hf_percent is not None:
        level = "warn" if hf_percent >= 80 else ("info" if hf_percent >= 50 else "neutral")
        badges.append(_badge("HFpEF-Wahrscheinlichkeit", f"{hf_percent:.0f}%", level))

    req = summary.get("require_fields") or []
    if req:
        badges.append(_badge("Fehlende Detailangaben", ", ".join(req[:5]) + (" …" if len(req) > 5 else ""), "warn"))

    tags = summary.get("tags") or []
    tags_html = ""
    if tags:
        tags_html = "<div style='margin-top:10px;color:#333;'><b>Hinweise:</b><ul style='margin:6px 0 0 18px;'>" + "".join([f"<li>{t}</li>" for t in tags]) + "</ul></div>"

    return (
        "<div style='padding:12px;border:1px solid #e5e5e5;border-radius:14px;background:#fff;'>"
        f"<div style='font-size:14px;color:#333;'><b>{APP_TITLE}</b></div>"
        f"<div style='margin-top:6px;display:flex;flex-wrap:wrap;'>{''.join(badges)}</div>"
        f"{tags_html}"
        "</div>"
    )

# -----------------------------
# Case builder
# -----------------------------
def build_case(ui: Dict[str, Any]) -> Dict[str, Any]:
    exam_date = parse_date(ui.get("exam_date")) or date.today()
    birth = parse_date(ui.get("birthdate"))
    age = calc_age_years(birth, exam_date)

    height_cm = _to_float(ui.get("height_cm"))
    weight_kg = _to_float(ui.get("weight_kg"))
    bsa = calc_bsa_dubois(height_cm, weight_kg)
    bmi = calc_bmi(height_cm, weight_kg)

    rap = _to_float(ui.get("rap"))
    spap = _to_float(ui.get("spap"))
    dpap = _to_float(ui.get("dpap"))
    mpap = _to_float(ui.get("mpap"))
    pawp = _to_float(ui.get("pawp"))
    co = _to_float(ui.get("co"))
    ci = _to_float(ui.get("ci"))
    hr = _to_float(ui.get("hr"))

    if mpap is None and spap is not None and dpap is not None:
        mpap = calc_mean(spap, dpap)
    if ci is None:
        ci = calc_ci(co, bsa)

    tpg = calc_tpg(mpap, pawp)
    dpg = calc_dpg(dpap, pawp)
    pvr = _to_float(ui.get("pvr"))
    if pvr is None:
        pvr = calc_pvr(mpap, pawp, co)

    exercise_done = bool(ui.get("exercise_done"))
    ex_co = _to_float(ui.get("ex_co")) if exercise_done else None
    ex_mpap = _to_float(ui.get("ex_mpap")) if exercise_done else None
    ex_pawp = _to_float(ui.get("ex_pawp")) if exercise_done else None
    ex_spap = _to_float(ui.get("ex_spap")) if exercise_done else None

    mpap_co_slope = _to_float(ui.get("mpap_co_slope"))
    pawp_co_slope = _to_float(ui.get("pawp_co_slope"))
    if mpap_co_slope is None and exercise_done:
        mpap_co_slope = calc_slope(mpap, co, ex_mpap, ex_co)
    if pawp_co_slope is None and exercise_done:
        pawp_co_slope = calc_slope(pawp, co, ex_pawp, ex_co)

    ci_peak = calc_ci(ex_co, bsa) if (exercise_done and ex_co is not None) else None
    delta_spap = (ex_spap - spap) if (exercise_done and ex_spap is not None and spap is not None) else None

    sex = ui.get("sex") or None
    hb = _to_float(ui.get("hb_g_dl"))
    anemia = anemia_present(hb, sex)
    anemia_type = ui.get("anemia_type") or None

    bnp_kind = ui.get("bnp_kind") or "NT-proBNP"
    bnp_value = _to_float(ui.get("bnp_value"))
    sbp = _to_float(ui.get("sbp"))
    egfr = _to_float(ui.get("egfr"))

    # echo/CMR
    lvef = _to_float(ui.get("lvef"))
    la_enlarged = bool(ui.get("la_enlarged"))
    af = bool(ui.get("atrial_fibrillation"))
    e_over_eprime = _to_float(ui.get("e_over_eprime"))
    pasp_echo = _to_float(ui.get("pasp_echo"))
    hf_res = hfpef_probability_model(age, bmi, e_over_eprime, pasp_echo, af)

    # S'/RAAI
    sprime = _to_float(ui.get("sprime_cm_s"))
    ra_area = _to_float(ui.get("ra_area_cm2"))
    sprime_raai = None
    if sprime is not None and ra_area is not None and bsa:
        raai = ra_area / bsa
        if raai > 0:
            sprime_raai = sprime / raai

    # definitions (default; rulebook can override in env)
    has_ph_rest = (mpap is not None and mpap > 20)
    has_postcap = (pawp is not None and pawp > 15)
    has_precap_component = (pvr is not None and pvr > 2.0)
    borderline_ph = (mpap is not None and 20 < mpap <= 24 and (pvr is None or pvr <= 2.0) and (pawp is None or pawp <= 15))
    lvef_preserved = (lvef is not None and lvef >= 50)

    # congestion heuristic
    ivc_mm = _to_float(ui.get("ivc_diam_mm"))
    ivc_coll = _to_float(ui.get("ivc_collapse_pct"))
    congestion_likely = False
    if rap is not None and rap >= 10:
        congestion_likely = True
    if ivc_mm is not None and ivc_coll is not None and ivc_mm >= 21 and ivc_coll < 50:
        congestion_likely = True

    # high flow heuristic
    high_flow = False
    if co is not None and co >= 8:
        high_flow = True
    if ci is not None and ci >= 4:
        high_flow = True

    # cteph markers
    cteph_markers = bool(ui.get("vq_positive") or ui.get("ct_chronic_thrombo") or ui.get("mosaic_perfusion"))

    derived = {
        "exam_date": exam_date.isoformat(),
        "age": age,
        "bsa": bsa,
        "bmi": bmi,
        "mpap": mpap,
        "tpg": tpg,
        "dpg": dpg,
        "pvr": pvr,
        "ci": ci,
        "exercise_done": exercise_done,
        "mpap_co_slope": mpap_co_slope,
        "pawp_co_slope": pawp_co_slope,
        "ci_peak": ci_peak,
        "delta_spap": delta_spap,
        "sprime_raai": sprime_raai,
        "hfpef_prob": hf_res.prob,
        "hfpef_percent": hf_res.percent,
        "hfpef_note": hf_res.note,
        "has_ph_rest": has_ph_rest,
        "has_postcap": has_postcap,
        "has_precap_component": has_precap_component,
        "borderline_ph": borderline_ph,
        "lvef_preserved": lvef_preserved,
        "congestion_likely": congestion_likely,
        "high_flow": high_flow,
        "cteph_markers": cteph_markers,
        "ild_present": bool(ui.get("ild_present")),
        "emphysema_present": bool(ui.get("emphysema_present")),
        "ltot_present": bool(ui.get("ltot_present")),
        "ltot_flow_l_min": _to_float(ui.get("ltot_flow_l_min")),
        "lufu_done": bool(ui.get("lufu_done")),
        "virology_positive": bool(ui.get("virology_positive")),
        "immunology_positive": bool(ui.get("immunology_positive")),
        "abdomen_sono_done": bool(ui.get("abdomen_sono_done")),
        "portal_hypertension": bool(ui.get("portal_hypertension")),
        "bnp_marker": bnp_kind,
        "entresto": bool(ui.get("entresto")),
        "anemia_present": anemia,
        "anemia_type": anemia_type,
        "vq_done": bool(ui.get("vq_done")),
    }

    # exercise pattern default (thresholds will be applied later in env)
    if exercise_done:
        derived["exercise_pattern"] = classify_exercise_pattern(mpap_co_slope, pawp_co_slope, 3.0, 2.0)
    else:
        derived["exercise_pattern"] = None

    # PVR severity & ci_low default
    derived["pvr_severity"] = pvr_severity(pvr, 2.0, 5.0, 10.0)
    derived["ci_low"] = (ci is not None and ci < 2.0)

    # scores
    esc3 = esc3_overall(ui.get("who_fc") or None, _to_float(ui.get("sixmwd_m")), bnp_kind, bnp_value)
    esc4 = esc4_overall(ui.get("who_fc") or None, _to_float(ui.get("sixmwd_m")), bnp_kind, bnp_value)
    reveal = reveal_lite2_score(ui.get("who_fc") or None, _to_float(ui.get("sixmwd_m")), bnp_kind, bnp_value, sbp, hr, egfr)

    return {
        "schema": UI_SCHEMA,
        "ui": ui,
        "derived": derived,
        "scores": {"esc3": esc3, "esc4": esc4, "reveal": reveal},
    }

# -----------------------------
# Report rendering context
# -----------------------------
def build_render_ctx(case: Dict[str, Any], decision: Decision) -> SafeDict:
    ui = case.get("ui") or {}
    d = case.get("derived") or {}

    mpap = _to_float(d.get("mpap"))
    pawp = _to_float(ui.get("pawp"))
    pvr = _to_float(d.get("pvr"))
    ci = _to_float(d.get("ci"))
    co = _to_float(ui.get("co"))

    mpap_phrase = f"mPAP {fmt_num(mpap,0)} mmHg" if mpap is not None else "mPAP —"
    pawp_phrase = f"PAWP {fmt_num(pawp,0)} mmHg" if pawp is not None else "PAWP —"
    pvr_phrase = f"PVR {fmt_num(pvr,1)} WU" if pvr is not None else "PVR —"
    ci_phrase = f"CI {fmt_num(ci,2)} L/min/m²" if ci is not None else (f"CO {fmt_num(co,2)} L/min" if co is not None else "CI/CO —")

    oxygen_mode = ui.get("oxygen_mode") or "—"
    oxygen_flow = _to_float(ui.get("oxygen_flow_l_min"))
    if str(oxygen_mode).lower().startswith("raum"):
        oxygen_sentence = "Raumluft"
    elif str(oxygen_mode).lower().startswith("o2") or str(oxygen_mode).lower().startswith("sauer"):
        oxygen_sentence = f"O2 {fmt_num(oxygen_flow,0)} L/min" if oxygen_flow is not None else "O2"
    else:
        oxygen_sentence = str(oxygen_mode)

    step_up = ui.get("step_up_present")
    if isinstance(step_up, str):
        step_up = {"ja": True, "nein": False}.get(step_up.strip().lower(), None)

    if step_up is True:
        step_up_sentence = "Sättigungssprung in der Stufenoxymetrie."
    elif step_up is False:
        step_up_sentence = "Kein relevanter Sättigungssprung in der Stufenoxymetrie."
    else:
        step_up_sentence = "Stufenoxymetrie: Bewertung unklar."

    rest_ph_sentence = "keine PH in Ruhe" if d.get("has_ph_rest") is False else ("PH in Ruhe" if d.get("has_ph_rest") is True else "unklar")

    # slopes
    mpap_co_slope = _to_float(d.get("mpap_co_slope"))
    pawp_co_slope = _to_float(d.get("pawp_co_slope"))

    # previous RHK comparison sentence
    prev_date = str(ui.get("prev_rhk_date") or "").strip()
    prev_summary = (ui.get("prev_rhk_summary") or "").strip()
    comparison_sentence = ""
    if prev_date and prev_summary:
        comparison_sentence = f"Im Vergleich zu RHK {prev_date} {prev_summary}."
    elif prev_date and any(ui.get(k) is not None for k in ("prev_mpap","prev_pawp","prev_ci","prev_pvr")):
        pm = fmt_num(_to_float(ui.get("prev_mpap")), 0)
        pw = fmt_num(_to_float(ui.get("prev_pawp")), 0)
        pci = fmt_num(_to_float(ui.get("prev_ci")), 2)
        ppvr = fmt_num(_to_float(ui.get("prev_pvr")), 1)
        comparison_sentence = f"Im Vergleich zu RHK {prev_date} Verlauf (mPAP {pm} mmHg, PAWP {pw} mmHg, CI {pci} L/min/m², PVR {ppvr} WU)."

    ctx = SafeDict({
        "ci_phrase": ci_phrase,
        "co_method_desc": ui.get("co_method") or "—",
        "oxygen_sentence": oxygen_sentence,
        "systemic_sentence": "",
        "exam_type_desc": ui.get("exam_type") or "—",
        "step_up_sentence": step_up_sentence,
        "mpap_phrase": mpap_phrase,
        "pawp_phrase": pawp_phrase,
        "pvr_phrase": pvr_phrase,
        "mPAP_CO_slope": fmt_num(mpap_co_slope, 1),
        "PAWP_CO_slope": fmt_num(pawp_co_slope, 1),
        "rest_ph_sentence": rest_ph_sentence,
        "comparison_sentence": comparison_sentence,
        "lufu_summary": (ui.get("lufu_summary") or "").strip(),
        "cv_stauung_phrase": "Hinweis auf zentrale Stauung." if d.get("congestion_likely") else "Keine zentrale Stauung.",
        "pv_stauung_phrase": "Hinweis auf pulmonalvenöse Stauung." if d.get("has_postcap") else "Keine pulmonalvenöse Stauung.",
        "pressure_resistance_short": join_nonempty([mpap_phrase, pawp_phrase, pvr_phrase]),
        "step_up_from_to": ui.get("step_up_from_to") or "—",
        "provocation_sentence": "",
        "provocation_type_desc": "Belastung" if d.get("exercise_done") else "—",
        "provocation_result_sentence": "",
        "borderline_ph_sentence": rest_ph_sentence,
        "pvr_sev_phrase": d.get("pvr_severity") or "",
        "therapy_neutral_sentence": "",
        "therapy_plan_sentence": "",
        "therapy_escalation_sentence": "",
        "delta_sPAP": fmt_num(_to_float(d.get("delta_spap")), 0),
        "CI_peak": fmt_num(_to_float(d.get("ci_peak")), 2),
    })
    return ctx

# -----------------------------
# Reports
# -----------------------------
def build_patient_report(case: Dict[str, Any], decision: Decision) -> str:
    ui = case.get("ui") or {}
    d = case.get("derived") or {}
    scores = case.get("scores") or {}
    hf = {
        "percent": d.get("hfpef_percent"),
        "note": d.get("hfpef_note"),
    }

    name = join_nonempty([str(ui.get("first_name") or "").strip(), str(ui.get("last_name") or "").strip()], " ")
    if not name.strip():
        name = "Patient:in"

    paragraphs: List[str] = []
    paragraphs.append(f"Patientenbericht – verständliche Zusammenfassung für {name}")
    paragraphs.append("")
    paragraphs.append("Was wurde gemacht?")
    paragraphs.append(
        "Wir haben eine Untersuchung durchgeführt, bei der Druck- und Flusswerte im rechten Herzen "
        "und in den Lungengefäßen gemessen werden. Damit kann man Ursachen für Luftnot und geringe Belastbarkeit besser einordnen."
    )
    paragraphs.append("")
    paragraphs.append("Was ist das wichtigste Ergebnis?")
    if decision.primary_dx:
        dx = decision.primary_dx.replace("PH", "einen erhöhten Druck im Lungenkreislauf")
        paragraphs.append(dx)
    else:
        paragraphs.append("Aus den vorliegenden Messwerten ergibt sich kein eindeutiger Hauptbefund.")

    paragraphs.append("")
    paragraphs.append("Was bedeutet das für mich?")
    if hf.get("percent") is not None and hf["percent"] >= 50:
        paragraphs.append(
            "Es gibt Hinweise darauf, dass das Herz sich in der Entspannungsphase nicht ganz leicht füllt. "
            "Das kann dazu führen, dass sich Druck nach hinten in Richtung Lunge aufbaut – besonders bei Belastung."
        )
    if d.get("ild_present") or d.get("emphysema_present") or d.get("ltot_present"):
        paragraphs.append(
            "Es gibt außerdem Hinweise, dass auch die Lunge bzw. die Sauerstoffversorgung eine Rolle spielt. "
            "Darum sind Lungenfunktion und Bildgebung wichtig."
        )
    if d.get("cteph_markers"):
        paragraphs.append(
            "Es gibt Hinweise, dass (alte) Blutgerinnsel in den Lungengefäßen eine Rolle spielen könnten. "
            "Das wird gezielt mit speziellen Untersuchungen abgeklärt."
        )

    paragraphs.append("")
    paragraphs.append("Wie geht es weiter?")
    if decision.recommendations:
        paragraphs.append("Als nächste Schritte empfehlen wir (je nach Gesamtsituation):")
        for r in decision.recommendations[:10]:
            rr = r.replace("V/Q", "eine spezielle Lungen-Durchblutungsuntersuchung")
            rr = rr.replace("CT", "Computertomographie")
            rr = rr.replace("Echo", "Herzultraschall")
            paragraphs.append(f"- {rr}")
    else:
        paragraphs.append(
            "Die nächsten Schritte hängen von Beschwerden, Vorerkrankungen und Vorbefunden ab. "
            "Gegebenenfalls sind zusätzliche Untersuchungen sinnvoll."
        )

    paragraphs.append("")
    paragraphs.append("Worauf sollte ich achten?")
    paragraphs.append(
        "Bitte melden Sie sich zeitnah, wenn Sie deutlich schlechter Luft bekommen, neue Brustschmerzen, Ohnmachtsanfälle, "
        "starke Wassereinlagerungen (geschwollene Beine) oder eine deutliche Leistungsminderung bemerken."
    )
    paragraphs.append("")
    paragraphs.append("Hinweis: Dieser Bericht ersetzt kein ärztliches Gespräch. Bitte besprechen Sie alles mit Ihrem Behandlungsteam.")
    return "\n".join(paragraphs).strip() + "\n"

def build_doctor_report(case: Dict[str, Any], decision: Decision) -> Tuple[str, str, str]:
    ui = case.get("ui") or {}
    d = case.get("derived") or {}
    scores = case.get("scores") or {}

    exam_date = d.get("exam_date") or _now_iso()
    ident = join_nonempty([str(ui.get("last_name") or "").strip(), str(ui.get("first_name") or "").strip()], ", ")
    if ui.get("birthdate"):
        ident = join_nonempty([ident, str(ui.get("birthdate"))], " | ")

    mpap = _to_float(d.get("mpap"))
    pawp = _to_float(ui.get("pawp"))
    pvr = _to_float(d.get("pvr"))
    ci = _to_float(d.get("ci"))
    co = _to_float(ui.get("co"))
    tpg = _to_float(d.get("tpg"))
    dpg = _to_float(d.get("dpg"))

    hemo_parts = [
        f"RAP {fmt_unit(_to_float(ui.get('rap')),'mmHg',0)}" if _to_float(ui.get("rap")) is not None else "",
        f"PA {fmt_num(_to_float(ui.get('spap')),0)}/{fmt_num(_to_float(ui.get('dpap')),0)} (m {fmt_num(mpap,0)}) mmHg" if _to_float(ui.get("spap")) is not None and _to_float(ui.get("dpap")) is not None and mpap is not None else "",
        f"PAWP {fmt_unit(pawp,'mmHg',0)}" if pawp is not None else "",
        f"CO {fmt_unit(co,'L/min',2)}" if co is not None else "",
        f"CI {fmt_unit(ci,'L/min/m²',2)}" if ci is not None else "",
        f"PVR {fmt_unit(pvr,'WU',1)}" if pvr is not None else "",
        f"TPG {fmt_unit(tpg,'mmHg',0)}" if tpg is not None else "",
        f"DPG {fmt_unit(dpg,'mmHg',0)}" if dpg is not None else "",
    ]
    hemo_line = " | ".join([p for p in hemo_parts if p]) or "—"

    ex_lines = []
    if d.get("exercise_done"):
        ex_lines.append(
            "Belastung: "
            + " | ".join(
                [
                    f"mPAP/CO-Slope {fmt_num(_to_float(d.get('mpap_co_slope')),1)} mmHg/(L/min)" if _to_float(d.get("mpap_co_slope")) is not None else "",
                    f"PAWP/CO-Slope {fmt_num(_to_float(d.get('pawp_co_slope')),1)} mmHg/(L/min)" if _to_float(d.get("pawp_co_slope")) is not None else "",
                    f"ΔsPAP {fmt_unit(_to_float(d.get('delta_spap')),'mmHg',0)}" if _to_float(d.get("delta_spap")) is not None else "",
                    f"CI_peak {fmt_unit(_to_float(d.get('ci_peak')),'L/min/m²',2)}" if _to_float(d.get("ci_peak")) is not None else "",
                ]
            )
        )

    # Render bundle blocks
    ctx = build_render_ctx(case, decision)
    beur, empf = _render_bundle(decision.bundle or "", ctx)

    # Risk block: directly after diagnosis
    esc3 = scores.get("esc3") or {}
    esc4 = scores.get("esc4") or {}
    reveal = scores.get("reveal") or {}

    risk_lines = ["RISIKOSTRATIFIZIERUNG"]
    risk_lines.append(f"- ESC/ERS 3-Strata: {esc3.get('overall') or '—'} ({esc3.get('category') or '—'})")
    risk_lines.append(f"- ESC/ERS 4-Strata: {esc4.get('overall') or '—'} ({esc4.get('category') or '—'})")
    if reveal.get("score") is not None:
        risk_lines.append(f"- REVEAL Lite 2: {reveal.get('score')} ({reveal.get('risk')})")
    else:
        risk_lines.append("- REVEAL Lite 2: —")
    if d.get("hfpef_percent") is not None:
        risk_lines.append(f"- HFpEF-Wahrscheinlichkeit (Modell): {d.get('hfpef_percent'):.0f}% (Quelle im Tool)")

    # Modules: user + engine
    user_modules = ui.get("modules") or []
    if isinstance(user_modules, str):
        user_modules = [user_modules]
    all_modules = []
    for m in (decision.modules + list(user_modules)):
        if m and m not in all_modules:
            all_modules.append(m)

    mod_lines = []
    if all_modules:
        mod_lines.append("ZUSATZMODULE (ausgewählt/engine)")
        for mid in all_modules:
            txt = _render_block(mid, ctx).strip()
            mod_lines.append(f"- {txt}" if txt else f"- {mid}")

    extra_recs = []
    for r in decision.recommendations:
        if r and r not in extra_recs:
            extra_recs.append(r)

    req_block = ""
    if decision.require_fields:
        req_block = "WAS FEHLT NOCH?\n- " + "\n- ".join(decision.require_fields)

    tag_block = ""
    if decision.tags:
        tag_block = "HINWEISE (Engine)\n- " + "\n- ".join(decision.tags)

    lines: List[str] = []
    lines.append("BEFUNDKOPF")
    lines.append(join_nonempty([ui.get("exam_type") or "RHK", ui.get("setting") or "Ruhe", f"Datum {exam_date}", ident], " | "))
    if ui.get("story"):
        lines.append(f"Klinik/Kurzanamnese: {ui.get('story')}")
    lines.append("")
    lines.append("HÄMODYNAMIK")
    lines.append(f"- {hemo_line}")
    for ln in ex_lines:
        lines.append(f"- {ln}")
    if ui.get("sprime_cm_s") is not None or ui.get("ra_area_cm2") is not None:
        spr = _to_float(d.get("sprime_raai"))
        if spr is not None:
            lines.append(f"- Echo-Add-on: S'/RAAI (berechnet) {fmt_num(spr,2)}")

    lines.append("")
    lines.append("DIAGNOSE / KLASSIFIKATION")
    lines.append(decision.primary_dx or "—")
    lines.append("")
    lines.extend(risk_lines)
    lines.append("")
    lines.append("BEURTEILUNG")
    lines.append(beur if beur else "—")
    if tag_block:
        lines.append("")
        lines.append(tag_block)
    lines.append("")
    lines.append("EMPFEHLUNG")
    if empf:
        lines.append(empf)
    if extra_recs:
        lines.append("")
        lines.append("Zusatzhinweise/Empfehlungen (Engine):")
        for r in extra_recs:
            lines.append(f"- {r}")
    if mod_lines:
        lines.append("")
        lines.extend(mod_lines)
    if req_block:
        lines.append("")
        lines.append(req_block)

    doctor_txt = "\n".join(lines).strip() + "\n"

    internal_lines = [
        "INTERN / DEBUG",
        f"Bundle: {decision.bundle}",
        f"Fired rules: {', '.join(decision.fired_rules) if decision.fired_rules else '—'}",
        "",
        "Key derived:",
        f"- mpap: {d.get('mpap')}",
        f"- pawp: {ui.get('pawp')}",
        f"- pvr: {d.get('pvr')}",
        f"- ci: {d.get('ci')}",
        f"- exercise_pattern: {d.get('exercise_pattern')}",
    ]
    internal_txt = "\n".join(internal_lines).strip() + "\n"

    patient_txt = build_patient_report(case, decision)
    return doctor_txt, patient_txt, internal_txt

# -----------------------------
# Random example
# -----------------------------
def _rand(a: float, b: float) -> float:
    return random.uniform(a, b)

def generate_random_example_ui() -> Dict[str, Any]:
    scenarios = ["normal","precap_mild","precap_mod","precap_sev","ipcph_hfpef","cpcph_hfpef","cteph","group3_ild","exercise_linkskardial","shunt","portopulm"]
    scen = random.choice(scenarios)

    height = _rand(160, 190)
    weight = _rand(55, 105)
    bsa = calc_bsa_dubois(height, weight) or 1.8

    ui: Dict[str, Any] = {}
    ui["exam_date"] = _now_iso()
    ui["exam_type"] = random.choice(["Initial-RHK", "Verlaufskontrolle"])
    ui["setting"] = "Ruhe"
    ui["oxygen_mode"] = random.choice(["Raumluft", "O2"])
    ui["oxygen_flow_l_min"] = 2 if ui["oxygen_mode"] == "O2" else None
    ui["co_method"] = "Thermodilution"
    ui["sex"] = random.choice(["m", "w"])
    ui["birthdate"] = f"{random.randint(1940, 2002)}-01-01"
    ui["height_cm"] = round(height)
    ui["weight_kg"] = round(weight, 1)
    ui["last_name"] = random.choice(["Mustermann", "Beispiel", "Test"])
    ui["first_name"] = random.choice(["Max", "Erika", "Sam"])

    ui["story"] = random.choice([
        "Belastungsdyspnoe, Leistungsabfall, unklare Ursache.",
        "Bekannte pulmonale Hypertonie, Verlaufskontrolle.",
        "Dyspnoe und Müdigkeit, Abklärung PH/HFpEF.",
    ])

    # Defaults
    ui.update({
        "exercise_done": False,
        "ild_present": False,
        "emphysema_present": False,
        "mosaic_perfusion": False,
        "ct_chronic_thrombo": False,
        "vq_done": False,
        "vq_positive": False,
        "la_enlarged": False,
        "lvef": 60,
        "atrial_fibrillation": False,
        "lufu_done": False,
        "ltot_present": False,
        "virology_positive": False,
        "immunology_positive": False,
        "abdomen_sono_done": False,
        "portal_hypertension": False,
        "entresto": False,
        "anemia_type": None,
    })

    ui["bnp_kind"] = "NT-proBNP"
    ui["bnp_value"] = int(_rand(80, 1400))
    ui["sixmwd_m"] = int(_rand(120, 520))
    ui["who_fc"] = random.choice(["I","II","III","IV"])
    ui["sbp"] = int(_rand(95, 150))
    ui["egfr"] = int(_rand(35, 95))
    ui["hr"] = int(_rand(55, 105))
    ui["hb_g_dl"] = round(_rand(10.5, 15.5), 1)

    ui["e_over_eprime"] = round(_rand(6, 18), 1)
    ui["pasp_echo"] = int(_rand(25, 70))

    def set_hemo(mpap_range, pawp_range, pvr_range, rap_range=(3, 14)):
        mpap = _rand(*mpap_range)
        pawp = _rand(*pawp_range)
        pvr = _rand(*pvr_range)
        delta = max(5.0, mpap - pawp)
        co = delta / max(0.5, pvr)
        co = max(2.0, min(10.0, co))
        pvr = delta / co
        ui["mpap"] = round(mpap)
        ui["pawp"] = round(pawp)
        ui["co"] = round(co, 2)
        ui["rap"] = round(_rand(*rap_range))
        ui["spap"] = round(mpap + _rand(8, 25))
        ui["dpap"] = max(5, round(mpap - _rand(8, 15)))
        ui["pvr"] = None
        ui["ci"] = None

    if scen == "normal":
        set_hemo((12, 19), (6, 12), (0.8, 1.9))
    elif scen == "precap_mild":
        set_hemo((23, 32), (6, 12), (2.2, 4.0), rap_range=(4, 10))
    elif scen == "precap_mod":
        set_hemo((30, 45), (6, 12), (5.0, 8.5), rap_range=(6, 14))
    elif scen == "precap_sev":
        set_hemo((40, 65), (6, 12), (10.0, 18.0), rap_range=(10, 18))
    elif scen == "ipcph_hfpef":
        set_hemo((25, 40), (16, 25), (0.8, 2.0), rap_range=(6, 14))
        ui["la_enlarged"] = True
        ui["atrial_fibrillation"] = random.choice([True, False])
    elif scen == "cpcph_hfpef":
        set_hemo((28, 45), (18, 28), (2.5, 6.0), rap_range=(8, 16))
        ui["la_enlarged"] = True
        ui["atrial_fibrillation"] = random.choice([True, False])
    elif scen == "cteph":
        set_hemo((30, 55), (6, 12), (5.0, 12.0), rap_range=(8, 16))
        ui["ct_chronic_thrombo"] = True
        ui["mosaic_perfusion"] = True
        ui["vq_done"] = True
        ui["vq_positive"] = True
    elif scen == "group3_ild":
        set_hemo((28, 50), (6, 12), (3.0, 9.0), rap_range=(6, 14))
        ui["ild_present"] = True
        ui["ild_type"] = random.choice(["IPF","NSIP","unklar"])
        ui["ild_extent"] = random.choice(["mild","moderate","severe"])
        ui["ild_histology"] = random.choice(["ja","nein"])
        ui["ild_fibrosis_clinic"] = random.choice(["ja","nein"])
        ui["lufu_done"] = True
        ui["dlco_sb"] = int(_rand(25, 55))
        ui["ltot_present"] = random.choice([True, False])
        ui["ltot_flow_l_min"] = 2 if ui["ltot_present"] else None
    elif scen == "exercise_linkskardial":
        set_hemo((12, 19), (6, 12), (0.8, 1.9), rap_range=(3, 10))
        ui["exercise_done"] = True
        rest_co = ui["co"]
        ui["ex_co"] = round(min(12.0, float(rest_co) + _rand(2.0, 4.5)), 2)
        ui["ex_mpap"] = round(ui["mpap"] + _rand(10, 20))
        ui["ex_pawp"] = round(ui["pawp"] + _rand(8, 18))
        ui["ex_spap"] = round(ui["spap"] + _rand(15, 35))
        ui["setting"] = "Belastung"
        ui["la_enlarged"] = True
    elif scen == "shunt":
        set_hemo((12, 25), (6, 12), (0.8, 3.0), rap_range=(3, 12))
        ui["step_up_present"] = "ja"
        ui["step_up_from_to"] = "RA → RV"
    elif scen == "portopulm":
        set_hemo((25, 40), (8, 14), (2.0, 6.0), rap_range=(4, 12))
        ui["portal_hypertension"] = True
        ui["co"] = round(_rand(7.5, 10.5), 2)
        ui["spap"] = round(ui["mpap"] + _rand(10, 25))
        ui["dpap"] = max(6, round(ui["mpap"] - _rand(8, 14)))

    return ui

# -----------------------------
# Export helpers
# -----------------------------
def _write_tmp(text: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix="rhk_v18_", suffix=suffix)
    os.close(fd)
    Path(path).write_text(text or "", encoding="utf-8")
    return path

def export_json(case: Dict[str, Any]) -> str:
    return _write_tmp(json.dumps(case, ensure_ascii=False, indent=2), ".json")

def export_txt(text: str, name: str) -> str:
    return _write_tmp(text or "", f"_{name}.txt")

def export_pdf(text: str, name: str) -> Optional[str]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
    except Exception:
        return None
    fd, path = tempfile.mkstemp(prefix="rhk_v18_", suffix=f"_{name}.pdf")
    os.close(fd)
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    x = 15 * mm
    y = height - 20 * mm
    lh = 5 * mm
    for line in (text or "").splitlines():
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        c.drawString(x, y, line[:200])
        y -= lh
    c.save()
    return path

# -----------------------------
# UI build
# -----------------------------
def build_demo() -> gr.Blocks:
    # Rulebook file (write default if missing)
    rules_path = _APP_DIR / "rhk_rules_v18.yaml"
    if not rules_path.exists():
        try:
            rules_path.write_text(DEFAULT_RULEBOOK_YAML.strip() + "\n", encoding="utf-8")
        except Exception:
            pass
    rulebook = _load_yaml(rules_path)

    css = """
    <style>
    .container {max-width: 1500px; margin: 0 auto;}
    </style>
    """

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(css)
        state_case = gr.State(value=None)

        # Top bar (buttons at top)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"## {APP_TITLE}\n<span style='color:#666;font-size:12px;'>Regelwerk: rhk_rules_v18.yaml • Textbausteine: rhk_textdb.py</span>")
        with gr.Row():
            btn_example = gr.Button("🎲 Beispiel laden (random)", variant="secondary")
            btn_generate = gr.Button("✅ Befund erstellen / aktualisieren", variant="primary")
            btn_save = gr.Button("💾 Fall speichern", variant="secondary")
            up_load = gr.UploadButton("📂 Fall laden", file_types=[".json"], variant="secondary")

        gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=7):
                # INPUTS
                with gr.Tabs():
                    with gr.Tab("Klinik & Labor"):
                        with gr.Row():
                            last_name = gr.Textbox(label="Name", value="")
                            first_name = gr.Textbox(label="Vorname", value="")
                            sex = gr.Dropdown(label="Geschlecht", choices=["", "m", "w"], value="")
                            birthdate = gr.Textbox(label="Geburtsdatum (YYYY-MM-DD)", value="")
                        with gr.Row():
                            exam_date = gr.Textbox(label="Untersuchungsdatum (YYYY-MM-DD)", value=_now_iso())
                            exam_type = gr.Dropdown(label="Typ", choices=["Initial-RHK", "Verlaufskontrolle"], value="Initial-RHK")
                            setting = gr.Dropdown(label="Setting", choices=["Ruhe", "Belastung", "Volumenchallenge", "Vasoreaktivität"], value="Ruhe")
                        with gr.Row():
                            height_cm = gr.Number(label="Größe (cm)", value=None, precision=0)
                            weight_kg = gr.Number(label="Gewicht (kg)", value=None, precision=1)
                        story = gr.Textbox(label="Story / Kurz-Anamnese", value="", lines=3)

                        gr.Markdown("### Labor (optional)")
                        with gr.Row():
                            hb_g_dl = gr.Number(label="Hb (g/dl)", value=None, precision=1)
                            anemia_type = gr.Dropdown(label="Anämie-Typ (wenn Hb niedrig)", choices=["", "micro", "normo", "macro"], value="", visible=False)
                            thrombos = gr.Number(label="Thrombos (G/l)", value=None, precision=0)
                            crp = gr.Number(label="CRP (mg/l)", value=None, precision=0)
                        with gr.Row():
                            leukos = gr.Number(label="Leukos (G/l)", value=None, precision=1)
                            inr = gr.Number(label="INR", value=None, precision=2)
                            quick = gr.Number(label="Quick (%)", value=None, precision=0)
                            ptt = gr.Number(label="PTT (s)", value=None, precision=0)
                        with gr.Row():
                            krea = gr.Number(label="Krea (mg/dl)", value=None, precision=2)
                            egfr = gr.Number(label="eGFR (ml/min/1.73m²)", value=None, precision=0)
                            entresto = gr.Checkbox(label="Entresto/ARNI?", value=False)
                        with gr.Row():
                            bnp_kind = gr.Dropdown(label="BNP-Marker", choices=["NT-proBNP", "BNP"], value="NT-proBNP")
                            bnp_value = gr.Number(label="BNP/NT-proBNP", value=None, precision=0)

                        gr.Markdown("### Infektiologie / Immunologie (optional)")
                        with gr.Row():
                            virology_positive = gr.Checkbox(label="Virologie positiv?", value=False)
                            immunology_positive = gr.Checkbox(label="Immunologie positiv?", value=False)
                        with gr.Row():
                            virology_details = gr.Textbox(label="Details Virologie (wenn positiv)", value="", lines=2, visible=False)
                            immunology_details = gr.Textbox(label="Details Immunologie (wenn positiv)", value="", lines=2, visible=False)

                        gr.Markdown("### Abdomen / Leber (optional)")
                        with gr.Row():
                            abdomen_sono_done = gr.Checkbox(label="Abdomen-Sono durchgeführt?", value=False)
                            portal_hypertension = gr.Checkbox(label="Hinweis auf portale Hypertension?", value=False)
                        abdomen_findings = gr.Textbox(label="Abdomen-Sono: besondere Befunde? (Freitext)", value="", lines=2, visible=False)

                    with gr.Tab("RHK & Belastung"):
                        with gr.Row():
                            oxygen_mode = gr.Dropdown(label="O2-Modus", choices=["Raumluft", "O2"], value="Raumluft")
                            oxygen_flow_l_min = gr.Number(label="O2 Flow (L/min)", value=None, precision=0)
                            co_method = gr.Dropdown(label="CO-Methode", choices=["Thermodilution", "Fick_direkt", "Fick_indirekt"], value="Thermodilution")
                        gr.Markdown("### Ruhe-Hämodynamik")
                        with gr.Row():
                            rap = gr.Number(label="RAP (mmHg)", value=None, precision=0)
                            spap = gr.Number(label="sPAP (mmHg)", value=None, precision=0)
                            dpap = gr.Number(label="dPAP (mmHg)", value=None, precision=0)
                            mpap = gr.Number(label="mPAP (mmHg) – optional (wird sonst berechnet)", value=None, precision=0)
                        with gr.Row():
                            pawp = gr.Number(label="PAWP (mmHg)", value=None, precision=0)
                            co = gr.Number(label="CO (L/min)", value=None, precision=2)
                            ci = gr.Number(label="CI (L/min/m²) – optional (wird sonst berechnet)", value=None, precision=2)
                            hr = gr.Number(label="HF (/min)", value=None, precision=0)
                        pvr = gr.Number(label="PVR (WU) – optional (wird sonst berechnet)", value=None, precision=1)

                        gr.Markdown("### Stufenoxymetrie")
                        with gr.Row():
                            step_up_present = gr.Dropdown(label="Sättigungssprung?", choices=["unklar", "ja", "nein"], value="unklar")
                            step_up_from_to = gr.Textbox(label="Ort (z.B. RA → RV)", value="")

                        gr.Markdown("### Vasoreaktivität (optional)")
                        vasoreactivity_done = gr.Checkbox(label="Vasoreaktivitätstest durchgeführt?", value=False)
                        ino_responder = gr.Dropdown(label="Responder? (wenn durchgeführt)", choices=["", "ja", "nein"], value="", visible=False)

                        gr.Markdown("### Belastung (optional)")
                        exercise_done = gr.Checkbox(label="Belastungsdaten vorhanden", value=False)
                        with gr.Row():
                            ex_co = gr.Number(label="CO_peak (L/min)", value=None, precision=2)
                            ex_mpap = gr.Number(label="mPAP_peak (mmHg)", value=None, precision=0)
                            ex_pawp = gr.Number(label="PAWP_peak (mmHg)", value=None, precision=0)
                            ex_spap = gr.Number(label="sPAP_peak (mmHg)", value=None, precision=0)
                        with gr.Row():
                            mpap_co_slope = gr.Number(label="mPAP/CO-Slope – optional (wird sonst berechnet)", value=None, precision=2)
                            pawp_co_slope = gr.Number(label="PAWP/CO-Slope – optional (wird sonst berechnet)", value=None, precision=2)

                        gr.Markdown("### Vorheriger RHK (optional)")
                        with gr.Row():
                            prev_rhk_date = gr.Textbox(label="Datum Vor-RHK (z.B. 03/21 oder YYYY-MM-DD)", value="")
                            prev_rhk_summary = gr.Textbox(label="Freitext (z.B. stabiler Verlauf)", value="")
                        with gr.Row():
                            prev_mpap = gr.Number(label="Vor-RHK mPAP (mmHg)", value=None, precision=0)
                            prev_pawp = gr.Number(label="Vor-RHK PAWP (mmHg)", value=None, precision=0)
                            prev_ci = gr.Number(label="Vor-RHK CI (L/min/m²)", value=None, precision=2)
                            prev_pvr = gr.Number(label="Vor-RHK PVR (WU)", value=None, precision=1)

                    with gr.Tab("Lunge & Bildgebung / Echo / CMR"):
                        gr.Markdown("### Lungenfunktion / LTOT")
                        with gr.Row():
                            lufu_done = gr.Checkbox(label="Lufu durchgeführt?", value=False)
                            dlco_sb = gr.Number(label="DLCO SB (%)", value=None, precision=0)
                            ltot_present = gr.Checkbox(label="LTOT vorhanden?", value=False)
                            ltot_flow_l_min = gr.Number(label="LTOT Flow (L/min)", value=None, precision=0, visible=False)
                        lufu_summary = gr.Textbox(label="Lufu Summary (Freitext)", value="", lines=3)

                        gr.Markdown("### CT / Bildgebung Thorax")
                        with gr.Row():
                            vq_done = gr.Checkbox(label="V/Q durchgeführt?", value=False)
                            vq_positive = gr.Checkbox(label="V/Q pathologisch (Perfusionsdefekte)?", value=False)
                            ct_chronic_thrombo = gr.Checkbox(label="CT-Hinweis auf chronische Thromboembolien?", value=False)
                            mosaic_perfusion = gr.Checkbox(label="Mosaikperfusion?", value=False)
                        with gr.Row():
                            ild_present = gr.Checkbox(label="ILD?", value=False)
                            emphysema_present = gr.Checkbox(label="Emphysem?", value=False)

                        with gr.Row():
                            ild_type = gr.Dropdown(label="ILD-Typ", choices=["", "IPF", "NSIP", "HP", "Sarko", "unklar"], value="", visible=False)
                            ild_extent = gr.Dropdown(label="Ausmaß ILD", choices=["", "mild", "moderate", "severe"], value="", visible=False)
                        with gr.Row():
                            ild_histology = gr.Dropdown(label="Histologisch gesichert?", choices=["", "ja", "nein"], value="", visible=False)
                            ild_fibrosis_clinic = gr.Dropdown(label="An Fibroseambulanz angebunden?", choices=["", "ja", "nein"], value="", visible=False)

                        gr.Markdown("### Echo / CMR (optional)")
                        with gr.Row():
                            lvef = gr.Number(label="LVEF (%)", value=None, precision=0)
                            la_enlarged = gr.Checkbox(label="Linkes Atrium vergrößert?", value=False)
                            atrial_fibrillation = gr.Checkbox(label="Vorhofflimmern?", value=False)
                        with gr.Row():
                            e_over_eprime = gr.Number(label="E/e' (Ratio)", value=None, precision=1)
                            pasp_echo = gr.Number(label="PASP (Echo, mmHg)", value=None, precision=0)
                        with gr.Row():
                            ivc_diam_mm = gr.Number(label="V. cava inferior Durchmesser (mm)", value=None, precision=0)
                            ivc_collapse_pct = gr.Number(label="VCI Kollaps (%)", value=None, precision=0)

                        gr.Markdown("### Echo-Add-on: S'/RAAI (Yogeswaran et al.)")
                        with gr.Row():
                            sprime_cm_s = gr.Number(label="S' (cm/s)", value=None, precision=1)
                            ra_area_cm2 = gr.Number(label="RA-ESA (cm²)", value=None, precision=1)

                    with gr.Tab("Funktion & Scores"):
                        gr.Markdown("### Funktionelle Tests")
                        with gr.Row():
                            who_fc = gr.Dropdown(label="WHO-FC", choices=["", "I", "II", "III", "IV"], value="")
                            sixmwd_m = gr.Number(label="6MWD (m)", value=None, precision=0)
                            sbp = gr.Number(label="SBP (mmHg)", value=None, precision=0)
                        gr.Markdown("### HFpEF-Wahrscheinlichkeit (kontinuierliches Modell)")
                        gr.Markdown(
                            "Quelle/Referenz: https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.118.034646  \n"
                            "Dieses Tool nutzt ein kontinuierliches Wahrscheinlichkeitsmodell (Alter, BMI, E/e', PASP, Vorhofflimmern). "
                            "BMI wird intern bei 50 kg/m² gedeckelt."
                        )

                    with gr.Tab("Module & Export"):
                        all_blocks = getattr(textdb, "ALL_BLOCKS", None) or {}
                        p_ids = sorted([k for k in all_blocks.keys() if isinstance(k, str) and k.startswith("P")])
                        be_ids = sorted([k for k in all_blocks.keys() if isinstance(k, str) and k.startswith("BE")])
                        c_ids = sorted([k for k in all_blocks.keys() if isinstance(k, str) and k.startswith("C")])

                        def _make_choices(ids: List[str]) -> List[str]:
                            out = []
                            for bid in ids:
                                try:
                                    b = textdb.get_block(bid)
                                    title = b.title if b else bid
                                except Exception:
                                    title = bid
                                out.append(f"{bid} – {title}")
                            return out

                        with gr.Row():
                            modules_p = gr.CheckboxGroup(label="P-Module (Procedere)", choices=_make_choices(p_ids), value=[])
                            modules_be = gr.CheckboxGroup(label="BE-Module (Sicherheit/Zusatz)", choices=_make_choices(be_ids), value=[])
                        modules_c = gr.CheckboxGroup(label="C-Module (Kontext/Trigger)", choices=_make_choices(c_ids), value=[])

                        gr.Markdown("### Export")
                        with gr.Row():
                            btn_pdf_doc = gr.Button("📄 PDF Arztbefund", variant="secondary")
                            btn_pdf_pat = gr.Button("📄 PDF Patientenbericht", variant="secondary")
                            btn_txt_doc = gr.Button("🧾 TXT Arztbefund", variant="secondary")
                            btn_txt_pat = gr.Button("🧾 TXT Patientenbericht", variant="secondary")
                        file_out = gr.File(label="Download-Datei")

                    with gr.Tab("Debug"):
                        debug_out = gr.Textbox(label="Engine Debug", lines=18, value="")

                # Buttons at bottom (requested)
                gr.Markdown("---")
                with gr.Row():
                    btn_example_b = gr.Button("🎲 Beispiel laden (random)", variant="secondary")
                    btn_generate_b = gr.Button("✅ Befund erstellen / aktualisieren", variant="primary")
                    btn_save_b = gr.Button("💾 Fall speichern", variant="secondary")
                    up_load_b = gr.UploadButton("📂 Fall laden", file_types=[".json"], variant="secondary")

            with gr.Column(scale=5):
                dashboard = gr.HTML(value=render_dashboard_html({"bundle": "—", "primary_dx": "—"}))
                with gr.Tabs():
                    with gr.Tab("Arztbefund"):
                        out_doc = gr.Textbox(label="Befund (copy-ready)", lines=26)
                    with gr.Tab("Patientenbericht"):
                        out_pat = gr.Textbox(label="Patientenbericht (verständliche Sprache)", lines=26)
                    with gr.Tab("Intern"):
                        out_int = gr.Textbox(label="Intern", lines=18)
                    with gr.Tab("JSON"):
                        out_json = gr.Textbox(label="Fall JSON", lines=12)

        # -----------------------------
        # Conditional visibility
        # -----------------------------
        def _update_anemia_visibility(hb_val, sex_val):
            hb_f = _to_float(hb_val)
            anemia = anemia_present(hb_f, sex_val)
            return gr.update(visible=(anemia is True))

        def _update_ild_visibility(ild):
            vis = bool(ild)
            return (
                gr.update(visible=vis),
                gr.update(visible=vis),
                gr.update(visible=vis),
                gr.update(visible=vis),
            )

        hb_g_dl.change(_update_anemia_visibility, inputs=[hb_g_dl, sex], outputs=[anemia_type])
        sex.change(_update_anemia_visibility, inputs=[hb_g_dl, sex], outputs=[anemia_type])

        ild_present.change(_update_ild_visibility, inputs=[ild_present], outputs=[ild_type, ild_extent, ild_histology, ild_fibrosis_clinic])

        virology_positive.change(lambda v: gr.update(visible=bool(v)), inputs=[virology_positive], outputs=[virology_details])
        immunology_positive.change(lambda v: gr.update(visible=bool(v)), inputs=[immunology_positive], outputs=[immunology_details])
        abdomen_sono_done.change(lambda v: gr.update(visible=bool(v)), inputs=[abdomen_sono_done], outputs=[abdomen_findings])
        ltot_present.change(lambda v: gr.update(visible=bool(v)), inputs=[ltot_present], outputs=[ltot_flow_l_min])
        vasoreactivity_done.change(lambda v: gr.update(visible=bool(v)), inputs=[vasoreactivity_done], outputs=[ino_responder])

        # -----------------------------
        # Helper: checkbox labels -> ids
        # -----------------------------
        def _labels_to_ids(labels: List[str]) -> List[str]:
            ids: List[str] = []
            for it in labels or []:
                it = str(it)
                if " – " in it:
                    bid = it.split(" – ", 1)[0].strip()
                else:
                    bid = it.strip()
                if bid and bid not in ids:
                    ids.append(bid)
            return ids

        # -----------------------------
        # Gather UI values into dict
        # -----------------------------
        def _gather_ui(*vals) -> Dict[str, Any]:
            # The order matches generate_inputs below
            (
                last_name_v, first_name_v, sex_v, birthdate_v, exam_date_v, exam_type_v, setting_v,
                height_cm_v, weight_kg_v, story_v,

                hb_v, anemia_type_v, thrombos_v, crp_v, leukos_v, inr_v, quick_v, ptt_v, krea_v, egfr_v, entresto_v, bnp_kind_v, bnp_value_v,
                virology_pos_v, immunology_pos_v, virology_details_v, immunology_details_v,
                abdomen_sono_v, portal_htn_v, abdomen_findings_v,

                oxygen_mode_v, oxygen_flow_v, co_method_v,
                rap_v, spap_v, dpap_v, mpap_v, pawp_v, co_v, ci_v, hr_v, pvr_v,
                step_up_present_v, step_up_from_to_v,
                vasoreactivity_done_v, ino_responder_v,
                exercise_done_v, ex_co_v, ex_mpap_v, ex_pawp_v, ex_spap_v, mpap_co_slope_v, pawp_co_slope_v,

                prev_rhk_date_v, prev_rhk_summary_v, prev_mpap_v, prev_pawp_v, prev_ci_v, prev_pvr_v,

                lufu_done_v, dlco_v, ltot_present_v, ltot_flow_v, lufu_summary_v,
                vq_done_v, vq_pos_v, ct_chronic_v, mosaic_v, ild_present_v, emphysema_v,
                ild_type_v, ild_extent_v, ild_histology_v, ild_fibrosis_clinic_v,

                lvef_v, la_enlarged_v, af_v, e_over_eprime_v, pasp_echo_v, ivc_diam_v, ivc_collapse_v, sprime_v, ra_area_v,

                who_fc_v, sixmwd_v, sbp_v,

                modules_p_v, modules_be_v, modules_c_v,
            ) = vals

            anemia_type_clean = (anemia_type_v or "").strip() or None

            ui: Dict[str, Any] = {
                "last_name": (last_name_v or "").strip(),
                "first_name": (first_name_v or "").strip(),
                "sex": (sex_v or "").strip(),
                "birthdate": (birthdate_v or "").strip(),
                "exam_date": (exam_date_v or "").strip(),
                "exam_type": exam_type_v,
                "setting": setting_v,
                "height_cm": _to_float(height_cm_v),
                "weight_kg": _to_float(weight_kg_v),
                "story": (story_v or "").strip(),

                "hb_g_dl": _to_float(hb_v),
                "anemia_type": anemia_type_clean,
                "thrombos": _to_float(thrombos_v),
                "crp": _to_float(crp_v),
                "leukos": _to_float(leukos_v),
                "inr": _to_float(inr_v),
                "quick": _to_float(quick_v),
                "ptt": _to_float(ptt_v),
                "krea": _to_float(krea_v),
                "egfr": _to_float(egfr_v),
                "entresto": bool(entresto_v),
                "bnp_kind": bnp_kind_v,
                "bnp_value": _to_float(bnp_value_v),

                "virology_positive": bool(virology_pos_v),
                "immunology_positive": bool(immunology_pos_v),
                "virology_details": (virology_details_v or "").strip(),
                "immunology_details": (immunology_details_v or "").strip(),

                "abdomen_sono_done": bool(abdomen_sono_v),
                "portal_hypertension": bool(portal_htn_v),
                "abdomen_findings": (abdomen_findings_v or "").strip(),

                "oxygen_mode": oxygen_mode_v,
                "oxygen_flow_l_min": _to_float(oxygen_flow_v),
                "co_method": co_method_v,

                "rap": _to_float(rap_v),
                "spap": _to_float(spap_v),
                "dpap": _to_float(dpap_v),
                "mpap": _to_float(mpap_v),
                "pawp": _to_float(pawp_v),
                "co": _to_float(co_v),
                "ci": _to_float(ci_v),
                "hr": _to_float(hr_v),
                "pvr": _to_float(pvr_v),

                "step_up_present": step_up_present_v,
                "step_up_from_to": (step_up_from_to_v or "").strip(),

                "vasoreactivity_done": bool(vasoreactivity_done_v),
                "ino_responder": (True if str(ino_responder_v).strip().lower()=="ja" else (False if str(ino_responder_v).strip().lower()=="nein" else None)),

                "exercise_done": bool(exercise_done_v),
                "ex_co": _to_float(ex_co_v),
                "ex_mpap": _to_float(ex_mpap_v),
                "ex_pawp": _to_float(ex_pawp_v),
                "ex_spap": _to_float(ex_spap_v),
                "mpap_co_slope": _to_float(mpap_co_slope_v),
                "pawp_co_slope": _to_float(pawp_co_slope_v),

                "prev_rhk_date": (prev_rhk_date_v or "").strip(),
                "prev_rhk_summary": (prev_rhk_summary_v or "").strip(),
                "prev_mpap": _to_float(prev_mpap_v),
                "prev_pawp": _to_float(prev_pawp_v),
                "prev_ci": _to_float(prev_ci_v),
                "prev_pvr": _to_float(prev_pvr_v),

                "lufu_done": bool(lufu_done_v),
                "dlco_sb": _to_float(dlco_v),
                "ltot_present": bool(ltot_present_v),
                "ltot_flow_l_min": _to_float(ltot_flow_v),
                "lufu_summary": (lufu_summary_v or "").strip(),

                "vq_done": bool(vq_done_v),
                "vq_positive": bool(vq_pos_v),
                "ct_chronic_thrombo": bool(ct_chronic_v),
                "mosaic_perfusion": bool(mosaic_v),
                "ild_present": bool(ild_present_v),
                "emphysema_present": bool(emphysema_v),
                "ild_type": (ild_type_v or "").strip() or None,
                "ild_extent": (ild_extent_v or "").strip() or None,
                "ild_histology": (ild_histology_v or "").strip() or None,
                "ild_fibrosis_clinic": (ild_fibrosis_clinic_v or "").strip() or None,

                "lvef": _to_float(lvef_v),
                "la_enlarged": bool(la_enlarged_v),
                "atrial_fibrillation": bool(af_v),
                "e_over_eprime": _to_float(e_over_eprime_v),
                "pasp_echo": _to_float(pasp_echo_v),
                "ivc_diam_mm": _to_float(ivc_diam_v),
                "ivc_collapse_pct": _to_float(ivc_collapse_v),
                "sprime_cm_s": _to_float(sprime_v),
                "ra_area_cm2": _to_float(ra_area_v),

                "who_fc": (who_fc_v or "").strip(),
                "sixmwd_m": _to_float(sixmwd_v),
                "sbp": _to_float(sbp_v),

                "modules": _labels_to_ids((modules_p_v or []) + (modules_be_v or []) + (modules_c_v or [])),
            }
            return ui

        # -----------------------------
        # Core generate (single source of truth)
        # -----------------------------
        def _generate(*vals):
            ui = _gather_ui(*vals)
            case = build_case(ui)

            # Build env for engine
            env = dict(case.get("derived") or {})
            # Add some raw values for rule expressions:
            env.update({
                "pawp": _to_float(ui.get("pawp")),
                "co": _to_float(ui.get("co")),
                "spap": _to_float(ui.get("spap")),
                "dpap": _to_float(ui.get("dpap")),
                "lvef": _to_float(ui.get("lvef")),
                "la_enlarged": bool(ui.get("la_enlarged")),
                "atrial_fibrillation": bool(ui.get("atrial_fibrillation")),
                "vasoreactivity_done": bool(ui.get("vasoreactivity_done")),
                "ino_responder": ui.get("ino_responder"),
                "ild_type": ui.get("ild_type"),
                "ild_extent": ui.get("ild_extent"),
                "ild_histology": ui.get("ild_histology"),
                "ild_fibrosis_clinic": ui.get("ild_fibrosis_clinic"),
                "virology_details": ui.get("virology_details"),
                "immunology_details": ui.get("immunology_details"),
                "abdomen_findings": ui.get("abdomen_findings"),
            })

            thr = (rulebook.get("thresholds") or {}) if isinstance(rulebook, dict) else {}
            # Apply thresholds to env-derived flags (so rulebook can change)
            try:
                mpap_thr = float(thr.get("mpap_ph_mmHg", 20))
                pawp_thr = float(thr.get("pawp_postcap_mmHg", 15))
                pvr_thr = float(thr.get("pvr_precap_wu", 2.0))
                env["has_ph_rest"] = (env.get("mpap") is not None and float(env["mpap"]) > mpap_thr)
                env["has_postcap"] = (env.get("pawp") is not None and float(env["pawp"]) > pawp_thr)
                env["has_precap_component"] = (env.get("pvr") is not None and float(env["pvr"]) > pvr_thr)
                env["borderline_ph"] = (env.get("mpap") is not None and mpap_thr < float(env["mpap"]) <= mpap_thr + 4 and (env.get("pvr") is None or float(env["pvr"]) <= pvr_thr) and (env.get("pawp") is None or float(env["pawp"]) <= pawp_thr))
            except Exception:
                pass

            # Exercise pattern with thresholds
            if env.get("exercise_done"):
                try:
                    thr_m = float(thr.get("mpap_co_slope_mmHg_per_L_min", 3.0))
                    thr_w = float(thr.get("pawp_co_slope_mmHg_per_L_min", 2.0))
                    env["exercise_pattern"] = classify_exercise_pattern(_to_float(env.get("mpap_co_slope")), _to_float(env.get("pawp_co_slope")), thr_m, thr_w)
                except Exception:
                    pass

            # PVR severity with thresholds
            try:
                mild_ge = float(thr.get("pvr_mild_ge", 2.0))
                mod_ge = float(thr.get("pvr_moderate_ge", 5.0))
                sev_ge = float(thr.get("pvr_severe_ge", 10.0))
                env["pvr_severity"] = pvr_severity(_to_float(env.get("pvr")), mild_ge, mod_ge, sev_ge)
            except Exception:
                pass
            try:
                env["ci_low"] = (env.get("ci") is not None and float(env["ci"]) < float(thr.get("ci_low_lt", 2.0)))
            except Exception:
                pass

            decision = apply_rule_engine(rulebook, env)
            if not decision.bundle:
                decision.bundle = "K04"
                decision.primary_dx = decision.primary_dx or "Unklassifizierte Konstellation (Fallback)"

            doctor_txt, patient_txt, internal_txt = build_doctor_report(case, decision)

            summary = {
                "bundle": decision.bundle,
                "primary_dx": decision.primary_dx,
                "tags": decision.tags,
                "require_fields": decision.require_fields,
                "esc3": case.get("scores", {}).get("esc3"),
                "esc4": case.get("scores", {}).get("esc4"),
                "reveal": case.get("scores", {}).get("reveal"),
                "hfpef_percent": case.get("derived", {}).get("hfpef_percent"),
            }
            dash = render_dashboard_html(summary)
            debug = "Fired rules:\n- " + "\n- ".join(decision.fired_rules) if decision.fired_rules else "—"
            case_json = json.dumps(case, ensure_ascii=False, indent=2)
            return dash, doctor_txt, patient_txt, internal_txt, debug, case_json, case

        # -----------------------------
        # Inputs list (order must match _gather_ui unpacking)
        # -----------------------------
        generate_inputs = [
            last_name, first_name, sex, birthdate, exam_date, exam_type, setting,
            height_cm, weight_kg, story,

            hb_g_dl, anemia_type, thrombos, crp, leukos, inr, quick, ptt, krea, egfr, entresto, bnp_kind, bnp_value,
            virology_positive, immunology_positive, virology_details, immunology_details,
            abdomen_sono_done, portal_hypertension, abdomen_findings,

            oxygen_mode, oxygen_flow_l_min, co_method,
            rap, spap, dpap, mpap, pawp, co, ci, hr, pvr,
            step_up_present, step_up_from_to,
            vasoreactivity_done, ino_responder,
            exercise_done, ex_co, ex_mpap, ex_pawp, ex_spap, mpap_co_slope, pawp_co_slope,

            prev_rhk_date, prev_rhk_summary, prev_mpap, prev_pawp, prev_ci, prev_pvr,

            lufu_done, dlco_sb, ltot_present, ltot_flow_l_min, lufu_summary,
            vq_done, vq_positive, ct_chronic_thrombo, mosaic_perfusion, ild_present, emphysema_present,
            ild_type, ild_extent, ild_histology, ild_fibrosis_clinic,

            lvef, la_enlarged, atrial_fibrillation, e_over_eprime, pasp_echo, ivc_diam_mm, ivc_collapse_pct, sprime_cm_s, ra_area_cm2,

            who_fc, sixmwd_m, sbp,

            modules_p, modules_be, modules_c,
        ]
        generate_outputs = [dashboard, out_doc, out_pat, out_int, debug_out, out_json, state_case]

        # Buttons: generate
        btn_generate.click(_generate, inputs=generate_inputs, outputs=generate_outputs)
        btn_generate_b.click(_generate, inputs=generate_inputs, outputs=generate_outputs)

        # Ultra-interaktiv: jeder Input ändert Outputs
        for comp in generate_inputs:
            comp.change(_generate, inputs=generate_inputs, outputs=generate_outputs)

        # -----------------------------
        # Apply dict -> components
        # -----------------------------
        def _apply_ui_to_components(ui: Dict[str, Any]) -> List[Any]:
            # returns raw values for each component in generate_inputs order
            # modules selections cleared on load/example (user can re-select)
            values = []
            def g(key, default=None):
                return ui.get(key, default)
            # Restore module selections (ids -> checkbox labels) if present
            mod_ids = ui.get('modules', []) or []
            if isinstance(mod_ids, str):
                mod_ids = [mod_ids]
            if not isinstance(mod_ids, list):
                mod_ids = []
            def _choice_map(choices):
                m = {}
                for ch in (choices or []):
                    try:
                        sid = str(ch).split(' – ', 1)[0].strip()
                    except Exception:
                        sid = str(ch).strip()
                    if sid:
                        m[sid] = ch
                return m
            p_map = _choice_map(getattr(modules_p, 'choices', []) )
            be_map = _choice_map(getattr(modules_be, 'choices', []) )
            c_map = _choice_map(getattr(modules_c, 'choices', []) )
            p_sel = [p_map[i] for i in mod_ids if i in p_map]
            be_sel = [be_map[i] for i in mod_ids if i in be_map]
            c_sel = [c_map[i] for i in mod_ids if i in c_map]
            values.extend([
                g("last_name",""), g("first_name",""), g("sex",""), g("birthdate",""), g("exam_date",_now_iso()),
                g("exam_type","Initial-RHK"), g("setting","Ruhe"),
                g("height_cm",None), g("weight_kg",None), g("story",""),
                g("hb_g_dl",None), g("anemia_type",""), g("thrombos",None), g("crp",None), g("leukos",None),
                g("inr",None), g("quick",None), g("ptt",None), g("krea",None), g("egfr",None),
                bool(g("entresto",False)), g("bnp_kind","NT-proBNP"), g("bnp_value",None),
                bool(g("virology_positive",False)), bool(g("immunology_positive",False)), g("virology_details",""), g("immunology_details",""),
                bool(g("abdomen_sono_done",False)), bool(g("portal_hypertension",False)), g("abdomen_findings",""),
                g("oxygen_mode","Raumluft"), g("oxygen_flow_l_min",None), g("co_method","Thermodilution"),
                g("rap",None), g("spap",None), g("dpap",None), g("mpap",None), g("pawp",None), g("co",None), g("ci",None), g("hr",None), g("pvr",None),
                g("step_up_present","unklar"), g("step_up_from_to",""),
                bool(g("vasoreactivity_done",False)), ("ja" if g("ino_responder",None) is True else ("nein" if g("ino_responder",None) is False else "")),
                bool(g("exercise_done",False)), g("ex_co",None), g("ex_mpap",None), g("ex_pawp",None), g("ex_spap",None), g("mpap_co_slope",None), g("pawp_co_slope",None),
                g("prev_rhk_date",""), g("prev_rhk_summary",""), g("prev_mpap",None), g("prev_pawp",None), g("prev_ci",None), g("prev_pvr",None),
                bool(g("lufu_done",False)), g("dlco_sb",None), bool(g("ltot_present",False)), g("ltot_flow_l_min",None), g("lufu_summary",""),
                bool(g("vq_done",False)), bool(g("vq_positive",False)), bool(g("ct_chronic_thrombo",False)), bool(g("mosaic_perfusion",False)), bool(g("ild_present",False)), bool(g("emphysema_present",False)),
                g("ild_type",""), g("ild_extent",""), g("ild_histology",""), g("ild_fibrosis_clinic",""),
                g("lvef",None), bool(g("la_enlarged",False)), bool(g("atrial_fibrillation",False)), g("e_over_eprime",None), g("pasp_echo",None), g("ivc_diam_mm",None), g("ivc_collapse_pct",None), g("sprime_cm_s",None), g("ra_area_cm2",None),
                g("who_fc",""), g("sixmwd_m",None), g("sbp",None),
                p_sel, be_sel, c_sel,
            ])
            return values

        # Example (random) – set inputs + run generate once (no multi-trigger)
        def _load_example():
            ui = generate_random_example_ui()
            vals = _apply_ui_to_components(ui)
            dash, doc, pat, intern, dbg, jsn, case = _generate(*vals)
            return (*vals, dash, doc, pat, intern, dbg, jsn, case)

        # Load JSON file – set inputs + run generate once
        def _load_case(file: gr.File):
            if file is None:
                ui = {}
            else:
                try:
                    path = file.name  # type: ignore[attr-defined]
                except Exception:
                    path = str(file)
                try:
                    raw = json.loads(Path(path).read_text(encoding="utf-8"))
                    ui = raw.get("ui") if isinstance(raw, dict) else {}
                    if not isinstance(ui, dict):
                        ui = {}
                except Exception:
                    ui = {}
            vals = _apply_ui_to_components(ui)
            dash, doc, pat, intern, dbg, jsn, case = _generate(*vals)
            return (*vals, dash, doc, pat, intern, dbg, jsn, case)

        # Outputs for load functions: first all inputs (to populate UI), then outputs
        load_outputs = generate_inputs + generate_outputs

        btn_example.click(_load_example, inputs=[], outputs=load_outputs)
        btn_example_b.click(_load_example, inputs=[], outputs=load_outputs)

        up_load.upload(_load_case, inputs=[up_load], outputs=load_outputs)
        up_load_b.upload(_load_case, inputs=[up_load_b], outputs=load_outputs)

        # Save case -> file
        def _save_case(case_state: Any):
            if not isinstance(case_state, dict):
                return None
            return export_json(case_state)

        btn_save.click(_save_case, inputs=[state_case], outputs=[file_out])
        btn_save_b.click(_save_case, inputs=[state_case], outputs=[file_out])

        # Export
        def _export_pdf_doc(doc: str):
            p = export_pdf(doc, "arztbefund")
            return p
        def _export_pdf_pat(pat: str):
            p = export_pdf(pat, "patient")
            return p
        def _export_txt_doc(doc: str):
            return export_txt(doc, "arztbefund")
        def _export_txt_pat(pat: str):
            return export_txt(pat, "patient")

        btn_pdf_doc.click(_export_pdf_doc, inputs=[out_doc], outputs=[file_out])
        btn_pdf_pat.click(_export_pdf_pat, inputs=[out_pat], outputs=[file_out])
        btn_txt_doc.click(_export_txt_doc, inputs=[out_doc], outputs=[file_out])
        btn_txt_pat.click(_export_txt_pat, inputs=[out_pat], outputs=[file_out])

    return demo

# -----------------------------
# Embedded default rulebook YAML (written on first run)
# -----------------------------
DEFAULT_RULEBOOK_YAML = r"""meta:
  name: "RHK-Befundassistent Regelwerk"
  version: "18.0.0"
  guideline_reference: "ESC/ERS Pulmonary Hypertension Guideline (2022) – core definitions + pragmatic clinical pathways"
  notes: |
    Dieses Regelwerk ist deklarativ und kann erweitert werden.
    Regeln werden nach priority (hoch->niedrig) ausgewertet. Bei stop_processing endet die Auswertung.

thresholds:
  mpap_ph_mmHg: 20
  pawp_postcap_mmHg: 15
  pvr_precap_wu: 2.0
  mpap_co_slope_mmHg_per_L_min: 3.0
  pawp_co_slope_mmHg_per_L_min: 2.0
  pvr_mild_ge: 2.0
  pvr_moderate_ge: 5.0
  pvr_severe_ge: 10.0
  ci_low_lt: 2.0

rules:
  # ------------------------------------------------------------
  # 1) "Hard" decision paths with highest priority
  # ------------------------------------------------------------
  - id: R_SHUNT_STEPUP
    priority: 1000
    when: "step_up_present == True"
    actions:
      set_bundle: "K16"
      set_primary_dx: "Shunt-Verdacht (Links-Rechts-Shunt) aufgrund Sättigungssprung"
      add_modules: ["P09"]
      add_recommendations:
        - "Weiterführende Shunt-Diagnostik (Echo/TEE/Bubble, ggf. Kardio-MRT/CT je nach Fragestellung) und Quantifizierung erwägen."
      stop_processing: true

  - id: R_VASO_POS
    priority: 950
    when: "vasoreactivity_done == True and ino_responder == True"
    actions:
      set_bundle: "K17"
      set_primary_dx: "Präkapilläre PH, Vasoreagibilitätstest positiv"
      add_modules: ["P11"]
      add_recommendations:
        - "Therapiepfad bei positiver Vasoreagibilität gemäß Leitlinie/zentraler SOP erwägen; engmaschige Verlaufskontrollen."
      stop_processing: true

  - id: R_VASO_NEG
    priority: 940
    when: "vasoreactivity_done == True and ino_responder == False"
    actions:
      set_bundle: "K18"
      set_primary_dx: "Präkapilläre PH, Vasoreagibilitätstest negativ"
      add_modules: ["P11"]
      add_recommendations:
        - "Kein Vasoreagibilitäts-spezifischer Therapiepfad; weiteres Vorgehen nach Ätiologie und Risikoprofil."
      stop_processing: true

  # ------------------------------------------------------------
  # 2) Exercise logic (when exercise data present)
  # ------------------------------------------------------------
  - id: R_EX_NO_PH_NORMAL
    priority: 850
    when: "exercise_done == True and has_ph_rest == False and exercise_pattern == 'normal'"
    actions:
      set_bundle: "K01"
      set_primary_dx: "Keine PH in Ruhe, Belastungsreaktion unauffällig"
      add_recommendations:
        - "Bei persistierender Belastungsdyspnoe weitere Abklärung nach klinischem Kontext (kardiologisch/pneumologisch, ggf. Spiroergometrie)."
      stop_processing: true

  - id: R_EX_LINKSHEART
    priority: 840
    when: "exercise_done == True and has_ph_rest == False and exercise_pattern == 'linkskardial'"
    actions:
      set_bundle: "K02"
      set_primary_dx: "Keine PH in Ruhe, aber pathologische Belastungsreaktion mit linkskardialer Komponente"
      add_modules: ["P09"]
      add_recommendations:
        - "Kardiologische Abklärung/Optimierung (HFpEF/Diastolik, Klappenvitien, Rhythmus) empfohlen."
        - "PH-spezifische Therapie ist in dieser Konstellation in der Regel nicht primär indiziert."
      stop_processing: true

  - id: R_EX_PULMVASC
    priority: 830
    when: "exercise_done == True and (has_ph_rest == False or borderline_ph == True) and exercise_pattern == 'pulmvasc'"
    actions:
      set_bundle: "K03"
      set_primary_dx: "Auffällige pulmonalvaskuläre Belastungsreaktion (Belastungs-PH/early pulmonary vascular disease)"
      add_modules: ["P01","P11"]
      add_recommendations:
        - "Komplettierung der PH-Diagnostik (V/Q, CT/HRCT, Lufu/DLCO, Echo, Labor je nach Kontext) empfohlen."
      stop_processing: true

  # ------------------------------------------------------------
  # 3) Rest PH classification (ESC/ERS definitions)
  # ------------------------------------------------------------
  - id: R_NO_PH_REST
    priority: 700
    when: "has_ph_rest == False"
    actions:
      set_bundle: "K01"
      set_primary_dx: "Kein Hinweis auf PH in Ruhe"
      stop_processing: true

  - id: R_IPCPH
    priority: 680
    when: "has_ph_rest == True and has_postcap == True and has_precap_component == False"
    actions:
      set_bundle: "K14"
      set_primary_dx: "Postkapilläre PH (Gruppe 2, linksatriale/linksventrikuläre Genese führend)"
      add_modules: ["P09"]
      add_recommendations:
        - "Im Vordergrund: kardiologische Therapieoptimierung (Volumenmanagement, Blutdruck, Rhythmus, Klappentherapie nach Kontext)."
      stop_processing: true

  - id: R_CPCPH
    priority: 670
    when: "has_ph_rest == True and has_postcap == True and has_precap_component == True"
    actions:
      set_bundle: "K15"
      set_primary_dx: "Kombinierte prä- und postkapilläre PH (CpcPH)"
      add_modules: ["P01","P09"]
      add_recommendations:
        - "HF-/Linksherz-Optimierung (Dekongestion, Risikofaktoren, Rhythmus) UND DD-Abklärung der präkapillären Komponente im spezialisierten Setting."
      stop_processing: true

  - id: R_PRECAP_MILD
    priority: 660
    when: "has_ph_rest == True and has_postcap == False and has_precap_component == True and pvr_severity == 'mild'"
    actions:
      set_bundle: "K05"
      set_primary_dx: "Präkapilläre PH (leichtgradig)"
      add_modules: ["P01","P11"]
      add_recommendations:
        - "Differenzialdiagnostik (Gruppe 1 vs 3 vs 4 u.a.) gemäß Basisdiagnostik komplettieren."
      stop_processing: true

  - id: R_PRECAP_MOD
    priority: 650
    when: "has_ph_rest == True and has_postcap == False and has_precap_component == True and pvr_severity == 'moderate'"
    actions:
      set_bundle: "K06"
      set_primary_dx: "Präkapilläre PH (mittelgradig)"
      add_modules: ["P01","P11"]
      add_recommendations:
        - "Therapieeinleitung/Anbindung an PH-Zentrum je nach Kontext/Risiko erwägen; Verlaufskontrolle planen."
      stop_processing: true

  - id: R_PRECAP_SEV
    priority: 640
    when: "has_ph_rest == True and has_postcap == False and has_precap_component == True and (pvr_severity == 'severe' or ci_low == True)"
    actions:
      set_bundle: "K07"
      set_primary_dx: "Präkapilläre PH (hochgradig) mit Hinweis auf limitierte Förderleistung"
      add_modules: ["P02","P11"]
      add_recommendations:
        - "Engmaschige Kontrolle; bei Stauung Priorisierung Dekongestion. Therapieeskalation im spezialisierten Setting/Board diskutieren."
      stop_processing: true

  - id: R_BORDERLINE
    priority: 600
    when: "borderline_ph == True or (has_ph_rest == True and has_precap_component == False and has_postcap == False)"
    actions:
      set_bundle: "K04"
      set_primary_dx: "Grenzwertige / unklassifizierte Hämodynamik"
      add_modules: ["P11"]
      add_recommendations:
        - "Interpretation im klinischen Kontext; ggf. Provokation/Verlauf und Komplettierung Zusatzdiagnostik."
      stop_processing: true

  # ------------------------------------------------------------
  # 4) Etiology prioritisation (Guideline-inspired differential groups)
  # These rules do NOT override bundle if already set, but add tags/recommendations/modules.
  # ------------------------------------------------------------
  - id: R_CTEPH_PATH
    priority: 500
    when: "cteph_markers == True and (has_precap_component == True or exercise_pattern == 'pulmvasc')"
    actions:
      add_tags: ["DD: Gruppe 4 (CTEPD/CTEPH)"]
      add_modules: ["P10"]
      add_recommendations:
        - "CTEPD/CTEPH-Diagnostikpfad: V/Q-Szintigraphie (falls ausstehend), CT-PA/Angiographie je nach Vorbefund; Vorstellung im CTEPH-Board (PEA/BPA) erwägen."

  - id: R_GROUP3_ILD
    priority: 490
    when: "ild_present == True and (has_precap_component == True or exercise_pattern == 'pulmvasc')"
    actions:
      add_tags: ["DD: Gruppe 3 (Lungenerkrankung/ILD)"]
      add_modules: ["P08","P12"]
      add_recommendations:
        - "Hinweis auf Lungenerkrankung/ILD: pneumologische Mitbeurteilung, HRCT/ILD-Board/Fibroseambulanz je nach Kontext; Hypoxie/OSA/OHS mitbeurteilen."

  - id: R_GROUP3_EMPHY
    priority: 485
    when: "emphysema_present == True and (has_precap_component == True or exercise_pattern == 'pulmvasc')"
    actions:
      add_tags: ["DD: Gruppe 3 (COPD/Emphysem)"]
      add_modules: ["P12"]
      add_recommendations:
        - "Bei Emphysem/COPD-Konstellation: Optimierung pneumologischer Therapie, Abklärung Hypoxämie/LTOT-Indikation, Reha."

  - id: R_GROUP2_HFPEF_LA
    priority: 480
    when: "la_enlarged == True and lvef_preserved == True"
    actions:
      add_tags: ["Hinweis: HFpEF/diastolische Komponente"]
      add_recommendations:
        - "Bei LA-Vergrößerung und erhaltener LVEF: HFpEF/diastolische Dysfunktion mitbeurteilen (Echo, Klinik, Scores)."

  - id: R_GROUP2_HFPEF_PH
    priority: 470
    when: "la_enlarged == True and lvef_preserved == True and has_ph_rest == True"
    actions:
      add_tags: ["Konstellation: HFpEF + PH möglich"]
      add_recommendations:
        - "Konstellation vereinbar mit HFpEF+PH; Fokus auf Volumenstatus, Blutdruck, Rhythmus und Komorbiditäten."

  - id: R_GROUP2_HFPEF_CPC
    priority: 465
    when: "la_enlarged == True and lvef_preserved == True and has_ph_rest == True and has_postcap == True and has_precap_component == True"
    actions:
      add_tags: ["Konstellation: HFpEF + CpcPH möglich"]
      add_recommendations:
        - "Bei CpcPH im HFpEF-Setting: neben Linksherz-Management präkapilläre DD (Gruppe 1/3/4) im spezialisierten Setting aktiv prüfen."

  - id: R_PORTOPULM
    priority: 460
    when: "portal_hypertension == True and high_flow == True"
    actions:
      add_tags: ["DD: Portopulmonale PH / Hyperzirkulation"]
      add_recommendations:
        - "Bei portaler Hypertension/Hyperzirkulation: interdisziplinäre Betreuung (PH-/Leberzentrum); Hochfluss-Ursachen/Anämie/Infekt/Thyreotoxikose prüfen."

  # ------------------------------------------------------------
  # 5) Congestion / IVC / RAP
  # ------------------------------------------------------------
  - id: R_CONGESTION_IVC
    priority: 420
    when: "congestion_likely == True"
    actions:
      add_tags: ["Stauung: zentralvenös wahrscheinlich"]
      add_modules: ["P02"]
      add_recommendations:
        - "Bei Hinweis auf zentrale Stauung: Dekongestion/Diuretikamanagement und Kontrolle Nierenfunktion/Elekrolyte erwägen."

  # ------------------------------------------------------------
  # 6) Anemia logic (Hb low -> ask/act)
  # ------------------------------------------------------------
  - id: R_ANEMIA_REQUIRE_TYPE
    priority: 400
    when: "anemia_present == True and anemia_type is None"
    actions:
      require_fields: ["anemia_type"]
      add_recommendations:
        - "Hb erniedrigt: Anämie-Typisierung (mikro-/normo-/makrozytär) und Ursachenabklärung empfohlen."

  - id: R_ANEMIA_MICRO
    priority: 390
    when: "anemia_present == True and anemia_type == 'micro'"
    actions:
      add_recommendations:
        - "Mikrozytäre Anämie: Eisenstatus (Ferritin/TSAT) und Blutungsquellen (GI/Gyn) abklären; Substitution gemäß Standard."

  - id: R_ANEMIA_MACRO
    priority: 385
    when: "anemia_present == True and anemia_type == 'macro'"
    actions:
      add_recommendations:
        - "Makrozytäre Anämie: B12/Folat, Leber, Alkohol, Hämatologie-DD prüfen; Therapie gemäß Ursache."

  - id: R_ANEMIA_NORMO
    priority: 380
    when: "anemia_present == True and anemia_type == 'normo'"
    actions:
      add_recommendations:
        - "Normozytäre Anämie: Entzündung/Niere/Chronik/DD prüfen; ggf. Eisenstatus, Retikulozyten, Hämolyseparameter."

  # ------------------------------------------------------------
  # 7) Entresto / BNP logic
  # ------------------------------------------------------------
  - id: R_ENTRESTO_BNP_NOTE
    priority: 360
    when: "entresto == True and bnp_marker == 'BNP'"
    actions:
      add_recommendations:
        - "Unter Sacubitril/Valsartan (ARNI) kann BNP erhöht/geringer interpretierbar sein; NT-proBNP ist in der Regel besser verwertbar."

  # ------------------------------------------------------------
  # 8) ILD detail requirements
  # ------------------------------------------------------------
  - id: R_ILD_DETAILS
    priority: 340
    when: "ild_present == True"
    actions:
      require_fields: ["ild_type","ild_extent","ild_histology","ild_fibrosis_clinic"]
      add_recommendations:
        - "ILD: Typ/Ausmaß/histologische Sicherung/Anbindung an Fibroseambulanz dokumentieren (Konsequenzen für Gruppe-3-DD und Therapiepfad)."

  # ------------------------------------------------------------
  # 9) Infection/Immunology detail requirements
  # ------------------------------------------------------------
  - id: R_VIROLOGY_DETAILS
    priority: 330
    when: "virology_positive == True"
    actions:
      require_fields: ["virology_details"]

  - id: R_IMMUNOLOGY_DETAILS
    priority: 325
    when: "immunology_positive == True"
    actions:
      require_fields: ["immunology_details"]

  # ------------------------------------------------------------
  # 10) Abdomen sono details
  # ------------------------------------------------------------
  - id: R_ABDOMEN_DETAILS
    priority: 315
    when: "abdomen_sono_done == True"
    actions:
      require_fields: ["abdomen_findings"]

  # ------------------------------------------------------------
  # 11) LTOT details
  # ------------------------------------------------------------
  - id: R_LTOT_DETAILS
    priority: 310
    when: "ltot_present == True and ltot_flow_l_min is None"
    actions:
      require_fields: ["ltot_flow_l_min"]

  # ------------------------------------------------------------
  # 12) HFpEF score based recommendation (non-prescriptive)
  # ------------------------------------------------------------
  - id: R_HFPEF_SCORE_LIKELY
    priority: 280
    when: "hfpef_prob is not None and hfpef_prob >= 0.8"
    actions:
      add_tags: ["HFpEF-Scores: hohe Wahrscheinlichkeit"]
      add_recommendations:
        - "HFpEF-Score: hohe Wahrscheinlichkeit für diastolische Funktionsstörung – kardiologische Therapieoptimierung und Komorbiditätsmanagement priorisieren."
        - "SGLT2-Inhibitoren können bei HFpEF (je nach Kontraindikationen) erwogen werden; Umsetzung gemäß Leitlinie/Fachinformation und klinischem Setting."

  - id: R_HFPEF_SCORE_POSSIBLE
    priority: 270
    when: "hfpef_prob is not None and hfpef_prob >= 0.5 and hfpef_prob < 0.8"
    actions:
      add_tags: ["HFpEF-Scores: mögliche Wahrscheinlichkeit"]
      add_recommendations:
        - "HFpEF-Score: relevante Wahrscheinlichkeit – diastolische Komponente in Diagnostik/Management berücksichtigen."

  # ------------------------------------------------------------
  # 13) Missing core hemodynamics for classification
  # ------------------------------------------------------------
  - id: R_REQUIRE_MPAP
    priority: 200
    when: "mpap is None and (spap is not None and dpap is not None)"
    actions:
      # mpap can be calculated, so no require; note only
      add_tags: ["mPAP wurde aus sPAP/dPAP berechnet"]

  - id: R_REQUIRE_PAWP
    priority: 190
    when: "has_ph_rest == True and pawp is None"
    actions:
      require_fields: ["pawp"]

  - id: R_REQUIRE_CO_FOR_PVR
    priority: 185
    when: "has_ph_rest == True and pvr is None"
    actions:
      require_fields: ["co"]

  - id: R_REQUIRE_BSA_FOR_CI
    priority: 180
    when: "ci is None and co is not None and bsa is None"
    actions:
      require_fields: ["height_cm","weight_kg"]
# ------------------------------------------------------------
  # 14) Exercise adaptation type (homeometric vs heterometric) – heuristic
  # ------------------------------------------------------------
  - id: R_EX_ADAPT_HOMEOMETRIC
    priority: 260
    when: "exercise_done == True and delta_spap is not None and delta_spap >= 30 and ci_peak is not None and ci is not None and ci_peak >= ci"
    actions:
      add_tags: ["RV-Reaktion unter Belastung: eher homeometrische Adaptation (heuristisch)"]
      add_recommendations:
        - "Belastungsreaktion mit deutlichem ΔsPAP bei erhaltener/verbesserter Förderleistung kann auf eher homeometrischen Adaptationstyp hinweisen (heuristische Einordnung)."

  - id: R_EX_ADAPT_HETEROMETRIC
    priority: 255
    when: "exercise_done == True and delta_spap is not None and delta_spap >= 30 and ci_peak is not None and ci is not None and ci_peak < ci"
    actions:
      add_tags: ["RV-Reaktion unter Belastung: eher heterometrische Adaptation (heuristisch)"]
      add_recommendations:
        - "Belastungsreaktion mit deutlichem ΔsPAP bei abnehmender Förderleistung kann auf eher heterometrischen Adaptationstyp hinweisen (heuristische Einordnung)."

  # ------------------------------------------------------------
  # 15) Combined DD emphasis
  # ------------------------------------------------------------
  - id: R_DD_GROUP3_AND_4
    priority: 240
    when: "ild_present == True and cteph_markers == True"
    actions:
      add_tags: ["DD: kombinierte Hinweise Gruppe 3 UND Gruppe 4"]
      add_recommendations:
        - "Sowohl ILD- als auch CTEPH-Hinweise: interdisziplinäre Einordnung (PH/ILD/CTEPH-Konferenz) empfohlen."

  # ------------------------------------------------------------
  # 16) If key workup missing, suggest
  # ------------------------------------------------------------
  - id: R_SUGGEST_VQ_IF_PRECAP
    priority: 230
    when: "has_precap_component == True and vq_done != True"
    actions:
      add_recommendations:
        - "Bei präkapillärer Komponente: V/Q-Szintigraphie zum Ausschluss CTEPD/CTEPH wird empfohlen (falls noch nicht erfolgt)."

  - id: R_SUGGEST_LUFU_IF_GROUP3_HINT
    priority: 225
    when: "(ild_present == True or emphysema_present == True or ltot_present == True) and lufu_done != True"
    actions:
      add_recommendations:
        - "Bei Hinweis auf Lungenerkrankung/Hypoxie: Lungenfunktion inkl. DLCO (und BGA) komplettieren (falls noch nicht erfolgt)."
"""

def main() -> None:
    demo = build_demo()
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    demo.launch(server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
