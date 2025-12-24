
# rhk_app.py
"""
RHK-Befundassistent – Hauptprogramm (GUI + Rechner + Text-Engine).

Moderne GUI: Gradio (lokale Web-Oberfläche)

Dieses Script ist eine überarbeitete Version mit Fokus auf:
1) Multi-Modul Setting: Basis (Ruhe) + optional Belastung / Volumenchallenge / Vasoreaktivität
2) Beurteilung & Empfehlung werden robust aus der Textdatenbank gerendert (bevorzugt rhk_textdb.py)
3) Auto-Berechnungen: mPAP, AO_mean, CI, TPG, DPG, PVR, SVR, PVRi/SVRi usw. (sofern Eingaben vorhanden)
4) Stufenoxymetrie: Sättigungssprung wird automatisch erkannt (mit optionalem Override)
5) Zusatzfelder (u.a. Labore/Klinik) + Beispielwerte (Formular ist initial beispielhaft befüllt)

Voraussetzungen
- Python >= 3.9
- gradio (empfohlen: >= 4.x; unterstützt auch 6.x)
- Textbaustein-Datenbank im selben Ordner:
    - bevorzugt: rhk_textdb.py
    - alternativ: Befunddatenbank.py

Start:
    python rhk_app.py
"""

from __future__ import annotations

import json
import copy
import math
import os
import re
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Text-Datenbank importieren
# -----------------------------
# WICHTIG: Wir laden bevorzugt *die Datei neben diesem Script*,
# um Namenskollisionen mit evtl. installierten Modulen zu vermeiden.
#
# Priorität (neu → alt):
#   1) rhk_textdb_api.py   (YAML-backed TextDB: core.yaml + overrides.yaml)
#   2) rhk_textdb.py       (Legacy, rein Python)
#   3) Befunddatenbank.py  (Legacy)
import importlib.util

try:
    _APP_DIR = Path(__file__).resolve().parent
except NameError:
    # Jupyter / IPython / Interactive
    _APP_DIR = Path.cwd()

def _load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Kann Modul nicht laden: {path}")
    mod = importlib.util.module_from_spec(spec)
    # In sys.modules registrieren, damit Unterimporte/Reloads konsistent sind
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

textdb = None
_import_errors: List[str] = []

# 1) Prefer local files in the same folder as this script
for _filename, _modname in (
    ("rhk_textdb_api.py", "rhk_textdb_api"),
    ("rhk_textdb.py", "rhk_textdb"),
    ("Befunddatenbank.py", "Befunddatenbank"),
):
    _p = _APP_DIR / _filename
    if _p.exists():
        try:
            textdb = _load_module_from_path(_p, _modname)
            break
        except Exception as e:
            _import_errors.append(f"{_filename}: {e}")

# 2) Fallback: import by module name (only if local file not found / failed)
if textdb is None:
    for _mod in ("rhk_textdb_api", "rhk_textdb", "Befunddatenbank"):
        try:
            textdb = __import__(_mod)
            break
        except Exception as e:
            _import_errors.append(f"{_mod}: {e}")

if textdb is None:
    raise ImportError(
        "Keine Textbaustein-Datenbank gefunden.\n"
        "Lege 'rhk_textdb_api.py' (empfohlen) oder 'rhk_textdb.py' (Legacy) in denselben Ordner wie dieses Script.\n"
        + "\n".join(_import_errors)
    )


# -----------------------------
# GUI (Gradio)
# -----------------------------

try:
    import gradio as gr
except Exception as e:
    raise RuntimeError(
        "Gradio ist nicht installiert. Installiere es z.B. mit: pip install gradio\n"
        f"Originalfehler: {e}"
    )

# -----------------------------
# Helper: Zahlen / Formatierung / Parsing
# -----------------------------

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, bool):
            return None
        return float(x)
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_intish(v: float, tol: float = 1e-6) -> bool:
    return abs(v - round(v)) < tol


def fmt_num(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "nicht erhoben"
    if _is_intish(v):
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


def fmt_unit(v: Optional[float], unit: str, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "nicht erhoben"
    return f"{fmt_num(v, decimals)} {unit}"


def join_nonempty(parts: List[str], sep: str = " | ") -> str:
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


def _norm_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s or "").strip()
    # " .", " ,"
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    # multiple dots
    s = re.sub(r"\.\s*\.", ".", s)
    return s


class SafeDict(dict):
    """
    Für Textbausteine:
    - fehlende Schlüssel sollen NICHT crashen
    - Default: leerer String (damit optionale Sätze nicht "nicht erhoben" werden)
    """
    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return ""


# -----------------------------
# Rechner
# -----------------------------

@dataclass
class CalcResult:
    value: Optional[float]
    formula: Optional[str] = None


def calc_mean(sys_: Optional[float], dia: Optional[float]) -> CalcResult:
    if sys_ is None or dia is None:
        return CalcResult(None)
    mean = (sys_ + 2.0 * dia) / 3.0
    return CalcResult(mean, formula=f"mean = (sys + 2·dia)/3 = ({sys_} + 2·{dia})/3")


def calc_tpg(mpap: Optional[float], pawp: Optional[float]) -> CalcResult:
    if mpap is None or pawp is None:
        return CalcResult(None)
    return CalcResult(mpap - pawp, formula=f"TPG = mPAP − PAWP = {mpap} − {pawp}")


def calc_dpg(dpap: Optional[float], pawp: Optional[float]) -> CalcResult:
    if dpap is None or pawp is None:
        return CalcResult(None)
    return CalcResult(dpap - pawp, formula=f"DPG = dPAP − PAWP = {dpap} − {pawp}")


def calc_pvr(mpap: Optional[float], pawp: Optional[float], co: Optional[float]) -> CalcResult:
    if mpap is None or pawp is None or co is None or co == 0:
        return CalcResult(None)
    return CalcResult((mpap - pawp) / co, formula=f"PVR = (mPAP − PAWP) / CO = ({mpap} − {pawp})/{co}")


def calc_svr(aom: Optional[float], ram: Optional[float], co: Optional[float]) -> CalcResult:
    if aom is None or ram is None or co is None or co == 0:
        return CalcResult(None)
    return CalcResult((aom - ram) / co, formula=f"SVR = (AO_mean − RA_mean) / CO = ({aom} − {ram})/{co}")


def calc_bsa_dubois(height_cm: Optional[float], weight_kg: Optional[float]) -> CalcResult:
    if height_cm is None or weight_kg is None or height_cm <= 0 or weight_kg <= 0:
        return CalcResult(None)
    bsa = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
    return CalcResult(
        bsa,
        formula=(
            "BSA(DuBois) = 0.007184 · Höhe(cm)^0.725 · Gewicht(kg)^0.425 "
            f"= 0.007184 · {height_cm}^0.725 · {weight_kg}^0.425"
        ),
    )


def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if height_cm is None or weight_kg is None or height_cm <= 0 or weight_kg <= 0:
        return None
    return weight_kg / ((height_cm / 100.0) ** 2)


def calc_ci(co: Optional[float], bsa: Optional[float]) -> CalcResult:
    if co is None or bsa is None or bsa == 0:
        return CalcResult(None)
    return CalcResult(co / bsa, formula=f"CI = CO / BSA = {co}/{bsa}")


def calc_slope(p_rest: Optional[float], co_rest: Optional[float], p_peak: Optional[float], co_peak: Optional[float]) -> CalcResult:
    if p_rest is None or co_rest is None or p_peak is None or co_peak is None:
        return CalcResult(None)
    delta_co = co_peak - co_rest
    if delta_co == 0:
        return CalcResult(None)
    slope = (p_peak - p_rest) / delta_co
    return CalcResult(
        slope,
        formula=(
            "Slope = (P_peak − P_rest) / (CO_peak − CO_rest) "
            f"= ({p_peak} − {p_rest})/({co_peak} − {co_rest})"
        ),
    )


def calc_pvri(mpap: Optional[float], pawp: Optional[float], ci: Optional[float]) -> CalcResult:
    # PVRI = (mPAP-PAWP)/CI  [WU·m²]
    if mpap is None or pawp is None or ci is None or ci == 0:
        return CalcResult(None)
    return CalcResult((mpap - pawp) / ci, formula=f"PVRI = (mPAP − PAWP)/CI = ({mpap} − {pawp})/{ci}")


def calc_svri(aom: Optional[float], ram: Optional[float], ci: Optional[float]) -> CalcResult:
    if aom is None or ram is None or ci is None or ci == 0:
        return CalcResult(None)
    return CalcResult((aom - ram) / ci, formula=f"SVRI = (AO_mean − RA_mean)/CI = ({aom} − {ram})/{ci}")


# -----------------------------
# Rules / Cutoff Helper
# -----------------------------

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _rule(rules: Dict[str, Any], path: str, default: Optional[float] = None) -> Optional[float]:
    cur: Any = rules
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return _to_float(cur)


def _patch_engine_rules(base_rules: Dict[str, Any], legacy_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Mappt *GUI-Legacy-Overrides* (rest/exercise/stepox/volume/...) in das neue Engine-Rules-Schema.

    Hintergrund:
    - Die YAML-TextDB (rhk_textdb_api.py) enthält `RULES` in einem modernen Schema.
    - Die vorhandene GUI (Advanced-Tab) arbeitet historisch mit dem Legacy-Schema.

    Damit Engine-Suggestions (Add-ons, Plan) konsistent zu den im GUI gewählten Cutoffs sind,
    übertragen wir relevante Werte.
    """

    if not isinstance(base_rules, dict):
        out: Dict[str, Any] = {}
    else:
        out = copy.deepcopy(base_rules)

    if not isinstance(legacy_rules, dict):
        return out

    def _get(*p: str) -> Optional[float]:
        cur: Any = legacy_rules
        for part in p:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return _to_float(cur)

    def _set(path: List[str], value: Any) -> None:
        cur: Any = out
        for part in path[:-1]:
            if not isinstance(cur, dict):
                return
            if part not in cur or not isinstance(cur.get(part), dict):
                cur[part] = {}
            cur = cur[part]
        if isinstance(cur, dict):
            cur[path[-1]] = value

    # Ruhe-Cutoffs
    mpap_cut = _get("rest", "mPAP_ph_mmHg")
    pawp_cut = _get("rest", "PAWP_postcap_mmHg")
    pvr_cut = _get("rest", "PVR_precap_WU")
    if mpap_cut is not None:
        _set(["hemodynamic_definitions", "ph", "mPAP_gt_mmHg"], mpap_cut)
    if pawp_cut is not None:
        _set(["hemodynamic_definitions", "postcapillary_ph", "PAWP_gt_mmHg"], pawp_cut)
    if pvr_cut is not None:
        _set(["hemodynamic_definitions", "precapillary_ph", "PVR_gt_WU"], pvr_cut)

    # Belastung
    mpap_slope = _get("exercise", "mPAP_CO_slope_mmHg_per_L_min")
    pawp_slope = _get("exercise", "PAWP_CO_slope_mmHg_per_L_min")
    if mpap_slope is not None:
        _set(["exercise_definitions", "exercise_ph", "mPAP_CO_slope_gt_mmHg_per_L_min"], mpap_slope)
    if pawp_slope is not None:
        _set(["exercise_definitions", "postcapillary_cause_suspected", "PAWP_CO_slope_gt_mmHg_per_L_min"], pawp_slope)

    # Stepwise-Oximetrie Cutoffs
    thr_ra = _get("stepox", "thr_ra_pct")
    thr_rv = _get("stepox", "thr_rv_pct")
    thr_pa = _get("stepox", "thr_pa_pct")
    if thr_ra is not None:
        _set(["stepwise_oximetry", "stepup_thresholds_percent_points", "atrial_ge"], thr_ra)
    # Engine nutzt einen Cutoff für "ventricular_or_pa" -> nehme RV (falls gesetzt) sonst PA
    thr_vent = thr_rv if thr_rv is not None else thr_pa
    if thr_vent is not None:
        _set(["stepwise_oximetry", "stepup_thresholds_percent_points", "ventricular_or_pa_ge"], thr_vent)

    # Volumenchallenge
    pawp_post_thr = _get("volume", "pawp_post_thr_mmHg")
    if pawp_post_thr is not None:
        _set(["fluid_challenge", "positive_cutoff", "PAWP_post_gt_mmHg"], pawp_post_thr)

    return out


# -----------------------------
# Leitlinien: Klassifikationslogik (Ruhe + Belastung)
# -----------------------------

def classify_ph_rest(mpap: Optional[float], pawp: Optional[float], pvr: Optional[float], rules: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    mpap_gt = _rule(rules, "rest.mPAP_ph_mmHg", 20)
    pawp_gt = _rule(rules, "rest.PAWP_postcap_mmHg", 15)
    pvr_gt = _rule(rules, "rest.PVR_precap_WU", 2)

    if mpap is None or mpap_gt is None:
        return None, "unklar"

    ph_present = mpap > mpap_gt
    if not ph_present:
        return False, "keine PH in Ruhe"

    if pawp is None:
        return True, "PH (PAWP unklar)"

    postcap = pawp > pawp_gt if pawp_gt is not None else False
    if postcap:
        if pvr is not None and pvr_gt is not None and pvr > pvr_gt:
            return True, "cpcPH"
        return True, "postkapillär"

    if pvr is not None and pvr_gt is not None and pvr > pvr_gt:
        return True, "präkapillär"
    return True, "PH unklar (PVR nicht führend)"


def classify_exercise_pattern(mpap_co_slope: Optional[float], pawp_co_slope: Optional[float], rules: Dict[str, Any]) -> Optional[str]:
    mpap_thr = _rule(rules, "exercise.mPAP_CO_slope_mmHg_per_L_min", 3)
    pawp_thr = _rule(rules, "exercise.PAWP_CO_slope_mmHg_per_L_min", 2)
    if mpap_co_slope is None or pawp_co_slope is None or mpap_thr is None or pawp_thr is None:
        return None
    mpap_path = mpap_co_slope > mpap_thr
    pawp_path = pawp_co_slope > pawp_thr
    if (not mpap_path) and (not pawp_path):
        return "normal"
    if mpap_path and pawp_path:
        return "linkskardial"
    if mpap_path and (not pawp_path):
        return "pulmvasc"
    return "isoliert_pawp"


def pvr_severity(pvr: Optional[float], rules: Dict[str, Any]) -> Optional[str]:
    if pvr is None:
        return None

    mild_from: Optional[float] = None
    mod_from: Optional[float] = None
    sev_from: Optional[float] = None

    sev_cfg = rules.get("severity") if isinstance(rules, dict) else None
    if isinstance(sev_cfg, dict):
        pvr_cfg = sev_cfg.get("PVR_WU")
        if isinstance(pvr_cfg, dict):
            mild_from = _to_float(pvr_cfg.get("mild_ge"))
            mod_from = _to_float(pvr_cfg.get("moderate_ge"))
            sev_from = _to_float(pvr_cfg.get("severe_ge"))

    # Backward compatible keys
    if mild_from is None:
        mild_from = _rule(rules, "severity.PVR_mild_from_WU", None)
    if mod_from is None:
        mod_from = _rule(rules, "severity.PVR_moderate_from_WU", None)
    if sev_from is None:
        sev_from = _rule(rules, "severity.PVR_severe_from_WU", None)

    if mild_from is None or mod_from is None or sev_from is None:
        return None

    if pvr >= sev_from:
        return "schwer"
    if pvr >= mod_from:
        return "mittel"
    if pvr >= mild_from:
        return "leicht"
    return "unter Cut-off"


# -----------------------------
# Stufenoxymetrie: Auto-Step-Up
# -----------------------------

@dataclass
class StepUpResult:
    present: Optional[bool]
    from_site: Optional[str]
    to_site: Optional[str]
    delta_pct: Optional[float]
    from_to: str         # z.B. "SVC→RA (68→78%, +10%)"
    location_desc: str   # z.B. "(SVC→RA, +10%)"


def detect_step_up(
    sat_svc: Optional[float],
    sat_ra: Optional[float],
    sat_rv: Optional[float],
    sat_pa: Optional[float],
    rules: Dict[str, Any],
) -> StepUpResult:
    """
    Einfache (aber praktische) automatische Step-Up-Erkennung.

    Default-Schwellen (anpassbar über rules.stepox.*):
      - RA step-up (SVC->RA): >= 7%
      - RV step-up (RA->RV): >= 5%
      - PA step-up (RV->PA): >= 5%

    Rückgabe:
      present: True/False/None (None = unzureichende Daten)
      from_to/location_desc: Strings für Befundtexte
    """
    thr_ra = _rule(rules, "stepox.thr_ra_pct", 7)
    thr_rv = _rule(rules, "stepox.thr_rv_pct", 5)
    thr_pa = _rule(rules, "stepox.thr_pa_pct", 5)

    seq: List[Tuple[str, Optional[float]]] = [("SVC", sat_svc), ("RA", sat_ra), ("RV", sat_rv), ("PA", sat_pa)]
    candidates: List[Tuple[float, str, str, float]] = []  # (delta, from, to, thr)

    for (frm, v1), (to, v2) in zip(seq, seq[1:]):
        if v1 is None or v2 is None:
            continue
        delta = v2 - v1
        if to == "RA":
            thr = thr_ra
        elif to == "RV":
            thr = thr_rv
        else:
            thr = thr_pa
        if thr is None:
            continue
        if delta >= thr:
            candidates.append((delta, frm, to, thr))

    # Wenn keinerlei direkte Vergleichspunkte vorhanden → unklar
    if not any(v is not None for _, v in seq):
        return StepUpResult(None, None, None, None, from_to="", location_desc="")

    if not candidates:
        # Wenn mind. 2 benachbarte Werte vorhanden, kann man "kein Step-Up" sagen
        enough_pairs = any(v1 is not None and v2 is not None for (_, v1), (_, v2) in zip(seq, seq[1:]))
        if enough_pairs:
            return StepUpResult(False, None, None, None, from_to="", location_desc="")
        return StepUpResult(None, None, None, None, from_to="", location_desc="")

    # Nimm den größten Sprung
    candidates.sort(key=lambda x: x[0], reverse=True)
    delta, frm, to, thr = candidates[0]
    v1 = dict(seq).get(frm)
    v2 = dict(seq).get(to)
    from_to = f"{frm}→{to} ({fmt_num(v1,0)}→{fmt_num(v2,0)}%, +{fmt_num(delta,0)}%)" if (v1 is not None and v2 is not None) else f"{frm}→{to} (+{fmt_num(delta,0)}%)"
    loc = f"({frm}→{to}, +{fmt_num(delta,0)}%)"
    return StepUpResult(True, frm, to, delta, from_to=from_to, location_desc=loc)


# -----------------------------
# Volumenchallenge
# -----------------------------

@dataclass
class VolumeChallengeResult:
    performed: bool
    positive: Optional[bool]
    desc: str
    pawp_pre: Optional[float]
    pawp_post: Optional[float]
    mpap_pre: Optional[float]
    mpap_post: Optional[float]
    delta_pawp: Optional[float]


def classify_volume_challenge(
    volume_ml: Optional[float],
    infusion_type: str,
    pawp_pre: Optional[float],
    pawp_post: Optional[float],
    mpap_pre: Optional[float],
    mpap_post: Optional[float],
    rules: Dict[str, Any],
) -> VolumeChallengeResult:
    """
    Heuristik:
      - positiv, wenn PAWP_post >= pawp_post_thr (Default 18) ODER ΔPAWP >= delta_thr (Default 5)
    """
    performed = (volume_ml is not None and volume_ml > 0) or (pawp_post is not None) or (mpap_post is not None)
    if not performed:
        return VolumeChallengeResult(False, None, "", pawp_pre, pawp_post, mpap_pre, mpap_post, None)

    delta = None
    if pawp_pre is not None and pawp_post is not None:
        delta = pawp_post - pawp_pre

    pawp_post_thr = _rule(rules, "volume.pawp_post_thr_mmHg", 18)
    delta_thr = _rule(rules, "volume.delta_pawp_thr_mmHg", 5)

    positive: Optional[bool] = None
    if pawp_post is not None and pawp_post_thr is not None:
        positive = pawp_post >= pawp_post_thr
    if delta is not None and delta_thr is not None:
        positive = True if delta >= delta_thr else (positive if positive is not None else False)

    # Text
    vol_desc = ""
    if volume_ml is not None and volume_ml > 0:
        vol_desc = f"{fmt_num(volume_ml,0)} ml {infusion_type or 'Volumen'}".strip()
    else:
        vol_desc = infusion_type or "Volumenchallenge"

    desc = f"Volumenchallenge ({vol_desc}): PAWP {fmt_num(pawp_pre,0)}→{fmt_num(pawp_post,0)} mmHg"
    if delta is not None:
        desc += f" (Δ{fmt_num(delta,0)} mmHg)"
    if mpap_pre is not None or mpap_post is not None:
        desc += f", mPAP {fmt_num(mpap_pre,0)}→{fmt_num(mpap_post,0)} mmHg"
    if positive is True:
        desc += " – Hinweis auf relevante linkskardiale Komponente (Demaskierung)."
    elif positive is False:
        desc += " – kein relevanter PAWP-Anstieg."
    else:
        desc += " – Bewertung unklar (Werte unvollständig)."

    return VolumeChallengeResult(True, positive, desc, pawp_pre, pawp_post, mpap_pre, mpap_post, delta)


# -----------------------------
# Vasoreaktivität (iNO)
# -----------------------------

@dataclass
class VasoreactivityResult:
    performed: bool
    responder: Optional[bool]
    agent_desc: str
    response_desc: str


def classify_vasoreactivity(
    agent: str,
    ino_ppm: Optional[float],
    pre_mpap: Optional[float],
    post_mpap: Optional[float],
    pre_co: Optional[float],
    post_co: Optional[float],
    rules: Dict[str, Any],
) -> VasoreactivityResult:
    """
    Standard-Responderkriterium (vereinfacht, gängig):
      - mPAP-Abfall ≥ 10 mmHg UND mPAP_post ≤ 40 mmHg UND CO nicht vermindert
    """
    performed = (post_mpap is not None) or (post_co is not None) or (ino_ppm is not None)
    if not performed:
        return VasoreactivityResult(False, None, "", "")

    agent_desc = agent or "Vasoreaktivität"
    if ino_ppm is not None:
        agent_desc = f"{agent_desc} (iNO {fmt_num(ino_ppm,0)} ppm)"

    responder: Optional[bool] = None
    if pre_mpap is not None and post_mpap is not None:
        delta = pre_mpap - post_mpap
        co_ok = True
        if pre_co is not None and post_co is not None:
            co_ok = post_co >= pre_co - 0.1  # kleine Messstreuung tolerieren
        responder = (delta >= 10) and (post_mpap <= 40) and co_ok
    # else bleibt None

    # Response text
    parts = []
    if pre_mpap is not None or post_mpap is not None:
        parts.append(f"mPAP {fmt_num(pre_mpap,0)}→{fmt_num(post_mpap,0)} mmHg")
        if pre_mpap is not None and post_mpap is not None:
            parts.append(f"(Δ−{fmt_num(pre_mpap - post_mpap,0)} mmHg)")
    if pre_co is not None or post_co is not None:
        parts.append(f"CO {fmt_num(pre_co,2)}→{fmt_num(post_co,2)} L/min")
    resp = "Vasoreaktivität: " + ", ".join(parts) if parts else "Vasoreaktivität: Werte unvollständig"
    if responder is True:
        resp += " – Responderkriterium erfüllt."
    elif responder is False:
        resp += " – Responderkriterium nicht erfüllt."
    else:
        resp += " – Bewertung unklar."

    return VasoreactivityResult(True, responder, agent_desc, resp)


# -----------------------------
# Risiko: ESC/ERS 3-Strata & 4-Strata, REVEAL Lite 2 (wie bisher)
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
    cat = None
    if overall == 1:
        cat = "low"
    elif overall == 2:
        cat = "intermediate"
    elif overall == 3:
        cat = "high"
    return {"overall": overall, "mean": mean, "category": cat, "grades": {"WHO_FC": g_fc, "6MWD": g_6, "BNP": g_b}}


def esc4_overall(who_fc: Optional[str], sixmwd_m: Optional[float], bnp_kind: str, bnp_value: Optional[float]) -> Dict[str, Any]:
    g_fc = esc4_grade_who_fc(who_fc)
    g_6 = esc4_grade_6mwd(sixmwd_m)
    g_b = esc4_grade_bnp(bnp_value, bnp_kind)
    grades = [g for g in [g_fc, g_6, g_b] if isinstance(g, int)]
    overall, mean = aggregate_grade(grades, 4)
    cat = None
    if overall == 1:
        cat = "low"
    elif overall == 2:
        cat = "intermediate-low"
    elif overall == 3:
        cat = "intermediate-high"
    elif overall == 4:
        cat = "high"
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
        if s in ("I", "1"):
            pts["WHO_FC"] = -1
        elif s in ("II", "2"):
            pts["WHO_FC"] = 0
        elif s in ("III", "3"):
            pts["WHO_FC"] = 1
        elif s in ("IV", "4"):
            pts["WHO_FC"] = 2

    if sixmwd_m is not None:
        if sixmwd_m >= 440:
            pts["6MWD"] = -2
        elif sixmwd_m >= 320:
            pts["6MWD"] = -1
        elif sixmwd_m >= 165:
            pts["6MWD"] = 0
        else:
            pts["6MWD"] = 1

    if bnp_value is not None:
        k = (bnp_kind or "").strip().lower()
        if "nt" in k:
            if bnp_value < 300:
                pts["BNP"] = -2
            elif bnp_value < 1100:
                pts["BNP"] = 0
            else:
                pts["BNP"] = 2
        else:
            if bnp_value < 50:
                pts["BNP"] = -2
            elif bnp_value < 200:
                pts["BNP"] = 0
            elif bnp_value < 800:
                pts["BNP"] = 1
            else:
                pts["BNP"] = 2

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
    if score <= 6:
        risk = "low"
    elif score <= 8:
        risk = "intermediate"
    else:
        risk = "high"
    return {"score": score, "risk": risk, "points": pts, "note": None}


def _risk_badge(label: str, value: str, level: str) -> str:
    colors = {
        "low": ("#0f7b0f", "#eaffea"),
        "intermediate": ("#8a6d00", "#fff6d5"),
        "intermediate-low": ("#8a6d00", "#fff6d5"),
        "intermediate-high": ("#a14400", "#ffe7d5"),
        "high": ("#b00020", "#ffe5e9"),
        "unknown": ("#444", "#f2f2f2"),
    }
    fg, bg = colors.get(level, colors["unknown"])
    return (
        f"<div style='display:inline-block;border-radius:10px;padding:6px 10px;"
        f"border:1px solid {fg};background:{bg};color:{fg};margin-right:8px;'>"
        f"<b>{label}:</b> {value}</div>"
    )


def render_risk_html(esc3: Dict[str, Any], esc4: Dict[str, Any], reveal: Dict[str, Any]) -> str:
    parts: List[str] = []
    if esc3.get("overall") is None:
        parts.append(_risk_badge("ESC/ERS 3-Strata", "—", "unknown"))
    else:
        cat = esc3.get("category") or "unknown"
        parts.append(_risk_badge("ESC/ERS 3-Strata", f"{esc3['overall']} ({cat})", cat))

    if esc4.get("overall") is None:
        parts.append(_risk_badge("ESC/ERS 4-Strata", "—", "unknown"))
    else:
        cat = esc4.get("category") or "unknown"
        parts.append(_risk_badge("ESC/ERS 4-Strata", f"{esc4['overall']} ({cat})", cat))

    if reveal.get("score") is None:
        parts.append(_risk_badge("REVEAL Lite 2", "—", "unknown"))
    else:
        cat = reveal.get("risk") or "unknown"
        parts.append(_risk_badge("REVEAL Lite 2", f"{reveal['score']} ({cat})", cat))

    detail_lines: List[str] = []
    if esc3.get("grades"):
        detail_lines.append(f"ESC3-Grades: {esc3['grades']}")
    if esc4.get("grades"):
        detail_lines.append(f"ESC4-Grades: {esc4['grades']}")
    if reveal.get("points"):
        detail_lines.append(f"REVEAL-Points: {reveal['points']}")
    if reveal.get("note"):
        detail_lines.append(str(reveal["note"]))

    details_html = ""
    if detail_lines:
        details_html = "<div style='margin-top:8px;color:#555;font-size:12px;white-space:pre-wrap;'>" + "\n".join(detail_lines) + "</div>"

    return "<div>" + "".join(parts) + details_html + "</div>"


# -----------------------------
# Helper: Step-Ox format
# -----------------------------

def _format_step_ox(
    sat_svc: Optional[float],
    sat_ra: Optional[float],
    sat_rv: Optional[float],
    sat_pa: Optional[float],
    sat_ao: Optional[float],
    step_up_present: Optional[bool],
    step_up_location_desc: str,
) -> str:
    sat_parts: List[str] = []
    if sat_svc is not None:
        sat_parts.append(f"SVC {fmt_num(sat_svc, 0)}%")
    if sat_ra is not None:
        sat_parts.append(f"RA {fmt_num(sat_ra, 0)}%")
    if sat_rv is not None:
        sat_parts.append(f"RV {fmt_num(sat_rv, 0)}%")
    if sat_pa is not None:
        sat_parts.append(f"PA {fmt_num(sat_pa, 0)}%")
    if sat_ao is not None:
        sat_parts.append(f"AO {fmt_num(sat_ao, 0)}%")

    txt = "Stufenoxymetrie: " + (", ".join(sat_parts) if sat_parts else "nicht erhoben")
    if step_up_present is True:
        txt += f" | Sättigungssprung: ja {step_up_location_desc}".rstrip()
    elif step_up_present is False:
        txt += " | Sättigungssprung: nein"
    else:
        txt += " | Sättigungssprung: unklar"
    return txt


# -----------------------------
# K-Paket-Auswahl (neu: kombiniert)
# -----------------------------

def _pick_main_blocks_v2(
    modules: List[str],
    ph_present: Optional[bool],
    ph_type: str,
    pvr: Optional[float],
    ci: Optional[float],
    exercise_pattern: Optional[str],
    step_up_present: Optional[bool],
    cteph_suspected: bool,
    volume_positive: Optional[bool],
    rules: Dict[str, Any],
) -> Tuple[str, str, List[str], List[str]]:
    """
    Gibt Block-IDs (Beurteilung/Empfehlung) zurück.
    Hauptlogik:
      - Step-up hat Priorität (K16)
      - CTEPH-Pfad (K11) wenn markiert und präkapillär
      - Wenn Belastung gewählt und in Ruhe keine PH: K02/K03/K01/K04 je nach Slopes
      - Wenn Volumenchallenge positiv und in Ruhe keine PH: K04
      - Sonst Ruhe-Klassifikation: K01/K14/K15/K05-07/K04
    """
    extra_beur: List[str] = []
    extra_empf: List[str] = []

    has_ex = "Belastung" in modules
    has_vol = "Volumenchallenge" in modules

    if step_up_present is True:
        return "K16_B", "K16_E", extra_beur, extra_empf

    if cteph_suspected and ph_type in ("präkapillär", "PH (PAWP unklar)"):
        return "K11_B", "K11_E", extra_beur, extra_empf

    if has_ex and (ph_present is False or ph_present is None):
        if exercise_pattern == "linkskardial":
            return "K02_B", "K02_E", extra_beur, extra_empf
        if exercise_pattern == "pulmvasc":
            return "K03_B", "K03_E", extra_beur, extra_empf
        if exercise_pattern == "normal":
            return "K01_B", "K01_E", extra_beur, extra_empf
        return "K04_B", "K04_E", extra_beur, extra_empf

    if has_vol and (volume_positive is True) and (ph_present is False or ph_present is None):
        # Volumen-Demaskierung → kein "Normalbefund"
        return "K04_B", "K04_E", extra_beur, extra_empf

    if ph_present is False:
        return "K01_B", "K01_E", extra_beur, extra_empf
    if ph_present is None:
        return "K04_B", "K04_E", extra_beur, extra_empf

    if ph_type == "postkapillär":
        return "K14_B", "K14_E", extra_beur, extra_empf
    if ph_type == "cpcPH":
        return "K15_B", "K15_E", extra_beur, extra_empf

    if ph_type == "präkapillär":
        sev = pvr_severity(pvr, rules)
        ci_low = _rule(rules, "severity.CI_L_min_m2.severely_reduced_lt", None)
        if ci_low is None:
            ci_low = _rule(rules, "severity.CI_low_lt_L_min_m2", 2.0)
        if (ci is not None) and (ci_low is not None) and (ci < ci_low):
            return "K07_B", "K07_E", extra_beur, extra_empf
        if sev == "schwer":
            return "K07_B", "K07_E", extra_beur, extra_empf
        if sev == "leicht":
            return "K05_B", "K05_E", extra_beur, extra_empf
        return "K06_B", "K06_E", extra_beur, extra_empf

    return "K04_B", "K04_E", extra_beur, extra_empf


# -----------------------------
# Procedere renderer (+ Auto-Vorschläge)
# -----------------------------

def _render_procedere(planned_actions: List[str], render_ctx: SafeDict, main_bundle_id: Optional[str]) -> str:
    """
    Wenn planned_actions leer ist, werden – falls verfügbar – Vorschläge aus textdb.BUNDLES genommen.
    """
    actions = list(planned_actions or [])

    if not actions and main_bundle_id and hasattr(textdb, "BUNDLES"):
        bundles = getattr(textdb, "BUNDLES", {}) or {}
        b = bundles.get(main_bundle_id)
        if isinstance(b, dict):
            sugg = b.get("P_suggestions") or []
            if isinstance(sugg, list):
                actions.extend([str(x) for x in sugg if str(x).strip()])

    if not actions:
        return "- —"

    lines: List[str] = []

    def add_bullet(text: str) -> None:
        text = (text or "").rstrip()
        if not text:
            return
        parts = text.splitlines()
        lines.append(f"- {parts[0]}")
        for l in parts[1:]:
            lines.append(f"  {l}")

    for item in actions:
        item = (item or "").strip()
        if not item:
            continue
        if hasattr(textdb, "P_BLOCKS") and item in getattr(textdb, "P_BLOCKS"):
            b = textdb.get_block(item)
            if b:
                try:
                    add_bullet(_norm_spaces(str(b.template).format_map(render_ctx)))
                except Exception:
                    add_bullet(_norm_spaces(str(b.template)))
        else:
            add_bullet(_norm_spaces(item))

    return "\n".join(lines) if lines else "- —"


# -----------------------------
# Report Generator
# -----------------------------

class RHKReportGenerator:
    def generate_all(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        ctx = data.get("context", {}) or {}
        patient = data.get("patient", {}) or {}
        raw = data.get("raw_values", {}) or {}
        derived = data.get("derived_values", {}) or {}
        flags = data.get("interpretation_flags", {}) or {}
        planned_actions = data.get("planned_actions", []) or []
        local_rules = data.get("local_rules", {}) or {}
        qualitative = data.get("qualitative", {}) or {}
        clinical = data.get("clinical_context", {}) or {}
        additional = data.get("additional_measurements", {}) or {}
        internal = data.get("internal", {}) or {}

        rules = dict(getattr(textdb, "DEFAULT_RULES", {}) or {})
        rules = _deep_merge(rules, (local_rules.get("rules", {}) or {}))

        calc_steps: List[str] = []
        plaus_warnings: List[str] = []
        missing: List[str] = []

        pressures = (raw.get("pressures_mmHg", {}) or {})
        flow = (raw.get("flow", {}) or {})
        sats = (raw.get("sats_pct", {}) or {})

        # -------- Ruhe (Basis) --------
        RA_mean = _to_float(pressures.get("RA_mean"))
        PA_sys = _to_float(pressures.get("PA_sys"))
        PA_dia = _to_float(pressures.get("PA_dia"))
        PA_mean_in = _to_float(pressures.get("PA_mean"))
        PAWP_mean = _to_float(pressures.get("PAWP_mean"))
        AO_sys = _to_float(pressures.get("AO_sys"))
        AO_dia = _to_float(pressures.get("AO_dia"))
        AO_mean_in = _to_float(pressures.get("AO_mean"))

        CO = _to_float(flow.get("CO_L_min"))
        CI_in = _to_float(flow.get("CI_L_min_m2"))
        HR = _to_float(flow.get("HR_min"))

        height_cm = _to_float(patient.get("height_cm"))
        weight_kg = _to_float(patient.get("weight_kg"))

        BSA = calc_bsa_dubois(height_cm, weight_kg).value
        if BSA is not None:
            calc_steps.append(f"BSA(DuBois) = {fmt_num(BSA,2)} m²")

        mpap = PA_mean_in
        if mpap is None:
            r = calc_mean(PA_sys, PA_dia)
            mpap = r.value
            if r.formula and mpap is not None:
                calc_steps.append(f"mPAP: {r.formula} = {fmt_num(mpap,0)} mmHg")

        aom = AO_mean_in
        if aom is None:
            r = calc_mean(AO_sys, AO_dia)
            aom = r.value
            if r.formula and aom is not None:
                calc_steps.append(f"AO_mean: {r.formula} = {fmt_num(aom,0)} mmHg")

        CI = CI_in
        if CI is None:
            r = calc_ci(CO, BSA)
            CI = r.value
            if r.formula and CI is not None:
                calc_steps.append(f"CI: {r.formula} = {fmt_num(CI,2)} L/min/m²")

        TPG = _to_float(derived.get("TPG_mmHg"))
        if TPG is None:
            r = calc_tpg(mpap, PAWP_mean)
            TPG = r.value
            if r.formula and TPG is not None:
                calc_steps.append(f"TPG: {r.formula} = {fmt_num(TPG,0)} mmHg")

        DPG = _to_float(derived.get("DPG_mmHg"))
        if DPG is None:
            r = calc_dpg(PA_dia, PAWP_mean)
            DPG = r.value
            if r.formula and DPG is not None:
                calc_steps.append(f"DPG: {r.formula} = {fmt_num(DPG,0)} mmHg")

        PVR = _to_float(derived.get("PVR_WU"))
        if PVR is None:
            r = calc_pvr(mpap, PAWP_mean, CO)
            PVR = r.value
            if r.formula and PVR is not None:
                calc_steps.append(f"PVR: {r.formula} = {fmt_num(PVR,1)} WU")

        SVR = _to_float(derived.get("SVR_WU"))
        if SVR is None:
            r = calc_svr(aom, RA_mean, CO)
            SVR = r.value
            if r.formula and SVR is not None:
                calc_steps.append(f"SVR: {r.formula} = {fmt_num(SVR,1)} WU")

        PVRI = calc_pvri(mpap, PAWP_mean, CI).value
        SVRI = calc_svri(aom, RA_mean, CI).value

        # -------- Module Auswahl --------
        modules_raw = ctx.get("modules") or []
        if isinstance(modules_raw, str):
            modules: List[str] = [modules_raw]
        else:
            modules = [str(x) for x in (modules_raw or []) if str(x).strip()]
        has_ex = "Belastung" in modules
        has_vol = "Volumenchallenge" in modules
        has_vaso = "Vasoreaktivität" in modules

        # -------- Belastung --------
        ex = additional.get("exercise_peak", {}) or {}
        ex_CO = _to_float(ex.get("CO_L_min"))
        ex_mPAP = _to_float(ex.get("mPAP_mmHg"))
        ex_PAWP = _to_float(ex.get("PAWP_mmHg"))
        ex_sPAP = _to_float(ex.get("sPAP_mmHg"))

        mPAP_CO_slope = _to_float(derived.get("mPAP_CO_slope_mmHg_per_L_min"))
        PAWP_CO_slope = _to_float(derived.get("PAWP_CO_slope_mmHg_per_L_min"))
        if mPAP_CO_slope is None and has_ex:
            mPAP_CO_slope = calc_slope(mpap, CO, ex_mPAP, ex_CO).value
        if PAWP_CO_slope is None and has_ex:
            PAWP_CO_slope = calc_slope(PAWP_mean, CO, ex_PAWP, ex_CO).value

        exercise_pattern = classify_exercise_pattern(mPAP_CO_slope, PAWP_CO_slope, rules) if has_ex else None
        CI_peak = ex_CO / BSA if (has_ex and ex_CO is not None and BSA) else None
        delta_sPAP = (ex_sPAP - PA_sys) if (has_ex and ex_sPAP is not None and PA_sys is not None) else None

        # -------- Stufenoxymetrie --------
        sat_SVC = _to_float(sats.get("SVC"))
        sat_RA = _to_float(sats.get("RA"))
        sat_RV = _to_float(sats.get("RV"))
        sat_PA = _to_float(sats.get("PA"))
        sat_AO = _to_float(sats.get("AO"))

        step_mode = (flags.get("step_up_mode") or "auto").strip().lower()
        step_loc_override = (flags.get("step_up_location_override") or "").strip() or None

        step_auto = detect_step_up(sat_SVC, sat_RA, sat_RV, sat_PA, rules)

        if step_mode == "ja":
            step_up_present = True
            step_up_from_to = step_loc_override or (step_auto.from_to if step_auto.from_to else "")
            step_up_location_desc = f"({step_up_from_to})" if step_up_from_to else ""
        elif step_mode == "nein":
            step_up_present = False
            step_up_from_to = ""
            step_up_location_desc = ""
        else:
            # auto
            step_up_present = step_auto.present
            step_up_from_to = step_auto.from_to
            step_up_location_desc = step_auto.location_desc

        # -------- Volumenchallenge --------
        vc = additional.get("volume_challenge", {}) or {}
        vc_volume_ml = _to_float(vc.get("volume_ml"))
        vc_infusion = str(vc.get("infusion_type") or "NaCl 0.9%")
        vc_pawp_post = _to_float(vc.get("PAWP_post"))
        vc_mpap_post = _to_float(vc.get("mPAP_post"))
        # Pre: optional override, sonst Ruhe
        vc_pawp_pre = _to_float(vc.get("PAWP_pre")) or PAWP_mean
        vc_mpap_pre = _to_float(vc.get("mPAP_pre")) or mpap

        vol_res = classify_volume_challenge(vc_volume_ml, vc_infusion, vc_pawp_pre, vc_pawp_post, vc_mpap_pre, vc_mpap_post, rules) if has_vol else VolumeChallengeResult(False, None, "", vc_pawp_pre, vc_pawp_post, vc_mpap_pre, vc_mpap_post, None)

        # -------- Vasoreaktivität --------
        vaso = additional.get("vasoreactivity", {}) or {}
        vaso_agent = str(vaso.get("agent") or "iNO")
        vaso_ino_ppm = _to_float(vaso.get("ino_ppm"))
        vaso_mpap_post = _to_float(vaso.get("mPAP_post"))
        vaso_co_post = _to_float(vaso.get("CO_post"))
        vaso_mpap_pre = _to_float(vaso.get("mPAP_pre")) or mpap
        vaso_co_pre = _to_float(vaso.get("CO_pre")) or CO

        vaso_res = classify_vasoreactivity(vaso_agent, vaso_ino_ppm, vaso_mpap_pre, vaso_mpap_post, vaso_co_pre, vaso_co_post, rules) if has_vaso else VasoreactivityResult(False, None, "", "")

        
        # -------- abgeleitete Werte in JSON/Export zurückschreiben --------
        # (damit im JSON sichtbar ist, was automatisch berechnet wurde)
        try:
            BMI = calc_bmi(height_cm, weight_kg)
        except Exception:
            BMI = None

        derived_out = dict(derived) if isinstance(derived, dict) else {}
        derived_out.update(
            {
                "BSA_m2": BSA,
                "BMI_kg_m2": BMI,
                "mPAP_mmHg": mpap,
                "AO_mean_mmHg": aom,
                "CI_L_min_m2": CI,
                "TPG_mmHg": TPG,
                "DPG_mmHg": DPG,
                "PVR_WU": PVR,
                "SVR_WU": SVR,
                "PVRI_WU_m2": PVRI,
                "SVRI_WU_m2": SVRI,
                "mPAP_CO_slope_mmHg_per_L_min": mPAP_CO_slope,
                "PAWP_CO_slope_mmHg_per_L_min": PAWP_CO_slope,
                "CI_peak_L_min_m2": CI_peak,
                "delta_sPAP_mmHg": delta_sPAP,
                "stepup_present": step_up_present,
                "stepup_from_to": step_up_from_to,
                "volume_positive": vol_res.positive if has_vol else None,
                "vaso_responder": vaso_res.responder if has_vaso else None,
            }
        )
        data["derived_values"] = derived_out


        # -------- Plausibilitäten --------
        if mpap is not None and PAWP_mean is not None and PAWP_mean > mpap:
            plaus_warnings.append(f"PAWP ({fmt_num(PAWP_mean,0)} mmHg) > mPAP ({fmt_num(mpap,0)} mmHg) – Wedge/Signal prüfen.")
        if CO is not None and CO <= 0:
            plaus_warnings.append(f"CO ({fmt_num(CO,2)} L/min) ≤ 0 – unplausibel.")
        if PVR is not None and PVR < 0:
            plaus_warnings.append(f"PVR ({fmt_num(PVR,1)} WU) < 0 – unplausibel.")
        if CI is not None and CI <= 0:
            plaus_warnings.append(f"CI ({fmt_num(CI,2)} L/min/m²) ≤ 0 – unplausibel.")

        if mpap is None:
            missing.append("mPAP/PA_mean")
        if PAWP_mean is None:
            missing.append("PAWP_mean")
        if CO is None:
            missing.append("CO")
        if CI is None:
            missing.append("CI (BSA oder CI fehlt)")

        # -------- Klassifikation Ruhe --------
        use_thr = bool(local_rules.get("use_guideline_cutoffs", True))
        if use_thr:
            ph_present, ph_type = classify_ph_rest(mpap, PAWP_mean, PVR, rules)
        else:
            ph_present, ph_type = None, "unklar"

        pvr_sev = pvr_severity(PVR, rules)

        # -------- Auswahl Hauptpaket --------
        main_B_id, main_E_id, extra_beur_texts, extra_empf_texts = _pick_main_blocks_v2(
            modules=modules,
            ph_present=ph_present,
            ph_type=ph_type,
            pvr=PVR,
            ci=CI,
            exercise_pattern=exercise_pattern,
            step_up_present=step_up_present,
            cteph_suspected=bool(clinical.get("ctepd_cteph_suspected")),
            volume_positive=vol_res.positive if has_vol else None,
            rules=rules,
        )

        # -------- Kontext/Strings für Templates --------
        co_method_desc = {
            "Thermodilution": "Thermodilution",
            "Fick_direkt": "direkter Fick",
            "Fick_indirekt": "indirekter Fick",
        }.get(str(ctx.get("co_method") or ""), str(ctx.get("co_method") or "nicht angegeben"))

        oxygen = ctx.get("oxygen", {}) or {}
        oxy_mode = (oxygen.get("mode") or "nicht angegeben")
        oxy_flow = _to_float(oxygen.get("flow_l_min"))
        if str(oxy_mode).lower().startswith("raum"):
            oxygen_header = "Raumluft"
            oxygen_sentence = "Unter Raumluft."
        elif str(oxy_mode).lower().startswith("o2") or str(oxy_mode).lower().startswith("sauer"):
            oxygen_header = "O2" if oxy_flow is None else f"O2 {fmt_num(oxy_flow,0)} L/min"
            oxygen_sentence = f"Unter {oxygen_header}."
        else:
            oxygen_header = str(oxy_mode)
            oxygen_sentence = f"Unter {oxygen_header}."

        exam_type = ctx.get("exam_type") or "nicht angegeben"
        exam_type_desc = "Initial-RHK" if str(exam_type).lower().startswith("initial") else ("invasive Verlaufskontrolle" if str(exam_type).lower().startswith("verlauf") else str(exam_type))

        # Systemik (qualitativ)
        bp_status = qualitative.get("systemic_bp") or ""
        hr_status_raw = qualitative.get("systemic_hr") or ""
        rhythm_status = qualitative.get("rhythm") or ""
        systemic_sentence = ""
        if any([bp_status, hr_status_raw, rhythm_status]):
            systemic_sentence = "Systemisch " + join_nonempty([bp_status, hr_status_raw, rhythm_status], "/") + "."

        # Step-Up Satz
        if step_up_present is True:
            step_up_sentence = "Sättigungssprung in der Stufenoxymetrie."
        elif step_up_present is False:
            step_up_sentence = "Kein relevanter Sättigungssprung in der Stufenoxymetrie."
        else:
            step_up_sentence = "Stufenoxymetrie: Bewertung unklar."

        # Stauung (heuristisch)
        cv_stauung_phrase = ""
        if RA_mean is not None:
            if RA_mean >= 15:
                cv_stauung_phrase = "Ausgeprägte zentralvenöse Stauung."
            elif RA_mean >= 8:
                cv_stauung_phrase = "Leichtgradige zentralvenöse Stauung."
        pv_stauung_phrase = ""
        if PAWP_mean is not None:
            pawp_gt = _rule(rules, "rest.PAWP_postcap_mmHg", 15) or 15
            if PAWP_mean > pawp_gt:
                pv_stauung_phrase = "Hinweis auf pulmonalvenöse Stauung."

        pvr_sev_phrase = f"({pvr_sev}gradige Widerstandserhöhung)" if pvr_sev in ("leicht", "mittel", "schwer") else ""

        mpap_phrase = f"mPAP {fmt_num(mpap,0)} mmHg" if mpap is not None else "mPAP nicht erhoben"
        pawp_phrase = f"PAWP {fmt_num(PAWP_mean,0)} mmHg" if PAWP_mean is not None else "PAWP nicht erhoben"
        pvr_phrase = f"PVR {fmt_num(PVR,1)} WU {pvr_sev_phrase}".strip() if PVR is not None else "PVR nicht erhoben"

        ci_phrase = f"CI {fmt_num(CI,2)} L/min/m²" if CI is not None else (f"CO {fmt_num(CO,2)} L/min" if CO is not None else "HZV nicht erhoben")
        tpg_phrase = f"TPG {fmt_num(TPG,0)} mmHg" if TPG is not None else "TPG nicht erhoben"
        pressure_resistance_short = join_nonempty([mpap_phrase, pawp_phrase, tpg_phrase, pvr_phrase], ", ")

        rest_ph_sentence = "keine PH in Ruhe" if ph_present is False else ("PH in Ruhe" if ph_present is True else "PH-Bewertung in Ruhe unklar")
        borderline_ph_sentence = "Grenzwertige/unklare Hämodynamik" if ph_present is None else ("Grenzwertige Konstellation" if ph_present is False else "Hämodynamik im Grenzbereich")

        # PAWP slope phrase
        pawp_slope_thr = _rule(rules, "exercise.PAWP_CO_slope_mmHg_per_L_min", 2.0)
        if PAWP_CO_slope is None or pawp_slope_thr is None:
            PAWP_CO_slope_phrase = "unklarer PAWP/CO-Slope"
        else:
            PAWP_CO_slope_phrase = "nicht führend erhöhter PAWP/CO-Slope" if PAWP_CO_slope <= pawp_slope_thr else "pathologisch erhöhter PAWP/CO-Slope"

        # Provocation phrases (Volumen/Belastung)
        provocation_type_desc = ""
        provocation_result_sentence = ""
        provocation_sentence = ""
        if has_vol and vol_res.performed:
            provocation_type_desc = "Volumenchallenge"
            provocation_result_sentence = vol_res.desc
            provocation_sentence = vol_res.desc
        elif has_ex and exercise_pattern is not None:
            provocation_type_desc = "Belastung"
            provocation_result_sentence = f"mPAP/CO-Slope {fmt_num(mPAP_CO_slope,1)}, PAWP/CO-Slope {fmt_num(PAWP_CO_slope,1)}."
            provocation_sentence = provocation_result_sentence

        # Therapie-Sätze (default neutral)
        therapy_neutral_sentence = ""  # z.B. bewusst leer
        therapy_plan_sentence = "Therapieeinleitung/-anpassung im PH-Team/PH-Board gemäß Gesamtkonstellation prüfen."
        therapy_escalation_sentence = "Therapieeskalation im PH-Team/PH-Board entsprechend Risikoprofil prüfen."

        anticoagulation_plan_sentence = ""
        if internal.get("procedure", {}).get("anticoagulation"):
            anticoagulation_plan_sentence = "Antikoagulation gemäß Indikation sicherstellen."

        # Zusatz: Klinischer Kontext
        cteph_context_desc = str(clinical.get("cteph_context_desc") or "").strip()
        left_heart_context_desc = str(clinical.get("left_heart_context_desc") or "").strip()
        pvod_hint_desc = str(clinical.get("pvod_hint_desc") or "").strip()

        # Vor-RHK (optional) -> Satz für Beurteilung
        prev = (additional.get("previous_rhc") or {})
        comparison_sentence = ""
        if isinstance(prev, dict) and (prev.get("label") or prev.get("mpap_mmHg") or prev.get("pawp_mmHg") or prev.get("ci_L_min_m2") or prev.get("pvr_WU")):
            prev_label = (prev.get("label") or "").strip()
            prev_course = (prev.get("course_desc") or "").strip() or "Verlauf"
            parts: List[str] = []
            if prev.get("mpap_mmHg") is not None:
                parts.append(f"mPAP {fmt_num(prev.get('mpap_mmHg'),0)} mmHg")
            if prev.get("pawp_mmHg") is not None:
                parts.append(f"PAWP {fmt_num(prev.get('pawp_mmHg'),0)} mmHg")
            if prev.get("ci_L_min_m2") is not None:
                parts.append(f"CI {fmt_num(prev.get('ci_L_min_m2'),2)} L/min/m²")
            if prev.get("pvr_WU") is not None:
                parts.append(f"PVR {fmt_num(prev.get('pvr_WU'),1)} WU")

            if prev_label:
                comparison_sentence = (
                    f"Im Vergleich zu RHK {prev_label} {prev_course}"
                    + (f" ({', '.join(parts)})." if parts else ".")
                )
            elif parts:
                comparison_sentence = f"Im Vergleich zum Vor-RHK {prev_course} ({', '.join(parts)})."

        # -----------------------------------------------------------------
        # YAML-TextDB Engine (Add-ons + Kontext) – optional
        # -----------------------------------------------------------------

        engine_rules = _patch_engine_rules(getattr(textdb, "RULES", {}) or {}, rules)

        # Optionale Quality-/Confounder-Flags können (später) über additional['quality'] kommen.
        # Wenn nichts vorhanden ist, bleiben sie None.
        quality = additional.get("quality") if isinstance(additional, dict) else None
        quality = quality if isinstance(quality, dict) else {}

        # Engine-Input zusammenstellen (robuste Key-Namen + Synonyme)
        engine_data: Dict[str, Any] = {
            "mPAP": mpap,
            "sPAP": PA_sys,
            "dPAP": PA_dia,
            "PAWP": PAWP_mean,
            "RAP": RA_mean,
            "MAP": aom,
            "CO": CO,
            "CI": CI,
            "HR": HR,
            "BSA": BSA,
            # Oximetry
            "Sat_SVC": sat_SVC,
            "Sat_RA": sat_RA,
            "Sat_RV": sat_RV,
            "Sat_PA": sat_PA,
            "SvO2": sat_PA,
            "SaO2": sat_AO,
            # Optional: Belastung
            "mPAP_CO_slope": mPAP_CO_slope,
            "PAWP_CO_slope": PAWP_CO_slope,
            # Optional: Volumenchallenge / Vasoreaktivität
            "PAWP_post": vol_res.pawp_post if (has_vol and vol_res.performed) else None,
            "mPAP_post": vaso_mpap_post if (has_vaso and vaso_res.performed) else None,
            "CO_post": vaso_co_post if (has_vaso and vaso_res.performed) else None,
            # Optional: Messqualität
            "wedge_sat": _to_float(quality.get("wedge_sat_pct")) if quality else None,
            "respiratory_swings_large": quality.get("resp_swings_large") if quality else None,
            "obesity": quality.get("obesity") if quality else None,
            "copd": quality.get("copd") if quality else None,
            "mechanical_ventilation": quality.get("mechanical_ventilation") if quality else None,
            # Kontext
            "ctepd_cteph_suspected": bool(clinical.get("ctepd_cteph_suspected")) if isinstance(clinical, dict) else False,
        }

        engine_plan: Dict[str, Any] = {}
        addon_block_ids: List[str] = []
        engine_ctx: Dict[str, Any] = {}
        try:
            if hasattr(textdb, "suggest_plan"):
                engine_plan = textdb.suggest_plan(engine_data, engine_rules)  # type: ignore
                addon_block_ids = list(engine_plan.get("addon_blocks") or [])
                engine_ctx = dict(engine_plan.get("context") or {})
            elif hasattr(textdb, "build_context"):
                engine_ctx = dict(textdb.build_context(engine_data, engine_rules) or {})  # type: ignore
        except Exception as e:
            plaus_warnings.append(f"TextDB-Engine: {e}")
            engine_plan = {}
            addon_block_ids = []
            engine_ctx = {}

        # Render ctx
        render_ctx = SafeDict(
            {
                # Core phrases
                "ci_phrase": ci_phrase,
                "co_method_desc": co_method_desc,
                "step_up_sentence": step_up_sentence,
                "systemic_sentence": systemic_sentence,
                "oxygen_sentence": oxygen_sentence,
                "exam_type_desc": exam_type_desc,
                "mpap_phrase": mpap_phrase,
                "pawp_phrase": pawp_phrase,
            "tpg_phrase": tpg_phrase,
                "pvr_phrase": pvr_phrase,
                "pvr_sev_phrase": pvr_sev_phrase,
                "pressure_resistance_short": pressure_resistance_short,
                # exercise
                "mPAP_CO_slope": fmt_num(mPAP_CO_slope, 1) if mPAP_CO_slope is not None else "",
                "PAWP_CO_slope": fmt_num(PAWP_CO_slope, 1) if PAWP_CO_slope is not None else "",
                "PAWP_CO_slope_phrase": PAWP_CO_slope_phrase,
                "delta_sPAP": fmt_num(delta_sPAP, 0) if delta_sPAP is not None else "",
                "CI_peak": fmt_num(CI_peak, 2) if CI_peak is not None else "",
                # Step-up details
                "step_up_from_to": step_up_from_to,
                "step_up_location_desc": step_up_location_desc,
                # Stauung
                "cv_stauung_phrase": cv_stauung_phrase,
                "pv_stauung_phrase": pv_stauung_phrase,
                # Provocation
                "provocation_type_desc": provocation_type_desc,
                "provocation_result_sentence": provocation_result_sentence,
                "provocation_sentence": provocation_sentence,
                "rest_ph_sentence": rest_ph_sentence,
                "borderline_ph_sentence": borderline_ph_sentence,
                # Volume challenge placeholders (für B11/B12 falls genutzt)
                "volume_challenge_desc": f"{fmt_num(vc_volume_ml,0)} ml {vc_infusion}".strip() if vc_volume_ml is not None else (vc_infusion or "Volumenchallenge"),
                "PAWP_pre": fmt_num(vol_res.pawp_pre, 0),
                "PAWP_post": fmt_num(vol_res.pawp_post, 0),
                "mPAP_pre": fmt_num(vol_res.mpap_pre, 0),
                "mPAP_post": fmt_num(vol_res.mpap_post, 0),
                # Vasoreactivity placeholders
                "vasoreactivity_agent_desc": vaso_res.agent_desc,
                "iNO_response_desc": vaso_res.response_desc,
                "iNO_ppm": fmt_num(vaso_ino_ppm,0) if vaso_ino_ppm is not None else "",
                "iNO_o2_desc": "",
                "iNO_responder_statement": "",
                # Therapy placeholders
                "therapy_neutral_sentence": therapy_neutral_sentence,
                "therapy_plan_sentence": therapy_plan_sentence,
                "therapy_escalation_sentence": therapy_escalation_sentence,
                "therapy_examples_sentence": "",
                # Misc
                "comparison_sentence": comparison_sentence,
                "measurement_limitation_sentence": "",
                "severity_ph_sentence": "",
                "lufu_summary": str(additional.get("lufu_summary") or "").strip(),
                "lufu_context_sentence": "",
                "cteph_context_desc": cteph_context_desc,
                "anticoagulation_plan_sentence": anticoagulation_plan_sentence,
                "left_heart_context_desc": left_heart_context_desc,
                "pvod_hint_desc": pvod_hint_desc,
                "CI_value": fmt_num(CI,2) if CI is not None else "",
                "PAWP_value": fmt_num(PAWP_mean,0) if PAWP_mean is not None else "",
                "V_wave_short": "",
            }
        )

        # Engine-Kontext ergänzen (für Add-on-Templates)
        if engine_ctx:
            for k in (
                "dpg_phrase",
                "pac_phrase",
                "svi_phrase",
                "papi_phrase",
                "wedge_sat_qc_sentence",
                "hemo_risk_sentence",
            ):
                if k in engine_ctx and engine_ctx.get(k) is not None:
                    render_ctx[k] = engine_ctx.get(k)

        # Fallbacks, falls Engine-Kontext nicht liefert
        if "dpg_phrase" not in render_ctx:
            render_ctx["dpg_phrase"] = f"DPG {fmt_num(DPG,0)} mmHg" if DPG is not None else "DPG nicht erhoben"
        if "pac_phrase" not in render_ctx:
            render_ctx["pac_phrase"] = ""
        if "svi_phrase" not in render_ctx:
            render_ctx["svi_phrase"] = ""
        if "papi_phrase" not in render_ctx:
            render_ctx["papi_phrase"] = ""

        def safe_render(block_id: str) -> str:
            b = textdb.get_block(block_id) if hasattr(textdb, "get_block") else None
            if not b:
                return ""
            try:
                return _norm_spaces(str(b.template).format_map(render_ctx))
            except Exception:
                return _norm_spaces(str(b.template))

        # -------- Kopf / Ident --------
        dt = ctx.get("date") or "nicht angegeben"

        module_str = "Basis (Ruhe)"
        if modules:
            module_str += " + " + " + ".join(modules)

        p_last = (patient.get("last_name") or "").strip()
        p_first = (patient.get("first_name") or "").strip()
        p_birth = parse_date_yyyy_mm_dd(patient.get("birthdate"))
        ref_dt = parse_date_yyyy_mm_dd(dt) or date.today()
        p_age = calc_age_years(p_birth, ref_dt)
        p_ident = ""
        if p_last or p_first or p_birth:
            name_part = join_nonempty([p_last, p_first], ", ")
            dob_part = p_birth.isoformat() if p_birth else ""
            age_part = f"{p_age} J" if p_age is not None else ""
            p_ident = join_nonempty([name_part, dob_part, age_part], " | ")

        ph_dx_known = bool(patient.get("ph_diagnosis_known"))
        ph_dx_type = (patient.get("ph_diagnosis_type") or "").strip()
        ph_susp = bool(patient.get("ph_suspicion"))
        ph_susp_type = (patient.get("ph_suspicion_type") or "").strip()
        ph_dx_part = f"PH-Diagnose: {ph_dx_type}" if ph_dx_known and ph_dx_type else ""
        ph_susp_part = f"PH-Verdacht: {ph_susp_type}" if ph_susp and ph_susp_type else ""

        bef_kopf = join_nonempty(
            [
                exam_type_desc,
                f"Module: {module_str}",
                oxygen_header,
                f"CO via {co_method_desc}",
                f"Datum: {dt}",
                p_ident,
                ph_dx_part,
                ph_susp_part,
            ],
            " | ",
        )

        # -------- Hämodynamik --------
        pa_parts: List[str] = []
        if PA_sys is not None and PA_dia is not None:
            pa_parts.append(f"{fmt_num(PA_sys,0)}/{fmt_num(PA_dia,0)}")
        elif PA_sys is not None:
            pa_parts.append(f"sPAP {fmt_num(PA_sys,0)}")
        elif PA_dia is not None:
            pa_parts.append(f"dPAP {fmt_num(PA_dia,0)}")
        if mpap is not None:
            pa_parts.append(f"mPAP {fmt_num(mpap,0)}")
        pa_str = "PA: " + " | ".join(pa_parts) + " mmHg" if pa_parts else "PA: nicht erhoben"

        h1 = join_nonempty([f"RA_mean: {fmt_unit(RA_mean,'mmHg',0)}", pa_str, f"PAWP_mean: {fmt_unit(PAWP_mean,'mmHg',0)}"])
        h2 = join_nonempty(
            [
                f"AO_mean: {fmt_unit(aom,'mmHg',0)}",
                f"CO: {fmt_unit(CO,'L/min',2)}",
                f"CI: {fmt_unit(CI,'L/min/m²',2)}",
                f"PVR: {fmt_unit(PVR,'WU',1)} {pvr_sev_phrase}".strip(),
                f"PVRI: {fmt_unit(PVRI,'WU·m²',1)}" if PVRI is not None else "",
                f"TPG: {fmt_unit(TPG,'mmHg',0)}",
                f"DPG: {fmt_unit(DPG,'mmHg',0)}",
                f"SVR: {fmt_unit(SVR,'WU',1)}",
                f"SVRI: {fmt_unit(SVRI,'WU·m²',1)}" if SVRI is not None else "",
            ]
        )

        stufen_text = _format_step_ox(sat_SVC, sat_RA, sat_RV, sat_PA, sat_AO, step_up_present, step_up_location_desc)

        haemo_lines = [f"- {h1}", f"- {h2}"]
        if BSA is not None:
            haemo_lines.append(f"- BSA (DuBois): {fmt_num(BSA,2)} m²")
        haemo_lines.append(f"- {stufen_text}")

        if has_ex:
            ex_line = join_nonempty([
                f"Belastung: CO_peak {fmt_unit(ex_CO,'L/min',2)}",
                f"mPAP_peak {fmt_unit(ex_mPAP,'mmHg',0)}",
                f"PAWP_peak {fmt_unit(ex_PAWP,'mmHg',0)}",
                f"mPAP/CO-Slope {fmt_unit(mPAP_CO_slope,'mmHg/(L/min)',1)}",
                f"PAWP/CO-Slope {fmt_unit(PAWP_CO_slope,'mmHg/(L/min)',1)}",
                f"ΔsPAP {fmt_unit(delta_sPAP,'mmHg',0)}" if delta_sPAP is not None else "",
                f"CI_peak {fmt_unit(CI_peak,'L/min/m²',2)}" if CI_peak is not None else "",
            ], " | ")
            haemo_lines.append(f"- {ex_line}")

        if has_vol and vol_res.performed:
            haemo_lines.append(f"- {vol_res.desc}")

        if has_vaso and vaso_res.performed:
            haemo_lines.append(f"- {vaso_res.agent_desc}: {vaso_res.response_desc}")

        haemodynamik = "\n".join([ln for ln in haemo_lines if ln.strip() and not ln.endswith(": ") and not ln.endswith("|")])

        # -------- Methodik / Qualität (Auto, TextDB-Engine) --------
        methodik_lines: List[str] = []
        for bid in addon_block_ids or []:
            t = safe_render(str(bid))
            if t:
                methodik_lines.append(f"- {t}")
        methodik = "\n".join(methodik_lines) if methodik_lines else "- —"

        # -------- Beurteilung / Empfehlung --------
        beur_parts: List[str] = []

        # Strukturierte Hämodynamik-Zeile inkl. TPG (Slopes nur, wenn belastet wurde)
        struct_parts: List[str] = []
        struct_parts.append(f"mPAP {fmt_num(mpap,0)} mmHg" if mpap is not None else "mPAP nicht erhoben")
        struct_parts.append(f"PAWP {fmt_num(PAWP_mean,0)} mmHg" if PAWP_mean is not None else "PAWP nicht erhoben")
        struct_parts.append(f"TPG {fmt_num(TPG,0)} mmHg" if TPG is not None else "TPG nicht erhoben")

        pvr_struct = f"PVR {fmt_num(PVR,1)} WU" if PVR is not None else "PVR nicht erhoben"
        if PVR is not None and pvr_sev and pvr_sev != "normal":
            pvr_struct += f" ({pvr_sev})"
        struct_parts.append(pvr_struct)

        struct_parts.append(f"CI {fmt_num(CI,2)} L/min/m²" if CI is not None else "CI nicht erhoben")
        beur_parts.append("Hämodynamik (Struktur): " + ", ".join(struct_parts) + ".")

        if "Belastung" in modules:
            slope_mpap_txt = fmt_num(mPAP_CO_slope,1) if mPAP_CO_slope is not None else "nicht erhoben"
            slope_pawp_txt = fmt_num(PAWP_CO_slope,1) if PAWP_CO_slope is not None else "nicht erhoben"
            beur_parts.append(
                f"Belastung: mPAP/CO-Slope {slope_mpap_txt} mmHg/(L/min), "
                f"PAWP/CO-Slope {slope_pawp_txt} mmHg/(L/min)."
            )

        main_beur = safe_render(main_B_id) if main_B_id else ""
        if main_beur:
            beur_parts.append(main_beur)

        # Vergleichssatz (falls nicht bereits im Hauptbaustein enthalten)
        if comparison_sentence and (not main_beur or comparison_sentence not in main_beur):
            beur_parts.append(comparison_sentence)


        # Volumenchallenge als Zusatzbaustein
        if has_vol and vol_res.performed:
            if hasattr(textdb, "get_block"):
                # B11/B12 existieren in rhk_textdb als Legacy-Blöcke
                if vol_res.positive is True and textdb.get_block("B11"):
                    beur_parts.append(safe_render("B11"))
                elif vol_res.positive is False and textdb.get_block("B12"):
                    beur_parts.append(safe_render("B12"))
        # Vasoreaktivität als Zusatz
        if has_vaso and vaso_res.performed:
            if vaso_res.responder is True:
                beur_parts.append(safe_render("K17_B") or safe_render("B14"))
            elif vaso_res.responder is False:
                beur_parts.append(safe_render("K18_B") or safe_render("B13"))
            else:
                beur_parts.append(_norm_spaces(vaso_res.response_desc))

        for t in extra_beur_texts:
            if t and str(t).strip():
                beur_parts.append(str(t).strip())

        beurteilung = _norm_spaces(" ".join([p.strip() for p in beur_parts if p.strip()])) or "—"

        emp_parts: List[str] = []
        if main_E_id:
            emp_parts.append(safe_render(main_E_id))

        # Empfehlung zu Volumenchallenge (kurz)
        if has_vol and vol_res.performed and vol_res.positive is True:
            emp_parts.append("Bei positivem PAWP-Anstieg unter Volumenchallenge: HFpEF-/Linksherz-Abklärung und konsequentes Volumenmanagement empfehlen.")
        # Vasoreaktivität Empfehlung
        if has_vaso and vaso_res.performed:
            if vaso_res.responder is True:
                emp_parts.append(safe_render("K17_E") or safe_render("B14"))
            elif vaso_res.responder is False:
                emp_parts.append(safe_render("K18_E") or safe_render("B13"))

        for t in extra_empf_texts:
            if t and str(t).strip():
                emp_parts.append(str(t).strip())

        empfehlung = _norm_spaces(" ".join([p.strip() for p in emp_parts if p.strip()])) or "—"

        # Procedere (auto aus Bundle, wenn leer)
        main_bundle_id = None
        if main_B_id and main_B_id.endswith("_B") and len(main_B_id) >= 4:
            main_bundle_id = main_B_id.split("_", 1)[0]
        procedere = _render_procedere(planned_actions, render_ctx, main_bundle_id)

        # Zusatzparameter (Klinik/Labore) – optional
        functional = additional.get("functional", {}) or {}
        who_fc = functional.get("WHO_FC")
        sixmwd = _to_float(functional.get("sixmwd_m"))
        sbp = _to_float(functional.get("sbp_mmHg"))
        egfr = _to_float(functional.get("egfr_ml_min_1_73"))

        labs = additional.get("labs", {}) or {}
        bnp_kind = str(labs.get("bnp_kind") or "NT-proBNP")
        bnp_value = _to_float(labs.get("bnp_value"))
        hb = _to_float(labs.get("hb_g_dl"))
        ferritin = _to_float(labs.get("ferritin_ug_l"))
        tsat = _to_float(labs.get("tsat_pct"))

        zus_parts = []
        zus_parts.append(f"WHO-FC: {who_fc}" if who_fc else "")
        zus_parts.append(f"6MWD: {fmt_unit(sixmwd,'m',0)}" if sixmwd is not None else "")
        zus_parts.append(f"SBP: {fmt_unit(sbp,'mmHg',0)}" if sbp is not None else "")
        zus_parts.append(f"eGFR: {fmt_unit(egfr,'ml/min/1.73m²',0)}" if egfr is not None else "")
        zus_parts.append(f"{bnp_kind}: {fmt_unit(bnp_value,'',0)}".strip() if bnp_value is not None else "")
        zus_parts.append(f"Hb: {fmt_unit(hb,'g/dl',1)}" if hb is not None else "")
        zus_parts.append(f"Ferritin: {fmt_unit(ferritin,'µg/l',0)}" if ferritin is not None else "")
        zus_parts.append(f"TSAT: {fmt_unit(tsat,'%',0)}" if tsat is not None else "")

        zusatz_lines: List[str] = []
        if any(zus_parts):
            zusatz_lines.append("- " + join_nonempty(zus_parts, " | "))
        else:
            zusatz_lines.append("- —")

        # Erweiterte Zusatzbefunde (optional) – nur Ausgabe, wenn befüllt
        def yn(val: Optional[bool]) -> str:
            if val is True:
                return "ja"
            if val is False:
                return "nein"
            return "unklar"

        # Story / Kurz-Anamnese
        story_txt = (additional.get("history", {}) or {}).get("story")
        if story_txt:
            zusatz_lines.append(f"- Anamnese: {story_txt}")

        # Labor (Erweiterung)
        inr = _to_float(labs.get("inr"))
        quick = _to_float(labs.get("quick_pct"))
        krea = _to_float(labs.get("creatinine_mg_dl"))
        hst = _to_float(labs.get("harnstoff_hst_mg_dl"))
        ptt = _to_float(labs.get("ptt_s"))
        thrombos = _to_float(labs.get("thrombos_g_l"))
        crp = _to_float(labs.get("crp_mg_l"))
        leukos = _to_float(labs.get("leukos_g_l"))
        congestive = labs.get("congestive_organopathy")

        lab_ext_parts: List[str] = []
        if inr is not None:
            lab_ext_parts.append(f"INR {fmt_num(inr,2)}")
        if quick is not None:
            lab_ext_parts.append(f"Quick {fmt_num(quick,0)}%")
        if krea is not None:
            lab_ext_parts.append(f"Krea {fmt_num(krea,2)} mg/dl")
        if hst is not None:
            lab_ext_parts.append(f"Hst {fmt_num(hst,1)} mg/dl")
        if ptt is not None:
            lab_ext_parts.append(f"PTT {fmt_num(ptt,0)} s")
        if thrombos is not None:
            lab_ext_parts.append(f"Thrombos {fmt_num(thrombos,0)} G/l")
        if crp is not None:
            lab_ext_parts.append(f"CRP {fmt_num(crp,1)} mg/l")
        if leukos is not None:
            lab_ext_parts.append(f"Leukos {fmt_num(leukos,1)} G/l")
        if congestive is not None:
            lab_ext_parts.append(f"Congestive Organopathie: {yn(bool(congestive))}")

        if lab_ext_parts:
            zusatz_lines.append("- Labor: " + ", ".join(lab_ext_parts))

        # Blutgase / LTOT
        bg = (additional.get("blood_gases", {}) or {})
        if isinstance(bg, dict):
            bg_parts: List[str] = []
            ltot_present = bg.get("ltot_present")
            ltot_paused = bg.get("ltot_paused")
            rest = bg.get("rest", {}) or {}
            ex = bg.get("exercise", {}) or {}
            night = bg.get("night", {}) or {}

            if ltot_present is not None:
                bg_parts.append(f"LTOT: {yn(ltot_present)}")
            if ltot_paused:
                bg_parts.append("pausiert")
            if rest.get("pO2_mmHg") is not None or rest.get("pCO2_mmHg") is not None:
                bg_parts.append(
                    f"Ruhe pO₂ {fmt_unit(rest.get('pO2_mmHg'),'mmHg',0)}, pCO₂ {fmt_unit(rest.get('pCO2_mmHg'),'mmHg',0)}"
                )
            if ex.get("pO2_mmHg") is not None or ex.get("pCO2_mmHg") is not None:
                bg_parts.append(
                    f"Belastung pO₂ {fmt_unit(ex.get('pO2_mmHg'),'mmHg',0)}, pCO₂ {fmt_unit(ex.get('pCO2_mmHg'),'mmHg',0)}"
                )
            if night.get("pH") is not None or night.get("BE_mmol_l") is not None:
                ph_txt = fmt_num(night.get("pH"),2) if night.get("pH") is not None else "n/a"
                bg_parts.append(f"Nacht pH {ph_txt}, BE {fmt_unit(night.get('BE_mmol_l'),'mmol/l',1)}")

            if any(p for p in bg_parts if p):
                zusatz_lines.append("- Blutgase/LTOT: " + "; ".join([p for p in bg_parts if p]))

        # Infektiologie / Immunologie
        ii = (additional.get("infection_immunology", {}) or {})
        if isinstance(ii, dict) and (ii.get("virology_positive") is not None or ii.get("immunology_positive") is not None):
            ii_parts: List[str] = []
            if ii.get("virology_positive") is not None:
                ii_parts.append(f"Virologie positiv: {yn(ii.get('virology_positive'))}")
            if ii.get("immunology_positive") is not None:
                ii_parts.append(f"Immunologie positiv: {yn(ii.get('immunology_positive'))}")
            if ii_parts:
                zusatz_lines.append("- Infekt/Immuno: " + ", ".join(ii_parts))

        # Abdomen / Leber
        ab = (additional.get("abdomen_liver", {}) or {})
        if isinstance(ab, dict) and (ab.get("abdomen_sono_done") is not None or ab.get("portal_hypertension_hint") is not None):
            ab_parts: List[str] = []
            if ab.get("abdomen_sono_done") is not None:
                ab_parts.append(f"Abdomen-Sono: {yn(ab.get('abdomen_sono_done'))}")
            if ab.get("portal_hypertension_hint") is not None:
                ab_parts.append(f"Portale Hypertension: {yn(ab.get('portal_hypertension_hint'))}")
            if ab_parts:
                zusatz_lines.append("- Abdomen/Leber: " + ", ".join(ab_parts))

        # CT / Bildgebung Thorax
        ct = (additional.get("ct_imaging", {}) or {})
        if isinstance(ct, dict):
            ct_flags: List[str] = []
            if ct.get("ct_angio"):
                ct_flags.append("CT-Angio")
            if ct.get("lae"):
                ct_flags.append("LAE")
            if ct.get("ild"):
                ct_flags.append("ILD")
            if ct.get("emphysema"):
                ct_flags.append("Emphysem")
            if ct.get("embolism"):
                ct_flags.append("Embolie")
            if ct.get("mosaic_perfusion"):
                ct_flags.append("Mosaikperfusion")
            if ct.get("coronary_calcification"):
                ct_flags.append("Koronarkalk")
            if ct.get("pericardial_effusion"):
                ct_flags.append("Perikarderguss")

            ct_extra: List[str] = []
            if ct.get("ventricular"):
                ct_extra.append(f"ventrikulär: {ct.get('ventricular')}")
            if ct.get("cardiac_phenotype"):
                ct_extra.append(str(ct.get("cardiac_phenotype")))
            if ct.get("ild_desc"):
                ct_extra.append(f"ILD: {ct.get('ild_desc')}")
            if ct.get("emphysema_extent"):
                ct_extra.append(f"Emphysem: {ct.get('emphysema_extent')}")

            if ct_flags or ct_extra:
                ct_str = ", ".join(ct_flags)
                if ct_extra:
                    ct_str = (ct_str + " | " if ct_str else "") + "; ".join(ct_extra)
                zusatz_lines.append("- CT/Bildgebung: " + ct_str)

        # Vorerkrankungen
        com = (additional.get("comorbidities", {}) or {})
        if isinstance(com, dict) and (com.get("text") or com.get("ph_relevance")):
            com_line = (com.get("text") or "").strip()
            if com.get("ph_relevance"):
                com_line = (com_line + " | " if com_line else "") + f"PH-relevant: {com.get('ph_relevance')}"
            zusatz_lines.append("- Vorerkrankungen: " + com_line)

        # Medikamente
        meds = (additional.get("medications", {}) or {})
        if isinstance(meds, dict) and (
            meds.get("ph_current") is not None
            or meds.get("ph_current_desc")
            or meds.get("ph_current_since")
            or meds.get("ph_past") is not None
            or meds.get("ph_past_desc")
            or meds.get("other")
            or meds.get("diuretics") is not None
        ):
            m_parts: List[str] = []
            if meds.get("ph_current") is not None:
                cur = f"PH-Med: {yn(meds.get('ph_current'))}"
                if meds.get("ph_current_desc"):
                    cur += f" ({meds.get('ph_current_desc')})"
                if meds.get("ph_current_since"):
                    cur += f", seit {meds.get('ph_current_since')}"
                m_parts.append(cur)
            if meds.get("ph_past") is not None:
                past = f"PH-Med früher: {yn(meds.get('ph_past'))}"
                if meds.get("ph_past_desc"):
                    past += f" ({meds.get('ph_past_desc')})"
                m_parts.append(past)
            if meds.get("diuretics") is not None:
                m_parts.append(f"Diuretika: {yn(meds.get('diuretics'))}")
            if meds.get("other"):
                m_parts.append(f"Sonstige: {meds.get('other')}")
            if m_parts:
                zusatz_lines.append("- Medikamente: " + " | ".join(m_parts))

        # Lungenfunktion
        lf = (additional.get("lung_function", {}) or {})
        if isinstance(lf, dict) and (
            lf.get("done") is not None
            or any((lf.get("phenotype") or {}).values())
            or any(v is not None for v in (lf.get("values") or {}).values())
        ):
            lf_parts: List[str] = []
            if lf.get("done") is not None:
                lf_parts.append(f"durchgeführt: {yn(lf.get('done'))}")

            pheno = lf.get("phenotype") or {}
            pheno_list: List[str] = []
            if pheno.get("obstructive"):
                pheno_list.append("obstruktiv")
            if pheno.get("restrictive"):
                pheno_list.append("restriktiv")
            if pheno.get("diffusion"):
                pheno_list.append("Diffusionsstörung")
            if pheno_list:
                lf_parts.append("Phänotyp: " + ", ".join(pheno_list))

            vals = lf.get("values") or {}
            val_list: List[str] = []
            if vals.get("fev1") is not None:
                val_list.append(f"FEV1 {fmt_num(vals.get('fev1'),0)}")
            if vals.get("fvc") is not None:
                val_list.append(f"FVC {fmt_num(vals.get('fvc'),0)}")
            if vals.get("fev1_fvc") is not None:
                val_list.append(f"FEV1/FVC {fmt_num(vals.get('fev1_fvc'),2)}")
            if vals.get("dlco_sb") is not None:
                val_list.append(f"DLCO {fmt_num(vals.get('dlco_sb'),0)}")
            if val_list:
                lf_parts.append(", ".join(val_list))

            if lf_parts:
                zusatz_lines.append("- Lungenfunktion: " + " | ".join(lf_parts))

        # Echokardiographie
        echo = (additional.get("echocardiography", {}) or {})
        if isinstance(echo, dict) and (echo.get("done") is not None or echo.get("params")):
            e_line = f"Echo-Phänotyp: {yn(echo.get('done'))}" if echo.get("done") is not None else ""
            if echo.get("params"):
                e_line = (e_line + " | " if e_line else "") + str(echo.get("params"))
            zusatz_lines.append("- Echo: " + (e_line if e_line else "—"))

        # Funktionelle Tests (Erweiterung)
        syn = functional.get("syncope")
        cpet_ve = _to_float(functional.get("cpet_ve_vco2"))
        cpet_vo2 = _to_float(functional.get("cpet_vo2max"))
        ft_parts: List[str] = []
        if syn is not None:
            ft_parts.append(f"Synkope: {yn(syn)}")
        if cpet_ve is not None:
            ft_parts.append(f"VE/VCO₂ {fmt_num(cpet_ve,1)}")
        if cpet_vo2 is not None:
            ft_parts.append(f"VO₂max {fmt_num(cpet_vo2,1)}")
        if ft_parts:
            zusatz_lines.append("- Funktionell: " + ", ".join(ft_parts))

        # MRT / CMR
        cmr = (additional.get("cmr", {}) or {})
        if isinstance(cmr, dict) and (cmr.get("rvesvi") is not None or cmr.get("svi") is not None or cmr.get("rvef") is not None):
            cmr_parts: List[str] = []
            if cmr.get("rvesvi") is not None:
                cmr_parts.append(f"RVESVi {fmt_num(cmr.get('rvesvi'),0)} ml/m²")
            if cmr.get("svi") is not None:
                cmr_parts.append(f"SVi {fmt_num(cmr.get('svi'),0)} ml/m²")
            if cmr.get("rvef") is not None:
                cmr_parts.append(f"RVEF {fmt_num(cmr.get('rvef'),0)}%")
            if cmr_parts:
                zusatz_lines.append("- CMR: " + ", ".join(cmr_parts))

        # Abschluss (Freitext)
        closing = (additional.get("closing", {}) or {})
        if isinstance(closing, dict) and closing.get("suggestion"):
            zusatz_lines.append("- Abschluss: " + str(closing.get("suggestion")))

        zusatz = "\n".join(zusatz_lines)


        # Plausibilität
        plaus_lines: List[str] = []
        for w in plaus_warnings:
            plaus_lines.append(f"- Plausibilitätswarnung: {w}")
        for m in missing:
            plaus_lines.append(f"- Fehlend/unklar: {m}")
        if calc_steps:
            plaus_lines.append("- Rechenweg (kurz):")
            for c in calc_steps:
                plaus_lines.append(f"  - {c}")
        plaus = "\n".join(plaus_lines) if plaus_lines else "- —"

        main_out = (
            "BEFUNDKOPF\n"
            f"{bef_kopf}\n\n"
            "HÄMODYNAMIK\n"
            f"{haemodynamik}\n\n"
            "METHODIK / QUALITÄT (Auto)\n"
            f"{methodik}\n\n"
            "KLINIK / LABORE (optional)\n"
            f"{zusatz}\n\n"
            "BEURTEILUNG\n"
            f"{beurteilung}\n\n"
            "EMPFEHLUNG\n"
            f"{empfehlung}\n\n"
            "PROCEDERE\n"
            f"{procedere}\n\n"
            "PLAUSIBILITÄT / FEHLENDE ANGABEN\n"
            f"{plaus}\n"
        )

        # Interner Befund: Prozedur
        internal_lines: List[str] = []
        internal_lines.append("INTERNER BEFUND (nicht in Patientenbefund kopieren)\n")
        proc = internal.get("procedure", {}) or {}
        if proc:
            internal_lines.append("PROZEDUR / LOGISTIK")
            consent = proc.get("consent_done")
            internal_lines.append(f"- Aufklärung erfolgt: {'ja' if consent else 'nein' if consent is False else 'unklar'}")
            anticoag = proc.get("anticoagulation")
            if anticoag:
                internal_lines.append(f"- Antikoagulation: {anticoag}")
            access = proc.get("access_site")
            if access:
                internal_lines.append(f"- Zugang: {access}")
            internal_lines.append("")

        # TextDB-Engine (Auto) – Debug/Transparenz
        if engine_plan or addon_block_ids:
            internal_lines.append("TEXTDB-ENGINE (Auto)")
            try:
                rest_cls = (engine_plan.get("rest") or {}).get("rest_class") if isinstance(engine_plan, dict) else None
            except Exception:
                rest_cls = None
            if rest_cls:
                internal_lines.append(f"- rest_class: {rest_cls}")
            # PAWP-Unsicherheitsgründe
            try:
                pawp_unc = (engine_plan.get("pawp_uncertainty") or {}) if isinstance(engine_plan, dict) else {}
                reasons = pawp_unc.get("reasons") or []
                if reasons:
                    internal_lines.append("- PAWP-Unsicherheit: " + "; ".join([str(r) for r in reasons]))
            except Exception:
                pass
            if addon_block_ids:
                internal_lines.append("- Add-on-Blöcke: " + ", ".join([str(x) for x in addon_block_ids]))
            internal_lines.append("")

        internal_out = "\n".join(internal_lines).rstrip() + "\n"

        # Risiko html (Preview)
        esc3 = esc3_overall(who_fc, sixmwd, bnp_kind, bnp_value)
        esc4 = esc4_overall(who_fc, sixmwd, bnp_kind, bnp_value)
        reveal = reveal_lite2_score(who_fc, sixmwd, bnp_kind, bnp_value, sbp, HR, egfr)
        risk_html = render_risk_html(esc3, esc4, reveal)

        return main_out, internal_out, risk_html


generator = RHKReportGenerator()

_DEF = getattr(textdb, "DEFAULT_RULES", {}) or {}


def _def(path: str, fallback: Optional[float]) -> Optional[float]:
    try:
        return _rule(_DEF, path, fallback)
    except Exception:
        return fallback


# -----------------------------
# UI -> JSON builder
# -----------------------------

def _build_data_from_ui(
    dt: str,
    exam_type: str,
    modules: List[str],
    oxygen_mode: str,
    oxygen_flow: Optional[float],
    co_method: str,
    last_name: str,
    first_name: str,
    birthdate: str,
    height_cm: Optional[float],
    weight_kg: Optional[float],
    ph_dx_known: bool,
    ph_dx_type: str,
    ph_susp: bool,
    ph_susp_type: str,
    # Ruhe
    ra_mean: Optional[float],
    pa_sys: Optional[float],
    pa_dia: Optional[float],
    pa_mean: Optional[float],
    pawp_mean: Optional[float],
    ao_sys: Optional[float],
    ao_dia: Optional[float],
    ao_mean: Optional[float],
    co: Optional[float],
    ci_in: Optional[float],
    hr: Optional[float],
    # Sats
    sat_svc: Optional[float],
    sat_ra: Optional[float],
    sat_rv: Optional[float],
    sat_pa: Optional[float],
    sat_ao: Optional[float],
    step_up_mode: str,
    step_up_loc_override: str,
    # Prozedur
    consent_done: str,
    anticoag_yes: str,
    anticoag_desc: str,
    access_site: str,
    # Belastung
    ex_co: Optional[float],
    ex_mpap: Optional[float],
    ex_pawp: Optional[float],
    ex_spap: Optional[float],
    # Volumenchallenge
    vc_volume_ml: Optional[float],
    vc_infusion: str,
    vc_pawp_post: Optional[float],
    vc_mpap_post: Optional[float],
    # Vasoreaktivität
    vaso_agent: str,
    vaso_ino_ppm: Optional[float],
    vaso_mpap_post: Optional[float],
    vaso_co_post: Optional[float],
    # Klinik/Labore/Risiko
    who_fc: str,
    sixmwd_m: Optional[float],
    sbp_mmHg: Optional[float],
    egfr_ml_min_1_73: Optional[float],
    bnp_kind: str,
    bnp_value: Optional[float],
    hb_g_dl: Optional[float],
    ferritin_ug_l: Optional[float],
    tsat_pct: Optional[float],
    cteph_suspected: bool,
    cteph_context_desc: str,
    left_heart_context_desc: str,
    pvod_hint_desc: str,
    lufu_summary: str,
    # Procedere
    planned_choice: List[str],
    planned_free: str,
    # Advanced / Cutoffs
    use_guideline_cutoffs: bool,
    mpap_cut: Optional[float],
    pawp_cut: Optional[float],
    pvr_cut: Optional[float],
    slope_mpap: Optional[float],
    slope_pawp: Optional[float],
    pvr_mild: Optional[float],
    pvr_mod: Optional[float],
    pvr_sev: Optional[float],
    ci_low: Optional[float],
    # Stepox thresholds
    stepox_thr_ra: Optional[float],
    stepox_thr_rv: Optional[float],
    stepox_thr_pa: Optional[float],
    # Volume thresholds
    vc_pawp_post_thr: Optional[float],
    vc_delta_pawp_thr: Optional[float],
    *extra: Any,
) -> Dict[str, Any]:

    ctx = {
        "date": (dt or "").strip() or None,
        "exam_type": exam_type,
        "modules": modules or [],
        "oxygen": {"mode": oxygen_mode, "flow_l_min": oxygen_flow},
        "co_method": co_method,
    }
    patient = {
        "last_name": (last_name or "").strip() or None,
        "first_name": (first_name or "").strip() or None,
        "birthdate": (birthdate or "").strip() or None,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "ph_diagnosis_known": bool(ph_dx_known),
        "ph_diagnosis_type": (ph_dx_type or "").strip() if ph_dx_known else None,
        "ph_suspicion": bool(ph_susp),
        "ph_suspicion_type": (ph_susp_type or "").strip() if ph_susp else None,
    }

    raw_values = {
        "pressures_mmHg": {
            "RA_mean": ra_mean,
            "PA_sys": pa_sys,
            "PA_dia": pa_dia,
            "PA_mean": pa_mean,
            "PAWP_mean": pawp_mean,
            "AO_sys": ao_sys,
            "AO_dia": ao_dia,
            "AO_mean": ao_mean,
        },
        "flow": {"CO_L_min": co, "CI_L_min_m2": ci_in, "HR_min": hr},
        "sats_pct": {"SVC": sat_svc, "RA": sat_ra, "RV": sat_rv, "PA": sat_pa, "AO": sat_ao},
    }

    interpretation_flags = {
        "step_up_mode": (step_up_mode or "auto"),
        "step_up_location_override": (step_up_loc_override or "").strip() or None,
        "ino_responder": None,
        "volume_pawp_rise_significant": None,
    }

    # Intern procedure
    consent_bool = True if consent_done == "ja" else False if consent_done == "nein" else None
    anticoag_bool = True if anticoag_yes == "ja" else False if anticoag_yes == "nein" else None
    anticoag_text = (anticoag_desc or "").strip() or None
    if anticoag_bool is True:
        anticoag_text = anticoag_text or "ja"
    elif anticoag_bool is False:
        anticoag_text = "nein"
    access_site_txt = (access_site or "").strip() or None
    internal = {"procedure": {"consent_done": consent_bool, "anticoagulation": anticoag_text, "access_site": access_site_txt}}

    # Planned actions
    planned_actions: List[str] = []
    for ch in planned_choice or []:
        pid = ch.split(" – ", 1)[0].strip()
        if pid:
            planned_actions.append(pid)
    for ln in (planned_free or "").splitlines():
        ln = ln.strip()
        if ln:
            planned_actions.append(ln)

    additional_measurements = {
        "exercise_peak": {"CO_L_min": ex_co, "mPAP_mmHg": ex_mpap, "PAWP_mmHg": ex_pawp, "sPAP_mmHg": ex_spap},
        "volume_challenge": {"volume_ml": vc_volume_ml, "infusion_type": vc_infusion, "PAWP_post": vc_pawp_post, "mPAP_post": vc_mpap_post},
        "vasoreactivity": {"agent": vaso_agent, "ino_ppm": vaso_ino_ppm, "mPAP_post": vaso_mpap_post, "CO_post": vaso_co_post},
        "functional": {"WHO_FC": (who_fc or "").strip() or None, "sixmwd_m": sixmwd_m, "sbp_mmHg": sbp_mmHg, "egfr_ml_min_1_73": egfr_ml_min_1_73},
        "labs": {
            "bnp_kind": bnp_kind,
            "bnp_value": bnp_value,
            "hb_g_dl": hb_g_dl,
            "ferritin_ug_l": ferritin_ug_l,
            "tsat_pct": tsat_pct,
        },
        "lufu_summary": (lufu_summary or "").strip() or None,
    }

    clinical_context = {
        "ctepd_cteph_suspected": bool(cteph_suspected),
        "cteph_context_desc": (cteph_context_desc or "").strip() or None,
        "left_heart_context_desc": (left_heart_context_desc or "").strip() or None,
        "pvod_hint_desc": (pvod_hint_desc or "").strip() or None,
    }

    local_rules = {
        "use_guideline_cutoffs": bool(use_guideline_cutoffs),
        "rules": {
            "rest": {"mPAP_ph_mmHg": mpap_cut, "PAWP_postcap_mmHg": pawp_cut, "PVR_precap_WU": pvr_cut},
            "exercise": {"mPAP_CO_slope_mmHg_per_L_min": slope_mpap, "PAWP_CO_slope_mmHg_per_L_min": slope_pawp},
            "severity": {"PVR_mild_from_WU": pvr_mild, "PVR_moderate_from_WU": pvr_mod, "PVR_severe_from_WU": pvr_sev, "CI_low_lt_L_min_m2": ci_low},
            "stepox": {"thr_ra_pct": stepox_thr_ra, "thr_rv_pct": stepox_thr_rv, "thr_pa_pct": stepox_thr_pa},
            "volume": {"pawp_post_thr_mmHg": vc_pawp_post_thr, "delta_pawp_thr_mmHg": vc_delta_pawp_thr},
        },
    }

    
    # ------------------------------------------------------------------
    # Erweiterte optionale Felder (am Ende der GUI angehängt)
    # Hinweis: Wird hier zentral gesammelt, damit die Beurteilung/Empfehlung
    # später alles "irgendwo" verwerten bzw. ausgeben kann.
    # ------------------------------------------------------------------
    EXTRA_KEYS = [
        "story",
        "lab_inr", "lab_quick", "lab_krea", "lab_hst", "lab_ptt", "lab_thrombos", "lab_crp", "lab_leukos",
        "congestive_organopathy",
        "ltot_present", "ltot_paused",
        "bga_rest_po2", "bga_rest_pco2", "bga_ex_po2", "bga_ex_pco2", "bga_night_ph", "bga_night_be",
        "virology_positive", "immunology_positive",
        "abdomen_sono_done", "portal_hypertension",
        "ct_angio", "ct_lae", "ct_ild", "ct_ild_desc", "ct_emphysema", "ct_emphysema_extent",
        "ct_embolism", "ct_mosaic", "ct_coronary_calc", "ct_pericardial_effusion",
        "ct_ventricular", "ct_cardiac_phenotype",
        "comorbidities", "ph_relevance",
        "ph_meds_current", "ph_meds_current_desc", "ph_meds_since",
        "ph_meds_past", "ph_meds_past_desc",
        "meds_other", "diuretics_yes",
        "lufu_done", "lufu_obst", "lufu_restr", "lufu_diff",
        "lufu_fev1", "lufu_fvc", "lufu_fev1_fvc", "lufu_tlc", "lufu_rv", "lufu_dlco_sb", "lufu_dlco_va",
        "lufu_po2", "lufu_pco2", "lufu_ph", "lufu_be",
        "echo_done", "echo_params",
        "syncope", "cpet_ve_vco2", "cpet_vo2max",
        "cmr_rvesvi", "cmr_svi", "cmr_rvef",
        "prev_rhk_label", "prev_course_desc", "prev_mpap", "prev_pawp", "prev_ci", "prev_pvr",
        "closing_suggestion",
        # Messqualität / PAWP-Validierung (optional)
        "wedge_sat_pct", "qc_resp_swings_large", "qc_obesity", "qc_copd", "qc_mechanical_ventilation",
    ]

    extra_map = dict(zip(EXTRA_KEYS, extra)) if extra else {}

    # Story / Kurz-Anamnese
    story = (extra_map.get("story") or "").strip()
    if story:
        additional_measurements["history"] = {"story": story}

    # Messqualität / PAWP-Validierung (optional)
    # (wird von der TextDB-Engine genutzt, um PAWP-Unsicherheit / Under-wedge etc. abzubilden)
    q = {
        "wedge_sat_pct": _to_float(extra_map.get("wedge_sat_pct")),
        "resp_swings_large": bool(extra_map.get("qc_resp_swings_large")),
        "obesity": bool(extra_map.get("qc_obesity")),
        "copd": bool(extra_map.get("qc_copd")),
        "mechanical_ventilation": bool(extra_map.get("qc_mechanical_ventilation")),
    }
    # None raus, booleans bleiben (True/False) – so ist explizit dokumentiert, ob gesetzt.
    q = {k: v for k, v in q.items() if v is not None}
    if q:
        additional_measurements["quality"] = q

    if extra_map:
        # Labor (Erweiterung)
        labs_ext = additional_measurements.get("labs", {}) or {}
        labs_ext.update(
            {
                "inr": _to_float(extra_map.get("lab_inr")),
                "quick_pct": _to_float(extra_map.get("lab_quick")),
                "creatinine_mg_dl": _to_float(extra_map.get("lab_krea")),
                "harnstoff_hst_mg_dl": _to_float(extra_map.get("lab_hst")),
                "ptt_s": _to_float(extra_map.get("lab_ptt")),
                "thrombos_g_l": _to_float(extra_map.get("lab_thrombos")),
                "crp_mg_l": _to_float(extra_map.get("lab_crp")),
                "leukos_g_l": _to_float(extra_map.get("lab_leukos")),
                "congestive_organopathy": bool(extra_map.get("congestive_organopathy"))
                if extra_map.get("congestive_organopathy") is not None
                else None,
            }
        )
        additional_measurements["labs"] = labs_ext

        # Blutgase / LTOT
        additional_measurements["blood_gases"] = {
            "ltot_present": bool(extra_map.get("ltot_present")) if extra_map.get("ltot_present") is not None else None,
            "ltot_paused": bool(extra_map.get("ltot_paused")) if extra_map.get("ltot_paused") is not None else None,
            "rest": {
                "pO2_mmHg": _to_float(extra_map.get("bga_rest_po2")),
                "pCO2_mmHg": _to_float(extra_map.get("bga_rest_pco2")),
            },
            "exercise": {
                "pO2_mmHg": _to_float(extra_map.get("bga_ex_po2")),
                "pCO2_mmHg": _to_float(extra_map.get("bga_ex_pco2")),
            },
            "night": {
                "pH": _to_float(extra_map.get("bga_night_ph")),
                "BE_mmol_l": _to_float(extra_map.get("bga_night_be")),
            },
        }

        # Infektiologie / Immunologie
        additional_measurements["infection_immunology"] = {
            "virology_positive": bool(extra_map.get("virology_positive"))
            if extra_map.get("virology_positive") is not None
            else None,
            "immunology_positive": bool(extra_map.get("immunology_positive"))
            if extra_map.get("immunology_positive") is not None
            else None,
        }

        # Abdomen / Leber
        additional_measurements["abdomen_liver"] = {
            "abdomen_sono_done": bool(extra_map.get("abdomen_sono_done"))
            if extra_map.get("abdomen_sono_done") is not None
            else None,
            "portal_hypertension_hint": bool(extra_map.get("portal_hypertension"))
            if extra_map.get("portal_hypertension") is not None
            else None,
        }

        # CT / Bildgebung Thorax
        additional_measurements["ct_imaging"] = {
            "ct_angio": bool(extra_map.get("ct_angio")) if extra_map.get("ct_angio") is not None else None,
            "lae": bool(extra_map.get("ct_lae")) if extra_map.get("ct_lae") is not None else None,
            "ild": bool(extra_map.get("ct_ild")) if extra_map.get("ct_ild") is not None else None,
            "ild_desc": (extra_map.get("ct_ild_desc") or "").strip() or None,
            "emphysema": bool(extra_map.get("ct_emphysema")) if extra_map.get("ct_emphysema") is not None else None,
            "emphysema_extent": (extra_map.get("ct_emphysema_extent") or "").strip() or None,
            "embolism": bool(extra_map.get("ct_embolism")) if extra_map.get("ct_embolism") is not None else None,
            "mosaic_perfusion": bool(extra_map.get("ct_mosaic")) if extra_map.get("ct_mosaic") is not None else None,
            "coronary_calcification": bool(extra_map.get("ct_coronary_calc"))
            if extra_map.get("ct_coronary_calc") is not None
            else None,
            "pericardial_effusion": bool(extra_map.get("ct_pericardial_effusion"))
            if extra_map.get("ct_pericardial_effusion") is not None
            else None,
            "ventricular": (extra_map.get("ct_ventricular") or "").strip() or None,
            "cardiac_phenotype": (extra_map.get("ct_cardiac_phenotype") or "").strip() or None,
        }

        # Vorerkrankungen
        additional_measurements["comorbidities"] = {
            "text": (extra_map.get("comorbidities") or "").strip() or None,
            "ph_relevance": (extra_map.get("ph_relevance") or "").strip() or None,
        }

        # Medikamente
        additional_measurements["medications"] = {
            "ph_current": bool(extra_map.get("ph_meds_current"))
            if extra_map.get("ph_meds_current") is not None
            else None,
            "ph_current_desc": (extra_map.get("ph_meds_current_desc") or "").strip() or None,
            "ph_current_since": (extra_map.get("ph_meds_since") or "").strip() or None,
            "ph_past": bool(extra_map.get("ph_meds_past")) if extra_map.get("ph_meds_past") is not None else None,
            "ph_past_desc": (extra_map.get("ph_meds_past_desc") or "").strip() or None,
            "other": (extra_map.get("meds_other") or "").strip() or None,
            "diuretics": bool(extra_map.get("diuretics_yes"))
            if extra_map.get("diuretics_yes") is not None
            else None,
        }

        # Lungenfunktion (strukturiert)
        additional_measurements["lung_function"] = {
            "done": bool(extra_map.get("lufu_done")) if extra_map.get("lufu_done") is not None else None,
            "phenotype": {
                "obstructive": bool(extra_map.get("lufu_obst")) if extra_map.get("lufu_obst") is not None else None,
                "restrictive": bool(extra_map.get("lufu_restr")) if extra_map.get("lufu_restr") is not None else None,
                "diffusion": bool(extra_map.get("lufu_diff")) if extra_map.get("lufu_diff") is not None else None,
            },
            "values": {
                "fev1": _to_float(extra_map.get("lufu_fev1")),
                "fvc": _to_float(extra_map.get("lufu_fvc")),
                "fev1_fvc": _to_float(extra_map.get("lufu_fev1_fvc")),
                "tlc": _to_float(extra_map.get("lufu_tlc")),
                "rv": _to_float(extra_map.get("lufu_rv")),
                "dlco_sb": _to_float(extra_map.get("lufu_dlco_sb")),
                "dlco_va": _to_float(extra_map.get("lufu_dlco_va")),
                "pO2": _to_float(extra_map.get("lufu_po2")),
                "pCO2": _to_float(extra_map.get("lufu_pco2")),
                "pH": _to_float(extra_map.get("lufu_ph")),
                "BE": _to_float(extra_map.get("lufu_be")),
            },
        }

        # Falls lufu_summary nicht als Freitext gesetzt ist: aus strukturierten Werten ableiten
        if not (clinical_context.get("lufu_summary") or "").strip():
            lf = additional_measurements.get("lung_function") or {}
            pheno = (lf.get("phenotype") or {})
            vals = (lf.get("values") or {})

            pheno_parts = []
            if pheno.get("obstructive"):
                pheno_parts.append("obstruktiv")
            if pheno.get("restrictive"):
                pheno_parts.append("restriktiv")
            if pheno.get("diffusion"):
                pheno_parts.append("Diffusionsstörung")

            val_parts = []
            if vals.get("fev1") is not None:
                val_parts.append(f"FEV1 {fmt_num(vals.get('fev1'), 0)}")
            if vals.get("fvc") is not None:
                val_parts.append(f"FVC {fmt_num(vals.get('fvc'), 0)}")
            if vals.get("dlco_sb") is not None:
                val_parts.append(f"DLCO {fmt_num(vals.get('dlco_sb'), 0)}")

            lf_summary = ""
            if pheno_parts:
                lf_summary += "Phänotyp: " + ", ".join(pheno_parts)
            if val_parts:
                lf_summary = (lf_summary + "; " if lf_summary else "") + ", ".join(val_parts)
            if lf_summary:
                clinical_context["lufu_summary"] = lf_summary

        # Echo
        additional_measurements["echocardiography"] = {
            "done": bool(extra_map.get("echo_done")) if extra_map.get("echo_done") is not None else None,
            "params": (extra_map.get("echo_params") or "").strip() or None,
        }

        # Funktionelle Tests (Erweiterung)
        func_ext = additional_measurements.get("functional", {}) or {}
        func_ext.update(
            {
                "syncope": bool(extra_map.get("syncope")) if extra_map.get("syncope") is not None else None,
                "cpet_ve_vco2": _to_float(extra_map.get("cpet_ve_vco2")),
                "cpet_vo2max": _to_float(extra_map.get("cpet_vo2max")),
            }
        )
        additional_measurements["functional"] = func_ext

        # MRT / CMR
        additional_measurements["cmr"] = {
            "rvesvi": _to_float(extra_map.get("cmr_rvesvi")),
            "svi": _to_float(extra_map.get("cmr_svi")),
            "rvef": _to_float(extra_map.get("cmr_rvef")),
        }

        # Vor-RHK
        additional_measurements["previous_rhc"] = {
            "label": (extra_map.get("prev_rhk_label") or "").strip() or None,
            "course_desc": (extra_map.get("prev_course_desc") or "").strip() or None,
            "mpap_mmHg": _to_float(extra_map.get("prev_mpap")),
            "pawp_mmHg": _to_float(extra_map.get("prev_pawp")),
            "ci_L_min_m2": _to_float(extra_map.get("prev_ci")),
            "pvr_WU": _to_float(extra_map.get("prev_pvr")),
        }

        # Abschluss
        closing = (extra_map.get("closing_suggestion") or "").strip()
        if closing:
            additional_measurements["closing"] = {"suggestion": closing}

    return {
        "context": ctx,
        "patient": patient,
        "raw_values": raw_values,
        "derived_values": {},
        "interpretation_flags": interpretation_flags,
        "planned_actions": planned_actions,
        "local_rules": local_rules,
        "qualitative": {},  # aktuell leer; kann später angebunden werden
        "clinical_context": clinical_context,
        "additional_measurements": additional_measurements,
        "internal": internal,
    }


def _generate_all_outputs(*inputs):
    try:
        data = _build_data_from_ui(*inputs)
        main, internal, risk_html = generator.generate_all(data)
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        return main, internal, risk_html, json_str
    except Exception:
        tb = traceback.format_exc()
        err = "FEHLER bei der Befundgenerierung:\n" + tb
        # Fehlertext in alle Text-Ausgaben spiegeln, damit im UI sicher etwas erscheint
        return err, err, "", err



def _download_text(txt: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix="rhk_", suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt or "")
    return path


def _download_json(json_str: str) -> str:
    return _download_text(json_str, ".json")


def _download_report(report_str: str) -> str:
    return _download_text(report_str, ".txt")


def _download_internal(internal_str: str) -> str:
    return _download_text(internal_str, "_intern.txt")


# -----------------------------
# Live-Preview helpers (UI)
# -----------------------------

def ui_age_preview(birthdate_str: str, exam_date_str: str) -> str:
    dob = parse_date_yyyy_mm_dd(birthdate_str)
    dt = parse_date_yyyy_mm_dd(exam_date_str) or date.today()
    age = calc_age_years(dob, dt)
    if dob is None:
        return "**Alter:** —"
    return f"**Alter:** {age} Jahre (Stand {dt.isoformat()})"


def ui_anthro_preview(height_cm: Optional[float], weight_kg: Optional[float]) -> str:
    h = _to_float(height_cm)
    w = _to_float(weight_kg)
    bsa = calc_bsa_dubois(h, w).value
    bmi = calc_bmi(h, w)
    parts: List[str] = []
    parts.append(f"**BSA (DuBois):** {fmt_num(bsa, 2)} m²" if bsa is not None else "**BSA (DuBois):** —")
    parts.append(f"**BMI:** {fmt_num(bmi, 1)} kg/m²" if bmi is not None else "**BMI:** —")
    return "  ".join(parts)


def ui_calc_preview(
    height_cm: Optional[float],
    weight_kg: Optional[float],
    pa_sys: Optional[float],
    pa_dia: Optional[float],
    pa_mean: Optional[float],
    pawp_mean: Optional[float],
    ao_sys: Optional[float],
    ao_dia: Optional[float],
    ao_mean: Optional[float],
    ra_mean: Optional[float],
    co: Optional[float],
    ci_in: Optional[float],
    sat_svc: Optional[float],
    sat_ra: Optional[float],
    sat_rv: Optional[float],
    sat_pa: Optional[float],
    modules: List[str],
    ex_co: Optional[float],
    ex_mpap: Optional[float],
    ex_pawp: Optional[float],
    vc_volume_ml: Optional[float],
    vc_pawp_post: Optional[float],
    vaso_mpap_post: Optional[float],
    vaso_co_post: Optional[float],
    stepox_thr_ra: Optional[float],
    stepox_thr_rv: Optional[float],
    stepox_thr_pa: Optional[float],
) -> str:
    rules = dict(getattr(textdb, "DEFAULT_RULES", {}) or {})
    rules = _deep_merge(rules, {"stepox": {"thr_ra_pct": stepox_thr_ra, "thr_rv_pct": stepox_thr_rv, "thr_pa_pct": stepox_thr_pa}})

    h = _to_float(height_cm)
    w = _to_float(weight_kg)
    bsa = calc_bsa_dubois(h, w).value
    mpap = _to_float(pa_mean)
    if mpap is None:
        mpap = calc_mean(_to_float(pa_sys), _to_float(pa_dia)).value

    aom = _to_float(ao_mean)
    if aom is None:
        aom = calc_mean(_to_float(ao_sys), _to_float(ao_dia)).value

    ci = _to_float(ci_in)
    if ci is None:
        ci = calc_ci(_to_float(co), bsa).value

    pvr = calc_pvr(mpap, _to_float(pawp_mean), _to_float(co)).value
    tpg = calc_tpg(mpap, _to_float(pawp_mean)).value
    dpg = calc_dpg(_to_float(pa_dia), _to_float(pawp_mean)).value
    pvri = calc_pvri(mpap, _to_float(pawp_mean), ci).value
    svr = calc_svr(aom, _to_float(ra_mean), _to_float(co)).value

    step = detect_step_up(_to_float(sat_svc), _to_float(sat_ra), _to_float(sat_rv), _to_float(sat_pa), rules)

    # exercise slopes
    has_ex = "Belastung" in (modules or [])
    s_mpap = calc_slope(mpap, _to_float(co), _to_float(ex_mpap), _to_float(ex_co)).value if has_ex else None
    s_pawp = calc_slope(_to_float(pawp_mean), _to_float(co), _to_float(ex_pawp), _to_float(ex_co)).value if has_ex else None

    lines = []
    lines.append("### Live-Preview (berechnete Werte)")
    lines.append(f"- mPAP: **{fmt_unit(mpap,'mmHg',0)}**")
    lines.append(f"- AO_mean: **{fmt_unit(aom,'mmHg',0)}**")
    lines.append(f"- CI: **{fmt_unit(ci,'L/min/m²',2)}** (BSA {fmt_num(bsa,2)} m²)")
    lines.append(f"- PVR: **{fmt_unit(pvr,'WU',1)}** | PVRI: **{fmt_unit(pvri,'WU·m²',1)}**")
    lines.append(f"- TPG: **{fmt_unit(tpg,'mmHg',0)}** | DPG: **{fmt_unit(dpg,'mmHg',0)}**")
    lines.append(f"- SVR: **{fmt_unit(svr,'WU',1)}**")
    if step.present is True:
        lines.append(f"- Step-up: **JA** {step.location_desc}")
    elif step.present is False:
        lines.append("- Step-up: **NEIN**")
    else:
        lines.append("- Step-up: **UNKLAR** (zu wenige Daten)")

    if has_ex:
        lines.append(f"- Belastung: mPAP/CO-Slope **{fmt_unit(s_mpap,'mmHg/(L/min)',1)}**, PAWP/CO-Slope **{fmt_unit(s_pawp,'mmHg/(L/min)',1)}**")

    if "Volumenchallenge" in (modules or []) and (vc_volume_ml is not None or vc_pawp_post is not None):
        lines.append(f"- Volumenchallenge: Volumen {fmt_unit(_to_float(vc_volume_ml),'ml',0)} | PAWP_post {fmt_unit(_to_float(vc_pawp_post),'mmHg',0)}")

    if "Vasoreaktivität" in (modules or []) and (vaso_mpap_post is not None or vaso_co_post is not None):
        lines.append(f"- Vasoreaktivität: mPAP_post {fmt_unit(_to_float(vaso_mpap_post),'mmHg',0)} | CO_post {fmt_unit(_to_float(vaso_co_post),'L/min',2)}")

    return "\n".join(lines)


# -----------------------------
# Gradio App
# -----------------------------

EXAMPLE = {
    "dt": str(date.today()),
    "exam_type": "Initial-RHK",
    "modules": ["Belastung", "Volumenchallenge", "Vasoreaktivität"],
    "oxygen_mode": "Raumluft",
    "oxygen_flow": None,
    "co_method": "Thermodilution",
    "last_name": "Muster",
    "first_name": "Max",
    "birthdate": "1968-05-14",
    "height_cm": 175,
    "weight_kg": 82.0,
    "ph_dx_known": False,
    "ph_dx_type": "Gruppe 1 – PAH",
    "ph_susp": True,
    "ph_susp_type": "Gruppe 1 – PAH",
    # Ruhe
    "ra_mean": 8,
    "pa_sys": 48,
    "pa_dia": 20,
    "pa_mean": None,
    "pawp_mean": 12,
    "ao_sys": 125,
    "ao_dia": 75,
    "ao_mean": None,
    "co": 4.6,
    "ci_in": None,
    "hr": 78,
    # Sats
    "sat_svc": 68,
    "sat_ra": 70,
    "sat_rv": 70,
    "sat_pa": 71,
    "sat_ao": 97,
    # Optional: PAWP-/Messqualität
    "wedge_sat_pct": None,
    "qc_resp_swings_large": False,
    "qc_obesity": False,
    "qc_copd": False,
    "qc_mechanical_ventilation": False,
    "step_up_mode": "auto",
    "step_up_loc_override": "",
    # Prozedur
    "consent_done": "ja",
    "anticoag_yes": "unklar",
    "anticoag_desc": "",
    "access_site": "V. jugularis dextra",
    # Belastung
    "ex_co": 7.4,
    "ex_mpap": 35,
    "ex_pawp": 14,
    "ex_spap": 62,
    # Volumenchallenge
    "vc_volume_ml": 500,
    "vc_infusion": "NaCl 0.9%",
    "vc_pawp_post": 20,
    "vc_mpap_post": 30,
    # Vasoreaktivität
    "vaso_agent": "iNO",
    "vaso_ino_ppm": 40,
    "vaso_mpap_post": 24,
    "vaso_co_post": 4.8,
    # Risiko/Labore
    "who_fc": "II",
    "sixmwd_m": 380,
    "sbp_mmHg": 118,
    "egfr_ml_min_1_73": 78,
    "bnp_kind": "NT-proBNP",
    "bnp_value": 450,
    "hb_g_dl": 13.6,
    "ferritin_ug_l": 55,
    "tsat_pct": 18,
    "cteph_suspected": False,
    "cteph_context_desc": "",
    "left_heart_context_desc": "kein Anhalt für relevante Klappenvitien, HFpEF-DD je nach Kontext",
    "pvod_hint_desc": "",
    "lufu_summary": "FEV1 82% Soll, DLCO 55% Soll",
    # Procedere
    "planned_choice": [],
    "planned_free": "",

    # Zusatzdaten / Verlauf (Beispiel)
    "story": "Dyspnoe NYHA II seit 6 Monaten, keine Synkopen. Abklärung PH bei V. a. PAH.",
    "lab_inr": 1.0,
    "lab_quick": 95,
    "lab_krea": 1.0,
    "lab_hst": 30,
    "lab_ptt": 30,
    "lab_thrombos": 230,
    "lab_crp": 2,
    "lab_leukos": 7,
    "congestive_organopathy": False,

    "ltot_present": False,
    "ltot_paused": False,
    "bga_rest_po2": 75,
    "bga_rest_pco2": 38,
    "bga_ex_po2": 68,
    "bga_ex_pco2": 40,
    "bga_night_ph": 7.41,
    "bga_night_be": 0.5,

    "virology_positive": False,
    "immunology_positive": False,

    "abdomen_sono_done": True,
    "portal_hypertension": False,

    "ct_angio": True,
    "ct_lae": False,
    "ct_ild": False,
    "ct_ild_desc": "",
    "ct_emphysema": False,
    "ct_emphysema_extent": "",
    "ct_embolism": False,
    "ct_mosaic": False,
    "ct_coronary_calc": True,
    "ct_pericardial_effusion": False,
    "ct_ventricular": "normal",
    "ct_cardiac_phenotype": "keine eindeutige RV-Dilatation",

    "comorbidities": "Arterielle Hypertonie, Hypothyreose",
    "ph_relevance": "Keine TE-Anamnese; keine relevante Lungenerkrankung bekannt.",

    "ph_meds_current": False,
    "ph_meds_current_desc": "",
    "ph_meds_since": "",
    "ph_meds_past": False,
    "ph_meds_past_desc": "",
    "meds_other": "Ramipril, L-Thyroxin",
    "diuretics_yes": False,

    "lufu_done": True,
    "lufu_obst": False,
    "lufu_restr": False,
    "lufu_diff": True,
    "lufu_fev1": 82,
    "lufu_fvc": 88,
    "lufu_fev1_fvc": 0.76,
    "lufu_tlc": 95,
    "lufu_rv": 110,
    "lufu_dlco_sb": 55,
    "lufu_dlco_va": 0.70,
    "lufu_po2": 75,
    "lufu_pco2": 38,
    "lufu_ph": 7.41,
    "lufu_be": 0.5,

    "echo_done": True,
    "echo_params": "RA vergrößert, leichte TI, TAPSE 18 mm, TR Vmax 3.2 m/s",

    "syncope": False,
    "cpet_ve_vco2": 34,
    "cpet_vo2max": 15,

    "cmr_rvesvi": 70,
    "cmr_svi": 35,
    "cmr_rvef": 45,

    "prev_rhk_label": "03/21",
    "prev_course_desc": "stabiler Verlauf",
    "prev_mpap": 19,
    "prev_pawp": 7,
    "prev_ci": 3.24,
    "prev_pvr": 1.5,

    "closing_suggestion": "Vorschlag: PH-Basisdiagnostik komplettieren, Verlaufskontrolle in 3 Monaten (inkl. 6MWT/BNP/Echo).",
}


def build_app() -> Tuple[gr.Blocks, Any, Any]:
    themes_mod = getattr(gr, "themes", None)
    theme_cls = getattr(themes_mod, "Soft", None) if themes_mod is not None else None
    theme = theme_cls() if callable(theme_cls) else None

    css = """
    .container {max-width: 1350px !important;}
    .gradio-container {font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;}
    """

    with gr.Blocks(title="RHK-Befundassistent") as demo:
        gr.Markdown(
            """
# RHK-Befundassistent (GUI)

- **Basis (Ruhe)** ist immer aktiv.
- Wähle optional Zusatzmodule: **Belastung**, **Volumenchallenge**, **Vasoreaktivität**.
- Der Assistent berechnet automatisch (mPAP, AO_mean, CI, PVR, Slopes, Step-up u.a.).
- Das Formular ist initial **mit Beispielwerten** befüllt – für reale Fälle bitte überschreiben oder mit "Leeren" zurücksetzen.
"""
        )

        with gr.Row():
            btn_clear = gr.Button("Formular leeren")
            btn_example = gr.Button("Beispielwerte neu laden")

        with gr.Tabs():
            with gr.Tab("1) Stammdaten"):
                with gr.Row():
                    dt = gr.Textbox(label="Untersuchungsdatum (YYYY-MM-DD)", value=EXAMPLE["dt"], max_lines=1)
                    exam_type = gr.Dropdown(label="Exam type", choices=["Initial-RHK", "Verlaufskontrolle"], value=EXAMPLE["exam_type"])
                    modules = gr.CheckboxGroup(label="Zusatzmodule (optional)", choices=["Belastung", "Volumenchallenge", "Vasoreaktivität"], value=EXAMPLE["modules"])

                with gr.Row():
                    oxygen_mode = gr.Dropdown(label="O2-Modus", choices=["Raumluft", "O2"], value=EXAMPLE["oxygen_mode"])
                    oxygen_flow = gr.Number(label="O2 Flow (L/min)", value=EXAMPLE["oxygen_flow"], precision=0, visible=(EXAMPLE["oxygen_mode"] == "O2"))
                    co_method = gr.Dropdown(label="CO-Methode", choices=["Thermodilution", "Fick_direkt", "Fick_indirekt"], value=EXAMPLE["co_method"])

                with gr.Row():
                    last_name = gr.Textbox(label="Name", value=EXAMPLE["last_name"], max_lines=1)
                    first_name = gr.Textbox(label="Vorname", value=EXAMPLE["first_name"], max_lines=1)
                    birthdate = gr.Textbox(label="Geburtsdatum (YYYY-MM-DD)", value=EXAMPLE["birthdate"], max_lines=1)

                age_preview = gr.Markdown(value=ui_age_preview(EXAMPLE["birthdate"], EXAMPLE["dt"]))

                with gr.Row():
                    height_cm = gr.Number(label="Größe (cm)", value=EXAMPLE["height_cm"], precision=0)
                    weight_kg = gr.Number(label="Gewicht (kg)", value=EXAMPLE["weight_kg"], precision=1)

                anthro_preview = gr.Markdown(value=ui_anthro_preview(EXAMPLE["height_cm"], EXAMPLE["weight_kg"]))

                gr.Markdown("### PH-Anamnese (optional)")
                with gr.Row():
                    ph_dx_known = gr.Checkbox(label="PH-Diagnose bekannt", value=EXAMPLE["ph_dx_known"])
                    ph_dx_type = gr.Dropdown(
                        label="PH-Diagnose (Hauptgruppen)",
                        choices=[
                            "Gruppe 1 – PAH",
                            "Gruppe 1 – CTD-PAH",
                            "Gruppe 2 – Linksherzerkrankung",
                            "Gruppe 3 – Lungenerkrankung/Hypoxie",
                            "Gruppe 4 – CTEPH",
                            "Portopulmonale PH",
                            "Sonstige/unklar",
                        ],
                        value=EXAMPLE["ph_dx_type"],
                        visible=bool(EXAMPLE["ph_dx_known"]),
                    )
                with gr.Row():
                    ph_susp = gr.Checkbox(label="PH-Verdachtsdiagnose", value=EXAMPLE["ph_susp"])
                    ph_susp_type = gr.Dropdown(
                        label="PH-Verdacht (Hauptgruppen)",
                        choices=[
                            "Gruppe 1 – PAH",
                            "Gruppe 2 – Linksherzerkrankung",
                            "Gruppe 3 – Lungenerkrankung/Hypoxie",
                            "Gruppe 4 – CTEPH",
                            "Portopulmonale PH",
                            "Sonstige/unklar",
                        ],
                        value=EXAMPLE["ph_susp_type"],
                        visible=bool(EXAMPLE["ph_susp"]),
                    )

            with gr.Tab("2) RHK Basis (Ruhe)"):
                gr.Markdown("### Drucke (mmHg)")
                with gr.Row():
                    ra_mean = gr.Number(label="RA_mean", value=EXAMPLE["ra_mean"], precision=0)
                    pa_sys = gr.Number(label="PA_sys (sPAP)", value=EXAMPLE["pa_sys"], precision=0)
                    pa_dia = gr.Number(label="PA_dia", value=EXAMPLE["pa_dia"], precision=0)
                    pa_mean = gr.Number(label="PA_mean (optional)", value=EXAMPLE["pa_mean"], precision=0)
                with gr.Row():
                    pawp_mean = gr.Number(label="PAWP_mean", value=EXAMPLE["pawp_mean"], precision=0)
                    ao_sys = gr.Number(label="AO_sys", value=EXAMPLE["ao_sys"], precision=0)
                    ao_dia = gr.Number(label="AO_dia", value=EXAMPLE["ao_dia"], precision=0)
                    ao_mean = gr.Number(label="AO_mean (optional)", value=EXAMPLE["ao_mean"], precision=0)

                gr.Markdown("### Flow")
                with gr.Row():
                    co = gr.Number(label="CO (L/min)", value=EXAMPLE["co"], precision=2)
                    ci_in = gr.Number(label="CI (optional, L/min/m²)", value=EXAMPLE["ci_in"], precision=2)
                    hr = gr.Number(label="HR (/min)", value=EXAMPLE["hr"], precision=0)

                gr.Markdown("### Stufenoxymetrie (Sättigungen, %)")
                with gr.Row():
                    sat_svc = gr.Number(label="SVC", value=EXAMPLE["sat_svc"], precision=0)
                    sat_ra = gr.Number(label="RA", value=EXAMPLE["sat_ra"], precision=0)
                    sat_rv = gr.Number(label="RV", value=EXAMPLE["sat_rv"], precision=0)
                    sat_pa = gr.Number(label="PA", value=EXAMPLE["sat_pa"], precision=0)
                    sat_ao = gr.Number(label="AO", value=EXAMPLE["sat_ao"], precision=0)

                with gr.Row():
                    step_up_mode = gr.Dropdown(label="Sättigungssprung (Auto/Override)", choices=["auto", "ja", "nein"], value=EXAMPLE["step_up_mode"])
                    step_up_loc_override = gr.Textbox(label="Override-Ort (nur wenn 'ja')", value=EXAMPLE["step_up_loc_override"], max_lines=1, visible=(EXAMPLE["step_up_mode"] == "ja"))

                with gr.Accordion("Messqualität / PAWP-Validierung (optional)", open=False):
                    gr.Markdown(
                        "Optional: Angaben, die die Interpretation des PAWP beeinflussen können (Kompendium) – "
                        "z.B. Under-wedge-Verdacht, große respiratorische Schwankungen, Beatmung/PEEP."
                    )
                    wedge_sat_pct = gr.Number(label="Wedge-Sättigung (%) – optional", value=EXAMPLE.get("wedge_sat_pct"), precision=0)
                    with gr.Row():
                        qc_resp_swings_large = gr.Checkbox(label="Große respiratorische Druckschwankungen", value=EXAMPLE.get("qc_resp_swings_large", False))
                        qc_obesity = gr.Checkbox(label="Adipositas/erhöhter pleuraler Grunddruck wahrscheinlich", value=EXAMPLE.get("qc_obesity", False))
                    with gr.Row():
                        qc_copd = gr.Checkbox(label="COPD/Auto-PEEP als Confounder möglich", value=EXAMPLE.get("qc_copd", False))
                        qc_mechanical_ventilation = gr.Checkbox(label="Beatmung/PEEP", value=EXAMPLE.get("qc_mechanical_ventilation", False))

                with gr.Accordion("Prozedur / Logistik (nur intern)", open=False):
                    with gr.Row():
                        consent_done = gr.Dropdown(label="RHK-Aufklärung erfolgt?", choices=["unklar", "ja", "nein"], value=EXAMPLE["consent_done"])
                        anticoag_yes = gr.Dropdown(label="Antikoagulation?", choices=["unklar", "ja", "nein"], value=EXAMPLE["anticoag_yes"])
                    anticoag_desc = gr.Textbox(label="Welche Antikoagulation? (Freitext)", value=EXAMPLE["anticoag_desc"], lines=1, visible=False)
                    access_site = gr.Dropdown(
                        label="Zugang",
                        choices=["", "V. jugularis dextra", "V. jugularis sinistra", "V. femoralis dextra", "V. femoralis sinistra"],
                        value=EXAMPLE["access_site"],
                    )

                calc_preview = gr.Markdown(value=ui_calc_preview(EXAMPLE["height_cm"], EXAMPLE["weight_kg"], EXAMPLE["pa_sys"], EXAMPLE["pa_dia"], EXAMPLE["pa_mean"], EXAMPLE["pawp_mean"], EXAMPLE["ao_sys"], EXAMPLE["ao_dia"], EXAMPLE["ao_mean"], EXAMPLE["ra_mean"], EXAMPLE["co"], EXAMPLE["ci_in"], EXAMPLE["sat_svc"], EXAMPLE["sat_ra"], EXAMPLE["sat_rv"], EXAMPLE["sat_pa"], EXAMPLE["modules"], EXAMPLE["ex_co"], EXAMPLE["ex_mpap"], EXAMPLE["ex_pawp"], EXAMPLE["vc_volume_ml"], EXAMPLE["vc_pawp_post"], EXAMPLE["vaso_mpap_post"], EXAMPLE["vaso_co_post"], _def("stepox.thr_ra_pct", 7), _def("stepox.thr_rv_pct", 5), _def("stepox.thr_pa_pct", 5)))

            with gr.Tab("3) Belastung"):
                ex_group = gr.Column(visible=("Belastung" in EXAMPLE["modules"]))
                with ex_group:
                    gr.Markdown("Belastung (Peak) – aktiv wenn Modul **Belastung** gewählt ist.")
                    with gr.Row():
                        ex_co = gr.Number(label="CO_peak (L/min)", value=EXAMPLE["ex_co"], precision=2)
                        ex_mpap = gr.Number(label="mPAP_peak (mmHg)", value=EXAMPLE["ex_mpap"], precision=0)
                        ex_pawp = gr.Number(label="PAWP_peak (mmHg)", value=EXAMPLE["ex_pawp"], precision=0)
                        ex_spap = gr.Number(label="sPAP_peak (mmHg) – für ΔsPAP", value=EXAMPLE["ex_spap"], precision=0)

            with gr.Tab("4) Volumenchallenge"):
                vol_group = gr.Column(visible=("Volumenchallenge" in EXAMPLE["modules"]))
                with vol_group:
                    gr.Markdown("Volumenchallenge – aktiv wenn Modul **Volumenchallenge** gewählt ist.")
                    with gr.Row():
                        vc_volume_ml = gr.Number(label="Volumen (ml)", value=EXAMPLE["vc_volume_ml"], precision=0)
                        vc_infusion = gr.Dropdown(label="Infusion", choices=["NaCl 0.9%", "Ringer", "sonstiges"], value=EXAMPLE["vc_infusion"])
                    with gr.Row():
                        vc_pawp_post = gr.Number(label="PAWP_post (mmHg)", value=EXAMPLE["vc_pawp_post"], precision=0)
                        vc_mpap_post = gr.Number(label="mPAP_post (mmHg)", value=EXAMPLE["vc_mpap_post"], precision=0)

            with gr.Tab("5) Vasoreaktivität"):
                vaso_group = gr.Column(visible=("Vasoreaktivität" in EXAMPLE["modules"]))
                with vaso_group:
                    gr.Markdown("Vasoreaktivität – aktiv wenn Modul **Vasoreaktivität** gewählt ist.")
                    with gr.Row():
                        vaso_agent = gr.Dropdown(label="Agent", choices=["iNO", "Iloprost", "Adenosin", "sonstiges"], value=EXAMPLE["vaso_agent"])
                        vaso_ino_ppm = gr.Number(label="iNO (ppm)", value=EXAMPLE["vaso_ino_ppm"], precision=0)
                    with gr.Row():
                        vaso_mpap_post = gr.Number(label="mPAP_post (mmHg)", value=EXAMPLE["vaso_mpap_post"], precision=0)
                        vaso_co_post = gr.Number(label="CO_post (L/min)", value=EXAMPLE["vaso_co_post"], precision=2)

            with gr.Tab("6) Klinik / Labore / Risiko"):
                gr.Markdown("### Risiko / Verlauf (optional)")
                with gr.Row():
                    who_fc = gr.Dropdown(label="WHO-FC", choices=["", "I", "II", "III", "IV"], value=EXAMPLE["who_fc"])
                    sixmwd_m = gr.Number(label="6MWD (m)", value=EXAMPLE["sixmwd_m"], precision=0)
                with gr.Row():
                    sbp_mmHg = gr.Number(label="SBP (mmHg)", value=EXAMPLE["sbp_mmHg"], precision=0)
                    egfr_ml_min_1_73 = gr.Number(label="eGFR (ml/min/1.73m²)", value=EXAMPLE["egfr_ml_min_1_73"], precision=0)
                with gr.Row():
                    bnp_kind = gr.Dropdown(label="BNP-Marker", choices=["NT-proBNP", "BNP"], value=EXAMPLE["bnp_kind"])
                    bnp_value = gr.Number(label="BNP/NT-proBNP", value=EXAMPLE["bnp_value"], precision=0)

                gr.Markdown("### Weitere Labore (optional)")
                with gr.Row():
                    hb_g_dl = gr.Number(label="Hb (g/dl)", value=EXAMPLE["hb_g_dl"], precision=1)
                    ferritin_ug_l = gr.Number(label="Ferritin (µg/l)", value=EXAMPLE["ferritin_ug_l"], precision=0)
                    tsat_pct = gr.Number(label="Transferrinsättigung (TSAT, %)", value=EXAMPLE["tsat_pct"], precision=0)

                gr.Markdown("### Klinischer Kontext (optional, für Textbausteine)")
                with gr.Row():
                    cteph_suspected = gr.Checkbox(label="CTEPH/CTEPD-Verdacht", value=EXAMPLE["cteph_suspected"])
                    cteph_context_desc = gr.Textbox(label="CTEPH-Kontext (Freitext)", value=EXAMPLE["cteph_context_desc"], lines=1)
                left_heart_context_desc = gr.Textbox(label="Linksherz-Kontext (Freitext)", value=EXAMPLE["left_heart_context_desc"], lines=1)
                pvod_hint_desc = gr.Textbox(label="PVOD-Hinweise (Freitext)", value=EXAMPLE["pvod_hint_desc"], lines=1)

                gr.Markdown("### Lungenfunktion (Kurztext)")
                lufu_summary = gr.Textbox(label="Lufu Summary (Freitext)", value=EXAMPLE["lufu_summary"], lines=1)

                with gr.Accordion("Story / Kurz-Anamnese", open=False):
                    story = gr.Textbox(label="Story / Kurz-Anamnese", value=EXAMPLE["story"], lines=3)

                with gr.Accordion("Labor (erweitert)", open=False):
                    with gr.Row():
                        lab_inr = gr.Number(label="INR", value=EXAMPLE["lab_inr"], precision=2)
                        lab_quick = gr.Number(label="Quick (%)", value=EXAMPLE["lab_quick"], precision=0)
                        lab_krea = gr.Number(label="Krea (mg/dl)", value=EXAMPLE["lab_krea"], precision=2)
                        lab_hst = gr.Number(label="Hst / Harnstoff (mg/dl)", value=EXAMPLE["lab_hst"], precision=1)
                    with gr.Row():
                        lab_ptt = gr.Number(label="PTT (s)", value=EXAMPLE["lab_ptt"], precision=0)
                        lab_thrombos = gr.Number(label="Thrombos (G/l)", value=EXAMPLE["lab_thrombos"], precision=0)
                        lab_crp = gr.Number(label="CRP (mg/l)", value=EXAMPLE["lab_crp"], precision=1)
                        lab_leukos = gr.Number(label="Leukos (G/l)", value=EXAMPLE["lab_leukos"], precision=1)
                    congestive_organopathy = gr.Checkbox(
                        label="Hinweis auf congestive Organopathie?", value=EXAMPLE["congestive_organopathy"]
                    )

                with gr.Accordion("Blutgase / LTOT", open=False):
                    with gr.Row():
                        ltot_present = gr.Checkbox(label="LTOT vorhanden", value=EXAMPLE["ltot_present"])
                        ltot_paused = gr.Checkbox(label="pausiert", value=EXAMPLE["ltot_paused"])
                    gr.Markdown("#### BGA Ruhe")
                    with gr.Row():
                        bga_rest_po2 = gr.Number(label="pO₂ Ruhe (mmHg)", value=EXAMPLE["bga_rest_po2"], precision=0)
                        bga_rest_pco2 = gr.Number(label="pCO₂ Ruhe (mmHg)", value=EXAMPLE["bga_rest_pco2"], precision=0)
                    gr.Markdown("#### BGA Belastung")
                    with gr.Row():
                        bga_ex_po2 = gr.Number(label="pO₂ Belastung (mmHg)", value=EXAMPLE["bga_ex_po2"], precision=0)
                        bga_ex_pco2 = gr.Number(label="pCO₂ Belastung (mmHg)", value=EXAMPLE["bga_ex_pco2"], precision=0)
                    gr.Markdown("#### BGA Nacht")
                    with gr.Row():
                        bga_night_ph = gr.Number(label="pH Nacht", value=EXAMPLE["bga_night_ph"], precision=2)
                        bga_night_be = gr.Number(label="BE Nacht (mmol/l)", value=EXAMPLE["bga_night_be"], precision=1)

                with gr.Accordion("Infektiologie / Immunologie", open=False):
                    with gr.Row():
                        virology_positive = gr.Checkbox(label="Virologie positiv?", value=EXAMPLE["virology_positive"])
                        immunology_positive = gr.Checkbox(label="Immunologie positiv?", value=EXAMPLE["immunology_positive"])

                with gr.Accordion("Abdomen / Leber", open=False):
                    with gr.Row():
                        abdomen_sono_done = gr.Checkbox(
                            label="Abdomen-Sono durchgeführt", value=EXAMPLE["abdomen_sono_done"]
                        )
                        portal_hypertension = gr.Checkbox(
                            label="Hinweis auf portale Hypertension", value=EXAMPLE["portal_hypertension"]
                        )

                with gr.Accordion("CT / Bildgebung Thorax", open=False):
                    with gr.Row():
                        ct_angio = gr.Checkbox(label="CT-Angio", value=EXAMPLE["ct_angio"])
                        ct_lae = gr.Checkbox(label="LAE", value=EXAMPLE["ct_lae"])
                        ct_ild = gr.Checkbox(label="ILD", value=EXAMPLE["ct_ild"])
                        ct_emphysema = gr.Checkbox(label="Emphysem", value=EXAMPLE["ct_emphysema"])
                    with gr.Row():
                        ct_embolism = gr.Checkbox(label="Embolie", value=EXAMPLE["ct_embolism"])
                        ct_mosaic = gr.Checkbox(label="Mosaikperfusion", value=EXAMPLE["ct_mosaic"])
                        ct_coronary_calc = gr.Checkbox(label="Koronarkalk", value=EXAMPLE["ct_coronary_calc"])
                        ct_pericardial_effusion = gr.Checkbox(
                            label="Perikarderguss", value=EXAMPLE["ct_pericardial_effusion"]
                        )
                    with gr.Row():
                        ct_ventricular = gr.Dropdown(
                            label="Ventrikulär", choices=["normal", "auffällig"], value=EXAMPLE["ct_ventricular"]
                        )
                        ct_cardiac_phenotype = gr.Textbox(
                            label="Kardialer Phänotyp (Freitext)", value=EXAMPLE["ct_cardiac_phenotype"], lines=1
                        )
                    with gr.Row():
                        ct_ild_desc = gr.Textbox(label="ILD-Beschreibung (optional)", value=EXAMPLE["ct_ild_desc"], lines=1)
                        ct_emphysema_extent = gr.Textbox(
                            label="Emphysem-Ausmaß (optional)", value=EXAMPLE["ct_emphysema_extent"], lines=1
                        )

                with gr.Accordion("Vorerkrankungen", open=False):
                    comorbidities = gr.Textbox(
                        label="Relevante Vorerkrankungen (Freitext)", value=EXAMPLE["comorbidities"], lines=3
                    )
                    ph_relevance = gr.Textbox(
                        label="Relevant für PH? (Freitext / ja-nein)", value=EXAMPLE["ph_relevance"], lines=2
                    )

                with gr.Accordion("Medikamente", open=False):
                    with gr.Row():
                        ph_meds_current = gr.Checkbox(label="PH-Medikation aktuell", value=EXAMPLE["ph_meds_current"])
                        diuretics_yes = gr.Checkbox(label="Diuretika", value=EXAMPLE["diuretics_yes"])
                    ph_meds_current_desc = gr.Textbox(label="Welche? (aktuell)", value=EXAMPLE["ph_meds_current_desc"], lines=1)
                    ph_meds_since = gr.Textbox(label="Seit wann? (Freitext)", value=EXAMPLE["ph_meds_since"], lines=1)
                    ph_meds_past = gr.Checkbox(
                        label="PH-Medikation in der Vergangenheit", value=EXAMPLE["ph_meds_past"]
                    )
                    ph_meds_past_desc = gr.Textbox(label="Welche? (Vergangenheit)", value=EXAMPLE["ph_meds_past_desc"], lines=1)
                    meds_other = gr.Textbox(label="Sonstige Medikation (Freitext)", value=EXAMPLE["meds_other"], lines=2)

                with gr.Accordion("Lungenfunktion (strukturiert)", open=False):
                    lufu_done = gr.Checkbox(label="Lufu durchgeführt?", value=EXAMPLE["lufu_done"])
                    with gr.Row():
                        lufu_obst = gr.Checkbox(label="Obstruktiv", value=EXAMPLE["lufu_obst"])
                        lufu_restr = gr.Checkbox(label="Restriktiv", value=EXAMPLE["lufu_restr"])
                        lufu_diff = gr.Checkbox(label="Diffusionsstörung", value=EXAMPLE["lufu_diff"])
                    gr.Markdown("#### Einzelwerte (optional, z.B. % Soll)")
                    with gr.Row():
                        lufu_fev1 = gr.Number(label="FEV₁", value=EXAMPLE["lufu_fev1"], precision=0)
                        lufu_fvc = gr.Number(label="FVC", value=EXAMPLE["lufu_fvc"], precision=0)
                        lufu_fev1_fvc = gr.Number(label="FEV₁/FVC", value=EXAMPLE["lufu_fev1_fvc"], precision=2)
                        lufu_tlc = gr.Number(label="TLC", value=EXAMPLE["lufu_tlc"], precision=0)
                    with gr.Row():
                        lufu_rv = gr.Number(label="RV", value=EXAMPLE["lufu_rv"], precision=0)
                        lufu_dlco_sb = gr.Number(label="DLCO SB", value=EXAMPLE["lufu_dlco_sb"], precision=0)
                        lufu_dlco_va = gr.Number(label="DLCO SB/VA", value=EXAMPLE["lufu_dlco_va"], precision=2)
                        lufu_po2 = gr.Number(label="pO₂", value=EXAMPLE["lufu_po2"], precision=0)
                    with gr.Row():
                        lufu_pco2 = gr.Number(label="pCO₂", value=EXAMPLE["lufu_pco2"], precision=0)
                        lufu_ph = gr.Number(label="pH", value=EXAMPLE["lufu_ph"], precision=2)
                        lufu_be = gr.Number(label="BE", value=EXAMPLE["lufu_be"], precision=1)

                with gr.Accordion("Echokardiographie", open=False):
                    echo_done = gr.Checkbox(label="Echo-Phänotyp vorhanden?", value=EXAMPLE["echo_done"])
                    echo_params = gr.Textbox(label="Relevante Echo-Parameter (Freitext)", value=EXAMPLE["echo_params"], lines=3)

                with gr.Accordion("Funktionelle Tests (Erweiterung)", open=False):
                    syncope = gr.Checkbox(label="Synkope", value=EXAMPLE["syncope"])
                    with gr.Row():
                        cpet_ve_vco2 = gr.Number(label="CPET VE/VCO₂", value=EXAMPLE["cpet_ve_vco2"], precision=1)
                        cpet_vo2max = gr.Number(label="CPET VO₂max (ml/kg/min)", value=EXAMPLE["cpet_vo2max"], precision=1)

                with gr.Accordion("MRT / CMR", open=False):
                    with gr.Row():
                        cmr_rvesvi = gr.Number(label="RVESVi (ml/m²)", value=EXAMPLE["cmr_rvesvi"], precision=0)
                        cmr_svi = gr.Number(label="SVi (ml/m²)", value=EXAMPLE["cmr_svi"], precision=0)
                        cmr_rvef = gr.Number(label="RVEF (%)", value=EXAMPLE["cmr_rvef"], precision=0)

                with gr.Accordion("Vor-RHK (optional)", open=False):
                    prev_rhk_label = gr.Textbox(label="Vor-RHK (z.B. 03/21)", value=EXAMPLE["prev_rhk_label"], lines=1)
                    prev_course_desc = gr.Dropdown(
                        label="Verlauf (Vergleich)",
                        choices=["stabiler Verlauf", "gebessert", "progredient", "unklar"],
                        value=EXAMPLE["prev_course_desc"],
                    )
                    with gr.Row():
                        prev_mpap = gr.Number(label="mPAP (mmHg)", value=EXAMPLE["prev_mpap"], precision=0)
                        prev_pawp = gr.Number(label="PAWP (mmHg)", value=EXAMPLE["prev_pawp"], precision=0)
                        prev_ci = gr.Number(label="CI (L/min/m²)", value=EXAMPLE["prev_ci"], precision=2)
                        prev_pvr = gr.Number(label="PVR (WU)", value=EXAMPLE["prev_pvr"], precision=1)

                with gr.Accordion("Abschluss", open=False):
                    closing_suggestion = gr.Textbox(
                        label="Erster Vorschlag (Therapie / Procedere, Freitext)",
                        value=EXAMPLE["closing_suggestion"],
                        lines=4,
                    )

                risk_out = gr.HTML(value=render_risk_html(esc3_overall(EXAMPLE["who_fc"], _to_float(EXAMPLE["sixmwd_m"]), str(EXAMPLE["bnp_kind"]), _to_float(EXAMPLE["bnp_value"])), esc4_overall(EXAMPLE["who_fc"], _to_float(EXAMPLE["sixmwd_m"]), str(EXAMPLE["bnp_kind"]), _to_float(EXAMPLE["bnp_value"])), reveal_lite2_score(EXAMPLE["who_fc"], _to_float(EXAMPLE["sixmwd_m"]), str(EXAMPLE["bnp_kind"]), _to_float(EXAMPLE["bnp_value"]), _to_float(EXAMPLE["sbp_mmHg"]), _to_float(EXAMPLE["hr"]), _to_float(EXAMPLE["egfr_ml_min_1_73"]))))

            with gr.Tab("7) Procedere"):
                choices = []
                if hasattr(textdb, "P_BLOCKS"):
                    for pid in sorted(textdb.P_BLOCKS.keys()):
                        title = textdb.P_BLOCKS[pid].title
                        choices.append(f"{pid} – {title}")
                planned_choice = gr.CheckboxGroup(label="P-Module auswählen (optional)", choices=choices, value=EXAMPLE["planned_choice"])
                planned_free = gr.Textbox(label="Zusätzliche freie Maßnahmen (eine pro Zeile)", value=EXAMPLE["planned_free"], lines=6)

            with gr.Tab("8) Advanced"):
                use_guideline_cutoffs = gr.Checkbox(label="Leitlinien-Cutoffs anwenden", value=True)
                gr.Markdown("### Ruhe-Definitionen")
                with gr.Row():
                    mpap_cut = gr.Number(label="PH: mPAP > (mmHg)", value=_def("rest.mPAP_ph_mmHg", 20), precision=0)
                    pawp_cut = gr.Number(label="Postkap: PAWP > (mmHg)", value=_def("rest.PAWP_postcap_mmHg", 15), precision=0)
                    pvr_cut = gr.Number(label="Präkap: PVR > (WU)", value=_def("rest.PVR_precap_WU", 2), precision=1)

                gr.Markdown("### Belastung")
                with gr.Row():
                    slope_mpap = gr.Number(label="Belastung: mPAP/CO Slope >", value=_def("exercise.mPAP_CO_slope_mmHg_per_L_min", 3), precision=1)
                    slope_pawp = gr.Number(label="Belastung: PAWP/CO Slope >", value=_def("exercise.PAWP_CO_slope_mmHg_per_L_min", 2), precision=1)

                gr.Markdown("### Schweregrad PVR (optional)")
                with gr.Row():
                    pvr_mild = gr.Number(label="PVR leicht ab (WU)", value=_def("severity.PVR_WU.mild_ge", 2), precision=1)
                    pvr_mod = gr.Number(label="PVR mittel ab (WU)", value=_def("severity.PVR_WU.moderate_ge", 5), precision=1)
                    pvr_sev = gr.Number(label="PVR schwer ab (WU)", value=_def("severity.PVR_WU.severe_ge", 10), precision=1)
                    ci_low = gr.Number(label="CI low < (L/min/m²)", value=_def("severity.CI_L_min_m2.severely_reduced_lt", 2.0), precision=2)

                gr.Markdown("### Step-Ox Schwellen (Auto-Step-Up)")
                with gr.Row():
                    stepox_thr_ra = gr.Number(label="Step-up SVC→RA (≥ %)", value=_def("stepox.thr_ra_pct", 7), precision=0)
                    stepox_thr_rv = gr.Number(label="Step-up RA→RV (≥ %)", value=_def("stepox.thr_rv_pct", 5), precision=0)
                    stepox_thr_pa = gr.Number(label="Step-up RV→PA (≥ %)", value=_def("stepox.thr_pa_pct", 5), precision=0)

                gr.Markdown("### Volumenchallenge Schwellen")
                with gr.Row():
                    vc_pawp_post_thr = gr.Number(label="positiv wenn PAWP_post ≥ (mmHg)", value=_def("volume.pawp_post_thr_mmHg", 18), precision=0)
                    vc_delta_pawp_thr = gr.Number(label="positiv wenn ΔPAWP ≥ (mmHg)", value=_def("volume.delta_pawp_thr_mmHg", 5), precision=0)

            # -------------------------------------------------------------
            # TextDB Admin / Editor – Overrides direkt aus dem GUI
            # -------------------------------------------------------------
            with gr.Tab("9) TextDB / Bausteine"):
                gr.Markdown(
                    "### Textbausteine bearbeiten (overrides.yaml)\n"
                    "Diese Seite schreibt *nur* in `textdb/overrides.yaml`. `core.yaml` bleibt unverändert.\n\n"
                    "Workflow: **Draft speichern** → (optional) **Approve** → Report generieren (Templates werden sofort genutzt)."
                )

                def _block_choices() -> List[str]:
                    try:
                        blocks = textdb.list_blocks() if hasattr(textdb, "list_blocks") else []
                    except Exception:
                        blocks = []
                    if not blocks:
                        # Fallback auf ALL_BLOCKS Dict
                        blocks_dict = getattr(textdb, "ALL_BLOCKS", {}) or {}
                        blocks = list(blocks_dict.values())
                    # sort by id for stability
                    blocks_sorted = sorted(blocks, key=lambda b: str(getattr(b, "id", "")))
                    out = []
                    for b in blocks_sorted:
                        bid = str(getattr(b, "id", "") or "").strip()
                        title = str(getattr(b, "title", "") or "").strip()
                        if not bid:
                            continue
                        out.append(f"{bid} – {title}" if title else bid)
                    return out

                def _category_choices() -> List[str]:
                    try:
                        blocks = textdb.list_blocks() if hasattr(textdb, "list_blocks") else []
                    except Exception:
                        blocks = []
                    if not blocks:
                        blocks = list((getattr(textdb, "ALL_BLOCKS", {}) or {}).values())
                    cats = sorted({str(getattr(b, "category", "") or "").strip() for b in blocks if getattr(b, "category", None)})
                    return [c for c in cats if c]

                def _planned_choices() -> List[str]:
                    p_blocks = getattr(textdb, "P_BLOCKS", {}) or {}
                    items = sorted(p_blocks.items(), key=lambda kv: kv[0])
                    return [f"{pid} – {blk.title}" for pid, blk in items]

                def _parse_choice(choice: str) -> str:
                    return (choice or "").split(" – ", 1)[0].strip()

                with gr.Row():
                    db_reload_btn = gr.Button("TextDB neu laden")
                    db_status = gr.Markdown("")

                block_select = gr.Dropdown(label="Block auswählen", choices=_block_choices(), value=None, allow_custom_value=False)

                with gr.Row():
                    edit_id = gr.Textbox(label="ID", value="", max_lines=1)
                    edit_status = gr.Dropdown(label="Override-Status", choices=["draft", "approved"], value="draft")

                with gr.Row():
                    edit_title = gr.Textbox(label="Titel", value="", max_lines=1)
                    edit_category = gr.Dropdown(label="Kategorie", choices=_category_choices() or ["B", "E", "P", "BZ"], value=None)
                    edit_kind = gr.Textbox(label="Kind (optional)", value="", max_lines=1)

                edit_tags = gr.Textbox(label="Tags (kommagetrennt)", value="", max_lines=1)
                with gr.Row():
                    edit_priority = gr.Number(label="Priority (optional)", value=None, precision=0)
                    edit_applies = gr.Textbox(label="Applies_to (optional)", value="", max_lines=1)

                edit_template = gr.Textbox(label="Template", value="", lines=10)
                edit_variants = gr.Textbox(label="Variants (JSON, optional)", value="{}", lines=4)
                edit_notes = gr.Textbox(label="Notes (optional)", value="", lines=3)

                with gr.Row():
                    db_new_btn = gr.Button("Neu")
                    db_save_draft_btn = gr.Button("Draft speichern")
                    db_save_approve_btn = gr.Button("Speichern & Approve")
                    db_approve_btn = gr.Button("Approve (bestehenden Draft)")
                    db_discard_btn = gr.Button("Override verwerfen")

                def _load_block(choice: str):
                    bid = _parse_choice(choice)
                    if not bid:
                        return "", "draft", "", None, "", "", "", None, "", "", "{}", ""
                    b = textdb.get_block(bid) if hasattr(textdb, "get_block") else None
                    if not b:
                        return bid, "draft", "", None, "", "", "", None, "", "", "{}", ""
                    # override status (wenn API vorhanden)
                    try:
                        st = textdb.override_status(bid) if hasattr(textdb, "override_status") else None
                    except Exception:
                        st = None
                    st = st or "draft"
                    tags = getattr(b, "tags", []) or []
                    tags_txt = ", ".join([str(t) for t in tags])
                    variants = getattr(b, "variants", {}) or {}
                    try:
                        variants_txt = json.dumps(variants, ensure_ascii=False, indent=2)
                    except Exception:
                        variants_txt = "{}"
                    return (
                        getattr(b, "id", ""),
                        st,
                        getattr(b, "title", ""),
                        getattr(b, "category", None),
                        getattr(b, "kind", "") or "",
                        tags_txt,
                        getattr(b, "template", "") or "",
                        getattr(b, "priority", None),
                        getattr(b, "applies_to", "") or "",
                        getattr(b, "notes", "") or "",
                        variants_txt,
                        "",  # status message
                    )

                def _new_block():
                    return (
                        "NEW_ID",
                        "draft",
                        "",
                        None,
                        "",
                        "",
                        "",
                        None,
                        "",
                        "",
                        "{}",
                        "",
                    )

                def _save_block(bid, status, title, category, kind, tags_txt, template, priority, applies_to, notes, variants_txt, approve: bool = False):
                    bid = (bid or "").strip()
                    if not bid:
                        return gr.update(), gr.update(), "❌ Bitte eine ID angeben."
                    if not hasattr(textdb, "upsert_override_block"):
                        return gr.update(), gr.update(), "❌ TextDB-API unterstützt Overrides nicht (upsert_override_block fehlt)."
                    tags = [t.strip() for t in str(tags_txt or "").split(",") if t.strip()]
                    # variants json
                    variants: Dict[str, Any] = {}
                    if str(variants_txt or "").strip():
                        try:
                            variants = json.loads(variants_txt)
                            if not isinstance(variants, dict):
                                variants = {}
                        except Exception:
                            variants = {}
                    patch = {
                        "id": bid,
                        "title": str(title or "").strip(),
                        "category": str(category or "").strip() or None,
                        "kind": str(kind or "").strip() or None,
                        "tags": tags,
                        "priority": int(priority) if priority is not None and str(priority) != "" else None,
                        "applies_to": str(applies_to or "").strip() or None,
                        "template": str(template or "").strip(),
                        "variants": variants,
                        "notes": str(notes or "").strip() or None,
                    }
                    # None-Felder entfernen
                    patch = {k: v for k, v in patch.items() if v is not None}

                    st = "approved" if approve else str(status or "draft")
                    if st not in ("draft", "approved"):
                        st = "draft"
                    textdb.upsert_override_block(bid, patch, status=st)  # type: ignore
                    # UI-choices aktualisieren
                    return (
                        gr.update(choices=_block_choices(), value=f"{bid} – {patch.get('title','')}".strip(" – ")),
                        gr.update(choices=_planned_choices()),
                        f"✅ Gespeichert ({st}): {bid}",
                    )

                def _approve_block(bid: str):
                    bid = (bid or "").strip()
                    if not bid:
                        return gr.update(), gr.update(), "❌ Bitte ID angeben."
                    if not hasattr(textdb, "approve_override_block"):
                        return gr.update(), gr.update(), "❌ TextDB-API unterstützt Approve nicht (approve_override_block fehlt)."
                    textdb.approve_override_block(bid)  # type: ignore
                    return gr.update(choices=_block_choices(), value=f"{bid}"), gr.update(choices=_planned_choices()), f"✅ Approved: {bid}"

                def _discard_block(bid: str):
                    bid = (bid or "").strip()
                    if not bid:
                        return gr.update(), gr.update(), "❌ Bitte ID angeben."
                    if not hasattr(textdb, "discard_override_block"):
                        return gr.update(), gr.update(), "❌ TextDB-API unterstützt Discard nicht (discard_override_block fehlt)."
                    textdb.discard_override_block(bid)  # type: ignore
                    return gr.update(choices=_block_choices(), value=None), gr.update(choices=_planned_choices()), f"✅ Override verworfen: {bid}"

                def _reload_textdb():
                    if hasattr(textdb, "reload"):
                        textdb.reload()  # type: ignore
                    return gr.update(choices=_block_choices()), gr.update(choices=_planned_choices()), "✅ TextDB neu geladen."

                block_select.change(
                    _load_block,
                    inputs=[block_select],
                    outputs=[
                        edit_id,
                        edit_status,
                        edit_title,
                        edit_category,
                        edit_kind,
                        edit_tags,
                        edit_template,
                        edit_priority,
                        edit_applies,
                        edit_notes,
                        edit_variants,
                        db_status,
                    ],
                )

                db_new_btn.click(
                    _new_block,
                    inputs=[],
                    outputs=[
                        edit_id,
                        edit_status,
                        edit_title,
                        edit_category,
                        edit_kind,
                        edit_tags,
                        edit_template,
                        edit_priority,
                        edit_applies,
                        edit_notes,
                        edit_variants,
                        db_status,
                    ],
                )

                db_save_draft_btn.click(
                    lambda *a: _save_block(*a, approve=False),
                    inputs=[edit_id, edit_status, edit_title, edit_category, edit_kind, edit_tags, edit_template, edit_priority, edit_applies, edit_notes, edit_variants],
                    outputs=[block_select, planned_choice, db_status],
                )

                db_save_approve_btn.click(
                    lambda *a: _save_block(*a, approve=True),
                    inputs=[edit_id, edit_status, edit_title, edit_category, edit_kind, edit_tags, edit_template, edit_priority, edit_applies, edit_notes, edit_variants],
                    outputs=[block_select, planned_choice, db_status],
                )

                db_approve_btn.click(
                    _approve_block,
                    inputs=[edit_id],
                    outputs=[block_select, planned_choice, db_status],
                )

                db_discard_btn.click(
                    _discard_block,
                    inputs=[edit_id],
                    outputs=[block_select, planned_choice, db_status],
                )

                db_reload_btn.click(
                    _reload_textdb,
                    inputs=[],
                    outputs=[block_select, planned_choice, db_status],
                )

        gr.Markdown("---")
        with gr.Row():
            btn = gr.Button("Befund generieren", variant="primary")
            btn_json = gr.Button("JSON herunterladen")
            btn_txt = gr.Button("Befund (.txt) herunterladen")
            btn_int = gr.Button("Interner Befund (.txt) herunterladen")

        report_out = gr.Textbox(label="Befund (copy-ready)", lines=26)
        internal_out = gr.Textbox(label="Interner Befund", lines=10)
        json_out = gr.Textbox(label="JSON", lines=10)
        json_file = gr.File(label="Download JSON")
        report_file = gr.File(label="Download Befund (.txt)")
        internal_file = gr.File(label="Download Interner Befund (.txt)")

        # -----------------------------
        # Events
        # -----------------------------

        def _vis_bool(b: bool):
            return gr.update(visible=bool(b))

        ph_dx_known.change(_vis_bool, inputs=[ph_dx_known], outputs=[ph_dx_type])
        ph_susp.change(_vis_bool, inputs=[ph_susp], outputs=[ph_susp_type])

        def _vis_yes(sel: str):
            return gr.update(visible=(sel == "ja"))

        anticoag_yes.change(_vis_yes, inputs=[anticoag_yes], outputs=[anticoag_desc])


        # Sichtbarkeit: O2-Flow nur bei O2
        def _vis_o2(mode: str):
            return gr.update(visible=(str(mode).strip() == "O2"))

        oxygen_mode.change(_vis_o2, inputs=[oxygen_mode], outputs=[oxygen_flow])

        # Sichtbarkeit: Step-up Override-Ort nur wenn "ja"
        def _vis_step_override(mode: str):
            return gr.update(visible=(str(mode).strip().lower() == "ja"))

        step_up_mode.change(_vis_step_override, inputs=[step_up_mode], outputs=[step_up_loc_override])

        # Sichtbarkeit: Modul-Tabs (Inhalte) je nach Auswahl
        def _vis_modules(mods):
            mods = mods or []
            return (
                gr.update(visible=("Belastung" in mods)),
                gr.update(visible=("Volumenchallenge" in mods)),
                gr.update(visible=("Vasoreaktivität" in mods)),
            )

        modules.change(_vis_modules, inputs=[modules], outputs=[ex_group, vol_group, vaso_group])


        # Live previews
        height_cm.change(ui_anthro_preview, inputs=[height_cm, weight_kg], outputs=[anthro_preview])
        weight_kg.change(ui_anthro_preview, inputs=[height_cm, weight_kg], outputs=[anthro_preview])
        birthdate.change(ui_age_preview, inputs=[birthdate, dt], outputs=[age_preview])
        dt.change(ui_age_preview, inputs=[birthdate, dt], outputs=[age_preview])

        # calc preview inputs
        calc_inputs = [
            height_cm, weight_kg,
            pa_sys, pa_dia, pa_mean, pawp_mean,
            ao_sys, ao_dia, ao_mean,
            ra_mean,
            co, ci_in,
            sat_svc, sat_ra, sat_rv, sat_pa,
            modules,
            ex_co, ex_mpap, ex_pawp,
            vc_volume_ml, vc_pawp_post,
            vaso_mpap_post, vaso_co_post,
            stepox_thr_ra, stepox_thr_rv, stepox_thr_pa,
        ]
        for c in calc_inputs:
            c.change(ui_calc_preview, inputs=calc_inputs, outputs=[calc_preview])

        # risk preview
        def _risk_preview_ui(who_fc_val, sixmwd_val, bnp_kind_val, bnp_val, sbp_val, hr_val, egfr_val):
            esc3 = esc3_overall(who_fc_val, _to_float(sixmwd_val), str(bnp_kind_val), _to_float(bnp_val))
            esc4 = esc4_overall(who_fc_val, _to_float(sixmwd_val), str(bnp_kind_val), _to_float(bnp_val))
            reveal = reveal_lite2_score(who_fc_val, _to_float(sixmwd_val), str(bnp_kind_val), _to_float(bnp_val), _to_float(sbp_val), _to_float(hr_val), _to_float(egfr_val))
            return render_risk_html(esc3, esc4, reveal)

        risk_inputs = [who_fc, sixmwd_m, bnp_kind, bnp_value, sbp_mmHg, hr, egfr_ml_min_1_73]
        for c in risk_inputs:
            c.change(_risk_preview_ui, inputs=risk_inputs, outputs=[risk_out])

        # Generate
        inputs = [
            dt, exam_type, modules, oxygen_mode, oxygen_flow, co_method,
            last_name, first_name, birthdate, height_cm, weight_kg,
            ph_dx_known, ph_dx_type, ph_susp, ph_susp_type,
            # Ruhe
            ra_mean, pa_sys, pa_dia, pa_mean, pawp_mean, ao_sys, ao_dia, ao_mean,
            co, ci_in, hr,
            # Sats
            sat_svc, sat_ra, sat_rv, sat_pa, sat_ao,
            step_up_mode, step_up_loc_override,
            # Prozedur
            consent_done, anticoag_yes, anticoag_desc, access_site,
            # Belastung
            ex_co, ex_mpap, ex_pawp, ex_spap,
            # Volumenchallenge
            vc_volume_ml, vc_infusion, vc_pawp_post, vc_mpap_post,
            # Vasoreaktivität
            vaso_agent, vaso_ino_ppm, vaso_mpap_post, vaso_co_post,
            # Klinik/Labore/Risiko
            who_fc, sixmwd_m, sbp_mmHg, egfr_ml_min_1_73, bnp_kind, bnp_value,
            hb_g_dl, ferritin_ug_l, tsat_pct,
            cteph_suspected, cteph_context_desc, left_heart_context_desc, pvod_hint_desc, lufu_summary,
            # Procedere
            planned_choice, planned_free,
            # Advanced
            use_guideline_cutoffs, mpap_cut, pawp_cut, pvr_cut,
            slope_mpap, slope_pawp,
            pvr_mild, pvr_mod, pvr_sev, ci_low,
            stepox_thr_ra, stepox_thr_rv, stepox_thr_pa,
            vc_pawp_post_thr, vc_delta_pawp_thr,

            # Zusatzdaten (optional)
            story,
            lab_inr, lab_quick, lab_krea, lab_hst, lab_ptt, lab_thrombos, lab_crp, lab_leukos,
            congestive_organopathy,
            ltot_present, ltot_paused,
            bga_rest_po2, bga_rest_pco2, bga_ex_po2, bga_ex_pco2, bga_night_ph, bga_night_be,
            virology_positive, immunology_positive,
            abdomen_sono_done, portal_hypertension,
            ct_angio, ct_lae, ct_ild, ct_ild_desc, ct_emphysema, ct_emphysema_extent,
            ct_embolism, ct_mosaic, ct_coronary_calc, ct_pericardial_effusion,
            ct_ventricular, ct_cardiac_phenotype,
            comorbidities, ph_relevance,
            ph_meds_current, ph_meds_current_desc, ph_meds_since,
            ph_meds_past, ph_meds_past_desc,
            meds_other, diuretics_yes,
            lufu_done, lufu_obst, lufu_restr, lufu_diff,
            lufu_fev1, lufu_fvc, lufu_fev1_fvc, lufu_tlc, lufu_rv, lufu_dlco_sb, lufu_dlco_va,
            lufu_po2, lufu_pco2, lufu_ph, lufu_be,
            echo_done, echo_params,
            syncope, cpet_ve_vco2, cpet_vo2max,
            cmr_rvesvi, cmr_svi, cmr_rvef,
            prev_rhk_label, prev_course_desc, prev_mpap, prev_pawp, prev_ci, prev_pvr,
            closing_suggestion,
            # Optional: Messqualität / PAWP-Validierung
            wedge_sat_pct, qc_resp_swings_large, qc_obesity, qc_copd, qc_mechanical_ventilation,
        ]

        btn.click(_generate_all_outputs, inputs=inputs, outputs=[report_out, internal_out, risk_out, json_out])

        btn_json.click(_download_json, inputs=[json_out], outputs=[json_file])
        btn_txt.click(_download_report, inputs=[report_out], outputs=[report_file])
        btn_int.click(_download_internal, inputs=[internal_out], outputs=[internal_file])

        # Clear / Example buttons
        def _clear_form():
            # Rückgabe in exakt gleicher Reihenfolge wie "inputs" (ohne Outputs)
            return [
                str(date.today()), "Initial-RHK", [],
                "Raumluft", None, "Thermodilution",
                "", "", "", None, None,
                False, "Gruppe 1 – PAH", False, "Gruppe 1 – PAH",
                None, None, None, None, None, None, None, None,
                None, None, None,
                None, None, None, None, None,
                "auto", "",
                "unklar", "unklar", "", "",
                None, None, None, None,
                None, "NaCl 0.9%", None, None,
                "iNO", None, None, None,
                "", None, None, None, "NT-proBNP", None,
                None, None, None,
                False, "", "", "", "",
                [], "",
                True, _def("rest.mPAP_ph_mmHg", 20), _def("rest.PAWP_postcap_mmHg", 15), _def("rest.PVR_precap_WU", 2),
                _def("exercise.mPAP_CO_slope_mmHg_per_L_min", 3), _def("exercise.PAWP_CO_slope_mmHg_per_L_min", 2),
                _def("severity.PVR_WU.mild_ge", 2), _def("severity.PVR_WU.moderate_ge", 5), _def("severity.PVR_WU.severe_ge", 10), _def("severity.CI_L_min_m2.severely_reduced_lt", 2.0),
                _def("stepox.thr_ra_pct", 7), _def("stepox.thr_rv_pct", 5), _def("stepox.thr_pa_pct", 5),
                _def("volume.pawp_post_thr_mmHg", 18), _def("volume.delta_pawp_thr_mmHg", 5),

                # Zusatzdaten (optional)
                "",
                None, None, None, None, None, None, None, None,
                False,
                False, False,
                None, None, None, None, None, None,
                False, False,
                False, False,
                False, False, False, "", False, "",
                False, False, False, False,
                "normal", "",
                "", "",
                False, "", "",
                False, "",
                "", False,
                False, False, False, False,
                None, None, None, None, None, None, None,
                None, None, None, None,
                False, "",
                False, None, None,
                None, None, None,
                "", "stabiler Verlauf", None, None, None, None,
                "",
                # Optional: Messqualität / PAWP-Validierung
                None, False, False, False, False,
            ]

        def _load_example():
            e = EXAMPLE
            return [
                e["dt"], e["exam_type"], e["modules"],
                e["oxygen_mode"], e["oxygen_flow"], e["co_method"],
                e["last_name"], e["first_name"], e["birthdate"], e["height_cm"], e["weight_kg"],
                e["ph_dx_known"], e["ph_dx_type"], e["ph_susp"], e["ph_susp_type"],
                e["ra_mean"], e["pa_sys"], e["pa_dia"], e["pa_mean"], e["pawp_mean"], e["ao_sys"], e["ao_dia"], e["ao_mean"],
                e["co"], e["ci_in"], e["hr"],
                e["sat_svc"], e["sat_ra"], e["sat_rv"], e["sat_pa"], e["sat_ao"],
                e["step_up_mode"], e["step_up_loc_override"],
                e["consent_done"], e["anticoag_yes"], e["anticoag_desc"], e["access_site"],
                e["ex_co"], e["ex_mpap"], e["ex_pawp"], e["ex_spap"],
                e["vc_volume_ml"], e["vc_infusion"], e["vc_pawp_post"], e["vc_mpap_post"],
                e["vaso_agent"], e["vaso_ino_ppm"], e["vaso_mpap_post"], e["vaso_co_post"],
                e["who_fc"], e["sixmwd_m"], e["sbp_mmHg"], e["egfr_ml_min_1_73"], e["bnp_kind"], e["bnp_value"],
                e["hb_g_dl"], e["ferritin_ug_l"], e["tsat_pct"],
                e["cteph_suspected"], e["cteph_context_desc"], e["left_heart_context_desc"], e["pvod_hint_desc"], e["lufu_summary"],
                e["planned_choice"], e["planned_free"],
                True, _def("rest.mPAP_ph_mmHg", 20), _def("rest.PAWP_postcap_mmHg", 15), _def("rest.PVR_precap_WU", 2),
                _def("exercise.mPAP_CO_slope_mmHg_per_L_min", 3), _def("exercise.PAWP_CO_slope_mmHg_per_L_min", 2),
                _def("severity.PVR_WU.mild_ge", 2), _def("severity.PVR_WU.moderate_ge", 5), _def("severity.PVR_WU.severe_ge", 10), _def("severity.CI_L_min_m2.severely_reduced_lt", 2.0),
                _def("stepox.thr_ra_pct", 7), _def("stepox.thr_rv_pct", 5), _def("stepox.thr_pa_pct", 5),
                _def("volume.pawp_post_thr_mmHg", 18), _def("volume.delta_pawp_thr_mmHg", 5),

                # Zusatzdaten (optional)
                EXAMPLE.get("story", ""),
                EXAMPLE.get("lab_inr"), EXAMPLE.get("lab_quick"), EXAMPLE.get("lab_krea"), EXAMPLE.get("lab_hst"),
                EXAMPLE.get("lab_ptt"), EXAMPLE.get("lab_thrombos"), EXAMPLE.get("lab_crp"), EXAMPLE.get("lab_leukos"),
                EXAMPLE.get("congestive_organopathy", False),
                EXAMPLE.get("ltot_present", False), EXAMPLE.get("ltot_paused", False),
                EXAMPLE.get("bga_rest_po2"), EXAMPLE.get("bga_rest_pco2"), EXAMPLE.get("bga_ex_po2"),
                EXAMPLE.get("bga_ex_pco2"), EXAMPLE.get("bga_night_ph"), EXAMPLE.get("bga_night_be"),
                EXAMPLE.get("virology_positive", False), EXAMPLE.get("immunology_positive", False),
                EXAMPLE.get("abdomen_sono_done", False), EXAMPLE.get("portal_hypertension", False),
                EXAMPLE.get("ct_angio", False), EXAMPLE.get("ct_lae", False), EXAMPLE.get("ct_ild", False),
                EXAMPLE.get("ct_ild_desc", ""), EXAMPLE.get("ct_emphysema", False), EXAMPLE.get("ct_emphysema_extent", ""),
                EXAMPLE.get("ct_embolism", False), EXAMPLE.get("ct_mosaic", False), EXAMPLE.get("ct_coronary_calc", False),
                EXAMPLE.get("ct_pericardial_effusion", False),
                EXAMPLE.get("ct_ventricular", "normal"), EXAMPLE.get("ct_cardiac_phenotype", ""),
                EXAMPLE.get("comorbidities", ""), EXAMPLE.get("ph_relevance", ""),
                EXAMPLE.get("ph_meds_current", False), EXAMPLE.get("ph_meds_current_desc", ""), EXAMPLE.get("ph_meds_since", ""),
                EXAMPLE.get("ph_meds_past", False), EXAMPLE.get("ph_meds_past_desc", ""),
                EXAMPLE.get("meds_other", ""), EXAMPLE.get("diuretics_yes", False),
                EXAMPLE.get("lufu_done", False), EXAMPLE.get("lufu_obst", False), EXAMPLE.get("lufu_restr", False), EXAMPLE.get("lufu_diff", False),
                EXAMPLE.get("lufu_fev1"), EXAMPLE.get("lufu_fvc"), EXAMPLE.get("lufu_fev1_fvc"),
                EXAMPLE.get("lufu_tlc"), EXAMPLE.get("lufu_rv"), EXAMPLE.get("lufu_dlco_sb"), EXAMPLE.get("lufu_dlco_va"),
                EXAMPLE.get("lufu_po2"), EXAMPLE.get("lufu_pco2"), EXAMPLE.get("lufu_ph"), EXAMPLE.get("lufu_be"),
                EXAMPLE.get("echo_done", False), EXAMPLE.get("echo_params", ""),
                EXAMPLE.get("syncope", False), EXAMPLE.get("cpet_ve_vco2"), EXAMPLE.get("cpet_vo2max"),
                EXAMPLE.get("cmr_rvesvi"), EXAMPLE.get("cmr_svi"), EXAMPLE.get("cmr_rvef"),
                EXAMPLE.get("prev_rhk_label", ""), EXAMPLE.get("prev_course_desc", "stabiler Verlauf"),
                EXAMPLE.get("prev_mpap"), EXAMPLE.get("prev_pawp"), EXAMPLE.get("prev_ci"), EXAMPLE.get("prev_pvr"),
                EXAMPLE.get("closing_suggestion", ""),
                # Optional: Messqualität / PAWP-Validierung
                EXAMPLE.get("wedge_sat_pct"), EXAMPLE.get("qc_resp_swings_large", False), EXAMPLE.get("qc_obesity", False), EXAMPLE.get("qc_copd", False), EXAMPLE.get("qc_mechanical_ventilation", False),
            ]

        btn_clear.click(_clear_form, inputs=[], outputs=inputs)
        btn_example.click(_load_example, inputs=[], outputs=inputs)

    return demo, theme, css


if __name__ == "__main__":
    app, theme, css = build_app()
    try:
        app.launch(share=False, theme=theme, css=css)
    except TypeError:
        app.launch(share=False)
