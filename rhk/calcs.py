# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .util import to_float


@dataclass
class CalcResult:
    value: Optional[float]
    formula: Optional[str] = None


def calc_mean(sys_: Optional[float], dia: Optional[float]) -> CalcResult:
    if sys_ is None or dia is None:
        return CalcResult(None)
    mean = (sys_ + 2.0 * dia) / 3.0
    return CalcResult(mean, formula=f"(sys + 2·dia)/3 = ({sys_} + 2·{dia})/3")


def calc_bsa_dubois(height_cm: Optional[float], weight_kg: Optional[float]) -> CalcResult:
    if height_cm is None or weight_kg is None:
        return CalcResult(None)
    if height_cm <= 0 or weight_kg <= 0:
        return CalcResult(None)
    bsa = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
    return CalcResult(
        bsa,
        formula=f"0.007184·{height_cm}^0.725·{weight_kg}^0.425",
    )


def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if height_cm is None or weight_kg is None:
        return None
    if height_cm <= 0 or weight_kg <= 0:
        return None
    return weight_kg / ((height_cm / 100.0) ** 2)


def calc_ci(co: Optional[float], bsa: Optional[float]) -> CalcResult:
    if co is None or bsa is None or bsa == 0:
        return CalcResult(None)
    return CalcResult(co / bsa, formula=f"{co}/{bsa}")


def calc_tpg(mpap: Optional[float], pawp: Optional[float]) -> CalcResult:
    if mpap is None or pawp is None:
        return CalcResult(None)
    return CalcResult(mpap - pawp, formula=f"{mpap} - {pawp}")


def calc_dpg(dpap: Optional[float], pawp: Optional[float]) -> CalcResult:
    if dpap is None or pawp is None:
        return CalcResult(None)
    return CalcResult(dpap - pawp, formula=f"{dpap} - {pawp}")


def calc_pvr(mpap: Optional[float], pawp: Optional[float], co: Optional[float]) -> CalcResult:
    if mpap is None or pawp is None or co is None or co == 0:
        return CalcResult(None)
    return CalcResult((mpap - pawp) / co, formula=f"({mpap}-{pawp})/{co}")


def calc_pvri(mpap: Optional[float], pawp: Optional[float], ci: Optional[float]) -> CalcResult:
    # PVRI = (mPAP - PAWP)/CI  (WU·m²)
    if mpap is None or pawp is None or ci is None or ci == 0:
        return CalcResult(None)
    return CalcResult((mpap - pawp) / ci, formula=f"({mpap}-{pawp})/{ci}")


def calc_svr(aom: Optional[float], rap: Optional[float], co: Optional[float]) -> CalcResult:
    if aom is None or rap is None or co is None or co == 0:
        return CalcResult(None)
    return CalcResult((aom - rap) / co, formula=f"({aom}-{rap})/{co}")


def calc_slope(p_rest: Optional[float], co_rest: Optional[float], p_peak: Optional[float], co_peak: Optional[float]) -> CalcResult:
    if p_rest is None or co_rest is None or p_peak is None or co_peak is None:
        return CalcResult(None)
    delta_co = co_peak - co_rest
    if delta_co == 0:
        return CalcResult(None)
    slope = (p_peak - p_rest) / delta_co
    return CalcResult(slope, formula=f"({p_peak}-{p_rest})/({co_peak}-{co_rest})")


def calc_delta_spap(spap_rest: Optional[float], spap_peak: Optional[float]) -> Optional[float]:
    if spap_rest is None or spap_peak is None:
        return None
    return spap_peak - spap_rest


def calc_ci_peak(co_peak: Optional[float], bsa: Optional[float]) -> Optional[float]:
    if co_peak is None or bsa is None or bsa == 0:
        return None
    return co_peak / bsa


def sprime_raai(
    sprime_cm_s: Optional[float],
    ra_area_cm2: Optional[float],
    bsa_m2: Optional[float],
) -> Optional[float]:
    """
    Yogeswaran et al.: S'/RAAI where RAAI = RA area / BSA (cm²/m²).
    => S'/RAAI = sprime / (ra_area/bsa) = sprime * bsa / ra_area
    Unit: m²/(s·cm)
    """
    if sprime_cm_s is None or ra_area_cm2 is None or bsa_m2 is None:
        return None
    if ra_area_cm2 <= 0 or bsa_m2 <= 0:
        return None
    return sprime_cm_s * bsa_m2 / ra_area_cm2


def exercise_rv_adaptation(delta_spap: Optional[float], ci_rest: Optional[float], ci_peak: Optional[float]) -> Optional[str]:
    """
    Homeometric vs. heterometric (heuristic):
    - If ΔsPAP > 30 mmHg and CI under exercise is not worse (>= rest): homeometric.
    - Else if ΔsPAP > 30 and CI falls: heterometric.
    - Else: None.
    """
    if delta_spap is None or ci_rest is None or ci_peak is None:
        return None
    if delta_spap <= 30:
        return None
    if ci_peak >= ci_rest:
        return "homeometrisch"
    return "heterometrisch"
