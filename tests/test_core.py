import sys
from pathlib import Path

# Ensure repo root is on path (for rhk_textdb in this sandbox)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rhk.calcs import calc_mean, calc_pvr, calc_slope
from rhk.classify import classify_ph_rest, detect_step_up, classify_exercise_pattern
from rhk.generator import compute_all


def test_calc_mean():
    res = calc_mean(60, 20).value
    assert abs(res - 33.3333) < 1e-3


def test_classify_ph():
    rules = {"rest": {"mPAP_ph_mmHg": 20, "PAWP_postcap_mmHg": 15, "PVR_precap_WU": 2}}
    assert classify_ph_rest(19, 10, 3, rules).ph_type == "none"
    assert classify_ph_rest(25, 10, 3, rules).ph_type == "precap"
    assert classify_ph_rest(25, 20, 1, rules).ph_type == "ipcph"
    assert classify_ph_rest(25, 20, 3, rules).ph_type == "cpcph"


def test_step_up_detection():
    sats = {"SVC": 65, "IVC": 64, "RA": 75, "RV": 74, "PA": 73}
    present, loc, delta = detect_step_up(sats, threshold=7)
    assert present is True
    assert loc is not None
    assert delta is not None
    assert delta >= 7


def test_exercise_pattern():
    rules = {"exercise": {"mPAP_CO_slope_mmHg_per_L_min": 3.0, "PAWP_CO_slope_mmHg_per_L_min": 2.0}}
    assert classify_exercise_pattern(2.0, 1.0, rules) == "normal"
    assert classify_exercise_pattern(4.0, 3.0, rules) == "left_heart"
    assert classify_exercise_pattern(4.0, 1.0, rules) == "pulmonary_vascular"


def test_compute_all_basic():
    ui = {
        "height_cm": 180,
        "weight_kg": 80,
        "spap": 50,
        "dpap": 20,
        "pawp": 10,
        "rap": 5,
        "co_td": 5.0,
        "exercise_done": True,
        "mpap_peak": 45,
        "pawp_peak": 12,
        "co_peak": 8.0,
        "spap_peak": 90,
    }
    comp = compute_all(ui)
    assert comp.bsa is not None
    assert comp.mpap is not None  # computed from spap/dpap
    assert comp.pvr is not None
    assert comp.exercise_done is True
    assert comp.mpap_co_slope is not None
    assert comp.pawp_co_slope is not None
