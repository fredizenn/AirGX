from statistics import median

from pydantic import BaseModel, ConfigDict

from airgx.models import ADC_MAX, FINGERS


class FingerCalibration(BaseModel):
    model_config = ConfigDict(frozen=True)
    open_raw: int   # reading with the finger straight
    bent_raw: int   # reading with the finger fully bent


class CalibrationError(ValueError):
    pass


def calibrate(open_frames, fist_frames, min_separation=200):
    """Learn each finger's range from frames recorded with an open hand, then a fist."""
    result = []
    for i, name in enumerate(FINGERS):
        open_vals = [f.flex_raw[i] for f in open_frames if f.flex_raw[i] is not None]
        fist_vals = [f.flex_raw[i] for f in fist_frames if f.flex_raw[i] is not None]
        if not open_vals or not fist_vals:
            result.append(None)  # no sensor on this finger
            continue
        open_raw = round(median(open_vals))
        bent_raw = round(median(fist_vals))
        if abs(bent_raw - open_raw) < min_separation:
            raise CalibrationError(
                f"{name}: open ({open_raw}) and fist ({bent_raw}) are too close. "
                "Is the sensor attached, and did the finger really bend?"
            )
        result.append(FingerCalibration(open_raw=open_raw, bent_raw=bent_raw))
    return tuple(result)


def normalise(raw, cal, rail_margin=10, slack=0.35):
    """Raw ADC count → (bend 0..1, problem). bend is None when the reading can't be trusted."""
    if cal is None or raw is None:
        return None, None  # finger not fitted: not a problem

    if raw <= rail_margin or raw >= ADC_MAX - rail_margin:
        return None, "stuck at 0 or 4095 (loose wire?)"

    low, high = sorted((cal.open_raw, cal.bent_raw))
    span = high - low
    if raw < low - slack * span or raw > high + slack * span:
        return None, "far outside the calibrated range"

    bend = (raw - cal.open_raw) / (cal.bent_raw - cal.open_raw)
    return min(1.0, max(0.0, bend)), None


def hand_bends(frame, calibration):
    """All five fingers at once → (bends, problems)."""
    bends, problems = [], []
    for name, raw, cal in zip(FINGERS, frame.flex_raw, calibration):
        bend, problem = normalise(raw, cal)
        bends.append(bend)
        if problem:
            problems.append(f"{name}: {problem}")
    return bends, problems