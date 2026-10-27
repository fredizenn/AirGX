import pytest

from airgx.calibration import CalibrationError, FingerCalibration, calibrate, hand_bends, normalise
from airgx.models import SensorFrame, Vec3
from airgx.simulation import SENSOR_RANGE, Segment, SimulatedGlove


def frame_with(flex):
    return SensorFrame(seq=0, t=0.0, flex_raw=flex,
                       accel=Vec3(x=0, y=0, z=9.81), gyro=Vec3(x=0, y=0, z=0))


def record(glove, start, end, rate_hz=100):
    frames = [glove.read(start + i / rate_hz) for i in range(int((end - start) * rate_hz))]
    return [f for f in frames if f is not None]


def test_calibration_recovers_the_true_ranges():
    glove = SimulatedGlove([Segment(2.0, "flat"), Segment(2.0, "fist")])
    open_frames = record(glove, 0.5, 2.0)   # skip the first 0.5 s: hand still settling
    fist_frames = record(glove, 2.5, 4.0)
    cal = calibrate(open_frames, fist_frames)

    straight, bent = SENSOR_RANGE[1]                       # index finger
    assert cal[1].open_raw == pytest.approx(straight + 0.05 * (bent - straight), abs=20)
    assert cal[1].bent_raw == pytest.approx(straight + 0.92 * (bent - straight), abs=20)
    assert cal[3] is None and cal[4] is None               # ring, pinky not fitted


def test_two_different_gloves_agree_on_half_bent():
    mine = FingerCalibration(open_raw=1200, bent_raw=2900)
    friends = FingerCalibration(open_raw=1500, bent_raw=2400)
    assert normalise(2050, mine)[0] == pytest.approx(0.5)
    assert normalise(1950, friends)[0] == pytest.approx(0.5)


def test_slightly_past_the_range_is_clamped():
    cal = FingerCalibration(open_raw=1200, bent_raw=2900)
    assert normalise(1150, cal) == (0.0, None)
    assert normalise(3000, cal) == (1.0, None)


def test_sensor_wired_backwards_still_works():
    cal = FingerCalibration(open_raw=2900, bent_raw=1200)  # reading goes DOWN as it bends
    assert normalise(2050, cal)[0] == pytest.approx(0.5)


def test_loose_wire_is_flagged():
    cal = FingerCalibration(open_raw=1200, bent_raw=2900)
    bend, problem = normalise(4095, cal)
    assert bend is None and "loose wire" in problem


def test_wild_reading_is_flagged():
    cal = FingerCalibration(open_raw=1200, bent_raw=2900)
    bend, problem = normalise(300, cal)
    assert bend is None and problem


def test_unattached_sensor_fails_calibration():
    still = [frame_with((1300, 1200, 1250, None, None))] * 10
    with pytest.raises(CalibrationError, match="thumb"):
        calibrate(still, still)


def test_median_ignores_a_spike():
    open_frames = [frame_with((1300, 1200, 1250, None, None))] * 9
    open_frames.append(frame_with((1300, 4000, 1250, None, None)))  # one glitch
    fist_frames = [frame_with((2500, 2800, 2700, None, None))] * 10
    cal = calibrate(open_frames, fist_frames)
    assert cal[1].open_raw == 1200


def test_simulated_broken_wire_shows_up_in_hand_bends():
    cal = calibrate([frame_with((1300, 1200, 1250, None, None))],
                    [frame_with((2500, 2800, 2700, None, None))])
    glove = SimulatedGlove([Segment(1.0, "flat", stuck={1: 4095})])
    bends, problems = hand_bends(glove.read(0.5), cal)
    assert bends[1] is None
    assert problems == ["index: stuck at 0 or 4095 (loose wire?)"]