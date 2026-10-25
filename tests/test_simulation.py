"""Tests that SimulatedGlove produces believable, repeatable sensor data."""

import pytest

from airgx.simulation import Segment, SimulatedGlove


def test_same_seed_gives_identical_frames():
    """Two gloves with the same script and seed produce exactly the same frames, noise included."""
    script = [Segment(1.0, "flat"), Segment(1.0, "fist")]
    a = SimulatedGlove(script, seed=42)
    b = SimulatedGlove(script, seed=42)
    for i in range(50):
        t = i * 0.04
        assert a.read(t) == b.read(t)


def test_fist_reads_higher_than_flat():
    """Bending a finger into a fist raises its flex reading well above the flat-hand reading."""
    glove = SimulatedGlove([Segment(1.0, "flat"), Segment(1.0, "fist")])
    flat = glove.read(0.5)
    fist = glove.read(1.5)
    index = 1
    assert fist.flex_raw[index] > flat.flex_raw[index] + 1000


def test_unfitted_fingers_are_none():
    """Ring and pinky have no sensor on the prototype, so they read None."""
    frame = SimulatedGlove([Segment(1.0)]).read(0.0)
    assert frame.flex_raw[3] is None and frame.flex_raw[4] is None


def test_flat_hand_feels_gravity_on_z():
    """With the hand held level, all of gravity shows up on the z-axis."""
    frame = SimulatedGlove([Segment(1.0, "flat")]).read(0.5)
    assert frame.accel.x == pytest.approx(0.0, abs=0.3)
    assert frame.accel.z == pytest.approx(9.81, abs=0.3)


def test_tilting_30_degrees_down():
    """Tilting fingertips down 30° splits gravity: -g·sin30° on x, g·cos30° on z."""
    frame = SimulatedGlove([Segment(1.0, "flat", pitch_deg=30)]).read(0.5)
    assert frame.accel.x == pytest.approx(-4.9, abs=0.3)
    assert frame.accel.z == pytest.approx(8.5, abs=0.3)


def test_dropout_sends_nothing():
    """During a dropout segment, read() returns None instead of a frame."""
    glove = SimulatedGlove([Segment(1.0, "flat"), Segment(1.0, dropout=True)])
    assert glove.read(0.5) is not None
    assert glove.read(1.5) is None


def test_seq_counts_up():
    """Each frame's seq number is one higher than the frame before it, starting at 0."""
    glove = SimulatedGlove([Segment(1.0)])
    assert [glove.read(i * 0.01).seq for i in range(3)] == [0, 1, 2]
