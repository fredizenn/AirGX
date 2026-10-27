"""Tests that SensorFrame accepts good data and rejects bad data."""

from pydantic import ValidationError
import pytest

from airgx.models import SensorFrame, Vec3


def make_frame(**overrides):
    """Build a valid SensorFrame. Pass keyword arguments to swap out individual fields."""
    data = dict(
        seq=0,
        t=0.0,
        flex_raw=(1300, 1200, 1250, None, None),
        accel=Vec3(x=0, y=0, z=9.81),  # flat hand: gravity straight "up" the z-axis
        gyro=Vec3(x=0, y=0, z=0),  # not rotating
    )
    data.update(overrides)
    return SensorFrame(**data)


def test_valid_frame():
    """A normal frame is accepted, and fingers with no sensor are allowed to be None."""
    frame = make_frame()
    assert frame.flex_raw[3] is None


def test_rejects_wrong_channel_count():
    """Fewer than five flex readings is an error."""
    with pytest.raises(ValidationError):
        make_frame(flex_raw=(1, 2, 3))


def test_rejects_out_of_range_adc():
    """A flex reading above ADC_MAX (4095) is an error."""
    with pytest.raises(ValidationError):
        make_frame(flex_raw=(5000, 1200, 1250, None, None))


def test_frames_are_immutable():
    """Changing a field on an existing frame raises an error."""
    frame = make_frame()
    with pytest.raises(ValidationError):
        frame.seq = 5
