"""Data shapes for what the glove sends us, validated by pydantic."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

ADC_MAX = 4095  # highest reading a 12-bit ADC can give
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class Vec3(BaseModel):
    """A 3D reading (x, y, z) from the accelerometer or gyro. Can't be changed once made."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float


class SensorFrame(BaseModel):
    """One snapshot of every sensor on the glove at a single moment.

    seq:      packet counter, counts up from 0
    t:        timestamp in seconds
    flex_raw: one raw ADC reading per finger (see FINGERS); None = no sensor fitted
    accel:    accelerometer reading in m/s² (includes gravity)
    gyro:     gyroscope reading
    clutch:   True while the clutch button is held

    Frames are frozen: once made, they can't be edited.
    """

    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=0)
    t: float
    flex_raw: tuple[int | None, ...]
    accel: Vec3
    gyro: Vec3
    clutch: bool = False

    @field_validator("flex_raw")
    @classmethod
    def check_flex(cls, v):
        """Reject frames that don't have exactly one reading per finger, or that have readings outside 0..ADC_MAX."""
        if len(v) != len(FINGERS):
            raise ValueError(f"expected {len(FINGERS)} flex channels, got {len(v)}")
        for raw in v:
            if raw is not None and not 0 <= raw <= ADC_MAX:
                raise ValueError(f"flex reading {raw} is outside 0..{ADC_MAX}")
        return v
