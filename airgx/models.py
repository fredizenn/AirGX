from pydantic import BaseModel, ConfigDict, Field, field_validator

ADC_MAX = 4095
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

class Vec3(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    x: float
    y: float
    z: float
    
class SensorFrame(BaseModel):
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
        if len(v) != len(FINGERS):
            raise ValueError(f"expected {len(FINGERS)} flex channels, got{len(v)}")
        for raw in v:
            if raw is not None and not 0 <= raw <= ADC_MAX:
                raise ValueError(f"flex reading {raw} is outside 0..{ADC_MAX}")
        return v