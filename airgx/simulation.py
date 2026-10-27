"""A fake glove for development without hardware.

You describe a timeline of hand poses (a list of Segments), and SimulatedGlove
turns it into noisy SensorFrames like the ones the real glove sends.
"""

import math
import random
from dataclasses import dataclass, field

from airgx.models import ADC_MAX, SensorFrame, Vec3

G = 9.81  # gravity, m/s²

# How bent each finger is for each hand shape: 0 = straight, 1 = fully bent.
# Order: thumb, index, middle, ring, pinky
HAND_SHAPES = {
    "relaxed":   (0.15, 0.10, 0.10, 0.15, 0.15),
    "flat":      (0.10, 0.05, 0.05, 0.05, 0.05),
    "point":     (0.90, 0.05, 0.90, 0.90, 0.90),
    "fist":      (0.90, 0.92, 0.92, 0.92, 0.92),
    "thumbs_up": (0.05, 0.92, 0.92, 0.92, 0.92),
}

# What each fake sensor reads (in ADC counts) when its finger is straight / fully bent.
SENSOR_RANGE = (
    (1350, 2650),  # thumb
    (1200, 2900),  # index
    (1250, 2800),  # middle
    (1300, 2750),  # ring
    (1280, 2700),  # pinky
)

# First prototype: sensors on thumb, index and middle only.
FITTED = (True, True, True, False, False)

@dataclass(frozen=True)
class Segment:
    """One step of a glove script: hold a shape and tilt for `duration` seconds."""
    duration: float          # seconds
    shape: str = "relaxed"
    pitch_deg: float = 0.0   # + = fingertips tilted down
    roll_deg: float = 0.0    # + = right side tilted down
    clutch: bool = False
    dropout: bool = False    # glove goes silent (link lost)
    stuck: dict = field(default_factory=dict) # {finger index: raw value} - a broken wire


def smoothstep(k):
    """0→1 with a gentle start and finish, like CSS ease-in-out."""
    return k * k * (3 - 2 * k)

class SimulatedGlove:
    """Plays a list of Segments and returns noisy sensor readings for any time you ask about.

    script:       the Segments to play, in order
    seed:         random seed, so the same script always gives the same readings
    adc_noise:    flex sensor noise (standard deviation, in ADC counts)
    accel_noise:  accelerometer noise (standard deviation, in m/s²)
    transition_s: how long the hand takes to move from one segment's pose to the next
    """

    def __init__(self, script, seed=7, adc_noise=15.0, accel_noise=0.05, transition_s=0.2):
        self.script = script
        self.rng = random.Random(seed)
        self.adc_noise = adc_noise
        self.accel_noise = accel_noise
        self.transition_s = transition_s
        self.seq = 0

    @property
    def duration(self):
        """Total length of the script, in seconds."""
        return sum(seg.duration for seg in self.script)

    def _locate(self, t):
        """Which segment is playing at time t, and how far into it are we?"""
        start = 0.0
        for i, seg in enumerate(self.script):
            if t < start + seg.duration:
                return i, t - start
            start += seg.duration
        return len(self.script) - 1, self.script[-1].duration

    def _pose(self, t):
        """The noise-free hand at time t, blending from the previous segment."""
        i, into = self._locate(t)
        seg = self.script[i]
        prev = self.script[i - 1] if i > 0 else seg
        k = smoothstep(min(1.0, into / self.transition_s)) if self.transition_s > 0 else 1.0

        def blend(a, b):
            return a + (b - a) * k

        bends = [blend(a, b) for a, b in zip(HAND_SHAPES[prev.shape], HAND_SHAPES[seg.shape])]
        pitch = math.radians(blend(prev.pitch_deg, seg.pitch_deg))
        roll = math.radians(blend(prev.roll_deg, seg.roll_deg))
        return seg, bends, pitch, roll

    def read(self, t):
        """One SensorFrame for time t — or None if the glove is silent.

        Flex readings are each finger's bend mapped onto its SENSOR_RANGE, plus noise.
        The accelerometer only measures gravity, split across x/y/z by the hand's tilt.
        """
        seg, bends, pitch, roll = self._pose(t)
        if seg.dropout:
            return None

        flex = []
        for i, (bend, (straight, bent), fitted) in enumerate(zip(bends, SENSOR_RANGE, FITTED)):
            if not fitted:
                flex.append(None)
                continue
            if i in seg.stuck:
                flex.append(seg.stuck[i])
                continue
            raw = straight + bend * (bent - straight) + self.rng.gauss(0, self.adc_noise)
            flex.append(int(min(ADC_MAX, max(0, round(raw)))))

        def noise():
            return self.rng.gauss(0, self.accel_noise)

        accel = Vec3(
            x=-G * math.sin(pitch) + noise(),
            y=G * math.sin(roll) * math.cos(pitch) + noise(),
            z=G * math.cos(roll) * math.cos(pitch) + noise(),
        )
        frame = SensorFrame(
            seq=self.seq,
            t=t,
            flex_raw=tuple(flex),
            accel=accel,
            gyro=Vec3(x=0, y=0, z=0),  # made real in Episode 4
            clutch=seg.clutch,
        )
        self.seq += 1
        return frame
