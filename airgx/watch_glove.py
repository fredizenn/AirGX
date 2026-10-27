"""Play a scripted gesture sequence on the simulated glove and show it live in the terminal.

Run with:  python -m airgx.watch_glove
"""

import time

from airgx.simulation import Segment, SimulatedGlove

# Demo sequence: a few hand shapes, a tilt, then a link dropout and recovery.
SCRIPT = [
    Segment(1.0, "relaxed"),
    Segment(1.5, "fist"),
    Segment(1.5, "thumbs_up"),
    Segment(1.5, "flat", pitch_deg=25),
    Segment(1.0, dropout=True),
    Segment(1.0, "flat"),
]


def bar(raw, lo=1100, hi=3000, width=12):
    """Draw a raw flex reading as a text bar: empty at lo, full at hi. "-" means no sensor."""
    if raw is None:
        return "-".ljust(width)
    filled = round((raw - lo) / (hi - lo) * width)
    return ("█" * max(0, min(width, filled))).ljust(width)


def main():
    """Read the glove 10 times a second in real time and print one table row per reading."""
    glove = SimulatedGlove(SCRIPT)
    rate_hz = 10
    print(
        f"{'t':>4}  {'thumb':<12} {'index':<12} {'middle':<12} {'accel x':>8} {'accel z':>8}"
    )
    for i in range(int(glove.duration * rate_hz)):
        t = i / rate_hz
        f = glove.read(t)
        if f is None:
            print(f"{t:4.1f}  ... no data (glove silent) ...")
        else:
            thumb, index, middle = f.flex_raw[:3]
            print(
                f"{t:4.1f}  {bar(thumb)} {bar(index)} {bar(middle)} {f.accel.x:8.2f} {f.accel.z:8.2f}"
            )
        time.sleep(1 / rate_hz)


if __name__ == "__main__":
    main()
