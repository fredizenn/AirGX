from airgx.calibration import calibrate, hand_bends
from airgx.models import FINGERS
from airgx.simulation import Segment, SimulatedGlove


def record(glove, start, end, rate_hz=100):
    frames = [glove.read(start + i / rate_hz) for i in range(int((end - start) * rate_hz))]
    return [f for f in frames if f is not None]


def main():
    # --- Step 1: the calibration routine -------------------------------
    print("Hold your hand OPEN... then make a FIST...")
    glove = SimulatedGlove([Segment(2.0, "flat"), Segment(2.0, "fist")])
    calibration = calibrate(record(glove, 0.5, 2.0), record(glove, 2.5, 4.0))

    for name, cal in zip(FINGERS, calibration):
        if cal is None:
            print(f"  {name:<7} no sensor")
        else:
            print(f"  {name:<7} open={cal.open_raw}  fist={cal.bent_raw}")

    # --- Step 2: use it ------------------------------------------------
    print("\nNow the same glove, reported as % bent:")
    glove = SimulatedGlove([
        Segment(1.0, "flat"),
        Segment(1.0, "point"),
        Segment(1.0, "thumbs_up"),
        Segment(1.0, "fist", stuck={1: 4095}),  # index wire comes loose
    ])
    for i in range(0, 40, 5):
        t = i / 10
        bends, problems = hand_bends(glove.read(t), calibration)
        shown = ["  --" if b is None else f"{b:4.0%}" for b in bends[:3]]
        print(f"  t={t:.1f}  thumb {shown[0]}  index {shown[1]}  middle {shown[2]}  {'; '.join(problems)}")


if __name__ == "__main__":
    main()