from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    entry: str
    frame: str
    global_grid: str
    positions: str


ARMS = {
    "A0": Arm("real", "hi", "hi", "native"),
    "A1": Arm("small", "lo", "lo", "native"),
    "A2": Arm("small", "lo", "lo", "pi_hr"),
    "A3": Arm("pixel", "hi", "hi", "native"),
    "A4": Arm("pixel", "hi", "lo", "pi_hr"),
    "A5": Arm("token", "hi", "lo", "pi_hr"),
    "A6": Arm("pixel", "lo", "hi", "native"),
    "A7": Arm("pixel", "hi", "lo", "native"),
    "A8": Arm("real", "hi", "hi", "native"),
    # Salvation track B: real pixels for every view, sharp frame attention, pooled global attention.
    "A9": Arm("real", "hi", "lo", "pi_hr"),
    # Salvation track C: encode each view at its own size, lift tokens to the sharp grid,
    # then run the whole decoder on the sharp grid. No pixel upsampling.
    "A10": Arm("token", "hi", "hi", "native"),
}

RMFA_ARMS = frozenset({"A4", "A5", "A7", "A9"})
