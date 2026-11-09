from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    entry: str
    frame: str
    global_grid: str
    positions: str


ARMS = {
    "A1": Arm("small", "lo", "lo", "native"),
    "A2": Arm("small", "lo", "lo", "pi_hr"),
    "A3": Arm("pixel", "hi", "hi", "native"),
    "A4": Arm("pixel", "hi", "lo", "pi_hr"),
    "A5": Arm("token", "hi", "lo", "pi_hr"),
    "A6": Arm("pixel", "lo", "hi", "native"),
    "A7": Arm("pixel", "hi", "lo", "native"),
    "A8": Arm("real", "hi", "hi", "native"),
}
