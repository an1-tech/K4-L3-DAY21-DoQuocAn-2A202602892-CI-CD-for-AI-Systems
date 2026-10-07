"""Quality gate shared by CI and model publication."""

import argparse
import math
import sys

F1_THRESHOLD = 0.65


def check_quality(value) -> float:
    f1 = float(value)
    if not math.isfinite(f1) or not F1_THRESHOLD <= f1 <= 1.0:
        raise ValueError(f"FAILED: f1_score={f1}; required 0.65 <= F1 <= 1.0")
    return f1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", required=True)
    args = parser.parse_args()
    try:
        f1 = check_quality(args.f1)
    except (TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"PASSED: f1_score={f1:.4f} >= {F1_THRESHOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
