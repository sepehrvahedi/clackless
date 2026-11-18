"""Contributor tool: run the acceptance benchmark over several sample rates and seeds.

    python scripts/evaluate.py                 # default grid
    python scripts/evaluate.py --rates 48000 --seeds 1 2 3 --no-tts

Use it before/after changing the DSP to see whether a change is a real
improvement across conditions rather than on one lucky example.
"""

import argparse
import statistics
from collections import defaultdict

from clackless.evaluate import run_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rates", type=int, nargs="+", default=[16000, 44100, 48000])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--no-tts", action="store_true")
    args = p.parse_args()

    results = defaultdict(list)
    failures = 0
    for sr in args.rates:
        for seed in args.seeds:
            checks = run_benchmark(sr, args.duration, seed=seed, use_tts=not args.no_tts)
            for c in checks:
                results[(c.name, c.unit)].append(c.value)
                failures += not c.passed
            print(f"done: {sr} Hz, seed {seed}")

    width = max(len(name) for name, _ in results)
    print(f"\n{'check':<{width}}  {'min':>7} {'mean':>7} {'max':>7}")
    for (name, unit), values in results.items():
        print(
            f"{name:<{width}}  {min(values):7.2f} {statistics.mean(values):7.2f} {max(values):7.2f}  {unit}"
        )
    print(f"\n{failures} failed checks")


if __name__ == "__main__":
    main()
