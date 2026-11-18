"""Times the grouped kernels at 1, 2 and 4 threads and prints the speedup table.

Not part of tests/test_efficiency.py: that suite runs under CodSpeed in
simulation mode, which counts instructions and so can't see what threads do.
With --check every gated kernel's two-thread speedup on the uniform dataset
has to clear MIN_SPEEDUP, which is what the CI job runs.
"""

import argparse
import os
import sys
import time

import numpy as np
from coreforecast.grouped_array import GroupedArray

THREAD_COUNTS = [1, 2, 4]

# The gate is on two threads, not four: GitHub's four vCPUs are two cores plus
# their hyperthreads, where box-cox gets 2.3x rather than the 3.8x it gets on
# four real cores, so a four-thread floor would say more about the runner than
# about the code. Two threads measured 1.9x to 2.0x on both machines, and the
# regression this guards against (quantiles before the skip list got a
# per-thread coin toss) was 0.51x there, so the margin is wide either way.
CHECKED_THREADS = 2
MIN_SPEEDUP = 1.5

# The third field says whether --check gates the kernel. A gated kernel needs
# tens of milliseconds of single-thread work on the uniform dataset: below that
# the measurement sits inside the noise of a shared runner and the ~0.07 ms it
# costs to spawn the threads, and a gate that flakes gets re-run rather than
# read. The two rolling kernels are memory-bound and finish in a few
# milliseconds (rolling_mean got 1.78x on two threads in CI, where the gated
# kernels got 1.94x to 1.99x), so they are timed and printed but not gated.
KERNELS = [
    ("rolling_mean", lambda ga: ga._rolling_mean(1, 7, 1), False),
    ("rolling_max", lambda ga: ga._rolling_max(1, 7, 1), False),
    ("robust_iqr_stats", lambda ga: ga._robust_iqr_stats(), True),
    ("boxcox_loglik", lambda ga: ga._boxcox_loglik(-0.9, 2.0), True),
    ("rolling_quantile (w=7)", lambda ga: ga._rolling_quantile(1, 0.5, 7, 1), True),
    ("rolling_quantile (w=100)", lambda ga: ga._rolling_quantile(1, 0.5, 100, 1), True),
]

NAME_WIDTH = 26
CELL_WIDTH = 10


def build(rng, lengths):
    indptr = np.append(0, np.cumsum(lengths)).astype(np.int64)
    # positive, since box-cox takes the log of it
    return rng.uniform(low=1.0, high=100.0, size=indptr[-1]), indptr


def uniform(rng):
    return build(rng, rng.integers(low=1000, high=2000, size=1000))


def huge_group(rng):
    lengths = rng.integers(low=1000, high=2000, size=1000)
    return build(rng, np.append(lengths, lengths.sum()))


def tiny(rng):
    return build(rng, np.full(200, 20))


# Only the uniform case is checked. The other two are what the rest
# of the threading plan is about: handing work out dynamically so one huge
# group doesn't pin a thread, and not spawning at all for an array this small.
DATASETS = [
    ("uniform", uniform, 5, True),
    ("one huge group", huge_group, 5, False),
    ("tiny", tiny, 200, False),
]


def available_cores():
    # not hardware_concurrency: taskset and cgroup cpusets are what a CI runner
    # hands us, and only the affinity mask reflects them
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def time_it(fn, ga, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(ga)
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def cell(text):
    return f"| {text:>{CELL_WIDTH}} "


def run(name, dataset, repeats, floors):
    data, indptr = dataset
    print(f"\n## {name}: {len(indptr) - 1} groups, {data.size} elements")
    header = f"| {'kernel':<{NAME_WIDTH}}"
    header += "".join(cell(f"{t}t (ms)") for t in THREAD_COUNTS)
    header += "".join(cell(f"{t}t speedup") for t in THREAD_COUNTS[1:]) + "|"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    below = []
    for kernel_name, fn, gated in KERNELS:
        times = {
            t: time_it(fn, GroupedArray(data, indptr, num_threads=t), repeats)
            for t in THREAD_COUNTS
        }
        speedups = {t: times[1] / times[t] for t in THREAD_COUNTS[1:]}
        row = f"| {kernel_name:<{NAME_WIDTH}}"
        row += "".join(cell(f"{times[t]:.3f}") for t in THREAD_COUNTS)
        row += "".join(cell(f"{speedups[t]:.2f}") for t in THREAD_COUNTS[1:])
        print(row + "|")
        if floors and gated and speedups[CHECKED_THREADS] < MIN_SPEEDUP:
            below.append(
                f"{kernel_name}: {speedups[CHECKED_THREADS]:.2f}x on "
                f"{CHECKED_THREADS} threads, floor {MIN_SPEEDUP:.2f}x"
            )
    return below


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=f"exit non-zero if a {CHECKED_THREADS}-thread speedup is under "
        f"{MIN_SPEEDUP}x",
    )
    args = parser.parse_args()
    cores = available_cores()
    print(f"{cores} cores available")
    if args.check:
        if cores < CHECKED_THREADS:
            # passing here would make the gate disappear the day a runner shrinks
            sys.exit(f"--check needs {CHECKED_THREADS} cores to mean anything")
        names = ", ".join(name for name, _, gated in KERNELS if gated)
        print(f"gated at {MIN_SPEEDUP}x on {CHECKED_THREADS} threads: {names}")
    rng = np.random.default_rng(0)
    below = []
    for name, dataset, repeats, floors in DATASETS:
        below += run(name, dataset(rng), repeats, floors and args.check)
    if below:
        print("\nBelow the floor:")
        for line in below:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
