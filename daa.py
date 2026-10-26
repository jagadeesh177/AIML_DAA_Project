"""
DAA analyser: binary search on the answer.

    python daa.py --target 50                                          # fewest stations for 50% of the city's demand
    python daa.py --values "12,22,30,36,41,47,52,57,61,64" --target 50   # your own sorted list

Every run prints the input, each step of the binary search (low, high, mid and
the decision), the answer, and an analysis that compares the number of checks
(and, for the city, the time) with a linear search, which also confirms the answer.
"""
import argparse
import contextlib
import io
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from main import MAX_STATIONS, best_coverage, binary_search_min, learn_demand  # noqa: E402


def linear_search_min(check, low, high):
    """Try k = low, low + 1, ... and stop at the first k that is enough: O(n) checks."""
    for k in range(low, high + 1):
        if check(k)[1]:
            return k
    return None


class Checks:
    """Wraps a check function. It remembers every result, so both searches see the same
    numbers, and records which k each search asked for and how long each check took."""

    def __init__(self, compute):
        self.compute, self.memo, self.cost, self.asked = compute, {}, {}, []

    def __call__(self, k):
        self.asked.append(k)
        if k not in self.memo:
            t = time.perf_counter()
            self.memo[k] = self.compute(k)
            self.cost[k] = time.perf_counter() - t
        return self.memo[k]

    def take(self):
        """The k values asked for since the last call."""
        asked, self.asked = self.asked, []
        return asked


def banner(text):
    print("\n" + "=" * 70 + f"\n {text}\n" + "=" * 70)


def print_steps(steps, show):
    print(f"{'step':>4}{'low':>6}{'high':>6}{'mid':>6}{'value':>10}   decision")
    for i, (low, high, mid, value, enough) in enumerate(steps, 1):
        decision = "enough     -> answer is mid or smaller" if enough else "not enough -> answer is larger than mid"
        print(f"{i:>4}{low:>6}{high:>6}{mid:>6}{show(value):>10}   {decision}")


def analyse(checks, binary_asked, linear_asked, b_answer, l_answer, n, timed):
    banner("ANALYSIS: number of checks" + (" and time" if timed else ""))
    print(f"Range size n = {n}: binary search needs at most ceil(log2(n + 1)) = {math.ceil(math.log2(n + 1))} checks, "
          f"linear search up to {n}.")
    print(f"{'method':<32}{'checks':>8}" + (f"{'time (ms)':>12}" if timed else "") + "   values of k checked")
    for name, asked in (("Binary search  [O(log n)]", binary_asked), ("Linear search  [O(n)]", linear_asked)):
        ms = f"{sum(checks.cost[k] for k in asked) * 1000:>12.0f}" if timed else ""
        print(f"{name:<32}{len(asked):>8}{ms}   {asked}")
    same = b_answer == l_answer
    print(f"Linear search finds the same answer: {'yes' if same else 'NO'}")
    if not same:
        dips = [(a, b) for a in checks.memo for b in checks.memo
                if a < b and checks.memo[a][1] and not checks.memo[b][1]]
        if dips:
            a, b = min(dips)
            print(f"The checks are not monotonic here: {a} was enough but {b} was not. Binary search assumes "
                  f"that once k is enough, every larger k is enough too.")


# ---------------------------------------------------------------- input 1: the city
def city(target, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        a, b, mae, demand, used = learn_demand()
    total = sum(demand.values())
    goal = target / 100 * total

    banner("INPUT")
    print(f"City: 10 x 10 = 100 zones, total predicted demand {total:.1f} charging sessions/day")
    print(f"Target: {target:g}% of the demand = {goal:.1f} sessions/day")
    print(f"Search range: 1 to {MAX_STATIONS} stations (check = best coverage found by simulated annealing)")

    def compute(k):
        random.seed(seed * 1000 + k)          # the same k always gives the same plan, whichever search asks
        covered = best_coverage(demand, k)
        return covered, covered >= goal

    checks = Checks(compute)
    answer, steps = binary_search_min(checks, 1, MAX_STATIONS)
    binary_asked = checks.take()

    banner("OUTPUT: binary search steps")
    print_steps(steps, lambda v: f"{v / total:.1%}")
    if answer is None:
        print(f"Answer: not possible, even {MAX_STATIONS} stations cover less than {target:g}% of the demand")
    else:
        print(f"Answer: {answer} stations are needed to cover {target:g}% of the demand "
              f"({checks.memo[answer][0] / total:.1%} covered)")

    l_answer = linear_search_min(checks, 1, MAX_STATIONS)
    analyse(checks, binary_asked, checks.take(), answer, l_answer, MAX_STATIONS, timed=True)


# ---------------------------------------------------------------- input 2: the user's own sorted list
def values_list(text, target):
    try:
        values = [float(x) for x in text.split(",")]
    except ValueError:
        sys.exit(f"--values must be numbers separated by commas, e.g. \"12,22,30\" (got \"{text}\")")
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            sys.exit(f"--values must be sorted from smallest to largest: {values[i]:g} comes after "
                     f"{values[i - 1]:g}, so binary search cannot be used")

    banner("INPUT")
    print(f"Values for k = 1..{len(values)}: {', '.join(f'{v:g}' for v in values)}")
    print(f"Target: the first k whose value is at least {target:g}")

    checks = Checks(lambda k: (values[k - 1], values[k - 1] >= target))
    answer, steps = binary_search_min(checks, 1, len(values))
    binary_asked = checks.take()

    banner("OUTPUT: binary search steps")
    print_steps(steps, lambda v: f"{v:g}")
    if answer is None:
        print(f"Answer: no value reaches {target:g}")
    else:
        print(f"Answer: k = {answer} (value {values[answer - 1]:g} is the first one >= {target:g})")

    l_answer = linear_search_min(checks, 1, len(values))
    analyse(checks, binary_asked, checks.take(), answer, l_answer, len(values), timed=False)


def main():
    p = argparse.ArgumentParser(description="DAA analyser: binary search on the answer")
    p.add_argument("--target", type=float, required=True,
                   help="coverage target in percent (city), or the value to reach (with --values)")
    p.add_argument("--values", help='your own sorted list instead of the city, e.g. "12,22,30,36,41"')
    p.add_argument("--seed", type=int, default=1, help="random seed for the optimizer (city only)")
    args = p.parse_args()
    if args.values:
        values_list(args.values, args.target)
    elif not 0 < args.target <= 100:
        p.error("--target must be a percentage above 0 and at most 100")
    else:
        city(args.target, args.seed)


if __name__ == "__main__":
    main()
