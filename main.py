"""
EV Charge Planner (simple version)

Where should a city build 3 electric-vehicle charging stations?

  Step 1 (Module 3)  Learn charging demand from a small survey: linear regression
  Step 2 (Module 1)  Describe the problem as states, moves and a score
  Step 3 (Module 2)  Find a good plan with hill climbing and simulated annealing

Run:  python main.py
"""
import csv
import math
import random
from itertools import combinations

SIZE = 10          # the city is a 10 x 10 grid of zones
STATIONS = 3       # how many stations the city can afford


# ---------------------------------------------------------------------------
# STEP 1 - Machine learning: predict demand from the number of EVs in a zone
# ---------------------------------------------------------------------------
def load_csv(name):
    with open(name, newline="") as f:
        return list(csv.DictReader(f))


def clean(rows):
    """Data preparation: drop rows where the demand value is missing."""
    good = [r for r in rows if r["demand"].strip() != ""]
    print(f"Survey rows: {len(rows)}, missing demand: {len(rows) - len(good)}, used: {len(good)}")
    return good


def train_linear_regression(x, y):
    """Simple linear regression  y = a + b*x  (least squares formula)."""
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    b = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sum((xi - mean_x) ** 2 for xi in x)
    a = mean_y - b * mean_x
    return a, b


def mean_absolute_error(actual, predicted):
    return sum(abs(p - q) for p, q in zip(actual, predicted)) / len(actual)


# ---------------------------------------------------------------------------
# STEP 2 - The problem as states, moves and a score
# ---------------------------------------------------------------------------
#   State : the zones where the 3 stations are,  e.g. ((1,2), (6,6), (3,8))
#   Move  : shift ONE station one zone up / down / left / right
#   Score : total demand of all zones covered by a station
#           (a station covers its own zone and the 4 zones next to it;
#            a zone covered by two stations only counts once)
def covered_zones(state):
    zones = set()
    for r, c in state:
        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE:
                zones.add((r + dr, c + dc))
    return zones


def score(state, demand):
    return sum(demand[z] for z in covered_zones(state))


def random_state(k=STATIONS):
    all_zones = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return tuple(random.sample(all_zones, k))


def neighbours(state):
    """All states you get by moving one station by one zone."""
    result = []
    for i, (r, c) in enumerate(state):
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new = (r + dr, c + dc)
            if 0 <= new[0] < SIZE and 0 <= new[1] < SIZE and new not in state:
                result.append(state[:i] + (new,) + state[i + 1:])
    return result


# ---------------------------------------------------------------------------
# STEP 3 - Optimization algorithms
# ---------------------------------------------------------------------------
def hill_climbing(demand):
    """Keep moving to the best neighbour. Stop when no neighbour is better."""
    state = random_state()
    while True:
        best = max(neighbours(state), key=lambda s: score(s, demand))
        if score(best, demand) <= score(state, demand):
            return state                          # stuck: a local maximum
        state = best


def simulated_annealing(demand, temperature=80.0, cooling=0.9993, steps=8000, k=STATIONS):
    """Like hill climbing, but sometimes accepts a WORSE move.
    This happens often while the temperature is high and rarely once it has cooled,
    which lets it escape local maxima early on."""
    state = random_state(k)
    best = state
    for _ in range(steps):
        new = random.choice(neighbours(state))
        change = score(new, demand) - score(state, demand)
        if change > 0 or random.random() < math.exp(change / temperature):
            state = new
        if score(state, demand) > score(best, demand):
            best = state
        temperature *= cooling
    return best


def brute_force(demand):
    """Check every possible plan. Possible here (161,700 plans), but not for a real city."""
    all_zones = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return max(combinations(all_zones, STATIONS), key=lambda s: score(s, demand))


# ---------------------------------------------------------------------------
# STEP 4 - DAA: Binary search on the answer
# ---------------------------------------------------------------------------
#   Question: what is the SMALLEST number of stations that covers a target
#   share (e.g. 50%) of the city's demand?
#
#   Key fact: more stations can never cover LESS demand, so the answers are
#   "sorted":  1 station: no,  2: no,  ...,  7: yes,  8: yes,  ...
#   That means we can binary-search for the first "yes" instead of trying
#   every number. Searching 1..MAX_STATIONS needs only about log2(20) = 5 checks
#   instead of up to 20:  O(log n) checks instead of O(n).
MAX_STATIONS = 20


def best_coverage(demand, k):
    """Demand covered by k stations, using the best of 2 quick simulated-annealing runs."""
    return max(score(simulated_annealing(demand, steps=4000, k=k), demand) for _ in range(2))


def min_stations_needed(demand, target_share):
    """Binary search for the smallest k whose coverage reaches target_share of all demand.
    Returns (answer or None if even MAX_STATIONS is not enough, list of steps tried)."""
    goal = target_share * sum(demand.values())
    low, high = 1, MAX_STATIONS + 1          # high = MAX_STATIONS + 1 means "not possible"
    steps = []
    while low < high:
        mid = (low + high) // 2
        covered = best_coverage(demand, mid)
        enough = covered >= goal
        steps.append((mid, covered, enough))
        if enough:
            high = mid           # mid works: the answer is mid or smaller
        else:
            low = mid + 1        # mid is not enough: the answer is bigger
    return (low if low <= MAX_STATIONS else None), steps


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def print_map(state, demand):
    """S = station, # = covered zone, . = not covered."""
    covered = covered_zones(state)
    for r in range(SIZE):
        line = ""
        for c in range(SIZE):
            line += " S" if (r, c) in state else " #" if (r, c) in covered else " ."
        print("   " + line)


def learn_demand():
    """Step 1 in one call: clean the survey, train, test, predict every zone."""
    survey = clean(load_csv("survey.csv"))
    train, test = survey[:18], survey[18:]                     # 18 rows to learn, 6 to test
    a, b = train_linear_regression([float(r["cars"]) for r in train], [float(r["demand"]) for r in train])
    actual = [float(r["demand"]) for r in test]
    predicted = [a + b * float(r["cars"]) for r in test]
    mae = mean_absolute_error(actual, predicted)
    city = load_csv("city.csv")
    demand = {(int(z["row"]), int(z["col"])): a + b * int(z["cars"]) for z in city}
    return a, b, mae, demand, len(survey)


def main():
    random.seed(1)

    print("=" * 60 + "\nSTEP 1  Learn demand from the survey (linear regression)\n" + "=" * 60)
    a, b, mae, demand, used = learn_demand()
    print(f"Learned model:  demand = {a:.2f} + {b:.2f} x cars")
    print(f"Test error (MAE): {mae:.2f} charging sessions per day")
    print(f"Predicted demand for all {len(demand)} zones (only {used} were surveyed)")

    print("\n" + "=" * 60 + f"\nSTEP 2+3  Place {STATIONS} stations: run each algorithm 10 times\n" + "=" * 60)
    hc = [score(hill_climbing(demand), demand) for _ in range(10)]
    sa = [score(simulated_annealing(demand), demand) for _ in range(10)]
    best = brute_force(demand)
    best_score = score(best, demand)
    print(f"{'algorithm':<22}{'average':>9}{'worst':>8}{'best':>8}{'found the best plan':>22}")
    for name, results in [("Hill climbing", hc), ("Simulated annealing", sa)]:
        hits = sum(abs(s - best_score) < 1e-9 for s in results)
        print(f"{name:<22}{sum(results) / 10:>9.1f}{min(results):>8.1f}{max(results):>8.1f}{hits:>17} / 10")
    print(f"{'Brute force (check)':<22}{best_score:>9.1f}   <- the true best plan")

    plan = simulated_annealing(demand)
    print(f"\nSimulated annealing plan: stations at {sorted(plan)}, "
          f"covering {score(plan, demand):.1f} sessions/day")
    print_map(plan, demand)

    print("\n" + "=" * 60 + "\nSTEP 4 (DAA)  How many stations cover 50% of demand?  Binary search\n" + "=" * 60)
    answer, steps = min_stations_needed(demand, 0.50)
    total = sum(demand.values())
    for i, (k, covered, enough) in enumerate(steps, 1):
        print(f"  check {i}: {k:>2} stations cover {covered / total:5.1%}  ->  "
              f"{'enough, try fewer' if enough else 'not enough, try more'}")
    print(f"Answer: {answer} stations, found with {len(steps)} checks "
          f"(trying 1, 2, 3, ... one by one would need {answer} checks; for 1..{MAX_STATIONS} "
          f"binary search never needs more than 5)")


if __name__ == "__main__":
    main()
