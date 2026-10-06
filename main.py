"""
EV Charging Station Placement Planner

This project finds the optimal locations for electric vehicle charging stations
in a 10x10 city grid using:
  1. Linear Regression (Ordinary Least Squares) to predict demand from survey data.
  2. Local search optimization (Hill Climbing vs Simulated Annealing) to place stations.
  3. Binary Search to determine the minimum number of stations needed for a target demand.

Run:
  python main.py
"""
import csv
import math
import random
from itertools import combinations

# City grid dimensions and default constraints
SIZE = 10          # 10x10 grid (100 zones total)
STATIONS = 3       # Default number of stations to place


# ---------------------------------------------------------------------------
# 1. Demand Prediction (Linear Regression)
# ---------------------------------------------------------------------------
def load_csv(name):
    """Load rows from a CSV file into a list of dictionaries."""
    with open(name, newline="") as f:
        return list(csv.DictReader(f))


def clean(rows):
    """Data cleaning: remove rows with missing demand values."""
    good = [r for r in rows if r["demand"].strip() != ""]
    print(f"Survey rows: {len(rows)}, missing: {len(rows) - len(good)}, valid rows used: {len(good)}")
    return good


def train_linear_regression(x, y):
    """
    Fits a simple linear regression model: y = a + b * x
    Uses standard Ordinary Least Squares (OLS) formulas:
      b = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
      a = mean_y - b * mean_x
    """
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    b = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sum((xi - mean_x) ** 2 for xi in x)
    a = mean_y - b * mean_x
    return a, b


def mean_absolute_error(actual, predicted):
    """Calculate Mean Absolute Error between actual and predicted values."""
    return sum(abs(p - q) for p, q in zip(actual, predicted)) / len(actual)


# ---------------------------------------------------------------------------
# 2. State Space & Coverage Model
# ---------------------------------------------------------------------------
# State: Tuple of 3 station coordinates, e.g. ((2, 2), (2, 8), (7, 6))
# Move: Shift one station by one zone (up, down, left, right)
# Score: Total demand covered by all stations without double counting

def covered_zones(state):
    """
    Returns the set of all zones covered by the stations.
    Each station covers its own zone plus its 4 orthogonal neighbors.
    Using a set prevents double-counting overlapping zones.
    """
    zones = set()
    for r, c in state:
        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE:
                zones.add((r + dr, c + dc))
    return zones


def score(state, demand):
    """Calculate total demand served by the given station placement."""
    return sum(demand[z] for z in covered_zones(state))


def random_state(k=STATIONS):
    """Generate a random placement of k stations."""
    all_zones = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return tuple(random.sample(all_zones, k))


def neighbours(state):
    """Generate all valid states reachable by moving one station by one step."""
    result = []
    for i, (r, c) in enumerate(state):
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new = (r + dr, c + dc)
            if 0 <= new[0] < SIZE and 0 <= new[1] < SIZE and new not in state:
                result.append(state[:i] + (new,) + state[i + 1:])
    return result


# ---------------------------------------------------------------------------
# 3. Optimization Algorithms
# ---------------------------------------------------------------------------
def hill_climbing(demand):
    """
    Steepest-Ascent Hill Climbing:
    Greedily moves to the neighboring state with the highest score.
    Stops when no neighbor has a better score (local maximum).
    """
    state = random_state()
    while True:
        best = max(neighbours(state), key=lambda s: score(s, demand))
        if score(best, demand) <= score(state, demand):
            return state
        state = best


def simulated_annealing(demand, temperature=80.0, cooling=0.9993, steps=8000, k=STATIONS):
    """
    Simulated Annealing:
    Stochastic local search. Always accepts better moves.
    Accepts worse moves with probability P = exp(delta / T).
    Temperature cools gradually: T = T * cooling_rate.
    This allows the algorithm to escape local maxima early in the search.
    """
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
    """
    Checks all C(100, 3) = 161,700 combinations to find the exact global maximum.
    Used as an exact baseline to verify heuristic algorithm performance.
    """
    all_zones = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return max(combinations(all_zones, STATIONS), key=lambda s: score(s, demand))


# ---------------------------------------------------------------------------
# 4. Binary Search for Minimum Station Count
# ---------------------------------------------------------------------------
# Problem: What is the smallest number of stations needed to cover target_share (e.g. 50%)?
# Property: More stations never cover less demand (monotonic non-decreasing).
# Therefore, binary search can find the threshold in O(log N) evaluations instead of O(N).
MAX_STATIONS = 20


def best_coverage(demand, k):
    """Returns coverage achieved with k stations using simulated annealing."""
    return max(score(simulated_annealing(demand, steps=4000, k=k), demand) for _ in range(2))


def min_stations_needed(demand, target_share):
    """
    Binary searches the range [1, MAX_STATIONS] for the minimum station count.
    Returns (answer, step_history).
    """
    goal = target_share * sum(demand.values())
    low, high = 1, MAX_STATIONS + 1
    steps = []
    while low < high:
        mid = (low + high) // 2
        covered = best_coverage(demand, mid)
        enough = covered >= goal
        steps.append((mid, covered, enough))
        if enough:
            high = mid
        else:
            low = mid + 1
    return (low if low <= MAX_STATIONS else None), steps


# ---------------------------------------------------------------------------
# Terminal Display Helpers
# ---------------------------------------------------------------------------
def print_map(state, demand):
    """Prints a text map: S = Station, # = Covered Zone, . = Not Covered."""
    covered = covered_zones(state)
    for r in range(SIZE):
        line = ""
        for c in range(SIZE):
            line += " S" if (r, c) in state else " #" if (r, c) in covered else " ."
        print("   " + line)


def learn_demand():
    """Cleans survey data, fits regression model, and extrapolates to all 100 zones."""
    survey = clean(load_csv("survey.csv"))
    train, test = survey[:18], survey[18:]
    a, b = train_linear_regression([float(r["cars"]) for r in train], [float(r["demand"]) for r in train])
    actual = [float(r["demand"]) for r in test]
    predicted = [a + b * float(r["cars"]) for r in test]
    mae = mean_absolute_error(actual, predicted)
    city = load_csv("city.csv")
    demand = {(int(z["row"]), int(z["col"])): a + b * int(z["cars"]) for z in city}
    return a, b, mae, demand, len(survey)


def main():
    random.seed(1)

    print("=" * 60)
    print("Part 1: Predicting Demand from Survey Data (Linear Regression)")
    print("=" * 60)
    a, b, mae, demand, used = learn_demand()
    print(f"Regression model:   demand = {a:.2f} + {b:.2f} * cars")
    print(f"Test error (MAE):   {mae:.2f} sessions/day")
    print(f"Total city demand:  {sum(demand.values()):.1f} sessions/day across {len(demand)} zones")

    print("\n" + "=" * 60)
    print(f"Part 2: Placing {STATIONS} Stations (Algorithm Comparison - 10 Runs Each)")
    print("=" * 60)
    hc = [score(hill_climbing(demand), demand) for _ in range(10)]
    sa = [score(simulated_annealing(demand), demand) for _ in range(10)]
    best = brute_force(demand)
    best_score = score(best, demand)

    print(f"{'Algorithm':<22}{'Mean':>8}{'Min':>8}{'Max':>8}{'Optimal Hits':>16}")
    for name, results in [("Hill Climbing", hc), ("Simulated Annealing", sa)]:
        hits = sum(abs(s - best_score) < 1e-9 for s in results)
        print(f"{name:<22}{sum(results) / 10:>8.1f}{min(results):>8.1f}{max(results):>8.1f}{hits:>11} / 10")
    print(f"{'Brute Force (Exact)':<22}{best_score:>8.1f}{best_score:>8.1f}{best_score:>8.1f}{'10 / 10':>16}")

    plan = simulated_annealing(demand)
    print(f"\nOptimal locations: {sorted(plan)}")
    print(f"Total demand covered: {score(plan, demand):.1f} sessions/day ({(score(plan, demand)/sum(demand.values()))*100:.1f}%)")
    print_map(plan, demand)

    print("\n" + "=" * 60)
    print("Part 3: Minimum Stations for 50% Coverage (Binary Search)")
    print("=" * 60)
    answer, steps = min_stations_needed(demand, 0.50)
    total = sum(demand.values())
    for i, (k, covered, enough) in enumerate(steps, 1):
        print(f"  Check {i}: {k:>2} stations -> covers {covered / total:5.1%} demand | "
              f"{'enough -> try fewer' if enough else 'not enough -> try more'}")
    print(f"\nResult: {answer} stations needed for 50% coverage.")
    print(f"Found in {len(steps)} checks using binary search (linear search would take up to 20 checks).")
    print("=" * 60)


if __name__ == "__main__":
    main()
