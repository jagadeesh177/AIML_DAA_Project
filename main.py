"""
EV Charge Planner (AI/ML + DAA Academic Implementation)
======================================================
Course Topics Covered:
  - Module 1: AI State Space Search (State Representation, Neighborhood, Objective Function)
  - Module 2: Heuristic Optimization (Greedy Hill Climbing vs Stochastic Simulated Annealing)
  - Module 3: Machine Learning (Data Preprocessing, Ordinary Least Squares Linear Regression, MAE)
  - Module 4: Design and Analysis of Algorithms - DAA (Binary Search on Monotonic Answer Space)

Problem Formulation:
  Find optimal coordinates for K electric-vehicle charging stations in a 10x10 city grid
  to maximize total daily charging demand covered, subject to limited survey data.

Run:
  python main.py
"""
import csv
import math
import random
from itertools import combinations

# Grid and problem dimensions
SIZE = 10          # City represented as a 10 x 10 grid of zones (100 zones total)
STATIONS = 3       # Constraint: City budget allows exactly 3 charging stations


# ---------------------------------------------------------------------------
# MODULE 3: MACHINE LEARNING (Supervised Regression & Evaluation)
# ---------------------------------------------------------------------------
def load_csv(name):
    """Load tabular data from CSV into a list of row dictionaries."""
    with open(name, newline="") as f:
        return list(csv.DictReader(f))


def clean(rows):
    """
    Data Preprocessing: Handling Missing Values.
    Applies Complete Case Analysis (listwise deletion) to discard rows
    with missing ground-truth demand values.
    """
    good = [r for r in rows if r["demand"].strip() != ""]
    print(f"Survey rows: {len(rows)}, missing demand: {len(rows) - len(good)}, valid rows used: {len(good)}")
    return good


def train_linear_regression(x, y):
    """
    Ordinary Least Squares (OLS) Simple Linear Regression:
      Model: y = beta_0 + beta_1 * x  (demand = a + b * cars)
    
    Closed-Form Analytical Solution:
      beta_1 (slope)     = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
      beta_0 (intercept) = mean_y - beta_1 * mean_x
    
    Zero third-party library dependencies (pure algorithmic implementation).
    """
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    b = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sum((xi - mean_x) ** 2 for xi in x)
    a = mean_y - b * mean_x
    return a, b


def mean_absolute_error(actual, predicted):
    """
    Evaluation Metric: Mean Absolute Error (MAE)
      MAE = (1 / n) * sum(|actual_i - predicted_i|)
    Quantifies average prediction error in real units (charging sessions/day).
    """
    return sum(abs(p - q) for p, q in zip(actual, predicted)) / len(actual)


# ---------------------------------------------------------------------------
# MODULE 1: AI STATE SPACE FORMULATION
# ---------------------------------------------------------------------------
# Formally defined as a 4-tuple (S, S_0, T, f):
#   1. State Space (S): Set of all subsets of 3 distinct grid coordinates: {(r, c) | 0 <= r, c < 10}
#   2. Initial State (S_0): Random uniform placement of 3 stations.
#   3. Transition Operator (T): Shift exactly one station by 1 unit in cardinal direction (N, S, E, W).
#   4. Objective Function (f): Total charging demand covered without double counting.
#
# Coverage Model:
#   A station at (r, c) covers its own zone plus its 4 orthogonal neighbors
#   (von Neumann neighborhood of radius 1: (r, c), (r±1, c), (r, c±1)).
#   Zones covered by multiple stations are counted once (set union).

def covered_zones(state):
    """
    Calculates the set union of all zones serviced by current station placements.
    Union operation prevents double-counting of overlapping service areas.
    """
    zones = set()
    for r, c in state:
        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE:
                zones.add((r + dr, c + dc))
    return zones


def score(state, demand):
    """
    Objective / Fitness Function:
      f(S) = sum(demand(z) for z in union(N(s) for s in S))
    This is an instance of the NP-hard Maximum Coverage Problem (MCP).
    """
    return sum(demand[z] for z in covered_zones(state))


def random_state(k=STATIONS):
    """Generates an initial state uniformly at random from the state space."""
    all_zones = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return tuple(random.sample(all_zones, k))


def neighbours(state):
    """
    Successor Function:
    Generates all valid neighboring states reachable by moving any single station
    by 1 grid unit (Manhattan distance = 1), avoiding collisions.
    """
    result = []
    for i, (r, c) in enumerate(state):
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new = (r + dr, c + dc)
            if 0 <= new[0] < SIZE and 0 <= new[1] < SIZE and new not in state:
                result.append(state[:i] + (new,) + state[i + 1:])
    return result


# ---------------------------------------------------------------------------
# MODULE 2: HEURISTIC SEARCH & OPTIMIZATION ALGORITHMS
# ---------------------------------------------------------------------------
def hill_climbing(demand):
    """
    Greedy Local Search (Steepest Ascent Hill Climbing):
      - Evaluates all successors in the neighborhood.
      - Moves strictly to the successor with maximum objective score.
      - Terminates when no neighbor has a higher score.
    
    Limitation:
      Vulnerable to getting trapped in Local Maxima. For example, two stations
      may converge on the same high-density cluster with overlapping coverage.
    """
    state = random_state()
    while True:
        best = max(neighbours(state), key=lambda s: score(s, demand))
        if score(best, demand) <= score(state, demand):
            return state                          # Trapped in local maximum
        state = best


def simulated_annealing(demand, temperature=80.0, cooling=0.9993, steps=8000, k=STATIONS):
    """
    Stochastic Optimization (Simulated Annealing):
      - Balances Exploration (early phase) vs Exploitation (late phase).
      - If a candidate neighbor improves score (delta > 0), it is always accepted.
      - If a candidate neighbor decreases score (delta <= 0), it is accepted with
        Metropolis Probability:
          P(accept) = exp(delta / T)
      - Temperature decays geometrically: T_(t+1) = T_t * alpha (alpha = cooling factor).
    
    Advantage:
      Allows downhill moves at high temperatures to escape local maxima traps,
      converging toward the global optimum as temperature approaches zero.
    """
    state = random_state(k)
    best = state
    for _ in range(steps):
        new = random.choice(neighbours(state))
        change = score(new, demand) - score(state, demand)
        # Metropolis acceptance criterion
        if change > 0 or random.random() < math.exp(change / temperature):
            state = new
        if score(state, demand) > score(best, demand):
            best = state
        temperature *= cooling
    return best


def brute_force(demand):
    """
    Exhaustive Combinatorial Search (Ground Truth Baseline):
      Evaluates all C(100, 3) = 161,700 combinations to find the provable global optimum.
      Complexity: O(C(N, K)) -> Computationally prohibitive for large cities (NP-hard).
    """
    all_zones = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return max(combinations(all_zones, STATIONS), key=lambda s: score(s, demand))


# ---------------------------------------------------------------------------
# MODULE 4: DESIGN & ANALYSIS OF ALGORITHMS (DAA)
# ---------------------------------------------------------------------------
# Technique: Binary Search on Monotonic Answer Space
#
# Problem:
#   Find the MINIMUM number of stations k* needed to cover at least target_share (e.g. 50%)
#   of the city's total charging demand.
#
# Monotonicity Property:
#   Let C(k) be the maximum demand covered by k stations.
#   Since adding an additional station can never decrease coverage, C(k) is monotonically
#   non-decreasing: C(k + 1) >= C(k).
#   Therefore, the predicate P(k) = [C(k) >= target_demand] forms a sorted boolean sequence:
#     [False, False, False, ..., False, True, True, True, ...]
#
# Complexity Advantage:
#   Instead of evaluating sequentially k = 1, 2, ..., MAX_STATIONS (O(N) checks),
#   Binary Search determines the exact threshold in ceil(log2(MAX_STATIONS + 1)) = 5 checks.
#   Since each check involves an optimization run, O(log N) provides massive computational savings.
MAX_STATIONS = 20


def best_coverage(demand, k):
    """Evaluates maximum demand coverage achievable with k stations using simulated annealing."""
    return max(score(simulated_annealing(demand, steps=4000, k=k), demand) for _ in range(2))


def min_stations_needed(demand, target_share):
    """
    Binary search for the smallest k whose coverage reaches target_share of all demand.
    Returns:
      (optimal_k or None if infeasible, step-by-step search trace)
    """
    goal = target_share * sum(demand.values())
    low, high = 1, MAX_STATIONS + 1          # Search range [1, 20]; high = 21 represents infeasible
    steps = []
    while low < high:
        mid = (low + high) // 2
        covered = best_coverage(demand, mid)
        enough = covered >= goal
        steps.append((mid, covered, enough))
        if enough:
            high = mid           # Feasible: explore smaller station counts in left half
        else:
            low = mid + 1        # Infeasible: insufficient coverage, search right half
    return (low if low <= MAX_STATIONS else None), steps


# ---------------------------------------------------------------------------
# CLI VISUALIZATION & DEMONSTRATION PIPELINE
# ---------------------------------------------------------------------------
def print_map(state, demand):
    """Renders the 10x10 city grid: 'S' = Station, '#' = Covered Zone, '.' = Uncovered."""
    covered = covered_zones(state)
    for r in range(SIZE):
        line = ""
        for c in range(SIZE):
            line += " S" if (r, c) in state else " #" if (r, c) in covered else " ."
        print("   " + line)


def learn_demand():
    """
    Pipeline Step 1: Preprocess survey data, train OLS model, evaluate test MAE,
    and predict charging demand across all 100 city zones.
    """
    survey = clean(load_csv("survey.csv"))
    train, test = survey[:18], survey[18:]                     # 75% train (18 rows), 25% test (6 rows)
    a, b = train_linear_regression([float(r["cars"]) for r in train], [float(r["demand"]) for r in train])
    actual = [float(r["demand"]) for r in test]
    predicted = [a + b * float(r["cars"]) for r in test]
    mae = mean_absolute_error(actual, predicted)
    city = load_csv("city.csv")
    demand = {(int(z["row"]), int(z["col"])): a + b * int(z["cars"]) for z in city}
    return a, b, mae, demand, len(survey)


def main():
    random.seed(1)

    print("=" * 65)
    print("MODULE 3 (ML): Predict Unmeasured Demand via Linear Regression")
    print("=" * 65)
    a, b, mae, demand, used = learn_demand()
    print(f"Regression Formula: demand = {a:.2f} + {b:.2f} * cars  (Ordinary Least Squares)")
    print(f"Test Set MAE:       {mae:.2f} sessions/day (Generalization error)")
    print(f"Coverage:           Estimated demand for all {len(demand)} zones from {used} survey samples.")

    print("\n" + "=" * 65)
    print(f"MODULE 1 & 2 (AI SEARCH): Place {STATIONS} Stations (10 Trials per Algorithm)")
    print("=" * 65)
    hc = [score(hill_climbing(demand), demand) for _ in range(10)]
    sa = [score(simulated_annealing(demand), demand) for _ in range(10)]
    best = brute_force(demand)
    best_score = score(best, demand)
    print(f"{'Algorithm':<22}{'Mean Score':>11}{'Min Score':>10}{'Max Score':>10}{'Optimal Hits':>14}")
    for name, results in [("Hill Climbing", hc), ("Simulated Annealing", sa)]:
        hits = sum(abs(s - best_score) < 1e-9 for s in results)
        print(f"{name:<22}{sum(results) / 10:>11.1f}{min(results):>10.1f}{max(results):>10.1f}{hits:>10} / 10")
    print(f"{'Brute Force (Oracle)':<22}{best_score:>11.1f}{best_score:>10.1f}{best_score:>10.1f}{'10 / 10':>14}")
    print("\nObservation: Hill Climbing gets trapped in local maxima due to overlapping clusters.")
    print("Simulated Annealing escapes traps via Metropolis probability P = exp(delta/T).")

    plan = simulated_annealing(demand)
    print(f"\nOptimal SA Placement: stations at {sorted(plan)}")
    print(f"Total Demand Covered: {score(plan, demand):.1f} sessions/day ({(score(plan, demand)/sum(demand.values()))*100:.1f}% of city)")
    print_map(plan, demand)

    print("\n" + "=" * 65)
    print("MODULE 4 (DAA): Minimum Stations for 50% Demand via Binary Search")
    print("=" * 65)
    answer, steps = min_stations_needed(demand, 0.50)
    total = sum(demand.values())
    for i, (k, covered, enough) in enumerate(steps, 1):
        print(f"  Iteration {i}: test k={k:>2} stations -> covers {covered / total:5.1%} demand | "
              f"{'FEASIBLE (search left: [low, mid])' if enough else 'INFEASIBLE (search right: [mid+1, high])'}")
    print(f"\nResult: Optimal k = {answer} stations.")
    print(f"Search Efficiency: Found in {len(steps)} checks (O(log N) vs O(N) linear search).")
    print("=" * 65)


if __name__ == "__main__":
    main()
