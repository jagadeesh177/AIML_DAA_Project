# Project Report: EV Charging Station Placement Planner

**Author:** Jagadeesh  
**Project:** EV Charging Station Placement Planner  

---

## 1. Introduction

With the growing adoption of electric vehicles (EVs), municipal planners face the challenge of placing limited charging infrastructure where it maximizes public benefit. Because comprehensive charging telemetry is often unavailable for every city block, planners must:
1. Predict charging demand from available indicators (such as registered vehicle counts).
2. Determine optimal station coordinates under budget constraints without wasteful coverage overlap.
3. Determine the minimum number of charging stations required to meet specific municipal coverage targets.

This project implements an end-to-end computational solution to this problem without relying on external machine learning or optimization libraries.

---

## 2. Problem Statement

A city is modeled as a $10 \times 10$ spatial grid containing 100 discrete zones. The municipality has a budget to build $K = 3$ charging stations.

- **Survey Data:** Historical daily charging demand was measured in only 24 out of 100 zones.
- **Service Coverage:** A charging station placed at $(r, c)$ serves its home zone plus its four orthogonal adjacent neighbors (von Neumann neighborhood of radius 1).
- **Objective:** Maximize the total daily demand served by the stations. If a zone falls within the coverage area of multiple stations, its demand is counted only once (set union).

---

## 3. Methodology

### 3.1 Demand Estimation (Linear Regression)
To estimate demand across the unmeasured zones:
1. Survey data is loaded from `survey.csv` and cleaned by filtering out missing records (complete-case analysis).
2. A simple linear regression model is fit between vehicle count ($x$) and measured demand ($y$):
   $$\text{demand} = a + b \times \text{cars}$$
   The parameters are calculated using standard Ordinary Least Squares (OLS) equations:
   $$b = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad a = \bar{y} - b\bar{x}$$
3. The dataset is split into training (18 samples) and testing (6 samples) to evaluate generalization via Mean Absolute Error (MAE):
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
4. The learned parameters are applied to `city.csv` to compute predicted demand across all 100 zones.

### 3.2 State Space Formulation
The station placement task is modeled as a state-space search:
- **State ($S$):** A 3-tuple of distinct grid coordinates, e.g., $((2, 2), (2, 8), (7, 6))$.
- **Initial State:** A uniform random selection of 3 zones.
- **Successor Function:** Shifting one station by one step in any cardinal direction (up, down, left, right) while remaining on the grid.
- **Objective Function:**
  $$f(S) = \sum_{z \in \bigcup_{s \in S} N(s)} D(z)$$
  where $N(s)$ is the 5-zone neighborhood around station $s$ and $D(z)$ is the predicted demand in zone $z$.

### 3.3 Heuristic Optimization Algorithms
Two local search strategies are implemented and compared:

1. **Steepest-Ascent Hill Climbing:**
   - Evaluates all valid neighboring states and transitions to the state with the highest coverage score.
   - Terminates when no neighbor improves upon the current score.
   - Vulnerable to local maxima: when two stations start near the same busy cluster, moving one away temporarily decreases the score, trapping the algorithm in a suboptimal state.

2. **Simulated Annealing:**
   - Evaluates a randomly chosen neighbor at each iteration.
   - Always accepts moves that increase coverage ($\Delta E > 0$).
   - Accepts moves that decrease coverage ($\Delta E \le 0$) with probability:
     $$P = \exp\left(\frac{\Delta E}{T}\right)$$
   - The temperature $T$ cools gradually at each step ($T \leftarrow \alpha T$, $\alpha = 0.9993$, 8000 steps).
   - This probabilistic acceptance enables the search to escape local maxima early on and converge to the global optimum as the temperature drops.

3. **Brute Force Verification:**
   - Evaluates all $\binom{100}{3} = 161,700$ possible station combinations to identify the exact theoretical global maximum for benchmark comparison.

### 3.4 Minimum Station Sizing (Binary Search)
Municipalities often need to know the minimum number of stations required to cover a target percentage of total demand (e.g., 50%).

- **Monotonicity Property:** Adding an additional station can never decrease the total demand covered. Therefore, the coverage function $C(k)$ is monotonically non-decreasing with respect to station count $k$.
- **Binary Search Application:** Because the feasibility predicate $[C(k) \ge \text{Target}]$ is sorted (False, ..., False, True, ..., True), binary search finds the optimal station count in $\lceil \log_2(21) \rceil = 5$ evaluations over the range $[1, 20]$, compared to 20 evaluations required by linear search.

---

## 4. Experimental Results

### 4.1 Regression Performance
- **Fitted Model:** $\text{demand} = 3.29 + 0.41 \times \text{cars}$
- **Test Set MAE:** $0.98$ charging sessions/day
- **Total City Demand:** $1338.4$ charging sessions/day across 100 zones

### 4.2 Algorithm Comparison (10 Runs Each)

| Algorithm | Mean Score | Min Score | Max Score | Optimal Hits |
| :--- | :---: | :---: | :---: | :---: |
| **Hill Climbing** | 354.6 | 291.7 | 394.4 | 4 / 10 |
| **Simulated Annealing** | 392.3 | 373.0 | 394.4 | 9 / 10 |
| **Brute Force (Exact)** | 394.4 | 394.4 | 394.4 | 10 / 10 |

- The global optimum score is **394.4 sessions/day** (covering 29.5% of total city demand).
- Optimal station coordinates: `(2, 2)`, `(2, 8)`, and `(7, 6)`.
- Simulated Annealing reliably distributed stations across all three city demand hotspots, whereas Hill Climbing frequently trapped two stations in a single hotspot.

### 4.3 Capacity Sizing via Binary Search
Target: Cover at least 50% of total city demand (669.2 sessions/day).

| Check | Tested Stations ($k$) | Demand Covered | Share | Verdict |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 11 | 923.8 | 69.0% | Enough -> search lower range |
| 2 | 6 | 628.8 | 47.0% | Not enough -> search higher range |
| 3 | 9 | 798.1 | 59.6% | Enough -> search lower range |
| 4 | 8 | 758.6 | 56.7% | Enough -> search lower range |
| 5 | 7 | 696.0 | 52.0% | Enough -> search lower range |

**Result:** Exactly **7 stations** are required to achieve 50% coverage.

---

## 5. Conclusion

This project demonstrates the effective combination of regression modeling, local search optimization, and binary search for spatial facility location. By implementing Ordinary Least Squares and Simulated Annealing from mathematical foundations in pure Python, the implementation remains lightweight, reproducible, and completely free of third-party dependencies.
