# EV Charging Station Placement Planner

**Author:** Jagadeesh  
**Repository:** [github.com/jagadeesh177/ev-charge-planner](https://github.com/jagadeesh177/ev-charge-planner)

A computational tool to determine the optimal spatial placement of electric-vehicle (EV) charging stations across a city grid. The project combines supervised regression to estimate unmeasured charging demand, local search optimization to select station locations under budget constraints, and binary search to size the minimum number of stations needed to hit target coverage.

Implemented entirely in standard Python with zero external libraries.

---

## Problem Description

A city is divided into a $10 \times 10$ spatial grid (100 zones total). The municipality has a budget to deploy **3 high-capacity charging stations** and wants them placed where they will serve the maximum amount of charging demand.

- **Sparse Survey Data:** Daily charging demand was measured in only 24 out of the 100 zones.
- **Service Range:** Each station covers its home zone and its 4 adjacent neighbors (North, South, East, West).
- **Submodular Coverage:** Zones covered by multiple stations only count once (no double counting).

---

## Methodology

### 1. Demand Estimation (Linear Regression)
We fit a simple linear regression model relating vehicle counts to measured daily demand:

$$\text{demand} = a + b \times \text{cars}$$

The coefficients are computed using standard Ordinary Least Squares (OLS) closed-form equations:
- **Learned Model:** $\text{demand} = 3.29 + 0.41 \times \text{cars}$
- **Validation:** Mean Absolute Error (MAE) of **0.98 sessions/day** on unseen test data.
- The model is applied across all 100 city zones to predict full-city demand.

### 2. Location Optimization (Local Search)
Station placement is modeled as a state-space search where a state is a 3-tuple of zone coordinates. We compare two optimization algorithms:

- **Hill Climbing:** Greedy steepest-ascent search. Often gets trapped in local maxima when two stations cluster in the same busy zone.
- **Simulated Annealing:** Probabilistic local search using the Metropolis acceptance criterion:
  $$P(\text{accept}) = \exp\left(\frac{\Delta E}{T}\right)$$
  Accepts occasional downhill moves at high temperatures, allowing it to escape local traps and distribute stations across separated demand hotspots.
- **Brute Force (Exact Baseline):** Evaluates all $\binom{100}{3} = 161,700$ possible station combinations to verify the true global optimum.

### 3. Capacity Sizing (Binary Search on Answer Space)
To find the minimum number of stations required to cover a target share (e.g., 50%) of total city demand:
- Because adding stations can never decrease total demand covered, coverage is monotonically non-decreasing.
- Binary search over the range of $[1, 20]$ stations finds the exact minimum in at most **5 evaluations ($O(\log N)$)** instead of testing each number sequentially ($O(N)$).

---

## Experimental Results

Results over 10 independent runs of each algorithm:

| Algorithm | Average Score | Minimum | Maximum | Optimal Plans Found |
| :--- | :---: | :---: | :---: | :---: |
| **Hill Climbing** | 354.6 | 291.7 | 394.4 | 4 / 10 |
| **Simulated Annealing** | **392.3** | **373.0** | **394.4** | **9 / 10** |
| **Brute Force (Exact)** | 394.4 | 394.4 | 394.4 | 10 / 10 |

The global optimum score is **394.4 charging sessions/day** (covering 29.5% of total city demand), placing one station in each of the three major demand hotspots: `(2, 2)`, `(2, 8)`, and `(7, 6)`.

---

## How to Run

### Requirements
- Python 3.8 or higher.
- Built-in libraries only (`csv`, `math`, `random`, `itertools`, `http.server`). No pip packages required.

### 1. Interactive Web Dashboard
```bash
python app.py
```
Open **http://localhost:8000** in your browser to interact with the city map, run optimization algorithms, and test manual station placements.

### 2. Terminal Mode
```bash
python main.py
```
Runs the full pipeline in the terminal and outputs the regression model, algorithm comparison table, and binary search traces.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `main.py` | Core implementation: OLS regression, local search algorithms, and binary search. |
| `app.py` | Local HTTP web server for the interactive dashboard. |
| `index.html` | Frontend interface with city map, real-time score calculation, and controls. |
| `city.csv` | Vehicle count data for each of the 100 city zones. |
| `survey.csv` | Survey sample data with vehicle counts and measured daily demand. |
| `run_output.txt` | Sample terminal execution log from `main.py`. |
| `PROJECT_REPORT.md` | Detailed project report and methodology documentation. |
