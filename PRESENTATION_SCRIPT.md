# 🗣️ EV Charge Planner — Faculty Presentation Script

Use this script during your lab evaluation, project review, or viva presentation. It tells you **exactly what to say** and **what actions to perform on screen**.

---

## ⏱️ Version 1: The 60-Second Elevator Pitch
*(Use when the professor says: "Quickly tell me what you did.")*

> "Good morning, professors. Our project solves a real-world municipal infrastructure problem: **Where should a city place 3 EV charging stations to maximize demand coverage under a limited budget?**
>
> We integrated four academic modules into an end-to-end pipeline:
> 1. **Machine Learning (Module 3):** We trained a closed-form Ordinary Least Squares linear regression model on a 24-zone survey to predict charging demand for all 100 city zones with an MAE of 0.98.
> 2. **AI State Space & Optimization (Modules 1 & 2):** We formulated the problem as a state-space search and proved why greedy **Hill Climbing gets trapped in local maxima (only 40% success rate)**, while **Simulated Annealing achieves a 90% success rate** using the Metropolis acceptance criterion.
> 3. **DAA (Binary Search):** We proved that coverage is monotonic and applied **Binary Search on the Answer Space** to find the minimum number of stations needed for 50% coverage in **$O(\log N)$ checks (just 5 iterations)** instead of 20 linear evaluations.
>
> All algorithms were implemented natively in Python from mathematical first principles with zero black-box libraries."

---

## ⏱️ Version 2: The 3-Minute Live Project Demo
*(Use when presenting your screen or projector with `python app.py` running at `http://localhost:8000`)*

### Step 1: Open the Dashboard (0:00 - 0:45)
- **Action:** Open `http://localhost:8000`. Point to the header tags and map.
- **Say:**
  > "Here is our interactive dashboard. The city is represented as a $10 \times 10$ spatial grid of 100 zones.
  > Darker blue squares indicate higher predicted charging demand.
  > In **Step 1**, we used Ordinary Least Squares regression to predict unmeasured demand:
  > $\text{demand} = 3.29 + 0.41 \times \text{cars}$.
  > On our unseen test partition, the model achieves a Mean Absolute Error of 0.98 sessions/day."

### Step 2: Compare Optimization Algorithms (0:45 - 2:00)
- **Action:** Click the **⛰ Hill Climbing** button once or twice.
- **Say:**
  > "Now, in **Step 2**, we solve the NP-hard Maximum Coverage Problem.
  > Notice when I run **Hill Climbing**, it frequently gets trapped around a score of 354 or lower.
  > That's because Hill Climbing is greedy: two stations clump together in the same high-demand cluster. Shifting one station across the low-demand valley temporarily drops the score, so greedy search refuses to make the move."
- **Action:** Click the **🔥 Simulated Annealing** button.
- **Say:**
  > "Now watch when I run **Simulated Annealing**. It consistently reaches the global optimum of **394.4 sessions/day**, placing one station in each of the three distinct city hotspots.
  > It succeeds because early in the search, when temperature $T$ is high, it probabilistically accepts downhill moves using the **Metropolis criterion $P = e^{\Delta E / T}$**, jumping out of local traps."
- **Action:** Click **🎯 Brute Force Benchmark**.
- **Say:**
  > "To verify our results, we ran an exhaustive search checking all $\binom{100}{3} = 161,700$ possible plans. Brute force confirms that 394.4 is the true mathematical global maximum."

### Step 3: Demonstrate DAA Binary Search (2:00 - 3:00)
- **Action:** Scroll to **Step 3**, set slider to 50%, and click **🔍 Binary Search Minimum Stations**.
- **Say:**
  > "Finally, in our **DAA module**, we ask: *What is the minimum number of stations needed to cover 50% of the city's demand?*
  > Because adding stations cannot decrease coverage, the feasibility predicate is monotonic.
  > Therefore, we can binary search over the range of 1 to 20 stations.
  > Instead of running 20 sequential optimization passes in $O(N)$, binary search finds the exact threshold of **7 stations in just 5 iterations ($O(\log N)$)**.
  > Each check evaluates simulated annealing, so binary search saves over 70% of CPU execution time."

---

## ⏱️ Version 3: The 5-Minute In-Depth Academic Defense
*(Use if faculty asks detailed theoretical questions during viva)*

### 1. Mathematical Rigor & No External Libraries
> "A key feature of our project is that we intentionally did not import `scikit-learn`, `numpy`, or `pandas`.
> We derived the closed-form normal equations:
> $$\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}$$
> This guarantees complete visibility into data cleaning, training, and inference without hidden black boxes."

### 2. State Space & Neighborhood Topology
> "Our state space is formally defined as a 4-tuple $(S, S_0, T, f)$:
> - $S$: All 3-station subsets in $\{0,\dots,9\} \times \{0,\dots,9\}$.
> - $S_0$: Uniform random initial coordinate assignment.
> - $T$: 1-step cardinal perturbation (Manhattan distance = 1).
> - $f$: Submodular set-union objective: $f(S) = \sum_{z \in \bigcup N(s)} D(z)$ where $N(s)$ is the von Neumann neighborhood of radius 1."

### 3. Proof of Monotonicity for DAA Binary Search
> "Binary search on answer space requires proving monotonicity:
> Let $C(k) = \max_{|S|=k} f(S)$.
> Any optimal $k$-station configuration can be augmented with an additional station: $S_{k+1} = S_k \cup \{s^*\}$.
> Since all zone demands $D(z) \ge 0$, $\bigcup_{s \in S_{k+1}} N(s) \supseteq \bigcup_{s \in S_k} N(s)$, guaranteeing $C(k+1) \ge C(k)$.
> Thus the decision boolean array $[C(k) \ge \text{Target}]$ is sorted:
> $[\text{False}, \dots, \text{False}, \text{True}, \dots, \text{True}]$.
> This mathematically guarantees that binary search will converge to the first true value in $\lceil \log_2(21) \rceil = 5$ checks."

---

## 💡 Quick Tips for Impressing Your Faculty
1. **Click the `🎓 Faculty / Viva Guide` button on screen**:
   Show them that the syllabus mapping and mathematical equations are documented right inside the live web app.
2. **Point out the local maximum on the map**:
   When Hill Climbing scores 291.7 or 354.6, point to the screen and say: *"See how two stations are overlapping around row 2? That's the local trap."*
3. **Run `python main.py` in the terminal**:
   If the professor prefers the command line, `python main.py` prints the exact same structured academic report in clean ASCII format.
