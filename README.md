# ⚡ EV Charge Planner (AI/ML + DAA mini project)

## Topic
Choosing where a city should build electric-vehicle (EV) charging stations.

## Problem statement
A city can build only **3 charging stations** and wants them where they will serve the most people.
But charging demand was measured in only **25 of the city's 100 zones**. We need to:
1. **predict** the demand in every zone, and
2. **find** the 3 locations that cover the most demand.

## Solution (3 steps, one concept from each module)

| Step | Module | What we do |
|---|---|---|
| 1 | **Module 3: ML** | Clean the survey data (remove a missing value), then train **simple linear regression**, `demand = a + b × cars`, and test it on unseen rows |
| 2 | **Module 1: State space** | **State** = positions of the 3 stations · **Move** = shift one station by one zone · **Score** = total demand covered |
| 3 | **Module 2: Optimization** | **Hill climbing** (always take the best move) vs **simulated annealing** (sometimes take a worse move to escape local maxima) |
| 4 | **DAA: Binary search on the answer** | Find the **smallest number of stations** that covers a target share of demand (e.g. 50%) |

A station covers its own zone plus the 4 zones next to it.

## Results (from `python main.py`)
- **Model:** `demand = 3.29 + 0.41 × cars`, with a test error (MAE) of **0.98** sessions/day.
- **Optimization,** 10 runs each, compared with the true best plan (found by checking all 161,700 plans):

| Algorithm | Average score | Found the best plan |
|---|---|---|
| Hill climbing | 354.6 | 4 / 10 |
| **Simulated annealing** | **392.3** | **9 / 10** |

Hill climbing often gets **stuck** with two stations in the same busy area (a local maximum). Simulated
annealing accepts some worse moves early on and escapes that trap. Over 60 test runs it found the best plan
**97%** of the time (hill climbing: about 15%), placing one station in each of the city's three EV hotspots:
```
    . . . . . . . . . .
    . . # . . . . . # .
    . # S # . . . # S #        S = station
    . . # . . . . . # .        # = zone covered
    . . . . . . . . . .        . = not covered
    . . . . . . . . . .
    . . . . . . # . . .
    . . . . . # S # . .
    . . . . . . # . . .
    . . . . . . . . . .
```

## DAA part: binary search on the answer
**Question:** what is the smallest number of stations that covers 50% of the city's demand?

**Why binary search works:** adding a station can never cover *less* demand, so the answers are sorted:
`1 station → no, 2 → no, … 6 → no, 7 → yes, 8 → yes, …`. We look for the first "yes" by checking the middle
and throwing away half of the range each time, the same idea as the *Koko Eating Bananas* problem.

```
check 1: 11 stations cover 66.1%  ->  enough, try fewer
check 2:  6 stations cover 47.0%  ->  not enough, try more
check 3:  9 stations cover 60.3%  ->  enough, try fewer
check 4:  8 stations cover 56.9%  ->  enough, try fewer
check 5:  7 stations cover 50.5%  ->  enough, try fewer
Answer: 7 stations
```

**Complexity:** searching 1–20 stations takes at most ⌈log₂ 21⌉ = **5 checks** instead of up to 20 with a
linear search: **O(log n)** instead of **O(n)**. Each check runs the simulated-annealing planner, so saving
checks saves real time.

*Note:* each check uses the AI planner, which finds the best plan almost always but not with a mathematical
guarantee. For very large numbers of stations the coverage can come out a fraction of a percent lower than
it should, so the answer is a very good estimate rather than a proof.

## How to run
You only need Python 3, with no extra libraries.

**Website:**
```bash
python app.py
```
Then open **http://localhost:8000** in your browser. Click the buttons to run each algorithm, click the map
to place your own stations and compare your score, or use **Step 3** to binary-search how many stations are needed.

**Terminal only:**
```bash
python main.py
```

## Files
| File | What it is |
|---|---|
| `main.py` | All the AI/ML code (about 160 lines); also runs in the terminal |
| `app.py` | Small web server that shows `main.py`'s results as a website |
| `index.html` | The web page |
| `survey.csv` | The survey: number of EVs and the measured demand, for 25 zones |
| `city.csv` | Number of EVs in each of the 100 zones |
| `run_output.txt` | Saved output of a run |
