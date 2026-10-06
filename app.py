"""
EV Charge Planner - Web Server
==============================
Exposes an interactive dashboard demonstrating:
  - Module 3: Linear Regression (OLS) demand prediction
  - Module 1 & 2: State-Space Heuristic Search (Hill Climbing vs Simulated Annealing)
  - Module 4: DAA Binary Search on Monotonic Answer Space

Run:
  python app.py
Open:
  http://localhost:8000
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Ensure the working directory is always the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402

# Train ML model once at server startup
A, B, MAE, DEMAND, USED = main.learn_demand()
TOTAL_DEMAND = round(sum(DEMAND.values()), 1)


def plan_info(stations):
    """Calculates coverage, score, and percentage for given station positions."""
    covered = main.covered_zones(stations)
    sc = round(main.score(stations, DEMAND), 1)
    pct = round((sc / TOTAL_DEMAND) * 100, 1)
    return {
        "stations": [list(s) for s in stations],
        "covered": [list(z) for z in covered],
        "score": sc,
        "total_demand": TOTAL_DEMAND,
        "percent": pct
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/":                                   # Dashboard web page
            with open("index.html", "rb") as f:
                self.send(f.read(), "text/html; charset=utf-8")

        elif url.path == "/model":                            # Module 3 (ML) Results
            grid = [[round(DEMAND[(r, c)], 1) for c in range(main.SIZE)] for r in range(main.SIZE)]
            self.send({
                "a": round(A, 2),
                "b": round(B, 2),
                "mae": round(MAE, 2),
                "used": USED,
                "demand": grid,
                "total_demand": TOTAL_DEMAND,
                "stations": main.STATIONS
            })

        elif url.path == "/run":                              # Module 2 Optimization run
            algo = query.get("algo", ["anneal"])[0]
            if algo == "hill":
                stations = main.hill_climbing(DEMAND)
            elif algo == "anneal":
                stations = main.simulated_annealing(DEMAND)
            else:
                stations = main.brute_force(DEMAND)
            self.send(plan_info(stations))

        elif url.path == "/needed":                           # Module 4 (DAA) Binary Search
            try:
                target = int(query.get("target", ["50"])[0])
                if not 1 <= target <= 99:
                    raise ValueError
            except ValueError:
                return self.send_error(400, "target must be an integer percentage from 1 to 99")

            answer, steps = main.min_stations_needed(DEMAND, target / 100)
            self.send({
                "answer": answer,
                "max": main.MAX_STATIONS,
                "steps": [
                    {
                        "stations": k,
                        "covered": round(c, 1),
                        "share": round(c / TOTAL_DEMAND * 100, 1),
                        "enough": e
                    }
                    for k, c, e in steps
                ]
            })

        elif url.path == "/score":                            # Custom manual placement score
            text = query.get("stations", [""])[0]             # e.g. "2,2;7,6;2,8"
            try:
                stations = tuple(tuple(int(v) for v in p.split(",")) for p in text.split(";") if p)
                if any(len(s) != 2 or not (0 <= s[0] < main.SIZE and 0 <= s[1] < main.SIZE) for s in stations):
                    raise ValueError
            except ValueError:
                return self.send_error(400, "stations coordinates must be in format row,col;row,col")
            self.send(plan_info(stations))

        else:
            self.send_error(404)

    def log_message(self, *args):                             # Suppress noisy HTTP request logs in terminal
        pass


if __name__ == "__main__":
    port = 8000
    print("=" * 65)
    print("EV Charge Planner Academic Dashboard")
    print("=" * 65)
    print(f"Server active at: http://localhost:{port}")
    print("Press Ctrl+C to terminate the server.\n")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
