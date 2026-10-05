"""
The website for EV Charge Planner.

Run:   python app.py
Open:  http://localhost:8000

All the AI work is done by the functions in main.py. This file only
sends their results to the web page (index.html).
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # so the CSV files are found from anywhere
import main                                            # noqa: E402

A, B, MAE, DEMAND, USED = main.learn_demand()


def plan_info(stations):
    """Score and covered zones for a list of station positions."""
    return {"stations": [list(s) for s in stations],
            "covered": [list(z) for z in main.covered_zones(stations)],
            "score": round(main.score(stations, DEMAND), 1)}


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

        if url.path == "/":                                   # the web page
            with open("index.html", "rb") as f:
                self.send(f.read(), "text/html; charset=utf-8")

        elif url.path == "/model":                            # Step 1 results
            grid = [[round(DEMAND[(r, c)], 1) for c in range(main.SIZE)] for r in range(main.SIZE)]
            self.send({"a": A, "b": B, "mae": MAE, "used": USED, "demand": grid,
                       "stations": main.STATIONS})

        elif url.path == "/run":                              # Step 3: run an algorithm
            algo = query.get("algo", ["hill"])[0]
            if algo == "hill":
                stations = main.hill_climbing(DEMAND)
            elif algo == "anneal":
                stations = main.simulated_annealing(DEMAND)
            else:
                stations = main.brute_force(DEMAND)
            self.send(plan_info(stations))

        elif url.path == "/needed":                           # Step 4 (DAA): binary search
            try:
                target = int(query.get("target", ["50"])[0])
                if not 1 <= target <= 99:
                    raise ValueError
            except ValueError:
                return self.send_error(400, "target must be a percentage from 1 to 99")
            total = sum(DEMAND.values())
            answer, steps = main.min_stations_needed(DEMAND, target / 100)
            self.send({"answer": answer, "max": main.MAX_STATIONS,
                       "steps": [{"stations": k, "share": round(c / total * 100, 1), "enough": e}
                                 for k, c, e in steps]})

        elif url.path == "/score":                            # score the user's own plan
            text = query.get("stations", [""])[0]             # e.g. "2,2;7,6;2,8"
            try:
                stations = tuple(tuple(int(v) for v in p.split(",")) for p in text.split(";") if p)
                if any(len(s) != 2 or not (0 <= s[0] < main.SIZE and 0 <= s[1] < main.SIZE) for s in stations):
                    raise ValueError
            except ValueError:
                return self.send_error(400, "stations must look like 2,2;7,6")
            self.send(plan_info(stations))

        else:
            self.send_error(404)

    def log_message(self, *args):                             # keep the terminal quiet
        pass


if __name__ == "__main__":
    print("EV Charge Planner website is running.")
    print("Open http://localhost:8000 in your browser.  (Press Ctrl+C to stop.)")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
