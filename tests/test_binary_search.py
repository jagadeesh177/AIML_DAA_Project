"""DAA tests for binary search on the answer. Run with:  python -m unittest discover tests"""
import contextlib
import io
import math
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main                                          # noqa: E402
from daa import linear_search_min                    # noqa: E402
from main import binary_search_min                   # noqa: E402


def check_list(values, target):
    """check(k) for a sorted list: is the k-th value (1-based) at least the target?"""
    return lambda k: (values[k - 1], values[k - 1] >= target)


class TestBinarySearch(unittest.TestCase):
    def test_answer_in_the_middle(self):
        values = [12, 22, 30, 36, 41, 47, 52, 57, 61, 64]
        answer, steps = binary_search_min(check_list(values, 50), 1, len(values))
        self.assertEqual(answer, 7)
        self.assertEqual([s[2] for s in steps], [6, 9, 8, 7])

    def test_first_value_already_enough(self):
        answer, _ = binary_search_min(check_list([12, 22, 30], 5), 1, 3)
        self.assertEqual(answer, 1)

    def test_only_the_last_value_is_enough(self):
        answer, _ = binary_search_min(check_list([12, 22, 30], 30), 1, 3)
        self.assertEqual(answer, 3)

    def test_not_possible(self):
        values = list(range(1, 21))
        answer, steps = binary_search_min(check_list(values, 99), 1, 20)
        self.assertIsNone(answer)
        self.assertEqual([s[2] for s in steps], [11, 16, 19, 20])     # every check says "not enough"
        self.assertLessEqual(len(steps), math.ceil(math.log2(21)))

    def test_single_value(self):
        self.assertEqual(binary_search_min(check_list([7], 7), 1, 1)[0], 1)
        self.assertIsNone(binary_search_min(check_list([7], 8), 1, 1)[0])

    def test_repeated_values_return_the_first(self):
        values = [10, 20, 20, 20, 20, 30]
        self.assertEqual(binary_search_min(check_list(values, 20), 1, len(values))[0], 2)

    def test_matches_linear_search_and_log_bound(self):
        rng = random.Random(5)
        for _ in range(300):
            n = rng.randint(1, 64)
            values = sorted(rng.randint(0, 100) for _ in range(n))
            for target in range(-1, 102, 7):
                answer, steps = binary_search_min(check_list(values, target), 1, n)
                self.assertEqual(answer, linear_search_min(check_list(values, target), 1, n))
                self.assertLessEqual(len(steps), math.ceil(math.log2(n + 1)))


class TestStations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.demand = main.learn_demand()[3]

    def test_fifty_percent_needs_seven_stations(self):
        random.seed(7)
        answer, steps = main.min_stations_needed(self.demand, 0.50)
        self.assertEqual(answer, 7)
        self.assertLessEqual(len(steps), 5)

    def test_whole_city_is_not_possible(self):
        random.seed(7)
        answer, _ = main.min_stations_needed(self.demand, 1.00)
        self.assertIsNone(answer)

    def test_same_answer_as_before_the_refactor(self):
        # the original loop, kept here to show that binary_search_min did not change the behaviour
        def original(demand, share):
            goal, low, high = share * sum(demand.values()), 1, main.MAX_STATIONS + 1
            while low < high:
                mid = (low + high) // 2
                if main.best_coverage(demand, mid) >= goal:
                    high = mid
                else:
                    low = mid + 1
            return low if low <= main.MAX_STATIONS else None
        for share in (0.3, 0.5):
            random.seed(3)
            new = main.min_stations_needed(self.demand, share)[0]
            random.seed(3)
            self.assertEqual(new, original(self.demand, share))


if __name__ == "__main__":
    unittest.main()
