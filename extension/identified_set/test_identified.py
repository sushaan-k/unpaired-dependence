import unittest

import numpy as np

from identified import (SUPPORT, StarQuery, cell_from_log_odds, certify, first_order_bruteforce,
                        first_order_vertices, independence_start, kernel, log_odds, minimax,
                        scale, star_law, table)
from run_identified import load_recipients


def kl(p, q):
    return float(np.sum(np.where(p > 0, p * np.log(p / q), 0.0)))


class IdentifiedSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_recipients()
        cls.B = cls.data["interaction"]
        cls.K = kernel(cls.B)
        cls.mu = cls.data["rna_means"][95]
        cls.nu = cls.data["protein_means"][95]

    def test_star_law_has_exact_means_and_derivative(self):
        rng = np.random.default_rng(1)
        for centre in (0, 4, 8):
            s = rng.uniform(0.05, 0.95, 8)
            law, derivative = star_law(self.mu, centre, s)
            self.assertAlmostEqual(law.sum(), 1.0, places=14)
            np.testing.assert_allclose(law @ SUPPORT, self.mu, atol=1e-14, rtol=0)
            for k in range(8):
                e = np.zeros(8); e[k] = 1e-7
                numeric = (star_law(self.mu, centre, s + e)[0] - star_law(self.mu, centre, s - e)[0]) / 2e-7
                np.testing.assert_allclose(numeric, derivative[:, k], atol=1e-8, rtol=0)
        independent = star_law(self.mu, 3, independence_start(self.mu, 3))[0]
        product = np.prod(np.where(SUPPORT == 1, self.mu, 1 - self.mu), axis=1)
        np.testing.assert_allclose(independent, product, atol=1e-15, rtol=0)

    def test_first_order_range_matches_enumeration(self):
        rng = np.random.default_rng(2)
        for _ in range(2):
            B = rng.normal(size=(9, 9))
            mu, nu = rng.uniform(0.05, 0.95, 9), rng.uniform(0.05, 0.95, 9)
            i, j = rng.integers(9, size=2)
            fast = first_order_vertices(B, mu, nu, i, j)[0]
            np.testing.assert_allclose(fast, first_order_bruteforce(B, mu, nu, i, j), atol=1e-13)

    def test_star_gradient_matches_finite_differences(self):
        problem = StarQuery(self.K, self.mu, self.nu, 1, 8)
        z = np.random.default_rng(3).uniform(0.1, 0.9, 16)
        _, gradient = problem.value_and_gradient(z)
        numeric = []
        for k in range(16):
            e = np.zeros(16); e[k] = 1e-6
            numeric.append((problem.value(z + e)[0] - problem.value(z - e)[0]) / 2e-6)
        np.testing.assert_allclose(numeric, gradient, atol=1e-9, rtol=0)

    def test_minimax_equalizes_and_minimizes_worst_case(self):
        m, n, low, high = 0.3, 0.45, 0.02, 0.25
        theta, risk = minimax(low, high, m, n)
        star = table(cell_from_log_odds(theta, m, n), m, n)
        self.assertAlmostEqual(kl(table(low, m, n), star), kl(table(high, m, n), star), places=12)
        self.assertAlmostEqual(risk, kl(table(low, m, n), star), places=12)
        grid = np.linspace(-6, 6, 2001)
        worst = [max(kl(table(low, m, n), table(cell_from_log_odds(g, m, n), m, n)),
                     kl(table(high, m, n), table(cell_from_log_odds(g, m, n), m, n))) for g in grid]
        self.assertGreaterEqual(min(worst), risk - 1e-12)

    def test_log_odds_inverse(self):
        for theta in (-7.0, -1.3, 0.0, 0.4, 5.5):
            t = cell_from_log_odds(theta, 0.2, 0.7)
            self.assertAlmostEqual(float(log_odds(t, 0.2, 0.7)), theta, places=9)

    def test_scaling_reproduces_released_full_pattern_prediction(self):
        import sys
        from run_identified import RELEASE
        colon = np.load(RELEASE / "colon_generalization/results/predictions.npz")
        r, c = colon["marginals"][0]
        Q, _, _ = scale(self.K, r, c)
        cells = np.einsum("ui,uv,vj->ij", SUPPORT, Q, SUPPORT)
        np.testing.assert_allclose(cells.ravel(), colon["predictions"][0, 3, :, 1, 1], atol=1e-10, rtol=0)

    def test_released_arms_are_attained_points_of_the_same_interval(self):
        predictions = self.data["predictions"]
        m = np.repeat(self.data["rna_means"], 9, axis=1)
        n = np.tile(self.data["protein_means"], (1, 9))
        for arm in (0, 3):
            rows = predictions[:, arm, :, 1, 0] + predictions[:, arm, :, 1, 1]
            cols = predictions[:, arm, :, 0, 1] + predictions[:, arm, :, 1, 1]
            np.testing.assert_allclose(rows, m, atol=1e-9, rtol=0)
            np.testing.assert_allclose(cols, n, atol=1e-9, rtol=0)
            self.assertGreater(predictions[:, arm].min(), 0)

    def test_certified_value_is_a_feasible_query_cell(self):
        z = np.full(16, 0.3)
        t = certify(self.K, self.mu, self.nu, 2, 5, z)
        m, n = self.mu[2], self.nu[5]
        self.assertTrue(max(0.0, m + n - 1) < t < min(m, n))
        self.assertAlmostEqual(t, StarQuery(self.K, self.mu, self.nu, 2, 5).value(z, tol=1e-13)[0], places=11)

    def test_scaling_failure_is_reported(self):
        from identified import ScalingError
        r = np.full(512, 1 / 512)
        with self.assertRaises(ScalingError):
            scale(self.K, r, r, tol=1e-30, maxit=10)

    def test_free_search_gradient(self):
        from sensitivity import FreeQuery
        rng = np.random.default_rng(4)
        r0 = star_law(self.mu, 0, rng.uniform(.2, .8, 8))[0]
        c0 = star_law(self.nu, 0, rng.uniform(.2, .8, 8))[0]
        w = np.concatenate([np.log(r0), np.log(c0)])
        problem = FreeQuery(self.K, self.mu, self.nu, 0, 0)
        _, gradient = problem.evaluate(w)
        direction = rng.normal(size=1024)
        h = 1e-6
        plus = FreeQuery(self.K, self.mu, self.nu, 0, 0).evaluate(w + h * direction)[0]
        minus = FreeQuery(self.K, self.mu, self.nu, 0, 0).evaluate(w - h * direction)[0]
        self.assertAlmostEqual((plus - minus) / (2 * h), gradient @ direction, places=8)


if __name__ == "__main__":
    unittest.main()
