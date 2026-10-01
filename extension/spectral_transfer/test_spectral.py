import unittest

import numpy as np

from gaussian_transfer import Marginal, first_order, fit_interaction, joint_precision, profile_loglik, transfer


def random_cov(rng, d):
    A = rng.normal(size=(d, 2 * d))
    return A @ A.T / (2 * d) + 0.1 * np.eye(d)


class ClosedFormTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.p, self.q = 7, 5
        self.Sx, self.Sy = random_cov(self.rng, self.p), random_cov(self.rng, self.q)
        self.B = self.rng.normal(size=(self.p, self.q)) * 0.8

    def test_precision_has_block_minus_B_and_is_valid(self):
        C, gamma = transfer(Marginal(self.Sx, self.Sy), self.B)
        L = joint_precision(self.Sx, C, self.Sy)
        np.testing.assert_allclose(L[: self.p, self.p:], -self.B, atol=1e-12)
        self.assertTrue(np.all(gamma < 1) and np.all(gamma >= 0))
        self.assertGreater(np.linalg.eigvalsh(np.block([[self.Sx, C], [C.T, self.Sy]])).min(), 0)

    def test_first_and_third_order_terms(self):
        m = Marginal(self.Sx, self.Sy)
        for t in (1e-2, 5e-3):
            C = transfer(m, t * self.B)[0]
            cubic = self.Sx @ self.B @ self.Sy @ self.B.T @ self.Sx @ self.B @ self.Sy
            residual = C - t * first_order(m, self.B) + t ** 3 * cubic
            self.assertLess(np.abs(residual).max(), 50 * t ** 5 * np.abs(cubic).max() + 1e-14)

    def test_profile_likelihood_gradient_value_and_concavity(self):
        rng = self.rng
        people = [(100 + 50 * k, Marginal(random_cov(rng, self.p), random_cov(rng, self.q)),
                   rng.normal(size=(self.p, self.q)) * 0.05) for k in range(3)]
        B = 0.3 * self.B
        _, grad = profile_loglik(B, people)
        E = rng.normal(size=B.shape)
        numeric = (profile_loglik(B + 1e-6 * E, people)[0] - profile_loglik(B - 1e-6 * E, people)[0]) / 2e-6
        self.assertAlmostEqual(numeric, float(np.sum(grad * E)), delta=1e-5 * abs(numeric) + 1e-8)
        values = [profile_loglik(s * self.B, people)[0] for s in np.linspace(-1, 1, 21)]
        self.assertTrue(np.all(np.diff(values, 2) <= 1e-9))
        n, m, Sxy = people[0]
        S = np.block([[m.Sx, Sxy], [Sxy.T, m.Sy]])

        def exact(Bm):
            C = transfer(m, Bm)[0]
            Sigma = np.block([[m.Sx, C], [C.T, m.Sy]])
            return -0.5 * (np.linalg.slogdet(Sigma)[1] + np.trace(S @ np.linalg.inv(Sigma)))

        diff_profile = profile_loglik(0.2 * self.B, [(1, m, Sxy)])[0] - profile_loglik(0.5 * self.B, [(1, m, Sxy)])[0]
        self.assertAlmostEqual(diff_profile, exact(0.2 * self.B) - exact(0.5 * self.B), places=10)

    def test_fit_recovers_interaction_from_model_moments(self):
        rng = self.rng
        truth = rng.normal(size=(self.p, self.q)) * 0.5
        people = []
        for _ in range(4):
            m = Marginal(random_cov(rng, self.p), random_cov(rng, self.q))
            people.append((1000, m, transfer(m, truth)[0]))
        fitted, _ = fit_interaction(people, ridge=0.0)
        np.testing.assert_allclose(fitted, truth, atol=1e-5)

    def test_cell_coupling_matches_closed_form_for_gaussian_clouds(self):
        from colon_scale import cell_coupling
        rng = np.random.default_rng(1)
        x = rng.multivariate_normal(np.zeros(self.p), self.Sx, 4000)
        y = rng.multivariate_normal(np.zeros(self.q), self.Sy, 4000)
        B = 0.5 * self.B
        C_cells, _ = cell_coupling(B, x, y)
        C_closed = transfer(Marginal(np.cov(x.T, bias=True), np.cov(y.T, bias=True)), B)[0]
        self.assertLess(np.abs(C_cells - C_closed).max(), 0.02 * np.abs(C_closed).max())


class PairingFreeTests(unittest.TestCase):
    def test_gradients_and_recovery_under_shared_channel(self):
        from pairing_free import RANK, ecological, lowrank_negloglik
        rng = np.random.default_rng(5)
        p, q = 7, 5
        W = (rng.normal(size=(q, RANK)) @ rng.normal(size=(p, RANK)).T) * 0.1
        b, psi = rng.normal(size=q), rng.uniform(0.3, 1.0, q)
        units = []
        for _ in range(30):
            Sx = random_cov(rng, p)
            mx = rng.normal(size=p)
            units.append((400, mx, W @ mx + b, Sx, W @ Sx @ W.T + np.diag(psi)))
        z = rng.normal(size=q * RANK + p * RANK + 2 * q) * 0.3
        _, grad = lowrank_negloglik(z, units, p, q, 0.01)
        e = rng.normal(size=z.size)
        numeric = (lowrank_negloglik(z + 1e-6 * e, units, p, q, 0.01)[0]
                   - lowrank_negloglik(z - 1e-6 * e, units, p, q, 0.01)[0]) / 2e-6
        self.assertAlmostEqual(numeric, float(grad @ e), places=6)
        What, bhat, psihat, _ = ecological(units, 1e-8)
        np.testing.assert_allclose(What, W, atol=1e-6)
        np.testing.assert_allclose(psihat, psi, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
