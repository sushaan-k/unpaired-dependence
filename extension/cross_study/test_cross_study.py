"""Unit tests for the cross-study pairing-free extension (synthetic data only)."""

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spectral_transfer"))

import estimators as est  # noqa: E402
from identifiability import complement, floor, identified_subspace, radius  # noqa: E402
from permutation import derangement  # noqa: E402
from posthoc_mixture import decompose  # noqa: E402


def random_units(rng, S=12, p=9, q=6):
    units = []
    for s in range(S):
        A = rng.normal(size=(p, p))
        Sx = A @ A.T / p + np.eye(p)
        B = rng.normal(size=(q, q))
        Sy = B @ B.T / q + np.eye(q)
        units.append((f"u{s}", 50 + s, rng.normal(size=p), rng.normal(size=q), Sx, Sy))
    return units


class Likelihood(unittest.TestCase):
    def test_fast_equals_reference(self):
        rng = np.random.default_rng(1)
        p, q = 9, 6
        units = random_units(rng, p=p, q=q)
        for r in (1, 3, 6):
            z = np.concatenate([rng.normal(size=q * r) * 0.3, rng.normal(size=p * r) * 0.3,
                                rng.normal(size=q), rng.normal(size=q) * 0.2])
            v1, g1 = est.moment_negloglik(z, units, p, q, r, 1e-3)
            v2, g2 = est.moment_negloglik_fast(z, units, p, q, r, 1e-3)
            self.assertAlmostEqual(v1, v2, places=10)
            np.testing.assert_allclose(g1, g2, rtol=1e-9, atol=1e-11)

    def test_matches_pairing_free_d3(self):
        import pairing_free
        rng = np.random.default_rng(2)
        p, q, r = 12, 10, pairing_free.RANK
        units = random_units(rng, p=p, q=q)
        z = np.concatenate([rng.normal(size=q * r) * 0.3, rng.normal(size=p * r) * 0.3,
                            rng.normal(size=q), rng.normal(size=q) * 0.2])
        v1, g1 = est.moment_negloglik(z, units, p, q, r, 1e-3)
        v2, g2 = pairing_free.lowrank_negloglik(z, [u[1:] for u in units], p, q, 1e-3)
        self.assertEqual(v1, v2)
        np.testing.assert_array_equal(g1, g2)

    def test_fast_gradient_finite_difference(self):
        rng = np.random.default_rng(3)
        p, q, r = 7, 5, 2
        units = random_units(rng, p=p, q=q)
        z = np.concatenate([rng.normal(size=q * r) * 0.3, rng.normal(size=p * r) * 0.3,
                            rng.normal(size=q), rng.normal(size=q) * 0.2])
        _, g = est.moment_negloglik_fast(z, units, p, q, r, 1e-3)
        h = 1e-6
        for i in rng.choice(len(z), 10, replace=False):
            e = np.zeros_like(z)
            e[i] = h
            fd = (est.moment_negloglik_fast(z + e, units, p, q, r, 1e-3)[0]
                  - est.moment_negloglik_fast(z - e, units, p, q, r, 1e-3)[0]) / (2 * h)
            self.assertAlmostEqual(g[i], fd, delta=1e-6 * max(1, abs(fd)))


class Identification(unittest.TestCase):
    def test_means_identify_channel_on_mean_span_only(self):
        rng = np.random.default_rng(4)
        p, q, S, m = 10, 4, 30, 3
        basis, _ = np.linalg.qr(rng.normal(size=(p, m)))
        W = rng.normal(size=(q, p))
        units = []
        for s in range(S):
            mx = basis @ rng.normal(size=m) * 3
            units.append((s, 100, mx, W @ mx + 0.5, np.eye(p), np.eye(q)))
        Wh, b, *_ = est.ecological(units, 1e-8)
        np.testing.assert_allclose(Wh @ basis, W @ basis, atol=1e-6)            # identified on M
        np.testing.assert_allclose(Wh @ complement(basis), 0, atol=1e-6)       # nothing off M
        G = est.ecological(units, 1e-3)[4]
        VM, VU, h = identified_subspace(G, 1e-3 * np.trace(G))
        self.assertEqual(VM.shape[1], m)

    def test_radius_is_attained_and_never_exceeded(self):
        rng = np.random.default_rng(5)
        p, q = 8, 5
        A = rng.normal(size=(p, p)); Rx = A @ A.T / p + np.eye(p)
        Bm = rng.normal(size=(q, q)); Ry = Bm @ Bm.T / q + np.eye(q)
        W = rng.normal(size=(q, p)) * 0.1
        VM, _ = np.linalg.qr(rng.normal(size=(p, 3)))
        r, centre, sF = radius(Rx, Ry, W, VM)
        self.assertLess(sF, 1)
        # every member is centre + Rx^1/2 Qp Gamma D Ry^1/2 with ||Gamma|| <= 1
        from identifiability import psd_power
        Rxh, Rxih, Ryh = psd_power(Rx, 0.5), psd_power(Rx, -0.5), psd_power(Ry, 0.5)
        QN, _ = np.linalg.qr(Rxih @ VM)
        Qp = complement(QN)
        F = QN.T @ Rxh @ W.T @ psd_power(Ry, -0.5)
        w, V = np.linalg.eigh(np.eye(q) - F.T @ F)
        D = (V * np.sqrt(w)) @ V.T
        L, R = Rxh @ Qp, D @ Ryh
        best = 0.0
        for _ in range(2000):
            G = rng.normal(size=(Qp.shape[1], q))
            G /= np.linalg.norm(G, 2)
            member = centre + L @ G @ R
            K = Rxih @ member @ psd_power(Ry, -0.5)
            self.assertLessEqual(np.linalg.norm(K, 2), 1 + 1e-9)                  # valid joint law
            np.testing.assert_allclose(QN.T @ K, F, atol=1e-9)                      # same identified part
            best = max(best, np.linalg.norm(member - centre))
        self.assertLessEqual(best, r + 1e-9)
        Ul, _, Vlt = np.linalg.svd(L, full_matrices=False)
        Ur, _, Vrt = np.linalg.svd(R, full_matrices=False)
        k = min(Ul.shape[1], Ur.shape[1])
        G = Vlt.T[:, :k] @ Ur[:, :k].T                                            # aligned partial isometry
        self.assertAlmostEqual(np.linalg.norm(L @ G @ R), r, places=9)

    def test_floor_zero_when_channel_lies_in_M(self):
        rng = np.random.default_rng(6)
        p, q = 8, 4
        VM, _ = np.linalg.qr(rng.normal(size=(p, 3)))
        W = rng.normal(size=(q, 3)) @ VM.T
        A = rng.normal(size=(p, p)); Rx = A @ A.T / p + np.eye(p)
        self.assertLess(floor(Rx @ W.T, Rx, VM), 1e-10)


class Rules(unittest.TestCase):
    def test_edge_rule_extends(self):
        best, report = est.tune((1e-4, 1e-3, 1e-2, 1e-1, 1.0), lambda v: (np.log10(v) + 5.2) ** 2)
        self.assertEqual(report["extensions"], 2)
        self.assertFalse(report["at_edge"])
        self.assertAlmostEqual(best, 1e-5)

    def test_derangement_within_groups(self):
        rng = np.random.default_rng(7)
        groups = [list("abcd"), list("efg"), list("hi")]
        for _ in range(50):
            mapping = derangement(groups, rng)
            for g in groups:
                self.assertEqual(sorted(mapping[m] for m in g), sorted(g))
                self.assertTrue(all(mapping[m] != m for m in g))

    def test_pairing_free_moments_use_disjoint_halves(self):
        rng = np.random.default_rng(8)
        n = 400
        x, y = rng.normal(size=(n, 3)), rng.normal(size=(n, 2))
        half = np.repeat([0, 1], n // 2)
        train = {"x": x, "y": y, "patient": np.array(["p"] * n), "half": half}
        raw = est.pf_raw_moments(train, ["p"])[0]
        np.testing.assert_allclose(raw["mx"], x[:200].mean(0))
        np.testing.assert_allclose(raw["my"], y[200:].mean(0))
        self.assertEqual((raw["nx"], raw["ny"]), (200, 200))


class Mixture(unittest.TestCase):
    def test_between_plus_within_is_total(self):
        rng = np.random.default_rng(11)
        labels = rng.choice(np.array(["a", "b", "c"]), size=600)
        x, y = rng.normal(size=(600, 4)), rng.normal(size=(600, 3))
        x[labels == "b"] += 2.0
        y[labels == "c"] -= 1.5
        Cb, Cw, sx, sy, n = decompose(x, y, labels, ["a", "b", "c"])
        total = (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)
        np.testing.assert_allclose(Cb + Cw, total, atol=1e-12)
        self.assertEqual(n, 600)

    def test_means_fix_between_type_part_for_any_proportions(self):
        # Proposition S8 (2): with fixed type laws, the between-type part is the
        # between-type RNA covariance times the between-type slope.
        rng = np.random.default_rng(12)
        mx = rng.normal(size=(3, 5))
        Wb = rng.normal(size=(2, 5))
        my = mx @ Wb.T
        for pi in (np.array([.2, .3, .5]), np.array([.6, .1, .3])):
            xbar, ybar = pi @ mx, pi @ my
            between = sum(p * np.outer(a - xbar, b - ybar) for p, a, b in zip(pi, mx, my))
            sbx = sum(p * np.outer(a - xbar, a - xbar) for p, a in zip(pi, mx))
            np.testing.assert_allclose(between, sbx @ Wb.T, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
