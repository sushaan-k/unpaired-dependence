#!/usr/bin/env python3
"""Analytic regression checks for the bounded theory corrections.

These tests evaluate the identities and counterexamples themselves.  They do
not inspect manuscript text and do not regenerate any empirical result.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np


def mean_and_cov(support, probabilities):
    support = np.asarray(support, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    mean = probabilities @ support
    errors = support - mean
    covariance = (errors * probabilities[:, None]).T @ errors
    return mean, covariance


def exact_u_statistic_moments(support, probabilities, sample_size):
    """Enumerate the iid law of the order-two signal U-statistic."""
    support = np.asarray(support, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    values = []
    weights = []
    for indices in itertools.product(range(len(support)), repeat=sample_size):
        sample = support[list(indices)]
        total = sample.sum(axis=0)
        value = (total @ total - np.sum(sample * sample)) / (
            sample_size * (sample_size - 1)
        )
        values.append(value)
        weights.append(np.prod(probabilities[list(indices)]))
    values = np.asarray(values)
    weights = np.asarray(weights)
    mean = np.sum(weights * values)
    variance = np.sum(weights * (values - mean) ** 2)
    return mean, variance


def signal_variance(theta, covariance, sample_size):
    return (
        4 * theta @ covariance @ theta / sample_size
        + 2 * np.trace(covariance @ covariance)
        / (sample_size * (sample_size - 1))
    )


def oracle_linear_risk(signal_squared, noise_trace, sample_size):
    denominator = sample_size * signal_squared + noise_trace
    if denominator == 0:
        return 0.0
    return signal_squared * noise_trace / denominator


def minimum_pilot_size(dimension, signal_squared, relative_sd):
    """Smallest m making the exact spherical relative SD no larger than target."""
    m = 2
    while True:
        relative_variance = (
            4 / (m * signal_squared)
            + 2 * dimension / (m * (m - 1) * signal_squared**2)
        )
        if relative_variance <= relative_sd**2:
            return m
        m += 1


def positive_part_rule(sample):
    sample = np.asarray(sample, dtype=float)
    sample_size = len(sample)
    mean = sample.mean(axis=0)
    trace_estimate = np.sum((sample - mean) ** 2) / (
        sample_size * (sample_size - 1)
    )
    norm_squared = mean @ mean
    factor = max(1 - trace_estimate / norm_squared, 0) if norm_squared else 0
    raw_signal = norm_squared - trace_estimate
    return factor * mean, raw_signal, max(raw_signal, 0)


class PilotVarianceTests(unittest.TestCase):
    def test_exact_u_statistic_variance_for_finite_vector_law(self):
        support = np.array([[2, -1], [-1, 0], [0, 2], [1, 1]], dtype=float)
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        theta, covariance = mean_and_cov(support, probabilities)
        observed_mean, observed_variance = exact_u_statistic_moments(
            support, probabilities, sample_size=3
        )

        self.assertAlmostEqual(observed_mean, theta @ theta, places=13)
        self.assertAlmostEqual(
            observed_variance,
            signal_variance(theta, covariance, sample_size=3),
            places=12,
        )

    def test_independent_product_construction_has_prescribed_moments(self):
        # X is a random sign, epsilon is independent with covariance I, and
        # Y = X theta + epsilon.  Thus Z = XY has mean theta and covariance I.
        theta = np.array([0.1, -0.2, 0.3])
        dimension = len(theta)
        products = []
        probabilities = []
        for x in (-1.0, 1.0):
            for coordinate in range(dimension):
                for sign in (-1.0, 1.0):
                    epsilon = np.zeros(dimension)
                    epsilon[coordinate] = sign * np.sqrt(dimension)
                    y = x * theta + epsilon
                    products.append(x * y)
                    probabilities.append(1 / (4 * dimension))

        product_mean, product_covariance = mean_and_cov(products, probabilities)
        np.testing.assert_allclose(product_mean, theta, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            product_covariance, np.eye(dimension), atol=1e-14, rtol=0
        )
        observed_mean, observed_variance = exact_u_statistic_moments(
            products, probabilities, sample_size=3
        )
        self.assertAlmostEqual(observed_mean, theta @ theta, places=13)
        self.assertAlmostEqual(
            observed_variance,
            signal_variance(theta, product_covariance, sample_size=3),
            places=12,
        )

    def test_finite_variance_term_dominates_the_weak_signal_example(self):
        dimension = 1000
        signal_squared = 0.01
        sample_size = 100
        first_term_relative_sd = np.sqrt(
            4 * signal_squared / sample_size
        ) / signal_squared
        exact_relative_sd = np.sqrt(
            4 * signal_squared / sample_size
            + 2 * dimension / (sample_size * (sample_size - 1))
        ) / signal_squared

        self.assertAlmostEqual(first_term_relative_sd, 2.0, places=12)
        self.assertAlmostEqual(exact_relative_sd, 44.99, places=2)
        self.assertGreater(exact_relative_sd, 20 * first_term_relative_sd)

    def test_fixed_relative_precision_can_require_budget_over_sqrt_dimension(self):
        signal_squared = 0.01
        relative_sd = 0.5
        dimensions = (10_000, 1_000_000)
        pilot_sizes = [
            minimum_pilot_size(d, signal_squared, relative_sd)
            for d in dimensions
        ]
        scaled = [
            m / (np.sqrt(d) / signal_squared)
            for d, m in zip(dimensions, pilot_sizes)
        ]

        # The limiting constant is sqrt(2)/relative_sd.  In contrast, a
        # dimension-free multiple of budget/dimension cannot control this term.
        np.testing.assert_allclose(
            scaled, np.sqrt(2) / relative_sd, rtol=0.03, atol=0
        )
        self.assertAlmostEqual(pilot_sizes[1] / pilot_sizes[0], 10, delta=0.3)


class CenteringTests(unittest.TestCase):
    def test_independent_product_covariance_includes_residual_mean_terms(self):
        x_support = np.array([[-1, 0], [2, 1], [0, 3]], dtype=float)
        x_probabilities = np.array([0.2, 0.5, 0.3])
        y_support = np.array([[1, -2], [3, 0], [-1, 1]], dtype=float)
        y_probabilities = np.array([0.25, 0.5, 0.25])
        mean_x, covariance_x = mean_and_cov(x_support, x_probabilities)
        mean_y, covariance_y = mean_and_cov(y_support, y_probabilities)

        product_support = np.array(
            [np.outer(x, y).ravel() for x in x_support for y in y_support]
        )
        product_probabilities = np.array(
            [px * py for px in x_probabilities for py in y_probabilities]
        )
        _, product_covariance = mean_and_cov(
            product_support, product_probabilities
        )
        corrected = (
            np.kron(covariance_x, covariance_y)
            + np.kron(covariance_x, np.outer(mean_y, mean_y))
            + np.kron(np.outer(mean_x, mean_x), covariance_y)
        )

        np.testing.assert_allclose(product_covariance, corrected, atol=1e-13)
        self.assertGreater(
            np.linalg.norm(product_covariance - np.kron(covariance_x, covariance_y)),
            1,
        )

    def test_scalar_residual_centering_counterexample(self):
        residual_mean_x = 0.7
        residual_mean_y = -1.2
        exact_variance = (
            1 + residual_mean_x**2 + residual_mean_y**2
        )
        self.assertAlmostEqual(exact_variance, 2.93)
        self.assertNotEqual(exact_variance, 1.0)


class OracleInterpretationTests(unittest.TestCase):
    def test_bound_is_relative_to_noise_scale_not_uniformly_to_oracle(self):
        noise_trace = 1000.0
        largest_eigenvalue = 1.0
        kurtosis = 2.0
        sample_size = 100
        bound_squared = (
            36 * largest_eigenvalue / sample_size
            + 22 * kurtosis * noise_trace / sample_size**2
        )
        noise_root = np.sqrt(noise_trace / sample_size)
        noise_relative = np.sqrt(bound_squared) / noise_root

        self.assertAlmostEqual(
            noise_relative**2,
            36 / (noise_trace / largest_eigenvalue)
            + 22 * kurtosis / sample_size,
            places=14,
        )
        ratios = []
        for signal_squared in (1e-2, 1e-8):
            oracle_root = np.sqrt(
                oracle_linear_risk(signal_squared, noise_trace, sample_size)
            )
            relative_to_oracle = np.sqrt(bound_squared) / oracle_root
            multiplier = np.sqrt(
                (noise_trace / sample_size)
                / oracle_linear_risk(signal_squared, noise_trace, sample_size)
            )
            self.assertAlmostEqual(
                relative_to_oracle, noise_relative * multiplier, places=12
            )
            ratios.append(relative_to_oracle)
        self.assertGreater(ratios[1], 500 * ratios[0])

    def test_null_oracle_differs_from_positive_part_estimator(self):
        estimates = []
        raw_signals = []
        clipped_signals = []
        for sample in itertools.product((-1.0, 1.0), repeat=2):
            estimate, raw_signal, clipped_signal = positive_part_rule(
                np.asarray(sample)[:, None]
            )
            estimates.append(float(estimate[0]))
            raw_signals.append(raw_signal)
            clipped_signals.append(clipped_signal)

        self.assertAlmostEqual(np.mean(raw_signals), 0.0)
        self.assertAlmostEqual(np.mean(clipped_signals), 0.5)
        self.assertAlmostEqual(np.mean(np.square(estimates)), 0.5)
        self.assertEqual(oracle_linear_risk(0.0, 1.0, 2), 0.0)
        self.assertEqual(oracle_linear_risk(0.0, 0.0, 2), 0.0)


class InverseAndSupportTests(unittest.TestCase):
    def test_positive_definite_condition_map(self):
        pooled = np.array([[2.0, 0.3], [0.3, 1.0]])
        condition = np.array([[1.5, 0.1], [0.1, 0.7]])
        channel_transpose = np.array([[1.0, -0.4], [0.2, 0.8]])
        pooled_cross_covariance = pooled @ channel_transpose
        mapped = condition @ np.linalg.inv(pooled) @ pooled_cross_covariance
        np.testing.assert_allclose(mapped, condition @ channel_transpose, atol=1e-14)

    def test_pseudoinverse_map_requires_compatible_support(self):
        pooled = np.diag([2.0, 0.0])
        condition = np.diag([3.0, 0.0])
        channel_transpose = np.array([[1.0], [5.0]])
        pooled_cross_covariance = pooled @ channel_transpose
        mapped = condition @ np.linalg.pinv(pooled) @ pooled_cross_covariance
        np.testing.assert_allclose(mapped, condition @ channel_transpose, atol=0)

        incompatible_condition = np.diag([0.0, 1.0])
        incompatible_target = incompatible_condition @ channel_transpose
        incompatible_map = (
            incompatible_condition
            @ np.linalg.pinv(pooled)
            @ pooled_cross_covariance
        )
        self.assertGreater(np.linalg.norm(incompatible_target - incompatible_map), 1)

    def test_finite_positive_matrix_scaling_reaches_positive_marginals(self):
        interaction = np.array([[0.2, -0.5, 0.7], [1.0, 0.1, -0.3]])
        kernel = np.exp(interaction)
        row_target = np.array([0.35, 0.65])
        column_target = np.array([0.2, 0.5, 0.3])
        row_scale = np.ones(2)
        column_scale = np.ones(3)
        for _ in range(1000):
            row_scale = row_target / (kernel @ column_scale)
            column_scale = column_target / (kernel.T @ row_scale)
        joint = row_scale[:, None] * kernel * column_scale[None, :]

        np.testing.assert_allclose(joint.sum(axis=1), row_target, atol=1e-14)
        np.testing.assert_allclose(joint.sum(axis=0), column_target, atol=1e-14)
        self.assertTrue(np.all(joint > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
