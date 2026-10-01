# Predicting the saving: theory (manuscript Supplementary Note 13, Propositions S13-S15)

The earlier statements (`../semipaired/THEORY.md`, Propositions S10-S12) concern idealized estimators: Gaussian block
means with known noise and an untruncated factor (S10), and a Gaussian sequence model with spherical noise within
blocks (S12). This note (i) bounds the rule that is actually implemented, (ii) derives the saving of paired cells
that the unpaired block structure buys, and (iii) states what unpaired cells can and cannot reveal about it.
`check_theory.py` checks the statements numerically (`results/check_theory.json`); the checks are not proofs.

## Setting

Everything is conditional on the unpaired pool. Given the pool, the bases U and V, the scaling, the population means
used for centring and the block partition are fixed. Paired cells i = 1..B are independent draws from the
population. For block b (a product of an RNA eigen-block R_b and a protein eigen-block C_b, with d_b coefficients),
z_ib is the vector of the cell's coefficients of x~ y~' in the block (x~ = U'x, y~ = V'y), and

    theta_b = E z_ib,   Sigma_b = Cov z_ib,   tau_b = tr Sigma_b (per-cell noise trace),
    l_b = lambda_max(Sigma_b),   kappa_b = Var(||z_ib - theta_b||^2) / tau_b^2,   r_b^2 = ||theta_b||^2.

The implemented rule (`sp_estimators.block_js2`) computes the block mean m_b = (1/B) sum_i z_ib, the unbiased
estimate t_b = sum_i ||z_ib - m_b||^2 / (B (B-1)) of tr Cov(m_b) = tau_b / B, and returns
theta^_b = (1 - t_b / ||m_b||^2)_+ m_b. Paired-only James-Stein is the same rule with one block of all coefficients.

For a fixed number c, E||c m_b - theta_b||^2 = c^2 tau_b/B + (1-c)^2 r_b^2 for any distribution with these first two
moments. Its minimum over c,

    rho(r_b^2, tau_b; B) = r_b^2 tau_b / (B r_b^2 + tau_b)        (the oracle-linear risk),

is attained at c = r_b^2 / (r_b^2 + tau_b/B). It depends on the block only through its signal r_b^2 and its
per-cell noise trace tau_b.

## Proposition S13 (the implemented rule, without Gaussian assumptions)

If B >= 2 and E||z_ib||^4 < infinity, then

    | (E||theta^_b - theta_b||^2)^(1/2) - rho(r_b^2, tau_b; B)^(1/2) |  <=  (36 l_b/B + 22 kappa_b tau_b/B^2)^(1/2).

Proof (drop b). Write m = theta + e, s = tau/B, A = ||m||^2, mu = E A = r^2 + s, a* = 1 - s/mu in [0, 1) and
theta* = a* m, whose risk is rho. Because t >= 0, the implemented factor is a = clip(1 - t/A) to [0, 1] (a = 0 when
A = 0), so theta^ = a m. By Minkowski's inequality in L2,

    | (E||theta^ - theta||^2)^(1/2) - rho^(1/2) | <= (E||theta^ - theta*||^2)^(1/2) = (E[(a - a*)^2 A])^(1/2) =: Delta^(1/2).

Clipping to [0, 1] is 1-Lipschitz and leaves a* unchanged, so |a - a*| <= |t/A - s/mu| for A > 0, and
|a - a*| <= 1 always. On {A >= mu/2},

    (t/A - s/mu)^2 A = ((t - s) mu + s (mu - A))^2 / (A mu^2) <= 4 (t - s)^2 / mu + 4 s^2 (A - mu)^2 / mu^3.

On {A < mu/2}, (a - a*)^2 A <= A < mu/2, and P(A < mu/2) <= 4 Var(A)/mu^2 by Chebyshev's inequality. Since E t = s,

    Delta <= 4 Var(t)/mu + (2 + 4 s^2/mu^2) Var(A)/mu <= (4 Var(t) + 6 Var(A)) / mu.

With e_i = z_i - theta (independent, mean zero, covariance Sigma), expanding A = r^2 + 2 theta'e_bar + ||e_bar||^2,

    Var(A) = 4 theta' Sigma theta / B + 4 E[theta'e (||e||^2 - tau)] / B^2 + (Var||e||^2 + 2 (B-1) tr Sigma^2) / B^3,

and t is the unbiased trace of a sample covariance divided by B, a U-statistic of order two, so

    Var(t) = (Var||e||^2 / B + 2 tr Sigma^2 / (B (B-1))) / B^2.

Using theta'Sigma theta <= l r^2, tr Sigma^2 <= l tau, Var||e||^2 = kappa tau^2, the Cauchy-Schwarz bound
|E[theta'e (||e||^2 - tau)]| <= (l r^2)^(1/2) kappa^(1/2) tau and 4xy <= 2x^2 + 2y^2,

    Var(A) <= 6 l r^2 / B + 2 l tau / B^2 + 3 kappa tau^2 / B^3,     Var(t) <= kappa tau^2 / B^3 + l tau / B^2 (B >= 2),

so 4 Var(t) + 6 Var(A) <= 36 l r^2 / B + 16 l tau / B^2 + 22 kappa tau^2 / B^3. Dividing by mu = r^2 + tau/B gives
Delta <= 36 l/B + 22 kappa tau / B^2. QED.

Remarks. (1) Relative to rho <= tau/B, the error term is of order (l/tau)^(1/2) = d_eff^(-1/2), with
d_eff = tau/l the block's effective dimension, plus kappa/B. The rule attains the oracle-linear risk for blocks of
many effective dimensions and budgets above the kurtosis of the products; for the smallest blocks the bound is
loose. Numerically (72 cases, Gaussian, log-normal and Poisson margins, d = 15 and 120, B = 5 to 100) the gap never
exceeded 0.10 of the bound. (2) The statement with one block covers paired-only James-Stein. (3) Given the pool,
theta_b includes the outer product of the errors of the unpaired means used for centring, of order 1/n for n
unpaired cells per population; the risk against the population value changes by at most its norm (Minkowski).
(4) The constant 36 comes from a Chebyshev split and is not sharp. For Gaussian block means S10 gives the tighter
excess 4 l_b/B; in simulations the excess of the implemented rule over rho is close to 2 l_b/B when the block
carries no signal (the "leak" of James-Stein shrinkage towards zero).

## Proposition S14 (the saving is a divergence between where dependence lies and where noise lies)

Oracle-linear risks: with blocks, R_K(B) = sum_b rho(r_b^2, tau_b; B); with one block of all coefficients,
R_1(B) = rho(r^2, tau; B), where r^2 = sum_b r_b^2 and tau = sum_b tau_b (the trace over all coefficients is the sum
of the blocks' traces). For a relative risk eps in (0, 1), let B_K(eps) and B_1(eps) solve R_K(B) = eps r^2 and
R_1(B) = eps r^2, and let S(eps) = B_1(eps) / B_K(eps). Write w_b = r_b^2 / r^2 (where the dependence lies) and
pi_b = tau_b / tau (where the noise lies). Assume r^2 > 0 and tau_b > 0.

 (a) S depends on the dependence only through w: S(eps) = (1 - eps) / (eps beta), where beta solves
     Phi(beta) = sum_b w_b pi_b / (beta w_b + pi_b) = eps; B_1 = (tau/r^2)(1-eps)/eps and B_K = beta tau/r^2.
 (b) S(eps) >= 1, with equality (at any, equivalently every, eps) if and only if w = pi.
 (c) S(eps) <= 1 / min{pi_b : w_b > 0}, with equality when all dependence lies in one block.
 (d) S(eps) -> 1 / sum_{b: w_b > 0} pi_b as eps -> 0 (high accuracy: only blocks without any dependence are
     saved), and S(eps) -> sum_b w_b^2 / pi_b = 1 + chi^2(w || pi) as eps -> 1 (low accuracy).

Proof. rho(x, y; B) = xy / (Bx + y) = min over c of [c^2 y/B + (1-c)^2 x], a minimum of linear functions, so it is
jointly concave in (x, y) and positively homogeneous of degree one.
(a) With B = beta tau / r^2, rho(r_b^2, tau_b; B) / r^2 = w_b pi_b / (beta w_b + pi_b) and R_1(B)/r^2 = 1/(beta + 1).
Phi is continuous and strictly decreasing from Phi(0) = 1 to 0, so beta(eps) is unique.
(b) R_1(B) = min_c sum_b [c^2 tau_b/B + (1-c)^2 r_b^2] >= sum_b min_{c_b} [c_b^2 tau_b/B + (1-c_b)^2 r_b^2] = R_K(B),
with equality if and only if the blockwise minimizers r_b^2/(r_b^2 + tau_b/B) coincide, that is, r_b^2/tau_b is the
same in every block (w = pi). At B = B_1(eps), R_K <= eps r^2, and R_K decreases in B, so B_K <= B_1, strictly when
w != pi. If w = pi, R_K = R_1 for every B.
(c) Let pi_+ = min{pi_b : w_b > 0} and tau_+ = pi_+ tau. For w_b > 0, rho(r_b^2, tau_b; B) >= rho(r_b^2, tau_+; B)
(rho increases in its second argument), and x -> rho(x, tau_+; B) is concave with value 0 at 0, hence subadditive,
so R_K(B) >= rho(r^2, tau_+; B). Hence R_K(B) <= eps r^2 requires B >= (tau_+/r^2)(1-eps)/eps = pi_+ B_1. Equality
holds when all signal is in one block of noise share pi_+.
(d) As eps -> 0, beta -> infinity and Phi(beta) = beta^(-1) sum_{w_b > 0} pi_b + O(beta^(-2)). As eps -> 1,
beta -> 0 and Phi(beta) = 1 - beta sum_b w_b^2/pi_b + O(beta^2). QED.

Interpretation. The saving that unpaired bases buy is one when the dependence is spread over the blocks in
proportion to the per-cell noise, and grows as the dependence concentrates where the noise is small. It is not a
property of the unpaired data alone (pi) nor of the dependence alone (w), but of the two together.

## Proposition S15 (what unpaired cells reveal)

 (a) Noise shares. If x and y are independent (any distributions with finite fourth moments), the per-cell
     covariance of a product block is the Kronecker product Sigma_b = Cov(x~_R) (x) Cov(y~_C), so
     tau_b = tr Cov(x~_R) tr Cov(y~_C) and l_b = lambda_max(Cov x~_R) lambda_max(Cov y~_C): functions of the two
     marginal distributions, which unpaired cells estimate. If (x, y) is jointly Gaussian,
     tau_b = tr Cov(x~_R) tr Cov(y~_C) + r_b^2, and r_b^2 <= tr Cov(x~_R) tr Cov(y~_C) in the eigenbases of the two
     marginal covariances, so the dependence moves each tau_b by at most a factor of two, and by little when it is
     weak.
 (b) Signal shares are not revealed. Given two marginal Gaussian laws and any w on the simplex, there are jointly
     Gaussian laws with these marginals whose signal shares are w, for every small enough total signal r^2. Hence,
     as the dependence becomes weak, the savings compatible with the unpaired data fill the interval
     [1, 1/pi_min], with pi the independence noise shares and pi_min their minimum over blocks, and the budget
     B_K(eps) = beta(eps) tau / r^2 is not bounded above.
 (c) A pilot of m paired cells estimates each r_b^2 without bias by ||m_b||^2 - t_b, with variance at most
     2 Var(A) + 2 Var(t) = O(l_b r_b^2/m + l_b tau_b/m^2 + kappa_b tau_b^2/m^3) (moments as in S13). To leading
     order its relative error is at most 2 (d_eff,b m rho_b)^(-1/2), with rho_b = r_b^2/tau_b, so it falls below one
     half once m exceeds 16/(d_eff,b rho_b) cells, whereas the block's own shrinkage factor reaches one half at
     B = 1/rho_b cells. Predicting where the dependence lies therefore costs paired cells in proportion to the
     budget each block needs, divided by its effective dimension: cheap for blocks of many effective dimensions,
     about as expensive as the estimation itself for blocks of a few.

Proof. (a) Isserlis' theorem for centred jointly Gaussian variables gives
Cov(x~_k y~_l, x~_k' y~_l') = Cov(x~_k, x~_k') Cov(y~_l, y~_l') + Cov(x~_k, y~_l') Cov(x~_k', y~_l); summing the
diagonal (k = k', l = l') over the block gives tau_b. Under independence the second term vanishes for any
distributions, and the covariance factorizes into the Kronecker product. In the eigenbases,
Cov(x~_k, y~_l)^2 <= Var(x~_k) Var(y~_l) (Cauchy-Schwarz), and summing gives r_b^2 <= tau_b(independence).
(b) In the eigenbases of the marginal covariances, Lambda and M, choose unit-Frobenius matrices E_b supported on the
blocks and set Theta = c sum_b w_b^(1/2) E_b. The joint covariance [[Lambda, Theta], [Theta', M]] is positive
semidefinite if and only if ||Lambda^(-1/2) Theta M^(-1/2)||_op <= 1, which holds for c small enough; its signal
shares are w and r^2 = c^2, and by (a) its noise shares tend to their independence values as c -> 0. By S14(a), S is
a continuous function of w on the simplex (beta solves a continuous, strictly monotone equation); it equals 1 at
w = pi and 1/pi_min at the vertex of the block with the smallest noise share, so by connectedness it takes every
value in between, and S14(b, c) exclude values outside. B_K = beta tau / c^2 grows without bound as c -> 0.
(c) r_b^2 estimate: E(A - t) = mu - s = r^2; the variance bound follows from Var(A - t) <= 2 Var(A) + 2 Var(t) and
the bounds in the proof of S13 with B = m. To leading order in 1/m, Var(A - t) = 4 theta'Sigma theta / m <=
4 l r^2 / m, so the relative error is at most about (4 l / (m r^2))^(1/2) = 2 (d_eff m rho)^(-1/2). QED.

## Consequences for design

* Unpaired profiles alone bound the saving to [1, 1/pi_min] but cannot determine it or the budget.
* With the signal of each block known, the saving and the budget follow from S14, and the implemented rule attains
  them up to the terms of S13.
* A pilot turns the bound into an estimate at a cost in paired cells that grows with the budget itself (S15c); its
  uncertainty can be propagated by resampling the pilot cells.
