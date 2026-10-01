# Risk of the semi-paired estimator (idealized statements and proofs)

Ninth-session note. Propositions 1 and 3 below concern idealized estimators (Gaussian block means with known noise
and an untruncated factor; a Gaussian sequence model with spherical noise within blocks). They are not
finite-sample guarantees for the implemented rule, and numerical checks do not make them so. The implemented rule
(means of non-Gaussian per-cell products, plug-in noise trace, positive part) is bounded without Gaussian
assumptions in `../predictability/THEORY.md` (manuscript Proposition S13), which also derives the saving (S14) and
what unpaired data reveal about it (S15).

Notation. Paired calibration cells give per-cell products z_i = x_i y_i' (x centred RNA, y centred protein,
pooled-SD units); the pooled cross-covariance is theta = E z (p x q), estimated by the mean of B cells. Two
orthonormal bases are estimated from unpaired cells only: U (p x p), the eigenvectors of the latent (count-split)
RNA correlation, and V (q x q), the eigenvectors of the protein correlation. The coefficient matrix U' theta V is
partitioned into blocks b = (RNA eigen-block) x (protein eigen-block): RNA blocks 1-5, 6-20, 21-60 and the rest,
protein blocks 1-3 and the rest. theta_b is the vector of coefficients in block b, of dimension d_b. The map
M -> U' M V is an isometry of p x q matrices in the Frobenius norm, so the total squared error is the sum of the
blocks' squared errors. Because U and V are computed from cells disjoint from the paired ones, they are fixed with
respect to the paired sample, and every statement below holds conditionally on them.

## Proposition 1 (block James-Stein with correlated, unequal noise)

Assume theta_hat_b ~ N(theta_b, S_b) for an arbitrary positive semidefinite S_b (correlated and heteroscedastic;
S_b = Sigma_b / B, with Sigma_b the covariance of the per-cell product coefficients). Let
t_b = tr S_b, l_b = lambda_max(S_b), and delta_b = (1 - t_b / ||theta_hat_b||^2) theta_hat_b. If t_b >= 4 l_b,

    E ||delta_b - theta_b||^2  <=  ||theta_b||^2 t_b / (||theta_b||^2 + t_b)  +  4 l_b
                               <=  min(||theta_b||^2, t_b) + 4 l_b.

Proof. Write X = theta_hat_b and delta = X - c X / ||X||^2 with c = t_b. Stein's identity for a Gaussian vector with
covariance S gives E <X - theta, g(X)> = E tr(S Dg(X)) for weakly differentiable g. With g(X) = X/||X||^2,
tr(S Dg(X)) = tr S / ||X||^2 - 2 X'SX / ||X||^4. Hence

    E||delta - theta||^2 = tr S - 2c E[tr S/||X||^2 - 2 X'SX/||X||^4] + c^2 E[1/||X||^2]
                         <= tr S - c (2 tr S - c - 4 l) E[1/||X||^2]
                         =  t - t (t - 4 l) E[1/||X||^2]                      (c = t).

If t >= 4 l the bracket is nonnegative, and Jensen's inequality, E[1/||X||^2] >= 1/E||X||^2 = 1/(||theta||^2 + t),
gives E||delta - theta||^2 <= t (||theta||^2 + 4 l) / (||theta||^2 + t) <= ||theta||^2 t/(||theta||^2 + t) + 4 l. QED.

The condition t_b >= 4 l_b says that the block's noise has effective dimension tr S / lambda_max(S) of at least 4;
it is checked on the data from the per-cell products (effective_dimension.py). The idealizations are the
Gaussian law of the block means (a central-limit approximation), a known trace t_b (the estimator plugs in the
empirical trace of the per-cell products) and the plain rather than positive-part factor; with spherical noise
the positive part always lowers the risk (Baranchik), and the realized error of the rule actually used is
reported directly.

## Corollary (paired cells needed, and where the saving comes from)

Summing over blocks with S_b = Sigma_b / B,

    R(B) <= sum_b ||theta_b||^2 (tau_b/B) / (||theta_b||^2 + tau_b/B) + (4/B) sum_b rho_b,

tau_b = tr Sigma_b, rho_b = lambda_max(Sigma_b); the condition t_b >= 4 l_b reads tau_b >= 4 rho_b and does not
depend on B. Bounding each block's first term by tau_b/B for the blocks in any set A and by ||theta_b||^2 for the
others,

    R(B) <= sum_{b not in A} ||theta_b||^2 + (sum_{b in A} tau_b + 4 sum_b rho_b) / B.

Hence, for any target e > sum_{b not in A} ||theta_b||^2, a budget

    B >= (sum_{b in A} tau_b + 4 sum_b rho_b) / (e - sum_{b not in A} ||theta_b||^2)

guarantees R(B) <= e. Since the recovered fraction of the pooled target is 1 - ||delta - theta||^2 / ||theta||^2,
this is a guarantee on the expected recovered fraction, 1 - e / ||theta||^2, in the idealized setting.

James-Stein in marker coordinates (one block holding all p q coefficients) needs the same with A equal to
everything. The number of paired cells therefore scales with the noise of the blocks that carry the dependence,
not with the whole panel: the saving of the unpaired bases is the share of per-cell noise that falls outside the
blocks where the bases concentrate the dependence. Splitting the protein coordinates by the unpaired protein
eigenbasis concentrates it further when proteins co-vary in programs. A basis estimated from the paired cells
themselves is not independent of their noise, and random blocks do not concentrate the dependence, so neither
inherits the saving.

## Proposition 2 (condition map under a shared channel)

Let latent RNA z (measured RNA x = z + noise independent of everything else) and protein y = W z + e with e
independent of z, W shared by all conditions, and Cov(z) = Sigma_c in condition c. Then
Cov_c(x, y) = Sigma_c W', the pooled within-population cross-covariance is C = Sigma W' with Sigma the pooled
latent covariance, and therefore

    Cov_c(x, y) = Sigma_c Sigma^{-1} C.

Measured covariances would give the wrong map (Sigma_x,c Sigma_x^{-1} with the noise on both diagonals), which is
why the map uses latent covariances from count splitting. The estimator applies the map on the leading k latent
eigendirections (identity elsewhere) and shrinks it towards the identity by the split-half reliability of the
condition-minus-pooled covariance, both computed from unpaired cells only.

## What the statements cover in the implemented estimator (seventh session, final checks)

* **Bases.** U and V come from pool cells, which are disjoint from the reservoir, so given the unpaired cells they
  are fixed orthonormal bases independent of the paired cells, as Proposition 1 requires. Estimation error in the
  bases changes how well blocks concentrate the dependence (a_b, tau_b, rho_b), not the validity of the bound.
* **Cross-product blocks.** Proposition 1 holds for any partition of A = U' theta V, including the products of RNA
  and protein eigen-blocks; risks add because M -> U'MV is an isometry.
* **Noise.** S_b is arbitrary: correlated noise of unequal variance, as in per-cell products, is allowed.
* **Approximations.** Gaussian block means (central limit), known trace (estimated in practice), no positive part
  (truncated in practice), and centring on unpaired rather than exact population means, which adds the outer
  product of the independent, mean-zero errors of the unpaired RNA and protein means, of order 1/n.
* **Numerical check** (`bound_check.py`, `logs/bound_check_<dataset>.json`): with each reservoir as the population
  (OverCITE-seq 457 cells, Frangieh 21,171, Papalexi 2,256), bases from the pool and 200 draws with replacement per
  budget (25 to 800), the implemented estimator's mean squared error was below the bound at every budget in all three
  data sets (bound / risk 1.3 to 3.3) and below the idealized rule's; tau_b >= 4 rho_b held in every block except the
  Papalexi blocks formed with the protein direction that centred log-ratios leave without variance (zero signal and
  noise).
* **Condition map.** The implemented map is G_c = Sigma_c U_k Lambda_k^-1 U_k' + (I - U_k U_k') on the leading k = 120
  pooled latent eigendirections (U_k is p x k, Lambda_k is k x k; the manuscript's earlier U_k U_k' Lambda_k^-1 U_k'
  was a dimensional typo), with Sigma_c = D_x,c R_z,c D_x,c, shrunk towards the identity by the split-half
  reliability s_c, G^_c = I + s_c (G_c - I), and rows and columns rescaled by D^_x,c = I + s_c (D_x,c - I) and
  D^_y,c. Given the unpaired cells it is a fixed linear operator G_c(M) = D^_x,c^-1 G^_c M D^_y,c^-1, so by Minkowski,
  (E||G_c(theta~) - theta_c||^2)^(1/2) <= ||D^_x^-1 G^||_op ||D^_y^-1||_op R(B)^(1/2) + ||G_c(theta) - theta_c||.
  The second term, the map's own error, does not depend on B and is not controlled. It would vanish for the exact
  map Sigma_z,c Sigma_z^-1 of Proposition 2 with exact latent covariances, but the implemented map need not become
  exact even with exact inputs: with s_c = 1 its error under Proposition 2 is (Sigma_z - Sigma_z,c) P_-k W', where
  P_-k projects on the pooled eigendirections beyond the leading k, and the rescaling of latent correlations by
  measured standard deviations adds more (corrected in the ninth session; the earlier text said the error vanished).
* **What the bound does not establish.** It is an upper bound for this estimator. It gives no lower bound for other
  estimators, so it does not by itself explain why comparators need more paired cells, and it does not establish
  optimality (no minimax lower bound was derived). It states when unpaired bases save paired cells: dependence
  concentrated in leading unpaired eigen-blocks whose noise is small relative to the panel's. The generality
  benchmark (`../generality`, G4) tests this across data sets.

## Self-tuning variant (`selftune.py`, `DEV_SELFTUNE.md`)

V2 shrinks each coefficient of a block by tau^2 / (tau^2 + s_j), with s_j its own noise variance and
tau^2 = max(0, (||m||^2 - t)/d). With equal s_j it equals the fixed estimator's positive-part James-Stein factor, so
it differs only by preconditioning each coefficient by its noise. Its Stein unbiased risk estimate, exact for
Gaussian block means including the dependence of tau^2 on m, is
sum_j b_j^2 m_j^2 + 2 sum_j s_j (1 - b_j) + (4/d) sum_j m_j s_j (Sm)_j / (tau^2 + s_j)^2 - t with
b_j = s_j/(tau^2 + s_j). Choosing the block partition by SURE (V1, V3) overfitted at small budgets in development.

## Proposition 3 (near-optimality over dependence with given block norms; manuscript Proposition S12)

Idealized noise model: block means have spherical Gaussian noise, S_b = (tau_b / (d_b B)) I, independent across
blocks, d_b >= 4. Let P(r) be the priors on theta with E||a_b||^2 <= r_b^2 in every block. Then

    inf over estimators  sup over P(r)  Bayes risk  =  R*(B) = sum_b r_b^2 (tau_b/B) / (r_b^2 + tau_b/B),

and block James-Stein (which does not know r) has sup over P(r) of its Bayes risk <= R*(B) + 4 sum_b tau_b/(d_b B).

Proof. Lower bound: the Gaussian prior a_b ~ N(0, (r_b^2/d_b) I) is in P(r); its posterior mean has Bayes risk
R*(B), and no estimator does better under it. Attainment: the linear rule c_b X_b, c_b = r_b^2/(r_b^2 + tau_b/B),
has risk (1-c_b)^2 E||a_b||^2 + c_b^2 tau_b/B <= R* under every prior in P(r). James-Stein: t_b = tau_b/B and
l_b = tau_b/(d_b B), so t_b >= 4 l_b; Proposition 1 bounds its risk pointwise by
sum_b ||a_b||^2 t_b/(||a_b||^2 + t_b) + 4 l_b, concave and increasing in ||a_b||^2, so by Jensen its expectation
under any prior in P(r) is at most R*(B) + 4 sum_b l_b.

Consequences. (i) The paired budget any estimator needs to guarantee risk e over dependence with given block norms
solves R*(B) = e; block James-Stein needs at most the B solving R*(B) + 4 sum_b tau_b/(d_b B) = e. (ii) For one block
of all coefficients the budget halving the risk is tau/A (A = sum ||a_b||^2, tau = sum tau_b); the ratio of that to
the block-structured budget is the predicted saving of the structural law (`../generality/glaw.py`, G4). Scope: spherical
noise within blocks (per-cell products only approximate it); optimality relative to the unpaired block structure,
not against estimators exploiting other structure such as low rank.
