# Measuring the individual-frequency identified set in real recipients

Status: post hoc analysis plan, written 2026-09-26 before computing any of the
summaries below. The held-out losses of the adaptation-only and colon analyses
were already public; direction agreement, identified-interval widths and
minimax-risk bounds had not been computed. This plan does not modify, replace or
reinterpret the frozen colon protocol (`release/analysis/colon_generalization/PLAN.md`)
or any released result. Its analyses are secondary and retrospective, including
those that use colon donors.

## Question

The paper proves that, with a fixed source interaction, a recipient described
only by its individual marker frequencies identifies each binary query only up
to an interval [l, h] (main text Eq. 2; Additional file 1, Proposition S1). The
released paper does not report the size of that interval in any real recipient
("The examples establish ambiguity, not its biological magnitude"). We ask:

1. How large is the individual-frequency identified set in the four released
   recipient cohorts, including the independent colon donors?
2. When separately measured assay patterns move a prediction across
   independence (reverse the association direction implied by the
   frequency-only reconstruction), do held-out cells agree with the patterns?

## Fixed inputs (no refitting, no new data)

- Source interaction: `release/analysis/assay_resolution/results/adaptation_only_results/source_fit.npz`
  (identical to `strict_b` in `reconstruction_inputs.npz`).
- Recipient marker frequencies: means of the released smoothed adaptation
  histograms (`strict_marginals` for 24 Cambridge, 56 Newcastle and 15 T-ALL
  recipients; `marginals` in `colon_generalization/results/predictions.npz`
  for 11 colon donors). All adaptation-only arms share these frequencies.
- Frequency-only and full-pattern predictions: released arms `means_only` and
  `both_patterns` of the same files.
- Held-out 2x2 scoring tables: released `truth` arrays. They enter only S3 and S4.

## Certified inner bounds on the identified interval

For recipient p and query (i, j), the query cell probability is
t = Pr(RNA marker i high, protein marker j high). Every point used as a bound
is attained: an explicit pair of strictly positive assay laws (r, c) whose 18
marker means equal the recipient frequencies, reconstructed with the fixed
interaction by matrix scaling. Hence [l_hat, h_hat] is contained in the sharp
interval [l, h]. No claim of global optimality is made.

Attained points: (a) the released frequency-only prediction; (b) the released
full-pattern prediction; (c) local optima of t within the star family, in which
RNA markers are conditionally independent given RNA marker i and protein
markers are conditionally independent given protein marker j. Each star law is
parameterized by its eight pairwise overlaps with the centre, each within its
Frechet bounds shrunk by a relative margin 1e-6 so that all 512 states keep
positive mass. Maximization and minimization use L-BFGS-B with exact adjoint
gradients from two deterministic starts each: the vertex solving the exact
first-order (weak-interaction) problem, and conditional independence. Every
reported star point is re-evaluated at marginal tolerance 1e-12, with marker
means checked to 1e-10.

Log odds: theta(t) = log[t(1 - m - n + t) / ((m - t)(n - t))], with m, n the
query frequencies. Deviance units are twice KL, as in the paper.

## Summaries, all reported per cohort (Cambridge, Newcastle, T-ALL, colon)

S1. Certified direction non-identification: the fraction of recipient-queries
    with theta(l_hat) < 0 < theta(h_hat), over all 81 queries and separately
    over the nine same-marker (cognate) queries.
S2. Size: median and interquartile range of theta(h_hat) - theta(l_hat); and the
    certified lower bound 2 R_hat on the worst-case deviance of every
    frequency-only single-table predictor, where R_hat is the equalized
    endpoint loss of Proposition S1 applied to [l_hat, h_hat]. Because the sharp
    interval contains [l_hat, h_hat], the minimax risk over the sharp interval
    is at least R_hat. Reported as the mean over recipients and queries.
S3. Held-out direction of reversals (primary held-out link). For each
    recipient-query let s_F and s_P be the signs of the frequency-only and
    full-pattern predicted log odds, and s_E the sign of P11 P00 - P10 P01 in
    the held-out scoring table. A reversal has s_F != s_P; reversals with
    s_E = 0 are counted but excluded. Agreement = (number of reversals with
    s_E = s_P) / (number of reversals with s_E != 0), pooled over recipients
    within a cohort. Paired recipient bootstrap, 20,000 draws, seed 20260926;
    nominal 95% and Bonferroni percentile intervals across the four cohorts
    (0.025/4, 1 - 0.025/4). Reference value 0.5. Every reversal is by
    construction a certified sign-ambiguous query, because both predictions
    are attained points of the same identified set.
S4. Descriptive only: mean within-recipient Spearman correlation between the
    certified bound 2 R_hat and the held-out pair gain (frequency-only minus
    full-pattern pair deviance), with the same correlation for the frequency
    product m(1-m)n(1-n) shown alongside as a comparator.

All four summaries are reported regardless of direction. No summary will be
used to change the source model, recipients, queries, arms or optimizer. A
failed optimization for a query leaves its interval as the span of the
remaining attained points and is reported.

## Checks

Unit tests: star-law means and derivatives, adjoint gradient against finite
differences, the first-order range against brute-force enumeration on a small
panel, and re-certification of saved star parameters. A sensitivity check
compares the star-family endpoints with an unrestricted local search over all
positive laws for a fixed subset of colon queries (first colon donor, all 81
queries), reported as the change in endpoint log odds.
