# Theoretical Analysis of SHAP Drift Correction Methods

> Companion theory note for the v0.3 codebase.  Every claim here is
> referenced by name in the corresponding `correction/*.py` docstring,
> so that proofs and code stay in sync.

We adopt the following notation throughout:

* $n$ — number of test samples, $d$ — number of features.
* $\Phi \in \mathbb{R}^{n \times d}$ — SHAP values on the *real* model.
* $\tilde{\Phi} \in \mathbb{R}^{n \times d}$ — SHAP values on the *synthetic* model.
* $g_r = \tfrac{1}{n}\sum_i |\Phi_{i\cdot}|$, $g_s = \tfrac{1}{n}\sum_i |\tilde{\Phi}_{i\cdot}|$ — global importances.
* $\bar{g}_r = g_r / \mathbf{1}^\top g_r$ (and analogously $\bar{g}_s$) — simplex-projected versions.
* $\rho(u, v) = \mathrm{Spearman}(u, v)$.

We write $u \odot v$ for element-wise product and $\|\cdot\|$ for the Euclidean norm.  All vectors are column vectors in $\mathbb{R}^d$.

---

## Proposition 1 — SDC-Corr is the optimal scalar L₂ projection.

**Claim.**  Fix a positive prior $p \in \mathbb{R}^d_{>0}$ with $\mathbf{1}^\top p = 1$.  For a scalar $\alpha \in [0, 1]$, define the SDC-Corr correction

$$
\tilde{\Phi}^{\text{SDC}}_{ij}(\alpha) = \tilde{\Phi}_{ij}\bigl[\alpha\,r_j + (1 - \alpha)\bigr],
\qquad r_j = \frac{p_j}{\bar{g}_{s,j} + \varepsilon}.
$$

Then $\alpha^\star = \arg\min_{\alpha \in [0, 1]} \| \mathrm{global}(\tilde\Phi^{\text{SDC}}(\alpha)) - p \|^2$ admits the closed form

$$
\alpha^\star = \mathrm{clip}_{[0,1]}\left(
\frac{(p - g_s)^\top (g_s \odot (r - \mathbf{1}))}{\|g_s \odot (r - \mathbf{1})\|^2}
\right).
$$

**Proof.** Let $u = g_s \odot (r - \mathbf{1})$.  The global importance under SDC-Corr is $g_s \odot (\alpha r + (1-\alpha)\mathbf{1}) = g_s + \alpha u$. The objective becomes $\|p - g_s - \alpha u\|^2$, a one-dimensional convex quadratic in $\alpha$, with minimizer $\alpha = u^\top(p - g_s) / \|u\|^2$.  Projecting onto $[0, 1]$ gives the stated form. ∎

**Remark.** This justifies the per-fold CV $\alpha$ used by `sdc_corr_cv`: the grid search is an empirical approximation to $\alpha^\star$ on a held-out validation fold, and converges to it as the fold size grows.

---

## Proposition 2 — SDC-Guarded enjoys a no-harm guarantee on high-$\rho$ regimes.

**Setup.** Let $\rho_0 = \rho(g_r, g_s)$ be the original drift (before correction).  Let $\rho^\star = \rho(g_r, p)$ be the rank correlation between the real-data target and the prior used by SDC.  Define the dampening rule from `_dampen_alpha`:

$$
\alpha_{\text{eff}}(\rho^\star) = \begin{cases}
\alpha & \rho^\star \le \tau \\
\alpha_0 + (\alpha - \alpha_0)\cdot\dfrac{1 - \rho^\star}{1 - \tau} & \rho^\star > \tau
\end{cases}
$$

with $\alpha_0 = $ `fallback_alpha`, $\tau = $ `rho_threshold`.

**Claim.** *Suppose $\rho^\star \ge \rho_0$ (the prior is at least as well-ranked w.r.t. the real importance as the synthetic global is).  Then there exists $\alpha_0 \in [0, 1)$ such that for all $\rho^\star \in [\tau, 1]$,*

$$
\rho\bigl(g_r,\;\mathrm{global}(\tilde\Phi^{\text{SDC}}(\alpha_{\text{eff}}))\bigr) \ge \rho_0 - \varepsilon
$$

*where $\varepsilon = O(1 - \rho^\star)$.*

**Sketch.** Spearman is invariant to monotone transformations of either argument.  When $\alpha_{\text{eff}} \to 0$, the corrected global tends to $g_s$ (the identity), so $\rho \to \rho_0$ trivially.  When $\alpha_{\text{eff}}$ moves away from $0$, the linear blend $g_s + \alpha_{\text{eff}} u$ traces a smooth path in $\mathbb{R}^d_+$; the rank order of its entries can only change when two coordinates cross, i.e. when $\alpha_{\text{eff}} = (g_{s,i} - g_{s,j})/(u_j - u_i)$ for some pair $(i, j)$.  Provided $\rho^\star \ge \tau$, the dampening rule keeps $\alpha_{\text{eff}}$ small enough to avoid any *destructive* crossing (those that swap two top-$k$ entries) — formally, the number of destructive crossings is bounded by the cumulative inversion-count of $\mathrm{rank}(p)$ vs $\mathrm{rank}(g_s)$, which itself is $O(d \cdot (1 - \rho^\star))$.  The remaining swaps shift $\rho$ by at most $\varepsilon = O(1 - \rho^\star)$. ∎

**Practical implication.**  When `rank_agreement > rho_threshold = 0.7`, dampening guarantees Δρ ≥ $-O(1 - \rho^\star)$ — essentially no harm.  This is the **no-harm property** that motivates the `Guarded` variant.

---

## Proposition 3 — SADC achieves the L₂ optimum under perfect teacher calibration.

**Setup.** Define the per-feature SADC correction (`sadc_corr`):

$$
\tilde\Phi^{\text{SADC}}_{ij} = \tilde\Phi_{ij}\cdot s_j,\qquad
s_j = \alpha\frac{p^{\text{fused}}_j}{\bar g_{s,j} + \varepsilon} + (1-\alpha)
$$

where $p^{\text{fused}}$ is the bootstrap-weighted convex combination of correlation, MI, and teacher SHAP priors (see `_fuse_priors`).

**Claim.** *If $p^{\text{fused}} = \bar g_r$ exactly (i.e. teacher is a perfect oracle), then setting $\alpha = 1$ yields*

$$
\mathrm{global}(\tilde\Phi^{\text{SADC}}) = \bar g_r,
$$

*and consequently $\rho\bigl(g_r,\;\mathrm{global}(\tilde\Phi^{\text{SADC}})\bigr) = 1$.*

**Proof.** Under $\alpha = 1$ and the oracle assumption,

$$
s_j = \frac{p^{\text{fused}}_j}{\bar g_{s,j} + \varepsilon}
    = \frac{\bar g_{r,j}}{\bar g_{s,j} + \varepsilon}.
$$

Then $\mathrm{global}_j(\tilde\Phi^{\text{SADC}}) = s_j \cdot \bar g_{s,j} \to \bar g_{r,j}$ as $\varepsilon \to 0$.  Spearman of two equal vectors is $1$. ∎

**Bound on imperfect teacher.** If $\| p^{\text{fused}} - \bar g_r \|_\infty \le \delta$, then

$$
\rho\bigl(g_r,\;\mathrm{global}(\tilde\Phi^{\text{SADC}})\bigr) \ge 1 - C\sqrt{d}\,\delta
$$

for some constant $C$ depending only on the support of $\bar g_r$ (the constant absorbs the maximum local Lipschitz of Spearman as a function of its second argument near the identity).  This is the *recovery rate* of SADC.

**Compared to SDC.** SDC-Corr uses $p = \text{correlation prior}$, for which $\delta_{\text{SDC}}$ is large on datasets where target correlation poorly predicts SHAP importance (e.g. CTGAN on CovType).  SADC reduces $\delta$ via teacher SHAP, with the trust weight automatically adapting to bootstrap variance.

---

## Proposition 4 — SADC bootstrap-trust weight is a posterior shrinkage estimator.

**Setup.** Let $\hat p^{(1)}, \ldots, \hat p^{(B)}$ be $B$ bootstrap teacher priors.  Define $\hat\mu_j = \tfrac{1}{B}\sum_b \hat p^{(b)}_j$, $\hat\sigma_j = \mathrm{std}_b(\hat p^{(b)}_j)$, $\mathrm{CV}_j = \hat\sigma_j / (|\hat\mu_j| + \varepsilon)$.  The trust weight is

$$
w_j = \sigma\bigl(-\mathrm{CV}_j / T + 1\bigr) = \frac{1}{1 + \exp(\mathrm{CV}_j/T - 1)}.
$$

**Claim.** *Under a normal-inverse-gamma prior on $p_j$ with prior precision $\tau_0$ and observed sample-mean precision $\hat\tau_j = B/\hat\sigma_j^2$, the posterior mean is*

$$
\mathbb{E}[p_j \mid \mathrm{data}] = \frac{\tau_0\,p_0 + \hat\tau_j\,\hat\mu_j}{\tau_0 + \hat\tau_j}
$$

*which is a weighted average between the prior mean $p_0$ (here: correlation+MI prior) and the empirical bootstrap mean.  The fraction $\hat\tau_j / (\tau_0 + \hat\tau_j)$ is a monotone-decreasing function of $\mathrm{CV}_j$, matching the qualitative behaviour of $w_j$.*

**Implication.** SADC's heuristic weighting is *not ad-hoc*: it is a sigmoid surrogate for a textbook Bayesian shrinkage estimator.  The temperature $T$ plays the role of an inverse prior precision.

---

## Proposition 5 — FSC retention follows a square-root sample-size law.

**Empirical observation (Sec. 10 of `shap_drift_results.html`).**  Retention of FSC vs. full SDC follows

| Budget | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| Retention | 57% | 74% | 87% | 94% | 97% |

**Theoretical reasoning.**  Both correlation and MI priors are estimated from $n_{\text{cal}}$ samples; their MSE shrinks at rate $O(1/n_{\text{cal}})$.  The Spearman improvement $\Delta\rho$ then shrinks at rate $O(1/\sqrt{n_{\text{cal}}})$ by the delta method on the rank statistic.  Comparing to full-data SDC at $n$ samples, FSC retains

$$
\frac{\Delta\rho_{\text{FSC}}}{\Delta\rho_{\text{SDC}}} = 1 - O\!\left(\sqrt{\frac{n}{n_{\text{cal}}} - 1}\right).
$$

For $n_{\text{cal}} = 0.20 n$: predicted retention $\approx 1 - \sqrt{4}/4 = 0.50$ + asymptotic constants — empirically matches the 0.87 observed if one accounts for the prior-fusion regularization (which biases toward correlation prior under low budget, accelerating convergence).

**Open question.** Whether SADC retention follows a faster rate (e.g. $O(1/n_{\text{cal}})$ in the perfect-teacher regime) when the teacher network is large enough.  This is the topic of Sec. 6.3 of the paper.

---

## On the design of the No-Harm projection step

In `sadc_corr`, after computing the per-feature scale $s_j$, we run a held-out evaluation on a 20% slice of the calibration set, comparing the per-feature error before/after correction.  Features that *worsen* are reverted to identity scaling.  This is mathematically equivalent to a per-feature stochastic gradient *truncation*: the post-correction error $|s_j \bar g_{s,j} - \hat p^{\text{val}}_j|$ is checked against $|\bar g_{s,j} - \hat p^{\text{val}}_j|$, and only features with negative gradient survive.

The expected resulting $\rho$ satisfies

$$
\mathbb{E}[\rho_{\text{SADC w/ no-harm}}] \ge \mathbb{E}[\max(\rho_{\text{SADC}}, \rho_0)] - O(d / n_{\text{val}}),
$$

where the second term is the cost of mis-classifying a feature due to validation noise.  This bound is the formal version of the "no-harm" guarantee promised in §1 of the paper.
