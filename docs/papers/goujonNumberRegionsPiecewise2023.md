---
citekey: goujonNumberRegionsPiecewise2023
title: "On the Number of Regions of Piecewise Linear Neural Networks"
authors: [Alexis Goujon, Arian Etemadi, Michael Unser]
year: 2023
venue: arXiv preprint
doi: "10.48550/arXiv.2206.08615"
arxiv: "2206.08615"
url: null
kind: theory
source_type: pdf
source_sha256: d8e866d3f9b076b8f6ae935faec3cee5e627611713243bc71369734b48df264b
pages: 29
ingested: 2026-10-15
tags: [expressivity, linear-regions, piecewise-linear-networks, combinatorial-geometry, activation-functions]
concepts_introduced:
  - arrangement of convex partitions
  - convex linear region
  - projection region
  - activation complexity
  - knot density along a 1D path
  - compositional bound on the maximal number of linear regions
  - expected knot density of a CPWL network
concepts_used:
  - continuous piecewise-linear function
  - ReLU network
  - maxout network
  - hyperplane arrangement
  - Zaslavsky's theorem
  - convex partition
  - depth-width expressiveness
builds_on: [NIPS2014_fa6f2a46, serraBoundingCountingLinear2018, haninComplexityLinearRegions2019, haninDeepReLUNetworks2019, zaslavskyFacingArrangementsFacecount1975]
datasets: []
---

# On the Number of Regions of Piecewise Linear Neural Networks

**Goujon, Etemadi, Unser (2023)**, *arXiv preprint*. [arXiv:2206.08615](https://arxiv.org/abs/2206.08615)

!!! abstract "TL;DR"
    The number of linear regions of a network is currently bounded only for specific activations (ReLU, Maxout). This paper replaces the arrangement of hyperplanes with an *arrangement of convex partitions*, obtaining tight upper and lower bounds on the maximal number of convex linear regions of any continuous piecewise-linear (CPWL) network in terms of depth $L$, layer dimensions, and the regions of each activation. A second, stochastic framework bounds the *expected knot density* along any 1D path by $D_0^\ast(\lambda_0 W L)$: on average depth buys only a linear factor, so depth, width and activation complexity play identical roles.

## Problem and motivation

A feedforward network whose activations are CPWL produces a CPWL mapping, and hence partitions its input domain into regions on which the map is affine (§1). The number of these linear regions is a natural expressiveness metric, but it is architecture-specific: sharp bounds existed only for ReLU networks [@NIPS2014_fa6f2a46; @serraBoundingCountingLinear2018] and, via tropical geometry, for Maxout networks (§1, Remark 1). The goal is to give *general* bounds for arbitrary, possibly multivariate, CPWL activations — Maxout, GroupSort, deepsplines, PWLU, pooling — and to decide how the three "sources of complexity" (Figure 1) actually contribute: depth $L$, width (layer dimensions $d_1,\dots,d_{L+1}$), and activation complexity $\kappa$ (the number of linear regions of the activation). Two questions drive the paper: what is the *maximal* number of regions attainable by an architecture (§3), and what does a *typical*, randomly parameterised network produce (§4).

The paper notes the metric is ambiguous in the literature (projection regions vs. convex regions, §2.2.1–2.2.3) and fixes the convex-region convention, since most parameterisations (hat-basis splines, hinging hyperplanes, ReLU/Maxout compositions) have convex regions.

## Method

The technical core is a generalisation of hyperplane arrangements. A hyperplane arrangement is too narrow because only a small fraction of convex partitions arise from hyperplanes (§2.3), so the paper defines the **arrangement of convex partitions**: given convex partitions $\Pi_1,\dots,\Pi_N$, its regions are the non-empty intersections $A_{m_1,\dots,m_N} = \bigcap_{k=1}^N P^{(k)}_{m_k}$ (Definition 5, §2.3). A second ingredient is the **knot density** along a path: for a CPWL path $\gamma$ and function $f$, the characteristic function $\varphi_f^\gamma(t) = (\mathbb{1}_{\Omega_1}(\gamma(t)),\dots,\mathbb{1}_{\Omega_K}(\gamma(t)))$ records which projection region supports the active affine piece; the number $kt_f^\gamma$ of its jumps is the number of knots, and $\lambda_f^\gamma = kt_f^\gamma/\mathrm{Len}(\gamma)$ (§4.1, Definition 6). This is computable from $f$ and $\nabla f$ in any deep-learning library.

Three steps build the results. (a) Theorem 1 bounds the cardinality of an arrangement of convex partitions by a polynomial of degree $\min(d,N)$ — the dimension of the ambient space caps the growth. (b) Proposition 2 applies this to the elementary operations of the CPWL function space (sum, vectorisation), with matching lower bounds; composition is then handled layer by layer via the projection/preimage Lemmas 1–3 (§3.2–3.3, App. A). (c) The stochastic part bounds the expectation of the knot density for sums, vectorisations (§4.1, Propositions 3–4), for single layers (Propositions 5–7), and finally across depth (Theorem 3, Corollary 3). The lower bound of Theorem 2 is realised by a *deepspline* construction built from 1D sawtooth functions $sw_p$ with $sw_p\circ sw_q = sw_{pq}$ (App. A.4); this is why the lower bound applies to pointwise activations only (§3.4). All proofs are in Appendices A–D.

## Key equations and definitions

The upper bound on the cardinality of an arrangement of $N$ convex partitions with $n_1,\dots,n_N$ regions each (§3.1, Eq. 4) is the generalisation of the hyperplane bound (Eq. 4):

$$
\beta_d(n_1,\dots,n_N) = 1 + \sum_{k=1}^{\min(d,N)}\ \sum_{1\le \ell_1<\dots<\ell_k\le N}\ \prod_{q=1}^{k}\bigl(n_{\ell_q}-1\bigr)
$$

Setting $n_{\ell}=2$ recovers $\sum_{k=0}^{\min(d,N)}\binom{N}{k}$, the classical number of regions of $N$ hyperplanes in general position. The matching operation bounds for sums and vectorisation (Proposition 2, §3.2, Eqs. 6–7) read

$$
\kappa_{f_1+\dots+f_N} \le \beta_d(\kappa_1,\dots,\kappa_N), \qquad \kappa_{(f_1,\dots,f_N)} \le \beta_d(\kappa_1,\dots,\kappa_N),
$$

and both are sharp. Composition (Theorem 2, §3.3, Eq. 15) multiplies these layer-wise (Eq. 15):

$$
\prod_{\ell=1}^{L}\alpha_{\min(d_1,\dots,d_{L+1})}(\kappa_{\ell,\cdot}) \ \le\ \kappa_{\max}\ \le\ \prod_{\ell=1}^{L}\beta_{\min(d_1,\dots,d_\ell)}(\kappa_{\ell,1},\dots,\kappa_{\ell,d_{\ell+1}}),
$$

where the lower factor $\alpha$ is a max over assignments $\tau$ of the $d_{\ell+1}$ outputs to the $\min(d_1,\dots,d_{L+1})$ bottleneck coordinates (Eq. 16). For a network with dimensions $(d_{\rm in}, W,\dots,W, d_{\rm out})$ and uniform activation complexity $\kappa$, Corollary 1 gives the readable form with $d^\ast = \min(d_{\rm in},d_{\rm out})$ (Eq. 17):

$$
\bigl(\kappa^{\lfloor W/d^\ast\rfloor}\bigr)^{L d^\ast} \ \le\ \kappa_{\max}\ \le\ (\kappa W)^{L d^\ast}.
$$

The knot density itself (§4.1, Definition 6, Eq. 24) and the two stochastic bounds (§4.3, Eqs. 34–35) are

$$
\lambda_f^\gamma = \frac{kt_f^\gamma}{\mathrm{Len}(\gamma)},
$$

$$
\E[\lambda^\gamma_{f_{\theta_L}\circ\dots\circ f_{\theta_1}}] \le \lambda_0 W\,\frac{1-D_0^{L}}{1-D_0}\ (D_0\ne 1), \qquad \E[\lambda] \le \lambda_0 W L\ (D_0 = 1),
$$

$$
\E[\lambda^\gamma_{f_{\theta_L}\circ\dots\circ f_{\theta_1}}] \le D_0^\ast\,(\lambda_0 W L), \qquad D_0^\ast = \max(D_0,1).
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $f:\R^d\to\R^{d'}$ | CPWL function |
| $\Omega_k$ | projection region (support of affine piece $f_k$) |
| $\kappa_f$ | number of convex linear regions of $f$ (Definition 4) |
| $\Pi=(P_1,\dots,P_n)$ | convex $n$-partition of $\R^d$ |
| $\beta_d(n_1,\dots,n_N)$ | max cardinality of arrangement of $N$ convex partitions |
| $L$ | depth (number of composed layers) |
| $d_\ell$ | dimension of layer $\ell$'s input |
| $\kappa_{\ell,k}$ | activation complexity: convex regions of component $k$ at layer $\ell$ |
| $d^\ast$ | $\min(d_{\rm in},d_{\rm out})$ |
| $\gamma:S\to\R^d$ | 1D CPWL path, $\mathrm{Len}(\gamma)$ its length |
| $\lambda_f^\gamma$ | knot density of $f$ along $\gamma$ |
| $\lambda_0$ | bound on expected knot density of a layer (Assumption (i)) |
| $D_0$, $D_0^\ast$ | bound on expected directional derivative (Assumption (ii)), $\max(D_0,1)$ |
| $W$ | width (shared layer dimension in Corollary 3) |

## Main results

**Maximum number of regions (§3).** Depth is the only descriptor that can grow the *maximal* region count exponentially: Eq. 15 has depth as a product of $L$ factors, while width and activation complexity enter polynomially (degree $\min(d_1,\dots,d_\ell)$ per layer). Corollary 1 makes this explicit for a fixed input/output dimension with width $W$: the count is between $(\kappa^{\lfloor W/d^\ast\rfloor})^{Ld^\ast}$ and $(\kappa W)^{Ld^\ast}$ (Eq. 17). Corollary 2 extends both bounds to *projection* regions and to the number of distinct affine pieces. Applying Eq. 15 to concrete architectures (§3.4): for ReLU it reproduces the bound of [@NIPS2014_fa6f2a46] (Eq. 18), which is *not* tightest for ReLU — the image of $(-\infty,0]$ under ReLU is the singleton $\{0\}$, reducing the effective dimension — but is well suited to PReLU/Leaky-ReLU; Maxout (Eq. 20) and deepspline (Eq. 19) share the same polynomial factor $\bigl(\sum_{k=0}^{\min(d_1,\dots,d_\ell)}\binom{d_{\ell+1}}{k}\bigr)(\kappa-1)^k$, improving on [@serraBoundingCountingLinear2018]; the Maxout bound of Eq. 20 is claimed sharper than plugging $d_\ell=d$ into that earlier bound. GroupSort with group size $g_s$ gives Eq. 21, and the sort/GroupSort activation has $\kappa_{\rm GS}=(g_s!)^{n_g}$ regions with $(g_s/2)^{d/2}\le\kappa_{\rm GS}\le g_s^{\,d}$ (Eq. 11). The 2D PWLU with $M^2$ control points has exactly $2(M-1)^2$ triangular regions, giving Eq. 12–14.

**Average number of regions (§4).** Under two assumptions — (i) the expected knot density of each component is bounded by $\lambda_0$, and (ii) the expected directional derivative within the composition is bounded by $D_0$ — Theorem 3 and Corollary 3 bound the expected knot density along any 1D path by $O(\lambda_0 W L)$ (Eqs. 34–35): no exponential growth with depth on average. Assumption (i) is verified for ReLU, Maxout and GroupSort layers with random weights and biases: Proposition 5 gives $\E[\lambda^{\rm ReLU}]\le \sqrt{\E[w_1^2]}\sup_t\rho_b(t)$, tightening to $\sqrt{\sigma_w/(\pi\sigma_b)}$ (sic; printed, Eqs. 28–29) for Gaussians; Proposition 6 gives a quadratic-in-$K$ bound $\sqrt{K/2}\,\sigma_w\sup_t\rho_b(t)$ for rank-$K$ Maxout (Eq. 30); Proposition 7 gives $\sqrt{d}(g_s-1)\sigma_w\sup_t\rho_b(t)$ for GroupSort (Eq. 32). Assumption (ii) is defended by citing that deep ReLU networks preserve expected length in a realistic setup (§4.3.1). Figure 7 shows the region-splitting process saturating with depth for an absolute-value network (the region counts are printed inside the figure, which could not be read from the extracted text).

## Limitations stated by the authors

Independence of the randomly generated layers holds at initialisation but not during learning (§4.3.1) — the stochastic analysis therefore describes the whole function space an architecture offers, not trained networks. The lower bound of Theorem 2 applies only to CPWL NNs with pointwise activations, because its construction is a deepspline network (§3.4). The Maxout knot-density bound is quadratic in the number of units, and the authors "conjecture the existence of a tighter linear bound" (§4.2). Theorem 3 is "more natural at first sight" but Corollary 3's assumption is "closer to practical observations" (§4.3.1) — the two are not equivalent. The sharpness discussion of the compositional bounds is conditional: they believe the bounds "are sharp provided that the expected knot density is uniform … and that the expected norm of the directional derivative is uniform and isotropic" (§4.3.1).

## Critical assessment

The tightness claims for the *maximal* count are backed by explicit constructions (parallel hyperplanes per partition for Theorem 1; sawtooth compositions for Theorem 2), so the upper and lower bounds genuinely bracket the truth, and the paper states where they fail to match: Eq. 18 is deliberately *not* the tightest ReLU bound and the authors say so. The stochastic half is weaker: the expected knot density along a 1D path is a proxy for the expected region count, not the count itself, and the authors admit the full expectation "seems unfortunately very complex". The step from "expected path density" to "average number of regions" is intuition plus Figure 7, not a theorem. Assumption (ii) is a real modelling choice — it is exactly the "expected length grows with depth" question that separates pessimistic and optimistic views of depth (the paper's [43], outside the collection), so the headline "no exponential growth on average" inherits whatever that assumption hides, and the substitution of an unreadable figure for numbers means the average-case claim has no measured constant attached. The deepspline-only lower bound means the "depth and width and complexity contribute equally on average" conclusion, while stated for general CPWL networks, is empirically supported only through the bounds' prefactors. The manuscript is a December-2023 arXiv v2 with no peer-reviewed venue recorded in the entry. No code or data are released; none are needed for the theory. Open questions it raises: a tight linear Maxout density bound, and whether the compositional bounds are sharp without the uniformity assumption on knot density and gradients.

## Connections within the collection

It directly generalises the depth/width-dependent maximal-region bound of [@NIPS2014_fa6f2a46] (its reference [6]) and the ReLU count bound of [@serraBoundingCountingLinear2018] (its [27]); it recovers the former as Eq. 18 and improves on the latter for Maxout (Eq. 20). Its stochastic section extends the average-case analyses of [@haninDeepReLUNetworks2019] (few activation patterns) and [@haninComplexityLinearRegions2019] (complexity of linear regions) to generic CPWL activations. The proof of Theorem 1 uses the arrangement-of-hyperplanes counting result of [@zaslavskyFacingArrangementsFacecount1975] (its [37]) and Whitney's formula. It refers to the early expectation that path length grows exponentially with depth, due to [@raghuExpressivePowerDeep2017] (its [42]), and sides with the finding that ReLU networks preserve expected length instead.

It is closely related to other region-counting work in the collection: the polyhedral-decomposition and combinatorial-structure results of [@masdenAlgorithmicDeterminationCombinatorial2022], [@berzinsPolyhedralComplexExtraction2023] and [@liuReLUNeuralNetworks2023] study how the regions are cut out rather than how many there can be; [@brandenburgDecompositionPolyhedraPiecewise2024] studies the polyhedra of piecewise-linear functions more directly. The tropical-geometry route to region counts and decision boundaries appears in [@alfarraDecisionBoundariesNeural2023] and [@zhangTropicalGeometryDeep], which complement the combinatorial-geometry route taken here. The spline view of deep learning of [@balestrieroSplineTheoryDeep2018] is the conceptual framing of the projection-region picture used in §2.2.1, and expressivity/region-counting issues in convex and rational ReLU architectures are studied in [@gagneuxHowImproveExpressivity2025] and [@averkovEXPRESSIVENESSRATIONALRELU2025]. The paper's conclusion — that complex learnable activations are an alternative to depth under Lipschitz constraints — connects to the Lipschitz-controlled deep-spline line of work (its [45], [46], outside the collection). A sharp Maxout bound via tropical geometry, by Montúfar, Ren and Zhang (its [30]), appeared independently and is discussed in Remark 1; it is outside the collection, but shares a co-author with [@NIPS2014_fa6f2a46].
