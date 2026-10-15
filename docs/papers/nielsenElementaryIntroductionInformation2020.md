---
citekey: nielsenElementaryIntroductionInformation2020
title: "An Elementary Introduction to Information Geometry"
authors: [Frank Nielsen]
year: 2020
venue: Entropy 22(10), 1100
doi: "10.3390/e22101100"
arxiv: null
url: null
kind: tutorial
source_type: pdf
source_sha256: 218b7e31ee693f21a360cd575a6b3b152023b62f15e355ad9e8f479a34aaaea6
pages: 61
ingested: 2026-10-15
tags: [information-geometry, statistical-manifolds, fisher-information, dual-connections, bregman-divergence]
concepts_introduced:
  - information manifold
  - conjugate connection manifold
  - statistical manifold
  - Amari–Chentsov cubic tensor
  - dually flat manifold
  - Bregman divergence
  - Fisher information metric
  - α-connections
  - natural gradient descent
  - Fisher–Rao distance
concepts_used:
  - differential geometry
  - affine connection
  - Riemannian metric
  - Legendre–Fenchel duality
  - exponential family
  - mixture family
  - f-divergence
  - Kullback–Leibler divergence
  - information monotonicity
builds_on: []
datasets: []
---

# An Elementary Introduction to Information Geometry

**Nielsen (2020)**, *Entropy* 22(10), 1100. [doi:10.3390/e22101100](https://doi.org/10.3390/e22101100)

!!! abstract "TL;DR"
    A self-contained tutorial on the differential-geometric core of information geometry: a manifold
    $(M, g, \nabla, \nabla^*)$ equipped with a metric and a pair of conjugate (metric-coupled) affine
    connections. It builds the dualistic structure up from scratch, shows how it collapses to Riemannian
    geometry for self-dual Levi–Civita connections, how divergences and strictly convex generators give
    rise to dually flat (Bregman) geometries, and how the Fisher information metric is the unique
    statistically invariant one. Four worked applications—natural gradient, Bregman mirror descent,
    Bayesian hypothesis testing and mixture clustering—show what the machinery buys.

## Problem and motivation

The paper asks how to put a geometry on a family of models (or on any space of "information"), why one
geometric structure should be preferred over another, and what such a structure is good for (§1.1, p. 1).
Nielsen frames information sciences as the study of the "communication" between data and model families,
and information geometry (IG) as the differential geometry of that process, with model inference read as
a statistical decision problem (§1.1, p. 1). The gap the survey targets is pedagogical: the dualistic
structures of IG are usually met only in advanced textbooks, so the exposition introduces the necessary
differential geometry from zero, states the central definitions and theorems, and omits proofs (§1.2,
p. 2). Familiar distances are not enough because a metric distance (Riemannian geodesic length) and a
divergence (a smooth, generally asymmetric "distance") are different objects: a Riemannian distance is
never a divergence, while a squared Riemannian distance always is (§3.6, p. 15). The setting is a smooth
$D$-dimensional manifold $M$ handled in a single chart, with a metric tensor field $g$ and an affine
connection $\nabla$ treated as two independent structures (§2.1, p. 3).

## Method

The survey is organised as a three-part tutorial (§1.2, p. 2). **Part 1** (Section 2) defines a manifold
$(M, g, \nabla)$: $g$ gives an inner product on each tangent plane (lengths, angles, a Mahalanobis
metric on $T_p$, §2.2, pp. 4–6), while $\nabla$ gives covariant derivatives, parallel transport,
$\nabla$-geodesics as autoparallel curves, and curvature/torsion (§2.3, pp. 6–9). The fundamental theorem
of Riemannian geometry supplies the unique torsion-free metric-compatible Levi–Civita connection
$^{LC}\nabla$ (§2.4, p. 11).

**Part 2** (Section 3) is the heart. It introduces conjugate connection manifolds (CCMs) $(M, g, \nabla,
\nabla^*)$ where the two connections are coupled to $g$ so that dual parallel transport preserves the
metric (Def. 1–2, Property 1–2, §3.2, pp. 12–13); a CCM converts to a statistical manifold $(M, g, C)$
via the totally symmetric Amari–Chentsov cubic tensor $C$ (§3.3, p. 13), and any pair $(\nabla,
\nabla^*)$ generates a one-parameter family of $\alpha$-connections $(M, g, \nabla^{-\alpha},
\nabla^{\alpha})$ (§3.4, p. 14). Two constructions produce the initial conjugate pair: (i) from a
divergence $D$, following Eguchi, via its second and third derivatives at the diagonal (§3.6, pp. 15–16),
and (ii) from a regular parametric family $P = \{p_\theta\}$, giving the expected exponential and
mixture connections $^e\nabla, {}^m\nabla$ coupled to the Fisher information metric (§3.9, pp. 20–24).
When $D$ is a Bregman divergence of a strictly convex generator $F$, the result is a **dually flat
manifold** with global dual affine coordinates $\eta = \nabla F(\theta)$, $\theta = \nabla F^*(\eta)$
and Pythagorean theorems (§3.7, pp. 16–20). Section 3.10 asks which structures are statistically
*meaningful*: invariance under Markov embeddings singles out the Fisher metric (Thm. 8, p. 25) and
information monotonicity singles out the separable $f$-divergences (§3.10, pp. 24–27).

**Part 3** (Section 4) applies the machinery: natural gradient descent and its equivalence to Bregman
mirror descent and ordinary gradient on dually parameterised functions (§4.1, pp. 36–41), Chernoff
information as the best error exponent of Bayesian hypothesis testing on the exponential-family manifold
(§4.3, pp. 41–43), and Bregman $k$-means clustering of mixtures with prescribed components on the
mixture-family manifold (§4.4, pp. 43–44).

## Key equations and definitions

The metric tensor and the divergence-induced structure:

$$
g_{ij} := \langle e_i, e_j\rangle, \qquad
^D g_{ij} := \partial_i \partial_j' D(\theta : \theta')\big|_{\theta=\theta'} \quad \text{(Eqs. 4, 56, p. 15)}
$$

Conjugate connections are defined by the identity that makes dual parallel transport metric-preserving
(Def. 1, Eq. 40, p. 12):

$$
X\langle Y, Z\rangle = \langle \nabla_X Y, Z\rangle + \langle Y, \nabla^*_X Z\rangle,
\qquad \forall X, Y, Z \in \mathfrak{X}(M).
$$

The one-parameter family of $\alpha$-connections is generated by scaling the cubic tensor (§3.4,
Eqs. 50–52, p. 14):

$$
^\alpha\Gamma_{ij,k} = \Gamma^0_{ij,k} - \frac{\alpha}{2} C_{ij,k},
\qquad
g(\nabla^{\alpha}_X Y, Z) = g(^{LC}\nabla_X Y, Z) + \frac{\alpha}{2} C(X,Y,Z),
$$

where $C_{ijk} := \Gamma^*_{ij,k} - \Gamma_{ij,k}$ is the Amari–Chentsov tensor (Eq. 47, p. 14).

Bregman divergence of a strictly convex generator $F$, and the dually flat metric and cubic tensor it
induces (§3.7, Eqs. 62–65, p. 16):

$$
\begin{aligned}
B_F(\theta : \theta') &:= F(\theta) - F(\theta') - (\theta-\theta')^\top \nabla F(\theta'), \\
^F g &:= \nabla^2 F(\theta), \qquad ^F C_{ijk} := \partial_i \partial_j \partial_k F(\theta).
\end{aligned}
$$

The Fisher information matrix (Eq. 85, p. 21) and the Fisher information metric (Eq. 100, p. 24) of a
parametric family $P = \{p_\theta\}$:

$$
^P I(\theta) := E_\theta\!\left[\partial_i l\, \partial_j l\right]_{ij} \succeq 0,
\qquad
^P g(u, v) := (u)_\theta^\top\, ^P I(\theta)\, (v)_\theta .
$$

The natural gradient and its update rule (§4.1, Eqs. 201–203, p. 38):

$$
\mathrm{NG}\nabla L_\theta(\theta) := g_\theta^{-1}(\theta)\, \nabla_\theta L_\theta(\theta),
\qquad
\theta_{t+1} = \theta_t - \alpha_t\, \mathrm{NG}\nabla L_\theta(\theta_t).
$$

Chernoff information as the best exponent error (Eq. 219, p. 42):

$$
C[P_1, P_2] = -\log \min_{\alpha \in (0,1)} \int_{x \in X} p_1^{\alpha}(x)\, p_2^{1-\alpha}(x)\, \mathrm{d}\mu(x) \ge 0 .
$$

Fisher–Rao distance as the Riemannian geodesic length (Def. 6, Eq. 127, p. 28):

$$
D_\rho(p_{\theta_1}, p_{\theta_2}) = \int_0^1 \sqrt{\dot\gamma(t)^\top g_{\gamma(t)} \dot\gamma(t)}\, \mathrm{d}t .
$$

Theorems (stated, proofs omitted): the fundamental theorem of information geometry — a torsion-free
connection of constant curvature $k$ has a conjugate torsion-free connection of the same constant
curvature $k$ (Thm. 3, §3.5, p. 14), with the corollary that $a$-flatness is self-dual; uniqueness of the
Fisher information metric under Markov embeddings up to scaling (Thm. 8, p. 25); the dual Pythagorean
identities in a dually flat space (Thm. 6, p. 18); and uniqueness of $\nabla$-projections onto
$\nabla^*$-flat submanifolds (Thm. 7, p. 18). Chernoff information between two members of one exponential
family equals a Bregman divergence at the optimal exponent (Thm. 10, p. 42).

| Symbol | Meaning (in this paper) |
|---|---|
| $M$ | smooth $D$-dimensional manifold (model or parameter space) |
| $g,\ g_{ij}$ | metric tensor field (Fisher metric $^Pg$ for a parametric family $P$) |
| $\nabla, \nabla^*$ | conjugate affine connections (dual structures) |
| $\Gamma^k_{ij}$ | Christoffel symbols of the second kind |
| $C_{ijk}$ | Amari–Chentsov totally symmetric cubic tensor |
| $F, F^*$ | strictly convex Bregman generator and its Legendre–Fenchel conjugate |
| $\theta, \eta$ | primal (natural) and dual (expectation/moment) coordinates, $\eta=\nabla F(\theta)$ |
| $^P I(\theta)$ | Fisher information matrix of the family $P$ |
| $D(\theta : \theta')$ | parameter divergence (asymmetric); $B_F$ is the Bregman one |
| $\gamma, \gamma^\nabla$ | geodesic, geodesic of $\nabla$ |

## Main results

The survey's claims are structural rather than numerical. Its main teaching is the pipeline
$(M, g, \nabla, \nabla^*) \Rightarrow (M, g, C) \Rightarrow (M, g, \alpha C) \Rightarrow
(M, g, \nabla^{-\alpha}, \nabla^{\alpha})$ (§5.1, Eq. 227, p. 45): one conjugate pair yields a whole
one-parameter family, and flatness is inherited dually (Cor. 1–2, p. 15).

- **Riemannian geometry as the self-dual case.** $(M,g)$ is recovered as $(M, g, ^{LC}\nabla,
  ^{LC}\nabla)$, and the mean connection $\bar\nabla = (\nabla+\nabla^*)/2$ coincides with $^{LC}\nabla$
  (Property 3, §3.2, p. 13). But for a self-dual pair no distance is induced; the Riemannian geodesic
  distance is a separate object (§2.5, p. 11).
- **Dually flat geometry generalises Euclidean geometry.** A strictly convex generator $F$ gives
  vanishing $\nabla$-Christoffel symbols, dual global affine coordinates via Legendre–Fenchel duality,
  and asymmetric Pythagorean theorems; the canonical Legendre–Fenchel divergence $A_{F,F^*}$ reproduces
  the Bregman divergence (Eq. 68, p. 17). Symmetric Bregman divergences are exactly squared Mahalanobis
  distances (§3.13, p. 31).
- **Statistical invariance pins down the structure.** The Fisher information metric is the unique
  invariant metric tensor under Markov embeddings up to a scale constant (Thm. 8, p. 25), the
  $f$-divergences are the unique separable invariant divergences, and the $\pm 1$-connections they
  induce coincide with the expected exponential/mixture $\alpha$-connections with $\alpha = 2f'''(1)+3$
  (Eq. 124, p. 27). Invariant $f$-divergences satisfy $I_f[p_\theta : p_{\theta + d\theta}] \simeq
  \tfrac12\,^Pg_{ij}\, d\theta^i d\theta^j$ (Eqs. 121–122, p. 27).
- **Gauge freedom.** The FIM has a family of equivalent representations via the $\alpha$-embeddings
  $l_\alpha(x;\theta) := k_\alpha(p(x;\theta))$ (Eq. 134, p. 29), and the metric is not unique once
  invariance is dropped: $(\rho,\tau)$-monotone embeddings yield a gauge direction in the space of
  metric tensors, with the Fisher metric recovered at the self-dual point (§3.12, pp. 29–30).
- **Natural gradient.** Recommendation: $\mathrm{NG}$ descent is the Riemannian steepest-descent vector
  (Property 5), is invariant under smooth reparameterisation, and on a dually flat space equals the
  ordinary gradient on the dually parameterised loss, $\mathrm{NG}\nabla_\theta L_\theta =
  \nabla_\eta L_\eta$ (Property 7, §4.1.3, p. 40); Bregman mirror descent on the primal Hessian manifold
  is natural gradient descent on the dual one (Property 6, Eq. 205, p. 39). Natural evolution strategies
  approximate the search gradient by $\tilde\nabla J(\lambda) = \tfrac1n \sum_i f(x_i)\nabla\log
  p_\lambda(x_i)$ and the FIM by an empirical average (Eqs. 213–214, p. 40).
- **Applications.** On $(E, \mathrm{KL}^*)$ the best MAP error exponent is the Chernoff information
  $C[P_{\theta_1}:P_{\theta_2}] = B(\theta_1 : \theta^*_{12})$, the exponential geodesic meeting the
  $m$-bisector at $\theta^*_{12} = (1-\alpha^*)\theta_1 + \alpha^*\theta_2$ (Eq. 222–224, p. 42); on
  $(M, \mathrm{KL})$ a set of $w$-GMMs is clustered by Bregman $k$-means on a Monte Carlo dually flat
  approximation (§4.4, p. 44). Worked examples give closed forms: univariate Gaussian FIM
  $\mathrm{diag}(1/\sigma^2, 1/(2\sigma^4))$ (Eq. 97, p. 23), Poisson $I_\lambda(\lambda)=1/\lambda$ and
  Fisher–Rao distance $D_\rho(\lambda_1,\lambda_2)=2(\sqrt{\lambda_1}-\sqrt{\lambda_2})$ (Eqs. 182–183,
  p. 34), and the Gamma manifold whose Fisher–Rao distance has no closed form (§3.11, p. 35).

## Limitations stated by the authors

Proofs are omitted throughout for brevity (§Abstract, p. 1; §3.5 cites [16] for Thm. 3). The exposition
stays in the finite-dimensional parametric setting; non-parametric and quantum information geometry are
named only as directions (§5.3, p. 47). Coefficients of the metric, Christoffel symbols and skewness
tensors for common families are not tabulated, by design (§5.3, p. 47). The geometric characterisation of
the best error exponent is "not necessarily in closed-form" (Fig. 13 caption, p. 42), and the mixture
manifold's generator (negative differential entropy) is not available in closed form, so it is
approximated (§4.4, p. 44). The Fisher–Rao distance is generally unavailable in closed form — no closed
form is known for Gamma distributions and none for any two multivariate Gaussians except a few cases
(§3.11, pp. 35–36).

## Critical assessment

As a teaching document the survey is dense but disciplined; its self-contained build-up and the four
worked applications make the abstract dualistic language concrete. Three things are worth flagging. First,
the absence of proofs makes it a map, not a substitute for Amari's or Amari–Nagaoka's textbooks, which it
explicitly delegates to (§5.3, p. 47); a reader must trust the cited sources for Thm. 3 and the
invariance theorem. Second, several claims are asserted with pointers rather than argued (e.g. the
$f$-divergence/$\alpha$-connection correspondence, Eq. 124, p. 27, and the gauge-freedom discussion,
§3.12, pp. 29–30), so the survey is best used as an index into the field. Third, the "applications"
remain illustrative: the mixture-clustering demonstration uses synthetic $w$-GMMs and Monte Carlo
generators (Figs. 15–16, pp. 43–44), the Cramér–Rao experiment is a 200-trial toy on univariate normals
(Fig. 9, p. 22), and no code is released (only a companion webpage, §5.3, p. 48). For a reader coming
from deep learning, the gap is that the connection to *neural-network* geometry — parameter-space
FIM/neuron manifolds, natural-gradient training — is only name-checked in the applications list (§4, p.
36) and not developed. The most useful, reusable pieces are the equivalence $\mathrm{NG}\nabla_\theta
L_\theta = \nabla_\eta L_\eta$ (Property 7), the dually flat ↔ Bregman dictionary, and the invariance
results that justify choosing the Fisher metric at all. One small fidelity note: the rendered
extraction garbles the $\alpha$-connection signs; the printed Eq. (50) carries a minus and Eq. (51) a
plus on the $\tfrac{\alpha}{2} C$ term, consistent with index raising on a totally symmetric tensor.

## Connections within the collection

This tutorial is the natural reference for the information-geometric threads running through the
collection. [@arvanitidisPullingBackInformation2022] pulls the same Fisher-metric/divergence structure
back onto the latent space of a generative model — the "expected manifold" and pullback constructions
here (§3.9, §3.11) are the direct machinery it reuses. [@arvanitidisGeometricallyEnrichedLatent2020]
similarly equips latent spaces with a Riemannian (here Fisher–Rao) metric, the geometry whose curvature
and geodesic distance this survey defines (§2.2, §3.11). [@combeExploringInformationGeometry2025]
continues the programme of this survey, surveying more recent advances and their connections to
topological field theory, so it should be read as the modern companion. [@pouplinIdentifyingLatentDistances2023]
extends latent-distance geometry from Riemannian to Finslerian metrics, a gauge-freedom direction this
survey flags but does not pursue (§3.12). [@halversonMetricFlowsNeural2024] learns a metric with neural
networks, i.e. treats $g$ as a learned object rather than the fixed Fisher metric whose invariance is
proved here (Thm. 8), and [@villarFullyCovariantMachine2023] argues for geometric/covariant design in
machine learning along the same invariance-first lines (§3.10). Possibly related: [@bronsteinGeometricDeepLearning],
which builds deep-learning architectures from symmetry and geometric priors rather than from the
dualistic statistical structure developed here. The paper cites no works from the collection itself, so
these are thematic rather than explicit dependencies.
