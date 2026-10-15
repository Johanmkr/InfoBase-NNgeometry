---
citekey: pouplinIdentifyingLatentDistances2023
title: "Identifying Latent Distances with Finslerian Geometry"
authors: [Alison Pouplin, David Eklund, Carl Henrik Ek, Søren Hauberg]
year: 2023
venue: arXiv preprint
doi: 10.48550/arXiv.2212.10010
arxiv: "2212.10010"
url: null
kind: theory
source_type: pdf
source_sha256: 2b886e8250fa4d3b8632d2e412722c590a6eb2986160247ad9ccce94ab3570d1
pages: 36
ingested: 2026-10-15
tags: [finsler-geometry, latent-space, pullback-metric, gaussian-process, riemannian-geometry]
concepts_introduced:
  - Finsler metric of the expected norm
  - expected norm on random manifolds
  - Finsler indicatrix
  - Busemann-Hausdorff volume measure
concepts_used:
  - pullback metric
  - stochastic immersion
  - Gaussian process latent variable model
  - non-central Nakagami distribution
  - non-central Wishart distribution
  - geodesic
  - Jensen's inequality
builds_on: [arvanitidisPullingBackInformation2022]
datasets: [pinwheel-on-sphere, concentric-circles-on-sphere, font dataset, qPCR single-cells, MNIST, FashionMNIST]
---

# Identifying Latent Distances with Finslerian Geometry

**Pouplin, Eklund, Ek, Hauberg (2023)**, *arXiv preprint*. [arXiv:2212.10010](https://arxiv.org/abs/2212.10010)

!!! abstract "TL;DR"
    When the decoder $f$ of a generative model is stochastic, the pullback metric $G=J^\top J$ is a random
    field, so lengths on the latent space are random too. Taking the expectation of the metric tensor gives
    a Riemannian norm $\norm{\cdot}_R$ whose geodesics do *not* minimise the expected length, whereas taking
    the expectation of the norm directly gives $\norm{\cdot}_F$, which the authors prove is a Finsler metric.
    For Gaussian-process decoders the Finsler norm has a closed form, and the two norms are bounded both
    absolutely and relatively, converging at rate $O(1/D)$ in the data dimension $D$ — which justifies using
    the simpler Riemannian approximation in practice.

## Problem and motivation

Generative models learn a smooth decoder $f : Z \subset \R^q \to X \subset \R^D$, $x = f(z)$, and one wants
distances on the latent manifold that reflect the geometry of the data rather than the Euclidean metric,
which is not identifiable (§1, p. 1). Pulling back the Euclidean metric through $f$ gives the Riemannian
metric tensor $G_z = J^\top(z)J(z)$ whose geodesics are reparametrisation-invariant and therefore
identifiable (§1, p. 2). The difficulty is that $f$ is often stochastic (e.g. a Gaussian process in a GPLVM
(Lawrence, 2003)), so $G$ is a random matrix, $\norm{\cdot}_G$ is a random norm, and lengths become random
variables that cannot be manipulated with deterministic differential geometry (§2, p. 4).

The established remedy is to approximate the random metric tensor by its expectation
$\E[G]$, giving the *expected Riemannian* norm $\norm{\cdot}_R$ (Tosi et al., 2014; Arvanitidis et al., 2018).
The authors point out that this is ad hoc: the length $\int \sqrt{\dot\gamma^\top \E[G] \dot\gamma}$ is not
the expected length $\E[\int \sqrt{\dot\gamma^\top G \dot\gamma}]$ (§2.3, p. 5). They instead propose taking
the expectation of the *norm* itself, $\norm{u}_F := \E[\sqrt{u^\top G u}]$, and ask how far it is from the
Riemannian approximation (§2.4, p. 6). The setting assumes $f$ is a smooth *stochastic immersion*
(Definition 2.2, p. 5; Definition A.11, p. 22), which notably rules out the VAE decoder but covers
GPLVMs (§1.2, p. 4, footnote 1).

## Method

The construction is analytic rather than algorithmic (§2–§3). Given a stochastic metric tensor $G$

1. define the random norm $\norm{u}_G = \sqrt{u^\top G u}$, and the random manifold $(Z,g)$ (Definition 2.3, p. 5);
2. define the Riemannian competitor $\norm{u}_R = \sqrt{u^\top \E[G] u}$ (Definition 2.4, p. 5);
3. define the Finsler candidate $\norm{u}_F = \E[\norm{u}_G]$ (Definition 2.5, p. 6);
4. check the Finsler axioms (homogeneity, smoothness, strong convexity) and derive closed forms, bounds,
   and high-dimensional asymptotics for Gaussian processes (§2.4–§3.4);
5. illustrate on a GPLVM with synthetic and real data (§4).

For experiments the authors train a GPLVM in Pyro (Bingham et al., 2019) with and without stochastic active
sets (Moreno-Muñoz et al., 2022), obtain $\E[J]$ and $\Sigma$ of the Jacobian from the posterior, and
compute both curve energies; geodesics are obtained by minimising a spline/energy discretisation with the
Stochman library (Detlefsen et al., 2021) (§4, p. 11; §D.1, p. 34). The Finsler indicatrix has no closed
form, so it is traced numerically by contouring $F(x,v)=1$ (§C.3, p. 32).

## Key equations and definitions

The pullback metric and its induced norm (Definition 2.1, p. 4):

$$
g_z(u, v) = u^\top G v, \qquad G = J^\top J, \qquad \norm{u}_G = \sqrt{u^\top G u}
$$

The two competing deterministic norms (Definition 2.4–2.5, pp. 5–6):

$$
\norm{u}_R := \sqrt{u^\top \E[G]\, u}, \qquad \norm{u}_F := \E\!\left[\sqrt{u^\top G u}\right]
$$

Finsler axioms (Definition 2.6, p. 6; via Shen & Shen, 2016): positive homogeneity $F(\lambda v)=\lambda F(v)$,
smoothness on the slit tangent bundle $TZ\setminus\{0\}$, and strong convexity of
$g_{ij}(v)=\tfrac12 \partial^2 F^2/\partial v^i \partial v^j(v)$.

Closed form for a Gaussian-process decoder, $J \sim \mathcal{N}(\E[J], \Sigma)$ (Proposition 2.3, p. 6):

$$
\norm{v}_F = \sqrt{2}\,\sqrt{v^\top \Sigma v}\;\frac{\Gamma\!\left(\frac{D}{2}+\frac{1}{2}\right)}{\Gamma\!\left(\frac{D}{2}\right)}\;{}_1F_1\!\left(-\frac{1}{2}, \frac{D}{2}, -\frac{\omega}{2}\right),
\qquad \omega = (v^\top \Sigma v)^{-1}\,(v^\top \E[J]^\top \E[J]\, v)
$$

because $v^\top G v \sim W_1(D, v^\top\Sigma v, \omega)$ follows a one-dimensional non-central Wishart
distribution, whose square root is non-central Nakagami (Proposition 2.3 proof, p. 24).

Absolute bounds (Proposition 3.1, p. 8), with $\alpha = 2\left(\Gamma(\tfrac{D}{2}+\tfrac12)/\Gamma(\tfrac{D}{2})\right)^2$:

$$
\norm{v}_{\alpha\Sigma} \le \norm{v}_F \le \norm{v}_R
$$

The upper bound is Jensen's inequality on the concave square root; the lower bound uses the closed form.
Relative bound for a general stochastic immersion (Proposition 3.2, p. 9):

$$
0 \le \frac{\norm{v}_R - \norm{v}_F}{\norm{v}_R} \le \frac{\operatorname{Var}\!\left[\norm{v}_G^2\right]}{2\,\E\!\left[\norm{v}_G^2\right]^2}
$$

and its Gaussian-process specialisation (Proposition 3.3, p. 9):

$$
0 \le \frac{\norm{v}_R - \norm{v}_F}{\norm{v}_R} \le \frac{1}{D+\omega} + \frac{\omega}{(D+\omega)^2}
$$

Energy gap between the two constructions (Eklund & Hauberg, 2019; §2.3, p. 5):

$$
E_R(\gamma) - E_F(\gamma) = \int_0^1 \operatorname{Var}\!\left[\norm{\dot\gamma(t)}_G\right] \mathrm{d}t
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $Z \subset \R^q$, $X \subset \R^D$ | latent and data manifolds of dimension $q$, $D$ |
| $f : Z \to X$ | stochastic immersion (decoder) |
| $J$ | Jacobian of $f$; $J \sim \mathcal{N}(\E[J], \Sigma)$ for a GP |
| $G = J^\top J$ | stochastic pullback metric tensor (non-central Wishart) |
| $\norm{\cdot}_G$ | stochastic induced norm $\sqrt{u^\top G u}$ |
| $\norm{\cdot}_R$ | norm induced by the expected metric tensor $\E[G]$ |
| $\norm{\cdot}_F$ | Finsler norm $\E[\sqrt{u^\top G u}]$ |
| $\omega$ | non-centrality parameter $(v^\top\Sigma v)^{-1}(v^\top\E[J]^\top\E[J]v)$ |
| $L, E, V$ | curve length, curve energy, volume (subscripts $R$ Riemannian, $F$ Finsler, $\alpha\Sigma$ lower bound) |
| ${}_1F_1$ | confluent hypergeometric (Kummer) function |

## Main results

- **The expected norm is Finsler, not Riemannian** (Proposition 2.2, p. 7; Lemma B.1.1–B.1.3, pp. 23–24):
  $F(v)=\E[\sqrt{v^\top G v}]$ is positively homogeneous, smooth and strongly convex, but not bilinear, so it
  cannot come from an inner product. Stochastic manifolds are therefore "inherently more complex" than
  deterministic ones (§5, p. 13).
- **Closed form** for GP decoders (Proposition 2.3, p. 6), enabling direct computation of the Finsler metric
  and its curve energy (§D.1.2, p. 34).
- **Absolute bounds** sandwich $F$ between two Riemannian norms, $\norm{v}_{\alpha\Sigma}\le\norm{v}_F\le\norm{v}_R$
  (Proposition 3.1, p. 8), inherited by length, energy and Busemann-Hausdorff volume (Corollary 3.1, p. 9).
  The volume result required a hyperspherical-coordinate argument (§B.2.1, pp. 26–28).
- **Relative bounds** (Proposition 3.2–3.3, p. 9) show the two norms coincide when the variance vanishes —
  i.e. the metric is deterministic — or as $D$ grows.
- **High-dimensional convergence** (Corollary 3.3–3.4, pp. 10–11): for a GP on a bounded latent manifold,
  $L_R-L_F$, $E_R-E_F$ and $\norm{v}_R-\norm{v}_F$ are all $O(1/D)$, the volume ratio is printed as
  $O(\sqrt{D})$ (see critical assessment), and $R \sim_{+\infty} F$ for the length, energy, volume and norm.
- **Experiments** (§4, pp. 11–13). On 3-dimensional synthetic pinwheel/concentric-circle data mapped to a
  sphere (Fig. 5, p. 12; Fig. 2, p. 8), Riemannian geodesics avoid high-variance regions and follow longer
  paths in $\R^3$ while Finsler geodesics are "not perturbed by the variance term" and take shorter paths.
  On the font dataset ($D=256$) and qPCR single-cell data ($D=48$) the two geodesic families agree (Fig. 6,
  p. 13); the Riemannian/Finsler volume ratio near the data is of order $10^{-4}$ (Fig. 3, p. 10). On MNIST
  and FashionMNIST the posterior variance is low and no visible difference remains (Fig. 7, p. 14). Training
  details are in Table 2, p. 32 (e.g. pinwheel: 500 points, RBF kernel, lengthscale 0.24, 17000 steps).
  Code is released at https://github.com/a-pouplin/latent_distances_finsler.

## Limitations stated by the authors

- The derivations require a *smooth* stochastic process; they do not hold for VAE decoders, whose posterior
  independence makes the covariance a Dirac delta (footnote 1, p. 4). Results are therefore GP-specific.
- The Finsler view "does not immediately translate into a practical computational tool" because Finsler
  manifolds are computationally less developed than Riemannian ones (§5, p. 13); the paper's positive
  message is that the Riemannian approximation is good enough.
- The high-dimensional experiment is only an illustration (Fig. 4, p. 11): because a fresh GP would be
  needed at each $D$, the authors instead truncate a sampled Jacobian, so only one factor is varied and the
  data distribution itself is held fixed (§C.5, pp. 33–34).
- Finsler volume is not unique; only the Busemann-Hausdorff choice is treated (§3.1, p. 7; §A.3, p. 21),
  though the authors argue it coincides with Holmes-Thompson once the metric becomes Riemannian.

## Critical assessment

The mathematical core is clean and convincing: the Finsler axioms are verified in full (Lemma B.1.1–B.1.3,
pp. 23–24) and the closed form follows from standard non-central Wishart/Nakagami moment results. The main
theoretical claim — convergence at rate $O(1/D)$ — is a real justification for the widely used expected-metric
heuristic, and the empirical part supports it exactly in the high-$D$ regime (font, MNIST) where it matters.

Weak points, in my reading. First, the $O(\sqrt{D})$ volume bound in Corollary 3.3 (p. 11) is odd next to the
$O(1/D)$ claims for length and energy, and the printed statement that the ratio is $O(\sqrt{D})$ does not by
itself look like convergence; the polynomial expansion in $D$ would rather suggest behaviour like $O(1/D)$
times the volume prefactor, so the printed order merits a check against §B.2.3 (not read in full here).
Second, the experiment that is supposed to *demonstrate* the $D$-dependence uses Jacobian truncation rather
than retrained models (§C.5), so it tests the analytic bound numerically but not the modelling claim that
real high-dimensional latent geometry becomes Riemannian. Third, the failure for VAEs is a genuine scope
restriction: VAEs are the most common practical stochastic decoder, and Arvanitidis et al. (2018) obtained
the expected-metric approach precisely for VAEs — so the paper justifies a heuristic in a setting (GPLVM) that
is arguably less used than the one it excludes. Fourth, on synthetic data the qualitative distinction
(Riemannian avoids variance) is visible, but there is no quantitative error metric for the geodesics; the
evidence is visual. The absolute lower bound involves the function $\alpha\Sigma$ introduced in the paper,
which is useful for theory but not obviously deployable.

Overall this is a well-executed theory paper whose practical conclusion is deliberately modest ("use the
Riemannian metric"), and it is honest about that. The interesting open direction it opens is the converse:
characterising when the variance term is *not* negligible, i.e. where latent geometry genuinely needs Finsler
machinery.

## Connections within the collection

The paper is a direct successor of the expected-metric line of work: it adopts the Riemannian approximation
introduced for GPLVMs and VAEs and extends it with the Finsler alternative. The same group's
[@arvanitidisPullingBackInformation2022] is cited as the information-geometric route to handling the
randomness of the metric (taking straightest paths on a manifold of distributions rather than shortest paths
on the data manifold); this paper is complementary, keeping the data-manifold view but changing the norm
(§1.2, p. 4). It also sits in the broader "geometry of latent representations" theme of
[@arvanitidisGeometricallyEnrichedLatent2020], though it does not cite that specific paper. Because it
operates on the smooth GP decoder and its Jacobian, it is methodologically closer to the manifold-learning
and geometry-of-deep-models work in this collection (e.g. [@meilaManifoldLearningWhat2023]) than to the
ReLU/polyhedral line, and the two literatures are disjoint: nothing here concerns piecewise-linear networks
or their linear regions. Important related works outside the collection are Eklund & Hauberg (2019), whose
bound the paper builds on, and Arvanitidis et al. (2018, "Latent Space Oddity"), Tosi et al. (2014) for the
expected metric, and the Stochman library (Detlefsen et al., 2021) used for all geodesic computations.
