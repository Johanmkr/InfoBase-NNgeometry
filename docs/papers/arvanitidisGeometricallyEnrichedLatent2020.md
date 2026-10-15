---
citekey: arvanitidisGeometricallyEnrichedLatent2020
title: "Geometrically Enriched Latent Spaces"
authors: [Georgios Arvanitidis, Søren Hauberg, Bernhard Schölkopf]
year: 2020
venue: arXiv preprint
doi: 10.48550/arXiv.2008.00565
arxiv: "2008.00565"
url: null
kind: method
source_type: pdf
source_sha256: c7e46c8ec5530e4b2b5778519313781c464c0ec1958ddd02732783bed763eba5
pages: 23
ingested: 2026-10-15
tags: [latent-space-geometry, riemannian-geometry, generative-models, geodesics, interpretability]
concepts_introduced:
  - ambient Riemannian metric
  - geometrically enriched latent space
  - meaningful extrapolation
  - linear extrapolation generator
  - cost-based Riemannian metric
  - data-support-based Riemannian metric
concepts_used:
  - Riemannian metric
  - pull-back metric
  - geodesic
  - variational autoencoder
  - generative adversarial network
  - metric learning
  - radial basis function network
  - uncertainty quantification
builds_on: []
datasets: [synthetic mixture-of-Gaussians dataset, MNIST digits 0-1-2, MNIST digits 0-1-2-3, CelebA]
---

# Geometrically Enriched Latent Spaces

**Arvanitidis, Hauberg, Schölkopf (2020)**, *arXiv preprint*. [arXiv:2008.00565](https://arxiv.org/abs/2008.00565)

!!! abstract "TL;DR"
    The latent space of a generative model is given a Riemannian metric by treating the *ambient observation space* — not just the learned data manifold — as a Riemannian manifold whose metric can encode domain knowledge. Pulling this ambient metric back through the generator yields shortest paths that are identifiability-free *and* semantically controllable, e.g. interpolants that avoid the "blond" class. For stochastic generators a relaxed expected metric is derived; for deterministic generators a residual, linearly extrapolating architecture is proposed.

## Problem and motivation

Generative models parametrize the density of observations in an ambient space $X$ through a lower-dimensional latent space $Z$ (§1, p. 1). Latent representations are not unique — different smooth reparametrizations give identical densities — so straight lines in $Z$ are not shortest paths in any meaningful sense and make poor interpolants (§1, p. 1). Endowing $Z$ with a Riemannian metric that measures curve lengths in the ambient observation space $X$ removes this identifiability problem, since a smooth invertible transformation of $Z$ then leaves distances unchanged (§1, p. 1). The authors' observation is that such distances are *identifiable but not necessarily useful*: they merely follow the learned data manifold. Metric learning is taken as inspiration to instead equip $X$ itself with a Riemannian metric, so that curve lengths in $Z$ can encode high-level, task-specific information (§1, p. 1). The motivating example is a shortest-path interpolant whose ambient metric favours images of blond people, steering the path across the blond class rather than through the low-density region between classes (Fig. 1, p. 1). The setting assumes data near a low-dimensional embedded manifold $\mathcal{M} \subset X$ and a smooth generator $g$ parametrized by a VAE or a GAN (§2, §3.2).

## Method

The construction unifies two previously separate views (§2.1, §4). The first is the *pull-back* view: for an embedded $d$-dimensional manifold $\mathcal{M}$ inside a Riemannian ambient space $X = \R^D$, restricting the ambient metric $M_X(\cdot)$ to the tangent space of the generated manifold induces the metric $M(z) = J_g(z)^\top M_X(g(z)) J_g(z)$ on the latent space via the generator Jacobian $J_g$ (Eq. 3, §2.1; the same formula with a chart map is Eq. 1). The second is *metric learning in the ambient space*: a smoothly varying positive-definite tensor $M_X : X \to \R^{D \times D}$ is designed so that its magnitude is small near the data or near low-cost regions, which pulls shortest paths toward them (§3.1). The paper's proposal is to combine the two: learn a generator whose pull-back captures the data manifold, then moderate it with a designed ambient metric $M_X(\cdot)$ that adds the desired high-level behaviour (§4).

For **stochastic generators** (VAE with Gaussian likelihood, $g(z) = \mu(z) + \mathrm{diag}(\varepsilon)\,\sigma(z)$), the induced metric is random because $\varepsilon$ is; with $\varepsilon$ fixed across $Z$ for smoothness, it is Eq. 8. The true expected metric $M(z) = \E_{\varepsilon}[M_\varepsilon(z)]$ is too costly to evaluate and differentiate inside the geodesic solver, so the authors relax it by evaluating $M_X(\cdot)$ only at the expected generated point $\mu(z)$, giving Eq. 9 (§4, App. D). The relaxation is justified by the assumption that near latent codes $\sigma(z) \to 0$ so the first term dominates, while far from codes $J_\sigma$ dominates and the exact form of $M_X$ matters little (App. D, Fig. 12).

For **deterministic generators** (AE, GAN), the second term of Eq. 7 vanishes and the model cannot by construction capture $\mathcal{M}$ in $Z$; the fix is to make $g$ *extrapolate meaningfully* so that moving outside the prior support moves the generated manifold away from the data, where a data-support-based ambient metric has large magnitude (§4). The proposed architecture is the residual/linear-extrapolation generator of Eq. 10: a nonlinear network $f(z)$ plus a linear map built from the top eigenvectors of the empirical data covariance (§4, App. B). A **cost-based** ambient metric $M_X(x) = \alpha\, h(x) + \varepsilon$ (Eq. 6 without inversion) then makes shortest paths avoid high-cost regions. For high-dimensional images the ambient metric is built in a PCA-projected space $X' = \R^{d'}$ with $D > d' > d$, keeping the generator trained in $X$ but reducing the curse of dimensionality for the pairwise-distance-based metric (§5.1, App. B).

## Key equations and definitions

Pull-back metric induced by the generator (Eq. 3, §2.1, p. 3); $J_g(z) \in \R^{D \times d_Z}$ spans $T_{g(z)}\mathcal{M}_Z$:

$$
\langle v_x, v_x \rangle_x = \langle v_z, J_g(z)^\top M_X(g(z)) J_g(z)\, v_z \rangle = \langle v_z, M(z) v_z \rangle \qquad \text{(Eq. 3)}
$$

Curve length under a metric $M(c(t))$; its minimizers are geodesics (Eq. 2, §2):

$$
\mathrm{length}[\gamma(t)] = \int_0^1 \sqrt{\langle \dot\gamma(t), \dot\gamma(t) \rangle_{\gamma(t)}}\,\mathrm{d}t = \int_0^1 \sqrt{\langle \dot c(t), M(c(t)) \dot c(t) \rangle}\,\mathrm{d}t \qquad \text{(Eq. 2)}
$$

Proposed ambient diagonal metric: $h(x) \to 1$ near the data manifold and $\to 0$ away from it (Eq. 6, §3.1):

$$
M_X(x) = (\alpha \cdot h(x) + \varepsilon)^{-1} \cdot I_D \qquad \text{(Eq. 6)}
$$

Expected Riemannian metric of a VAE with RBF-modelled inverse variance (Eq. 7, §3.2); the second term grows with generator uncertainty:

$$
M(z) = \E_{p(\varepsilon)}\big[J_{g_\varepsilon}(z)^\top J_{g_\varepsilon}(z)\big] = J_\mu(z)^\top J_\mu(z) + J_\sigma(z)^\top J_\sigma(z) \qquad \text{(Eq. 7)}
$$

Stochastic pull-back with ambient metric (Eq. 8, §4), and its cheaper relaxation (Eq. 9, §4, where $\triangleq$ denotes the proposed definition):

$$
M_\varepsilon(z) = \big[J_\mu(z) + J_\sigma(z)\varepsilon\big]^\top M_X\big(\mu(z) + \mathrm{diag}(\varepsilon)\sigma(z)\big)\big[J_\mu(z) + J_\sigma(z)\varepsilon\big] \qquad \text{(Eq. 8)}
$$

$$
M(z) \triangleq J_\mu(z)^\top M_X(\mu(z)) J_\mu(z) + J_\sigma(z)^\top M_X(\mu(z)) J_\sigma(z) \qquad \text{(Eq. 9)}
$$

Linear-extrapolation deterministic generator with $U$ the top $d$ eigenvectors of the data covariance and $b$ the data mean (Eq. 10, §4):

$$
g(z) = f(z) + U \cdot \mathrm{diag}\big([\sqrt{\lambda_1}, \dots, \sqrt{\lambda_d}]\big) \cdot z + b \qquad \text{(Eq. 10)}
$$

Euler–Lagrange system whose boundary-value solution gives geodesics (Eq. 13, App. A):

$$
\ddot c(t) = -\tfrac{1}{2} M^{-1}(c(t))\Big[2(\dot c(t)^\top \otimes I_d)\tfrac{\partial \mathrm{vec}[M(c(t))]}{\partial c(t)}\dot c(t) - \tfrac{\partial \mathrm{vec}[M(c(t))]}{\partial c(t)}^\top(\dot c(t) \otimes \dot c(t))\Big] \qquad \text{(Eq. 13)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $X = \R^D$ | ambient observation space, viewed as a Riemannian manifold |
| $\mathcal{M} \subset X$ | embedded $d$-dimensional data manifold |
| $Z = \R^{d_Z}$ | latent space, trivial chart |
| $M_X(\cdot)$ | ambient Riemannian metric $X \to \R^{D \times D} \succ 0$ |
| $M(z)$ | latent (pull-back) metric in $Z$ |
| $g$ | smooth generator, $Z \to X$; assumed an immersion |
| $J_g(z)$ | Jacobian (basis of $T_{g(z)}\mathcal{M}_Z$) |
| $\mu(z), \sigma(z)$ | mean and standard deviation of a stochastic generator |
| $\beta(z) = (\sigma^2(z))^{-1}$ | inverse variance, modelled by a positive RBF network |
| $h(x)$ | positive function ($\to 1$ near data) used to build $M_X$ |
| $U, \lambda_k, b$ | top eigenvectors, eigenvalues and mean of the data covariance |
| $P, X' = \R^{d'}$ | PCA projection and the reduced ambient space |
| $q(z) \propto \tilde p_r(z)(1 + \sqrt{\vert M(z) \vert})^{-1}$ | metric-derived latent density |

## Main results

The experiments are demonstrations rather than quantitative benchmarks, so the evidence is visual.

**Deterministic generators (§5.1).** On a synthetic dataset of seven Gaussian clusters on a circle, a Wasserstein GAN with $Z = \R^2$ is trained and a density $q(z) \propto \tilde p_r(z)(1 + \sqrt{\vert M(z) \vert})^{-1}$ is defined using the learned metric; samples from $q(z)$ align better with the data manifold and avoid the disconnected gaps, e.g. no "ghostly" samples between clusters (Fig. 4, p. 7; Fig. 13, p. 19). Fitting a mixture of locally adaptive normal distributions to those samples gives an unsupervised alternative to a conditional GAN (§5.1, p. 7). On MNIST digits 0,1,2 with $Z = \R^5$, a PCA projection to $d' = 10$ is used to build $M_{X'}(\cdot)$; the resulting interpolants avoid off-manifold "shortcuts" that the plain shortest path takes (Fig. 5, p. 7; Fig. 14, p. 20). On CelebA with a pre-trained Progressive GAN ($Z = \R^{512}$), the ambient metric is built in $X' = \R^{1000}$ from an RBF cost whose centres are points classified as blond; only the proposed path avoids the high-cost ("blond") regions, while both the straight line and the naive shortest path cross them (Fig. 7, p. 8; Fig. 15, p. 21). Because computing the Jacobian of this generator is prohibitive, shortest paths are approximated by a finite-difference metric on a graph over a random 10-dimensional subspace of $Z$, with an ISOMAP-style Dijkstra search and cubic-spline smoothing (Eqs. 23–26, App. E.1, pp. 17–18).

**Stochastic generators (§5.2).** A VAE with $Z = \R^2$ ($d' = 10$ ambient) on MNIST digits 0,1,2,3 is used to show that combining ambient metrics steers paths: an LDA metric (digits 0,1,3 as one class) makes the path avoid the digit-2 region, a local covariance metric (Eq. 5) pulls it toward selected points, a cost-based RBF repels it from chosen neighbourhoods, and a linear combination enforces all constraints at once while still respecting the learned geometry (Fig. 8, p. 8; Fig. 17, p. 23). The mixture weights used are 1 (LDA), 0.001 (local diagonal inverse covariance) and 0.1 (cost) (App. E.2, p. 22). Kernel density estimation in $Z$ shows the metric separates digits better under the LDA part and lowers density near high-cost regions (Fig. 9, p. 8). An extra experiment confirms the meaningful extrapolation: the distance $\lVert b - \mu(z)\rVert_2$ increases as $z$ leaves the prior support, faster along the latent axis with the larger eigenvalue (Fig. 16, p. 22).

## Limitations stated by the authors

- Meaningful uncertainty quantification with parametric models remains an open problem; the RBF construction of $\sigma(\cdot)$ is a practical albeit principled workaround (§3.2, p. 5).
- The relaxation of the expected metric (Eq. 9) is only argued and checked empirically; a principled derivation is not given (§4, App. D).
- No guarantees for the generator's immersion property are offered: the Jacobian $J_g = J_f + A$ could in principle have cancelling columns, breaking full rank; "proper guarantees and analysis should be provided in the future" (App. B, p. 13).
- The linear projection to $X'$ is only acceptable if it does not change the data-manifold structure, and Euclidean distances are not meaningful for images, especially in high dimensions (§5.1, App. B).
- A principled method to estimate the mixture coefficients of several ambient metrics is left as a future problem (App. E.2, p. 22).
- GAN encoders may learn unmeaningful representations, which is why the stochastic-generator route is preferred (§3.2, p. 5).

## Critical assessment

The conceptual move — treating the *ambient* space rather than the embedded manifold as the carrier of domain knowledge — is clean and, as the authors note, previously untried (§4). The relaxation in Eq. 9 is a sensible engineering compromise given the cost of differentiating an expectation inside a boundary-value ODE solve.

The evidence, however, is entirely qualitative. No quantitative metric (reconstruction error, FID, classification accuracy of interpolants) is reported; the claims rest on side-by-side image grids that the reader must inspect. The experiments also rely on several hand-tuned pieces: the PCA projection dimension $d'$, the RBF bandwidths and scaling factors, and the ambient-metric mixture weights (0.001–1 across terms), the latter explicitly acknowledged as unprincipled (App. E.2). The CelebA demonstration is the weakest link: it substitutes a finite-difference, ISOMAP-style graph heuristic for the actual geodesic problem (App. E.1), uses a random 10-dimensional subspace of a 512-dimensional latent space, and cannot even evaluate the metric's smoothness because the generator uses ReLU (App. E.1, p. 17). It shows that a designed cost can steer a path, but not that the proposed pull-back framework is what makes it work.

The "meaningful extrapolation" claim rests on the conjecture that trained generators stay away from their activation limits so that Jacobians keep full rank (App. B, p. 12) — plausible but unproven, and the paper itself flags cancellation between $J_f$ and $A$ as a theoretical failure mode. No code or data release is mentioned, and the number of experimental knobs makes independent reproduction non-trivial. A useful next step, which the conclusion itself proposes, is to use the ambient geometry *during* training rather than only at inference (§6, p. 8).

## Connections within the collection

This paper is the direct methodological ancestor of [@arvanitidisPullingBackInformation2022], which removes the paper's reliance on a Gaussian likelihood (and hence on the reparametrizable $\mu+\varepsilon\sigma$ generator) by pulling back the Fisher–Rao metric of general decoder distributions to the latent space; the 2022 paper's abstract frames itself explicitly as fixing where "the existing theory … relies on the decoder being a Gaussian". Both are concerned with giving latent spaces a meaningful, identifiability-free geometry.

[@pouplinIdentifyingLatentDistances2023] attacks the same problem of meaningful latent distances from the opposite direction, using Finslerian rather than Riemannian structure, and can be read as a competing framework for the latent interpolation task studied here. [@nielsenElementaryIntroductionInformation2020] supplies the information-geometric background that the Ramanujan-style pull-back of the 2022 sequel builds on. The manifold-learning premise shared by all of these — data near a low-dimensional manifold — is surveyed in [@meilaManifoldLearningWhat2023]. The regularizer-as-ambient-metric idea is thematically close to [@chazalIntroductionTopologicalData2021] in that both seek structure (topology/density) beyond Euclidean distances, though the connection is conceptual rather than one of direct reuse. The foundations this paper actually builds on — Tosi et al. (2014), Arvanitidis et al. (2016, 2018, 2019), Hauberg et al. (2012) — are all outside the collection.
