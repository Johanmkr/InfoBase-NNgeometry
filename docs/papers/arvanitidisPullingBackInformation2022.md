---
citekey: arvanitidisPullingBackInformation2022
title: "Pulling Back Information Geometry"
authors: [Georgios Arvanitidis, Miguel González-Duque, Alison Pouplin, Dimitris Kalatzis, Søren Hauberg]
year: 2022
venue: Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 151
doi: "10.48550/arXiv.2106.05367"
arxiv: "2106.05367"
url: null
kind: method
source_type: pdf
source_sha256: 4b2174eb59d28c6ecbc76c92f512012ee020eb3816a6d67bea5d3e8e8d98e46a
pages: 23
ingested: 2026-10-15
tags: [information-geometry, latent-space-geometry, variational-autoencoders, fisher-rao-metric, geodesics]
concepts_introduced:
  - pullback Fisher-Rao metric
  - information geometric latent metric
  - black-box random geometry
  - KL-based curve energy
  - uncertainty regularization
concepts_used:
  - Fisher-Rao metric
  - Kullback-Leibler divergence
  - statistical manifold
  - Riemannian manifold
  - shortest path (geodesic)
  - variational autoencoder
  - reparameterization trick
  - identifiability
builds_on: [nielsenElementaryIntroductionInformation2020, arvanitidisGeometricallyEnrichedLatent2020]
datasets: [MNIST (digit 1 subset), synthetic circular data, CMU motion capture, MovieLens 25M]
---

# Pulling Back Information Geometry

**Arvanitidis, González-Duque, Pouplin, Kalatzis, Hauberg (2022)**, *AISTATS, PMLR 151*. [arXiv:2106.05367](https://arxiv.org/abs/2106.05367)
<!-- venue in the .bib is the arXiv @misc entry; corrected to AISTATS 2022 from the PDF's proceedings line (p. 1). -->

!!! abstract "TL;DR"
    VAE latent geometry has so far been induced by pulling the Euclidean metric back through the Jacobian of a Gaussian decoder, which requires a reparametrizable, location-scale likelihood. The paper instead views the decoder as a map $h: Z \to H$ into the *parameter space* of the likelihood, equips $H$ with the Fisher-Rao metric, and pulls that metric back to $Z$. Because the Fisher-Rao metric agrees locally with the KL divergence, shortest paths can be optimized by differentiating KL terms — a distribution-agnostic "black-box" latent geometry, demonstrated for Gaussian, Bernoulli, Beta, Dirichlet, Exponential and von Mises-Fisher decoders.

## Problem and motivation

Deep generative models assume data $x \in X$ lie near a low-dimensional manifold parametrized by latent codes $z \in Z = \R^d$, but a latent variable model only recovers a *probabilistic* estimate of that manifold, encoded in the decoder $p(x|z)$ (§1, p. 1). Latent geometry is the standard cure for the resulting identifiability problem — an invertible reparametrization $g$ with decoder $f \circ g^{-1}$ yields the same density — but classic differential geometry applies to deterministic manifolds, and the stochastic decoder is not one.

For a deterministic decoder $f$, the pullback metric $M(z) = J_f(z)^\top J_f(z)$ solves the problem (Eq. 1–2, p. 2), and the Gaussian case was extended to VAEs by reading $f(z) = \mu(z) + \sigma(z) \odot \epsilon$ as a random projection of a deterministic manifold, giving $M(z) = J_\mu(z)^\top J_\mu(z) + J_\sigma(z)^\top J_\sigma(z)$ (Eq. 3–4, p. 3). That route is "intrinsically linked with the simple reparametrization of the Gaussian" and only covers location-scale distributions (§2.1, p. 3), leaving categorical, Gamma, Beta, Dirichlet and von Mises-Fisher decoders without a principled geometry.

## Method

The reframing: the decoder's codomain is already the parameter space $H$ of a probability density. Given a likelihood $p(x|\eta)$, $\eta \in H$, the decoder is a map $h: Z \to H$ with $p(x|z) = p(x|h(z))$ (§3, p. 3), so instead of a random manifold in $X$ the model spans a *deterministic* manifold $h(Z)$ inside $H$ (Fig. 1, p. 2).

1. Equip $H$ with the Fisher-Rao metric $I_H(\eta)$, making it a statistical manifold (Def. 3.1, p. 3).
2. Pull that metric back through $h$: $M(z) = J_h(z)^\top I_H(h(z)) J_h(z)$ (Prop. 3.2, p. 4).
3. Because the Fisher-Rao metric is the local second-order approximation of the KL divergence, approximate curve energy along a discretized curve by a sum of KL terms between neighbouring decoded distributions (Eq. 10, p. 4), then minimize it.
4. Represent the curve as a cubic spline with fixed endpoints and optimize its control points by automatic differentiation, initializing from a graph shortest-path pass over a latent grid (§3.2, D.2, pp. 4, 19).
5. Optionally recover the full metric tensor from KL values at perturbed points (Prop. 3.4, p. 5), enough for exponential maps and statistics on the manifold (§3.4, p. 5).

Lengths are measured on $h(Z) \subset H$, so the construction is invariant to reparametrizations of $Z$ and inherits the identifiability protection of the classical pullback (§3.1, p. 4). A shortest path computed *inside* $H$ can nonetheless leave the data support and decode to nonsense, so the decoder is designed to *extrapolate to high uncertainty* away from the training codes (§3.1, p. 4, Fig. 3). This "uncertainty regularization" (Sec. C.2, p. 17) clusters training latent codes with KMeans, defines $D(z) = \min_j \norm{z - c_j}$, and blends the learned parameters with a maximum-uncertainty distribution through a translated sigmoid $\tilde\sigma_\beta(D(z))$ (Eq. 26–27, p. 18): $\sigma \to \infty$ for a Gaussian, $p \to 1/2$ for a Bernoulli, $\kappa \to 0$ for a von Mises-Fisher.

The categorical case is worked out in full: the Fisher-Rao matrix on the simplex is $\mathrm{diag}(1/\eta_1, \dots, 1/\eta_D)$ (Sec. B, p. 15) and distances are great-circle distances between $\sqrt{\eta}$ and $\sqrt{\eta'}$ (Eq. 11, p. 5). When no closed-form KL exists, Monte Carlo estimates through the reparameterization trick give gradients — the "black-box random geometry" of §3.4, p. 5 — and one `curve_energy` interface serves any decoder (D.1, p. 18).

## Key equations and definitions

Latent curve length under the deterministic pullback, and the equivalent energy functional:

$$
L(\gamma) = \int_0^1 \sqrt{\dot\gamma(t)^\top J_f(\gamma(t))^\top J_f(\gamma(t))\, \dot\gamma(t)}\,\mathrm{d}t, \qquad
E(\gamma) = \int_0^1 \dot\gamma(t)^\top M(\gamma(t))\, \dot\gamma(t)\,\mathrm{d}t \qquad \text{(Eq. 1–2, p. 2)}
$$

The first defines distance in $Z$ as length measured in $X$; the second is the locally convex functional minimized to find shortest paths.

$$
M(z) = J_\mu(z)^\top J_\mu(z) + J_\sigma(z)^\top J_\sigma(z) \qquad \text{(Eq. 4, p. 3)}
$$

The existing Gaussian-VAE metric: distances grow with model uncertainty, so shortest paths avoid low-density regions.

$$
I_H(\eta) = \int_X \big[\nabla_\eta \log p(x|\eta)\, \nabla_\eta \log p(x|\eta)^\top\big]\, p(x|\eta)\,\mathrm{d}x \qquad \text{(Eq. 7, p. 3)}
$$

The Fisher-Rao information matrix; a parameter space equipped with it is a statistical manifold (Def. 3.1, p. 3), a modelling decision tied to the likelihood and not to the data.

$$
\mathrm{KL}\big(p(x|\eta), p(x|\eta + \delta\eta)\big) = \tfrac{1}{2}\, \delta\eta^\top I_H(\eta)\, \delta\eta + o(\delta\eta^2) \qquad \text{(Eq. 8, p. 3)}
$$

The local coincidence of KL and Fisher-Rao turns geodesic optimization into a KL-sum problem. The main text calls it a second-order approximation (Prop. 3.1) while the appendix says "first order" (Prop. A.1, p. 11) for the same formula; see the critical assessment.

$$
M(z) = J_h(z)^\top I_H(h(z)) J_h(z) \qquad \text{(Prop. 3.2, p. 4)}
$$

The central construction: the Fisher-Rao metric of the likelihood parameters pulled back along the decoder. Prop. 3.3 (p. 4) shows it equals the Fisher-Rao metric computed with $z$ itself as parameter.

$$
E(\gamma) \propto \lim_{N\to\infty} \sum_{n=1}^{N-1} \mathrm{KL}\big(p(x|\gamma(t_n)),\, p(x|\gamma(t_{n+1}))\big), \qquad t_n = \tfrac{n}{N} \qquad \text{(Eq. 10, p. 4)}
$$

The energy used in every experiment: only a differentiable KL between neighbouring decoded distributions is needed (Prop. A.2, p. 12, gives $E(\gamma) \approx \tfrac{2}{\delta t} \sum_t \mathrm{KL}(p_t, p_{t+\delta t})$).

$$
M_{ii} \approx 2\,\mathrm{KL}_z(\delta e_i)/\varepsilon^2, \qquad
M_{ij} = M_{ji} \approx \big(\mathrm{KL}_z(\delta e_i + \delta e_j) - \mathrm{KL}_z(\delta e_i) - \mathrm{KL}_z(\delta e_j)\big)/\varepsilon^2 \qquad \text{(Prop. 3.4, p. 5)}
$$

Finite-difference recovery of the metric tensor from KL evaluations at basis-vector perturbations $\delta e_i = \varepsilon e_i$; it requires only that $h$ be a smooth immersion and the KL differentiable.

$$
\mathrm{dist}(\eta, \eta') = \arccos\!\big(\sqrt{\eta}^\top \sqrt{\eta'}\big), \qquad
E(\gamma) = \sum_{n=1}^{N-1}\Big(2 - 2\sqrt{h(\gamma(t_n))}^\top \sqrt{h(\gamma(t_{n+1}))}\Big) \qquad \text{(Eq. 11–12, p. 5)}
$$

The categorical case: Fisher-Rao distance on the simplex is spherical distance between element-wise square roots, and the energy follows from $\theta^2 \approx 2 - 2\cos\theta$.

| Symbol | Meaning (in this paper) |
|---|---|
| $Z = \R^d$ | latent space, intrinsic coordinates |
| $X = \R^D$ | observation space |
| $H$ | parameter space of the likelihood; statistical manifold |
| $f: Z \to X$ | deterministic decoder / learned manifold map |
| $h: Z \to H$ | decoder viewed as mapping latent codes to likelihood parameters |
| $J_f, J_h$ | Jacobians of $f$ and of $h$ |
| $M(z)$ | induced Riemannian metric (pullback) in $Z$ |
| $I_H(\eta)$ | Fisher-Rao information matrix on $H$ |
| $p(x\|z) = p(x\|h(z))$ | decoder distribution |
| $\gamma: [0,1] \to Z$ | latent curve; $L(\gamma)$, $E(\gamma)$ its length and energy |
| $\epsilon \sim \mathcal{N}(0, I_D)$ | Gaussian decoder noise |
| $\tilde\sigma_\beta(D(z))$ | uncertainty-regularization slider (Eq. 26) |
| $\mathrm{Log}_\mu, \mathrm{Exp}_\mu$ | logarithmic and exponential maps, used for the LAND density |

The appendix adds the geodesic ODE (Eq. 13, p. 12), closed-form Fisher-Rao matrices for six distributions (Sec. A.1, pp. 12–14), and the grid interpolation used for the LAND fit (Eq. 29, p. 23).

## Main results

**Fisher-Rao vs. Euclidean pullback with Gaussian decoders (§4.1, pp. 5–6).** Four VAEs are trained on an MNIST subset containing only the digit 1 (Adam, $\alpha = 10^{-5}$, batch 32; Tab. 3, p. 19), varying the metric (Euclidean vs. Fisher-Rao) and the uncertainty model (regularized vs. plain). From the shortest-path plots (Fig. 4, p. 6) the authors conclude that (1) the Fisher-Rao pullback is on par with the Euclidean pullback, and (2) uncertainty regularization is instrumental — without it neither metric yields a sensible geometry.

**Fisher-Rao pullback for various distributions on toy data (§4.2, pp. 6–7).** On noisy circular data in $Z_{\text{toy}} = \R^2$, metrics are pulled from the statistical manifolds of multivariate Normal, Bernoulli, Beta, Dirichlet and Exponential distributions, decoded by the untrained MLPs of Tab. 4, p. 20. The shortest paths recover the true circular structure (Fig. 5, p. 6), except for Bernoulli, where some paths fail to converge; the authors blame regularization that is too weak "since Bernoulli distributions with parameters close to 1/2 are already highly entropic" (p. 7).

**Motion capture with products of von Mises-Fisher distributions (§4.3, p. 7).** A 26-bone skeleton (each limb a point on a sphere, following Tournier et al. 2009) is modelled by a VAE decoding a product of 26 von Mises-Fisher distributions (Adam, $\alpha = 10^{-3}$, batch 16, KL annealing 0.01, extrapolation $\kappa \to 0.1$; Tab. 5, p. 20); the KL has no closed form and is Monte Carlo estimated. On sequence Seq.69_06 of the CMU motion capture database the shortest path follows the data and reproduces the periodicity of walking, whereas the Euclidean straight line decodes to "uncharted territory of the latent space" and an implausible motion (Fig. 6, p. 7).

**Numerical quality of the metric approximation (§4.4, pp. 7–8).** On the von Mises-Fisher latent space the approximated $M(z)$ is compared with $\mathrm{KL}(p(x|z), p(x|z+\delta z))$ for perturbations on a circle of radius $\varepsilon = 0.1$. The average error stays below $\varepsilon^2$ inside and outside the data support but grows at the support boundary, and some metrics have negative determinant (Fig. 7, p. 8). Against closed-form metrics (Sec. D.4, p. 21) the relative Frobenius error is $5.32 \cdot 10^{-4} \pm 9.63 \cdot 10^{-4}$ (Normal) and $1.73 \cdot 10^{-5} \pm 1.17 \cdot 10^{-5}$ (Beta).

**Statistics on manifolds (§4.5, p. 8).** A locally adaptive normal distribution (LAND) (Arvanitidis et al. 2016) is fitted to the von Mises-Fisher latent representations, with density $\rho(z) = C(\mu,\Sigma)\exp(-0.5\,\mathrm{Log}_\mu(z)^\top \Gamma \mathrm{Log}_\mu(z))$; since shortest paths follow the data, so does the density (Fig. 7, p. 8). Fitting required the grid-interpolated metric of Eq. 29 (p. 23), repeated Monte Carlo KL evaluation being prohibitive.

**Movie preferences via latent interpolants (§4.6, p. 8).** A Bernoulli VAE is fitted to MovieLens 25M restricted to the 60 most popular movies and 10000 users who have seen between 2 and 30 of them (each movie seen by at least 600 users; Sec. D.6, p. 22, batch 256; Tab. 6, p. 21). Under cosine similarity of decoded preference vectors, the pulled-back shortest path stays among locally similar users while linear interpolation does not (Fig. 8, p. 8).

## Limitations stated by the authors

The method "only works well for decoders with well-calibrated uncertainties": the decoder must be highly entropic where training data is sparse, otherwise shortest paths stop following the data (§6, p. 9) — a hurdle shared with the Gaussian-decoder approach, principled calibration being "currently lacking" (§6, p. 9). The KL-based metric approximation is imprecise near the support boundary and can be indefinite (§4.4, p. 8), which "warrant[s] further research on more stable ways of approximating pulled back metrics" (p. 8). The Bernoulli toy case shows the regularization can be too weak in high-entropy regimes (§4.2, p. 7), and other divergences and Wasserstein geometry are left unexplored (§5, p. 8; §6, p. 9).

## Critical assessment

The conceptual move is clean: a probabilistic decoder *is* a map into a statistical manifold, so the Fisher-Rao metric is the natural reference geometry, and the KL identity makes it implementable with nothing but a differentiable divergence. Generality is real — only differentiability of the KL is assumed — and the motion-capture experiment shows a distribution with no closed-form Fisher-Rao metric is usable at all. The empirical section, however, is qualitative: §4.1, §4.3 and §4.6 rest on plots of shortest paths and decoded samples, with no quantitative measure of geometric quality, no ablation over spline control points or initialization, and no head-to-head against the Gaussian baseline. The "on par" claim of §4.1 is a visual judgement, not a measurement; only §4.4 and D.4 report numbers.

Uncertainty regularization is the weakest link: the method inherits the quality of the decoder's uncertainty extrapolation, tuned here by KMeans clusters and $\beta$ values ($-2.5$ to $-5.5$; Tabs. 4–6), and the Bernoulli failure shows this is fragile where the likelihood is nearly uniform. A missing baseline is the Fisher-Rao metric of the *latent* variable — which Prop. 3.3 proves equivalent — or the natural-gradient view, mentioned only to distinguish the spaces explored (§5, p. 8).

Minor issues: the KL expansion is called second order in Prop. 3.1 and "first order" in Prop. A.1 (p. 11) for the same formula; MovieLens user selection disagrees between the main text ("less than 30 movies", p. 7), Tab. 6 ("more than two … less than 30", p. 21) and D.6 ("between 2 and 30", p. 22); and the code link is unresolved in the PDF (D, p. 18). The cost is significant — Prop. 3.4 needs two decoder passes per diagonal and six per off-diagonal entry (D.5, p. 22) — and the LAND fit works only after the metric is replaced by a kernel-smoothed grid interpolation (Eq. 29), Riemannian but no longer exactly the pullback. Open questions: whether other divergences give better-behaved geometries, whether uncertainty can be calibrated without hand-tuned heuristics, and whether indefinite approximations can be avoided by construction.

## Connections within the collection

The paper's mathematical backbone is information geometry, and its introduction of the Fisher-Rao metric and the KL expansion follows [@nielsenElementaryIntroductionInformation2020] closely (§3, p. 3, Prop. 3.1). It continues the authors' own latent-geometry line, whose collection representative is [@arvanitidisGeometricallyEnrichedLatent2020]: that work enriches latent metrics with alternative geometric structures, whereas here the observation-space reference metric is replaced by the Fisher-Rao metric in *distribution* space (§2, p. 2). [@villarFullyCovariantMachine2023] argues for geometric constructions covariant under the problem's symmetries, the invariance this metric buys — a thematic rather than technical relation. Also on the "metrics learned from data" thread are [@pouplinIdentifyingLatentDistances2023] (same group, Finslerian rather than information-geometric) and [@halversonMetricFlowsNeural2024], possibly related rather than direct predecessors; for background, [@combeExploringInformationGeometry2025] surveys information geometry and [@meilaManifoldLearningWhat2023] covers the manifold hypothesis estimated here.
