---
citekey: halversonMetricFlowsNeural2024
title: "Metric Flows with Neural Networks"
authors: [James Halverson, Fabian Ruehle]
venue: "Machine Learning: Science and Technology 5(4) 045020"
doi: "10.1088/2632-2153/ad8533"
arxiv: "2310.19870"
url: null
kind: method
source_type: pdf
source_sha256: ff73088b9f9ffdbc1611330dbcf9c9d9707db231840d354ca666c137bd1509ca
pages: 36
ingested: 2026-10-15
tags: [neural-tangent-kernel, metric-flows, calabi-yau, ricci-flow, feature-learning, riemannian-geometry]
concepts_introduced:
  - neural network metric flow
  - metric neural tangent kernel
  - infinite neural network metric flow
  - local metric flow
  - batched feature clustering
concepts_used:
  - neural tangent kernel
  - gradient flow
  - Ricci flow
  - Perelman functional
  - Calabi-Yau metric
  - Kähler potential
  - feature learning
builds_on: []
datasets: [Fermat Quintic, Fermat Cubic]
---

# Metric Flows with Neural Networks

**Halverson, Ruehle (2024)**, *Machine Learning: Science and Technology* 5(4) 045020. [arXiv:2310.19870](https://arxiv.org/abs/2310.19870), [doi:10.1088/2632-2153/ad8533](https://doi.org/10.1088/2632-2153/ad8533)

!!! abstract "TL;DR"
    When a Riemannian metric is represented by a neural network and the network is trained by gradient descent, the metric evolves as a flow governed by a *metric neural tangent kernel*, which is generally non-local, mixes metric components, and changes with time. In the infinite-width limit the kernel freezes, and additional (strong) architectural assumptions make the flow local, at which point Perelman's gradient-flow formulation of Ricci flow is recovered as a special case. Numerically, frozen-kernel methods learn approximate Calabi-Yau metrics far worse than finite-width networks, and the paper shows the metric kernel of a successful finite network evolves during training — evidence that feature learning, not the fixed kernel, does the work.

## Problem and motivation

No nontrivial compact Calabi-Yau (CY) metric is known in closed form: Calabi and Yau guarantee that a Ricci-flat Kähler metric exists when certain criteria hold, but Yau's proof is non-constructive (§1, p. 3). Neural networks are now the state of the art for approximating them — "thirty minutes on a modern laptop gives an approximate Calabi-Yau metric on par with those that would take a decade to compute with Donaldson's algorithm" (§1, p. 3). A training run is a sequence of metric updates, hence a *flow in the space of metrics* in the continuous limit. The questions are: what is the mathematical theory of metric flows induced by continuous neural-network gradient descent, and how do they relate to flows, such as Ricci flow, for which CY metrics are fixed points (§1, pp. 1–4)?

## Method

The metric $g_{ij}(x)$ on a manifold $X$ is represented by a deep network with parameters $\theta^I$, $I=1,\dots,N$ (§2.1, p. 5). Gradient descent on a scalar loss $L[g]$ induces a metric flow through the chain rule, Eqs. (2.1)–(2.2); a batch loss or a measure-weighted loss gives Eqs. (2.5)–(2.8). All contain the distinguished object

$$\Theta_{ijkl}(x,x') := \frac{\partial g_{ij}(x)}{\partial\theta^I}\frac{\partial g_{kl}(x')}{\partial\theta^I}, \qquad \text{(Eq. 2.7, p. 6)}$$

the *metric-NTK*. It is not a 4-tensor: the $(ij)$ indices transform at $x$, the $(kl)$ at $x'$, and invariance of updates under $x'$-diffeomorphisms requires an invariant measure (§2.1, p. 7). The general flow is complicated in three ways: the kernel evolves in time, it is non-local (updating $g_{ij}(x)$ from $g_{kl}(y)$ at distant $y$), and it mixes metric components (§2.1, p. 7).

Two standard NTK observations simplify this (§2.2, pp. 7–9). In the $N\to\infty$ limit the kernel becomes a parameter-independent expectation $\bar\Theta$ fixed by architecture and initialization distribution, Eq. (2.9); for the model linearized in the parameters it is additionally $t$-independent — the frozen NTK, Eq. (2.11) — and deep networks in this limit evolve as linear models ("lazy learning"), so the frozen kernel governs the dynamics to controllable accuracy (§2.2, p. 8). Such a flow is an *infinite neural network metric flow*: analytically tractable, solved by kernel regression in function space rather than parameter-space descent, yet still non-local and component-mixing.

Locality and the absence of mixing are then imposed architecturally (§2.3, pp. 9–14). Giving each metric component its own identically-architected network removes mixing, $\bar\Theta_{ijkl} = \delta_{ik}\delta_{jl}\bar\Theta$ (Eqs. 2.19–2.23). Locality requires the component kernel to become $\delta(x-x')\bar\Omega(x)$ (Eq. 2.24), which two ingredients deliver: a Gaussian-suppressed kernel whose width $\sigma\to0$ becomes a Dirac delta (Eq. 2.27), and a rescaling trick — multiplying a network by a deterministic $h(x)$ multiplies its NTK by $h(x)h(x')$ (Eqs. 2.15–2.16) — used with $h=\tilde g^{-1/4}$ to cancel the metric determinant in the volume measure (Eq. 2.28), giving $\lim_{\sigma\to0}\bar\Theta_\sigma = \delta(x-x')\bar\Omega(x)$ and the *local metric flow* Eq. (2.14). Crucially, in the discrete (batch) case locality would update only training points, so the local flow only makes sense for continuous flows (§2.3, p. 9); backing off $\sigma\to0$ instead gives a parametrically non-local generalization (§2.3, p. 12; §2.4, p. 15).

A concrete realization is a cosine-activation network with only the last layer's $a_i$ trained, Eq. (2.32), whose $N\to\infty$ NTK is a Gaussian, Eq. (2.35), of width $\sigma=\sqrt d/\sigma_w$ after normalization by Eq. (2.36); $\sigma_w\to\infty$ then gives a local flow (§2.3.2, pp. 13–14). It modifies an architecture of Halverson, and Gauss-net (Halverson, Maiti, Stoner) has a Gaussian NNGP kernel that can be repurposed similarly (§2.3.2, p. 14).

Metric flows of the form $dg_{ij}/dt=-u_{ij}(x)$ (Eq. 2.38), in particular Ricci flow Eq. (2.39), are then asked to arise here. Perelman showed Ricci flow is a gradient flow after a $t$-dependent diffeomorphism, with functional $F[\phi,g]$ (Eq. 2.41) and dilaton $\phi$ (§2.4, pp. 14–15); since a local metric flow is a local gradient flow, the loss $l = -F[\phi,g]/\bar\Omega(x)$ (Eq. 2.42) realizes it, and finite $\sigma$ gives a non-local generalization (§2.4, p. 15). Appendix B collects the Kähler-Ricci and Calabi flow analogs, e.g. $d\varphi/dt = -2(\eta-1)$ from the ultra-local functional $F(\eta)=(\eta-1)^2$ (Eq. B.21, pp. 28–32).

## Key equations and definitions

$$
\frac{dg_{ij}(x)}{dt} = \frac{dg_{ij}(x)}{d\theta^I}\frac{d\theta^I}{dt}, \qquad \frac{d\theta^I}{dt} = -\frac{\partial L[g]}{\partial\theta_I} \qquad \text{(Eqs. 2.1–2.2, p. 5)}
$$

Training is equivalent to flowing the metric; Einstein summation is implied.

$$
\frac{dg_{ij}(x)}{dt} = -\int d\mu(x')\,\Theta_{ijkl}(x,x')\frac{\delta l(x')}{\delta g_{kl}(x')} \qquad \text{(Eq. 2.8, p. 6)}
$$

The general flow with a continuous (full-manifold) loss; the batch version replaces $\int d\mu(x')$ by $\sum_{x'\in B}$ (Eq. 2.5).

$$
\bar\Theta_{ijkl}(x,x') = \delta(x-x')\delta_{ik}\delta_{jl}\bar\Omega(x) \qquad \text{(Eq. 2.13, p. 9)}
$$

The locality-and-no-mixing condition on the frozen metric-NTK; when it holds the flow collapses to the local gradient flow $\dfrac{dg_{ij}(x)}{dt} = -\bar\Omega(x)\dfrac{\delta l(x)}{\delta g_{ij}(x)}$ (Eq. 2.14).

$$
\bar\Theta(x,x') = \frac{A^2}{2}\exp\left(-\frac{1}{2}\frac{\sigma_w^2}{d}\,\lvert x-x'\rvert^2\right), \qquad \sigma = \frac{\sqrt{d}}{\sigma_w} \qquad \text{(Eqs. 2.35, 2.37, pp. 13–14)}
$$

The frozen NTK of the cosine-activation architecture of Eq. (2.32); it is Gaussian, and $\sigma_w\to\infty$ makes it a (normalized) delta function.

$$
\frac{dg_{ij}}{dt} = \frac{\delta F[\phi,g]}{\delta g_{ij}(x)} = -2\left[R_{ij} + \nabla_i\nabla_j\phi\right], \qquad F[\phi,g] = \int_X \left(R + \lvert\nabla\phi\rvert^2\right)e^{-\phi}\,dV \qquad \text{(Eqs. 2.40–2.41, pp. 14–15)}
$$

Perelman's gradient-flow formulation of Ricci flow, made a neural network metric flow by the loss choice $l(x) = -F[\phi,g]/\bar\Omega(x)$ (Eq. 2.42). $R_{ij}$ is the Ricci tensor, $\phi$ the dilaton.

$$
g^{(n+1)}_{ij}(x) = g^{(n)}_{ij}(x) - \Delta t\sum_{x'\in B}\Theta_{ijkl}(x,x')\frac{\delta\ell(x')}{\delta g^{(n)}_{kl}} \qquad \text{(Eq. 3.5, p. 19)}
$$

The first-order backward-Euler discretization of the flow used in the numerical experiments; $\Delta t$ is a hyperparameter.

$$
\Theta_{ijkl}(x,x') = \delta_{ik}\delta_{jl}\,e^{-d(x,x')/2\sigma} \qquad \text{(Eq. 3.6, p. 20)}
$$

The Gaussian kernel used in the experiments; $d(x,x')$ is the shortest geodesic distance in the ambient Fubini-Study metric and $\sigma$ a hyperparameter. (The PDF text layer is ambiguous between $-d/2\sigma$ and $-d^2/2\sigma^2$.)

$$
\sigma := \lvert 1-\eta\rvert, \qquad \eta = \frac{J^n}{\kappa\lvert\Omega\rvert^2} \qquad \text{(Eq. B.11, p. 30)}
$$

The *sigma loss* used as $\ell$: the CY metric has $\eta=1$ and $\sigma=0$; $\Omega$ is the holomorphic top form and $J$ the Kähler form.

| Symbol | Meaning (in this paper) |
|---|---|
| $g_{ij}(x)$ | Riemannian metric component, represented by a neural network |
| $\theta^I$ | neural network parameters, $I=1,\dots,N$ |
| $N$ | network width (hyperparameter controlling the infinite-width limit) |
| $\Theta_{ijkl}(x,x')$ | metric-NTK, Eq. (2.7) |
| $\bar\Theta$ | frozen (deterministic, $t$-independent) NTK, in the infinite-width linearized regime |
| $L[g]$, $l(x)$ | scalar loss functional and pointwise loss |
| $\bar\Omega(x)$ | the local kernel's scalar function, Eq. (2.13) |
| $\sigma$ | Gaussian width / non-locality scale of the kernel ($\sigma\to0$ = local) |
| $\sigma_w$ | variance parameter of the cosine-network weights |
| $\phi$ | dilaton; $F[\phi,g]$ Perelman functional |
| $R_{ij}$, $R$ | Ricci tensor, scalar curvature |
| $\eta$, $\sigma$ (sigma loss) | the ratio $J^n/(\kappa\lvert\Omega\rvert^2)$ and its deviation from 1 (the paper flags the symbol clash with the Gaussian width) |

## Main results

**Theory.** Every neural network gradient-descent training of a metric is a metric flow governed by the metric-NTK (Eq. 2.8). Infinite-width, locality and no-component-mixing are independent simplifications; imposing all three plus the loss choice Eq. (2.42) reproduces Perelman's Ricci flow gradient flow (Eq. 2.40), and any gradient flow of a local energy functional can similarly be realized, with a non-local generalization at finite $\sigma$ (§2.4, p. 15). The assumptions needed for Ricci flow are called "both ad hoc and strong" and "very non-generic in the space of neural network metric flows" (§1, p. 4; §4, p. 25).

**Numerics.** The benchmark is the Fermat Quintic, $z_0^5+z_1^5+z_2^5+z_3^5+z_4^5=0$ (Eq. 3.1, p. 15). A baseline finite network (3 layers, 64 nodes, ReLU, ~50 epochs, 100k points, following the `cymetic` works [13, 14]) gains a factor 10–100 in validation loss (§3, pp. 15–16). Kernel methods (ReLU NTK via `neural-tangents`, a Gaussian kernel, and a nearest-neighbour "Delta" kernel) are trained through Eq. (3.5) with the sigma loss and Bayesian-optimized hyperparameters (§3.2, pp. 19–20), but "failed to achieve a gain of more than a factor of $\sim 5$ in the test loss" (§3.2, p. 20). With zero geodesic distance to train points the Gaussian and Delta kernels reach a factor $\sim10^6$ and the ReLU kernel $\sim5$, but the gain falls to $\sim1$ at distance 1 (Fig. 3 left, pp. 21–22). The mean geodesic distance is about $0.3$ and $0.1$ for $10^3$ and $10^5$ points, so the expected test gain is $\sim5$ regardless of sample size (Fig. 3 right, p. 22). A log-log fit gives $\mu = .968\times N_\text{pts}^{-0.184}$ (Eq. 3.7, p. 22), near the theoretical $(N_\text{pts})^{-1/d}$ exponent $-0.167$ for real dimension $d=6$. Extrapolating, gain $\sim100$ needs $10^{12}$ points with the Gaussian kernel ($10^8$ for Delta), versus $10^5$ for the finite network, and test gain $10^6$ would need $\sim10^{35}$ points (p. 22). The authors attribute the gap to the fixed kernel's inability to learn features; since the CY metric in a Kähler class is unique, a working kernel must be a very special function, given by inverse Weierstrass $\wp$-functions for the two-torus (p. 23).

**Kernel evolution.** Using the `cymetic` multiplicative model (3 hidden layers, 128 neurons, gelu, 50 epochs), the empirical metric-NTK (9×9×10×10) is tracked per epoch for 10 fixed train and 10 fixed test points. Its eigenvalue spectrum shifts clearly between epoch 0 and epoch 50, and the Wasserstein distance between the current and initial spectra grows steadily (Fig. 4, pp. 22–23). This is direct evidence of feature learning in successful finite networks.

## Limitations stated by the authors

- The assumptions leading to Perelman's Ricci flow — infinite width, locality, elimination of component mixing — are "ad hoc and strong" and non-generic (§1, p. 4; §4, p. 25).
- The local flow only makes sense for continuous flows: with a discrete batch, delta-function locality would update only training points (§2.3, p. 9).
- The semi-locality induced by the Gaussian suppression in the kernel is itself an assumption made to reach local flows (§2.3.1, p. 12).
- The batched feature-clustering algorithm "is not perfect and depends on the random order of mini batches and clusters"; measured silhouette scores were .016 (batched), $-.011$ (full), and $-0.032$ (Birch/Euclidean baseline), i.e. all near zero, indicating weak cluster separation (§3.1, pp. 17–18).
- The kernel methods "work excellently on the train set, but generalization to test points is challenging", and performance was "very sensitive to the hyperparameters", easily producing exploding gradients or non-positive-definite metric updates (§3, p. 15; §3.2, p. 20).

## Critical assessment

Theoretical core is clean and its main claim — that the general object is the evolving, non-local, component-mixing metric-NTK of which Ricci flow is a corner — is well supported, and Fig. 4 shows a *successful* finite network's metric-NTK genuinely evolving. The causal claim is nevertheless correlational: the paper never trains the *same* finite network with its kernel frozen at initialization, which would isolate feature learning as the cause. The comparison is not fully apples-to-apples either — the finite baseline uses 100k points while the kernel runs use ~1000–2500 (Table 1) — and the extrapolation to $10^{12}$/$10^{35}$ points (p. 22) rests on a fitted $\mu\propto N_\text{pts}^{-0.184}$ and on a noise-perturbation surrogate for geodesic-distance sampling that only puts test points *approximately* on the CY (§3.2, p. 21). Kernel optimization clips updates, which "suffers substantially" (p. 20), so part of the failure may be optimization rather than expressivity. A hidden assumption is using the ambient Fubini-Study geodesic distance as a feature-space distance when no canonical feature metric exists. Reproducibility is plausible but incomplete: the paper names `cymetic`, `neural-tangents`, `scikit-learn` and `BayesianOptimization`, but states no data or code release. A few equations are garbled in the PDF text layer (Eqs. 2.26, 3.6), making re-derivation from the text error-prone. Open questions: which flow a given architecture induces; whether non-NTK kernels help; whether the evolved kernel of a successful finite network can be characterized analytically (pp. 23–25).

## Connections within the collection

This paper stands apart from the collection's centre of gravity (geometry of ReLU/piecewise-linear networks, tropical geometry, TDA). It builds on no collection paper: the only overlap is a passing mention of the geometric deep learning primer [@bronsteinGeometricDeepLearning] as background in its neural-network primer (App. A, p. 27). Its subject — putting a Riemannian metric on a space and letting learning move it — is conceptually adjacent to entries using information/Riemannian geometry of learned representations: [@arvanitidisPullingBackInformation2022] (pullback metrics on latent spaces) and [@pouplinIdentifyingLatentDistances2023] (Finslerian latent geometry) are possibly related, as is the information-geometry survey [@combeExploringInformationGeometry2025]. The frozen-kernel-versus-feature-learning theme touches the collection's recurring question of what a trained network's geometry is, cf. how ReLU networks partition space [@haninComplexityLinearRegions2019; @goujonNumberRegionsPiecewise2023]. Outside the collection, the theory rests on the NTK literature (Jacot et al.; Lee et al.) and Perelman's entropy paper, and the numerics on the neural-network Calabi-Yau metric line (Anderson et al.; Douglas et al.; Jejjala et al.; Larfors et al.; Gerdes & Krippendorf).
