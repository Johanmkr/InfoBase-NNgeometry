---
citekey: raghuExpressivePowerDeep2017
title: "On the Expressive Power of Deep Neural Networks"
authors: [Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, Jascha Sohl-Dickstein]
year: 2017
venue: Proceedings of the 34th International Conference on Machine Learning, PMLR 70, pp. 2847–2854
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: 865fece7060b79b2a47db57a1a7842759337a894facd6eb932bf62241920364b
pages: 8
ingested: 2026-10-15
tags: [neural-network-expressivity, trajectory-length, depth-separation, relu-networks, regularization]
concepts_introduced:
  - trajectory length
  - activation pattern
  - neuron transition
  - trajectory regularization
  - upper bound on number of activation patterns
concepts_used:
  - piecewise linear network
  - ReLU activation
  - hard tanh activation
  - linear region
  - hyperplane arrangement
  - batch normalization
builds_on: [NIPS2014_fa6f2a46, pascanuNumberResponseRegions2013]
datasets: [MNIST, CIFAR-10]
---

# On the Expressive Power of Deep Neural Networks

**Raghu, Poole, Kleinberg, Ganguli, Sohl-Dickstein (2017)**, *Proceedings of the 34th International Conference on Machine Learning, PMLR 70*.

!!! abstract "TL;DR"
    The paper replaces worst-case, hand-constructed depth separations with *measures of expressivity* that are computable for any weight setting: the number of activation patterns, the number of neuron transitions along a one-dimensional input trajectory, and the arc length of that trajectory in each layer. The unifying quantity is *trajectory length*, which provably grows exponentially with depth and only polynomially (as a base) with width; the same growth explains why trained networks are far more sensitive to perturbations of lower-layer weights, and motivates *trajectory regularization*, which matches batch normalization on CIFAR-10.

## Problem and motivation

Expressivity asks how architectural properties (depth, width, layer type) constrain the functions a network family can compute, *independently* of a particular weight choice (§1). Prior depth-separation results (Pascanu et al. 2013; Montufar et al. 2014) lower-bound a shallow network's capacity by exhibiting one carefully hand-coded weight setting $W_0$ whose deep network beats every shallow network of equal parameter count; they say nothing about the "typical" case, have no matching upper bounds, and give no route to performance (§1, p. 1). The paper's framing: start after random initialization, measure how the *input* is deformed by the network, then connect the measurements to behaviour during and after training (§1, p. 2). Input space is intractable, so all measures are taken along a one-dimensional *trajectory* $x(t)$ (§2, p. 2).

## Method

The approach is a small family of computable combinatorial and geometric measures, evaluated on fully connected ReLU and hard-tanh networks with $n$ hidden layers of width $k$ (denoted $A_{(n,k)}$), inputs in $\R^m$, weights initialized as $\mathcal{N}(0, \sigma_w^2/k)$ (input scaling), biases $\mathcal{N}(0,\sigma_b^2)$ (§2.2, p. 4).

1. **Activation patterns.** For fixed $W$ and input $x$, record for every neuron which linear piece of its activation function it is on; this string is $AP(F_A(x;W))$, and the number of distinct patterns seen while sweeping $x$ along $x(t)$ is $A(F_A(x(t);W))$ (§2.1, p. 3). Each pattern corresponds to a different linear map, so counting patterns measures the non-linearity actually available.
2. **Neuron transitions.** $T(F_A(x(t);W))$ is the number of times output neurons switch linear region along the trajectory; this is the number of linear regions crossed (§2.1, p. 3).
3. **Trajectory length.** The arc length $l(x(t))$ of the image $z^{(d)}(t)$ of the trajectory after layer $d$ (§2.2, p. 4). This is the quantity that unifies the others: it grows exponentially in depth, and transitions are (under an assumption) proportional to it.
4. **Applications.** Measure trajectory length during training; relate its growth to sensitivity to layer-wise weight noise; regularize it directly.

Experiments sweep circular trajectories between MNIST and CIFAR-10 datapoints through fully connected and convolutional networks of varying depth, width and initialization scale, and count transitions/trajectory length at train steps (§2.1.1, §2.2, §3).

## Key equations and definitions

A trajectory is a curve $x(t)$, $t\in[0,1]$, with $x(0)=x_0$, $x(1)=x_1$; the paper's examples are the line $x(t)=tx_1+(1-t)x_0$ and the circular arc (used in all experiments) (§2, p. 2).

$$
x(t) = \cos\!\left(\tfrac{\pi t}{2}\right) x_0 + \sin\!\left(\tfrac{\pi t}{2}\right) x_1
$$

The activation pattern of a network is a string over $\{0,1\}^{\text{num neurons}}$ (ReLU) or $\{-1,0,1\}^{\text{num neurons}}$ (hard tanh) recording the linear region of every neuron's activation function, for input $x$ and weights $W$ (§2.1, p. 3).

Trajectory length is the standard arc length of the curve (integrated over $t\in[0,1]$) (§2.2, p. 4):

$$
l(x(t)) = \int \left\lvert \frac{dx(t)}{dt} \right\rvert \, dt
$$

**Theorem 1** (upper bound on activation patterns, §2.1, p. 3; proof by induction on regions of a hyperplane arrangement, given in the Appendix, which is not part of this file). For a fully connected network with $n$ hidden layers of width $k$ and inputs in $\R^m$,

$$
A(F_{A_{n,k}}(\R^m; W)) = O\!\left(k^{mn}\right) \ \text{(ReLU)}, \qquad O\!\left((2k)^{mn}\right) \ \text{(hard tanh)}
$$

Tight: it matches the construction of Montufar et al. 2014. Writing the bound as $U(n,k,m)$ and holding the total neuron count $N$ fixed, the paper derives the chain $U(1,N,m) < U(2,\tfrac{N}{2},m) < \cdots < U(n-1,\tfrac{N}{n-1},m) < U(n,k,m)$, i.e. depth beats width at equal parameters.

**Theorem 2** (region geometry, §2.1, p. 3). For ReLU or hard tanh activations, input space is partitioned into convex polytopes, and $F_A(\R^m;W)$ is a different linear function on each. Figure 1 illustrates the "bending" of later-layer activation boundaries.

**Theorem 3** (growth of trajectory length, §2.2, pp. 4–5). Let $F_A(x',W)$ be a ReLU or hard-tanh random network and $x(t)$ a one-dimensional trajectory with $x(t+\delta)$ having a non-trivial perpendicular component to $x(t)$ for all $t,\delta$ (i.e. not a line); let $z^{(d)}(t)$ be the image of the trajectory in layer $d$. Then (proof in the Appendix, not in this file; tight in the limits of large $\sigma_w$ and $k$)

$$
\mathbb{E}\!\left[l(z^{(d)}(t))\right] \geq O\!\left( \left( \frac{\sigma_w \sqrt{k}}{\sqrt{k+1}} \right)^{\!d} l(x(t)) \right) \qquad \text{(a), ReLUs}
$$

$$
\mathbb{E}\!\left[l(z^{(d)}(t))\right] \geq O\!\left( \left( \frac{\sigma_w \sqrt{k}}{\sqrt{\sigma_w^2+\sigma_b^2} + k\sqrt{\sigma_w^2+\sigma_b^2}} \right)^{\!d} l(x(t)) \right) \qquad \text{(b), hard tanh}
$$

*(sic: the relation is printed as "$\geq O(\cdot)$" in both parts; the intended reading is an order-of-magnitude growth statement.)* The key structural claim is that $d$ is an exponent while $k$ only enters the base.

**Theorem 4** (transitions proportional to trajectory length, §2.2, p. 5). For a hard-tanh network, under the assumption that during the sweep all neurons are saturated except at rapidly crossed saturation endpoints,

$$
\begin{aligned}
g(k,\sigma_w,\sigma_b,n) &= O\!\left( \left( \frac{\sqrt{k}}{\sqrt{1 + \sigma_b^2/\sigma_w^2}} \right)^{\!n} \right), \\
T(F_{A_{n,k}}(x(t);W)) &= O(g(k,\sigma_w,\sigma_b,n))
\end{aligned}
$$

The paper notes that $g$ coincides with the expression of Theorem 3 when $\sigma_w$ is large and dominates $\sigma_b$.

| Symbol | Meaning (in this paper) |
|---|---|
| $A$ (architecture) | depth, width, layer types; $A_{(n,k)}$: $n$ hidden layers of width $k$ |
| $A(F_A(x(t);W))$ | number of distinct activation patterns along the trajectory (notation overloaded) |
| $AP(F_A(x;W))$ | activation pattern at a single input |
| $F_A(x;W)$ | function computed by architecture $A$ with parameters $W$ |
| $x(t)$, $z^{(d)}(t)$ | input trajectory; its image after layer $d$ |
| $l(\cdot)$ | arc length (trajectory length) |
| $T(F_A(x(t);W))$ | number of output-neuron transitions along the trajectory |
| $m$, $n$, $k$ | input dimension, number of hidden layers, layer width |
| $\sigma_w$, $\sigma_b$ | weight and bias initialization scales |
| $d$ | layer index in Theorem 3 |

## Main results

- **Exponential depth, not width.** The number of transitions grows roughly exponentially with depth but far more slowly (essentially as a base) with width, for a circular trajectory between MNIST datapoints (Fig. 2, p. 4); the same holds for trajectory length, including convolutional CIFAR-10 networks (Fig. 4, p. 5), where pooling layers *reduce* length (layers 3 and 7 in that figure).
- **Initialization scale matters.** Transition counts and trajectory length depend on the initialization scale $\sigma_w$; there is a threshold below which the exponential regime is not entered (Figs. 2, 4). Training itself pushes $\sigma_w^2$ upward: a CIFAR-10 network initialized at $\sigma_w^2=2$ has all layers' scaled weight variance grow during training (Fig. 9, p. 7), so even networks not initialized in the exponential regime are pushed into it.
- **Lower layers matter more.** Adding weight noise of magnitude $0.25$ to layer 1 of a trained CIFAR-10 conv net drops accuracy by $0.7$, while the same noise on layer 5 drops it by about $0.02$ (Fig. 6, p. 6). Conversely, training exactly one layer (all others frozen at random initialization) on MNIST, width $k=100$, $\sigma_w^2=1$, hard tanh, gives monotonically better performance the *lower* the trained layer, with the number of trained parameters held fixed (Fig. 7, p. 6).
- **Batch norm reduces trajectory length.** Without batch norm, trajectory length increases during training (Fig. 8, p. 6: $1189 \to 156250$ across the conv layers at the final step); with batch norm the batch-norm layers reduce the length at the same depth, stabilizing the representation (Fig. 10, p. 7).
- **Trajectory regularization matches batch norm.** Replacing every batch-norm layer by a trajectory-regularization layer gives comparable CIFAR-10 test accuracy, and considerably better than neither, while needing no train/test-time discrepancy (Fig. 11, p. 7).

## Limitations stated by the authors

The trajectory-length analysis (Thm. 3) is for fully connected networks; the convolutional growth is empirical (§2.2, p. 5). Theorem 4's transitions-length proportionality rests on the assumption that neurons are saturated except at rapidly crossed saturation endpoints (§2.2, p. 5). The paper says further gains from trajectory regularization "might be achieved through more sophisticated use of this method" (§4, p. 7), i.e. the presented version is basic.

## Critical assessment

The central contribution is a *positive*, all-weights statement that reverses the direction of prior lower-bound constructions: an upper bound on activation patterns that matches Montufar's hand-crafted lower bound, so those constructions are not extremal in a hidden way. Two caveats are worth stating. First, the printed "$\geq O(\cdot)$" in Theorem 3 is mathematically awkward (a lower bound on an expectation against a big-$O$); it reads like a typo for the asymptotic growth-rate claim, and Theorem 4's derivation treats it as such. Second, the appendix — where the proofs of Theorems 1–3 and Figures 12–18 live — is not part of this file, so the arguments cannot be checked from the paper as saved; the main text gives only a sketch (recurrence on $\delta z^{(d)}$ and its perpendicular component, §2.2, p. 5). The "all weights are not equal" claim is supported by two experiments (noise sensitivity and single-layer training), both on small image datasets (MNIST, CIFAR-10) with one architecture family each; the monotonicity in Fig. 7 is also partly confounded by which layer's parameters are trained even though the count is held fixed. As a *measure*, trajectory length is extremely sensitive to initialization scale, which makes cross-architecture comparisons delicate; the paper is careful to note the $\sigma_w^2=2$ case sits outside the exponential regime at step 0. The connection to adversarial examples is raised but not tested.

## Connections within the collection

The paper is a direct generalization of the depth-separation work it cites: it turns the hand-coded linear-region count of Montufar et al. 2014 [@NIPS2014_fa6f2a46] into an upper bound valid for *all* weight settings, and it uses the linear-region/response-region framing of Pascanu et al. 2013 [@pascanuNumberResponseRegions2013] as the baseline construction (*). It belongs to the same "count the linear pieces" programme later sharpened by Hanin and Rolnick's bounds on the number of linear regions [@haninComplexityLinearRegions2019] and activation patterns [@haninDeepReLUNetworks2019], by Serra et al. [@serraBoundingCountingLinear2018], and by Goujon et al. [@goujonNumberRegionsPiecewise2023]; the convex-polytope partition of Theorem 2 is the starting point of the polyhedral-complex literature in this collection, e.g. the combinatorial extraction of the regions [@masdenAlgorithmicDeterminationCombinatorial2022], their subdivision structure [@berzinsPolyhedralComplexExtraction2023; @brandenburgDecompositionPolyhedraPiecewise2024] and their use in visualizing decision boundaries [@humayunSplineCamExactVisualization2024]. The "weights are not equal" observation is thematically close to the space-folding and symmetry accounts of ReLU networks [@lewandowskiSpaceFoldsReLU2024; @grigsbyHiddenSymmetriesReLU], which offer a different explanation of why depth, and particular weights, do more work. Outside the collection, the natural follow-ups are the mean-field/curvature view of the same authors (Poole et al. 2016) and the adversarial-examples question raised in §4.
