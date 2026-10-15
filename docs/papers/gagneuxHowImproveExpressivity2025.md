---
citekey: gagneuxHowImproveExpressivity2025
title: "How to Improve Expressivity of Convex ReLU Neural Networks?"
authors: [Anne Gagneux, Mathurin Massias, Emmanuel Soubies, Rémi Gribonval]
year: 2025
venue: null
doi: null
arxiv: null
url: null
kind: method
source_type: pdf
source_sha256: 3db0229f921d3be261a6965cf90faf71ae6d0285ccb34f9575501d6d8dbc3927
pages: 5
ingested: 2026-10-15
tags: [relu-networks, convexity, input-convex-neural-networks, expressivity, regularization]
concepts_introduced:
  - convex ReLU network characterization
  - isolated neuron
  - convex regularisation
concepts_used:
  - input convex neural network
  - activation regions
  - continuous piecewise linear function
  - path-lifting
  - path-activations
  - polyhedral decomposition
builds_on: [berzinsPolyhedralComplexExtraction2023, haninDeepReLUNetworks2019]
datasets: []
---

# How to Improve Expressivity of Convex ReLU Neural Networks?

**Gagneux, Massias, Soubies, Gribonval (2025)**, *HAL preprint*. [hal-05216923](https://hal.science/hal-05216923v1)
<!-- venue null: the .bib entry is @unpublished with no journal or proceedings. -->

!!! abstract "TL;DR"
    Convex functions are usually implemented with Input Convex Neural Networks (ICNNs), whose nonnegativity constraints are crude and hurt expressivity. Building on the companion characterization of convex ReLU networks, this short paper gives a pedagogical account of when a ReLU network is convex, shows empirically that convex ReLU networks vastly outnumber ICNNs at fixed architecture, and proposes a differentiable penalty that softly enforces convexity during training — recovering the accuracy of an unconstrained network (MSE 0.84) while staying convex, against 1.71 for the ICNN.

## Problem and motivation

Learning convex functions with neural networks matters for learning convex regularizers in inverse problems and for optimal transport maps (§1). The dominant approach are Input Convex Neural Networks (ICNNs): a standard multilayer perceptron whose hidden weight matrices are constrained to be nonnegative (§2.1). This guarantees convexity by design and needs only a small architectural change, but ICNNs "demonstrate poor expressivity when scaling up" (§1). The nonnegativity constraint looks arbitrary, which raises the paper's question: is it the only way to enforce convexity, or do many more convex ReLU networks exist that are not ICNNs?

A first answer, due to Chen, Shi and Zhang (outside the collection), is that any convex piecewise affine function on a compact domain can be implemented by an ICNN — but the constructive proof produces one neuron per affine piece and as many layers as pieces, useless in practice. The authors' companion paper [Gagneux et al., "Convexity in ReLU Neural Networks: beyond ICNNs?", outside the collection, ref. [5]] turns this into a statement *for a fixed architecture*: there exist convex functions implemented by a given ReLU network that are not implementable by any ICNN of the same width and depth. The present paper has two aims: a pedagogical dive into that characterization, and *empirical evidence* that ICNNs are only a small fraction of all convex ReLU networks, together with a new regularisation promoting convexity (§1).

## Method

The argument is built in three steps. First, a general principle for convex continuous piecewise linear (CPWL) functions: convexity need only be checked across *neighboring* regions, i.e. regions sharing a boundary of affine dimension $d-1$, and there via condition (1) (§2.3). This is a *minimal* set of convexity conditions, which matters because the goal is the smallest set of constraints that still guarantees convexity.

Second, the paper moves from CPWL functions to ReLU networks. A ReLU network implements a CPWL function whose regions are the *activation regions* — the sets of inputs on which every hidden neuron's activation $a_\nu(x,\theta) = \mathbb{1}_{z_\nu(x,\theta)>0}$ is constant (definition adopted from [@haninDeepReLUNetworks2019, Definition 1]; [@haninDeepReLUNetworks2019, Lemma 1] shows these are convex and partition the input space, §2.2). Between two neighboring activation regions at least one neuron switches; the paper focuses on points where exactly one neuron switches, the points of the set $X_\nu$. A neuron whose $X_\nu$ is nonempty is called *isolated* (Def. 2.1); activation switches at intersections of more than two regions (several neurons switching at once) are treated as degenerate and excluded, with genericity discussed in the companion paper (§2.4). With this, convexity of a ReLU network reduces to a finite set of sign conditions on sums of products of weights.

Third, to write those conditions the paper uses the path-norm toolkit of Gonon, Brisebarre, Riccietti and Gribonval (outside the collection, ref. [6]): the output is rewritten as $f(x,\theta)=\Phi(\theta)A(x,\theta)$ (Eq. 2), the *path-lifting* $\Phi(\theta)$ collecting, one row per path from a neuron to the output, the product of weights along that path, and the *path-activations* $A(x,\theta)$ recording which paths are active. Sub-graphs with a hidden neuron $\nu$ as single input give $\Phi^{\nu\to}(\theta)$ and the set $\mathfrak{a}^{\nu\to}$ of activation patterns realized on $X_\nu$ (§2.4.3).

The algorithmic contribution follows from Thm. 2.3. The naive check is prohibitively expensive: the number of paths grows exponentially with depth, and enumerating activation patterns grows exponentially with the number of neurons. Two ingredients make it tractable (§3): each scalar product in condition (4) is evaluated by a forward pass on a slightly modified network, avoiding explicit storage of $\Phi^{\nu\to}(\theta)$; and the sets $\mathfrak{a}^{\nu\to}$ — the frontiers of the CPWL function — are computed with the edge-subdivision algorithm of [@berzinsPolyhedralComplexExtraction2023], an efficient frontier extractor for fully-connected feedforward ReLU networks.

## Key equations and definitions

Neighboring-region convexity test (§2.3, Eq. 1): $f$ is convex iff for every neighboring pair $R_1 \sim R_2$ there is a pair of points $x_1 \in \mathrm{int}(R_1)$, $x_2 \in \mathrm{int}(R_2)$ with

$$
\langle \nabla f(x_1) - \nabla f(x_2),\, x_1 - x_2 \rangle \geq 0.
$$

Path-lifting decomposition (§2.4.3, Eq. 2), with $\Phi(\theta)$ the path-lifting vector and $A(x,\theta)$ the path-activations matrix (Eq. 2, p. 3):

$$
f(x,\theta) = \Phi(\theta)\, A(x,\theta).
$$

Gradient on an activation region (§2.4.3, Eq. 3): since $A(\cdot,\theta)$ is constant on an activation region $R_1$ with matrix $A_1$ (Eq. 3, p. 4),

$$
\nabla f(x_1) = A_1^{\top}\, \Phi(\theta).
$$

Characterization of convex ReLU networks (§2.4.3, Thm. 2.3): under a non-degeneracy assumption (Assumption 7.1 of the companion paper), $f_\theta$ is convex if and only if for every hidden neuron $\nu \in H$ (Eq. 4, p. 4),

$$
\min_{a \in \mathfrak{a}^{\nu\to}} \langle a,\, \Phi^{\nu\to}(\theta) \rangle \geq 0, \qquad \text{where } \mathfrak{a}^{\nu\to} := \{ a^{\nu\to}(x,\theta) : x \in X_\nu \}.
$$

Each binary $a$ is the activation pattern of a $(d-1)$-dimensional boundary where $\nu$ switches (§2.4.3). If all elementary vectors are present in $\mathfrak{a}^{\nu\to}$ for every $\nu$, each scalar product has a single term and (4) collapses to the ICNN nonnegativity constraints; when elementary vectors are missing, (4) is strictly less restrictive — this is exactly the gap the paper exploits.

Convex regularisation (§4.2, Eq. 5), the soft penalty that drives negative constraint values up to zero (Eq. 5, p. 5):

$$
R(\theta) := \lambda \sum_{\nu \in H} \sum_{a \in \mathfrak{a}^{\nu\to}} \max\!\left(0,\, -\langle a,\, \Phi^{\nu\to}(\theta) \rangle\right).
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $N$, $N_{\text{in}}$, $N_{\text{out}}$, $H$ | all neurons; input neurons; output neuron; hidden neurons |
| $\nu$ | a neuron |
| $\theta$ | network parameters |
| $z_\nu(x,\theta)$ | pre-activation of neuron $\nu$ at input $x \in \R^d$ |
| $a_\nu(x,\theta)$ | activation $\mathbb{1}_{z_\nu(x,\theta)>0}$ |
| $R_1 \sim R_2$ | neighboring regions (shared boundary, affine dimension $d-1$) |
| $X_\nu$ | inputs where, in every small enough neighbourhood, only $\nu$'s activation switches (Def. 2.1) |
| $G = (N,E)$ | DAG describing the architecture (neurons as vertices, weights as edges) |
| $\Phi(\theta)$ | path-lifting: one row per path, product of weights along it |
| $A(x,\theta)$ | path-activations matrix, rows = paths, $d+1$ columns |
| $\Phi^{\nu\to}(\theta)$, $\mathfrak{a}^{\nu\to}$ | path-lifting of the sub-graph with $\nu$ as single input; its realizable activation patterns |
| $\lambda$ | regularisation strength |
| $d$ | input dimension |

## Main results

**Theory.** One hidden layer: convex ReLU networks and ICNNs implement the same functions. Prop. 2.2 states that any convex function implemented by a one-hidden-layer ReLU network with weighted input skip connections can also be implemented by a one-hidden-layer ICNN *with the same width* (§2.4.1). Two hidden layers: the equivalence already fails. The characterization (4) becomes a condition on some sums of products of weights being nonnegative, which is strictly weaker than requiring every hidden weight to be nonnegative (§2.4.2), and the companion paper exhibits a convex two-hidden-layer ReLU network not implementable by any ICNN of the same architecture.

**Experiments — how rare are ICNNs?** Random networks are sampled: last-layer weights from a folded normal (nonnegative, a necessary convexity condition), other hidden layers from a normal (possibly negative entries). For two-hidden-layer networks of varying widths (10k draws, $d=2$) and for varying depth (5k draws, $n=(w,\dots,w)$, $d=3$), the number of convex ReLU networks is compared with the number of ICNNs (§4.1). Convex ReLU networks "largely out-number" ICNNs; the count of convex networks grows with depth at fixed width — somewhat unexpectedly — and there are more convex networks in "decoder-like" architectures ($n_2 > n_1$) than in "encoder-like" ones. Beyond three hidden layers the number of ICNNs among the sampled convex networks is consistently zero, so it is not plotted. The input dimension is reported to have no effect (not reported in the figures), as expected for ICNNs, whose constraint acts on hidden layers only. The numeric cell counts of the width/depth heatmaps (Fig. 6) could not be extracted reliably from the file's text layer, so only the trends stated in the text are reported here.

**Experiments — regularisation.** A 2D CPWL target is approximated from uniform samples on $[-10,10]^2$ with an $\ell_1$ loss, comparing an unconstrained ReLU network, an ICNN, and a convex-regularised ReLU network, all with architecture $n=(4,8)$ and the same random initialisation (§4.2). Table 1 reports the mean square error on $N_{\text{val}} = 20$K uniformly sampled points, mean $\pm$ standard deviation over 10 initialisations: unconstrained $0.90 \pm 0.53$, ICNN $1.71 \pm 0.90$, convex regularisation $0.84 \pm 0.73$ (Tab. 1, p. 5). The regularised network thus attains the best accuracy of the three while being pushed towards convexity — the paper's headline claim that the penalty trades little accuracy for convexity, whereas the ICNN pays a large accuracy penalty.

## Limitations stated by the authors

The main bottleneck is computational: the sets $\mathfrak{a}^{\nu\to}$ require enumerating frontiers and regions, whose number grows exponentially with the number of neurons, and the penalty (5) requires these sets at every optimisation step. The conclusion states plainly that "a computational challenge is to scale it to practical networks" and that doing so "would open new ways to train more expressive convex networks" (§5). Degenerate cases — frontiers at which several neurons switch simultaneously — are excluded by the non-degeneracy assumption behind Thm. 2.3, with genericity discussed only in the companion paper (§2.4, §2.4.3).

## Critical assessment

This is a careful but deliberately short preprint: essentially a pedagogical restatement of the companion paper's characterization plus a first numerical exploration, with no theorem proved here (Thm. 2.3 is recalled, not proved) and no formal complexity analysis of the regulariser. The empirical evidence is suggestive rather than decisive: the "convex ReLU networks vastly outnumber ICNNs" claim rests on random sampling from folded-normal/normal weights, not on a natural measure over convex functions, so it shows the *architecture constraint set* is large, not that ICNNs fail on realistic tasks. The single approximation experiment uses one 2D synthetic target, one architecture, and one $\lambda$; the standard deviations are large (ICNN $1.71 \pm 0.90$), so the ordering of $0.84$ versus $0.90$ is within noise and the convex-regularised network is itself not guaranteed convex — the penalty is soft. The regularisation (5) is only meaningful once $\mathfrak{a}^{\nu\to}$ is known exactly, i.e. it presupposes a full polyhedral decomposition of the current network, which the authors acknowledge is the practical blocker; no surrogate or sampling scheme for large networks is proposed. Code is released (github.com/annegnx/beyondicnn), which helps reproduction. The most useful open questions the paper raises are quantitative: whether the regulariser with an approximate $\mathfrak{a}^{\nu\to}$ still yields convex networks in practice, and whether hard constraints derived from (4) can replace the crude nonnegativity of ICNNs at realistic scale.

## Connections within the collection

The paper's algorithmic feasibility rests directly on [@berzinsPolyhedralComplexExtraction2023], whose edge-subdivision routine computes the frontiers of the CPWL function — the sets $\mathfrak{a}^{\nu\to}$ — and is cited as the tool for enumerating convexity conditions (§3). The definition of activation regions and their convex-partition property is adopted from [@haninDeepReLUNetworks2019, Definition 1; Lemma 1] (§2.2). Conceptually it sits next to the polyhedral-complex view of ReLU networks developed in [@liuReLUNeuralNetworks2023] and [@brandenburgDecompositionPolyhedraPiecewise2024], where the geometry of regions and their boundaries governs what a network can represent; the convexity condition (4) is precisely a statement about slopes across neighbouring regions and so about that geometry. It is also close in spirit to [@averkovEXPRESSIVENESSRATIONALRELU2025], which studies expressiveness of ReLU-type networks through algebraic constraints on weights rather than via region counting. Finally the count-of-regions literature — [@goujonNumberRegionsPiecewise2023; @serraBoundingCountingLinear2018; @masdenAlgorithmicDeterminationCombinatorial2022] — provides the combinatorial background against which the enumeration cost of $\mathfrak{a}^{\nu\to}$ should be judged. The ICNN baseline and the companion characterization (Amos et al. 2017; Gagneux et al. 2025, "beyond ICNNs?") are outside this collection.
