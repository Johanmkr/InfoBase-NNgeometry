---
citekey: liuIntegratingGeometriesReLU2023a
title: "Integrating geometries of ReLU feedforward neural networks"
authors: [Yajing Liu, Turgay Caglar, Christopher Peterson, Michael Kirby]
year: 2023
venue: Frontiers in Big Data 6, article 1274831
doi: "10.3389/fdata.2023.1274831"
arxiv: null
url: null
kind: method
source_type: pdf
source_sha256: 55aea7356c0798f038ba774700e9176b2eb60bead4428fd7a0f95e1aa87c4485
pages: 13
ingested: 2026-10-15
tags: [relu-networks, polyhedral-decomposition, adversarial-examples, hamming-distance, linear-regions]
concepts_introduced:
  - binary vector representation of polyhedra
  - traversal-and-pruning method
  - Hamming distance between polyhedra
  - bisection method along shortest Euclidean path
  - Chebyshev center of a polyhedron
  - active bit
concepts_used:
  - ReLU feedforward neural network
  - polyhedral decomposition
  - linear region
  - piecewise affine function
  - linear programming
  - redundant inequality
builds_on: [liuReLUNeuralNetworks2023, vincentReachablePolyhedralMarching2021, masdenAlgorithmicDeterminationCombinatorial2022, sattelbergLocallyLinearAttributes2023]
datasets: [MNIST, CIFAR-10, toy 2D quadratic, toy 3D quadratic]
---

# Integrating geometries of ReLU feedforward neural networks

**Liu, Caglar, Peterson, Kirby (2023)**, *Frontiers in Big Data* 6:1274831. [doi:10.3389/fdata.2023.1274831](https://doi.org/10.3389/fdata.2023.1274831)
<!-- Yajing Liu is marked deceased (†) on the title page -->

!!! abstract "TL;DR"
    A ReLU feedforward network is a continuous piecewise affine map whose linear regions form a polyhedral decomposition of the input space. The paper packages the decomposition into an open-source toolbox that, for any data point, returns its binary activation vector, the minimal linear model of its polyhedron, its active bits (non-redundant facets), and all neighbouring polyhedra by single-bit flips. It is used to compare polyhedra via three lenses — Hamming distance between binary vectors, a bisection walk along the shortest Euclidean path, and the Chebyshev center and inscribed radius — finding that adversarial (FGSM) images live in systematically smaller polyhedra with larger Hamming distances than training and test points.

## Problem and motivation

A ReLU network computes a piecewise affine map $M:\R^m\to\R^n$; the pieces are convex polyhedra of a decomposition of the input space, and every point of one polyhedron shares both its activation pattern and its affine map. The aim is to make this geometry *usable*: to build a toolbox that integrates the several descriptions of a ReLU network (binary vector, linear inequality model, neighbour graph, inscribing ball) so that the network can be interrogated empirically (§1, §2). The motivation is diagnostic: polyhedron count measures expressivity, and polyhedron shape and placement bear on generalization, clustering and decision boundaries (§3.3, §5). Prior work gives bounds on the number of linear regions and algorithms to enumerate them; the stated contribution is a public, systematic implementation of the binary-vector representation plus the linear-model toolbox, with metrics previously used only in abstract form (Hamming similarity, Chebyshev centers) (§2, §4.2). Code is released at `https://github.com/cglrtrgy/GoL_Toolbox` (§2, Data availability statement).

## Method

**Model and binary vector.** An $(L+1)$-layer FFNN with ReLU on the first $L$ layers and a linear output layer maps $\R^m\to\R^n$ (§2.1). For an input $x$, the activation state of hidden unit $j$ in layer $i$ is a bit $s_{i,j}(x)\in\{0,1\}$ recording the sign of the pre-activation; stacking all bits gives the *binary vector* $s(x)$ of length $h=\sum_i h_i$. All points in one polyhedron share the same binary vector, so a polyhedron is identified by a bit string (§2.2).

**Linear model and active bits.** For a fixed binary vector $s$, the polyhedron is the intersection of half-spaces $Ax\le c$ (§2.2). Each facet corresponds to one inequality; an *active bit* is a non-redundant inequality. Redundancy is tested by solving, for row $i$, the LP $\max_x a_i^\top x$ over the other constraints: if the optimum does not exceed $c_i$, row $i$ is redundant and is dropped, yielding the minimal model $(A',c')$ and the set of active-bit indices (§2.2).

**Traversal-and-Pruning (Algorithm 1).** Starting from a random point $x^{(0)}$, compute $s(x^{(0)})$, store it with label 0, and repeatedly take any label-0 vector, solve the LP for its active bits, and flip each active bit one at a time to generate candidate neighbours; add the new strings, label them 0, and label the current one 1. When every stored vector has label 1, all polyhedra of the input space (or of a bounded region) have been enumerated (§2.4). Two polyhedra share a facet iff their binary vectors differ in exactly one active bit, so the graph of single-active-bit flips is the dual graph of the decomposition. The authors sketch an induction argument for completeness and connectivity (§2.4).

**Three geometric lenses.**

1. *Hamming distance* between two points' binary vectors, the number of activation-pattern differences, a proxy for the minimum number of polyhedra boundaries separating them (§3.1).
2. *Bisection method* (Algorithm 2): find the fraction $\delta$ with $C=A+\delta(B-A)$ exactly one Hamming step from the current left endpoint, then advance; the collinear samples have consecutive Hamming distance 1, approximating the polyhedra crossed along the Euclidean segment (§3.2).
3. *Chebyshev center*: for a bounded polyhedron in minimal form $A'x\le c'$, an LP maximizes the radius $r$ of the largest inscribed ball with center $x_c$; the center is a "representative point" and the radius a coarse size indicator (§3.3).

## Key equations and definitions

Layerwise feedforward map with ReLU on the hidden layers (§2.1, Eq. 2, p. 3):

$$
F_i(x) = \mathrm{ReLU}(W_i F_{i-1}(x) + b_i) \qquad \text{(Eq. 2, p. 3)}
$$

Binary bit of unit $j$ in hidden layer $i$ (§2.2, Eq. 3):

$$
s_{i,j}(x) =
\begin{cases}
1 & \text{if } w_{i,j}F_{i-1}(x) + b_{i,j} > 0,\\
0 & \text{if } w_{i,j}F_{i-1}(x) + b_{i,j} \le 0.
\end{cases}
\qquad \text{(Eq. 3, p. 3)}
$$

The full binary vector stacks all hidden layers (§2.2, Eq. 4):

$$
s(x) = \bigl[\,s_1(x)^\top \ \cdots \ s_L(x)^\top\,\bigr]^\top \in \R^h, \qquad h = \sum_{i=1}^{L} h_i
\qquad \text{(Eq. 4, p. 3)}
$$

Linear model of the polyhedron carrying $s(x)$ (§2.2, Eq. 5):

$$
Ax \le c, \quad
A = \bigl[A_1^\top \ A_2^\top \ \cdots \ A_L^\top\bigr]^\top, \quad
c = \bigl[c_1^\top \ c_2^\top \ \cdots \ c_L^\top\bigr]^\top,
\qquad \text{(Eq. 5, p. 3)}
$$

with $A_j = \mathrm{diag}(s'_j)\,\hat W_j$ and $c_j = \mathrm{diag}(s'_j)\,(-\hat b_j)$, where $s'_j$ is the sign vector ($s'_{i,j}=1$ if $s_{i,j}=0$, and $-1$ if $s_{i,j}=1$) and the recursive quantities are $\hat W_j = W_j\,\mathrm{diag}(s_{j-1})\,\hat W_{j-1}$, $\hat b_j = W_j\,\mathrm{diag}(s_{j-1})\,\hat b_{j-1} + b_j$ for $2\le j\le L$, with $\hat W_1=W_1$, $\hat b_1=b_1$ (§2.2). *(The extraction is ambiguous about whether the display prints $s_j$ or $s'_j$ in $A_j,c_j$; the sign vector $s'_j$ is the one defined for expressing inequality direction, §2.2.)*

Within the polyhedron the network is the single affine map (§2.2):

$$
G(x) = W_{L+1}\,\mathrm{diag}(s_L)\,\hat W_L\,x + W_{L+1}\,\mathrm{diag}(s_L)\,\hat b_L + b_{L+1}
\qquad \text{(§2.2)}
$$

Redundancy test for the $i$th inequality, with $\tilde A,\tilde c$ the rows/entries other than $i$ (§2.2, Eq. 6):

$$
\max_{x}\ a_i^\top x \quad \text{s.t. } \tilde A x \le \tilde c
\qquad \text{(Eq. 6, p. 3)}
$$

Chebyshev center LP for a bounded polyhedron $A'x\le c'$ (§3.3, Eq. 7):

$$
\max_{r,\,x_c}\ r \quad \text{s.t. } a_i'^\top x_c + r\,\norm{a_i'} \le c_i', \quad 1\le i\le l
\qquad \text{(Eq. 7, p. 5)}
$$

which is recast as the LP $\min_x -e^\top x$ s.t. $\hat A x \le c'$ and solved with `cvxpy` (§3.3, Eq. 8).

| Symbol | Meaning (in this paper) |
|---|---|
| $L$ | number of hidden layers (network has $L+1$ layers) |
| $W_i, b_i$ | weight matrix and bias of layer $i$ |
| $h_i$ | number of nodes in hidden layer $i$ |
| $F_i(x)$ | output of layer $i$ at input $x$ |
| $s(x)$ | binary vector of $x$; length $h=\sum_i h_i$ |
| $s'_j$ | sign vector of layer $j$ |
| $\hat W_j, \hat b_j$ | recursively folded weights/biases |
| $A, c$ | constraint matrix/vector of the polyhedron $Ax\le c$ |
| $A', c'$ | minimal (redundancy-free) constraint set |
| $a_i$ | $i$th row of $A$ |
| $x_c, r$ | Chebyshev center and inscribed radius |
| $M$ | the network's piecewise affine map $\R^m\to\R^n$ |

## Main results

**Toy networks.** For $f_1(x_1,x_2)=x_1^2+x_2^2-0.4$ with two hidden layers of 10 nodes each, Algorithm 1 returns 218 polyhedra inside $[-1,1]^2$ and 237 in $\R^2$, of which 211 are bounded and 26 unbounded (§4, Fig. 1); polyhedra grow and equal Euclidean distances span fewer Hamming steps away from the training region. A 3D model (three hidden layers of 10 nodes) yields 2,212 polyhedra in $[-1,1]^3$ (§4.1.3, Fig. 2). Sweeping architectures (50 initializations each), more layers at fixed width and more nodes at fixed depth both raise the polyhedron count and lower validation MSE: at 5 nodes/layer the average count rises from 55.10 (2 layers) to 588.80 (10 layers), average MSE 0.001389→0.000635; at 3 layers from 110.20 (5 nodes) to 2,225.60 (25 nodes), average MSE 0.000651→0.000014 (Tab. 1, p. 7).

**MNIST.** Five-hidden-layer networks with 300/350/400/450/500 nodes per layer all reach >98% train and test accuracy (§4.2). The average number of active bits per polyhedron grows with width: 658.30 (SD 64.07) at 300 nodes to 975.00 (SD 124.29) at 500 nodes (Tab. 2, p. 8); the redundancy LPs cost $O(m^3 h)$ each and scale with node count (§4.2, Fig. 3). Nearest-neighbour classification accuracy is higher under Hamming distance than under Euclidean distance on both train and test sets — e.g. structure 1: Hamming 99.50% (train) / 98.26% (test) versus Euclidean 97.37% / 95.58% (Tab. 3, p. 8).

**Fixed Euclidean distance, differing Hamming distance.** Holding the Euclidean distance at 0.5, 0.75 and 1.0 from a reference point, average Hamming distances for training and test points nearly coincide, but FGSM adversarial points differ sharply and consistently more: at distance 0.5, training 52.97±13.80 versus FGSM 75.27±13.37 (structure 1), and 129.98±56.91 versus 219.92±99.16 (structure 3); at distance 1.0, 131.82±35.73 versus 151.09±36.89 (structure 1) (Tab. 4, p. 9). The authors stress that the ranges overlap, so Hamming distance alone cannot decide adversarialness (§4.2.1.1).

**Bisection along the shortest Euclidean path.** For six nearest-neighbour pairs, the Hamming distance is close to the number of polyhedra crossed when it is small but increasingly underestimates it as it grows: pair 6 has Hamming distance 320 versus 351 polyhedra along the path (Tab. 5, p. 9). Over 1000 pairs, raising the number of layers from 1 to 5 (200 nodes each) leaves the mean number of polyhedra nearly flat (21→49) while the maximum grows from 41 to 303 (Fig. 4, p. 10).

**Chebyshev centers and radii.** On MNIST (1000 samples per set, structure 2) the inscribed radii are Train 0.23±0.081, Test 0.21±0.079, Adversarial 0.13±0.092 (Tab. 6, p. 11); training and test polyhedra have nearly coincident centers, adversarial ones are perceptibly smaller and differently centered (Fig. 5, 6). CIFAR-10 (8 hidden layers of 400 nodes, train accuracy 99.39%, test 53.59%) shows the same ordering at fixed Euclidean distance 0.1 with mean Hamming distances 109.47, 130.19, 142.80 (SD 40.44, 50.66, 63.42) for train, test, adversarial, and inscribed radii Train 0.038, Test 0.021, Adversarial 0.017 (Tab. 7, p. 12). Here training and test Hamming profiles *diverge*, which the authors attribute to the network's poor generalization on CIFAR-10 (§5).

## Limitations stated by the authors

- Exact polytope quantities are out of reach: vertex counting is combinatorial and bounding-sphere radius, exact volume infeasible; the Chebyshev center is only a coarse volume proxy (§3.3).
- Unbounded polyhedra must be bounded by adding per-dimension bounds before the Chebyshev LP applies (§3.3).
- The toolbox is demonstrated on toy problems, MNIST and CIFAR-10, i.e. modest architectures; "much larger real-world datasets" and "much more intricate" networks are left to future work (Conclusion).
- Redundancy testing is expensive ($O(m^3 h)$ per LP and one LP per bit); GPU scalability is future work (§4.2, Fig. 3).
- The paper calls itself "a proof of concept" (Conclusion).

## Critical assessment

The toolbox is a genuine contribution and the release of code matters, but the research claims are weaker than the tooling. The paper is mostly a methods/demonstration paper: the traversal-and-pruning enumeration and the binary-vector model are explicitly inherited from Liu et al. (2023) and Vincent and Schwager (2021) (§2.2–2.4), so the constructions named as introduced largely rename existing ones; the integration, not the pieces, is new.

Several reported quantities deserve caution. The traversal argument is only sketched: the "induction" paragraph (§2.4) asserts connectivity rather than proving that all polyhedra of an *unbounded* space are reached. Table 7 as printed is internally inconsistent — for the test set the reported "Min" 0.089 exceeds the "Max" 0.069 (Tab. 7, p. 12, sic). The CIFAR-10 result, with 53.59% test accuracy, undercuts the geometric story the paper wants to tell: the authors attribute the train/test Hamming divergence to "suboptimal generalization", an interpretation with no control (e.g. a better-generalizing CIFAR model). Adversarial examples come from one method (FGSM) on one network per architecture; the acknowledged overlap in Hamming distributions is never quantified, so the adversarial-detection framing is suggestive at best. Numbers are means/SDs over 30–1000 samples with no significance testing, and the nearest-neighbour comparison (Tab. 3) is a weak baseline.

Before trusting the geometry conclusions: does the Chebyshev-radius ordering survive normalization (radii are input-space units, so "smaller" adversarial polyhedra may simply sit elsewhere), and are the training seeds and CIFAR training recipe reproducible?

## Connections within the collection

The paper is a direct companion to [@liuReLUNeuralNetworks2023]: the binary vector, the linear model and the traversal method are all "from Liu et al. (2023)", and this paper adds the toolbox and the Hamming/Chebyshev analyses (§2.2–2.4). Its enumeration-by-neighbour-flipping is described as aligning with the reachable polyhedral marching of [@vincentReachablePolyhedralMarching2021], with the difference that it also emits binary vectors (§2.4). The combinatorial encoding of the polyhedral complex of [@masdenAlgorithmicDeterminationCombinatorial2022] is a fuller, all-dimensions counterpart to the facet-level encoding used here. The intuition about polyhedral structure and its redundancy echoes [@sattelbergLocallyLinearAttributes2023] (§2.2). The region-count bounds the toy experiments reference — [@NIPS2014_fa6f2a46], [@pascanuNumberResponseRegions2013], [@raghuExpressivePowerDeep2017], [@haninDeepReLUNetworks2019], [@serraBoundingCountingLinear2018], [@goujonNumberRegionsPiecewise2023] and [@haninComplexityLinearRegions2019] — are the theoretical backdrop to the depth/width sweep (§2, §4.1.2). The tropical reading of ReLU networks in [@zhangTropicalGeometryDeep] is another "geometry of ReLU" perspective named in the review (§2), and the visualization aims overlap with [@humayunSplineCamExactVisualization2024] and the complex extraction of [@berzinsPolyhedralComplexExtraction2023]. The Hamming/adjacency graph used here is close to the ReLU transition graphs of [@dhayalkarDiscreteFunctionalGeometry2025] (possibly related; the papers do not cite each other). Works outside the collection central to the setup are Jamil et al. (2022, 2023), whose binary-vector representation and adversarial-detection experiments this paper builds on.
