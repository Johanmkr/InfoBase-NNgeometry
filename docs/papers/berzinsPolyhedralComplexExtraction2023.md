---
citekey: berzinsPolyhedralComplexExtraction2023
title: "Polyhedral Complex Extraction from ReLU Networks Using Edge Subdivision"
authors: [Arturs Berzins]
year: 2023
venue: Proceedings of the 40th International Conference on Machine Learning (PMLR 202)
doi: "10.48550/arXiv.2306.07212"
arxiv: "2306.07212"
url: null
kind: method
source_type: pdf
source_sha256: 7ae3c3f5fe9b1e9000dbcf0070da1a82574327ee830d8c091c887f3deb5fb354
pages: 11
ingested: 2026-10-15
tags: [relu-networks, polyhedral-complex, piecewise-linear, computational-geometry, gpu-acceleration, implicit-shapes]
concepts_introduced:
  - edge subdivision
  - sign-vector
  - polyhedral complex extraction
  - 1-skeleton
  - perturbation of sign-vectors
  - intersecting edge
  - geometric loss on the polyhedral complex
concepts_used:
  - polyhedral complex
  - hyperplane arrangement
  - folded hyperplane arrangement
  - ReLU network
  - piecewise affine function
  - linear regions
  - intersection-poset
  - H-representation
  - V-representation
builds_on: [raghuExpressivePowerDeep2017, humayunSplineCamExactVisualization2024, serraBoundingCountingLinear2018, haninDeepReLUNetworks2019, haninComplexityLinearRegions2019]
datasets: []
---

# Polyhedral Complex Extraction from ReLU Networks Using Edge Subdivision

**Berzins (2023)**, *Proceedings of the 40th International Conference on Machine Learning (PMLR 202)*. [arXiv:2306.07212](https://arxiv.org/abs/2306.07212)

!!! abstract "TL;DR"
    A ReLU network is a continuous piecewise-affine (CPWA) function supported on a polyhedral complex, but extracting that complex exactly is combinatorially expensive. Berzins argues that the standard region-subdivision view recomputes the same new vertices on all $2^{D-1}$ regions sharing an edge, and instead subdivides only the edges (the 1-skeleton) of the arrangement. The method is fully parallel, runs in linear time and memory in the number of vertices, executes on a GPU, and is roughly 20 times faster than the region-subdivision baseline SplineCam on 2D inputs.

## Problem and motivation

Exact extraction of the polyhedral complex induced by a ReLU network is a prerequisite for using network geometry in expressivity, robustness, training, and neural-implicit-shape applications (§1, p. 1). Three families of exact methods exist: mixed-integer linear programming [@serraBoundingCountingLinear2018], state-flipping, and region subdivision, the "most natural and widely discussed" one, in which the complex is refined neuron by neuron and layer by layer (§2.1, p. 3). Sampling neuron states on a grid only approximates the complex (it yields a lower bound on the number of regions) (§2.1).

The paper's central observation is that region subdivision is redundant. Because the activation is continuous, a folded hyperplane stays continuous across folds, so the two new cells that appear on neighbouring regions share the same vertex on their common edge (Fig. 4). Processing each region independently therefore recomputes each new vertex $2^{D-1}$ times (an edge has $2^{D-1}$ ascendant regions in an unbounded arrangement), and in H-representation performs the same redundancy check on all those regions (§4, p. 5). The goal is to exploit this continuity and operate only on the unique vertices and edges.

## Method

The method keeps, at each iteration, a bounded-domain 1-skeleton: a vertex set $V$ and an edge set $E$, where each bounded edge has exactly two vertices. For each neuron of each layer sequentially, edge subdivision performs five steps (Fig. 3):

1. Evaluate the network at all vertices $V$;
2. obtain the sign-vectors of the vertices w.r.t. the current folded hyperplane;
3. find *splitting edges*, i.e. edges whose two vertices have opposite signs;
4. for each splitting edge, locate the new vertex by linearly interpolating vertex positions weighted by pre-activations, and split the edge;
5. build *intersecting edges* connecting the new vertices across each splitting 2-face.

Step 5 is the subtle one. New edges are also formed where the hyperplane cuts a 2-face, but tracking all $k$-cells is impractical for $k>1$ because a bounded $k$-cell may have an arbitrary number of facets. The paper intersects 2-faces *implicitly*: every bounded splitting 2-face has exactly two splitting edges, so once the splitting edges are known, the 2-faces are recovered by perturbation (§4.2.2, p. 6).

*Perturbation* turns sign-vectors into combinatorics (§4.1). For an unbounded domain, a $k$-cell has $D-k$ zeros in its sign-vector, and flipping one zero at a time to $+$ or $-$ produces its $2(D-k)$ parent cells; iterating from the 1-skeleton reconstructs the whole intersection-poset, including the regions. On a bounded domain defined by $m$ half-spaces, boundary cells have zeros among the first $m$ signs; perturbing only toward the interior, a $k$-cell has $z + 2(D-k-z)$ parents, where $z$ is the number of zeros in the first $m$ signs. This also gives the (optional) region-counting post-processing.

A by-product is that the sign-vector of a cell is inherited along the subdivision, so the combinatorial structure is carried by tensor rows rather than pointer graphs; this is what makes steps 0–4 standard tensor operations on a GPU (§4, §5.1).

## Key equations and definitions

The sign-vector of a point assigns its position relative to each hyperplane of the arrangement (Eq. 1, §3.3):

$$
\sigma_i(x) =
\begin{cases}
+ & \text{if } x \in H_i^{+},\\
0 & \text{if } x \in H_i,\\
- & \text{if } x \in H_i^{-}.
\end{cases}
\qquad \text{(Eq. 1)}
$$

A cell $C$ of the arrangement equals the intersection of the closed half-spaces selected by its sign-vector $\sigma(C)$; this is how cells are read off combinatorially (Eq. 2, §3.3):

$$
C = \bigcap_{i=1}^{m} H_i^{\sigma_i(C)} =: H_{\sigma(C)}. \qquad \text{(Eq. 2)}
$$

The network is a composition of $L$ ReLU layers, and its folded hyperplanes are the zero-level sets of the pre-activations, evaluated at the input $x$ through the previous layers (§3.4):

$$
f_\Theta(x) = (f_\Theta^{(L)} \circ \cdots \circ f_\Theta^{(1)})(x), \qquad \text{(Eq. 3)}
$$

$$
x^{(l)} = f_\Theta^{(l)}(x^{(l-1)}) = \mathrm{ReLU}\!\left(W^{(l)} x^{(l-1)} + b^{(l)}\right). \qquad \text{(Eq. 4)}
$$

The $i$-th neuron of layer $l$ induces the folded hyperplane $H_i^{(l)} := \{\,x \in \R^D \mid W_i^{(l)\top} x^{(l-1)} + b_i^{(l)} = 0\,\}$ (§3.4, unnumbered). The number of parents of a $k$-cell on a bounded domain, $z + 2(D-k-z)$, is the combinatorial engine of the method (§4.1).

| Symbol | Meaning (in this paper) |
|---|---|
| $D$ | input dimension |
| $x^{(l)}$ | activations of layer $l$; $x^{(0)} = x$ |
| $W^{(l)}, b^{(l)}$ | weights and biases of layer $l$ |
| $H_i^{(l)}$ | folded hyperplane of neuron $i$ in layer $l$ |
| $\sigma(x), \sigma(C)$ | sign-vector of a point / of a cell |
| $C$ | a cell of the polyhedral complex; $H_{\sigma(C)}$ its half-space intersection |
| $\mathcal{C}$, $\lvert\mathcal{C}\rvert$ | polyhedral complex and its domain |
| $\mathcal{C}^k$ | the $k$-skeleton of $\mathcal{C}$ |
| $V, E, \hat{E}$ | vertices, edges, splitting edges at the current iteration |
| $z$ | number of zeros in the first $m$ signs of a cell (boundary zeros) |
| $c = 4\pi A / P^2$ | shape compactness used as a geometric loss |
| $A, P$ | area and perimeter of the shape boundary |

## Main results

**Validation.** For a fixed 4-layer, width-10 network the neurons whose sign-vector entry is $0$ evaluate to $0$ at each extracted vertex up to numerical error; the maximum error is roughly seven orders of magnitude smaller than the extent of the validation hypercube domain (Fig. 6, left; §5.1.1, p. 7). This is stated as a necessary but not sufficient condition.

**Performance.** Counting vertices and edges on the bounded unit hypercube for 4-layer networks of width 10, 20, 40 and dimensions $D = 1,\dots,10$ on an NVIDIA RTX 3090, vertex and edge counts grow subexponentially in $D$ while the number of regions grows exponentially (Fig. 8, §5.1.2). Runtime and memory scale log-linearly in the number of vertices, which the paper attributes to the sub-optimal sorting-based implementation of step 5 (Fig. 7). Against SplineCam [@humayunSplineCamExactVisualization2024], a region-subdivision method restricted to $D=2$, edge subdivision is on average 20 times faster over the considered tests, because SplineCam uses CPU graph structures (Fig. 9).

**Vertex distribution.** Running on a $D = [-100, 100]^D$ hypercube with five random 4-layer, width-10 networks, the distance of vertices from the origin is bimodal: an exponentially decaying interior mode ($r < 100$) from the folded hyperplane arrangement, and a boundary mode ($r \geq 100$) from the domain facets (Fig. 6, right; §5.2). The authors note a trained network would likely concentrate the first mode near the data.

**Geometric loss.** With a network of two hidden layers of 50 neurons ($D=2$) first trained as a neural implicit representation of the Stanford bunny, using shape compactness $c = 4\pi A/P^2$ (equal to 1 for a circle) as the loss deforms the boundary contour from the bunny to a circle in 100 iterations of Adam (Fig. 10; §5.3). Any loss depending on vertex positions — edge lengths, angles, areas, volumes, curvatures — can be optimized this way.

**Pruning.** Two strategies are demonstrated. Parameter pruning removes neurons whose folded hyperplane does not intersect the shape (dying ReLUs, or hyperplanes meeting the domain but not the shape), compressing a $2,50,50,1$ network to $2,25,19,1$ and reducing parameters from 2751 to 589 (Fig. 11; §5.4.1). Pruning during edge subdivision discards an edge when both its vertices share signs w.r.t. all future neurons; on the 2D bunny this reduces edges from $757/2424/2576$ to $301/76/228$ and vertices from $399/1240/1316$ to $305/118/194$ after each layer (Fig. 12; §5.4.2).

## Limitations stated by the authors

The method is impractical for $D>8$ even for small networks, due to the exponential growth of the complex (mentioned in §1, p. 2, and shown in Fig. 8). Operating on a bounded domain inserts artificial boundary cells that can dominate the count, which is undesirable for performance; replacing the hypercube by a simplex (linear vertices, quadratic edges in $D$) is suggested, and extending to unbounded domains is future work (§5.2, §6). Step 5 is only log-linear because hash-table pairing on the GPU is non-trivial, and replacing sorting with hash tables is named as the main outlook for linear time and memory (§5.1, §6). The method assumes a generic arrangement (citing Theorem 3 of Grigsby & Lindsey, 2022, outside the collection) and is presented for plain ReLU networks, though the authors say the key ideas generalize to any CPWA network (§3.4, §6). Extension to non-generic arrangements and other CPWA architectures is future work (§6).

## Critical assessment

The redundancy argument is convincing and the collapse to a 1-skeleton is genuinely simpler than tracking regions, so the engineering win (GPU tensor ops, 20× over SplineCam) is plausible. But several claims rest on thin evidence. The validity check (§5.1.1) only verifies that zero-sign neurons vanish at the extracted vertices — a necessary condition the authors explicitly call insufficient — and the geometric error in vertex *positions* is "closely related, but not quantified here"; the headline "seven orders of magnitude" therefore measures pre-activation residual, not complex accuracy. The performance study is confined to randomly initialised 4-layer networks of width $\leq 40$ and $D \leq 10$; no trained large-scale network is timed, and the region counts (which would be the honest measure of complex size) are not reported for those cases. The only external baseline is SplineCam, which is itself limited to $D=2$; no comparison is made against mixed-integer approaches [@serraBoundingCountingLinear2018], Analytical Marching (Lei & Jia, 2020/2021, outside the collection) or cell-enumeration methods such as RPM (outside the collection), so "previous methods" in the 20× claim means one 2D baseline. The pruning-during-subdivision numbers are for a single 2D shape and the condition is acknowledged as merely sufficient. Reproducibility is better than most: code is released (`github.com/arturs-berzins/relu_edge_subdivision`), the algorithm is fully specified, and the complexity analysis is given in Appendix A. Open directions the paper itself implies: an actual hash-table step 5, an unbounded-domain variant, and whether the implicit 2-face construction can be turned into a level-set-only extractor competitive with Analytical Marching.

## Connections within the collection

The paper is a direct extension of *region subdivision*, the method described by [@raghuExpressivePowerDeep2017] and implemented by the works it competes with; [@humayunSplineCamExactVisualization2024] (SplineCam) is both the closest relative — region subdivision specialised to $D=2$ and the only benchmark here — and the baseline it beats. It uses [@haninDeepReLUNetworks2019] twice: for the $o(N/D!)$ bound on the number of regions and, crucially, for the bound $|\hat{E}| < |E| D/i$ (their Theorem 5) that turns the per-step counting into the claimed overall $O(|V|)$. The related-work discussion of complexity and region-subdivision implementations rests on [@haninComplexityLinearRegions2019], and the exact-counting alternative it contrasts with is the MILP of [@serraBoundingCountingLinear2018]. It also cites the foundational region-counting line [@NIPS2014_fa6f2a46; @pascanuNumberResponseRegions2013], which the collection covers.

Other collection papers address the same object — the polyhedral complex of a CPWA network — from different angles and are likely relevant to anyone studying this paper: algorithmic recovery of the full combinatorial structure [@masdenAlgorithmicDeterminationCombinatorial2022], cell-by-cell enumeration for verification [@vincentReachablePolyhedralMarching2021], the dual complex and persistent homology of the decomposition [@liuReLUNeuralNetworks2023], the geometry of space folds [@lewandowskiSpaceFoldsReLU2024], the preimage of rectifier activities [@carlssonPREIMAGERECTIFIERNETWORK2017], locally linear attributes of regions [@sattelbergLocallyLinearAttributes2023], and region-counting bounds [@goujonNumberRegionsPiecewise2023]. The max-affine spline view of the complex it invokes as motivation [@balestrieroSplineTheoryDeep2018] is closely related. The author's later geometry-informed neural networks [@berzinsGeometryInformedNeuralNetworks2024] build on the neural-implicit-shape setting that motivates the bunny experiment here (possibly related; the two papers share an author rather than a method).
