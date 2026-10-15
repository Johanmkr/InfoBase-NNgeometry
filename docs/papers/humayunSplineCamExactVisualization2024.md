---
citekey: humayunSplineCamExactVisualization2024
title: "SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries"
authors: [Ahmed Imtiaz Humayun, Randall Balestriero, Guha Balakrishnan, Richard Baraniuk]
year: 2024
venue: arXiv preprint
doi: 10.48550/arXiv.2302.12828
arxiv: "2302.12828"
url: null
kind: method
source_type: pdf
source_sha256: 9eb85c2ab68fdf5fe9568f823606d789b242da6ab19585c45102ead5c262114d
pages: 21
ingested: 2026-10-15
tags: [exact-geometry, decision-boundary, spline-theory, linear-regions, visualization, interpretability]
concepts_introduced:
  - SplineCam
  - exact spline partition
  - partition statistics
  - decision boundary sampling
  - boundary back-projection
concepts_used:
  - continuous piecewise-linear networks
  - max-affine spline operator
  - linear regions
  - hyperplane arrangement
  - ReLU
  - implicit neural representation
  - signed distance function
  - positional encoding
builds_on: [balestrieroSplineTheoryDeep2018]
datasets: [MNIST, Fashion-MNIST, CIFAR-10, tinyImageNet-200, MetFaces, Stanford Bunny]
---

# SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries

**Humayun, Balestriero, Balakrishnan, Baraniuk (2024)**, *arXiv preprint*. [arXiv:2302.12828](https://arxiv.org/abs/2302.12828)

!!! abstract "TL;DR"
    SplineCam is the first provably exact method for computing the input-space partition and decision boundary of a deep network built from continuous piece-wise linear (CPWL) activations over a chosen 2-D slice of the input space. It enumerates the exact linear regions layer by layer, back-projects each deeper-layer hyperplane onto the current region, and finds the region cycles with a graph algorithm; no sampling or architecture simplification is used. The resulting exact geometry supports decision-boundary visualization and sampling, and two partition statistics — region volume and number of regions — that let architectures and training regimes (convolutional vs. fully connected, with vs. without data augmentation) be compared quantitatively.

## Problem and motivation

The parameters of a deep network cannot be read as its function: permuting rows of $W^\ell,b^\ell$ and columns of $W^{\ell+1}$, or rescaling $W^\ell,b^\ell$ by $\kappa$ and $W^{\ell+1}$ by $1/\kappa$ for a ReLU network, all leave the input–output mapping unchanged (§1, p. 2). Activation-based interpretability is instead vulnerable to feature-adversarial attacks that change activations without crossing the decision boundary (§1, p. 2). Existing methods for locating the decision boundary — sampling it, or minimizing a loss that is zero on the boundary — are approximate and have focused on better losses and sampling strategies rather than exactness (§1, p. 2).

For networks whose nonlinearities are CPWL — (leaky-)ReLU, absolute value, maxout, max-pooling — the whole network is itself CPWL: affine on each cell of a partition of its domain. Prior work used this partition for expressivity measures, complexity bounds and robustness certificates, but all of it approximated the partition (§1, p. 2). The authors' claim is blunt: there is no exact (up to machine precision) method to compute the decision boundary of a deep network (§1, p. 2), and they set out to build one, restricted to 2-D domains of the input space for tractability and visualization.

## Method

SplineCam builds the exact input-space partition $\Omega$ of a network $S$ on a 2-polytope $P \subset \R^S$ (§2.2, p. 3). It exploits the CPWL structure: layer 1's rows are hyperplanes $h^1_i$, so partitioning $P$ with them gives $\Omega^1$; for each region $\omega \in \Omega^1$ the layer-2 hyperplanes are pulled back onto $\omega$ using the region's affine parameters (Lemma 2), giving a refined $\Omega^2$; this is repeated through layer $L$ (§2.2, p. 3).

Concretely:

1. **Find partitions (Alg. 1).** Given a 2-polytope $P$ and hyperplanes $H$, solve for all intersections of each $h_i$ with the other hyperplanes and with the edges of $P$ to get nodes $N$; sort the nodes on each $h_i$ and connect consecutive ones to get edges $E$; the cells are the cycles of the resulting planar graph (§2.2, Alg. 1, p. 3).
2. **Find cycles (Alg. 2).** From an undirected graph $G=(E,N)$ with boundary edges, make the graph bidirectional and, starting from a boundary edge $e_s$, run a breadth-first search to find the shortest return path, which yields one cycle; remove the traversed edges and repeat (§2.2, Alg. 2 and Fig. 3, p. 4). Non-boundary edges may be traversed twice, once per direction.
3. **Back-project and repeat.** For each region, deeper-layer hyperplanes are projected onto the region's tangent space via Lemma 2, the partition graph is recomputed, and the process repeats to layer $L$ (§2.2, p. 3).

Because everything but the cycle search is vectorizable and runs on GPU, and because the number of intersections, edges and cycles of a line arrangement is $O(n^2)$ (Alg. 1's step reduces to an arrangement of lines problem), the method scales: with $n \to \infty$ the expected number of edges per cycle is $O(1)$ and the average-case cost of Alg. 2 is $O(n)$ (§2.2, p. 4).

The library wraps any PyTorch *sequential* network, takes a polytopal 2-D region of interest and its orthogonal projection matrix, and includes a `verify()` check that the affine and forward computations agree (§E, p. 20). Max-pool layers are replaced by average pooling, and VGG weights are stored sparsely because double precision is needed to solve the linear systems (§B, p. 12).

## Key equations and definitions

$$
z^\ell = \sigma\!\left(W^\ell z^{\ell-1} + b^\ell\right), \quad \ell = 1,\dots,L \qquad \text{(Eq. 1, p. 1)}
$$

The network is a composition of $L$ layers; $\sigma$ applies an element-wise CPWL activation.

$$
S^\ell(x) = \sum_{\omega \in \Omega} \left(A^\ell_\omega x + b^\ell_\omega\right)\mathbb{1}\{x \in \omega\} \qquad \text{(Lemma 1, Eq. 2, p. 3)}
$$

The composition of layers $1$ to $\ell$ is an affine map per region $\omega$, written with the indicator $\mathbb{1}\{\cdot\}$ (the paper prints the indicator as ✶). This is what makes the partition the right object: knowing $A^\ell_\omega,b^\ell_\omega$ for all $\omega$ is knowing the function.

$$
A^\ell_\omega = \prod_{i=1}^{\ell} \operatorname{diag}(q^\omega_i)\, W^i
\qquad\text{(Eq. 3, p. 3)}
$$

$$
b^\ell_\omega = \operatorname{diag}(q^\omega_\ell)\, b^\ell + \sum_{i=1}^{\ell-1}\left(\prod_{j=i+1}^{\ell}\operatorname{diag}(q^\omega_j)\, W^j\right)\operatorname{diag}(q^\omega_i)\, b^i
\qquad\text{(Eq. 4, p. 3)}
$$

Here $q^\omega_i$ is the point-wise derivative of $\sigma$ at the pre-activation $W^i z^{i-1}+b^i$ for region $\omega$; by a result of Balestriero et al. it is unique for any $\omega$ (§2.1, p. 3). The sub/superscript stacking of Eqs. 3–4 is garbled in the extracted text; the forms above are the standard per-region affine parameters and the paper's notation, but the exact index ranges should be checked against the published PDF.

$$
h^\ell_i \triangleq \left\{z \in \R^{\ell-1} : \langle w^\ell_i, z\rangle + b^\ell_i = 0\right\} \qquad \text{(Eq. 5, p. 3)}
$$

The hyperplane induced by the $i$-th row $(w^\ell_i,b^\ell_i)$ of layer $\ell$.

$$
\operatorname{proj}_\omega(h^\ell_i) = \left\{x \in \R^S : \langle w^\ell_i, A^{\ell-1}_\omega x + b^{\ell-1}_\omega\rangle + b^\ell_i = 0\right\} \qquad \text{(Lemma 2, Eq. 6, p. 3)}
$$

Back-projection of a layer-$\ell$ hyperplane onto region $\omega$'s tangent space in the input space.

**Theorem 1 (§2.2, p. 3).** For a binary classifier with a single output neuron, the decision boundary in $\R^{L-1}$ is the hyperplane $h^L_1$; in the input space it is $\bigcup_{\omega \in \Omega}\{\operatorname{proj}_\omega(h^L_1) \cap \omega\}$. The proof is a repeated application of Lemma 2 back to layer 1.

The supplementary background formalizes each layer as a **max-affine spline operator**:

$$
\mathrm{MASO}(v; \{A_r, b_r\}_{r=1}^R) = \max_{r=1,\dots,R} A_r v + b_r \qquad \text{(Eq. 7, p. 11)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $S$ | input-space dimension |
| $x \in \R^S$ | input vector |
| $z^\ell$ | activations of layer $\ell$ |
| $W^\ell, b^\ell$ | weights and biases of layer $\ell$ |
| $\sigma$ | element-wise CPWL activation |
| $\Omega$ | input-space partition into $S$-dimensional regions |
| $\omega \in \Omega$ | one region (linear region) of the partition |
| $A^\ell_\omega, b^\ell_\omega$ | per-region affine parameters of layers $1..\ell$ |
| $q^\omega_i$ | point-wise derivative of $\sigma$ at layer $i$ for region $\omega$ |
| $h^\ell_i$, $w^\ell_i$, $b^\ell_i$ | hyperplane of row $i$ of layer $\ell$ |
| $P$ | the 2-polytope (2-D region of interest) being partitioned |
| $H$ | set of hyperplanes intersecting $P$ |
| $R$ | number of affine pieces of a MASO |
| $\mathrm{MASO}$ | max-affine spline operator (a CPWL layer) |

## Main results

- **Exactness and scale.** SplineCam computes the exact partition and boundary with no sampling or simplification, on 2-D domains, for fully connected, convolutional and skip-connected (VGG11/VGG16, 138M params) networks (§2.2, p. 4). A single-layer MLP with width 1000 and input dimension 8002 (8M params) yields 132569 regions in 134 s (Fig. 18, p. 17); VGG16 pre-trained on tinyImageNet takes ≈7 minutes for 1K regions (§2.2, p. 4).
- **Architecture comparison (Tab. 1, p. 7).** On the same 2-D slice, convolutional networks form far more regions of smaller volume than MLPs, even with fewer parameters. For MLP (44,860 params) vs. CONV (39,780 params): MNIST average region volume 3.144e-4 vs. 1.134e-5 and number of regions 318 vs. 8814; Fashion-MNIST 4.991e-4 vs. 3.54e-5 and 1364 vs. 28222. The average number of vertices is 4 in every row, matching the text's claim that the average number of edges per region converges to 4 (§2.2, p. 4). The fourth column is printed under a header "Ecc." with values 102e7, 36e7, 17e7, 14e7 — see the critical assessment; the relative ordering does show smaller (not larger) eccentricity for the convolutional networks, matching the text (§4.1, p. 7).
- **Task complexity.** Identical width-256 depth-6 MLPs trained on an easy SDF (ESDF) and a convoluted one (HSDF) from a MetFaces image develop very different partitions: the harder HSDF has a finer partition and higher region density, suggesting harder tasks use more of the parameters (§3.1, p. 5). In a 3-D SDF of the Stanford Bunny, many deeper-layer neurons place their boundaries near the zero level set, not only the output neuron, which the authors read as hierarchical signal fitting (§3.1, p. 6).
- **Positional encoding.** A width-10 depth-5 ReLU MLP fit to an image with periodic position encoding shows per-neuron boundaries repeating periodically, increasing the number of regions and weight sharing; this is the authors' candidate explanation for PE's faster convergence (Fig. 4, §3.2, pp. 5–6). With PE approximated by 5 linear pieces per period, performance matched continuous sine/cosine (§3.2, p. 6).
- **Decision-boundary samples.** Points sampled on the boundary between Tabby and Egyptian Cat (tinyImageNet) look like ambiguous blends of the anchor images; some MNIST 2-vs-3 boundary samples clearly look like digit 3, which the authors call a possibly sub-optimal boundary (§3.1, Fig. 6, p. 6; §D.3, Fig. 19, p. 18). Boundary samples are linear combinations of the three anchors, with weights set by the boundary geometry (Fig. 9, p. 8).
- **Data augmentation.** Averaged over 90 tinyImageNet test samples, DA raises partition density at test points for VGG11 (a distinct ARV/NR difference) but much less for VGG16; both DA models reach ≈51% tinyImageNet-200 accuracy (§4.2 and Fig. 8, pp. 7–8).
- **Training dynamics.** For a depth-5 MLP on MNIST, ARV is bimodal for small widths and unimodal with low-volume tails for width ≥64; the lower tails appear only for training samples and are read as possible memorization (§D.1, Fig. 14, p. 15). For an 8-layer CNN on CIFAR-10, ARV near training samples decreases monotonically with width in most cases (§D.1, p. 13).
- **Slice stability.** Across 20 random orientations of the 2-D slice around 5 tinyImageNet samples, the between-orientation ARV standard deviation (5.5993e-08 … 1.2935e-07) is an order of magnitude below the in-domain region-volume standard deviation (7.3955e-07 … 3.0617e-06), suggesting one 2-D slice may suffice for characterizing a network (§D.2, p. 14).

## Limitations stated by the authors

The method computes partitions on 2-D domains, chosen for tractability and visualization, not on the full input space (§2.2, p. 3). Region finding is single-threaded and can be a bottleneck for networks with more than one layer; distributing regions across threads is proposed but not implemented (§B, p. 12). Double precision is required, creating memory bottlenecks for large convolutional layers, mitigated by sparse storage (§B, p. 12). Max-pool layers are replaced by average pooling and strided convolutions for the VGG experiments "for simplicity" (§B, p. 12). Complexities are stated in an average-case, empirical sense: the $O(n^2/5000)$ wall-time bound holds "within the range of $n$ in question" (§C, p. 13).

## Critical assessment

The core contribution — an exact, layer-by-layer partition and boundary over a 2-D slice, with back-projection — is clean and genuinely different from the sampling and LP-based approximations it replaces. The claim of "first provably exact" is scoped to 2-D domains and, as the authors note, competes with exponential-complexity LP methods and one-dimensional sampling methods (§2.2, p. 4); that scope should be read as the headline.

The empirical sections are demonstration-heavy rather than controlled. The architecture comparison (Tab. 1) is informative but small — one 2-D slice per dataset, no variance over slices or seeds reported there. The eccentricity column is the weakest point: the extracted table prints values 102e7, 36e7, 17e7, 14e7 under a header "Ecc.", yet ARE is defined as a max/min pairwise-distance ratio, which is ≥1 and cannot be ~10⁹ (§4.1, p. 7). Either the printed exponents are garbled or the column is something other than eccentricity; the numbers as extracted should not be relied on. Relatedly, the SDF thresholds disagree between text and caption: §3.1 says the MetFaces image is thresholded at .001 and .05 (p. 5) while Fig. 5's caption says 0.01 and 0.5 (p. 5) — a 10× discrepancy the paper does not resolve.

The data-augmentation experiment is suggestive but thin: 90 samples, one architecture family, and the ARV/NR difference largely disappears for the larger VGG16, which the authors explain by parameter count (§4.2, p. 8) but do not test. The "possible memorization" reading of low-volume training tails (§D.1, p. 15) is an interpretation, not a measured link to generalization; the authors themselves call it "possible". The PE conclusion likewise treats correlation (more regions, more weight sharing) as a possible cause of faster convergence (§3.2, p. 5), without an ablation isolating it. Code and a Colab demo are released (§E, p. 20; §1, p. 2), which makes reproduction of the library straightforward, though the GPU/memory setup (8× RTX 8000) is heavy.

The paper's real payoff is the tool: exact boundaries, provably-on-boundary samples, and per-region affine maps invite re-examination of robustness, active learning and initialization — directions the conclusion names but does not pursue (§5, p. 8).

## Connections within the collection

SplineCam is the computational continuation of the CPWL/max-affine-spline view of deep networks developed by [@balestrieroSplineTheoryDeep2018]; Lemma 1 and the uniqueness of $q^\omega$ are taken from that line of work (§2.1, p. 3). It cites [@NIPS2014_fa6f2a46] and [@raghuExpressivePowerDeep2017] for the use of partition density as an expressivity proxy and [@haninComplexityLinearRegions2019] for the link between partition density and function complexity (§1, p. 2, §4, p. 6), and it invokes [@phuongFunctionalVsParametric2019] as the example of parameter symmetries that motivate geometric (rather than weight-based) interpretation (§1, p. 2). Its "are all linear regions created equal?" framing is the question posed by Gamba et al. (outside the collection).

It is a natural counterpart to the region-counting literature in the collection: [@goujonNumberRegionsPiecewise2023], [@serraBoundingCountingLinear2018] and [@pascanuNumberResponseRegions2013] bound or count regions analytically, where SplineCam enumerates them exactly on a slice; [@masdenAlgorithmicDeterminationCombinatorial2022] and [@vincentReachablePolyhedralMarching2021] also compute exact combinatorial structure / reachable polytopes and are the closest algorithmic relatives, possibly related in that they could be cross-validated against SplineCam. Its decision-boundary view connects to the tropical-geometry account of boundaries in [@alfarraDecisionBoundariesNeural2023] and [@zhangTropicalGeometryDeep], and to the polyhedral-complex extraction of [@berzinsPolyhedralComplexExtraction2023]. The observations about neurons that stay inactive on the domain (§3.2, p. 6) echo the activation-pattern themes of [@haninDeepReLUNetworks2019], and the space-folding / symmetry papers [@lewandowskiSpaceFoldsReLU2024; @grigsbyHiddenSymmetriesReLU] address the parameter-vs-function problem SplineCam sidesteps by working in input space.
