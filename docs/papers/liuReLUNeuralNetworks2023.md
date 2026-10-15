---
citekey: liuReLUNeuralNetworks2023
title: "ReLU Neural Networks, Polyhedral Decompositions, and Persistent Homology"
authors: [Yajing Liu, Christina M. Cole, Chris Peterson, Michael Kirby]
year: 2023
venue: "Proceedings of the 2nd Annual Workshop on Topology, Algebra, and Geometry in Machine Learning (TAG-ML) at the 40th International Conference on Machine Learning"
doi: null
arxiv: null
url: null
kind: method
source_type: pdf
source_sha256: 8352148211d0b9cae1103d2b81d9cb36bd9b9b79b9fbda95d69446258288fef6
pages: 13
ingested: 2026-10-15
tags: [relu-networks, polyhedral-decomposition, persistent-homology, topological-data-analysis, activation-patterns]
concepts_introduced:
  - binary vector of a ReLU network
  - polyhedral decomposition of the input space
  - dual graph of the polyhedral decomposition
  - Hamming distance between binary vectors
  - active bit
concepts_used:
  - ReLU feedforward neural network
  - persistent homology
  - Vietoris–Rips filtration
  - barcode
  - linear region
  - linear programming
builds_on: [vincentReachablePolyhedralMarching2021, balestrieroSplineTheoryDeep2018, sattelbergLocallyLinearAttributes2023, masdenAlgorithmicDeterminationCombinatorial2022, haninDeepReLUNetworks2019]
datasets: [ImageNet validation set]
---

# ReLU Neural Networks, Polyhedral Decompositions, and Persistent Homology

**Liu, Cole, Peterson, Kirby (2023)**, *TAG-ML Workshop at ICML 2023*.

!!! abstract "TL;DR"
    A ReLU feedforward network partitions its input space into convex polyhedra, each labelled by the network's ReLU activation pattern as a binary vector. The paper argues that the Hamming distance between these binary vectors is a coarse but robust proxy for the dual-graph distance between polyhedra, and shows empirically that feeding that distance into Vietoris–Rips persistent homology still recovers the homology of manifolds sampled in input space — including for ImageNet images pushed through ResNet-50, a network trained for a purpose unrelated to topology.

## Problem and motivation

A ReLU feedforward neural network (FFNN) $F : \R^m \to \R^n$ is piecewise linear, and the ReLU activation pattern of every hidden node assigns each input point a binary vector $s(x)$; points sharing that vector form a convex polyhedron, so the network induces a finite polyhedral decomposition of input space together with a dual graph whose vertices are the polyhedra (§1, p. 1). The granularity of this decomposition raises a concrete question: is the network's own combinatorial geometry fine enough to destroy the geometric structure of data, or coarse enough to be a usable summary? The paper's stated answer is that the induced labelling is "a coarse quantization of input space", yet "sufficiently robust" that persistent homology built on it detects homological signals of manifolds from samples (§Abstract, p. 1). The motivation is that the distance measure is handed to the analyst for free by any trained network, regardless of what the network was trained for, and that its coarseness may suppress noise that distorts the Euclidean metric (§5, p. 8). The setting is networks with ReLU activations on hidden layers and an affine output layer; no assumption is made about training objective, depth or architecture family beyond that.

## Method

The construction has three stages.

**Activation-pattern labelling.** For each hidden layer $i$ the network attaches a bit $s_{i,j}(x)$ to node $j$; stacking the per-layer vectors gives the binary vector $s(x)$ supported on the $h=\sum_i h_i$ hidden nodes (Eq. 4–5). The paper records the convention that a bit is $1$ if ReLU "was not applied at the node and 0 if ReLU was applied at the node" (§2.3, bullet list, p. 4) — equivalently, $s_{i,j}(x)=1$ exactly when the pre-activation $w_{i,j}F_{i-1}(x)+b_{i,j}$ is positive (Eq. 4), the usual convention that the node is active. The set of points with a fixed bit vector is the convex polyhedron $P=\{x : Ax \le c\}$, and the network restricted to $P$ is a single affine map $G(x)$ (§2.3, p. 3).

**Enumerating polyhedra.** Because a random bit vector may or may not correspond to a non-empty polyhedron (and may not be full-dimensional), Algorithm 1 (Appendix A, p. 11) tests each of the $2^h$ candidate bit vectors by solving a feasibility LP $\min 0$ s.t. $Ax \le c$ (Eq. 15) and keeps the full-dimensional feasible ones. Algorithm 2 (Appendix A, p. 11) replaces this brute force with a traversal: find the *active bits* of a polyhedron's bit vector (those whose inequality is not redundant, determined by the redundancy LP of Eq. 10), flip one active bit at a time to obtain all neighbours sharing a facet, and iterate until the closure of the reached set is complete. This is a bounded-region modification of the reachable polyhedral marching algorithm [@vincentReachablePolyhedralMarching2021]; enumerating a box adds the box constraints to $A$ and appends boundary-hitting flags as extra bits (§3.1, p. 4).

**Topological pipeline.** Each sampled data point inherits the bit vector of the polyhedron it lies in; a pairwise Hamming-distance matrix $D$ is formed and fed to Vietoris–Rips persistent homology (Eq. 13–14), with barcodes computed by Ripser (§4.2, p. 6). Examples 4.4–4.5 additionally combine two distance matrices (from a slight rotation of the same sample set) by entrywise max and min, i.e. a monotone Boolean **and** / **or** combination (§4.2, p. 7).

## Key equations and definitions

$$
ReLU(a) = \begin{cases} a & \text{if } a > 0 \\ 0 & \text{if } a \le 0 \end{cases} \qquad \text{(Eq. 2, p. 2)}
$$

$$
s_{i,j}(x) = \begin{cases} 1 & \text{if } w_{i,j}F_{i-1}(x) + b_{i,j} > 0 \\ 0 & \text{if } w_{i,j}F_{i-1}(x) + b_{i,j} \le 0 \end{cases} \qquad \text{(Eq. 4, p. 2)}
$$

$$
s(x) = [s_1^\top(x) \dots s_L^\top(x)]^\top \in \R^h, \qquad h = \sum_{i=1}^{L} h_i \qquad \text{(Eq. 5, p. 2)}
$$

$$
s'_{i,j} = \begin{cases} 1 & \text{if } s_{i,j} = 0 \\ -1 & \text{if } s_{i,j} = 1 \end{cases} \qquad \text{(Eq. 6, p. 2)}
$$

$$
diag(s'_1)(W_1 x + b_1) \le 0 \qquad \text{(Eq. 7, p. 2)}
$$

$$
Ax \le c, \qquad A = [A_1^\top \dots A_L^\top]^\top,\ c = [c_1^\top \dots c_L^\top]^\top \qquad \text{(Eq. 9, pp. 2–3)}
$$

$$
A_i = diag(s'^{i})\hat W_i, \qquad c_i = diag(s'^{i})(-\hat b_i) \qquad \text{(§2.3, p. 3)}
$$

$$
A(t)_{i,j} = \begin{cases} 0 & \text{if } D^{(i,j)} > t \text{ or } i = j \\ 1 & \text{else} \end{cases} \qquad \text{(Eq. 13, p. 6)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $F_i(x)$ | output of layer $i$ on input $x$; $F_0(x)=x$ |
| $h_i$ | number of nodes in hidden layer $i$; $h$ = total hidden nodes |
| $h_0 = m$, $h_{L+1} = n$ | input and output dimensions |
| $W_i, b_i$ | weight matrix and bias vector of layer $i$ |
| $w_{i,j}, b_{i,j}$ | $j$-th row / entry of $W_i, b_i$ |
| $s_i(x)$, $s(x)$ | per-layer and stacked binary (activation) vectors |
| $s'^{i}$ | sign vector: $+1$ where $s_i$ is $0$, $-1$ where $s_i$ is $1$ |
| $\hat W_j, \hat b_j$ | recursively folded weight/bias for layer $j$ (§2.3, p. 2) |
| $A, c$ | stacked half-space data of the polyhedron $\{x : Ax \le c\}$ |
| $A', c'$ | minimal (non-redundant) subset of rows defining the same polyhedron |
| $G(x)$ | the single affine map equal to $F$ on the polyhedron |
| $D$ | distance matrix (Euclidean or Hamming) between sampled points |
| $A(t)$ | threshold adjacency matrix at scale $t$ |

The central structural facts are stated without full proof in the main text: the dual graph is bipartite (proof/argument sketched in Appendix B, pp. 11–12), the affine maps of two polyhedra sharing an $(m-1)$-face agree on that face, and "two convex polyhedra $P_1, P_2$ share an $(m-1)$-dimensional face, [iff] the binary vectors associated to each of the polyhedra differ in one bit" (§2.3, p. 4, bullet list). The observation that the Hamming distance approximates the shortest path length in the dual graph is stated as an approximation, not an equality (§1, p. 1; §2.3, p. 4).

## Main results

**Polyhedron counts on two small models.** Model (11), $2\text{-}3\text{-}3\text{-}3$, fit to $f_1(x_1,x_2)=x_1^2+x_2^2-2/3$ on $10{,}000$ uniform points in $[-1,1]^2$, has $25$ polyhedra in the box $[-1,1]^2$ and $27$ in $\R^2$, of which $21$ are bounded and $6$ unbounded (Fig. 1a; §3.2, p. 5). Model (12), $3\text{-}10\text{-}10\text{-}10\text{-}10$, fit to $f_2(x_1,x_2,x_3)=(x_1-1)^2+2x_2^2+x_3^2+1/3$ on $125{,}000$ uniform points in $[-1,1]^3$ (batch size 50, early stopping), has $1858$ polyhedra in $[-1,1]^3$ and $3331$ in $\R^3$ (Fig. 1b; §3.2, p. 5). Both are far below the $2^h$ a priori bound, consistent with the $\sim h^m$ activation-pattern count they attribute to [@haninDeepReLUNetworks2019] (§3.1, p. 4).

**Recovering a circle in $\R^3$.** Sampling the circle $(0,\cos t,\sin t)$ at 20 evenly spaced points through model (12) yields 20 bit vectors, 19 distinct; the $H_0$ barcode of the $19\times19$ Hamming matrix shows connectivity at Hamming distance 4, the $H_1$ barcode shows a short-lived spurious loop at distance 3 and the genuine circle signal at distance 4 (Example 4.1; Fig. 3a, p. 7). At 500 samples only 41 of the 500 bit vectors are distinct; $H_0$ connects at Hamming distance 2 and $H_1$ shows a long-lived loop beginning at Hamming distance 2 (Example 4.2; Fig. 3b, p. 7).

**ResNet-50 on ImageNet images.** Using the pre-trained ResNet-50 (input dimension $150528$, more than $6{,}000{,}000$ nodes) on three ImageNet validation images $A_1,A_2,A_3$ (miniature poodle, Persian cat, Saluki), the 50 points $\sin\theta A_1 + \cos\theta A_2$ produced a strong circle signal in the $50\times50$ Hamming matrix but also "homological noise (or homological dust)" in $H_0$, while the 50 points $A_3 + \sin\theta A_1 + \cos\theta A_2$ produced a strong circle signal and no noise (Example 4.3; Fig. 5, p. 8). Combining two rotated copies of the sample set by max and by min (Examples 4.4–4.5, pp. 7–8) gave, in the first case, an overall strengthening of the signal length for the max (**and**) combination and an earlier start for the min (**or**) combination, with the authors declining to call either better; in the second case no homological noise appeared (§4.2, pp. 7–8). An appendix example with five random orthogonal $224\times224\times3$ tensors and 100 points on a 2-torus-like family produces $H_1$ and $H_2$ barcodes (Appendix C, Ex. C.1, p. 12; Fig. 9, p. 13).

## Limitations stated by the authors

Themselves say all exact decomposition methods are "woefully inadequate for the large and deep networks that are commonplace today" — the algorithms are demonstrated only on small networks, and the ResNet-50 experiments deliberately *avoid* enumerating its polyhedra, using only the bit vectors of the sampled points (§3, p. 4; §4.2, p. 7). They note that the Hamming distance is only "an approximation for the smallest number of polyhedral steps" (§1, p. 1). In Example 4.3 they state they "were unsure why (or if)" the presence of negative entries in the sample points relates to the homological noise (§4.2, p. 7). The future-work section conjectures (not proves) that the decomposition's coarseness helps with noise, and defers manifold examples ($SO(3)$, higher-dimensional tori, fibre bundles) and a quantitative "polytopes landscape" to later work (§5, p. 8).

## Critical assessment

This is a short workshop paper whose contribution is an observation plus a demonstration rather than a theorem: the two structural facts it leans on most (the one-bit/facet correspondence and the dual graph being bipartite) are stated as easy verifications in a bullet list and Appendix B, with the bipartite argument specialised to the 2-D example. The empirical evidence is thin — 20, 50, 100 sample points, one circle and one torus-like family, single runs with no error bars, no comparison against Euclidean-distance barcodes on the same data, and barcodes read visually rather than scored. The claim that the Hamming metric is "robust" is illustrated, not quantified: there is no bound relating Hamming distance to dual-graph distance, only the qualitative approximation remark. The most striking part, that a 6-million-node ImageNet classifier trained for classification still exposes the topology of a synthetic image circle, is genuinely surprising but is shown once per example; whether it survives other images, other architectures, or random networks is untested here. There is also a small wording inconsistency: the bullet list on p. 4 glosses bit $1$ as "ReLU was not applied", while Eq. 4 defines bit $1$ by positivity of the pre-activation — the two agree for the standard definition of an active node, but the gloss invites confusion. Sub-questions the paper leaves open include whether the bounded-region enumeration stays tractable at all beyond toy inputs, and how to choose the monotone Boolean combination (max/min) of several distance matrices in practice rather than by trial. No code or data release is mentioned.

## Connections within the collection

The enumeration machinery is a bounded-region variant of reachable polyhedral marching [@vincentReachablePolyhedralMarching2021], and the paper credits the Hamming-metric idea and the network-imposed input metrics to [@balestrieroSplineTheoryDeep2018], whose max-affine spline view of deep networks is the same piecewise-linear picture in different language. Its study of behaviour from the decomposition plus the per-polyhedron affine map follows [@sattelbergLocallyLinearAttributes2023]. For enumerating the combinatorial structure it points to [@masdenAlgorithmicDeterminationCombinatorial2022], and for the fact that the number of realised activation patterns is far below $2^h$ it relies on [@haninDeepReLUNetworks2019]. Because its central object is the polyhedral decomposition and its dual graph, it sits directly alongside the other decomposition papers of the collection — the polyhedral-complex extraction of [@berzinsPolyhedralComplexExtraction2023], the region counting of [@goujonNumberRegionsPiecewise2023] and [@serraBoundingCountingLinear2018], and the symmetry-based analysis of region combinatorics in [@grigsbyHiddenSymmetriesReLU] — and it addresses the same observation space that [@zhangTropicalGeometryDeep] and [@alfarraDecisionBoundariesNeural2023] treat tropically. The persistent-homology side connects it to the topological-data-analysis papers of the collection, [@carlssonPREIMAGERECTIFIERNETWORK2017] (homology of decision boundaries), [@chazalIntroductionTopologicalData2021] and [@fuLatentTopologyInduction2022].
