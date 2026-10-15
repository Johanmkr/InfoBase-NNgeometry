---
citekey: sattelbergLocallyLinearAttributes2023
title: "Locally linear attributes of ReLU neural networks"
authors: [Ben Sattelberg, Renzo Cavalieri, Michael Kirby, Chris Peterson, Ross Beveridge]
year: 2023
venue: Frontiers in Artificial Intelligence 6
doi: "10.3389/frai.2023.1255192"
arxiv: "2012.01940"
url: null
kind: method
source_type: pdf
source_sha256: 7f22572dd1d9902af294d390629e2c37a8d4c842e85b96ef0a87fb7e1f8e820c
pages: 17
ingested: 2026-10-15
tags: [relu-networks, linear-regions, polyhedral-decomposition, model-similarity, network-compression]
concepts_introduced:
  - dual space of linear regions
  - local linear map clustering
  - affine map between network linear regions
  - polytope evolution during training
concepts_used:
  - linear regions of ReLU networks
  - piecewise linear representation
  - activation pattern
  - Jacobian of a neural network
  - convex polytope
  - k-means clustering
  - tropical rational function
builds_on: [raghuExpressivePowerDeep2017, NIPS2014_fa6f2a46, pascanuNumberResponseRegions2013, haninComplexityLinearRegions2019, haninDeepReLUNetworks2019, liuReLUNeuralNetworks2023, zhangTropicalGeometryDeep]
datasets: [MNIST, XOR, circle vs. annulus]
---

# Locally linear attributes of ReLU neural networks

**Sattelberg, Cavalieri, Kirby, Peterson, Beveridge (2023)**, *Frontiers in Artificial Intelligence*. [doi:10.3389/frai.2023.1255192](https://doi.org/10.3389/frai.2023.1255192)

!!! abstract "TL;DR"
    A ReLU network is a continuous piecewise-linear map whose linear pieces sit on convex polytopes; the authors treat the affine map of each polytope as a point in a "dual" space and study that point cloud instead of the polytope geometry. Two experiments follow: k-means clustering the sampled affine maps of MNIST networks recovers near-original accuracy from as few as 10 cluster centers for every network except ResNet, and a least-squares affine map between the affine maps of two different networks preserves much of their accuracy. The conclusion is that trained networks of quite different architectures converge to similar, and substantially redundant, piecewise-linear solutions.

## Problem and motivation

ReLU networks are continuous piecewise-linear maps $f:\R^d \to \R^o$, and the weights partition the input space into convex polytopes, each carrying one affine map (§2.1, p. 2). The authors argue that most work on these linear regions has gone into expressivity and counting bounds — how many regions a network *can* have as a function of width and depth [@raghuExpressivePowerDeep2017; @NIPS2014_fa6f2a46; @pascanuNumberResponseRegions2013] — which "do not say anything about the expressivity of a *given* neural network" (§1, p. 1). Empirically, trained networks turn out to have far fewer linear regions than the worst case: on any one-dimensional subspace the region count grows only linearly [@haninComplexityLinearRegions2019], and after training it grows polynomially in the number of ReLU nodes with degree equal to the input dimension [@haninDeepReLUNetworks2019].

The paper's target is different: to build intuition on how the regions behave, how their number might be reduced, and how similar the region structures of two independently trained networks are. Because enumerating all $m$ regions (§2.1, Eq. 1) is infeasible for realistic nets, the authors work with *sampled* affine maps and with the rows of the region's weight matrix rather than the polytopes themselves (§2.4, p. 8).

## Method

**Iterated-perceptron picture (§2.3, pp. 5–6).** Each ReLU node adds a non-linearity along a line in the *output* space of the previous layer; mapped back to the original input space that line becomes a trajectory that changes direction only where it crosses a boundary from an earlier layer. Figure 5 renders this for a circle-versus-annulus classifier layer by layer: first-layer boundaries are straight lines in the input plane, later ones are piecewise-linear and can only "break" at previous boundaries. The paper supplies animations of how these structures evolve during training (§3.1, p. 10, with video links).

**Dual space (§2.1, p. 4).** A row of the region's weight matrix $\mathbf{W}_i$ concatenated with the matching bias entry is a point in $\R^{d+1}$, viewed as a "dual" to the network's output along which the linear-function behaviour can be analysed.

**Clustering local linear functions (§2.4.1, p. 8).** Given an output $j$, stack the sampled affine maps into a matrix $\mathbf{C}_j$ (Eq. 7), plus $\tilde{\mathbf{C}}_j$ from a held-out set. Steps:

1. Compute $\mathbf{C}_j$ (train) and $\tilde{\mathbf{C}}_j$ (test) rows from the Jacobian at each sampled input.
2. Fit k-means with $k$ centers on the rows of $\mathbf{C}_j$.
3. Assign every row of $\tilde{\mathbf{C}}_j$ to its nearest center.
4. Use that center as the linear map for that output.
5. Classify by the largest resulting output; compare with original accuracy.

If $k=1$ keeps near-original accuracy the network behaves holistically as a linear map; if it is near random, the transformation is genuinely non-linear. Note the reported cluster counts are 10× smaller than the true number, because each of the ten digit outputs is clustered separately (Tab. 1, p. 15).

**Affine maps between networks (§2.4.2, p. 8).** For two networks, per output $j$, fit $\mathbf{M}_j \in \R^{(d+1)\times(d+1)}$ minimising Eq. 8 — a least-squares map from one network's linear-region weights to the other's. The result is no longer a continuous function, because the bias is part of the mapped vector, so mapped regions need not join at their boundaries (§2.4.2, p. 9).

## Key equations and definitions

Piecewise-linear representation of the network (§2.1, Eq. 1, p. 2), with $f:\R^d\to\R^o$ (Eq. 1):

$$
f(x) =
\begin{cases}
\mathbf{W}_1 x + b_1, & \text{if } \mathbf{A}_1 x \le c_1 \\
\mathbf{W}_2 x + b_2, & \text{if } \mathbf{A}_2 x \le c_2 \\
\quad\vdots \\
\mathbf{W}_m x + b_m, & \text{if } \mathbf{A}_m x \le c_m
\end{cases}
$$

The affine map $(\mathbf{W}_i,b_i)$ is valid exactly on the polytope $\mathbf{A}_i x \le c_i$; the number of regions $m$ grows polynomially in width and exponentially in depth.

XOR example (§2.2, Eq. 2, p. 4): the network in Figure 2B (Eq. 2),

$$
f(x,y) = \begin{bmatrix} 2 & -4 \end{bmatrix}
\max\left\{
\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix},
\ \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\right\} - 1
$$

Its induced regions (§2.2, Eq. 3, p. 4) (Eq. 3):

$$
f(x) =
\begin{cases}
-2x - 2y + 1, & 0 \le x + y \quad \text{Both ReLUs activated} \\
2x + 2y + 1, & -1 \le x + y \le 0 \quad \text{Top activated; bottom not} \\
-1, & x + y \le -1 \quad \text{Neither ReLU activated}
\end{cases}
$$

The "fourth" case (bottom activated, top not) would give $-4x-4y-1$ but lives on the empty polytope $0 \le x+y \le -1$, so it is unreachable — an example of why the region count is below $2^{\#\text{ReLUs}}$ (§2.2, p. 4).

Dual point (§2.1, Eq. 4, p. 4): the row of the region's weight matrix and its bias form

$$
\begin{bmatrix} -1 & -1 & 0 \end{bmatrix}\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = -x - y
$$

so the region $-x-y$ induces the dual point $(-1,-1,0)$.

Per-output affine maps (§2.4, Eqs. 5–7, p. 8): with rows $w_{i,j}^T$ and biases $b_{i,j}$ (Eqs. 5–6),

$$
\mathbf{W}_i =
\begin{bmatrix} w_{i,1}^T \\ w_{i,2}^T \\ \vdots \\ w_{i,o}^T \end{bmatrix},
\quad
\mathbf{C}_j =
\begin{bmatrix} w_{1,j}^T & b_{1,j} \\ w_{2,j}^T & b_{2,j} \\ \vdots \\ w_{m,j}^T & b_{m,j} \end{bmatrix} \in \R^{m\times(d+1)}
$$

and Eq. 7 is the same with $p$ sampled rows instead of all $m$. The regression objective (§2.4.2, Eq. 8, p. 8) (Eq. 8):

$$
\norm{\mathbf{C}_{j,\text{network1}}\mathbf{M}_j - \mathbf{C}_{j,\text{network2}}}_2
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $f:\R^d\to\R^o$ | ReLU network, $d$ inputs, $o$ outputs |
| $m$ | number of linear regions |
| $\mathbf{W}_i, b_i$ | affine map on region $i$; $\mathbf{W}_i$ is the network's Jacobian there |
| $\mathbf{A}_i, c_i$ | inequalities cutting out region $i$ |
| $w_{i,j}, b_{i,j}$ | $j$-th output's weight row and bias on region $i$ |
| $\mathbf{C}_j$ | matrix of sampled affine maps for output $j$ |
| $\tilde{\mathbf{C}}_j$ | same, built from held-out inputs |
| $\mathbf{M}_j$ | least-squares map between two networks' $\mathbf{C}_j$ |
| $k$ | number of k-means cluster centers |

## Main results

**Visualizations (§2.3, §3.1).** For two-input problems the whole region structure is viewable. Layers beyond the first only refine boundaries laid down earlier; in the circle-versus-annulus case the final decision boundary closely follows a second-hidden-layer boundary, suggesting later layers are unnecessary (§2.3, p. 6). A three-node single-hidden-layer network often fails to find a closed region and reaches only 50% accuracy, while a three-hidden-layer, eight-node-per-layer network creates the closed region in fewer epochs because the simple network is a subnetwork of it (§3.1, p. 11) — visual support for the lottery-ticket hypothesis and for the greater importance of early layers [@raghuExpressivePowerDeep2017].

**Clustering (Tab. 1, p. 15; §3.2, p. 14).** Correct test labels out of 10,000 (original accuracies in the first row):

| # Clusters | Dense | Conv1 | Conv2 | Inception | ResNet |
|---|---|---|---|---|---|
| Original | 9603 | 9807 | 9804 | 9908 | 9892 |
| 1 | 8679 | 6766 | 6366 | 974 | 1432 |
| 10 | 9231 | 8639 | 8672 | 9660 | 8166 |
| 100 | 9434 | 9382 | 9421 | 9689 | 8458 |
| 1000 | 9508 | 9586 | 9603 | 9695 | 8982 |
| 10000 | 9554 | 9696 | 9673 | 9752 | 9381 |

The dense single-hidden-layer network (96.03%) keeps 8679/10000 with a single cluster, i.e. it is close to a linear classifier. Inception (99.08%) collapses to 974/10000 with $k=1$ but recovers to 9660 with $k=10$ — better than the original dense network — so its local piecewise-linear map compresses to roughly ten regions. The two convolutional networks (98.07%, 98.04%) and ResNet (98.92%) stay poor even at $k=10{,}000$ (Conv1 9696, Conv2 9673, ResNet 9381), i.e. ResNet cannot be simplified by this clustering.

**Affine maps (Tab. 2, p. 16; §3.3).** Correct labels out of 10,000 when mapping the linear regions of one network onto another (rows "from", columns "to"; diagonal = original accuracy):

| From \ To | Dense | Conv1 | Conv2 | Inception | ResNet |
|---|---|---|---|---|---|
| Dense | 9603 | 9536 | 9519 | 9290 | 9068 |
| Conv1 | 9567 | 9807 | 9776 | 9662 | 9588 |
| Conv2 | 9562 | 9786 | 9804 | 9644 | 9579 |
| Inception | 8868 | 9488 | 9511 | 9908 | 9536 |
| ResNet | 9320 | 9738 | 9739 | 9838 | 9892 |

No mapped network beats its original accuracy, but many stay high. Dense and the two convolutional networks map to each other near their originals (e.g. Conv1→Dense 9567). Inception replicates the others poorly (Inception→Dense 8868, below Dense's own 9603), consistent with its lack of a good single-region representation. ResNet maps into everything well except Dense (ResNet→Inception 9838, ResNet→Conv1 9738), suggesting its linear regions contain the information the others use. The asymmetry is the interesting part: ResNet approximates Inception, but Inception cannot approximate ResNet even though their original accuracies are nearly equal (§3.3, p. 15).

## Limitations stated by the authors

MNIST is high-dimensional but a simple dataset, so the conclusions may not extend to harder data; the authors call for extensions to more complex datasets (§4, p. 16). All five networks were trained with similar methodologies, so how different training schemes or regularisation affect region similarity is untested; they point to the Zhang and Wu study as a template (§4, p. 16). The methods require running inputs through the network and computing Jacobians, and the produced functions are non-continuous when biases are mapped (§2.4.2, p. 8). Visualizations are only practical for two-dimensional inputs, and even simple MNIST networks put almost every image on a unique polytope (<1% overlap), so all experiments use sampled points rather than the full region set (§2.4.1, p. 8).

## Critical assessment

My reading. The paper is honest that it is exploratory: the two headline claims (regions are highly redundant; different networks are affinely related at the region level) are supported by one dataset and five networks with similar training recipes, which the authors concede. The evidence is stronger for the clustering result than for the affine-mapping result: Table 1 is a clean ablation over $k$, whereas Table 2 has no baseline for "how much accuracy survives a *random* affine map", so it is unclear how much of the preserved accuracy reflects genuine geometric agreement versus the fact that all networks solve the same task ($\S3.3$ even notes accuracy alone may hide qualitative differences). The claimed asymmetry between Inception and ResNet (ResNet→Inception nearly recovers accuracy, Inception→ResNet does not) rests on single entries of Table 2 (9838 vs 9536) without a significance or seed study, and the interpretation that Inception "identifies information that may not generalize" is speculative. The dual-space framing is intuitive rather than formal — no theorem links dual-point structure to network behaviour — and the paper does not engage the tropical-geometry view of the same objects beyond citing it [@zhangTropicalGeometryDeep]. Code is released (Data availability statement, p. 16), and the training details are specific enough to reproduce, though the key hyperparameters are only listed in prose. On the credit side, the sampling-based formulation is genuinely practical: it treats the network interior as a black box and so applies to architectures (convolutional, Inception, ResNet) that region-enumeration methods cannot handle.

## Connections within the collection

The paper positions itself as an extension of the visualization work of Raghu et al. [@raghuExpressivePowerDeep2017], which is also its main source for the "earlier layers matter more" finding reproduced in Figure 5 and §3.1. Its sampling of local linear maps as bit-string-like activation-pattern objects is close in spirit to the activation-pattern analysis of [@liuReLUNeuralNetworks2023] — which this paper explicitly contrasts with, since that method needs fully connected networks and does not extend to convolution or pooling (§2.4, p. 7). The counting results it builds on are those of [@NIPS2014_fa6f2a46] and [@pascanuNumberResponseRegions2013], and the empirical "few regions" counterparts are [@haninComplexityLinearRegions2019] and [@haninDeepReLUNetworks2019]. The equivalence it invokes between ReLU networks, piecewise-linear functions and tropical rational functions is that of [@zhangTropicalGeometryDeep] (§2.1, p. 2; §2.4, p. 7). The idea of comparing networks by their local geometry — as opposed to their features — is possibly related to the symmetry and functional-equivalence questions of [@grigsbyHiddenSymmetriesReLU] and [@phuongFunctionalVsParametric2019], though neither is cited by this paper.
