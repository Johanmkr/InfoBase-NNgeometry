---
citekey: carlssonPREIMAGERECTIFIERNETWORK2017
title: "The Preimage of Rectifier Network Activities"
authors: [Stefan Carlsson, Hossein Azizpour, Ali Razavian, Josephine Sullivan, Kevin Smith]
year: 2017
venue: ICLR 2017 Workshop track
doi: null
arxiv: null
url: "https://openreview.net/forum?id=HJcLcw9xg"
kind: method
source_type: pdf
source_sha256: 9e0a8c9081e3cc4246c5347df59abe187c8e4ee240779386ff621ad0431a279a
pages: 3
ingested: 2026-10-15
tags: [relu-networks, preimages, piecewise-linear-manifolds, activation-patterns, network-geometry]
concepts_introduced:
  - preimage of layer activities
  - complete preimage computation
  - reciprocal nullspace basis of the weight matrix
  - piecewise linear preimage manifold
  - subspace coordinates
concepts_used:
  - rectified linear unit
  - fully connected layer
  - nullspace hyperplane
  - full rank weight matrix
  - activation pattern
  - piecewise linear function
builds_on: []
datasets: []
---

# The Preimage of Rectifier Network Activities

**Carlsson, Azizpour, Razavian, Sullivan, Smith (2017)**, *ICLR 2017 Workshop track*. [OpenReview](https://openreview.net/forum?id=HJcLcw9xg)

!!! abstract "TL;DR"
    For a fully connected rectifier network the paper gives a closed-form, weight-only procedure that returns the *complete* preimage of a target activity vector in a layer, where earlier work recovered single inputs by heuristic optimisation and regularisation. The preimage of a layer is described as a piecewise linear manifold in the input space, built from the reciprocal (dual) basis of the layer's weight matrix: each unit with zero output contributes one free nonnegative direction, and the active units pin down the remaining coordinates. The contribution is a three-page workshop paper: it states the construction and illustrates it on a 3-unit, 2-dimensional example, and says the empirical evaluation was still in progress (§1; Abstract).

## Problem and motivation

The paper asks a question about the geometry of a trained rectifier network: given a target vector of activities $x^{(l+1)}$ at some layer, which inputs $x$ produce it? The set of all such inputs is the *preimage* $P(x^{(l+1)})$ (§1, p. 1). The authors take the setting of a fully connected multi-layer network in which the map at layer $(l)$ is a matrix $W$ and bias $b$ followed by a ReLU, so between successive layers $x^{(l+1)} = [Wx^{(l)}+b]^+$ (§1, p. 1; unnumbered eq.). The problem is motivated as the inverse of a forward pass, the same question probed by Mahendran & Vedaldi (2015, 2016) — outside this collection — who recovered "preimages" by heuristic optimisation and regularisation. The authors argue that such demonstrations were not guaranteed to be complete, and that a weight-only characterisation is needed. Their stated conclusion is that the most general preimage is "piecewise linear manifolds in the input space with possibly multiple branches", so the manifold, not a single point, is the object to compute (Abstract).

The paper's own framing is deliberately modest: the abstract presents the procedure as complementing the optimisation-based work, and adds that the procedure's ability to extract complete preimages, and the general structure of the preimage manifolds, were being evaluated empirically at the time of writing (Abstract).

## Method

The construction turns a target activity vector into a set of linear equalities and inequalities and then reads the solution set off in a basis adapted to the layer's weight matrix.

1. **Split the output units.** Fix $x^{(l+1)}$ and let $i_1,\dots,i_p$ index the components that are $=0$ and $j_1,\dots,j_q$ those that are $>0$; with $x$ in $n$-dimensional space, $p+q=n$ (§1, p. 1). A zero output means the pre-activation was $\le 0$ (rectified away), an active output means it was $>0$. This is the activation pattern of the target; the preimage is the part of input space consistent with it.
2. **Write the constraint set.** Equality of the active coordinates and the sign constraint on the rectified coordinates give the system in Eq. 1 (§1, p. 1). The detailed form of $P(x^{(l+1)})$ given at the top of §1 is exactly this: for active units $w_i^T x + b_i = x_i^{(l+1)}$, for rectified units $w_i^T x + b_i \le 0$.
3. **Build the nullspace hyperplanes.** For each kernel $w_i$ (row $i$ of $W$), define its nullspace $\Pi_i = \{x : w_i^T x + b_i = 0\}$ — a hyperplane in the space of *activities of layer $(l)$* (§1, p. 1).
4. **Build the reciprocal basis.** Intersect the complementary set of hyperplanes to get the one-dimensional subspace $\pi_i = \Pi_1 \cap \dots \cap \Pi_{i-1} \cap \Pi_{i+1} \cap \dots \cap \Pi_n$, and choose $e_i$ spanning $\pi_i$ with direction and length fixed by $w_i^T e_i = 1$ (§1, pp. 1–2). The dimension count is the argument: each additional hyperplane lowers dimension by one, so intersecting $n-1$ of them leaves a line. The assumed full rank of $W$ makes $\{e_1,\dots,e_n\}$ a basis of the input space, and the authors note that the matrix $(e_1,\dots,e_n)$ is the inverse of $W$ (§1, p. 2). Computation is by finding the nullspace of $W$ with row $i$ deleted.
5. **Read the constraints in this basis.** Because $e_i$ lies in every other kernel's nullspace, $w_j^T x = \alpha_j$, i.e. the subspace coordinates are just the pre-activations, so each of Eq. 1's inequalities is a simple bound on one coordinate (§1, p. 2).
6. **Assemble the preimage.** Assign the active coordinates their target values and let the rectified coordinates range over the allowed half-line; the result is the manifold of §1, p. 2–3.

The worked illustration is a network of three units per layer and 2-dimensional inputs (§1, p. 2; Fig. 1). For the all-positive octant ($w_i^T x > 0$ for all $i$) the map is invertible and the preimage is a single point. For a point with exactly one coordinate rectified, the preimage is a ray along the corresponding $e_i$; when two coordinates are rectified simultaneously, it is a 2-dimensional cone (grey shaded area in Fig. 1). The general statement the authors extract is that points outside the all-positive region lie on a linear submanifold spanned by the $e_i$ of the rectified units (§1, p. 2).

For a multi-level network the same argument is applied layer by layer: a preimage at one level is a piece of a linear manifold at that level, of dimension equal to the number of units with positive output for that element, and mapping back to the input space yields piecewise linear manifolds whose points all map to the same element (§1, p. 3). The authors present these manifolds as "fundamental building blocks for mapping input distributions to node outputs at any level of the network" (§1, p. 3).

## Key equations and definitions

Layer map (§1, p. 1, unnumbered):

$$
x^{(l+1)} = \left[Wx^{(l)} + b\right]^+ \qquad\text{(§1, p. 1, unnumbered)}
$$

Preimage of a target activity, in the compact form and the expanded form given in §1:

$$
P(x^{(l+1)}) = \left\{x : x^{l+1} = [Wx + b]^+\right\} \qquad\text{(§1, p. 1, unnumbered)}
$$

$$
P(x^{(l+1)}) = \left\{x : w_i^T x + b_i = x_i^{l+1}\ \forall x_i^{l+1} > 0,\quad w_i^T x + b_i \le 0\ \forall x_i^{l+1} = 0\right\} \qquad\text{(§1, p. 1, unnumbered)}
$$

The constraint system (Eq. 1, p. 1), written for the rectified indices $i_1,\dots,i_p$ with $p+q=n$:

$$
w_{i_1}^T x^{(l)} + b_{i_1} \le 0,\; w_{i_2}^T x^{(l)} + b_{i_2} \le 0,\; \dots,\; w_{i_p}^T x^{(l)} + b_{i_p} \le 0
$$
$$
w_{j_1}^T x^{(l)} + b_{j_1} > 0,\; w_{j_2}^T x^{(l)} + b_{j_2} > 0,\; \dots,\; w_{j_q}^T x^{(l)} + b_{j_q} > 0 \qquad\text{(Eq. 1, p. 1)}
$$

Nullspace hyperplanes and the reciprocal basis (§1, pp. 1–2, unnumbered, printed as $\Pi_i$ and $\pi_i$):

$$
\Pi_i = \left\{x : w_i^T x + b_i = 0,\ i = 1\dots n\right\}, \qquad
\pi_i = \Pi_1 \cap \Pi_2 \cap \dots \cap \Pi_{i-1} \cap \Pi_{i+1} \cap \dots \cap \Pi_n
$$

Normalisation of the basis and the coordinate identity (§1, p. 2, unnumbered):

$$
w_i^T e_i = 1, \qquad w_j^T e_i = 0 \ (i \ne j), \qquad x = \sum_{i=1}^{n} \alpha_i e_i, \qquad w_j^T x = \sum_{i} \alpha_i w_j^T e_i = \alpha_j
$$

The activity of output unit $j$ (Eq. 2, p. 2):

$$
x_j^{(l+1)} = \left[w_j^T x^{(l)} + b_j\right]^+ \qquad\text{(Eq. 2, p. 2)}
$$

The preimage manifold (§1, p. 2, unnumbered; printed as $x = \alpha_{i_1}e_{i_1} + \alpha_{i_2}e_{i_2} + \dots \alpha_{i_p}e_{i_p} - \alpha_{j_1}e_{j_1} - \alpha_{j_2}e_{j_2} - \dots \alpha_{j_q}e_{j_q}$, "where all $\alpha_i \ge 0$"):

$$
x = \alpha_{i_1}e_{i_1} + \alpha_{i_2}e_{i_2} + \dots + \alpha_{i_p}e_{i_p} - \alpha_{j_1}e_{j_1} - \alpha_{j_2}e_{j_2} - \dots - \alpha_{j_q}e_{j_q}, \qquad \alpha_i \ge 0 \quad\text{(sic; subscripts as printed, see below)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $l$ | layer index; $x^{(l)}$ is the activity vector entering layer $l$ |
| $W$, $b$ | weight matrix and bias vector of the layer's affine map |
| $[\,\cdot\,]^+$ | ReLU, applied componentwise (negatives to zero) |
| $w_i$, $b_i$ | $i$-th row of $W$ ("kernel") and its bias component |
| $n$ | dimension of the layer's output = dimension of the input to the layer |
| $p$, $q$ | number of output units that are zero / positive for the target ($p+q=n$) |
| $i_1,\dots,i_p$ | indices of output components equal to $0$ |
| $j_1,\dots,j_q$ | indices of output components greater than $0$ |
| $\Pi_i$ | hyperplane $\{x : w_i^T x + b_i = 0\}$, the nullspace of kernel $i$ |
| $\pi_i$ | one-dimensional subspace $\bigcap_{j\ne i}\Pi_j$ |
| $e_i$ | basis vector spanning $\pi_i$, normalised so that $w_i^T e_i = 1$ |
| $\alpha_i$ | coordinates of $x$ in the basis $\{e_i\}$; $\alpha_j = w_j^T x$ |

## Main results

There are no experiments in the file; the result is the construction itself and the characterisation it implies. On the algorithmic side, the paper's claim is that the complete preimage is computable from the weights alone, in the basis $\{e_i\}$ obtained by deleting rows of $W$ and taking nullspaces (§1, p. 2). On the geometric side, it claims three things:

- the preimage of a single layer's activity is a linear manifold, a ray when exactly one unit is rectified and a higher-dimensional cone when several are (Fig. 1; §1, p. 2);
- the preimage of an activity of a deeper layer, mapped back to the input space, is a piecewise linear manifold, possibly with several branches (Abstract; §1, p. 3);
- the all-rectified case is degenerate: the grey region of Fig. 1 (right) consists of inputs that all map to output activity $(0,0)$ and are therefore "irreversibly mixed" — the map is not injective there (§1 p. 2; Fig. 1 caption, p. 3).

The worked example of Fig. 1 (left) is the only quantitative content: for three kernels, the three pairwise intersections of the hyperplanes give $e_1,e_2,e_3$, and the preimages shown are an arrow along $e_3$ for the pattern $(+,+,-)$, an arrow along $e_2$ for $(+,-,+)$, and the shaded area for $(+,-,-)$ (§1, p. 2).

## Limitations stated by the authors

The authors state that this is a workshop-track note and that they "are presently empirically evaluating the procedure and it's ability to extract complete preimages as well as the general structure of preimage manifolds" (Abstract) — so no empirical validation, no comparison with the optimisation-based preimages, and no complexity claim for the multi-layer case are given. They also flag the degenerate case $p=0$ (nothing rectified), where the preimage is a single point and the problem is trivially solved by the linear map (§1, p. 1).

## Critical assessment

My reading, not the paper's. This is a three-page note that states a construction and stops; several things a reader would want are absent. First, the inequalities in Eq. 1 and the coordinate requirements conflict: after Eq. 1 defines $i_1,\dots,i_p$ as the *zero* output indices and $j_1,\dots,j_q$ as the *positive* ones, the procedure on p. 2 says "identify indices $j=i_1,\dots,i_p$ where $w_j^T x^{(l)}+b_j > 0$ and $j=j_1,\dots,j_q$ where $w_j^T x^{(l)}+b_j \le 0$" — the two lists are swapped relative to Eq. 1. Likewise the sentence "Since for $j=i_1,\dots,i_p$ we will have $\alpha_j > 0$ and for $j=j_1,\dots,j_q$ we will have $\alpha_j \le 0$" contradicts $\alpha_j = w_j^T x$ and Eq. 1, where the *rectified* units have $w_i^T x + b_i \le 0$ (so $\alpha_i \le -b_i$, necessarily negative only if $b_i > 0$). These look like an index swap in the write-up rather than a substantive error, but they make the printed formula unusable without repair.

Second, the displayed manifold is malformed as printed: $x = \alpha_{i_1}e_{i_1} + \dots - \alpha_{j_1}e_{j_1} - \dots$ is followed by "where all $\alpha_i \ge 0$" and then, on p. 3, by a second version "$x - \alpha_{j_1}e_{j_1} - \dots - \alpha_{j_q}e_{j_q}$ where all $\alpha_i \ge 0$" that has lost its right-hand side. Reconstructed, the intent is clear: the *active* units ($j$) fix their coordinates at the target value through $\alpha_j = x_j^{(l+1)} - b_j$, while the *rectified* units ($i$) are free in the allowed half-space, giving rays along $\pm e_i$; the preimage is the target point plus a nonnegative cone generated by the $e_i$ of the rectified units. I take the paper to mean the latter, and note the printing is garbled.

Third, the word "complete" in the abstract is asserted, not proved: nothing shows that the union over activation patterns covers a full neighbourhood, and the "full rank" hypothesis is invoked once ("the assumed full rank of the mapping $W$") without discussion of the rank-deficient case, which is exactly where nullspaces become positive-dimensional and the construction fails. Since ReLU networks are routinely rank-deficient in individual layers, this is a real restriction. Fourth, the multi-layer claim — that mapping layer-level pieces back to the input space gives piecewise linear manifolds — is stated with no account of how many branches arise or how the pieces compose across layers, and the 3-unit/2-input example is far too small to demonstrate the "possibly multiple branches" of the abstract. No code, data or numerical experiment accompanies the note.

The construction is nevertheless conceptually sharp: the observation that the pre-activations are exactly the coordinates in the reciprocal basis makes the sign pattern of a ReLU layer a coordinate-sign condition, which is a clean way to see the layer's preimage as a cone in a basis determined by the weights. Later work outside this collection (the polyhedral-decomposition literature) arrives at the same local structure from the side of counting regions; this note is best read as an early, informal statement of that picture.

## Connections within the collection

Nothing in this paper cites another paper of the collection: the only references are to Mahendran & Vedaldi (2015, 2016), which are outside the collection, so `builds_on` is empty. Its subject matter places it squarely among the collection's polyhedral-geometry papers, and the relations below are mine.

The layer-level picture — an affine map, a sign pattern, and a cone of inputs sharing it — is the same object studied quantitatively by [@haninDeepReLUNetworks2019] and [@NIPS2014_fa6f2a46], where activation patterns and linear regions are counted rather than inverted; those works are the counting counterparts of this note's inversion. [@pascanuNumberResponseRegions2013] counts response regions of piecewise linear networks and shares the decomposition into polytopes; [@serraBoundingCountingLinear2018] bounds their number. The geometric object appearing here — pieces of linear manifolds glued along hyperplanes, with regions of the input collapsing to a single output — is studied directly by [@haninComplexityLinearRegions2019] and by the polyhedral-complex works [@berzinsPolyhedralComplexExtraction2023], [@liuReLUNeuralNetworks2023] and [@masdenAlgorithmicDeterminationCombinatorial2022], which give algorithms and combinatorics for the complex of linear regions that this paper's preimages are the cells of. The idea that a ReLU layer's local behaviour is captured by a sign pattern whose preimage is a cone is developed into a full spline theory by [@balestrieroSplineTheoryDeep2018], and the same "space folding/mixing" intuition is pursued experimentally by [@lewandowskiSpaceFoldsReLU2024]. Verification tools such as [@vincentReachablePolyhedralMarching2021] attack a related reachability question with polyhedral marching. Inversion of a network to recover inputs also underlies the diagnostic uses made of the geometry in [@raghuExpressivePowerDeep2017] and [@sattelbergLocallyLinearAttributes2023]; those are possibly related rather than direct continuations, since neither cites or is cited by this note.
