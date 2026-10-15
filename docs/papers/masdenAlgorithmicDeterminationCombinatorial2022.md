---
citekey: masdenAlgorithmicDeterminationCombinatorial2022
title: "Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks"
authors: [Marissa A. Masden]
year: 2022
venue: arXiv preprint
doi: "10.48550/arXiv.2207.07696"
arxiv: "2207.07696"
url: null
kind: method
source_type: pdf
source_sha256: 6dc3aff1ab63ed71cad0bea92cee288564b013b5132ee5d16b67e1075a4a79f0
pages: 23
ingested: 2026-10-15
tags: [relu-networks, polyhedral-complex, linear-regions, decision-boundary-topology, computational-geometry]
concepts_introduced:
  - sign sequence
  - sign sequence cubical complex
  - supertransversal neural network
  - sign sequence product
  - canonical polyhedral complex duality
  - decision boundary cochain complex
concepts_used:
  - canonical polyhedral complex
  - bent hyperplane
  - hyperplane arrangement
  - activation pattern
  - node map
  - generic neural network
  - polyhedral complex
  - Betti number
builds_on: [haninDeepReLUNetworks2019]
datasets: []
---

# Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks

**Masden (2022)**, *arXiv preprint*. [arXiv:2207.07696](https://arxiv.org/abs/2207.07696)

!!! abstract "TL;DR"
    The canonical polyhedral complex of a ReLU network has, at present, only its top-dimensional regions described (by activation patterns). Masden shows that labelling each cell by the *signs* of the network's intermediate node maps — a *sign sequence* in $\{-1,0,1\}^N$ that is zero exactly where the cell is cut by a bent hyperplane — is injective, and that the map to the hypercube is a combinatorial duality onto a pure cubical complex whose face relations obey a sign-sequence product. Since a vertex's sign sequence follows from which hyperplanes were intersected and the remaining signs of the network evaluation, the whole face poset can be recovered from vertex locations alone, giving an algorithm that is numerically stable and polynomial in the number of hidden units. Experiments on random networks show that width alone does not raise the topology of the decision boundary, while width together with depth does.

## Problem and motivation

For a fully connected ReLU network $F = A_m \circ F_{m-1} \circ \cdots \circ F_1$, the canonical polyhedral complex $C(F)$ decomposes input space into the cells on which $F$ is affine — the object whose regions are counted as a measure of expressivity (§1, p. 1). Prior work establishes the high-level cellular structure of $C(F)$ but not explicit lower-dimensional information: the $(n_0-1)$-skeleton is described as preimages of layer hyperplanes, while arbitrary $k$-skeletons for $k < n_0-1$ and general face relations are unexamined (§2, p. 2). Counting results capture only the top-dimensional regions [@NIPS2014_fa6f2a46; @serraBoundingCountingLinear2018; @haninDeepReLUNetworks2019], local models treat the structure at a single hyperplane intersection, and the preimage-based statistics of [@haninComplexityLinearRegions2019] are statistical rather than combinatorial (§2, p. 2).

The natural labelling by *activation patterns* — binary strings recording which neurons are active — is provably insufficient. Masden's Theorem 15 (stated in §1, p. 1, proved in Appendix A.2) exhibits two networks $F_1, F_2: \R^2 \to \R^4 \to \R$ whose top-dimensional cells carry the *same* set of activation-pattern strings, yet $C(F_1)$ and $C(F_2)$ are not combinatorially equivalent and their decision boundaries are not homotopy equivalent: one is bounded, the other unbounded (Fig. 7). Knowing which activation patterns occur therefore does not determine even the pairwise intersection properties of the regions, and intersecting top-dimensional regions directly is numerically unstable (§1, p. 1). Masden's alternative is to *work up from vertices*: locate the vertices of $C(F)$ and, for each, record the signs of every node map. The framework throughout is the one of the transversality paper of Grigsby and Lindsey (outside the collection), on which the definition of $C(F)$ and of transversality on cells is built.

## Method

**Sign sequences and supertransversality.** Writing $F_{ij}$ for the $(i,j)$-th node map (the activation of the $j$-th neuron of layer $i$ pulled back to input space), the *sign sequence* of a cell (Def. 13, App. A.2) is

$$
s_{ij}(C) = \mathrm{sgn}(F_{ij}(C)),
$$

so that $s(C) \in \{-1,0,1\}^N$ with $N$ the number of intermediate neurons, and a zero entry exactly where $C$ lies on a bent hyperplane $F_{ij}^{-1}(0)$. Theorem 14 proves $s$ is well defined and injective on cells of $C(F)$. The results require the network to be *generic* (each layer's hyperplane arrangement is generic) and *supertransversal* — each layer map transverse on cells of the previous arrangement to the interior of all cells of the later-layer complex (Def. 11) — a condition shown to be full measure in parameter space and fibrewise generic in the biases (Lemma 12, App. A.2). For such networks the number of zeros in $s(C)$ equals the codimension of $C$: a $k$-cell has exactly $n_0-k$ zeros (Lemma 16).

**Duality with a cubical complex.** The image $S(F)$ of $s$ is a pure, $n_0$-dimensional subcomplex of the hypercube $[-1,1]^N$ with its product CW structure (Thm. 20): vertices of $C(F)$ correspond to $n_0$-cells (cubes) of $S(F)$, and in general $k$-cells of $S(F)$ to codimension-$k$ cells of $C(F)$. The face poset of $C(F)$ is the *opposite* poset of that of $S(F)$ (Lemma 21). Because $S(F)$ is pure, knowing which $n_0$-cells occur determines every face relation in it; hence *the sign sequences of the vertices of $C(F)$ determine the whole face poset of $C(F)$* (Corollary, §3.2).

**The product.** Sign sequences carry a product compatible with faces (Lemma 18): for cells $C, D$ (Lemma 18, App. A.2),

$$
\bigl(S(C)\cdot S(D)\bigr)_{ij} =
\begin{cases}
S(C)_{ij} & \text{if } S(C)_{ij} \neq 0,\\[2pt]
S(D)_{ij} & \text{otherwise},
\end{cases}
$$

which is well defined as a sign sequence of some cell $E$ with $C \le E$, making the sign sequences a semigroup. It yields the face-poset tests of Lemma 19: $C \le D \iff S(C)\cdot S(D) = S(D)$; the product commutes iff $C$ and $D$ have a common coface; and $S(C)\cdot S(D) = S(C)$ iff every bent hyperplane containing $D$ also contains $C$.

**Algorithm.** Vertices are found layer by layer, iteratively refining $C(F^{(k)})$ (Computing Sign Sequences, §4.1). The $0$-cells of the first layer are the solutions of $\{W_\alpha x = b_\alpha : \alpha \subseteq [n_1],\, \lvert\alpha\rvert = n_0\}$, and a solution has $s_i = 0$ exactly for $i \in \alpha$ (Lemma 22). To pass from layer $k-1$ to $k$, loop over the regions $C$ of $C(F^{(k-1)})$ and, for each, solve systems pairing $1 \le \ell \le n_0$ new bent hyperplanes $A_{kj}(x)=0$ with $n_0-\ell$ earlier ones $A_{ij}(x)=0$ ($i<k$) that span a face of $C$; a solution is kept as a vertex iff it agrees with $C$ on every *un-crossed* node map $i \le k-1$ (Lemma 23). Crucially, whether an entry of a vertex's sign sequence is zero is decided by which hyperplanes were intersected, not by evaluating $F_{ij}$ numerically, so machine-epsilon errors in the linear solve do not corrupt the combinatorics (§4.3, p. 7). The runtime is polynomial in the number of hidden units and exponential in the input dimension $n_0$ (§1, p. 2; §4.3).

**Decision-boundary topology.** The cells with $S(C)_N = 0$ are those with $F(C) = 0$. Restricting the mod-two cellular coboundary of the cubical complex $S(F)$ to cells whose last sign is also zero yields a cochain complex whose coboundary is dual to the (mod-two) boundary of these cells (Lemma 21, §4.2). With one "vertex at infinity" adjoined to the unbounded edges, this computes the cochain complex of the one-point compactification of the decision boundary, whose Betti numbers can be obtained with standard chain-complex software (§4.2, p. 7).

## Key equations and definitions

The sign-sequence duality and its algebra are above (Def. 13, Lemmas 16, 18, 19, Thm. 20). The remaining definitions the algorithm turns on:

| Symbol | Meaning (in this paper) |
|---|---|
| $F = G \circ F_m \circ \cdots \circ F_1$ | ReLU network map, $F: \R^{n_0} \to \R$ |
| $n_0,\dots,n_m,1$ | architecture; $n_i$ neurons in hidden layer $i$ |
| $F_{ij}$ | node map, activation of neuron $j$ in layer $i$ pulled back to input space |
| $N$ | total number of intermediate neurons |
| $C(F)$ | canonical polyhedral complex of $F$ |
| $F_{ij}^{-1}(0)$ | bent hyperplane of node map $(i,j)$ |
| $s(C) = S(C)$ | sign sequence of cell $C$ in $\{-1,0,1\}^N$ |
| $S(F)$ | sign sequence cubical complex, dual to $C(F)$ |
| $k$-cell | cell of dimension $k$; has $n_0-k$ zeros in its sign sequence |
| $R^{(i)}$ | hyperplane-arrangement complex of layer $i$ |
| $W_\alpha, b_\alpha$ | weight rows and biases selected by index set $\alpha \subseteq [n_1]$ |
| $\beta_i$ | $i$-th Betti number of the compactified decision boundary |

The dimension bookkeeping (Lemma 16) is what makes the complex pure; the product (Lemma 18) is what lets vertex signs generate every face. Betti numbers of the one-point compactification split into bounded and unbounded components of the decision boundary by $\beta_0 - 1$ and $\beta_{n_0-1} - \beta_0 + 1$ respectively (§5.2, p. 8).

## Main results

**Theory.** $s$ is well defined and injective (Thm. 14); $S(F)$ is a pure $n_0$-dimensional cubical complex dual to $C(F)$, with the $k$-cells of $S(F)$ matching codimension-$k$ cells of $C(F)$ (Thm. 20); the face poset of $C(F)$ is the opposite of that of $S(F)$ and their mod-two boundary maps are dual (Lemma 21); and sign sequences form a semigroup whose product reproduces face relations (Lemma 18, 19). Theorem 15 shows activation patterns alone are insufficient, motivating the construction. Supertransversality excludes only a measure-zero set of parameters (Lemma 12).

**Experiments.** Fifty networks of each architecture $(k,n,1)$ ("shallow") and $(k,n,n,1)$ ("deep") with $2 \le k \le 4$ are randomly initialised with standard-normal weights and biases; $C(F)$ is computed with the algorithm (a PyTorch linear-algebra solver) and the Betti numbers of the compactified decision boundary from the boundary map of Lemma 21 via a Sage chain complex (§5.1, p. 8). This is claimed to be the first exact determination of decision-boundary topology for input dimension above two with more than one hidden layer (§5, p. 8).

The topology of shallow networks is nearly constant across input dimensions, whereas deep networks show greater variability and a distribution that shifts with width (§5.2, p. 8; Fig. 4). Bounded components are rare and rarer in higher input dimension: 8.1% of $(2,n,1)$ networks have at least one bounded component, versus 0.05% of $(4,n,1)$ (§5.2, p. 8). For unbounded components, shallow networks never exceeded 3, with mode 1 and averages 1.0, 1.03, 1.04 for $n_0 = 2, 3, 4$; deep networks averaged 1.1, 1.4, 1.4 and reached maxima of 5, 7, 12 (§5.2, p. 8). Table 2 reports, e.g., $\beta_0 = 1.06 \pm 0.034$ and $\beta_{n_0-1} = 1.06 \pm 0.059$ for $(2,5,1)$ versus $\beta_0 = 1.04 \pm 0.03$ and $\beta_{n_0-1} = 1.88 \pm 0.30$ for $(4,8,8,1)$: deeper architectures show greater variability and greater change with width in $\beta_{n_0-1}$ (Tab. 2, p. 9). The conclusion drawn is that depth influences topological expressivity more strongly than the number of intermediate neurons, at least at initialisation (§5.2, p. 10).

## Limitations stated by the authors

The algorithm is exponential in the input dimension, so realistically only low-dimensional slices of a network's true decision boundary can be studied empirically (§6, p. 10). The experiments are confined to initialisation and to networks of depth at most two hidden layers and width at most 15 (§5.1, p. 8). The theory holds only for generic, supertransversal networks, though the excluded set is measure zero (Lemma 12). The construction requires at least $n_0$ hidden units in the first layer for $C(F)$ to have vertices (Thm. 20). The presence of unbounded cells means the coboundary map is not quite a cellular chain complex unless a vertex at infinity is added (§4.2, p. 7).

## Critical assessment

The theoretical core is convincing and unusually complete: injectivity of the sign sequence, the duality with a pure cubical complex, the product, and the vertex-driven algorithm fit together with full proofs, and the decision to determine zeros combinatorially (from intersected hyperplanes) rather than numerically is a genuine and well-argued fix for the instability of direct polygon intersection. The self-reported claim of being the first publicly available code for the full face poset is plausible given the related-work survey, and the code is released (github.com/mmasden/canonicalpoly).

The empirical part is thinner than the theory. All results are at initialisation, on 50 networks per architecture, with only averages and standard errors reported — no significance tests, and the "width in the presence of depth" conclusion rests on small differences (e.g. $\beta_{n_0-1} = 1.88 \pm 0.30$ for $(4,8,8,1)$) over a narrow width range. The conjecture that deep networks need more width before reaching a "Gaussian regime" is offered without evidence (§5.2, p. 8). The extracted Table 2 text is partly garbled in the source file, so one entry must be read with care. Finally, the framework inherits from Grigsby–Lindsey the assumption of genericity/supertransversality; whether the combinatorial classification is stable under perturbation near the measure-zero exceptional parameters — the paper conjectures vertex sets change only at finitely many training thresholds (§6, p. 11) but proves nothing here — is left open.

## Connections within the collection

The paper is a competitor and complement to other exact-extraction methods in the collection. [@berzinsPolyhedralComplexExtraction2023] also recovers the polyhedral complex exactly but works from the 1-skeleton by edge subdivision, in $O(\lvert V\rvert)$ time and on a GPU, and explicitly contrasts its approach with the region-based alternatives; Masden instead works from vertices and sign sequences, trading the GPU-amenable 1-skeleton for a combinatorial signature that gives the full face poset directly. [@humayunSplineCamExactVisualization2024] and [@vincentReachablePolyhedralMarching2021] pursue explicit enumeration from the region/verification side, while [@liuReLUNeuralNetworks2023] extracts the dual decomposition and its persistent homology — the same dual cubical object Masden constructs, studied topologically rather than combinatorially. The question of adjacency of regions and connectivity of lower-dimensional facets, which Masden identifies as undocumented, is treated locally for space folds by [@lewandowskiSpaceFoldsReLU2024] and via preimages of rectifier activities by [@carlssonPREIMAGERECTIFIERNETWORK2017]; locally linear attributes of individual regions are the focus of [@sattelbergLocallyLinearAttributes2023].

For the counting side of expressivity, the paper's motivation rests on [@NIPS2014_fa6f2a46], [@serraBoundingCountingLinear2018], [@haninDeepReLUNetworks2019] and [@haninComplexityLinearRegions2019], and its polynomial-runtime claim uses the polynomial number of regions at initialisation established by [@haninDeepReLUNetworks2019]. The tropical-geometry characterisations of the polyhedral complex [@zhangTropicalGeometryDeep; @alfarraDecisionBoundariesNeural2023] are the main competing description of the same object: Masden traces them to integer-coefficient tropical rational functions and notes (as of writing) applicability limited to single hidden layers (§2, p. 2). The same authors' work on ReLU symmetries, [@grigsbyHiddenSymmetriesReLU], concerns parameter-space redundancy rather than the input-space complex and is only indirectly related. The transversality framework of Grigsby and Lindsey (2020) is the foundation of the entire construction but is outside the collection, as are the hyperplane-arrangement theory of Aguiar–Mahajan and Grunert's piecewise-linear Morse theory.
