---
citekey: phuongFunctionalVsParametric2019
title: "Functional vs. Parametric Equivalence of ReLU Networks"
authors: [Mary Phuong, Christoph H. Lampert]
year: 2019
venue: International Conference on Learning Representations (ICLR)
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: 89437ca71e4c9588177a2307db1c32535efd477ed20f588cc583fabeeeab354f
pages: 23
ingested: 2026-10-15
tags: [relu-networks, network-symmetries, piecewise-linear, identifiability, linear-regions]
concepts_introduced:
  - functional equivalence of ReLU networks
  - fold-set
  - piece-wise linear surface
  - transparent ReLU network
  - general ReLU network
  - canonical representation of a piece-wise linear surface
  - dependency graph of a piece-wise linear surface
concepts_used:
  - ReLU network
  - permutation of units
  - positive scaling symmetry of weights
  - linear region
  - zero-level set
  - piece-wise hyperplane
builds_on: [raghuExpressivePowerDeep2017, serraBoundingCountingLinear2018, haninComplexityLinearRegions2019]
datasets: []
---

# Functional vs. Parametric Equivalence of ReLU Networks

**Phuong, Lampert (2019)**, *ICLR*. (The PDF is headed "Published as a conference paper at ICLR 2020"; the `.bib` entry gives 2019.)
<!-- The extracted PDF has no DOI/arXiv metadata; ICLR proceedings page not present in the file. -->

!!! abstract "TL;DR"
    For feed-forward ReLU architectures with non-increasing widths, the authors prove that permutation of
    units and positive scaling of incoming/outgoing weights are the *only* weight-space transformations
    that leave the implemented function unchanged — provided one restricts attention to a specifically
    constructed network. The proof introduces the *fold-set* (the non-differentiable points of a
    piece-wise linear surface) and shows that the boundaries between linear regions, organised into a
    *dependency graph*, determine the weights up to permutation and scaling. There are no experiments.

## Problem and motivation

Over-parameterisation is usually credited with redundant function representation, yet the *exact*
redundancy of the ReLU parameterisation had not been quantified. Two function-preserving weight
transformations are classical for feed-forward architectures (§1, p. 1):

1. permutation of units within a layer, $W_l \mapsto P W_l$, $b_l \mapsto P b_l$, $W_{l+1} \mapsto W_{l+1} P^{-1}$ (Eq. 1–2);
2. positive scaling of all incoming weights of a unit together with inverse scaling of its outgoing
   weights, written per layer with a diagonal matrix $M$ as $W_l \mapsto M W_l$, $b_l \mapsto M b_l$,
   $W_{l+1} \mapsto W_{l+1} M^{-1}$ (Eq. 3–4).

The question is whether these exhaust the transformations $T$ of the weight space with
$h_{T(\theta)}(z) = h_\theta(z)$ for all $z$ in a domain $Z \subseteq \R^{d_0}$. Earlier work resolved
the analogous question for smooth activations: for tanh networks Chen et al. (1993) showed that any
smooth map preserving the function of *all* networks is a composition of permutations and sign flips,
and Fefferman & Markel (1994) showed the input-output map determines architecture and weights up to
permutations and sign flips (§2, p. 2). The authors state that no such result existed for ReLU, which
is non-saturating, asymmetric and non-smooth, and that this is the gap they close. The result is
positioned against *inverse stability* work (Petersen et al. 2018; Berner et al. 2019), which asks
whether functionally close networks are parameterically close — here the question is exact equality of
functions.

## Method

The paper is a mathematical proof; there is no algorithm, training procedure or dataset. The argument
is built from three definitions and a chain of lemmas (§3–§6).

**Setting.** A ReLU network is a composition of affine maps and the element-wise ReLU
$\sigma(u)_i = \max\{0, u_i\}$; the parameters are the weight matrices $W_l \in \R^{d_l \times d_{l-1}}$
and biases $b_l \in \R^{d_l}$, with $d_L = 1$ (§3, p. 3). Two restricted classes are introduced. A
network is **general** if it avoids three degeneracies: no unit has a local optimum of $h^{1:l}_i$ at
value exactly zero; certain products of masked weight matrices have maximal rank (Eq. 8); and no two
units realise linear functions that are scalar multiples of each other (§3, p. 3). "Almost all"
networks are general — sampling weights from a distribution with a density suffices (footnote 1;
Lemma A.10–A.12). A network is **transparent** if for every input and every layer $l \in [L-1]$ at
least one unit is active: $\exists i$ with $h^{1:l}_i(z) \ge 0$ (§3, p. 3).

**Fold-sets.** Because ReLU networks are piece-wise linear, the boundaries between linear regions
carry information. The *fold-set* $F(f)$ of a continuous piece-wise linear $f$ is the set of its
non-differentiable points (§4, p. 3). Lemma 1 shows that for a general and transparent network the
fold-set is exactly the union of the units' zero-level sets (Eq. 9), so the end-to-end map reveals
the hidden units — this motivates the name "transparent".

**Piece-wise linear surfaces.** Section 5 formalises the picture: first-layer units are linear, so
their zero-level sets are hyperplanes that partition $Z$; second-layer units are linear on each cell
of that partition, so their zero-level sets are hyperplanes *within* cells, and so on. This yields the
notions of a *piece-wise hyperplane* (a hyperplane restricted to one partition cell) and a
*piece-wise linear surface of order $\ell$*, a union of piece-wise hyperplanes whose dependencies
have depth at most $\ell$ (§5, p. 4). The operator $\bigcup_k S$ collects all piece-wise linear
surfaces of order at most $k$ contained in $S$ (Eq. 10); the *canonical representation* is the unique
decomposition into distinct piece-wise hyperplanes whose partial unions reproduce $\bigcup_k S$
(Lemma A.5–A.6). It assigns each piece-wise hyperplane a well-defined "layer". The *dependency graph*
then has the piece-wise hyperplanes as vertices and an edge $H^l_i \to H^k_j$ iff $l < k$ and
$\mathrm{relint}\,H^l_i \cap \mathrm{cl}\,H^k_j \ne \emptyset$ — i.e. $H^k_j$ "bends at" $H^l_i$
(§5, p. 6).

**Proof strategy.** The key geometric observation (Lemma 3) is that a fold with a chain of $m$
descendants in the dependency graph must originate from a layer at least $m$ layers below the last
one. Hence a *well-connected* fold-set identifies which hyperplanes belong to the first layer. Lemma 4
constructs a network whose fold-set has, at every layer, $d_l$ directed paths of length $L-1-l$ with
distinct starting vertices, guaranteeing this connectivity. Theorem 1 then follows by induction over
layers: recover the first layer's hyperplanes, apply the same argument to the truncated network
$h^{l:L}$, and finish with the last layer, which is a single linear function on a full-dimensional
domain (§6, p. 6–8).

## Key equations and definitions

Network and truncated networks (§3, Eq. 6–7):

$$
h = h_L \circ h_{L-1} \circ \dots \circ h_1, \qquad h_l(z) = W_l z + b_l, \qquad
h^{l:k} = h_k \circ h_{k-1} \circ \dots \circ h_l \qquad \text{(Eq. 6–7)}
$$

Fold-set of a general and transparent network (§4, Lemma 1, Eq. 9) — the union of all units'
zero-level sets:

$$
F(h) = \bigcup_{l,i} \left\{ z \mid h^{1:l}_i(z) = 0 \right\} \qquad \text{(Eq. 9, p. 4)}
$$

Main theorem (Thm. 1, Eq. 5 / Eq. 11): for an architecture with non-increasing widths
$d_0 \ge d_1 \ge \dots \ge d_{L-1} \ge 2$, $d_L = 1$, there exists a ReLU network
$h$ with parameters $(W_1,b_1,\dots,W_L,b_L)$ such that every *general* $\tilde h$ with the same
architecture and $\tilde h(z) = h(z)$ on $Z$ satisfies

$$
\begin{aligned}
W_1 &= M_1 P_1 W_1^0, &\quad b_1 &= M_1 P_1 b_1^0, \\
W_l &= M_l P_l W_l^0 P_{l-1}^{-1} M_{l-1}^{-1}, &\quad b_l &= M_l P_l b_l^0,
  &\quad l &\in \{2,\dots,L-1\}, \\
W_L &= W_L^0 P_{L-1}^{-1} M_{L-1}^{-1}, &\quad b_L &= b_L^0,
\end{aligned}
\qquad \text{(Eq. 11)}
$$

with permutation matrices $P_l$ and positive diagonal matrices $M_l$, and $(W^0_1,b^0_1,\dots,W^0_L,b^0_L)$
the parameters of $\tilde h$.

Layer-wise recovery (Eq. 16): the zero-level sets determine each unit's parameters up to a scalar per
unit,

$$
(W_l[i,:],\, b_l[i]) = m_i \left( W_l^0[\sigma(i),:],\, b_l^0[\sigma(i)] \right) \qquad \text{(Eq. 16, p. 8)}
$$

for a permutation $\sigma$ of $[d_l]$.

| Symbol | Meaning (in this paper) |
|---|---|
| $Z \subseteq \R^{d_0}$ | bounded open non-empty input domain |
| $d_l$ | width of layer $l$; $d_L = 1$ |
| $W_l, b_l$ | weights and biases of layer $l$ |
| $h^{1:l}_i$ | pre-activation of unit $i$ in layer $l$ (truncated network) |
| $F(f)$ | fold-set of $f$: the points where $f$ is non-differentiable |
| $F^l_i(h)$ | unit $(l,i)$ fold-set: where $h^{1:l}_i(z) = 0$ and $h^{1:l}_i$ is positive nearby |
| $\mathcal{P}_Z(S)$ | partition of $Z$ induced by $S$ (connected components of $Z \setminus S$) |
| $\bigcup_k S$ | largest piece-wise linear surface of order $\le k$ contained in $S$ (Eq. 10) |
| $H^l_i$ | piece-wise hyperplane (vertex of the dependency graph) |
| $P_l, M_l$ | permutation matrix, positive diagonal matrix at layer $l$ |
| $I_l(z)$ | activation indicator diagonal, entries $1_{h^{1:l}_i(z) \ge 0}$ |

## Main results

- **Theorem 1 (Thm. 1, §1 p. 2 and §6 p. 6).** For any architecture with non-increasing widths and
  $d_L = 1$, there *exists* a ReLU network $h$ such that the input-output mapping determines the
  parameters of any general equivalent network uniquely up to permutations and positive scalings (Eq. 11).
  The quantifier is existential over $h$: the theorem does not claim every network is identifiable.
- **Lemma 1 (p. 4).** For a general transparent network, $F(h) = \bigcup_{l,i}\{z \mid h^{1:l}_i(z)=0\}$
  (Eq. 9): the end-to-end function exposes the hidden units' zero sets.
- **Lemma 2 (p. 4).** The fold-set of a general transparent ReLU network is a piece-wise linear
  surface of order at most $L-1$.
- **Lemma 3 (p. 7).** Each piece-wise hyperplane $H^k_j$ in the canonical representation is contained
  in some unit's zero-level set $\{z \mid h^{1:l}_i(z)=0\}$ with $l \le k$, and a directed path of
  length $m$ starting at $H^k_j$ forces $l \ge m$. This is the mechanism that assigns layers.
- **Lemma 4 (p. 7).** There exists a general transparent network whose layer-$l$ fold-set has $d_l$
  directed paths of length $L-1-l$ with distinct starting vertices, i.e. a well-connected dependency
  graph.
- **Discussion of assumptions (§6, p. 8).** The proof needs non-increasing widths so that the
  interiors $\mathrm{int}\,Z^{l-1}$ are non-empty; the authors *conjecture the theorem does not hold*
  for more general architectures. Almost the whole proof carries over to leaky ReLU, except the step
  showing $M_l$ has positive diagonal entries. No numerical results, figures or tables of measurements
  are reported; Figure 1–4 are schematic illustrations of fold-sets and dependency graphs.

## Limitations stated by the authors

- Non-increasing widths and $d_L = 1$ are essential to the argument; the authors conjecture the result
  fails for wider intermediate layers, and note that prior theory sometimes assumes much wider
  intermediate layers than the input dimension, where redundancy may be larger (§6, p. 8; §7, p. 9).
- The theorem only asserts existence of an identifiable network, not that typical or trained networks
  are identifiable. "It could be that 'typical networks' do not have well connected dependency graphs,
  and are therefore not easily identifiable" (§7, p. 9).
- Exact equivalence classes may be small while *approximate* equivalence is much easier to achieve, so
  small functional classes need not imply little redundancy in practice (§7, p. 9).
- The leaky-ReLU extension is incomplete — positivity of the diagonal scalings is not proved, though
  the authors "cannot rule out that a simple workaround exists" (§6, p. 8).
- Boundedness of $Z$ is used to guarantee transparency of the first layer and "can be lifted if we
  consider a slightly modified definition of transparency" (§6, p. 8).

## Critical assessment

My reading. The central result is carefully quantified and easy to overstate: Theorem 1 gives, for each
eligible architecture, *one* special network whose function determines its parameters; it is silent
about general networks, and the discussions in §6–§7 are unusually candid that this is the weak point.
The paper therefore answers "how redundant can the ReLU parameterisation be forced to be?" rather than
"how redundant is it for networks we train?". The geometric machinery (fold-sets, canonical
decompositions, dependency graphs) is the more durable contribution and is reused as a language for
linear-region structure. The generality claims rest on measure-theoretic arguments in Appendix A.3
(Lemmas A.10–A.12): "general" excludes a closed zero-measure set, and Lemma A.7 and Lemma A.9 are
left to the reader, so a reader reproducing the proof must fill small gaps. Transparency, the key
technical hypothesis, is a global activation-count condition that is hard to certify for a given
trained network, and the proof's own construction enforces it by hand, which limits the practical
reach. There are no experiments at all, and none would be straightforward — the objects are exact
equivalence classes of a real analytic-type construction. The result invites the natural open
questions the authors raise: whether approximate equivalence classes are large, whether trained
networks have connected dependency graphs, and whether the conjecture (failure for non-nested widths)
is true. Reproducibility of the *statement* is high — it is a theorem — but it inherits the usual
caveat that no code or numerical check accompanies it.

## Connections within the collection

The paper belongs to the linear-region / piece-wise-linear-geometry thread of the collection. It draws
directly on the empirical-theoretical picture of fold geometry from [@raghuExpressivePowerDeep2017]
(cited in §2, p. 2 and Fig. 2 caption, p. 5) and on the region-counting framework of
[@serraBoundingCountingLinear2018] (§2, p. 2), and it states that "somewhat similar proof techniques"
were used by [@haninComplexityLinearRegions2019] for counting linear decision regions (§2, p. 2).

Its main theorem is the exact-equivalence counterpart to the *hidden-symmetry* question of
[@grigsbyHiddenSymmetriesReLU]: that work proves that for architectures with no layer narrower than
the input there exist parameter settings with no hidden symmetries, i.e. where the symmetries are
exactly permutation and positive scaling — the same conclusion Phuong & Lampert reach
constructively, but under the *opposite* width condition (non-increasing widths here vs. no layer
narrower than the input there). Reconciling the two width regimes would be a natural next step, and is
listed here as possibly related rather than claimed equivalent.

The study of fold-sets also sits next to the space-folding line of
[@lewandowskiSpaceFoldsReLU2024], which quantifies how straight paths fold across layers, and to the
boundary/preimage views of [@carlssonPREIMAGERECTIFIERNETWORK2017] and
[@dhayalkarDiscreteFunctionalGeometry2025] (transition graphs of activation patterns). The
piece-wise-linear-surface and piece-wise-hyperplane definitions are close in spirit to the polyhedral
decompositions used by [@masdenAlgorithmicDeterminationCombinatorial2022],
[@brandenburgDecompositionPolyhedraPiecewise2024] and [@goujonNumberRegionsPiecewise2023], and connect
to the tropical-geometry view of decision boundaries [@alfarraDecisionBoundariesNeural2023;
@zhangTropicalGeometryDeep]. Outside the collection, the classical identifiability results it extends
are Chen et al. (1993) and Fefferman & Markel (1994).
