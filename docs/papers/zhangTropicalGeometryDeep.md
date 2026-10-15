---
citekey: zhangTropicalGeometryDeep
title: "Tropical Geometry of Deep Neural Networks"
authors: [Liwen Zhang, Gregory Naitzat, Lek-Heng Lim]
year: 2018
venue: Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: b0ce24ae87e498d12d25544f8adcc8db1247d00cc46804453f97dd1a29d14fa7
pages: 9
ingested: 2026-10-15
tags: [tropical-geometry, relu-networks, linear-regions, expressiveness, piecewise-linear, polytopes]
concepts_introduced:
  - tropical rational function
  - tropical rational map
  - tropical hypersurface
  - dual subdivision of the Newton polygon
  - zonotope characterization of shallow ReLU networks
  - tropical characterization of ReLU networks
concepts_used:
  - tropical semiring
  - max-plus algebra
  - ReLU network
  - linear region
  - Minkowski sum
  - Newton polygon
  - difference-of-convex function
builds_on: [NIPS2014_fa6f2a46, raghuExpressivePowerDeep2017]
datasets: []
---

# Tropical Geometry of Deep Neural Networks

**Zhang, Naitzat, Lim (2018)**, *Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80*.

!!! abstract "TL;DR"
    Feedforward ReLU networks with integer weights and threshold activations are shown to be *exactly* the tropical rational maps — tropical quotients of two tropical polynomials — via an explicit layer-wise recursion. Reading a network as a tropical geometric object then gives a polytope (zonotope) picture of a one-hidden-layer network, a bound on the number of connected decision regions and a containment of the decision boundary in a tropical hypersurface, and an upper bound $O(n^{d(L-1)})$ on the number of linear regions that reproduces the known deep-vs-shallow gap.

## Problem and motivation

Theoretical understanding of why deep ReLU networks work remains incomplete: complexity measures such as VC-dimension and Rademacher complexity fail to explain their generalization behaviour, and the promise of implicit regularization is unresolved (§1). Earlier work argues that deep architectures use parameters more efficiently than shallow ones, using exponentially fewer parameters for certain function families; the paper aims to give that intuition a precise algebraic-geometric setting by bridging neural networks and tropical geometry, a branch of algebraic geometry that is "relatively obscure outside pure mathematics" (§1). The starting analogy is rational approximation: approximating a target by a ratio of two polynomials instead of a single polynomial improves quality without increasing degree, and the paper claims the same analogue holds for ReLU networks. The central claim is that the family of functions represented by feedforward ReLU networks with integer weights is *exactly* the family of tropical rational maps, from which a semifield structure, zonotope building blocks, decision-boundary results and a depth-separation bound are derived (§1).

## Method

Work with the *tropical semiring* $\mathbb{T} := \R \cup \{-\infty\}$ (the max-plus algebra) under tropical addition $\oplus$, multiplication $\odot$ and quotient $\oslash$ (Def. 2.1). Over this semifield one defines tropical monomials, polynomials and, crucially for networks, rational functions — tropical quotients of two tropical polynomials (Defs. 2.2–2.4) — and vector-valued polynomial and rational maps $\mathrm{Pol}(d,p)$, $\mathrm{Rat}(d,p)$ (Def. 2.5). A tropical polynomial $f$ defines a convex piecewise-linear function, while a tropical rational function is a difference-of-convex (DC) function, which is exactly the structure a nonconvex network should have (§2).

The neural-network setting is standard fully connected feedforward networks with the assumptions (a) integer weight matrices $A^{(l)} \in \Z^{n_l \times n_{l-1}}$, (b) real bias vectors, and (c) threshold activations $\rho^{(l)}(x) := \max\{x, t^{(l)}\}$ with $t^{(l)} \in (\R \cup \{-\infty\})^{n_l}$ (§4). The integer-weight restriction is argued to be no loss of generality (rational weights approximate reals, then clear denominators; a positive rescaling of all weights and biases does not change the network). The activation class contains ReLU ($t=0$) and identity ($t=-\infty$) and extends to leaky ReLU, absolute value, and with extra work max pooling/maxout, but not to tanh or sigmoid.

The method is then a layer-wise induction: split each integer weight matrix as $A = A_+ - A_-$ with nonnegative integer parts; a one-layer network coordinate is a difference of two tropical polynomials, and this decomposition is applied recursively to propagate tropical polynomial representations through the layers (Prop. 5.1). The result is that the preactivation and output of layer $l+1$ are again tropical rational functions, giving the theorem that every network satisfying (a)–(c) is a tropical rational map (Thm. 5.2), and conversely that any tropical rational function is realizable as a network of depth controlled by the number of monomials (Thm. 5.4). With the tropical dictionary established, geometry is read off: linear regions correspond to vertices on upper faces of the polytope $P(f)$, and the layer-to-layer transformation of these polytopes is governed by the tropical sum, product and power rules (Props. 3.1–3.2).

## Key equations and definitions

Tropical arithmetic (Def. 2.1):

$$
x \oplus y := \max\{x,y\}, \qquad x \odot y := x + y, \qquad x \oslash y := x - y \qquad \text{(Def. 2.1, §2)}
$$

A tropical polynomial is a finite tropical sum of monomials, and a tropical rational function a tropical quotient of two:
$$
f(x) = c_1 x^{\alpha_1} \oplus \cdots \oplus c_r x^{\alpha_r}, \qquad f \oslash g = f \odot g^{\odot(-1)} \qquad \text{(Defs. 2.3–2.4, §2)}
$$
so that every network is $F \oslash G = F \odot G^{\odot(-1)}$ with $F,G$ tropical polynomial maps (Thm. 5.2).

The tropical hypersurface of $f$ is where the maximum is attained by two or more monomials, i.e. where $f$ is not linear:
$$
T(f) := \left\{ x \in \R^d : c_i x^{\alpha_i} = c_j x^{\alpha_j} = f(x) \ \text{for some } i \neq j \right\} \qquad \text{(Def. 3.1, §3)}
$$
It divides $\R^d$ into convex polyhedral cells on which $f$ is linear; the number of linear regions is thus bounded by the number of vertices of the dual subdivision.

The polytope $P(f)$ is the convex hull of the lifted exponents, and this lift generates the dual subdivision of the Newton polygon (Eq. 1, §3):
$$
P(f) := \operatorname{Conv}\{ (\alpha_i, c_i) \in \R^d \times \R : i = 1,\dots,r \} \qquad \text{(Eq. 1, §3)}
$$
with $P(f^{\odot a}) = aP(f)$ for $a \in \N$ (Prop. 3.1) and, for $f,g$ tropical polynomials, $P(f \odot g) = P(f) + P(g)$ and $P(f \oplus g) = \operatorname{Conv}\{V(P(f)) \cup V(P(g))\}$ (Prop. 3.2).

The layer-wise recursion, in the non-tropical form of Prop. 5.1 (with $t := t^{(l+1)}$ and writing $F^{(l)}, G^{(l)}$ for the layer's tropical polynomial maps):
$$
F^{(l+1)}(x) = \max\{H^{(l+1)}(x),\, G^{(l+1)}(x) + t\},\quad
G^{(l+1)}(x) = A_+ G^{(l)}(x) + A_- F^{(l)}(x),
$$
$$
H^{(l+1)}(x) = A_+ F^{(l)}(x) + A_- G^{(l)}(x) + b \qquad \text{(Prop. 5.1, §5)}
$$
In tropical arithmetic the coordinate form (Eq. 2, §5) is
$$
f_i^{(l+1)} = h_i^{(l+1)} \oplus \bigl(g_i^{(l+1)} \odot t_i^{(l+1)}\bigr),
$$
$$
g_i^{(l+1)} = \Bigl(\bigodot_{j=1}^{n_l} (g_j^{(l)})^{\odot a_{ij}^+}\Bigr) \odot \Bigl(\bigodot_{j=1}^{n_l} (f_j^{(l)})^{\odot a_{ij}^-}\Bigr),\quad
h_i^{(l+1)} = \Bigl(\bigodot_{j=1}^{n_l} (f_j^{(l)})^{\odot a_{ij}^+}\Bigr) \odot \Bigl(\bigodot_{j=1}^{n_l} (g_j^{(l)})^{\odot a_{ij}^-}\Bigr) \odot b_i^{(l+1)} \qquad \text{(Eq. 2, §5)}
$$
(The extracted text garbles the operators and exponent signs of Eq. 2; the form above is the one consistent with the unambiguous non-tropical statement of Prop. 5.1.)

Decision boundary: if $\nu = f \oslash g$ and $s$ is injective with decision threshold $c$, the boundary is contained in a tropical hypersurface,
$$
B \subseteq T\bigl(s^{-1}(c) \odot g \ominus f\bigr), \qquad \text{where } s^{-1}(c) \odot g(x) \ominus f(x) = \max\{f(x),\, g(x) + s^{-1}(c)\} \qquad \text{(Eq. 3, §6.1)}
$$
the containment being strict in general because $T$ may further split a positive or negative region into several linear regions.

| Symbol | Meaning (in this paper) |
|---|---|
| $\mathbb{T} = \R \cup \{-\infty\}$ | tropical semiring (max-plus algebra) |
| $\oplus, \odot, \oslash$ | tropical sum, product, quotient |
| $x^{\odot a}$ | tropical power, $x^{\odot a} = a x$ |
| $f, g \in \mathbb{T}[x_1,\dots,x_d]$ | tropical polynomials |
| $f \oslash g$ | tropical rational function |
| $\mathrm{Pol}(d,p), \mathrm{Rat}(d,p)$ | tropical polynomial / rational maps |
| $T(f)$ | tropical hypersurface of $f$ |
| $\Delta(f)$, $P(f)$ | Newton polygon, lifted polytope of $f$ |
| $N(F)$ | number of linear regions of $F$ |
| $A^{(l)}, b^{(l)}, t^{(l)}$ | weights, biases, thresholds of layer $l$ |
| $n_l$ | width (number of nodes) of layer $l$, $n_0 = d$, $n_L = p$ |
| $r_f, r_g$ | number of monomials of $f$, $g$ |

## Main results

**Tropical characterization (Thm. 5.2).** Every feedforward network satisfying (a)–(c) is a function $\nu : \R^d \to \R^p$ whose coordinates are tropical rational functions of the input, i.e. $\nu = F \oslash G = F \odot G^{\odot(-1)}$ with $F, G$ tropical polynomial maps; so $\nu$ is a tropical rational map.

**Equivalence (Thm. 5.4).** (i) $\nu : \R^d \to \R$ is a tropical rational function if and only if it is a feedforward network satisfying (a)–(c). (ii) A tropical rational function $f \oslash g$ can be represented by an $L$-layer network with $L \le \max\{\lceil \log_2 r_f \rceil, \lceil \log_2 r_g \rceil\} + 2$, where $r_f, r_g$ are the numbers of monomials of $f$ and $g$. Combined with Cor. 5.3 (integer-weight ReLU network with linear output is a tropical rational function) and Prop. 5.5 ($\nu$ is a continuous piecewise-linear function with integer coefficients iff it is a tropical rational function), the paper concludes the equivalence of three families: tropical rational functions, continuous piecewise-linear functions with integer coefficients, and networks satisfying (a)–(c). It notes the converse fails without integrality — tropical *signomials* (real, possibly negative exponents) give a weaker Prop. 5.6 (every ReLU network is a tropical rational signomial map) that falls outside tropical algebraic geometry.

**Decision boundaries (Prop. 6.1).** For an $L$-layer network with $t^{(L)} = -\infty$, an injective score $s$ and threshold $c$ in its range: (i) $B$ divides $\R^d$ into at most $N(f)$ connected positive regions and at most $N(g)$ connected negative regions; (ii) $B$ is contained in the tropical hypersurface of $s^{-1}(c) \odot g \ominus f$ (Eq. 3). Related: the paper cites (Arora et al., 2018, Thm. 2.1) for the analogous real-weight equivalence between $L$-layer ReLU networks and $d$-variate continuous piecewise-linear functions with real coefficients, with $L \le \lceil \log_2(d+1)\rceil + 1$.

**Zonotopes as building blocks (Lemma 6.2).** $P(g_i^{(1)})$ and $P(h_i^{(1)})$ are points, $P(f_i^{(1)})$ is a line segment, and the $P(g_i^{(2)}), P(h_i^{(2)})$ of the second layer are zonotopes; from layer 2 on, $P(g_i^{(l+1)})$ and $P(h_i^{(l+1)})$ are weighted Minkowski sums of the previous layer's polytopes plus $\{b_i e\}$. Hence zonotopes are the geometric building blocks, connecting networks to the extensive convex-geometry literature on zonotopes and hyperplane arrangements.

**Deep vs shallow (Thm. 6.3).** For an $L$-layer network with $t^{(L)} = -\infty$ and $n_l \ge d$ for $l = 1,\dots,L-1$, the number of linear regions is at most
$$
\prod_{l=1}^{L-1} \sum_{i=0}^{d} \binom{n_l}{i} \quad\text{linear regions, and in particular } O\bigl(n^{d(L-1)}\bigr) \text{ when } n_1 = \cdots = n_{L-1} = n \ge d \qquad \text{(Thm. 6.3, §6.3)}
$$
The proof for $L = 2$ follows from Lemma 6.2 and Cor. 3.4 (zonotope vertex bound), with $L \ge 3$ deferred to Section D.7 of the supplement. The paper states explicitly that this bound does *not* improve on that of [@raghuExpressivePowerDeep2017] and is more restrictive (it applies only to (a)–(c)); the point is that tropical geometry recovers it. It notes this closely matches the lower bound $(n/d)^{d(L-1)} n^d$ of (Montúfar et al., 2014, Cor. 5) when all widths equal $n \ge d$, leading to the conclusion that the number of linear regions grows polynomially in the width $n$ and exponentially in the depth $L$.

## Limitations stated by the authors

The paper restricts attention to integer weights and threshold activations (a)–(c); the framework does not apply to tanh or sigmoid activations, and extends to leaky ReLU, absolute value, and only "with some extra effort" to max pooling and maxout nets (§4). It concedes the Thm. 6.3 bound is not an improvement over [@raghuExpressivePowerDeep2017] and is more restrictive (§6.3), so the contribution there is methodological rather than quantitative. Tropical rational signomials (Prop. 5.6), though a more general characterization, are deliberately not pursued because they lie outside tropical algebraic geometry (§5). The authors frame the whole article as "a first step ... a proof-of-concept" and note that only a small part of tropical geometry is touched (§7). The number of linear regions is adopted as the complexity measure, following [@NIPS2014_fa6f2a46] and [@raghuExpressivePowerDeep2017].

## Critical assessment

This is a foundational and much-cited paper, and the core equivalence (Thm. 5.2/5.4) is clean and convincing: the recursion genuinely produces the network's output, and the tropical dictionary is set up carefully enough to be reused. Its main weakness is that the headline quantitative result is explicitly not new — Thm. 6.3 reproduces the bound of [@raghuExpressivePowerDeep2017] under stronger assumptions, so the depth-separation claim "a deeper network is exponentially more expressive" (abstract) rests on the pre-existing lower bound of [@NIPS2014_fa6f2a46] rather than on anything proved here. The claim of being "for the first time" an explicit connection between ReLU networks and tropical geometry is correct as far as the equivalence of *families* goes, but the closely related equivalence with continuous piecewise-linear integer-coefficient functions (Prop. 5.5) partially overlaps the earlier real-weight result the authors themselves acknowledge (Arora et al., 2018, Thm. 2.1).

Several technical points deserve scrutiny. The integer-weight restriction is argued to be lossless, but the "clear denominators" argument changes the network's scale and shifts which tropical rational function is represented, so the exact-family equivalence is genuinely only for integer weights; the real-weight case is handled only by the weaker signomial statement (Prop. 5.6), which the authors then abandon. The $\lceil \log_2 r \rceil + 2$ depth bound in Thm. 5.4(ii) is a realizability bound and does not address whether the resulting network is trainable or the representation minimal. There are no experiments, no code, and no numerical validation of any bound — expected for a theory paper, but it means the contributions are entirely structural. The paper also leans on the reader accepting "linear regions" as the right complexity proxy, a choice that later work (e.g. [@haninDeepReLUNetworks2019]) showed can be misleading. Finally, Eq. 2's extraction in the source PDF is ambiguous in its operators and exponent signs; I reconstructed it to be consistent with the unambiguous non-tropical Prop. 5.1, but a reader checking against the original should verify this transcription.

## Connections within the collection

The paper is the root of the tropical-geometry thread in this collection. It builds directly on the linear-region counting framework of [@NIPS2014_fa6f2a46] (whose lower bound $(n/d)^{d(L-1)} n^d$ it pairs with its upper bound) and on [@raghuExpressivePowerDeep2017], whose bound it reproduces; both are used as components, not passing mentions. Its main descendant in the collection is [@alfarraDecisionBoundariesNeural2023], which takes the tropical view of decision boundaries and builds practical decision-boundary algorithms on it.

It sits alongside the piecewise-linear/polyhedral-decomposition line of work: [@pascanuNumberResponseRegions2013] and [@serraBoundingCountingLinear2018] count response/linear regions by other means; [@haninComplexityLinearRegions2019] and [@haninDeepReLUNetworks2019] refine and partly challenge the region-counting picture; [@goujonNumberRegionsPiecewise2023] and [@brandenburgDecompositionPolyhedraPiecewise2024] continue the combinatorial analysis of regions and decompositions; [@masdenAlgorithmicDeterminationCombinatorial2022] gives combinatorial algorithms for the same linear-region structure; and [@carlssonPREIMAGERECTIFIERNETWORK2017], [@liuReLUNeuralNetworks2023], [@grigsbyHiddenSymmetriesReLU] study the underlying polyhedral complexes and their symmetries. The zonotope/hyperplane-arrangement link points to the combinatorics gathered in [@pakLecturesDiscretePolyhedral], and to the arrangement face-count work of Zaslavsky (in the collection but without a readable attachment). The spline viewpoint of [@balestrieroSplineTheoryDeep2018] and the rational-network expressiveness work of [@averkovEXPRESSIVENESSRATIONALRELU2025] address adjacent questions of how piecewise-polynomial/rational structure shapes network expressiveness. The geometric complexity theme also recurs in [@keupOrigamiDimensionsHow2022] and the space-folding analyses of [@lewandowskiSpaceFoldsReLU2024] and [@lewandowskiExploitingSpaceFolding2026].
