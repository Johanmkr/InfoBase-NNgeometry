---
citekey: grigsbyHiddenSymmetriesReLU
title: "Hidden Symmetries of ReLU Networks"
authors: [J. Elisenda Grigsby, Kathryn Lindsey, David Rolnick]
year: 2023
venue: Proceedings of the 40th International Conference on Machine Learning, PMLR 202
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: 3db6c5171785c941dcefa822d62a6acc7818a85f5fdb7c796d3ab347f7144353
pages: 27
ingested: 2026-10-15
tags: [relu-networks, parameter-symmetry, polyhedral-geometry, hyperplane-arrangements, functional-dimension]
concepts_introduced:
  - hidden symmetry
  - positive-axis hyperplane
  - transverse pairwise intersection condition
  - stably unactivated neuron
  - never coactive neurons
concepts_used:
  - activation pattern
  - bent hyperplane
  - supertransversal network
  - functional dimension
  - polyhedral complex
  - permutation symmetry
  - positive scaling symmetry
  - linear regions assumption
builds_on: [phuongFunctionalVsParametric2019, haninDeepReLUNetworks2019, masdenAlgorithmicDeterminationCombinatorial2022]
datasets: []
---

# Hidden Symmetries of ReLU Networks

**Grigsby, Lindsey, Rolnick (2023)**, *ICML, PMLR 202*. Equal contribution (Grigsby, Lindsey).

!!! abstract "TL;DR"
    Parameter space is an imperfect proxy for the function class of a ReLU architecture: beyond
    neuron permutation and positive scaling, some parameters admit *hidden* symmetries. The paper
    proves that whenever no hidden layer is narrower than the input, a positive-measure set of
    parameters has no hidden symmetries at all, by explicitly constructing parameters whose bent
    hyperplanes pairwise intersect (TPIC) and satisfy the Linear Regions Assumption. Experiments
    that probe the functional dimension at initialization show that the probability of *no* hidden
    symmetries falls with depth and rises with width and input dimension.

## Problem and motivation

For a fixed architecture $(n_0,\dots,n_d)$ (input dimension $n_0$, hidden widths $n_1,\dots,n_{d-1}$,
output $n_d$), the parameter space $\Omega=\R^D$ is used as a proxy for the represented function
class, but two parameters can determine the same function (§1, p. 1). Two families of symmetries are
universally known:

- **Permutation (P):** reordering the neurons of a hidden layer together with their incoming and
  outgoing weights and biases.
- **Scaling (S):** for a single neuron, multiplying incoming weights and bias by $c>0$ and dividing
  outgoing weights by $c$.

The question is when these are the *only* symmetries — i.e. when the parameters admit no **hidden
symmetries** (Def. F.4). Prior work leaves the question open for most architectures: reverse
engineering succeeds under a geometric condition on the bent hyperplane arrangement, which
immediately covers all depth-2 networks and a positive-measure subset of depth-3 networks
(Rolnick & Kording 2020, outside the collection), and the complementary result is known for
*non-widening* architectures $n_0\ge n_1\ge\dots\ge n_d$ [@phuongFunctionalVsParametric2019].
The present paper closes the complementary class: architectures whose hidden layers are all at least
as wide as the input (§2, p. 2). It also asks how *often* hidden symmetries occur, using the
functional dimension of a parameter as a computable proxy (§6, p. 6).

## Method

**Theory (§4, §5, App. E).** The strategy is constructive, via the transverse pairwise intersection
condition (TPIC, Def. B.9): a generic, supertransversal parameter satisfies TPIC if every pair of
bent hyperplanes in adjacent layers has non-empty transverse intersection. A sharpened version of
Rolnick & Kording's Theorem 2 (Lemma D.15) says that any parameter satisfying TPIC and the Linear
Regions Assumption (LRA) on a neighborhood of the pairwise intersections is recoverable from
$F_\theta$ up to permutation and positive rescaling — hence has no hidden symmetries. TPIC is an
open condition (Cor. D.13), as is the cell realizing a given ternary activation pattern (Prop. D.12);
so it suffices to exhibit a *single* such parameter, and the conclusion propagates to an open
positive-measure neighborhood minus a measure-zero algebraic set (Lemmas D.11, D.14).

The construction is by induction on depth. The base case $d=2$ uses **positive-axis hyperplanes**
$H=\{\vec x\mid \vec w\cdot\vec x+b=0\}$ with all $w_i>0$, $b<0$: these meet every ray of the
non-negative orthant $\mathcal O_{\ge0}$, and since $F^1$ maps almost every unbounded ray into
$\mathcal O_{\ge0}$ (Lemma C.4), a sufficiently high-bias positive-axis hyperplane meets the image
of every first-layer hyperplane. Lemma 4.2 pulls such intersections back through the layer map. For
the inductive step one keeps the previous bent hyperplanes and their intersections, and chooses the
final layer so that its new bent hyperplanes intersect those of the penultimate layer; pulling back
requires the activation region images to be full-dimensional, supplied by Prop. 4.3 and Cor. 4.4.

**Mechanisms (§5).** The paper lists four ways hidden symmetries arise:
(i) a *stably unactivated* neuron (Def. A.2), whose positive half-space misses the image of the
domain; (ii) two neurons in adjacent layers that are *never coactive* (Def. A.5), making the weight
between them free; (iii) a later ReLU collapsing linear regions built by earlier layers (violating
LRA); (iv) the relevant image lying in an affine subspace of positive codimension, giving a
1-parameter family of hyperplanes that yield the same map (Prop. 5.1).

**Experiments (§6).** Hidden symmetries are hard to detect directly, so the paper approximates the
functional dimension $\dim_{\mathrm{fun}}(\theta)$ (Def. F.2) by the rank of the Jacobian of the
unrolled evaluations at $m$ random points. $\dim_{\mathrm{fun}}$ equals the theoretical upper bound
exactly when the parameters have no hidden symmetries, since the bound is obtained by quotienting by
(P) and (S) (Lemma F.6). Weights are i.i.d. $\mathcal N(0,2/\text{fan-in})$ (He init), biases i.i.d.
normal with variance $0.01$; $m=100\times$ the maximum possible $\dim_{\mathrm{fun}}$, and each
architecture uses 5000 parameter draws.

## Key equations and definitions

Layer maps of the network (§3, Eqs. 1–2, p. 2):

$$
F^i(x) := \begin{cases} \sigma(W^i x + b^i) & 1\le i<d\\ W^i x & i=d,\end{cases}
\qquad F_\theta = F^d\circ\dots\circ F^1.
$$

The pre-activation map of the $i$th neuron of layer $\ell$ and its bent hyperplane (§3, Eq. 3, p. 3):

$$
z^{(\ell),i}(x) = \pi_i\!\left(W^\ell\!\left(F^{(\ell-1)}(x)\right) + b^\ell\right),
\qquad \hat H^\ell_i := (z^{(\ell),i})^{-1}\{0\}.
$$

Concrete parameters for a proof are positive-axis hyperplanes (§4, p. 4):

$$
H := \{\vec x \in \R^n \mid \vec w\cdot\vec x + b = 0\}, \qquad w_i>0\ \forall i,\quad b<0.
$$

Image dimension of an activation region (Prop. 4.3, p. 4) — for almost all $\theta$, an activation
region $C$ with pattern $s_C=(s^1_C,\dots,s^d_C)$ maps to a polyhedral set of dimension
$\min\{\dim(s^1_C),\dots,\dim(s^d_C),k\}$, where $k=n_0$ and $\dim(s^\ell_C)$ counts the $+1$'s.
Corollary: if every $\dim(s^\ell_C)=k$, then $F_\theta$ is a homeomorphism on the interior of $C$.

Functional dimension and its estimator (§6, Eq. 4, p. 7; Def. F.2, p. 23):

$$
\dim_{\mathrm{fun}}(\theta) := \sup\{\operatorname{rank}(J_{E_Z}(\theta)) \mid Z\subset\R^{n_0}\ \text{finite, parametrically smooth}\},
$$

$$
\dim_{\mathrm{fun}}(\theta) \approx \operatorname{rank}\!\begin{bmatrix}\nabla_\theta F_\theta(z_1)\\ \vdots\\ \nabla_\theta F_\theta(z_m)\end{bmatrix}.
$$

No hidden symmetries (Def. F.4, p. 24): $\theta$ has none iff the semigroup generated by (P) and
(S) acts transitively on its fiber $\{\tilde\theta \mid F_{\tilde\theta}=F_\theta\}$.

| Symbol | Meaning (in this paper) |
|---|---|
| $(n_0,\dots,n_d)$ | architecture: input dim., hidden widths, output dim. |
| $\Omega=\R^D$, $D=-\n_d+\sum_i n_i(n_{i-1}+1)$ | parameter space and its dimension |
| $F_\theta$ | network function determined by $\theta=(W^1,b^1,\dots,W^d)$ |
| $F^{(\ell)}=F^\ell\circ\dots\circ F^1$ | composed layer maps from the domain |
| $\hat H^\ell_i$ | bent hyperplane of the $i$th neuron of layer $\ell$ |
| $\mathcal O_{\ge0}$ | non-negative orthant |
| $\dim_{\mathrm{fun}}(\theta)$ | functional dimension of the parameter $\theta$ |

## Main results

**Theorem 4.1 (p. 3).** For any feedforward ReLU architecture with $n_0 \le n_\ell$ for all $\ell$,
a positive-measure subset of $\Omega$ has no hidden symmetries. The proof (App. E, Thm. E.1) is the
constructive induction sketched above; it establishes that the constructed neighborhood contains no
hidden symmetries away from a measure-zero algebraic set. This is the exact complement of the
non-widening case [@phuongFunctionalVsParametric2019]; the boundary case that remains open after
both results is, for instance, $(n_0,n_1,n_2,n_3,n_4)$ with $n_0<n_1$ and $n_2<n_0$ (§2, p. 2).

**Proposition 5.1 (p. 5).** If the part of the image where a $(k+1)$-th-layer neuron is non-negative
lies in an affine hyperplane $S$ in general position with $H$, and no unbounded ray of the image has
slope orthogonal to $\vec n_H$, then there is a 1-parameter family of distinct hyperplanes $H_t$
with $\eta\circ F^k\circ\dots\circ F^1 = f_{H_t}\circ F^k\circ\dots\circ F^1$: the neuron's
hyperplane is not uniquely determined. This is mechanism (iv).

**Experiments (Fig. 6, p. 7).** For depth $d=4$ and widths $n_0=n_1=\dots=n_{d-1}\in\{5,10,15\}$,
the fractions of networks with the maximum possible functional dimension (black dots) are
$25\%$, $48\%$, $66\%$ respectively; for depth $6$ they are $1\%$, $3\%$, $5\%$ (5000 draws each).
So at initialization the probability of no hidden symmetries **decreases with depth** and **increases
with width and input dimension** (Fig. 6, p. 7). When only the hidden widths vary and the input
dimension is fixed at 5, the probability instead *slightly decreases* with width (Fig. 7, pp. 8–9).
The distributions are unimodal when the probability of full dimension is low, but strongly multimodal
when it is high, with the peaks apparently spaced by the network width (Fig. 6 inset, p. 7); the
authors note that neither mechanism (i) nor (ii) would give a spacing of one width, so mechanisms
(iii) or (iv) may be at play, and say the phenomenon needs further study (§6, p. 8). Appendix G
shows the conclusions are robust to the sample size $m$: with $m$ set to twice (instead of 100
times) the maximum functional dimension, and 20000 draws, the same patterns appear (Figs. 10–11,
App. G, p. 26).

## Limitations stated by the authors

- The theorem is only for hidden layers at least as wide as the input; the general architecture
  remains unresolved, and the paper gives an explicit open example (§2, p. 2).
- The mechanism list is *partial*: it is "a partial list", not an exhaustive classification of
  hidden symmetries (§5, p. 5).
- The experiments approximate $\dim_{\mathrm{fun}}$, which is a lower bound (attained values equal
  the upper bound are guaranteed accurate, but lower values are not certified) (§6, p. 8).
- The explanations offered for the depth/width trends are heuristic and "further study is required";
  the authors flag the counteracting effect that increasing width enlarges the support of the
  distribution (§6, p. 8).
- The spacing of the multimodal peaks is only observed empirically and does not match the two
  mechanisms that would have simple spacing predictions (§6, p. 8).

## Critical assessment

The theoretical contribution is solid and cleanly placed: Grigsby–Lindsey–Rolnick's TPIC-plus-LRA
route to "no hidden symmetries" is the same one used earlier, and the genuine work is the
construction that realizes TPIC for all wide-enough architectures. The complementarity with the
non-widening case [@phuongFunctionalVsParametric2019] is a nice structural fact, and the authors are
candid that the two results together still do not cover all architectures. The construction is
technical and heavily appendix-dependent (App. C–E), so a reader wanting to re-derive Thm. 4.1 must
follow the whole chain of lemmas — the main text's proof sketch is honest about this. The
"positive measure" conclusion is the substantive part; existence of a single such parameter would be
weak, and the open-ness lemmas (D.13, D.14) are what upgrade it.

The empirical part is weaker evidence than it first appears, as the authors themselves acknowledge:
$\dim_{\mathrm{fun}}$ is approximated, and a low estimate can reflect an insufficient number of
sample points rather than an actual hidden symmetry. The reported percentages therefore bound
"no hidden symmetries" from above as probabilities, not exactly. There is no code or data statement
that I could find in the PDF, which makes reproduction of the experiments harder; the setup is
simple enough (He init, unit-normal inputs, Jacobian rank) to be reproducible in principle. The
mechanism analysis is suggestive rather than conclusive: four mechanisms are listed but not
quantified, and the observation that mode spacing equals the network width is left unexplained. The
link to generalization in the conclusion ("lower functional dimension should induce local flatness
of the loss landscape", §7, p. 8) is a conjecture, explicitly labelled as such, and is not tested.

Overall: a convincing and useful structural result about the geometry of parameter space, paired with
a suggestive but preliminary empirical study.

## Connections within the collection

The paper sits squarely in the ReLU-geometry literature of this collection. Its central objects —
bent hyperplanes and ternary activation patterns — are the vocabulary of
[@haninDeepReLUNetworks2019], whose Lemma 8 supplies the polynomial formula for
$F_k(\theta,x)$ used in Lemma D.9, and whose counting of activation regions underlies
[@haninComplexityLinearRegions2019], [@pascanuNumberResponseRegions2013] and
[@serraBoundingCountingLinear2018]. Supertransversality, the canonical polyhedral complex and the
injectivity of the ternary labelling on cells are taken from
[@masdenAlgorithmicDeterminationCombinatorial2022], and the same polyhedral-complex machinery is
used by [@berzinsPolyhedralComplexExtraction2023] to extract the complex and by
[@goujonNumberRegionsPiecewise2023], [@brandenburgDecompositionPolyhedraPiecewise2024] to count
regions; the mechanism (iii) "collapse" of linear regions is exactly the failure of the Linear
Regions Assumption studied in that line.

It directly addresses the same question as [@phuongFunctionalVsParametric2019] (functional versus
parametric equivalence), providing the complementary architecture class. The four mechanisms connect
to the rest of the collection: mechanism (ii), never-coactive neurons, is the local picture behind
[@sattelbergLocallyLinearAttributes2023] and the spline/LRA-based descriptions of
[@balestrieroSplineTheoryDeep2018] and [@humayunSplineCamExactVisualization2024]; mechanism (i),
stably unactivated neurons, and the geodesic/Finsler view of latent geometry in
[@pouplinIdentifyingLatentDistances2023] are only loosely related. The "space fold" picture of
[@lewandowskiSpaceFoldsReLU2024] and the transition-graph view of [@dhayalkarDiscreteFunctionalGeometry2025]
study the same folded domain and could potentially refine the naive mode-spacing explanation.
Possibly related: [@carlssonPREIMAGERECTIFIERNETWORK2017] and [@liuReLUNeuralNetworks2023] treat the
preimage/combinatorial structure of the same arrangement. The functional-dimension work of
Grigsby et al. (2022b) that this paper relies on and cites throughout is *outside* the collection
and therefore has no citekey here.
