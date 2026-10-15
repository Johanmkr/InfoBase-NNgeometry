---
citekey: serraBoundingCountingLinear2018
title: "Bounding and Counting Linear Regions of Deep Neural Networks"
authors: [Thiago Serra, Christian Tjandraatmadja, Srikumar Ramalingam]
year: 2018
venue: "Proceedings of the 35th International Conference on Machine Learning, PMLR 80"
doi: "10.48550/arXiv.1711.02114"
arxiv: "1711.02114"
url: null
kind: theory
source_type: pdf
source_sha256: f7ff450d275ad0cdfe3ae9c9bdf16dc930a83a1c3472a564c6b9c7332e9869f4
pages: 19
ingested: 2026-10-15
tags: [linear-regions, relu-networks, mixed-integer-programming, expressiveness, piecewise-linear]
concepts_introduced:
  - bottleneck effect
  - exact counting of linear regions
  - mixed-integer linear formulation of rectifier networks
  - upper bound on linear regions
  - upper bound for multi-layer maxout networks
  - exact maximal number of regions for one-dimensional input
concepts_used:
  - linear region
  - activation pattern
  - rectified linear unit
  - piecewise linear function
  - hyperplane arrangement
  - maxout unit
  - mixed-integer linear programming
builds_on: [NIPS2014_fa6f2a46, pascanuNumberResponseRegions2013, raghuExpressivePowerDeep2017, zaslavskyFacingArrangementsFacecount1975]
datasets: [MNIST]
---

# Bounding and Counting Linear Regions of Deep Neural Networks

**Serra, Tjandraatmadja, Ramalingam (2018)**, *ICML, PMLR 80*. [arXiv:1711.02114](https://arxiv.org/abs/1711.02114)

!!! abstract "TL;DR"
    The paper tightens the upper and lower bounds on how many linear regions a
    rectifier network can realize, by tracking the *dimension of the image* of
    each region through the layers; the new upper bound is exact when the input
    dimension is one, and it yields a first upper bound for multi-layer maxout
    networks. It also gives a mixed-integer linear formulation of a trained
    network that allows the *exact* number of regions to be counted, and uses it
    on small MNIST networks.

## Problem and motivation

A rectifier network computes a piecewise linear (PWL) function
$F : \R^{n_0} \to \R^{m}$, and its expressiveness is often measured by how many
linear regions it can carve the input space into: more regions are taken to mean
more representational power (§1). Prior work had bounded this count:
Montúfar et al. (2014) gave an upper bound of $2^N$ for $N$ units and a lower
bound, Raghu et al. (2017) an asymptotic upper bound $O(n^{L n_0})$ for equal
widths $n$ and constant input dimension, Montúfar (2017) the tighter
$\prod_{l=1}^{L}\sum_{j=0}^{d_l}\binom{n_l}{j}$ with $d_l=\min\{n_0,\dots,n_l\}$,
and Arora et al. (2018) a stronger lower bound (§3, p. 3). No bound was known
for multi-layer maxout networks, and no method existed to compute the true count
for a given trained network — so the tightness of any of these bounds was
untestable (§1, §5). The setting is the usual feedforward network with ReLU
activations, a single linear output layer, and regions defined by activation
patterns (Def. 1, §2).

## Method

The paper's engine is a dimensional argument. For a region $S$ up to layer $l$
one considers the dimension of the image $\dim(h^l(S))$; because the network is
linear on $S$, this is the rank of the effective weight matrix, and it can only
decrease as further layers partition $S$ (§3.1). Lemma 4 sharpens Zaslavsky's
arrangement bound to rank-deficient hyperplanes, Lemma 5 turns it into a bound
on how many subregions layer $l$ can add inside $S$, and Lemma 6 bounds
$\dim(h^l(S))$ recursively. Chaining these gives a recurrence $R(l,d)$ for the
maximal number of regions obtainable from a region of image dimension $d$ using
layers $l,\dots,L$, whose solution is Theorem 1. The same technique, applied to
the hyperplanes $w_j x + b_j = w_{j'} x + b_{j'}$ that extend each maxout
boundary segment, yields the maxout bound (Thm. 10). For counting, each ReLU is
encoded with a binary activation variable and a big-$M$ disjunction, and the
region count is obtained by enumerating integer solutions of the resulting
mixed-integer linear program with the one-tree approach (§5, Appendix I).

## Key equations and definitions

Network activations, with the output layer linear (§2, p. 2):

$$
h^1 = \max\{0, W^1 x + b^1\}, \qquad h^l = \max\{0, W^l h^{l-1} + b^l\}, \qquad y = W^{L+1} h^L \qquad \text{(§2, p. 2)}
$$

Zaslavsky's region count for an arrangement of $n$ hyperplanes in $\R^d$, the
base case the whole paper builds on (§2, p. 3; [@zaslavskyFacingArrangementsFacecount1975]):

$$
(\text{regions}) \le \sum_{s=0}^{d} \binom{n}{s} \qquad \text{(§2, p. 3)}
$$

The main upper bound (Thm. 1, p. 4):

$$
(\text{maximal number of regions}) \le \sum_{(j_1,\dots,j_L)\in J}\ \prod_{l=1}^{L}\binom{n_l}{j_l}, \qquad J = \Bigl\{(j_1,\dots,j_L)\in\Z^L : 0\le j_l\le \min\{n_0,\, n_1-j_1,\dots,n_{l-1}-j_{l-1},\, n_l\}\ \forall l\Bigr\} \qquad \text{(Thm. 1, p. 4)}
$$

The index set forces each $j_l$ below the widths of all preceding layers after
subtracting the dimensions already "used up", and this is exactly what produces
the bottleneck effect.

Two ingredients, the second bounding how the image dimension shrinks
(Lem. 5, 6, pp. 5–6):

$$
(\text{subregions of } S) \le \sum_{j=0}^{\min\{n_l,\ \dim(h^{l-1}(S))\}}\binom{n_l}{j}, \qquad \dim(h^l(S)) \le \min\{|S^l|,\ \dim(h^{l-1}(S^0))\} \qquad \text{(Lem. 5, 6, pp. 5–6)}
$$

The exact one-dimensional result (Thm. 7, p. 6):

$$
\max = \prod_{l=1}^{L}(n_l + 1) \qquad \text{(Thm. 7, p. 6, for } n_0=1,\ n_l\ge 3\text{)}
$$

The improved lower bounds (Thm. 8, 9, p. 6), the first in terms of per-layer
widths, the second in terms of total size:

$$
\prod_{l=1}^{L-1}\Bigl(\bigl\lfloor n_l/n_0\bigr\rfloor + 1\Bigr)^{n_0}\sum_{j=0}^{n_0}\binom{n_L}{j} \qquad\text{and}\qquad 2^{\sum_{j=0}^{n_0-1}\binom{m-1}{j}}\,(w+1)^{L-1} \qquad \text{(Thm. 8, 9, p. 6)}
$$

The maxout upper bound (Thm. 10, p. 7), a product over layers of arrangements of
the $\tfrac{k(k-1)}{2}n_l$ pairwise-difference hyperplanes:

$$
\prod_{l=1}^{L}\sum_{j=0}^{d_l}\binom{\tfrac{k(k-1)}{2}\,n_l}{j}, \qquad d_l = \min\{n_0,n_1,\dots,n_l\} \qquad \text{(Thm. 10, p. 7)}
$$

Mixed-integer encoding of one ReLU, with a complementary variable
$\bar h^l_i = \max\{0,-W^l_i h^{l-1}-b^l_i\}$ and a binary $z^l_i$
(§5, p. 7, constraints (1)–(6)):

$$
W^l_i h^{l-1} + b^l_i = h^l_i - \bar h^l_i,\quad h^l_i \le M z^l_i,\quad \bar h^l_i \le M(1-z^l_i),\quad h^l_i \ge 0,\quad \bar h^l_i \ge 0,\quad z^l_i \in \{0,1\}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $n_0$ | input dimension; $n_l$ width (number of neurons) of hidden layer $l$; $m$ output dimension |
| $L$ | number of hidden layers; $W^l$ the $n_l \times n_{l-1}$ weight matrix, $b^l$ its bias |
| $h^l$ | vector of activations of layer $l$; $h^l_i$ its $i$-th entry |
| $S^l$ | activation set of layer $l$; $S=(S^1,\dots,S^L)$ the activation pattern, also called a region |
| $F$ | the PWL function $\R^{n_0}\to\R^{m}$ realized by the network |
| $\dim(h^l(S))$ | dimension of the image of region $S$ under the map up to layer $l$ |
| $R(n_0,n_1,\dots,n_L)$ | exact maximal number of regions; $B(\cdot)$ its upper bound from Thm. 1 |
| $d_l$ | $\min\{n_0,n_1,\dots,n_l\}$ |
| $k$ | rank (number of linear pieces) of a maxout unit |
| $z^l_i$ | binary activation variable of neuron $i$ in layer $l$; $M$ a big-$M$ constant |
| $X$ | the (bounded, polyhedral) input domain of the counting problem |

## Main results

The upper bound of Theorem 1 is strictly stronger than Montúfar (2017): relaxing
$n_l-j_l$ to $n_l$ recovers that bound, and for constant $n_0$ and equal widths
it matches the best known asymptotic $O(n^{Ln_0})$ of Raghu et al. (2017)
(§3.1, p. 4). Two consequences are highlighted. *Bottleneck effect:* subtracting a
neuron from an earlier layer lowers the bound at least as much as subtracting it
from a later one; for two layers with $n_0 > \max\{n_1,n_2\}$ moving a neuron from
the first to the second layer strictly decreases the bound (Prop. 2, p. 4,
Prop. 13/14 and App. A). *Deep vs shallow:* Corollary 3 shows that for
$L\ge 2$, $n\ge 1$, $n_0 \ge Ln$,
$R(n_0,n,\dots,n) < R(n_0,Ln)$ and the ratio of the two tends to $0$ as
$L\to\infty$ — a deep network is *worse* than a shallow one of the same total
width once the input dimension exceeds the neuron count (§3.1, p. 4, App. A).
When the input dimension is one, the construction of Theorem 7 attains the
Theorem 1 bound exactly, improving the $n^L$ replication of Montúfar et al.
(2014) to $(n+1)^L$ by keeping every region's image one-dimensional
(§3.2, p. 6, App. D). The lower bounds improve those of Montúfar et al. (2014)
and Arora et al. (2018) by one region per replicated factor ($\lfloor n_l/n_0\rfloor$
becomes $\lfloor n_l/n_0\rfloor+1$, and $w$ becomes $w+1$), at the cost of the
stricter condition $n_l \ge 3n_0$ (Thm. 8, 9, §3.3, p. 6). Asymptotically a
network of width $n$ with $n_0 \ge n/3$ realizes $\Omega(2^{\frac{2}{3}Ln})$
regions (App. B, p. 12).

For counting, Theorem 11 states that the constraints (1)–(6) reproduce the
network's input–output map whenever $|W^l_i h^{l-1}+b^l_i|\le M$ for all
reachable $h^{l-1}$; Corollary 19 gives mixed-integer representability for
polytopal input, and Corollary 21 that regions remain countable for unrestricted
$X$ if $M$ is large enough (§5, App. I–J). On MNIST, ten networks are trained for
each configuration of two hidden layers summing to 22 neurons (plus a 10-unit
output), for 20 epochs, and all regions are counted within $0 \le x \le 1$
(§6, p. 8). The paper reports that the Theorem 1 bound is the tightest of the
three compared (Fig. 1, p. 1), and that across all 21 configurations the median
counting time per configuration lies between $1.1$ s and $9.9$ s (Tab. 1, p. 19).
The actual average/min/max region counts are given only graphically in Fig. 1
(no numeric table is printed), so they are not reproduced here. Figures 6 and 7
plot the count against training cross-entropy and test misclassification rate;
the authors report "initial evidence" that accuracy correlates with the region
count for configurations where no layer is too small, while a very thin layer
(1–2 neurons) can give high errors despite many regions (§6–§7, pp. 8–9).

## Limitations stated by the authors

The counting formulation assumes a bounded, polyhedral input (natural when each
$x_i$ has known bounds) and a sufficiently large $M$ (§5, p. 7, App. H). The
experiment is limited to small two-hidden-layer networks with 22 neurons because
of the cost of exact counting. The correlation between region count and accuracy
is described only as an initial observation, not a demonstrated law (§7). The
bottleneck effect is proved for the *bound*, not for the true maximum; the
authors say so explicitly and only note that trained networks behave similarly
(§3.1, p. 4).

## Critical assessment

My reading. The dimensional argument is the conceptual contribution: it replaces
an enumeration of activation patterns by a rank bookkeeping that is monotone
through the layers, and the same lemma then gives the maxout bound almost for
free — a genuinely economical proof. The counter-intuitive Corollary 3 is a
genuine correction to the "depth always wins" picture, though it speaks about
$\max$ over all weight configurations at fixed total width, not about trained
networks. Two weaknesses stand out. First, the counting experiment is
necessarily tiny: 22 neurons, two hidden layers, so no claim about realistic
architectures is tested, and the accuracy–region correlation rests on plots with
no fitted statistic or confidence statement. Second, the maxout bound is the
first of its kind but crude — the rank relaxation throws away the very
dimension information that made the ReLU bound tight, only partially restored by
the $\tfrac{k(k-1)}{2}$ factor. The construction behind Theorem 7 (App. D) is
fiddly and the derivation of the weights and biases from the breakpoints is
sketched ("left to show that there exists a solution"), so reproducibility of
that corner needs the reader's own algebra. The big-$M$ values needed in
practice are only discussed in the appendix (App. H) and are known to be a
numerical weak point of such formulations.

## Connections within the collection

Directly extends the linear-region bounds of Montúfar et al. (2014)
[@NIPS2014_fa6f2a46] and the region-counting framing of Pascanu et al. (2013)
[@pascanuNumberResponseRegions2013], and improves the asymptotic upper bound of
Raghu et al. (2017) [@raghuExpressivePowerDeep2017]. The whole calculation rests
on Zaslavsky's hyperplane-arrangement face-count formula
[@zaslavskyFacingArrangementsFacecount1975], whose summary is unavailable in this
collection (no readable attachment). The later average-case results of Hanin and
Rolnick [@haninComplexityLinearRegions2019; @haninDeepReLUNetworks2019] are the
natural complement: they show trained networks use far fewer regions than these
worst-case maxima, a tension the present experiments on 22-neuron networks begin
to probe. Goujon et al. [@goujonNumberRegionsPiecewise2023] continue the
worst-case bound programme, and Averkov et al.
[@averkovEXPRESSIVENESSRATIONALRELU2025] study expressiveness for bounded depth.
The exact-counting idea is close in spirit to the polyhedral enumeration of
Vincent and Schwager [@vincentReachablePolyhedralMarching2021] and to
Masden's combinatorial determination of region structure
[@masdenAlgorithmicDeterminationCombinatorial2022] and the exact visualization of
Humayun et al. [@humayunSplineCamExactVisualization2024]. The same PWL geometry is
expressed differently by the spline picture of Balestriero and Baraniuk
[@balestrieroSplineTheoryDeep2018], by tropical geometry
[@zhangTropicalGeometryDeep; @alfarraDecisionBoundariesNeural2023], and by
polyhedral decomposition [@brandenburgDecompositionPolyhedraPiecewise2024;
@liuReLUNeuralNetworks2023]; these are alternative formalisms for the objects this
paper counts rather than results it uses. Possibly related: the preimage/complexity
view of ReLU activations [@carlssonPREIMAGERECTIFIERNETWORK2017] and the
transition-graph representation [@dhayalkarDiscreteFunctionalGeometry2025].
