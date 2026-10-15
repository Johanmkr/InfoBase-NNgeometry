---
citekey: pascanuNumberResponseRegions2013
title: "On the Number of Response Regions of Deep Feed Forward Networks with Piece-Wise Linear Activations"
authors: [Razvan Pascanu, Guido Montúfar, Yoshua Bengio]
year: 2013
venue: arXiv preprint
doi: null
arxiv: "1312.6098"
url: null
kind: theory
source_type: pdf
source_sha256: 27b1ac7a687606e570b42f8bd579eb144c69da59e403de3475d1d51e6f35d31f
pages: 17
ingested: 2026-10-15
tags: [relu-networks, linear-regions, hyperplane-arrangements, expressivity, piecewise-linear]
concepts_introduced:
  - response region
  - region of linearity
  - region replication across layers
  - absolute-value rectifier construction
  - deep-shallow separation by region counting
concepts_used:
  - rectified linear unit
  - piecewise linear function
  - hyperplane arrangement
  - general position
  - Zaslavsky's theorem
  - multilayer perceptron
  - affine map
builds_on: [zaslavskyFacingArrangementsFacecount1975]
datasets: []
---

# On the Number of Response Regions of Deep Feed Forward Networks with Piece-Wise Linear Activations

**Pascanu, Montúfar, Bengio (2013)**, *arXiv preprint*. [arXiv:1312.6098](https://arxiv.org/abs/1312.6098)

!!! abstract "TL;DR"
    The paper counts the number of regions on which a rectifier MLP is exactly linear — its *response regions* — as a measure of expressivity, and compares deep against shallow models at equal size. Using combinatorial geometry (hyperplane arrangements) it shows that a shallow net with $kn$ units has $O(k^{n_0} n^{n_0})$ regions, while a $k$-layer net of width $n$ has $\Theta((\lfloor n/n_0\rfloor)^{k-1} n^{n_0})$; the deep count per parameter grows exponentially in depth and polynomially faster in width.

## Problem and motivation

Deep networks are widely believed to represent some functions far more efficiently than shallow ones, but the authors argue that theoretical evidence for this is thin: earlier results concern units quite unlike the ones used in practice (logic gates, threshold units, polynomials) (§1, p. 2). They propose to compare architectures *within* the family of piecewise linear functions, by counting the number of *regions of linearity* the model defines on the input space — "the number of pieces available to the model in order to approximate some arbitrary nonlinear function" (§1, p. 2). Rectifier units and other piecewise linear activations (e.g. maxout) are the target, since a composition of piecewise linear functions is itself piecewise linear.

The count is honestly framed as an *upper bound* on flexibility: the linear pieces are not independent (parameters are shared across regions), so a large region count need not translate into a large reachable function class; the authors in fact read the correlation between regions as the source of a deep model's ability to generalise to structured functions (§1, p. 2). Setting: rectifier MLPs with linear output units, $n_0$ inputs, and hidden layers of widths $n_1,\dots,n_k$; the quantity of interest is $R(n)$, the *maximal* number of response regions over all functions in the class $F_n$ realisable with that profile (§2, p. 3).

## Method

The analysis is a study of hyperplane arrangements, built up in three steps.

1. **Preliminaries (§2, pp. 3–4).** Lemma 1: the response regions of a width-$k$ rectifier layer are the non-empty intersections $R_{j_1}\cap\cdots\cap R_{j_k}$ of the regions of its scalar coordinates, so a vector-valued rectifier layer is handled coordinate-wise. Lemma 2: the number of (linear) output units does not affect the maximal region count. Lemma 3: a linear layer followed by a rectifier layer is absorbed into a single rectifier layer, since affine maps compose to affine maps.

2. **One hidden layer (§3, pp. 4–6).** Each rectifier unit $i$ is active on one side of the hyperplane $H_i = \{x : W^{(1)}_{i,:}x + b^{(1)}_i = 0\}$ and dead on the other, so the read-out of a single-layer MLP is exactly the number of regions of a hyperplane arrangement $A = \{H_i\}$. For arrangements in *general position* the count is given by the special case of Zaslavsky's theorem [@zaslavskyFacingArrangementsFacecount1975] (Prop. 1).

3. **Multiple hidden layers (§4, pp. 6–13).** The key geometric construction makes a lower layer *replicate* an arrangement built on its intermediate outputs inside every input-space region. Prop. 3 shows any arrangement can be scaled and shifted so that all its regions meet the unit ball; Prop. 4 builds, for any $n_0,n$, an $n_0$-dimensional arrangement of $n$ hyperplanes in which for every consecutive range $a<b$ there is a region $R$ with $R^+ = \{a,a+1,\dots,b\}$ (i.e. exactly units $a$ through $b$ active); Prop. 5 shows that several regions can be mapped by affine maps with a shared bias onto a common set containing the unit ball. Chaining these gives the main Theorem 1. Section 5 gives a concrete, more efficient construction: pairs of rectifiers are summed to compute $|x_i|$, so each layer folds the space about coordinate axes and re-folds the image.

## Key equations and definitions

The rectifier unit and the single-hidden-layer MLP (§3, Eqs. 1–3, pp. 4–5):

$$
\operatorname{rect}(s) = I(s)\,s, \qquad
I(s) = \begin{cases} 1, & s > 0 \\ 0, & \text{otherwise} \end{cases}
$$

$$
f(x) = W^{(\mathrm{out})}\,\operatorname{diag}\!\big(I(W^{(1)}x + b^{(1)})\big)\,(W^{(1)}x + b^{(1)}) + b^{(\mathrm{out})}
$$

The $I(\cdot)$ vector selects the active modes; the extraction of Eq. 3 from the PDF is garbled, and the grouping shown above is a reconstruction (see critical assessment).

A *region of linearity* (or *response region*) of $f : \R^{n_0}\to\R^{n_L}$ is a maximal connected open set $R$ on which $f|_R$ is linear (§2, p. 3); $R(n)$ is the maximum of the region count over $F_n$.

Zaslavsky's theorem for arrangements in general position (§3, Prop. 1, p. 5), and its plane-lines special case (Eq. 4, p. 5):

$$
r(A) = \sum_{s=0}^{n_0} \binom{m}{s}, \qquad b(A) = \binom{m-1}{n_0}, \qquad
r(A_m) = \binom{m}{2} + m + 1 .
$$

Maximal region count of a single hidden layer (Prop. 2, p. 6):

$$
R(n_0;n_1;n_y) = \sum_{j=0}^{n_0} \binom{n_1}{j} .
$$

The absolute-value construction of §5 (Eq. 5, p. 11): four rectifiers $f_1=\max\{0,[1,0]^\top x\}$, $f_2=\max\{0,[-1,0]^\top x\}$, $f_3=\max\{0,[0,1]^\top x\}$, $f_4=\max\{0,[0,-1]^\top x\}$ are combined so that their pair-sums are $|x_1|,|x_2|$:

$$
\begin{bmatrix} g_1(x) \\ g_2(x) \end{bmatrix}
= \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}
\begin{bmatrix} f_1(x) \\ f_2(x) \\ f_3(x) \\ f_4(x) \end{bmatrix}
= \begin{bmatrix} \lvert x_1 \rvert \\ \lvert x_2 \rvert \end{bmatrix}.
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $n_0$ | number of input units (input dimension) |
| $n_l$ | number of units in hidden layer $l$, $l=1,\dots,k$ |
| $k$ | number of hidden layers |
| $n_{\mathrm{out}}$, $n_y$ | number of linear output units |
| $F_n$, $n=(n_0,\dots,n_L)$ | class of functions computed by a rectifier MLP with that layer profile |
| $R(n)$ | maximal number of response regions over $F_n$ |
| $I(s)$ | indicator of $s>0$ |
| $H_i$ | hyperplane $\{x: W_{i,:}x + b_i = 0\}$ of unit $i$ |
| $A$, $r(A)$, $b(A)$ | hyperplane arrangement; number of regions; number of relatively bounded regions |
| $R^+$ | set of indices of hidden units active (non-zero) on region $R$ |
| $S$ | $n_0$-dimensional unit ball |
| $m$ | number of hyperplanes in an arrangement |

## Main results

**Theorem 1 (main, §4, p. 10; proof pp. 10–11).** A model with $n_0$ inputs and $k$ hidden layers of widths $n_1,\dots,n_k$ can divide the input space into

$$
\prod_{i=1}^{k-1} \left\lfloor \frac{n_i}{n_0} \right\rfloor \cdot \sum_{j=0}^{n_0} \binom{n_k}{j}
$$

or possibly more regions. The proof is constructive: the first layer realises the arrangement of Prop. 4, giving $p=\lfloor n_1/n_0\rfloor$ input regions each activating a *disjoint* set of $n_0$ units; a fictitious intermediate linear layer (absorbed by Lemma 3) maps each such region onto a set containing the unit ball; the next layer's arrangement is then replicated inside every one of them, and the process iterates. Only the last layer uses an arrangement in general position, contributing $\sum_{j=0}^{n_0}\binom{n_k}{j}$.

**Asymptotics (App. A, Prop. 6, pp. 14–15).** With $n_0=O(1)$, a single-layer net with $kn$ units has $R(n_0;kn;1)=O(k^{n_0} n^{n_0})$, whereas a $k$-layer net of width $n$ satisfies $R = \Theta((\lfloor n/n_0\rfloor)^{k-1} n^{n_0})$. The multiplicative factor $\lfloor n/n_0\rfloor$ grows faster than $k$ (Polynomial in $n$) and, when $n>2n_0$ is fixed, exponentially in $k$.

**Regions per parameter (App. A, Props. 7–8, pp. 15–16).** A deep model with $k$ layers of width $n$ has $O(kn^2)$ parameters and a region-to-parameter ratio of order $(\lfloor n/n_0\rfloor)^{k-1} n^{n_0-2}/k$; a shallow model with $kn$ hidden units has $O(kn)$ parameters and ratio $O(k^{n_0-1} n^{n_0-1})$. Hence deep models yield exponentially more regions per parameter in depth and at least order-$(k-2)$ polynomially more in width, and there are deep models with fewer parameters but more regions than a shallow counterpart (§6, p. 14).

**A small, efficient deep model (§5, Thm. 2, p. 13).** A rectifier net with $n_0$ inputs and $k$ hidden layers of width exactly $2n_0$ has at least

$$
2^{(k-1)n_0} \sum_{j=0}^{n_0} \binom{2n_0}{j}
$$

response regions — proved by pairing rectifiers to compute absolute values layer by layer. For $n_0=2$: a shallow model with $4n_0$ units ($=8$) has at most $37$ regions, while the two-layer model with $2n_0$ units per layer produces $44$; for $6n_0$ hidden units a shallow model has at most $79$, while the three-layer model reaches $176$ (§5, p. 13). Figure 9(c) shows a 4-layer model with a total of 24 hidden units whose function would need at least 137 hidden units in a shallow model (§5, Fig. 9, p. 13).

## Limitations stated by the authors

- The region count is only an *upper bound* on flexibility; the pieces are correlated and the model may not be able to learn the optimal slope on each (§1, p. 2).
- The treatment assumes linear output units; the authors argue this is not a restriction, since the output activation is not parametrised and can be inverted on $\tanh$/sigmoid targets, or one of its preimages chosen for softmax (§6, p. 14).
- The paper is framed as "a first step": the construction is given for rectifier MLPs, and extension to other piecewise linear activations (maxout) and to convolutional rectifier nets is proposed as future work, not proved (§1, p. 2; §6, p. 15).
- There are no experiments; results are lower and upper bounds on region counts, not measurements of trained networks.

## Critical assessment

The paper's clean contribution is the *replication* mechanism: once a layer maps several input regions to sets each containing the unit ball, an arrangement placed afterwards is copied into every one of them, so region counts multiply across depth. This is exactly the combinatorial argument later systematised in the collection's [@NIPS2014_fa6f2a46] and the tropical-geometry line of work. Two caveats stand out.

First, the extraction of the display mathematics from the arXiv PDF is badly garbled, and I had to reconstruct the formulas in Theorem 1, Theorem 2, Proposition 6 and Proposition 8 from the surrounding prose and figure captions; the reconstruction of Theorem 1 is confirmed by the figure captions of Figs. 6–7 and by the proof text ($p=\lfloor n_1/n_0\rfloor$), and Theorem 2 by the paper's own numbers ($2^{2\cdot 1}\cdot(1+4+6)=44$ for $k=2,n_0=2$; $2^{2\cdot 2}\cdot 11=176$ for $k=3$). Equation 3 is the least certain: as printed the indicator vector appears to multiply the layer's pre-activations, and the grouping above is my reading. Readers should verify the exact constants against the published version.

Second, the claims are asymptotic and existential. Theorem 1 is a *lower* bound realised by one carefully chosen weight assignment, not by the weights a network actually learns, and the comparison with shallow models is at equal hidden-unit count rather than equal parameters everywhere (the parameter-controlled version is deferred to Props. 7–8). The honest framing — that a region count is an upper bound and that region correlation is desirable for generalisation — is a strength of the writing, but the paper gives no evidence that trained networks approach these counts. The remark that the advantage is recovered when data lie near a low-dimensional manifold (so that effective $n_0$ is small and $n \gg n_0$) is interesting but left as speculation (§6, p. 14).

## Connections within the collection

- It uses hyperplane-arrangement face-counting, specifically Zaslavsky's theorem [@zaslavskyFacingArrangementsFacecount1975] (Zaslavsky, 1975, Theorem A, §3, p. 5), for the single-layer count and the last-layer term of Theorem 1.
- It is the direct precursor of the collection's [@NIPS2014_fa6f2a46] (Montúfar, Pascanu, Cho, Bengio 2014, "On the Number of Linear Regions of Deep Neural Networks"), which extends the region-counting program — same authors, same geometric construction — to maxout and related piecewise linear units. The two should be read together.
- It is the foundational reference for the later region-counting literature in the collection: [@serraBoundingCountingLinear2018] (tighter bounds and an exact counting method for bounded inputs), [@haninComplexityLinearRegions2019] and [@haninDeepReLUNetworks2019] (average-case rather than worst-case counts), [@goujonNumberRegionsPiecewise2023] (asymptotic counts of regions for piecewise linear networks), [@brandenburgDecompositionPolyhedraPiecewise2024], [@masdenAlgorithmicDeterminationCombinatorial2022] and [@vincentReachablePolyhedralMarching2021] (computing the polyhedral decomposition), and [@raghuExpressivePowerDeep2017] (a different expressivity measure, trajectory length, with the same deep-vs-shallow spirit) — all possibly related through the region-counting paradigm rather than by direct citation here.
- The absolute-value construction of §5 and the piecewise linear view connect to the spline reading of ReLU networks in [@balestrieroSplineTheoryDeep2018] and, more distantly, the tropical-geometry treatment in [@zhangTropicalGeometryDeep].
