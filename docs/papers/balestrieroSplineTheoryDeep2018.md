---
citekey: balestrieroSplineTheoryDeep2018
title: "A Spline Theory of Deep Networks"
authors: [Randall Balestriero, Richard G. Baraniuk]
year: 2018
venue: Proceedings of the 35th International Conference on Machine Learning (PMLR 80)
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: 4fee2e55c6ae76b22ba99d350d7d96601dc0d2c0eb58348d7c9473434fb17199
pages: 5
ingested: 2026-10-15
tags: [spline-theory, max-affine-spline-operator, relu-networks, piecewise-linear, matched-filters, orthogonal-templates]
concepts_introduced:
  - max-affine spline operator
  - affine spline operator
  - signal-dependent affine mapping of a network
  - matched filter template of a network
  - orthogonal template penalty
  - partition-based signal distance
  - hierarchical vector quantization by a network
concepts_used:
  - spline approximation
  - piecewise affine function
  - rectified linear unit
  - max-pooling
  - matched filter
  - cross-entropy loss
  - backpropagation
  - input space partition into linear regions
builds_on: []
datasets: [MNIST, CIFAR10, CIFAR100, SVHN]
---

# A Spline Theory of Deep Networks

**Balestriero, Baraniuk (2018)**, *Proceedings of the 35th International Conference on Machine Learning (PMLR 80)*.

!!! abstract "TL;DR"
    Any deep network assembled from affine maps and piecewise-affine convex nonlinearities (ReLU, leaky ReLU, $|\cdot|$, max-pooling) is a composition of *max-affine spline operators* (MASOs). Conditioned on its input, such a network computes a signal-dependent affine map $z=A[x]x+B[x]$, so its output is a bank of matched filters whose rows are the network's class-specific, signal-dependent templates. Penalising those templates to be mutually orthogonal improves CIFAR100 accuracy and reduces overfitting with no architectural change, and the spline partition of input space yields a new distance between signals.

## Problem and motivation

Deep networks are empirically dominant but understood mostly as black boxes; the authors list five existing research camps and place themselves in a sixth, approximation theory (§1, p. 1). The question is structural: what *are* the input–output maps that a deep network computes, and can a classical function-approximation object describe them exactly rather than by analogy? The paper's answer is that the object is a spline: a piecewise-affine function whose pieces are glued along the boundaries of a partition of the input domain. The setting is a standard supervised deep network $f_\Theta:\R^D\to\R^C$ fed with vectorised signals $x\in\R^D$ (§2, p. 2). The claim is that for the *current standard-practice* nonlinearities — ReLU, leaky ReLU, absolute value, max-pooling, all piecewise-affine and convex — the network is not merely approximable by a spline but *is* a composition of spline operators, and that this description explains three empirical facts: the network's output as template matching, its tendency to memorise training data, and the hierarchical organisation of its internal representations.

## Method

The engine is the **max-affine spline operator**. A classical multivariate affine spline $s[\alpha,\beta,\Omega]$ glues affine pieces $\varphi_r(x)=\langle[\alpha]_{r,\cdot},x\rangle+[\beta]_r$ on the regions $\omega_r$ of a partition $\Omega$ (Eq. 3, p. 3). The difficulty with splines is that the partition ("knots") must be optimised jointly with the slopes; but if the spline is constrained to be globally convex it can be rewritten with no explicit partition as a max of affine functions (Eq. 4, p. 3), an *adaptive partitioning* spline whose partition is induced by $\alpha,\beta$ (Magnani & Boyd 2009, outside the collection). Stacking $K$ such functions gives a MASO $S[A,B]:\R^D\to\R^K$ with slopes $A\in\R^{K\times R\times D}$ and offsets $B\in\R^{K\times R}$ (Eq. 5, p. 3).

The method then proceeds in four steps, each a named contribution:

1. **Every layer is a MASO** (§4.1, pp. 4–5). A fully connected or convolution operator is a *degenerate* MASO with $R=1$ (Prop. 1); a piecewise-affine convex activation is a MASO with $R=2$, the two pieces being the branches below and above the kink (Prop. 2, e.g. ReLU takes $[A_\sigma]_{k,1,\cdot}=0$ and $[A_\sigma]_{k,2,\cdot}=e_k$); a piecewise-affine convex pooling operator is a MASO (Prop. 3). Composing one affine layer with one activation or pooling operator gives a layer that acts as $f^{(\ell)}(z^{(\ell-1)}(x))=A^{(\ell)}[x]z^{(\ell-1)}(x)+B^{(\ell)}[x]$ (Prop. 4, Eq. 6), and any network built from these operators is a composition of MASOs equivalent to a global affine spline operator (Thm. 1). The composition is *not* itself a MASO in general; it remains one when every non-first component is non-decreasing in each output dimension, which negative weights in $W$ or $C$ violate (§4.1, p. 4).
2. **A convex special case** (Thm. 2, p. 4): if all weight matrices have nonnegative entries and the activation and pooling operators are non-decreasing, piecewise-affine and convex, the *whole network* is a single MASO and is globally convex in each output dimension. This is exactly the input-convex-network regime.
3. **Collapsing the composition** (§4.2): substituting Eq. 5 into the layer recursion gives a closed-form, signal-dependent affine expression for a standard ReLU+max-pooling CNN, $z^{(L)}_{\mathrm{CNN}}(x)=W^{(L)}(A_{\mathrm{CNN}}[x]x+B_{\mathrm{CNN}}[x])+b_W^{(L)}$ (Eq. 8), the inner factor $A_{\mathrm{CNN}}[x]x+B_{\mathrm{CNN}}[x]$ being the featurisation that feeds the final linear classifier. Analogous formulas for ResNets and RNNs are derived in the extended companion (Balestriero & Baraniuk 2018, outside the collection).
4. **Two applications follow mechanically**: interpret the output as matched filtering (§5) and read the network's implicit input-space partition (§7).

For the matched-filter reading, the $c$-th output entry is the inner product of $x$ with the template in the $c$-th row of $W^{(L)}A[x]$ (§5.1, p. 5), and cascading the per-layer maxima shows that the network performs a *hierarchical, greedy* maximisation over region indices, globally optimal only under Thm. 2 (Eq. 10; Cor. 1). Templates are computed by backpropagation: $A[x]_c = \mathrm{d}[z^{(L)}(x)]_c/\mathrm{d}x$, and then $b[x]_c = z^{(L)}(x)_c - \langle A[x]_{c,\cdot}, x\rangle$ (§5.2, p. 5). An idealised analysis of the cross-entropy optimum (Prop. 5) yields the collinear templates of Eq. 12. To mitigate the matched-filter assumption of additive white Gaussian noise, the loss is augmented with a penalty forcing the rows of the final classifier to be mutually orthogonal (Eq. 13, §6, p. 6); orthogonality of $W^{(L)}$ is stated to be *necessary* for the templates to be orthogonal for all inputs.

For the geometric reading, the partition indices $[t^{(\ell)}(x)]_k$ selected by the argmax at each layer (Eq. 14) encode which region the signal falls in; comparing these integer codes gives a partition-based distance between two signals (Eq. 15, §7.3).

!!! note "Background"
    A *matched filter* is the linear filter that maximises output signal-to-noise ratio for a known deterministic signal in additive white Gaussian noise, so the optimal detector correlates the observation with a scaled copy of the candidate signal; a bank of matched filters, one per class, is then the optimal classifier for that noise model. This is textbook detection theory and is not taken from the collection.

## Key equations and definitions

A multivariate affine spline on a partition $\Omega=\{\omega_1,\dots,\omega_R\}$ with slopes $\alpha\in\R^{R\times D}$ and offsets $\beta\in\R^R$ (Eq. 3, p. 3):

$$
s[\alpha,\beta,\Omega](x) = \sum_{r=1}^{R}\big(\langle[\alpha]_{r,\cdot}, x\rangle + [\beta]_r\big)\,\mathbf{1}(x\in\omega_r) =: \langle\alpha[x], x\rangle + \beta[x]
$$

If the spline is globally convex it can be written without specifying $\Omega$ (Eq. 4, p. 3):

$$
s[\alpha,\beta,\Omega](x) = \max_{r=1,\dots,R}\ \langle[\alpha]_{r,\cdot}, x\rangle + [\beta]_r
$$

A max-affine spline operator with slopes $A\in\R^{K\times R\times D}$ and offsets $B\in\R^{K\times R}$ (Eq. 5, p. 3):

$$
S[A,B](x) =
\begin{bmatrix}
\max_{r=1,\dots,R}\langle[A]_{1,r,\cdot},x\rangle+[B]_{1,r}\\
\vdots\\
\max_{r=1,\dots,R}\langle[A]_{K,r,\cdot},x\rangle+[B]_{K,r}
\end{bmatrix}
=: A[x]\,x + B[x]
$$

A single layer acts as a signal-dependent affine map (Eq. 6, p. 4):

$$
f^{(\ell)}(z^{(\ell-1)}(x)) = A^{(\ell)}[x]\,z^{(\ell-1)}(x) + B^{(\ell)}[x]
$$

The closed form for a standard ReLU+max-pooling CNN ($A_\sigma$ for activations, $A_\rho$ for pooling; no offset terms because ReLU and max-pooling have none) (Eq. 7, p. 5):

$$
\begin{aligned}
z^{(L)}_{\mathrm{CNN}}(x) &= W^{(L)}\Big(\prod_{\ell=L-1}^{1} A^{(\ell)}_\rho[x]A^{(\ell)}_\sigma[x]C^{(\ell)}\Big)x \\
&\quad + W^{(L)}\sum_{\ell=1}^{L-1}\Big(\prod_{j=L-1}^{\ell+1} A^{(j)}_\rho[x]A^{(j)}_\sigma[x]C^{(j)}\Big)\big(A^{(\ell)}_\rho[x]A^{(\ell)}_\sigma[x]b^{(\ell)}_C\big) \\
&\quad + b^{(L)}_W
\end{aligned}
$$

The optimal templates at the constrained cross-entropy minimum (Prop. 5, p. 6), for unit-norm inputs $\norm{x_n}_2=1$ (Eq. 12, p. 6):

$$
[A^\star[x_n]]_{c,\cdot} =
\begin{cases}
+\sqrt{\dfrac{(C-1)\alpha}{C}}\; x_n, & c = y_n\\
-\sqrt{\dfrac{\alpha}{C(C-1)}}\; x_n, & c \neq y_n
\end{cases}
$$

The orthogonality penalty added to the cross-entropy loss (Eq. 13, p. 6):

$$
L_{CE} + \lambda\sum_{c_1\neq c_2}\big\lvert\langle[W^{(L)}]_{c_1,\cdot},[W^{(L)}]_{c_2,\cdot}\rangle\big\rvert^2
$$

The region index selected at layer $\ell$ (Eq. 14, p. 8), and the partition-based distance between two signals (Eq. 15, p. 8):

$$
[t^{(\ell)}(x)]_k = \arg\max_r\ \langle[A^{(\ell)}]_{k,r,\cdot}, z^{(\ell-1)}(x)\rangle + [B^{(\ell)}]_{k,r}
$$

$$
d\big(t^{(\ell)}(x_1), t^{(\ell)}(x_2)\big) = 1 - \frac{\sum_{k=1}^{D^{(\ell)}}\mathbf{1}\big([t^{(\ell)}(x_1)]_k = [t^{(\ell)}(x_2)]_k\big)}{D^{(\ell)}}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $D, C$ | input and output dimension of the network |
| $L, \ell$ | number of layers; layer index |
| $z^{(\ell)}(x)$ | feature map at layer $\ell$; $z^{(0)}(x)=x$, $z^{(L)}(x)=:z$ |
| $\Theta$ | all network parameters |
| $W^{(\ell)}, b_W^{(\ell)}$ | fully connected weights and bias at layer $\ell$ |
| $C^{(\ell)}, b_C^{(\ell)}$ | convolution matrix and bias at layer $\ell$ |
| $\sigma, \rho$ | scalar activation; (max-)pooling policy |
| $A_\sigma^{(\ell)}[x], A_\rho^{(\ell)}[x]$ | signal-dependent MASO matrices of the activation / pooling operator |
| $\Omega=\{\omega_1,\dots,\omega_R\}$ | partition of the domain into $R$ regions; $\omega_r$ a region |
| $\alpha\in\R^{R\times D}, \beta\in\R^{R}$ | slopes and offsets of a multivariate affine spline |
| $A\in\R^{K\times R\times D}, B\in\R^{K\times R}$ | slopes and offsets of a MASO; $K$ its output dimension |
| $A[x], B[x]$ | signal-dependent matrix and vector selected by the argmax, Eq. 5 |
| $r_k(x)$ | argmax (active hyperplane) index for output dimension $k$ |
| $A_{\mathrm{CNN}}[x], B_{\mathrm{CNN}}[x]$ | accumulated signal-dependent featurisation, Eq. 7 |
| $A^\star[x_n]$ | optimal (memorised) templates of Prop. 5 |
| $\alpha$ | re-used in Eq. 12 as the regularization bound $\sum_c\norm{A[x_n]_{c,\cdot}}^2<\alpha$ |
| $\lambda$ | weight of the orthogonality penalty, Eq. 13 |
| $t^{(\ell)}(x)$ | region index vector selected at layer $\ell$, Eq. 14 |
| $R^{(\ell)}$ | number of regions of the layer-$\ell$ input space |
| $x_1, x_2$ | two signals compared by the partition distance, Eq. 15 |

## Main results

- **Structural (Thm. 1, p. 4).** Networks built from fully connected/convolution, piecewise-affine convex activation, and piecewise-affine convex pooling operators are compositions of MASOs and hence equivalent to a global affine spline operator. Consequently the output is always a *signal-dependent affine function of the input* (Eq. 5 and Eq. 8), which is the paper's central take-away.
- **Convexity (Thm. 2, p. 4).** With nonnegative weights and non-decreasing activation/pooling, the entire network is one MASO and globally convex in each output dimension; there the hierarchical greedy template matching of Eq. 10 becomes globally optimal (Cor. 1).
- **Template structure (Prop. 5, Fig. 2, p. 6).** At the constrained cross-entropy optimum the optimal templates are collinear with the input itself, positive and large for the correct class and negative and small for every other class (Eq. 12), which pushes the softmax towards a one-hot vector. Figure 2 confirms the predicted bimodality of $\langle[A^\star[x_n]]_{c,\cdot}, x_n\rangle$ for a largeCNN on MNIST and CIFAR10, with the effect stronger on the easier MNIST problem.
- **Template visualisation (Fig. 1, p. 6).** For an MNIST "0" the three class templates ($0,3,7$) give inner products $32.7$, $-27.7$, $-27.8$; for a CIFAR10 "truck" the templates (truck, bird, dog) give $30.6$, $-12.5$, $-16.2$.
- **Orthogonal templates (Fig. 3, p. 7).** CIFAR100 with largeCNN: benefits "emerge distinctly as soon as $\lambda>0$", with improved final accuracy, improved generalisation and visibly less overfitting; the figure plots mean, standard deviation and maximum test accuracy over 15 runs for a sweep of $\lambda$, plus training error for $\lambda=0$ versus $\lambda=1$. The main text gives no numeric accuracies and reports only this one configuration; it states that the trends for SVHN, CIFAR10 and CIFAR100 and for smallCNN, largeCNN and ResNet4-4 are similar, with full results in the extended companion (outside the collection).
- **Partition geometry (Fig. 4, §7).** A ReLU partitions its layer input into up to $2^{D^{(\ell)}}$ regions and max-pooling into up to $\#R^{D^{(\ell)}}$ (§7.1; $\#R$ = pooling region size); the overall partition is the intersection of the per-layer partitions, which the authors read as hierarchical vector quantisation. Under the partition distance (Eq. 15), nearest neighbours of a test image at successive layers ($\ell=1,\dots,6$ of a smallCNN) move from sharing colour and shape to sharing class, while becoming *further* apart in Euclidean distance (Fig. 4).

## Limitations stated by the authors

- The framework covers piecewise-affine convex nonlinearities; sigmoid and tanh are only approximable by an affine spline operator, not representable as a MASO (§4.1, p. 4).
- A composition of MASOs is a MASO only under the non-decreasing condition; negative weights break it, so the global-convexity results (Thm. 2, Cor. 1) are a special case (§4.1, p. 4).
- "Unfortunately there is no simple formula for the partition of the signal space" (§7.2, p. 8); the partition is only probed empirically through the region indices.
- Because of the very large number of possible regions and limited training data, many regions are empty or contain a single training point (§7.2, p. 8).
- The matched-filter noise model is "overly simplistic" for most machine-learning problems, and the orthogonality penalty is imported from non-Gaussian detection, where the paper concedes little prior work exists (§6, p. 6).
- The appendices with proofs and the full experimental details are in the supplementary material, and a "significantly extended account … with numerous new results" is in the companion publication (§1, p. 2; §6, p. 7).

## Critical assessment

The conceptual contribution is real and has aged well: identifying standard networks with max-affine splines is the cleanest way to say "a ReLU network is piecewise affine on a polyhedral complex", and the MASO bookkeeping ($A[x]$, $B[x]$, region indices) is the notation that the later tropical-geometry and linear-region literature converges to. My reservations are about the evidence, not the framing.

First, the empirical claim is thin as printed. The main text reports one architecture/dataset pair (CIFAR100, largeCNN), and even there only a figure; no table, no accuracy values, no numeric baseline at $\lambda=0$, and no confidence interval beyond the shaded standard deviation. Every other combination is deferred to a companion (Balestriero & Baraniuk 2018) that is *not* in this collection, so nothing beyond Fig. 3 can be checked from the file.

Second, the mechanism of the orthogonality gain is under-identified. The paper argues that orthogonality of $W^{(L)}$ is *necessary* for orthogonal templates for all inputs; it is not sufficient (templates are rows of $W^{(L)}A[x]$, and $\langle W_{c_1}A, W_{c_2}A\rangle = \langle W_{c_1}, W_{c_2}AA^{\mathsf T}\rangle$), so the penalty enforces a necessary condition, and nothing isolates whether the accuracy gain comes from template orthogonality, from an implicit shrinkage of $\norm{W^{(L)}}$, or from a change in the effective learning rate of the last layer. No ablation with a norm-matched alternative regulariser is shown. Third, Prop. 5 is so idealised (arbitrary MASO matrices reachable, unit-norm inputs, a particular norm constraint) that Fig. 2's bimodality is better read as a sanity check of the picture than as a test of the theory. Fourth, the partition distance (Eq. 15) is a Hamming distance over region indices with no invariance theory, and its validation is qualitative ("Visual inspection … highlights") — no retrieval accuracy or clustering metric is reported. Fifth, no code or data release is mentioned.

Two smaller points. The printed Eq. 11 carries what looks like a superscript slip on the left-hand side: the extracted text reads $z^{(L-1)}(x)$ while the right-hand side applies the final classifier $W^{(L)}$ and adds $b_W^{(L)}$, which produces $z^{(L)}(x)$ (c.f. Eq. 10); I therefore omitted it above. (I could not view the rendered page — this model has no vision — so the slip is inferred from the extracted text, not confirmed visually.) And the symbol $\alpha$ carries two unrelated meanings (spline slopes in Eq. 3 and the Prop. 5 regularization bound), which invites confusion. The natural follow-ups — does the orthogonality penalty help at scale, and is the induced partition the same object counted by the linear-region literature — are left open.

## Connections within the collection

The paper is a root of this collection's ReLU-geometry cluster. Its partition $\Omega$ is exactly the object whose region count the counting papers bound and compute: the classical depth/width region counts [@NIPS2014_fa6f2a46], the response-region bounds of [@pascanuNumberResponseRegions2013], the tighter bounds of [@serraBoundingCountingLinear2018], the average-case analysis of [@haninComplexityLinearRegions2019] and [@haninDeepReLUNetworks2019], and the general piecewise-linear count of [@goujonNumberRegionsPiecewise2023] — all of which ask how many pieces the spline of this paper has, so they are the quantitative counterpart of its partition picture. Algorithms that recover the partition's combinatorial structure, such as [@carlssonPREIMAGERECTIFIERNETWORK2017], [@masdenAlgorithmicDeterminationCombinatorial2022], [@berzinsPolyhedralComplexExtraction2023] and [@lewandowskiSpaceFoldsReLU2024], operationalise the region index $t^{(\ell)}(x)$ of Eq. 14 on real networks.

The MASO formalism is mathematically a *tropical* (max-plus) polynomial, and the collection contains two tropical-geometry treatments that reach the same piecewise-linear picture by a different route: [@zhangTropicalGeometryDeep] and, for decision boundaries, [@alfarraDecisionBoundariesNeural2023]. That this paper does not make the tropical connection is a matter of timing (both are contemporaneous), so the relation is one of synonymy rather than dependence — possibly related, with differing notation. The most direct descendant in the collection is [@humayunSplineCamExactVisualization2024], whose SplineCam visualises a network by exploiting precisely the piecewise-affine spline structure advocated here; template-based visualisation and exact region visualisation are two uses of the same decomposition. The symmetry/equivalence questions of [@phuongFunctionalVsParametric2019] and [@grigsbyHiddenSymmetriesReLU] bear on how many distinct MASOs map to one function, and [@keupOrigamiDimensionsHow2022] and [@dhayalkarDiscreteFunctionalGeometry2025] study the same space-folding/transition structure from a statistical-physics and graph viewpoint respectively. The orthogonal-template penalty has no counterpart in the collection: the other papers study geometry, this is the one that turns geometry into a training regulariser.
