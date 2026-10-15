---
citekey: alfarraDecisionBoundariesNeural2023
title: "On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective"
authors: [Motasem Alfarra, Adel Bibi, Hasan Hammoud, Mohamed Gaafar, Bernard Ghanem]
year: 2023
venue: IEEE Transactions on Pattern Analysis and Machine Intelligence 45(4)
doi: "10.1109/TPAMI.2022.3201490"
arxiv: null
url: null
kind: method
source_type: pdf
source_sha256: 7db0978087535d04c752b6b7a1b45f1af781827ce23ac39b3d8c4ecbb0d7bff0
pages: 11
ingested: 2026-10-15
tags: [tropical-geometry, decision-boundaries, reLU-networks, network-pruning, adversarial-attacks]
concepts_introduced:
  - decision boundaries polytope
  - tropical superset of decision boundaries
  - convex hull of two zonotopes as dual subdivision
  - tropical network pruning
  - tropical adversarial attack
concepts_used:
  - tropical semiring
  - tropical polynomial
  - tropical rational function
  - Newton polytope
  - dual subdivision
  - zonotope
  - ReLU network
  - lottery ticket hypothesis
  - mixed norm group sparsity
builds_on: [zhangTropicalGeometryDeep]
datasets: [SVHN, CIFAR-10, CIFAR-100, MNIST]
---

# On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective

**Alfarra, Bibi, Hammoud, Gaafar, Ghanem (2023)**, *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [DOI](https://doi.org/10.1109/TPAMI.2022.3201490)

!!! abstract "TL;DR"
    For a bias-free (Affine, ReLU, Affine) network, the inputs where two logits tie lie in a tropical hypersurface $T(R(x))$ whose dual subdivision is the convex hull of two zonotopes with generators given by the weights (Thm. 2, §3). Calling this polytope the *decision boundaries polytope*, the paper reads three tasks through it: the lottery ticket hypothesis (its orientation survives lottery-ticket re-initialisation), pruning (a row-wise surrogate keeps the polytope while >90% of the classifier's parameters go), and adversarial attacks (perturbing the first layer moves the polytope and flips predictions).

## Problem and motivation

Understanding why deep networks work remains elusive, and decision boundaries are one of the objects used to probe it: earlier work studied their curvature and smoothness, their convergence to a linear SVM in the last layer, and their behaviour under input perturbations (§1, p. 5027). The piecewise-linear ReLU has attracted tropical geometry here, because a ReLU network is a tropical rational map: [@zhangTropicalGeometryDeep] showed the equivalence and used it to count linear regions, recovering the bounds of [@NIPS2014_fa6f2a46] more simply (§1, p. 5027, §2, p. 5028).

What was missing is a geometric description of the decision boundaries themselves — the inputs on which two logits tie — rather than of linear regions or of the network function. This paper derives one for $f(x) = B\max(Ax+c_1,0)+c_2$ with $A \in \mathbb{Z}^{p\times n}$, $B \in \mathbb{Z}^{2\times p}$: those boundaries are contained in a tropical hypersurface whose dual subdivision polytope is the convex hull of two zonotopes built from $A$ and $B$ (Contribution (i), §1, p. 5028; Thm. 2, p. 5029). The results are stated for bias-free two-output networks; biases, the multi-class case and all proofs are in the online supplement, not part of the file summarised here. Pruning and adversarial attacks are then cast as preserving or moving that polytope (Contributions (ii)–(iii), §1, p. 5028).

## Method

**Tropical preliminaries.** In the tropical semiring $\mathbb{T}=(\mathbb{R}\cup\{-\infty\},\oplus,\odot)$ addition is max and multiplication is standard addition; the *tropical hypersurface* $T(f)$ of a tropical polynomial is the set where two or more monomials attain the max. The Newton polytope $D(f)$ and dual subdivision $\delta(f)$ (the upper faces of $P(f)=\mathrm{ConvHull}\{(a_i,c_i)\}$ projected onto $\mathbb{R}^d$) encode the regions where $f$ is linear, and the edges of $\delta(f)$ are normal to $T(f)$ (§2, pp. 5028–5029). Theorem 1 of [@zhangTropicalGeometryDeep] is the bridge to networks: a feedforward network with integer weights, real biases and piecewise-linear activations has tropical rational coordinates, $f_i = H_i \oslash Q_i$ (§2, p. 5029).

**The geometric representation.** With $f_1 = H_1 \oslash Q_1$ and $f_2 = H_2 \oslash Q_2$, the equality $f_1(x)=f_2(x)$ is the tropical equality $H_1(x)\odot Q_2(x) = H_2(x)\odot Q_1(x)$, which is a face of the tropical polynomial $R(x) = H_1 \odot Q_2 \oplus H_2 \odot Q_1$. Hence $B$ lies inside $T(R(x))$, and it suffices to describe $\delta(R(x))$ — Theorem 2 shows this dual subdivision is the convex hull of two zonotopes in $\mathbb{R}^n$ with one line segment per hidden unit $j$ (§3, p. 5029; Fig. 2). Since fast vertex-enumeration algorithms for zonotopes assume segments anchored at the origin, Proposition 1 converts a zonotope with segments $[u_{i1},u_{i2}]$ into one with segments $[u_{i1}-u_{i2},0]$ plus a shift $\sum_i u_{i2}$, and Corollary 1 turns this into explicit generator matrices $G_1, G_2$ (§3, p. 5030). The polytope is then read through the orientation of its edges, whose normals are parallel to (a superset of) the decision boundaries.

**Lottery ticket perspective.** A 2-input, 2-output, 40-node single-hidden-layer network is pruned by removing the smallest $x\%$ of its weights and retrained with four initialisations — the lottery-ticket one, Xavier, standard Gaussian, zero-mean Gaussian (variance $0.1$) — with $\delta(R(x))$ plotted along the trajectory (§4, pp. 5030–5031, Fig. 3).

**Tropical pruning.** Because preserving $\delta(R(x))$ preserves the superset $T(R(x))$, pruning is cast as keeping the pruned network's polytope close to the original one, Problem (1), for which Problem (2) is a surrogate on the generator matrices with mixed-norm penalties that zero out whole rows of $\tilde A$ (i.e. hidden units). Problem (2) is not quadratic but separates over the rows of $\tilde A$ and $\tilde B$, each sub-problem having a closed-form solution (the $\tilde A$-row update is a scaled, thresholded average of the two generator rows), so one alternating iteration suffices in all experiments. Deeper networks use a greedy pass over consecutive (Affine, ReLU, Affine) blocks from input to output (§5, pp. 5031–5033).

**Tropical adversarial attacks.** Instead of searching for an input perturbation that crosses a decision boundary, one fixes $x_0$ and perturbs the first linear layer by $\tilde A_1$; the two views coincide when $A_1h = \tilde A_1 x_0$ (Eq. 4). The attack minimises $D_1(h)+D_2(\tilde A_1)$ subject to a margin-type loss constraint, a box constraint, budgets on $\Vert h\Vert_1$ and $\Vert \tilde A_1\Vert_{1,1}$, and $A_1h=\tilde A_1x_0$; $D_2$ measures how far the two zonotopes of Theorem 2 move. It is solved by a penalty method on the equality constraints, each penalty step by ADMM (§6, pp. 5033–5035).

## Key equations and definitions

Tropical addition and multiplication (Def. 1, §2, p. 5028):

$$
x \oplus y = \max\{x,y\}, \qquad x \odot y = x + y, \qquad \forall x,y \in \mathbb{T}.
$$

Tropical hypersurface of $f(x) = c_1 x^{a_1} \oplus \cdots \oplus c_n x^{a_n}$ (Def. 4, §2, p. 5028):

$$
T(f) := \{x \in \R^d : c_i x^{a_i} = c_j x^{a_j} = f(x) \text{ for some } a_i \neq a_j\}.
$$

Zonotope from generators $u^1,\dots,u^L$ and from a generator matrix $U$ (Def. 6, §2, p. 5029):

$$
Z(u^1,\dots,u^L) := \Bigl\{\textstyle\sum_{i=1}^{L} x_i u^i : 0 \le x_i \le 1\Bigr\}, \qquad Z_U := \{U^\top x : \forall x \in [0,1]^L\}.
$$

Tropical characterisation of networks (Thm. 1, due to [@zhangTropicalGeometryDeep]; §2, p. 5029): a feedforward network with integer weights, real biases and piecewise-linear activations is $f: \R^n \to \R^k$ whose coordinates are tropical rationals, $f(x) = H(x) \oslash Q(x) = H(x) - Q(x)$ with $H, Q$ tropical polynomials.

Theorem 2 (§3, p. 5029). For a bias-free network $f: \R^n \to \R^2$ with $A \in \mathbb{Z}^{p \times n}$, $B \in \mathbb{Z}^{2 \times p}$, let $R(x) = H_1(x) \odot Q_2(x) \oplus H_2(x) \odot Q_1(x)$. With $B = \{x \in \R^n : f_1(x) = f_2(x)\}$ the decision boundaries,

$$
B \subseteq T(R(x)), \qquad \delta(R(x)) = \mathrm{ConvHull}\bigl(Z_{G_1}, Z_{G_2}\bigr).
$$

$Z_{G_1}$ is a zonotope in $\R^n$ with line segments $\{(B^+(1,j)+B^-(2,j))\,[A^-(j,:), A^+(j,:)]\}_{j=1}^p$ and shift $(B^-(1,:)+B^+(2,:))A^-$; $Z_{G_2}$ has segments $\{(B^-(1,j)+B^+(2,j))\,[A^+(j,:), A^-(j,:)]\}_{j=1}^p$ and shift $(B^+(1,:)+B^-(2,:))A^-$, where $A^+ = \max(A,0)$ and $A^- = \max(-A,0)$ — with the same convention for $B$. No proof is given in the main text (appendix, not in this file).

Corollary 1 (§3, p. 5030): the generators can be taken as

$$
G_1 = \mathrm{Diag}\bigl[(B^+(1,:)) + (B^-(2,:))\bigr]A, \qquad G_2 = \mathrm{Diag}\bigl[(B^+(2,:)) + (B^-(1,:))\bigr]A,
$$

both with shift $(B^-(1,:) + B^+(2,:) + B^+(1,:) + B^-(2,:))A^-$, where $\mathrm{Diag}(v)$ puts $v$ on the diagonal.

Tropical pruning, surrogate problem (Problem 2, §5, p. 5032), with $\Vert C \Vert_{2,1} = \sum_{i=1}^{n}\Vert C(i,:)\Vert_2$ for $C \in \R^{n \times k}$:

$$
\min_{\tilde A, \tilde B} \; \tfrac{1}{2}\Vert \tilde G_1 - G_1\Vert_F^2 + \tfrac{1}{2}\Vert \tilde G_2 - G_2 \Vert_F^2 + \lambda_1 \Vert \tilde G_1\Vert_{2,1} + \lambda_2 \Vert \tilde G_2 \Vert_{2,1}.
$$

Closed-form row updates (§5, p. 5032), with $c_1 = \mathrm{ReLU}(B(1,:)) + \mathrm{ReLU}(-B(2,:))$, $c_2 = \mathrm{ReLU}(B(2,:)) + \mathrm{ReLU}(-B(1,:))$ and $c_1^i$ the $i$-th entry:

$$
\tilde A(i,:) = \max\Biggl\{\biggl(1 - \frac{\lambda_1\sqrt{c_1^i} + \lambda_2\sqrt{c_2^i}}{c_1^i + c_2^i}\biggr)\frac{c_1^i G_1(i,:) + c_2^i G_2(i,:)}{c_1^i + c_2^i},\; 0\Biggr\},
$$

$$
\tilde B(1,j) = \max\Biggl\{0, \; \frac{\tilde A(j,:)^\top G_1^+(j,:) - \lambda_1 \Vert \tilde A(j,:)\Vert_2}{\Vert \tilde A(j,:)\Vert_2^2}\Biggr\}, \qquad G_1^+ = \mathrm{Diag}(B^+(1,:))A .
$$

(The bracket/grouping structure of these two displays is reconstructed from the printed page; see the report note on the PDF text layer and the critical assessment below.)

Adversarial dual view and problem (Eqs. 3–6, §6, pp. 5034–5035), where $g$ is the network after the first layer and the loss is cross-entropy:

$$
\min_h D(h) \quad \text{s.t.} \quad \operatorname*{argmax}_i f_i(x_0 + h) = t \neq c \qquad \text{(Eq. 3)}
$$

$$
g\bigl((A_1 + \tilde A_1)x_0\bigr) = g\bigl(A_1 x_0 + \tilde A_1 x_0\bigr) = g(A_1 x_0 + A_1 h) = f(x_0 + h) \qquad \text{(Eq. 4)}
$$

$$
\min_{h,\tilde A_1} D_1(h) + D_2(\tilde A_1) \quad \text{s.t.} \quad \mathrm{loss}\bigl(g(A_1(x_0+h)), t\bigr) \le -1, \;\; \mathrm{loss}\bigl(g((A_1+\tilde A_1)x_0), t\bigr) \le -1,
$$

$$
(x_0+h) \in [0,1]^n, \quad \Vert h\Vert_1 \le \lambda_1, \quad \Vert \tilde A_1\Vert_{1,1} \le \lambda_2, \quad A_1 h = \tilde A_1 x_0 \qquad \text{(Eq. 5)}
$$

$$
D_2(\tilde A_1) = \frac{1}{2}\sum_{j=1}^{2}\Bigl(\bigl\Vert \mathrm{Diag}\bigl(B^+(j,:)\bigr)\tilde A_1 \bigr\Vert_F^2 + \bigl\Vert \mathrm{Diag}\bigl(B^-(j,:)\bigr)\tilde A_1 \bigr\Vert_F^2\Bigr). \qquad \text{(Eq. 6)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $\mathbb{T}$, $\oplus$, $\odot$ | tropical semiring; max; standard addition |
| $f_i = H_i \oslash Q_i$ | tropical rational representation of logit $i$ |
| $R(x)$ | tropical polynomial $H_1 \odot Q_2 \oplus H_2 \odot Q_1$; $B \subseteq T(R(x))$ |
| $B$ | decision boundaries $\{x : f_1(x) = f_2(x)\}$ |
| $T(f)$, $\delta(f)$, $D(f)$ | tropical hypersurface; dual subdivision; Newton polytope |
| $Z_U$, $G_1$, $G_2$ | zonotope of generator matrix $U$; generators of the two zonotopes |
| $A$, $B$, $c_1$, $c_2$ | first weight matrix, output weight matrix, biases |
| $A^+$, $A^-$ | $\max(A,0)$, $\max(-A,0)$ |
| $n$, $p$, $2$ | input dimension, hidden width, number of outputs |
| $\tilde A$, $\tilde B$ | pruned (sparser) weight matrices |
| $\lambda_1$, $\lambda_2$ | sparsity trade-offs (Problem 2); perturbation budgets (Eq. 5) |
| $h$, $\tilde A_1$ | input perturbation; first-layer perturbation |
| $D_1(h)$, $D_2(\tilde A_1)$ | input-perturbation and polytope-perturbation costs |

## Main results

**Lottery ticket hypothesis (§4, pp. 5030–5031, Fig. 3).** The orientation of $\delta(R(x))$ stays close to the original polytope throughout pruning when the pruned network is re-initialised exactly as the lottery-ticket algorithm prescribes, whereas Xavier, standard Gaussian and zero-mean Gaussian initialisations deviate much more and accuracy degrades. The authors stress that this refines the hypothesis: several networks can have similar accuracy with different decision boundaries. Studying the polytopes of $H_{\{1,2\}}$, $Q_{\{1,2\}}$ instead would be inconclusive, since different such polytopes can give the same boundaries.

**Tropical pruning (§5, pp. 5031–5033).** Toy networks ((Affine, ReLU, Affine), $100/200/300$ hidden nodes, two 2-D datasets) are competitive with, and often better than, decision-boundary-agnostic baselines. AlexNet and VGG16 (classifier sizes $(256,512,10)$ and $(512,512,10)$, last dimension $100$ for CIFAR-100; baseline test accuracies $92\%,74\%,43\%$ and $92\%,92\%,70\%$ on SVHN/CIFAR-10/CIFAR-100 respectively) lose more than $90\%$ of their classifier parameters "without affecting the accuracy". In area under the pruning-accuracy curve, tropical pruning beats the magnitude-based baselines (class blind, class uniform, class distribution) by $3\%$, $3\%$ and $2\%$ on SVHN, CIFAR-10 and CIFAR-100 for AlexNet, and is comparable on SVHN and CIFAR-10 for VGG16 while slightly worse on CIFAR-100. Against the tropical pruning of Smyrnis and Maragos (SM) on LeNet/MNIST, test accuracy (neurons kept, SM, ours) in % is (100, 98.60, 98.84), (90, 95.71, 98.82), (75, 95.05, 98.8), (50, 95.52, 98.71), (25, 91.04, 98.36), (10, 92.79, 97.99), (5, 92.93, 94.91) — a margin reaching $7\%$. Retraining only the biases can suffice, since biases translate rather than rotate the polytope.

**Tropical adversarial attacks (§6, pp. 5035–5036, Table 1).** On a 2-D synthetic binary problem (50 hidden nodes) perturbing first-layer weights moves the polytope and misclassifies a fixed $x_0$ (Fig. 7). On MNIST, single perturbations flip digits ($9 \to 8,4,3,2,1$; likewise for $5,4,1,8$), sometimes as small as $\lambda = 0.1$ with $x_0 \in [0,1]^n$. Robust accuracy over 1000 randomly selected correctly classified MNIST samples ($\lambda_1 \in \{0.1, 0.2\}$, $\lambda_2 \in \{0.0,1.0,2.0\}$; $\lambda_2 = 0$ recovers the cited structured adversarial attack) is: $\lambda_1 = 0.1$ — $44.8$ (baseline), $46.1$ ($\lambda_2 = 1.0$), $43.8$ ($\lambda_2 = 2.0$); $\lambda_1 = 0.2$ — $12.6$, $11.3$, $11.0$.

## Limitations stated by the authors

- The theory covers only shallow bias-free networks of the form (Affine, ReLU, Affine); biases, multi-class outputs and proofs are relegated to the online supplement, and deeper networks are handled by a greedy block-wise heuristic (§3, p. 5029, §5, p. 5033).
- No theoretical reason is offered for why the lottery-ticket pruning algorithm works; the argument is empirical (§4, p. 5030).
- The pruning method is not claimed to beat state-of-the-art pruning, and the attack not to beat state-of-the-art attacks; the contribution is the geometric perspective (§5, p. 5032, §6, p. 5036).
- Extending the characterisation to CNNs and graph convolutional networks is left as future work (§7, p. 5036).

## Critical assessment

The central statement is an *inclusion*, $B \subseteq T(R(x))$, not an equality, and nothing bounds how much larger $T(R(x))$ is than $B$. The applications inherit that slack: the pruning surrogate (Problem 2) matches generator matrices rather than the polytope distance of Problem (1), and no guarantee is given that a small generator mismatch implies a small boundary mismatch. The experiments carrying the paper's weight (AlexNet/VGG16 pruning) run outside the theorem's assumptions — bias-free, shallow, two-output — relying entirely on the greedy block heuristic, so the theory motivates the algorithm rather than validating it.

The empirical evidence is thin in places: pruning is reported as AUC curves without seeds or error bars and with a single alternating iteration; the LeNet/MNIST comparison is one architecture against one tropical baseline; and the adversarial study uses 1000 MNIST samples and deliberately avoids comparison with strong attacks, so Table 1 is best read as an ablation of $\lambda_2$ within the authors' own formulation.

Two transcription-level issues deserve flagging. First, Eq. (5) prints the loss constraints as $\mathrm{loss}(\cdot) \le -1$ although the text calls the loss cross-entropy, which can never be $\le -1$; presumably a margin-based loss (negative logit gap) is meant (sic). Second, the two closed-form row updates are dense fractional displays whose delimiter structure the PDF's text layer does not preserve; they are reconstructed as faithfully as the extracted glyph positions allow and should be checked against the printed page. The $\sqrt{\cdot}$ on $c_1^i,c_2^i$ is unusual for a least-squares-plus-norm objective and is worth re-deriving.

No code or data link is given in the main text, and the parts a re-implementer most needs (derivation of the updates, multi-class extension, bias handling, ADMM details) are in supplementary material unavailable here. The explicit polytopes are drawn only for $n = 2$ inputs; since the vertex count of a zonotope grows as $O(L^{n-1})$, the decision boundaries polytope is a genuinely low-dimensional object of study, and the paper does not discuss what replaces it in high input dimension.

## Connections within the collection

- [@zhangTropicalGeometryDeep] is the direct foundation: Theorem 1 (networks are tropical rationals) is restated from it, the paper positions itself as taking that work "several steps further" by treating the multi-class decision boundaries, and it attributes the single-output special case of the first part of Theorem 2 to it (§2, p. 5029, §3, p. 5029).
- [@NIPS2014_fa6f2a46] is cited as [10] for the counting of linear regions and is presented in the introduction as motivating the study of piecewise-linear expressivity; the paper notes that [@zhangTropicalGeometryDeep] recovered those bounds more simply (§1, p. 5027, §2, p. 5028).
- The counting-of-regions thread of the collection is adjacent, though not cited by this paper: [@goujonNumberRegionsPiecewise2023], [@serraBoundingCountingLinear2018], [@haninComplexityLinearRegions2019] and [@NIPS2014_fa6f2a46] bound or count the linear regions into which a network partitions its input space, whereas the object here is the tie set $B$ between two logits.
- Possibly related, and likewise not cited: [@humayunSplineCamExactVisualization2024] (SplineCam) computes and visualises the exact spline/polyhedral geometry and decision boundaries of trained ReLU networks — a computational counterpart to this paper's analytic polytope; [@balestrieroSplineTheoryDeep2018] develops the spline view whose partition is the same polyhedral decomposition; [@berzinsPolyhedralComplexExtraction2023], [@masdenAlgorithmicDeterminationCombinatorial2022] extract the polyhedral complex; [@keupOrigamiDimensionsHow2022], [@lewandowskiSpaceFoldsReLU2024] study folding; [@grigsbyHiddenSymmetriesReLU] study function-preserving symmetries.

Named plainly, outside the collection: the lottery ticket hypothesis of Frankle and Carbin (the empirical target of §4), the tropical polynomial division pruning of Smyrnis and Maragos (the SM baseline of §5), and the classical tropical-geometry references of Maclagan and Sturmfels and of Gritzmann and Sturmfels for hypersurface–subdivision duality and zonotope vertices.
