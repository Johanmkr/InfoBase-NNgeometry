---
citekey: haninComplexityLinearRegions2019
title: "Complexity of Linear Regions in Deep Networks"
authors: [Boris Hanin, David Rolnick]
year: 2019
venue: "Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97"
doi: "10.48550/arXiv.1901.09021"
arxiv: "1901.09021"
url: null
kind: theory
source_type: pdf
source_sha256: 3affe4649f3d980ef3ed2034fcce5ec4a05d2f2e3cbeacb33d82c1691594dce5
pages: 16
ingested: 2026-10-15
tags: [linear-regions, relu-networks, expressivity, initialization, adversarial-robustness]
concepts_introduced:
  - average number of linear regions along a line
  - volume density of linear region boundaries
  - distance to the nearest region boundary
  - practical versus theoretical expressivity
concepts_used:
  - ReLU network
  - piecewise linear activation
  - hyperplane arrangement
  - co-area formula
  - He initialization
  - expressivity of depth
builds_on: [NIPS2014_fa6f2a46, raghuExpressivePowerDeep2017, serraBoundingCountingLinear2018]
datasets: [MNIST]
---

# Complexity of Linear Regions in Deep Networks

**Hanin, Rolnick (2019)**, *ICML 2019, PMLR 97*. [arXiv:1901.09021](https://arxiv.org/abs/1901.09021)

!!! abstract "TL;DR"
    The theoretical maximum number of linear regions of a piecewise-linear network is exponential in depth, but special constructions realizing it are numerically fragile. Hanin and Rolnick instead count *average* regions for random weights/biases: the expected number of regions along any 1D line is proportional to the number of neurons (not exponential), the expected $(n_{\rm in}{-}1)$-dimensional volume of the region boundaries inside a set $K$ is at most $O(\#\{\text{neurons}\}/{\rm vol}(K))$ times ${\rm vol}(K)$, and the expected distance to the nearest boundary is bounded below by a constant over the number of neurons. MNIST experiments show these quantities stay near their initialization values throughout training, so practical expressivity is far below the theoretical maximum.

## Problem and motivation

A network with a piecewise-linear activation (ReLU, hard tanh) partitions its input space into *linear regions*, on each of which it computes a single affine function; the number of regions is a standard proxy for expressivity (§1, §2.1). Two extremes are known: a single region is attainable (all weights/biases zero), while exponential-in-depth counts are attainable by special weight configurations, e.g. Telgarsky's sawtooth with $2^n$ teeth using $3n+4$ neurons (§1; Fig. 3). Which extreme *typical* networks occupy — before and after training — was open. The paper sets out to characterize the average behavior for random weights and biases, and, because region counts alone say little about how regions are shaped, also introduces the *volume of the boundary* between regions as a second complexity measure, which controls the typical distance from a random input to its nearest region boundary and hence relates to stability under adversarial perturbation (§1, §2.2). The authors stress a distinction between *theoretical expressivity* (what an architecture can represent) and *practical expressivity* (what naturally occurring parameters represent) (§1). The setting is a depth-$d$ fully connected net with input dimension $n_{\rm in}$, output dimension 1, hidden widths $n_j$, under non-degeneracy assumptions A1–A2 (§5); the results also cover general continuous piecewise-linear activations with $T$ breakpoints and arbitrary feed-forward connectivity (§2, App. A).

## Method

The technical vehicle is a change-of-variables identity for the expected boundary volume, derived from the co-area formula.

- **Boundary as switching neurons.** $B_N$ (Eq. 4) is the set where $\nabla N$ is discontinuous. It is decomposed by co-dimension into pieces $B_{N,k}$ (Eq. 5), each piece being locally a co-dimension-$k$ hyperplane.
- **Neurons that switch.** For a neuron $z$ with pre-activation $z(x)$ and bias $b_z$, define $S_z = \{x : z(x) - b_z \in \{\lambda_1,\dots,\lambda_T\}\}$ (Eq. 17), the inputs at which the neuron crosses a breakpoint of the activation. Only neurons lying on an *open* path to the output ("good" neurons, §5) actually create a discontinuity, giving the refined set $\tilde S_z$ (Eq. 18).
- **Representation.** With probability 1, $B_N$ is the disjoint union of the $\tilde S_z$, and $B_{N,k}$ is the set where exactly $k$ neurons switch (Props. 9–10, App. B–C).
- **Expected volume.** Pushing the volume element through the map $x \mapsto (z_1(x),\dots,z_k(x))$ gives the expectation as an integral over $K$ of a Jacobian norm times the bias density (Prop. 11, Thm. 3, Eq. 8). Under initialization assumptions on gradient magnitude and bias density this yields explicit upper and lower bounds (Cor. 4, Eq. 9–10; Cor. 7, Eq. 15).
- **From volume to distance.** Chebyshev plus the boundary volume gives the lower bound on distance to the nearest boundary (Cor. 5/8, §E).

Experiments (fully connected nets, He-normal weights, biases $\mathcal N(0,10^{-6})$, Adam at rate $10^{-3}$, vectorized MNIST input dimension 784, test accuracy 95–98%) count regions along lines exactly, layer by layer (§3.1), and estimate the distance to the nearest boundary at 10 000 random points (§3.2); a 2D slice through three MNIST examples is visualized (§3.3).

## Key equations and definitions

The universal upper bound on the region count and the mean-gradient assumption (§2.1, Eqs. 1–2):

$$
\max \#\{\text{regions}\} \le 2^{\#\{\text{neurons}\}}, \qquad \min \#\{\text{regions}\} = 1,\quad \forall N, \qquad \text{(Eq. 1)}
$$

$$
\E\bigl[\norm{\nabla z(x)}\bigr] \le C,\quad \text{some } C>0. \qquad \text{(Eq. 2)}
$$

Theorem 1 (informal, §2.1): for input/output dimension 1 and stimuli confined to $I \subset \R$,

$$
\E\bigl[\#\{\text{regions in } I\}\bigr] \approx |I| \cdot T \cdot \#\{\text{neurons}\},
$$

with $T$ the number of breakpoints of the activation ($T=1$ for ReLU). The volume-density generalization and the exact distance formula (§2.2, Eqs. 3–4) are

$$
\frac{\mathrm{vol}_{n_{\rm in}-1}(B_N \cap K)}{\mathrm{vol}_{n_{\rm in}}(K)}, \qquad B_N = \{x \mid \nabla N(x) \text{ is not continuous at } x\}, \qquad \text{(Eqs. 3–4)}
$$

$$
\mathrm{distance}(x,B_N) = \min_{\text{neurons } z} \frac{|z(x)-b_z|}{\norm{\nabla z(x)}}. \qquad \text{(unnumbered, §2.2, p. 4)}
$$

Theorem 2 (informal, §2.2), for $K \subset \R^{d_{\rm in}}$ bounded: $\E[\mathrm{vol}_{n_{\rm in}-1}(B_N\cap K)/\mathrm{vol}_{n_{\rm in}}(K)] \approx T\cdot\#\{\text{neurons}\}$, and for $x \sim \mathrm{Unif}([0,1]^{n_{\rm in}})$, $\E[\mathrm{distance}(x,B_N)] \ge C(\#\{\text{neurons}\})^{-1}$.

The main theorem (Thm. 3, §5, Eq. 8) computes the expected $(n_{\rm in}-k)$-dimensional volume of $B_{N,k}$ exactly:

$$
\E\bigl[\mathrm{vol}_{n_{\rm in}-k}(B_{N,k}\cap K)\bigr] = \sum_{\substack{\text{distinct neurons}\\ z_1,\dots,z_k \text{ in } N}} \int_K \E\bigl[\Upsilon_{z_1,\dots,z_k}(x)\bigr]\, \mathrm{d}x, \qquad \text{(Eq. 8)}
$$

$$
\Upsilon_{z_1,\dots,z_k}(x) = \norm{J_{z_1,\dots,z_k}(x)}\,\rho_{b_{z_1},\dots,b_{z_k}}\bigl(z_1(x),\dots,z_k(x)\bigr) \cdot \mathbb{1}\{z_j \text{ good at } x\ \forall j\}, \qquad \text{(Eq. 13)}
$$

where $J_{z_1,\dots,z_k}$ is the $k\times n_{\rm in}$ Jacobian of $x\mapsto(z_1(x),\dots,z_k(x))$ and $\norm{J} := \det\bigl(J J^{\mathsf T}\bigr)^{1/2}$. Under A1–A2 and symmetric, variance-$2/{\rm fan\text{-}in}$ weights (Cor. 4), the $(n_{\rm in}-k)$-boundary volume obeys, with $\binom{\#\{\text{neurons}\}}{k}$ the binomial coefficient,

$$
\frac{\E\bigl[\mathrm{vol}_{n_{\rm in}-k}(B_{N,k}\cap K)\bigr]}{\mathrm{vol}_{n_{\rm in}}(K)} \le \binom{\#\{\text{neurons}\}}{k}\,(C_{\rm grad}\cdot C_{\rm bias})^k, \qquad \text{(Eq. 9)}
$$

$$
C_{\rm bias} = \sup_z \sup_{b\in\R}\rho_{b_z}(b), \qquad C_{\rm grad} = \sup_z \sup_{x\in\R^{n_{\rm in}}} \E\bigl[\norm{\nabla z(x)}^{2k}\bigr]^{1/k} \le C e^{C\sum_{j=1}^{d} 1/n_j}, \qquad \text{(Eq. 9)}
$$

and (Eq. 10) provides a matching lower bound $\binom{\#\{\text{neurons}\}}{k} c_{\rm bias}^k$, with $c_{\rm bias} = \inf_{|b|\le\eta}\rho_{b_z}(b)$ and an explicit $\eta$. Corollary 5 then gives, for $x$ uniform on a compact $K$,

$$
\E\bigl[\mathrm{distance}(x,B_N)\bigr] \ge \frac{c}{C_{\rm bias}C_{\rm grad}\,\#\{\text{neurons}\}}. \qquad \text{(Cor. 5, §5)}
$$

Appendix A restates everything for a general continuous piecewise-linear $\varphi$ with breakpoints $\lambda_1<\dots<\lambda_T$ (Eq. 11) and arbitrary feed-forward connectivity (Thm. 6, Eq. 12; Cor. 7, Eq. 15), and App. C–E give the proofs.

| Symbol | Meaning (in this paper) |
|---|---|
| $N$ | feed-forward ReLU net, input dim $n_{\rm in}$, output dim 1, depth $d$, widths $n_j$ |
| $P_j$ | polytope / linear region on which $N$ is affine |
| $B_N$ | boundary of linear regions: points where $\nabla N$ is discontinuous (Eq. 4) |
| $B_{N,k}$ | the "$(n_{\rm in}-k)$-dimensional piece" of $B_N$ (Eq. 5) |
| $z(x)$, $b_z$, $\ell(z)$ | pre-activation, bias and layer index of neuron $z$ (Eq. 6) |
| $Z(x)$ | post-activation $Z(x)=\mathrm{ReLU}(z(x))=\max\{0,z(x)-b_z\}$ (Eq. 7) |
| $J_{z_1,\dots,z_k}(x)$ | $k\times n_{\rm in}$ Jacobian of $x\mapsto(z_1(x),\dots,z_k(x))$ |
| $\rho_{b_{z_1},\dots,b_{z_k}}$ | joint density of the biases (Assumption A1) |
| $\Upsilon_{z_1,\dots,z_k}(x)$ | integrand of Thm. 3 (Eq. 13) |
| "good" neuron at $x$ | lies on a path to the output with every neuron open at $x$ (§5) |
| $T$ | number of breakpoints of the activation ($1$ for ReLU) |
| $C_{\rm grad}, C_{\rm bias}, c_{\rm bias}$ | gradient bound, bias-density sup, bias-density inf (Eqs. 9–10) |
| $T_\epsilon(B_N)$ | tube of radius $\epsilon$ around $B_N$ (§E) |

## Main results

**Average regions along a line are linear in the neurons (§2.1, §3.1).** Theorem 1 gives $\E[\#\{\text{regions in }I\}]\approx|I|\cdot T\cdot\#\{\text{neurons}\}$, so depth does not enter: a deep and a shallow net with the same neuron count have the same average complexity along a curve. Experiments confirm this for all architectures: at initialization the number of regions along a line is a constant times the number of neurons, the constant "very close to 1" (§3.1); during training it stays within a small constant of that value, first dipping slightly (through 0.5 epochs) and then rebounding (Fig. 5).

**Boundary volume and distance to the boundary (§2.2, §5).** Theorem 3/Cor. 4 bound the expected boundary volume by $\binom{\#\{\text{neurons}\}}{k}(C_{\rm grad}C_{\rm bias})^k$ per unit volume, and Cor. 5 gives $\E[\mathrm{distance}(x,B_N)] \ge c/(C_{\rm bias}C_{\rm grad}\#\{\text{neurons}\})$. Experimentally the distance to the nearest boundary stays between $0.4/\#\{\text{neurons}\}$ and $1.5/\#\{\text{neurons}\}$ throughout training (Fig. 6(a)); at initialization all architectures share the same value of $\#\{\text{neurons}\}\cdot\mathrm{distance}$ (Fig. 6(b)). The log-distance distribution varies over many orders of magnitude and is only slightly altered by training — a slight skew toward larger regions (Fig. 7, depth 4 width 16, after 1 and 20 epochs).

**Two phases of training (§3.2, Fig. 6(c)).** Plotting $\#\{\text{neurons}\}\cdot\mathrm{distance}$ against test accuracy suggests regions first expand (generalization) then contract (memorization); the authors connect this to "learn patterns first" [Arpit et al. 2017, outside the collection] and label it speculation.

**2D slices (§3.3).** In a plane through three MNIST examples (labels 0, 1, 2) of a depth-3 width-64 net, the number of regions is found to be on the order of $(64\times 3)^2 \approx 4\times 10^4$, as the heuristic $\#\{\text{regions}\}\sim\#\{\text{neurons}\}^{n_{\rm in}}$ would predict; regions expand then contract (Fig. 8). The count is exact, obtained by splitting regions neuron by neuron, and terminates quickly precisely because few regions exist.

**Conclusion (§6).** Boundary volume depends on the number of neurons, not on depth, both at initialization and during training; the authors conclude "deeper networks do not learn more complex functions than shallow networks" on this measure, and explicitly do not propose a replacement theory for the success of deep learning.

## Limitations stated by the authors

The total-region bound is only along 1D curves (or the boundary volume): a bound on the total number of regions in all of $\R$ is left to future work, believed to be proportional to the number of neurons (§2.1, Fig. 5). The heuristic $\#\{\text{regions}\}=\#\{\text{neurons}\}^{n_{\rm in}}$ for input dimension $>1$ is "work in progress" (§2.2). Bounds are for weights/biases at initialization, and while the authors argue Theorem 3 "suggests" the behavior during training, this is not a direct consequence (§1, §2.1). The two-phase-learning reading of Fig. 6(c) is "speculation" (§3.2). The generalization to arbitrary feed-forward connectivity is stated as not worked out in detail (§2).

## Critical assessment

The strength of the paper is its exactness: Theorem 3 is an identity, not merely a bound, and Cor. 4's upper bound is tight up to $C_{\rm grad}, C_{\rm bias}$; the assumptions A1–A2 are genuinely mild (any independent density-carrying weights/biases), so the result is not an artifact of Gaussian initialization. The distance bound Cor. 5, however, is a lower bound obtained by Chebyshev from the boundary volume — it says the *typical* distance is at least $\sim 1/\#\{\text{neurons}\}$ but not that it is that large, and indeed the experiments only pin it within the factor-$4$ window $[0.4,1.5]/\#\{\text{neurons}\}$ (Fig. 6(a)). The framing "practical expressivity is far below the theoretical maximum" is convincing for the *average* measures studied, but the paper is careful that it studies only one measure of complexity; the claim in §6 that deeper networks "do not learn more complex functions" is stronger than the evidence and is hedged in the text as a suggestion. Two internal numbering slips should be read cautiously: Cor. 5 in §5 is proved as "Corollary 8" in §E, and the Cor. 4 statement's bound uses a "k" factor that the surrounding prose does not always track (the $\binom{\#\{\text{neurons}\}}{k}$ coefficient is square; the version in App. A, Cor. 7, carries an extra $T$ and drops the binomial). The empirical study is a single dataset (MNIST) with small fully connected nets, so the "training does not change the regime" conclusion rests on one setting. The counting method is only tractable because regions are few — the very theorem under test is what makes the experiment computationally feasible, so the experiments cannot falsify the theorem in a regime where regions proliferate. Reproducibility is good in principle (all quantities are explicitly defined), but no code is released in the paper. The distance-to-boundary measure as a robustness proxy is motivated but never tested against actual adversarial attacks.

## Connections within the collection

It builds directly on the counting literature: it cites the maximal-region constructions of [@NIPS2014_fa6f2a46] and the refined width-dependent bounds of [@serraBoundingCountingLinear2018] and [@raghuExpressivePowerDeep2017] as the theoretical upper bounds that its average-case results undercut, and frames its contribution as the "practical" counterpart to those "maximal" counts. The companion paper [@haninDeepReLUNetworks2019] by the same authors studies the same average-case question through activation *patterns*, complementing the geometric region/boundary view taken here; both drive the average-case message that typical networks are far from the exponential maximum. The stochastic, path-based analysis of [@goujonNumberRegionsPiecewise2023] is a direct descendant of Theorem 1, generalizing the "linear in neurons" heuristic to arbitrary CPWL activations and giving an explicit expected knot density. The distance-to-boundary quantity introduced here is a precursor to the sensitivity-and-generalization measurements of later work, and to the exact region-visualization tools of [@humayunSplineCamExactVisualization2024] and [@berzinsPolyhedralComplexExtraction2023], which compute the polyhedral decomposition that this paper counts statistically. The spline framing of [@balestrieroSplineTheoryDeep2018] offers a complementary view of the same piecewise-linear structure.
