---
citekey: haninDeepReLUNetworks2019
title: "Deep ReLU Networks Have Surprisingly Few Activation Patterns"
authors: [Boris Hanin, David Rolnick]
year: 2019
venue: Advances in Neural Information Processing Systems 32 (NeurIPS 2019)
doi: "10.48550/arXiv.1906.00904"
arxiv: "1906.00904"
url: null
kind: theory
source_type: pdf
source_sha256: 640eb97bef910bd6504b38594af5756d8e4382ef0a0d1b9b15811288cbc0a98a
pages: 19
ingested: 2026-10-15
tags: [relu-networks, activation-regions, linear-regions, expressivity, initialization, memorization]
concepts_introduced:
  - activation region
  - activation pattern
  - linear region
  - bent hyperplane
  - partial activation region
  - expected number of activation regions
  - scale equivariance of zero-bias ReLU networks
concepts_used:
  - hyperplane arrangement
  - ReLU network
  - depth-width expressivity
  - neural network initialization
  - gradient norm of pre-activation
  - memorization capacity
builds_on: [haninComplexityLinearRegions2019, NIPS2014_fa6f2a46, raghuExpressivePowerDeep2017]
datasets: [MNIST, random 2D points with binary labels]
---

# Deep ReLU Networks Have Surprisingly Few Activation Patterns

**Hanin, Rolnick (2019)**, *NeurIPS 2019*. [arXiv:1906.00904](https://arxiv.org/abs/1906.00904)

!!! abstract "TL;DR"
    A ReLU network partitions its input space into activation regions, one per on/off pattern of its neurons. Worst-case counts grow exponentially with depth, but this paper proves that, at a random parameter draw, the *expected* number of activation regions per unit volume is at most $(T\cdot\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$ — independent of depth — and shows experimentally that this bound is both tight and essentially unchanged by training, even on memorization tasks that should reward more regions.

## Problem and motivation

The expressivity argument for depth says that per parameter deep ReLU networks represent richer function classes than shallow ones, quantified by worst-case bounds on the number of activation patterns that grow exponentially in depth (§1). But what is learned in practice may fall far short of what is expressible; the paper contrasts the set $F_{\text{express}}$ of functions an architecture can represent, the set $F_{\text{init}}$ of functions at initialization, and the set $F_{\text{learn}}$ reachable by gradient descent (Fig. 1, p. 2). The question is where the typical trained network sits. Prior work is dominated by large *lower* bounds on the *maximum* region count for a fixed architecture; here the authors pursue *small upper* bounds on the *average* count for a typical parameter vector (§1, §1.1). Two concepts that the literature often conflates are separated: **activation regions**, which are convex, and **linear regions**, which need not be (§2.1).

## Method

The paper is a theory paper (Theorem 5, proved in Appendices B–C) reinforced by experiments that try to force the region count upward (§4).

*Definitions.* An activation pattern $A$ assigns a sign to every neuron; the corresponding activation region collects the inputs on which every pre-activation is on the correct side of its bias (Definition 1, §2.1). Activation regions are exactly the connected components left after the "bent hyperplanes" $H_z(\theta)=\{x:z(x;\theta)=b_z\}$, one per neuron, are removed (Lemmas 1–2). Linear regions are instead the components left after the discontinuity set $B_N(\theta)=\{x:\nabla N(\cdot;\theta)\text{ is discontinuous at }x\}$ is removed (Definition 2); the number of linear regions never exceeds the number of activation regions, and generically the two counts are close (Lemma 3, Lemma 8, App. B.3–B.4).

*Main theorem.* Theorem 5 bounds the expected number of non-empty activation regions intersecting a cube $C$ of side length $\delta$, per unit volume, by $(T\,\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$, under four distributional assumptions: the weights have a density, biases have conditional densities, pre-activation gradients have bounded moments ($C_{\mathrm{grad}}$), and bias densities are bounded ($C_{\mathrm{bias}}$) (§3.1). Because only these mild regularity conditions are needed, the bound applies to correlated weight/bias distributions and hence "throughout training" as well as at initialization; arbitrary connectivity between adjacent layers is allowed, only weight tying is excluded. Appendix C generalizes it to $r$-*partial* activation regions (Theorem 10, Eq. 9), where $r$ neurons are allowed to sit exactly on their hyperplane; the proof tracks the sets $X_{N,k}$ of inputs where exactly $k$ neurons are critical and integrates their $(n_{\mathrm{in}}-k)$-dimensional volume against the $k$-skeleton $C_k$ of $C$ (Eqs. 10–12).

*Two heuristics.* §3.2 gives a one-dimensional argument: each region boundary is a solution of $z(x)=b_z$, and a random level $b_z$ can be crossed many times only if the graph of $z$ oscillates over a large range, requiring large total variation; a unit-counting argument then raises the per-neuron rate to the power $n_{\mathrm{in}}$. §3.3 argues the sharpness is depth-independent: under the $2/\text{fan-in}$ variance normalization, the gradients of pre-activations in every layer have covariance $\mathrm{Cov}[\nabla z(x)] = 2\,\mathrm{Id}$ (Eq. 17 of [14], cited §3.3), so bent hyperplanes in every layer have the same normal-vector statistics and contribute equally, per neuron, to the region count.

## Key equations and definitions

Activation pattern and region (Definition 1, §2.1):

$$
\begin{aligned}
A &= \{a_z;\ z \text{ a neuron in } N\}\in\{-1,1\}^{\#\text{neurons}}, \\
R(A;\theta) &= \{x\in\R^{n_{\mathrm{in}}} \mid (-1)^{a_z}\bigl(z(x;\theta)-b_z\bigr) > 0,\ z \text{ a neuron in } N\}.
\end{aligned}
$$

Bent hyperplanes and the connectivity description (Eqs. 2–3, Lemma 2):

$$
\begin{aligned}
H_z(\theta) &:= \{x\in\R^{n_{\mathrm{in}}} \mid z(x;\theta) = b_z\}, \\
\{\text{non-empty } R(A;\theta)\} &= \text{connected components}\Bigl(\R^{n_{\mathrm{in}}}\setminus \bigcup_{\text{neurons } z} H_z(\theta)\Bigr).
\end{aligned}
$$

Arrangement count for one hidden layer in general position (Eq. 4, §2.2):

$$
\#\text{connected components} = \sum_{i=0}^{n}\binom{m}{i}\quad (m\ge n \text{ hyperplanes in } \R^n).
$$

Theorem 5 (Eq. 5, §3.1), under the four assumptions above with constants $C_{\mathrm{grad}},C_{\mathrm{bias}}>0$ and $T=T(C_{\mathrm{grad}},C_{\mathrm{bias}})$, for every cube $C$ of side length $\delta>\delta_0$:

$$
\begin{aligned}
\frac{\E\bigl[\#\text{non-empty activation regions of } N \text{ in } C\bigr]}{\mathrm{vol}(C)} &\le \frac{(T\,\#\text{neurons})^{n_{\mathrm{in}}}}{n_{\mathrm{in}}!}, \\
&\qquad \#\text{neurons}\ge n_{\mathrm{in}}.
\end{aligned}
$$

The two regularity constants come from (Conditions 3–4, §3.1):

$$
\sup_{x\in\R^{n_{\mathrm{in}}}}\E\bigl[\norm{\nabla z(x)}^m\bigr] \le C_{\mathrm{grad}}^m, \qquad
\sup_{b_1,\dots,b_k\in\R}\rho_{b_{z_1},\dots,b_{z_k}}(b_1,\dots,b_k) \le C_{\mathrm{bias}}^k .
$$

Generalization to $r$-partial regions (Theorem 10, Eq. 9, App. C):

$$
\begin{aligned}
&\frac{\E\bigl[\#r\text{-partial activation regions of } N \text{ in } C\bigr]}{\mathrm{vol}(C)}\\
&\quad \le
\begin{cases}
\dfrac{(T\,\#\text{neurons})^{n_{\mathrm{in}}}}{2^{\,r}\,n_{\mathrm{in}}!}, & \#\text{neurons}\ge n_{\mathrm{in}},\\[2mm]
2^{\#\text{neurons}}, & \#\text{neurons}\le n_{\mathrm{in}}.
\end{cases}
\end{aligned}
$$

The two volume bounds the proof rests on are, for bounded $S\subset\R^{n_{\mathrm{in}}}$ (Eq. 10) and the $k$-skeleton $C_k$ of $C$ (Eq. 12):

$$
\begin{aligned}
\frac{\E\bigl[\mathrm{vol}_{n_{\mathrm{in}}-k}(X_{N,k}\cap S)\bigr]}{\mathrm{vol}_{n_{\mathrm{in}}}(S)} &\le T^k\binom{\#\text{neurons}}{k}, \\
\#\{R(A;\theta)\cap C \neq \emptyset\} &\le \sum_{k=r}^{n_{\mathrm{in}}}\binom{k}{r}2^{k-r}\,\#V_k .
\end{aligned}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $n_{\mathrm{in}}$, $n_{\mathrm{out}}$ | input and output dimension ($n_{\mathrm{out}}=1$ in Theorem 5) |
| $N$, $\theta$ | ReLU net and its vector of trainable parameters |
| $z(x;\theta)$, $b_z$ | pre-activation and bias of neuron $z$ |
| $A$, $a_z\in\{-1,1\}$ | activation pattern and the sign it assigns to neuron $z$ |
| $R(A;\theta)$ | activation region of pattern $A$ |
| $H_z(\theta)$ | bent hyperplane $\{z(x;\theta)=b_z\}$ |
| $B_N(\theta)$ | discontinuity set of $\nabla N(\cdot;\theta)$ (linear-region boundaries) |
| $C$, $\delta$ | cube in $\R^{n_{\mathrm{in}}}$ of side length $\delta$ |
| $\#\text{neurons}$ | total number of neurons (all hidden layers) |
| $T$, $C_{\mathrm{grad}}$, $C_{\mathrm{bias}}$ | constants depending only on the gradient/bias bounds |
| $X_{N,k}$, $C_k$, $V_k$ | inputs with exactly $k$ critical neurons; $k$-skeleton of $C$; $X_{N,k}\cap C_k$ |

## Main results

**Theory.** For any feedforward ReLU net with no tied weights, random weights and biases satisfying the four conditions, the expected activation-region *density* is bounded by $(\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$ up to the constant $T$, with no dependence on depth (Theorem 5, Eq. 5). Remark 1 (§3.1) states the heuristic that the average total number of activation patterns over all of $\R^{n_{\mathrm{in}}}$ is at most $(\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$ — its exact value for a depth-1 network, by Eq. 4. At initialization, with independent biases, $C_{\mathrm{bias}}$ is just the maximum bias density; for wide fully connected nets with variance $2/\text{fan-in}$ weights, $C_{\mathrm{grad}}$ is bounded by a function of $\sum_j 1/n_j$ (the reciprocals of hidden widths), so it is small for wide networks and "bounded as soon as the network is stable enough to begin training" (§3.1). At initialization the paper expects depth-independent *matching lower bounds* to leading order, so that two ReLU nets with the same number of neurons should have the same average region count; the effect of depth, they conjecture, appears in the *variance* rather than the mean (§2.2, §3.3).

**Scaling identities.** Multiplying all biases by $c$ gives $N(x)=N^{c\,\mathrm{bias}}(cx)/c$, so uniform bias rescaling leaves the global region count unchanged (Lemma 6, §4.2). Multiplying all weights by $c$ is equivalent to rescaling the biases of every layer differently, $N^{c\,\mathrm{weight}}(x)=c^{d}N^{1\,\mathrm{bias}/c}(x)$ (§4.2).

**Experiments (§4).** Regions are counted in 1D/2D cross-sections by adding neurons one at a time and splitting the regions each neuron cuts (App. A); training uses Adam, learning rate $10^{-3}$, batch 128, i.i.d. normal weights of variance $2/\text{fan-in}$ and biases of variance $10^{-6}$ unless stated. MNIST, depth-3 networks: for a $[20;20;20]$ net the 2D cross-section count starts at approximately $(\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$ as predicted, and over 20 epochs (averaged over 10 runs, 5 cross-sections each) it "first decreas[es] slightly and then rebound[s], but never increas[es] exponentially" (Fig. 3). A depth-3 width-32 net (96 neurons) trained on MNIST with varying label corruption shows the region count along lines predicted to be ≈96, rising only slightly with the amount of memorization (Fig. 4, averaged over 40 runs and 100 lines). Memorizing random 2D points with binary labels: Theorem 5 predicts $96^2/2! = 4608$ regions, and all runs start there; the count increases with memorization but only by a small constant factor, and the hardest setting fails to learn at all (Fig. 5). Varying the learning rate ($10^{-3}$ maximal, $10^{-2}$ too large, causing failure) and the initialization scale changes the count only mildly (Fig. 6). A depth-3 width-64 net on MNIST shows the same behaviour (§4, Fig. 7).

## Limitations stated by the authors

The result is stated for ReLU nets with no tied weights and arbitrary connectivity; the authors believe analogous results hold for residual and convolutional networks but "have not verified the technical details" (§5). Theorem 5 needs the bounded-gradient and bounded-bias-density conditions 3–4, which exclude identically zero biases as stated; Appendix D treats that case separately, showing via scale equivariance that a zero-bias net has *no more* regions than the same net with biases (Proposition 14), and arguing informally that the count should then grow like $n_{\mathrm{in}}(\#\text{neurons})^{n_{\mathrm{in}}-1}$ rather than $n_{\mathrm{in}}(\#\text{neurons})^{n_{\mathrm{in}}}$ (App. D, p. 19). There is "no *a priori* reason" that $C_{\mathrm{grad}}$ and $C_{\mathrm{bias}}$ stay bounded during training; this is supported only by numerical experiment, with a conjecture that they remain bounded for wide nets governed by the neural tangent kernel (§3.1). The bound is on the *local*, per-unit-volume count rather than the global count, and on the expectation, not on individual draws (§3.1, §4.2). The paper does not claim to influence architecture selection (§5).

## Critical assessment

This is the reference point for the "practical vs. theoretical expressivity" gap in the region-counting literature, and its strength is precisely that the bound needs only mild distributional assumptions, which lets it survive training. Two caveats matter when reading it. First, the central object — the number of activation regions as a function of $\theta$ — is a random variable, and Theorem 5 constrains only its mean; the authors themselves guess depth shows up in the variance, but prove nothing there, so "depth does not increase complexity" is a statement about averages only. Second, the constants $T$, $C_{\mathrm{grad}}$, $C_{\mathrm{bias}}$ are not effective: the bound can be vacuous if the network is not stable, and the experiments that support the "throughout training" message are all small (depth 3, width 32–64) and use a counting procedure in cross-sections, which is a proxy for the full region count, not the count itself. The memorization experiments are the right stress test, but they too are tiny — random 2D binary-label tasks and corrupted MNIST labels — and the observed increases (a "small constant factor") are read off plots rather than tabulated. The scale-equivariance discussion (Lemma 15, Proposition 14) is a genuinely useful structural remark that is not widely known, and the authors flag its main claim as weak. I see no code or data release mentioned, and Appendix A describes the counting algorithm only in prose, so reproduction is possible but not turnkey. None of this undermines the theorem; it limits how far the "learned nets are simple" reading can be pushed from this evidence alone.

## Connections within the collection

Theorem 5 was conjectured, and weaker average-case bounds proved, in the companion ICML paper [@haninComplexityLinearRegions2019], the closest relative here: that paper supplies the convergence-in-distribution machinery for region counts, while this one supplies the expectation bound and the experiments. The worst-case counts it contrasts against are those of [@NIPS2014_fa6f2a46] (the depth-dependent maximum number of linear regions, its reference [25]) and the exponential expressivity of [@raghuExpressivePowerDeep2017] (its [29]); the bounding-and-counting constructions of [@serraBoundingCountingLinear2018] (its [32]) are the algorithmic counterpart of its maximum-side bounds. The scale-equivariance property of zero-bias ReLU networks (Lemma 15) is a special case of the homogeneity that also underlies tropical-geometry treatments such as [@zhangTropicalGeometryDeep] and [@alfarraDecisionBoundariesNeural2023], which describe the same max-plus structure through polyhedral/tropical lenses.

Within the collection's region-counting work, it sits with [@goujonNumberRegionsPiecewise2023], which generalizes the average-case analysis to arbitrary continuous piecewise-linear activations, and with the early response-region count of [@pascanuNumberResponseRegions2013] on the combinatorial side. The distinction it draws between convex activation regions and non-convex linear regions reappears in the extraction and enumeration methods of [@masdenAlgorithmicDeterminationCombinatorial2022], [@berzinsPolyhedralComplexExtraction2023] and [@liuReLUNeuralNetworks2023], and in the exact visualization of [@humayunSplineCamExactVisualization2024], all of which compute the very partition this paper bounds statistically. The spline/compositional view of [@balestrieroSplineTheoryDeep2018] and the bottleneck/decision-boundary analyses of [@brandenburgDecompositionPolyhedraPiecewise2024] and [@liuIntegratingGeometriesReLU2023a] study, from other angles, the same question of how many pieces a trained ReLU network actually uses. The forward/backward stability results the paper leans on for the boundedness of $C_{\mathrm{grad}}$ (its references [14]–[16]) are outside the collection.
