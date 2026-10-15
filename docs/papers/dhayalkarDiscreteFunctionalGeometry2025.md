---
citekey: dhayalkarDiscreteFunctionalGeometry2025
title: "Discrete Functional Geometry of ReLU Networks via ReLU Transition Graphs"
authors: [Sahil Rajesh Dhayalkar]
year: 2025
venue: "2025 5th International Conference on Robotics, Automation, and Artificial Intelligence (RAAI)"
doi: "10.1109/RAAI67517.2025.11423317"
arxiv: "2505.11692"
url: null
kind: theory
source_type: pdf
source_sha256: c55459e00ab0288c6118d429dcbc4a749981c405c2c68d4fae10f8fd1255ccea
pages: 13
ingested: 2026-10-15
tags: [relu-networks, linear-regions, graph-theory, expressivity, vc-dimension, pruning]
concepts_introduced:
  - relu transition graph
  - rtg entropy
  - rtg size bound
  - rtg connectivity
  - rtg diameter bound on vc-dimension
  - rtg degree-sparsity
  - functional compression via rtg pruning
concepts_used:
  - activation pattern
  - linear region
  - relu network
  - piecewise linear function
  - hyperplane arrangement
  - vc-dimension
  - shannon entropy
  - hamming distance
builds_on: [NIPS2014_fa6f2a46, zaslavskyFacingArrangementsFacecount1975]
datasets: [synthetic 2D uniform grid]
---

# Discrete Functional Geometry of ReLU Networks via ReLU Transition Graphs

**Dhayalkar (2025)**, *2025 5th International Conference on Robotics, Automation, and Artificial Intelligence (RAAI)*. [arXiv:2505.11692](https://arxiv.org/abs/2505.11692)

!!! abstract "TL;DR"
    The paper defines the *ReLU Transition Graph* (RTG): an undirected graph whose
    nodes are the linear regions of a ReLU network and whose edges join regions that
    differ by exactly one active neuron. On top of it four definitions, four theorems
    and three lemmas claim bounds on RTG size, entropy, diameter, VC-dimension and
    prunability, each "validated" by a PyTorch experiment on synthetic 2D grids.
    Theorem 1 and Lemma 1 restate known hyperplane-arrangement facts; Theorem 3
    (VC-dimension $\le$ RTG diameter) is the genuinely new-looking claim, and its
    proof is the weakest part of the paper.

*This summary was written from the arXiv preprint [arXiv:2505.11692v2](https://arxiv.org/abs/2505.11692) (28 May 2025), because the attachment in this collection is a paywalled IEEE Xplore landing page without full text. The published version is the RAAI 2025 paper of record, pp. 153–159, DOI 10.1109/RAAI67517.2025.11423317. The two do not read as the same text: the `.bib` abstract of the published paper announces "strong expansion, binomial degree distributions, and spectral properties", "bounds on generalization via spectral gap and edge-wise KL divergence", "region entropy saturates under overparameterization" and a framework for "diagnos[ing] … generalization" — none of which appears in the preprint summarised here, whose results are the size bound, connectivity, the entropy-versus-degree inequality, the diameter/VC bound and pruning. Everything below describes the preprint only; the published paper could not be read. Locators are the preprint's own (§4.2, Thm. 2, Fig. 1, Tab. 1).*

## Problem and motivation

Region counting quantifies a ReLU network's expressivity by the number of linear pieces it induces, and bounds on that count are the classical toolkit (§1, p. 1). The paper's argument is that counting says nothing about how those regions are *organised*: "much less is known about the global structure connecting these regions—how they transition, cluster, and contribute to the network's functional topology" (§1, p. 1). The proposed object is therefore not a count but a graph over the regions, with edges given by minimal transitions — a single neuron flipping state (§1, p. 1). Setting: fully connected feedforward ReLU networks of depth $L$ and width $n$ per layer, $f : \R^d \to \R^k$ (§3, p. 3), with the empirical work restricted to $d = 2$ (§5.1, p. 6).

## Method

The method is almost entirely definitional and combinatorial; the empirical half is a sampling exercise.

1. **Definitions** (Defs. 1–4, §3.1–§3.4, p. 3): activation pattern $a(x)$, linear region $R_a$, the RTG itself, and the entropy of the region-volume distribution. Definitions 1 and 2 are the standard activation-pattern/PWL-region pair.
2. **Structural results** (§4, pp. 4–5): a size bound for $|V|$ (Thm. 1), a Hamming-1 adjacency lemma (Lem. 1), connectivity of the whole graph (Thm. 2), an entropy lower bound in terms of average degree (Lem. 2), a VC-dimension bound by graph diameter (Thm. 3), a degree-sparsity lemma (Lem. 3) and a compression theorem derived from it (Thm. 4). Proofs are deferred to Appendix A.1–A.7.
3. **Construction from samples** (§5.1, p. 6): evaluate the network on $100{,}000$ points of a uniform grid in $[-1,1]^2$, record the binary activation pattern $a(x) \in \{0,1\}^{nL}$ of every point, take the set of distinct patterns as nodes and join patterns at Hamming distance exactly 1. Where the resulting graph is disconnected, statistics are reported for the largest connected component.
4. **Validation** (§5.2–§5.3.5, pp. 6–8): each result is checked against a deliberately simple statistic — node counts against the bound (Fig. 1, Tab. 1), sampled Hamming-1 pairs against an edge count, connected components per seed, $H(G)$ against $\log(d_{\mathrm{avg}}+1)$ (Fig. 2), a "VC proxy" against the diameter (Fig. 3), the fraction of nodes with $\deg(v) \le 2 d_{\mathrm{avg}}$, and finally the sup-norm error after deleting the lowest-degree half of the nodes and smoothing (Fig. 4).

Depths $L \in \{2,3,4\}$, widths $n \in \{4,8,16,32,64,128\}$, five seeds per configuration, mean with standard deviation and 95\% Student-$t$ intervals (§5.1, p. 6).

## Key equations and definitions

The region of an activation pattern, and RTG adjacency as shared facet (Def. 2, 3, p. 3):

$$
R_a = \{x \in \R^d \mid a(x) = a\}, \qquad (v_i,v_j) \in E \iff \dim\bigl(\overline{R_{a_i}} \cap \overline{R_{a_j}}\bigr) = d-1
$$

Region-volume distribution and RTG entropy (Def. 4, p. 3):

$$
p_v = \frac{\operatorname{Vol}(R_v)}{\sum_{u \in V} \operatorname{Vol}(R_u)}, \qquad H(G) = -\sum_{v \in V} p_v \log p_v
$$

The RTG size bound of Theorem 1 (§4.1, p. 4; proved in App. A.1, p. 10):

$$
|V| \le \left(\sum_{i=0}^{d} \binom{n}{i}\right)^{\!L}
$$

Entropy grows with average degree — the only inequality relating a graph quantity to a geometric one (Lem. 2, §4.4, p. 5; proved in App. A.4, p. 12):

$$
H(G) \ge \log(d_{\mathrm{avg}} + 1)
$$

VC-dimension is bounded by RTG diameter (§4.5, p. 5; proved in App. A.5, p. 12):

$$
\operatorname{VC}(\mathcal{F}) \le \operatorname{diam}(G)
$$

Degree sparsity: at least half the nodes are within twice the average degree (Lem. 3, §4.6, p. 5; proved in App. A.6, p. 12):

$$
S = \{v \in V \mid \deg(v) \le 2 d_{\mathrm{avg}}\}, \qquad |S| \ge \tfrac{1}{2}|V|
$$

Compression: replacing the pruned regions by a linear interpolation of neighbouring region outputs is claimed to cost at most $\delta$ in sup norm, while shrinking the graph by a factor $1-\alpha$ (Thm. 4 and App. A.7, pp. 5, 13):

$$
\begin{aligned}
f_{\mathrm{core}}(x) &= f(x)\,\mathbf{1}\!\left[x \notin \bigcup_{v \in S} R_v\right] + \tilde{f}(x)\,\mathbf{1}\!\left[x \in \bigcup_{v \in S} R_v\right], \\
\|f - f_{\mathrm{core}}\|_\infty &\le \delta, \\
|V_{\mathrm{core}}| &\le (1-\alpha)|V|
\end{aligned}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $d$, $k$ | input dimension and output dimension of $f$ |
| $L$, $n$ | number of hidden layers; width (ReLU neurons) per hidden layer |
| $m$ | total number of ReLU units, $m = nL$ |
| $a(x) \in \{0,1\}^m$ | activation pattern: $a_i(x) = \mathbf{1}[z_i(x) > 0]$ for pre-activation $z_i$ |
| $R_a$ | linear region of pattern $a$; a convex polyhedron on which $f$ is affine |
| $G = (V,E)$ | ReLU Transition Graph; $v \in V$ a region, $E$ the Hamming-1 adjacency |
| $d_{\mathrm{avg}}$ | average degree of the nodes of $G$ |
| $p_v$, $\operatorname{Vol}(R_v)$ | normalised region volume; volume of region $R_v$ |
| $H(G)$ | RTG entropy of the volume distribution |
| $\mathcal{F}$ | function class realised by the network |
| $\operatorname{VC}(\mathcal{F})$, $\operatorname{diam}(G)$ | VC-dimension; longest shortest path in $G$ |
| $S$, $\alpha$, $\delta$ | set of low-degree nodes; pruning fraction; error tolerance |
| $\tilde{f}$, $f_{\mathrm{core}}$, $B$ | interpolation between surviving region outputs; compressed function; negligible boundary set |

## Main results

- **Size bound (Thm. 1, §4.1, p. 4).** The stated bound is the one for an arrangement of $n$ hyperplanes in $\R^d$ raised to the power $L$, i.e. the per-layer product of the Zaslavsky count. The paper says so explicitly, and credits the arrangement theory and its ReLU application to prior work; its own contribution is described as "refram[ing] this combinatorial result within the RTG framework".
- **Adjacency (Lem. 1, §4.2, p. 4).** Two activation patterns differing in one bit give regions sharing a $(d-1)$-dimensional facet. Also presented as a restatement of a known property, cited. Validation: 100/100 sampled Hamming-1 pairs connected and 0/100 sampled Hamming-$>$1 pairs connected, in every configuration and seed (§5.3, p. 6) — with the caveat that the graph was *built* from the Hamming-1 rule, so the first number is a property of the code, not of the geometry.
- **Connectivity (Thm. 2, §4.3, p. 4).** For a continuous ReLU network with non-degenerate hyperplanes the RTG is connected. Empirically 1 component for all $(L,n)$ except $(L{=}4,n{=}128)$, which averaged $4.40$ components (std $2.06$, CI95 $1.80$) across seeds (§5.3.1, p. 7); the authors attribute this to finite-sample sampling sparsity under heavy overparameterisation.
- **Entropy (Lem. 2, §4.4, p. 5).** $H(G) \ge \log(d_{\mathrm{avg}}+1)$, checked by plotting $H(G)$ against $\log(d_{\mathrm{avg}}+1)$ for all configurations (Fig. 2, p. 7). The experiment computes $H(G) = \log|V|$ "assuming a uniform region volume distribution" (§5.3.2, p. 7), i.e. the opposite of Definition 4, so Fig. 2 does not test the quantity the entropy definition is about.
- **VC-dimension (Thm. 3, §4.5, p. 5).** The claim is a graph-theoretic upper bound on capacity. Validation uses a proxy, "the number of distinct activation patterns observed over 10 randomly sampled input points" (§5.3.3, p. 7), plotted against the diameter for each configuration; all points lie below the diagonal (Fig. 3, p. 7). A proxy bounded by ~10 cannot be compared to diameters that grow with $|V|$ in a way that tests the theorem.
- **Sparsity (Lem. 3, §4.6, p. 5).** At least half of the nodes have degree at most $2 d_{\mathrm{avg}}$. Measured: 100\% of nodes satisfy the condition in all configurations and seeds (mean $1.00$, std $0.00$, CI95 $0.00$) (§5.3.4, p. 8). The proof is a two-line averaging argument and the measurement is the Markov-style statement itself, so both are tautological.
- **Compression (Thm. 4, §4.7, p. 5).** Pruning the lowest-degree 50\% of nodes and smoothing over the pruned regions is claimed to leave the function within $\delta$ in sup norm. Reported errors: $0.154$ at $(L{=}2,n{=}64)$ and $0.127$ at $(L{=}4,n{=}32)$, printed only in Fig. 4 without a table (§5.3.5, p. 8). Note the mismatch of quantities: the experiment prunes *regions* (50\% of nodes) and reports an output-space error, whereas the theorem bounds the number of nodes removed by $(1-\alpha)|V|$ with $\alpha$ "depending on $d_{\mathrm{avg}}$ and the geometry of region overlaps" — a free parameter, never instantiated.
- **Size numbers (Tab. 1, App. A.8, p. 13).** Mean $|V|$ rises with width and depth: $15.5$ at $(L{=}2,n{=}4)$, $4399.0$ at $(L{=}2,n{=}128)$, $6136.0$ at $(L{=}3,n{=}128)$, $7062.5$ at $(L{=}4,n{=}128)$, all far below the printed bounds ($5.3\times10^1$ up to $4.36\times10^{15}$). The table is referred to as "Table 1 (provided in Appendix A.9)" in §5.2 (p. 6) but sits in App. A.8 (sic).

## Limitations stated by the authors

Restricted to fully connected ReLU MLPs and synthetic low-dimensional inputs, which "may not fully capture complexities in high-dimensional or convolutional networks"; the entropy estimates assume uniform region volumes, which "could oversimplify real data distributions" (§7, p. 8). Extending RTG construction to convolutional, residual and attention models and to real datasets is listed as future work (§6, §7).

## Critical assessment

My reading. The RTG is a reasonable indexing device — it is exactly the region-adjacency graph that the polyhedral-decomposition literature builds — and naming it, together with the entropy of the region-volume distribution, is a legitimate contribution. As a paper it is however much weaker than its framing. Three problems are structural rather than cosmetic.

First, **Theorem 1 is not a new bound and its proof is invalid**. The proof in App. A.1 (p. 10) starts from the correct product form $|V| \le (\sum_{i=0}^{d}\binom{n}{i})^L$ and then "loosens this slightly" by the printed inequality $(\sum_i a_i)^L \le \sum_i a_i^L$ for "non-negative terms" — which is false in the direction stated for $L \ge 2$ and more than one non-zero term (the multinomial expansion gives the reverse). The numerical column confirms which form was used: at $(L{=}2,n{=}4)$ it prints $5.3\times10^1 = 1+4^2+6^2$, whereas the correct product form gives $11^2 = 121$. So the table compares the experiments against a quantity that is *not* an upper bound; the conclusion "all empirical values remain below the bound" rests on the wrong side of a wrong inequality. The proof heading is also wrong — "Proof of Theorem 1: Existence of FSM-Emulating Neural Networks" (p. 10) is a copy-paste from another manuscript of the author.

Second, **Theorem 3's proof is a sketch presented as a proof**. The argument (App. A.5, p. 12) claims that a $D$-ball of the RTG contains at most $\sum_{k=0}^{D}\binom{nL}{k}$ activation patterns, is "strictly less than $2^N$ unless $D \ge N$", and concludes $\operatorname{VC}(\mathcal{F}) \le \operatorname{diam}(G)$. The intermediate step is not established: the ball count is a crude bound on reachable patterns, the "see Sauer's Lemma" appeal is not carried out, and no relation is derived between $N$ (the number of shattered points) and the pattern count. Since VC-dimension for a ReLU network of $L$ hidden layers of width $n$ is $\Theta(nL\log(nL))$-ish while $\operatorname{diam}(G)$ is at least the RTG dimension-like quantity, the inequality may be true, but this text does not show it. The empirical "validation" cannot rescue it either: a proxy that is at most 10 by construction is compared to a diameter that grows into the thousands.

Third, **three of the six "validations" are tautologies**. Lemma 1's edge test asks the graph whether it contains the edges it was built from. Lemma 3's test measures the same averaging inequality that the proof proves. Lemma 2's test replaces the volume distribution with the uniform one, in which the entropy is $\log|V|$ by definition, so $H(G) \ge \log(d_{\mathrm{avg}}+1)$ becomes a statement about average degree versus node count rather than about region volumes. The remaining two are weak: the $(L{=}4,n{=}128)$ disconnection is dismissed as sampling, which is plausible, but the sampling claim is not tested (raising the grid density would settle it); and Theorem 4 is checked against a different quantity than it is stated for.

None of this makes the object useless — the RTG appears in later work on region adjacency and the experiments do produce the rough scaling behaviour one expects — but the theorem/validation structure should not be read as establishing the claimed bounds. Code and data are not released, and no repository is mentioned; the experiments are small enough that re-running them would cost little, which is the redeeming feature.

## Connections within the collection

The size bound of Theorem 1 is the Zaslavsky arrangement count composed over layers, which is precisely the result of Montúfar et al. [@NIPS2014_fa6f2a46] for ReLU networks, and the paper's own text says its bound came from that line of work; the face-count formula behind it is Zaslavsky's [@zaslavskyFacingArrangementsFacecount1975], whose summary is unavailable in this collection (no readable attachment) and which is cited here as [20]. Lemmas 1 and 2 are stated as restatements of known adjacency and counting behaviour credited to Montúfar et al. and to Hanin and Rolnick [@haninComplexityLinearRegions2019] (also cited, but not used as a component). Serra et al. [@serraBoundingCountingLinear2018] and the region-topology work of Raghu et al. [@raghuExpressivePowerDeep2017] appear only as related work on counting and on the role of depth.

The RTG itself is the same object that the exact polyhedral-decomposition line of the collection constructs and reasons about: Masden's combinatorial determination of the linear regions [@masdenAlgorithmicDeterminationCombinatorial2022], Vincent and Schwager's reachable polyhedral marching [@vincentReachablePolyhedralMarching2021], Berzins' edge-subdivision extraction [@berzinsPolyhedralComplexExtraction2023], the persistent-homology view of the polyhedral complex [@liuReLUNeuralNetworks2023] and SplineCam's exact visualisation [@humayunSplineCamExactVisualization2024] all operate on region adjacency, and Brandenburg et al.'s decomposition polyhedra [@brandenburgDecompositionPolyhedraPiecewise2024] and Lewandowski et al.'s space-folding picture [@lewandowskiSpaceFoldsReLU2024; @lewandowskiExploitingSpaceFolding2026] are the closest structural neighbours for the transition-between-regions viewpoint. The counting-versus-adjacency distinction that motivates the paper is exactly the empirical gap Hanin and Rolnick document [@haninDeepReLUNetworks2019]. These are families of related objects, not results this preprint uses; the preprint's own bibliography contains none of them, so the relations are of the "same object, different literature" kind rather than dependencies. The pruning-based compression claim is a graph-restricted analogue of functional-equivalence/pruning results such as the functional-vs-parametric equivalence of Phuong and Lampert [@phuongFunctionalVsParametric2019]; possibly related, and the preprint does not cite it.
