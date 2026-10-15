---
citekey: lewandowskiSpaceFoldsReLU2024
title: "On Space Folds of ReLU Neural Networks"
authors: [Michal Lewandowski, Hamid Eghbalzadeh, Bernhard Heinzl, Raphael Pisoni, Bernhard A. Moser]
year: 2024
venue: Transactions on Machine Learning Research
doi: null
arxiv: null
url: null
kind: method
source_type: pdf
source_sha256: f6fd21157162e0fda916a25e1072fce5fcaff827e0e6a7003c45d290029568f3
pages: 18
ingested: 2026-10-15
tags: [relu-networks, space-folding, activation-space, hamming-distance, linear-regions]
concepts_introduced:
  - space folding measure
  - activation space convexity
  - Hamming activation space
  - observable activation pattern
concepts_used:
  - ReLU neural network
  - activation pattern
  - linear region
  - Hamming distance
  - binarization
  - range measure
builds_on: [NIPS2014_fa6f2a46, raghuExpressivePowerDeep2017, keupOrigamiDimensionsHow2022, phuongFunctionalVsParametric2019]
datasets: [CantorNet, MNIST]
---

# On Space Folds of ReLU Neural Networks

**Lewandowski, Eghbalzadeh, Heinzl, Pisoni, Moser (2024)**, *Transactions on Machine Learning Research*.
<!-- Printed as "Published in Transactions on Machine Learning Research (02/2025)"; the .bib gives year 2024. No DOI or arXiv id in the .bib or on the PDF. -->

!!! abstract "TL;DR"
    ReLU layers are often described as "folding" the input space, but until now that description was only qualitative. The authors make it quantitative: they map straight paths from Euclidean input space into the discrete Hamming activation space and measure how much the Hamming distance to the start point can decrease along the way. The resulting *space folding measure* $\chi \in [0,1]$ is zero for a straight image path and grows with folding; a lemma shows folding is unobservable in one-hidden-layer networks. On CantorNet and MNIST the measure grows with depth for well-trained nets, and their fraction of folding paths jumps from ≈0.35 to ≈0.99 in larger networks.

## Problem and motivation

Successive ReLU layers have been interpreted as folding operators that repeatedly bend the input space (Montúfar et al., 2014; Raghu et al., 2017; Keup and Helias, 2022), but this is only an intuition: "the concept of space folding by neural networks remains largely qualitative, with no prior work attempting to quantify these effects" (§1, p. 2). The paper's premise is that folding is hard to see in the continuous input space, but becomes observable in the *discrete activation space* — the binary vectors of neuron activation signs — where linear regions become points of a Hamming cube (§1, p. 2).

The setting: a ReLU network $N : X \to Y$ with $N$ hidden neurons, a straight line in the Euclidean input space, and the sequence of activation patterns along it. In Euclidean space a straight walk monotonically increases the distance to the start; in the Hamming activation space this monotonicity can fail, and that failure is exactly what is measured (§1, p. 2; §5, p. 6). The paper also positions itself relative to distance-alteration work (Lipschitz constants, length distortion), which it says "do[es] not touch on the activation space" nor on the monotonicity of a mapped straight line (§2, p. 3).

## Method

**Binarization and activation space.** The ReLU $\sigma(x) := \max(x, 0)$ is applied element-wise; the concatenation of all hidden-layer activations is binarized by $\pi(v_i) = 1$ if $v_i > 0$ and $0$ otherwise, giving an *activation pattern* in the Hamming hypercube $H_N := \{0,1\}^N$ (§3, p. 4). A *linear region* is a cell of the partition on which the network is affine (§3, p. 4).

**Convexity transfer.** Definition 1 calls a subset of the Hamming cube convex if every shortest path between two of its points stays in it. Lemma 1 proves that a union of activation regions is convex in $\R^n$ **iff** the set of their activation patterns is convex in the Hamming space — for hyperplanes that cut the whole input space. This is the paper's key structural result: it justifies that a straight input path corresponds to a shortest path in the Hamming space, and it implies that folding effects are invisible in networks with a single hidden layer (§4, pp. 5–6).

**The measure.** Sample $n$ points $(1-\lambda_i)\mathbf{x}_1 + \lambda_i \mathbf{x}_2$ on the straight segment, propagate each to an activation pattern, and form the path $\Gamma = (\pi_1, \dots, \pi_n)$. Two range measures are tracked: the maximum Hamming distance reached from the start, $r_1(\Gamma)$, and the total distance travelled on the hypercube, $r_2(\Gamma)$. Their ratio is the *flatness* $\phi(\Gamma) = r_1/r_2$, and the space folding measure is $\chi(\Gamma) := 1 - \phi(\Gamma)$ (Eq. 4). $\chi = 0$ iff the image path is straight (no folding); larger $\chi$ means more folding. Lemma 2 proves $0 \le \chi \le 1$ (for paths crossing more than one region). The *global* version $\Phi_N := \sup_{\Gamma \in X} \chi(\Gamma)$ (Eq. 5) is defined but not computed.

**Algorithm.** Alg. 1 computes $\chi$ in $O(n \cdot (N + C))$ per pair, where $O(C)$ is one inference pass. Since the number of linear regions crossed by a segment is at most the $N$ hidden neurons, $n$ need not exceed $N$. For two classes of sizes $|C_1|, |C_2|$ the total cost is $O(n \cdot (N + C) \cdot |C_1| \cdot |C_2|)$ (Eq. 6); the authors propose $k$-means clustering of each class to centroids to reduce it (§5, pp. 7–9).

## Key equations and definitions

$$
\Delta_i := d_H(\pi_{i+1}, \pi_1) - d_H(\pi_i, \pi_1) \qquad \text{(Eq. 1, p. 6)}
$$

The step-by-step change in Hamming distance to the start point; negative values signal folding.

$$
r_1(\Gamma) = \max_{i \in \{1,\dots,n-1\}} \sum_{j=1}^{i} \bigl(d_H(\pi_{j+1}, \pi_1) - d_H(\pi_j, \pi_1)\bigr) = \max_{i \in \{1,\dots,n-1\}} d_H(\pi_i, \pi_1) \qquad \text{(Eq. 2, p. 7)}
$$

Maximum Hamming distance to the start reached along the path (a range/diameter measure).

$$
r_2(\Gamma) = \sum_{i=1}^{n-1} d_H(\pi_i, \pi_{i+1}) \qquad \text{(Eq. 3, p. 7)}
$$

Total distance travelled on the hypercube.

$$
\chi(\Gamma) := 1 - \phi(\Gamma) = 1 - \frac{\max_{i \in \{1,\dots,n\}} d_H(\pi_i, \pi_1)}{\sum_{i=1}^{n-1} d_H(\pi_i, \pi_{i+1})} \qquad \text{(Eq. 4, p. 7)}
$$

The space folding measure; the deviation from convexity of the image of a straight line.

$$
\Phi_N := \sup_{\Gamma \in X} \chi(\Gamma) \qquad \text{(Eq. 5, p. 7)}
$$

Global folding. Note the index ranges differ between Eq. 2 ($i \in \{1,\dots,n-1\}$) and Eq. 4 ($i \in \{1,\dots,n\}$); this appears to be a small inconsistency in the printed paper, harmless because the last pattern only adds a term.

$$
O\bigl(n \cdot (N + C) \cdot |C_1| \cdot |C_2|\bigr) \qquad \text{(Eq. 6, p. 8)}
$$

Total cost of evaluating the measure over all pairs of two classes.

**CantorNet** (App. A, pp. 15–16) is defined by the generating function $A(x) := \max\{-3x+1, 0, 3x-2\}$ nested as $A^{(k+1)}(x) := A(A^{(k)}(x))$, with decision region

$$
R_k := \{(x,y) \in [0,1]^2 : y \le (A^{(k)}(x)+1)/2\} \qquad \text{(Eq. 7, p. 15)}
$$

and realized as the zero-preimage of a ReLU network with $W_1 = \begin{pmatrix} -3 & 0 \\ 3 & 0 \\ 0 & 1\end{pmatrix}$, $b_1 = \begin{pmatrix} 1 \\ -2 \\ 0\end{pmatrix}$, $W_2 = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 0 & 1\end{pmatrix}$, $W_L = \begin{pmatrix} -\tfrac12 & 1\end{pmatrix}$, $b_L = -\tfrac12$ (Eq. 8, p. 15). Kanji-free transcription from the extracted text; the matrix layout is as printed.

| Symbol | Meaning (in this paper) |
|---|---|
| $N$ | ReLU network; also the number of hidden neurons |
| $X, Y$ | input and output spaces of $N$ |
| $\sigma$ | ReLU, $\max(x,0)$, element-wise |
| $W_k, b_k$ | weights and biases at layer $k$ |
| $\pi$ | binarization map; also an activation pattern in $H_N$ |
| $H_N$ | Hamming hypercube $\{0,1\}^N$ |
| $d_H(u,v)$ | Hamming distance, $\lvert\{u_i \ne v_i\}\rvert$ |
| $R_\pi$ | linear region with activation pattern $\pi$ |
| $\Gamma$ | path $(\pi_1, \dots, \pi_n)$ in the Hamming activation space |
| $\lambda_i$ | interpolation weights on the input segment |
| $r_1, r_2$ | range measures: max distance from start; total path length |
| $\phi$ | space flatness $r_1/r_2$ |
| $\chi$ | space folding measure $1-\phi$ |
| $\Phi_N$ | global folding, $\sup_\Gamma \chi(\Gamma)$ |
| $n$ | number of sampled intermediate points |
| $C$ | cost of one network inference pass, $O(C)$ |
| $C_1, C_2$ | classes of the two endpoints |
| $A^{(k)}$ | CantorNet generating function at recursion depth $k$ |

## Main results

**Theoretical.** Lemma 1 (equivalence of convexity in input and Hamming spaces for full-space hyperplanes; proof given, §4, pp. 5–6) and Lemma 2 ($0 \le \chi \le 1$; proof given, §5, p. 7). The direct consequence of Lemma 1 is that "space folding effects are observable only in networks with at least two hidden layers", and it gives a second reading of the XOR problem (§4, p. 6).

**CantorNet.** For recursion level $k=1$, the path $\Gamma : \pi_6 \to \pi_5 \to \pi_4$ has $\chi(\Gamma) = 0$ if only first-layer activations are used but $\chi(\Gamma) = \tfrac12$ when all layers are included (§6.1, p. 9). For $k=2$ on the segment between $(0, \tfrac34)$ and $(1, \tfrac34)$, $\chi(\Gamma) \approx 0.7$ (§6.1, p. 9; Fig. 6, p. 10). In the discussion the authors quote CantorNet values "close to 0.8" (§7, p. 11) — slightly higher than the $\approx 0.7$ of the worked path, so the two numbers come from different paths.

**MNIST, fixed 60 hidden neurons.** Architectures $2{\times}30, 3{\times}20, 4{\times}15, 5{\times}12, 6{\times}10, 10{\times}6$ (layers×neurons), trained 30 epochs. A single hidden layer is deliberately omitted because Lemma 1 predicts no folding (§6.1, p. 9, footnote 2). For the best seed (4), the Pearson correlation between depth and the aggregated median of maxima of non-zero folding across the 100 digit pairs is $0.987$ (§6.1, p. 9). Networks that failed to reach high validation accuracy showed much lower aggregated folding, which the authors read as "the symmetries that arise with depth help with achieving some generalization ability" (§6.1, p. 10).

**Larger networks.** For $2{\times}300$ and $3{\times}200$, folding *values* barely change, but the ratio of paths that exhibit folding rises from $0.35 \pm 0.1$ ($2{\times}30$) and $0.44 \pm 0.15$ ($3{\times}20$) to $0.97 \pm 0.04$ ($2{\times}300$) and $0.99 \pm 0.02$ ($3{\times}200$) (§6.1, p. 10; App. D, p. 18). (App. D, p. 18 mislabels the deeper architectures as "$2{\times}30$ and $3{\times}20$" (sic; likely $2{\times}300$ and $3{\times}200$, as in §6.1).)

## Limitations stated by the authors

- The measure is expensive; the choice of evenly spaced intermediate points "may be suboptimal", and computing the exact set of crossed linear regions is left as future work (§5, p. 8; §7, p. 11).
- Lemma 1 assumes hyperplanes that cut the whole input space, true only for the first hidden layer; deeper regions can be adjacent yet differ in Hamming distance by more than one (§4, p. 6).
- The extreme case $\chi \to 1$ (a looping path) is "theoretically possible, might not be realizable in practice" (Fig. 4, p. 8).
- Applicability to attention layers is not established; normalization and residual layers are addressed only informally (§6.3, pp. 10–11).
- The clustering study is "preliminary" and "require[s] validation on additional datasets" (App. C, p. 16).

## Critical assessment

My reading. The central contribution — transfer the convexity question from input space to the discrete activation space and measure the deficit with a bounded range metric — is clean and, as far as I can tell, new. Lemma 1 is the load-bearing piece and its proof is short and correct in outline; its restriction to full-space hyperplanes is honestly flagged, and the paper is careful not to overclaim for deeper layers.

The evidence is thinner than the framing suggests, however. The strongest claim ("the maximal value of $\chi$ is associated with the network's generalization capacity", §7, p. 11) rests on a single correlation coefficient ($0.987$) computed across *architectures of differing depth at a fixed neuron budget* for *one* lucky random seed (4). Depth and width trade off along that family, so the correlation does not separate depth from architecture or from the training-accuracy confound the authors themselves observe (poorly trained nets fold less). The generalization hypothesis is explicitly "we hypothesize" (§7, p. 11) and should be read as a conjecture. The larger-network result is more convincing in spirit — folding *does* become ubiquitous at $2{\times}300$/$3{\times}200$ — but only two architectures are reported, and the deeper-network experiment in App. D appears to have been run on very few seeds.

The measure also has a definitional wrinkle: $r_1$ is the maximum distance *from the start point*, so it is anchored at $\pi_1$ and depends on which endpoint is chosen; the paper does not discuss this asymmetry, nor the sensitivity to the equally spaced $\lambda_i$ it admits is "suboptimal" (§5, p. 8). $\Phi_N$ (Eq. 5) is defined but never estimated, so the "global" claim stays formal. Finally, no code or trained-parameter release is mentioned in the paper or visible in the .bib, which limits reproducibility of the numbers (the authors state they "store their parameters", §6.1, p. 9, but not where).

The paper is a useful, honest first step: it converts an intuition into a bounded, computable quantity and shows where it does and does not fire. Treat its correlation with accuracy as motivation, not as a result.

## Connections within the collection

- **Direct extension.** The same group's follow-up [@lewandowskiExploitingSpaceFolding2026] generalizes this measure to a wider class of activation functions via equivalence classes of inputs and links folding to adversarial geometry; it is the natural continuation of §7 here.
- **Builds on.** The folding-as-layers reading is from [@NIPS2014_fa6f2a46, §2.4] and [@raghuExpressivePowerDeep2017]; the "paper folding" analogy and the claim that folding manufactures linear separability are from [@keupOrigamiDimensionsHow2022]; the early, unfinished exploration of the folding operation is [@phuongFunctionalVsParametric2019, App. A.2].
- **Convexity/activation-space background.** Definition 1 is adapted from Moser et al., *Tessellation-filtering ReLU neural networks* (IJCAI 2022, outside the collection); the activation-space tessellation viewpoint overlaps [@carlssonPREIMAGERECTIFIERNETWORK2017] and the region-counting tradition of [@NIPS2014_fa6f2a46; @raghuExpressivePowerDeep2017; @serraBoundingCountingLinear2018; @haninComplexityLinearRegions2019; @goujonNumberRegionsPiecewise2023], which this paper cites as expressivity proxies.
- **Symmetry, adjacent framing.** The self-similarity/hidden-symmetry discussion draws on [@grigsbyHiddenSymmetriesReLU]; the fractal benchmark CantorNet is from the authors' own workshop paper (Lewandowski et al., 2024, outside the collection) and echoes the polyhedral-complex and spline views of ReLU geometry represented here by [@balestrieroSplineTheoryDeep2018], [@masdenAlgorithmicDeterminationCombinatorial2022] and [@humayunSplineCamExactVisualization2024].
- **Contrast.** Unlike the combinatorial region-counting work above, this paper does not count regions; it measures the distortion of paths through them, which makes it complementary to [@berzinsPolyhedralComplexExtraction2023] and [@liuIntegratingGeometriesReLU2023a] rather than competing.
