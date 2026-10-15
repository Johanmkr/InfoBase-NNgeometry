---
citekey: meilaManifoldLearningWhat2023
title: "Manifold Learning: What, How, and Why"
authors: [Marina Meilă, Hanyu Zhang]
year: 2023
venue: arXiv preprint
doi: 10.48550/arXiv.2311.03757
arxiv: "2311.03757"
url: null
kind: survey
source_type: pdf
source_sha256: 770104e267d3d063cd6f4b3e2c7722084bfdc6f2a4176ab82883bb65f0b0e9ad
pages: 33
ingested: 2026-10-15
tags: [manifold-learning, dimensionality-reduction, riemannian-geometry, spectral-embedding, intrinsic-dimension]
concepts_introduced:
  - manifold assumption
  - embedding algorithm
  - one-shot embedding algorithm
  - attraction-repulsion algorithm
  - repeated eigendirection problem
  - independent eigendirection selection
  - push-forward Riemannian metric
  - renormalized graph Laplacian
concepts_used:
  - Riemannian manifold
  - Riemannian metric
  - geodesic distance
  - isometric embedding
  - graph Laplacian
  - Laplace-Beltrami operator
  - intrinsic dimension
  - diffusion maps
  - spectral embedding
  - principal component analysis
builds_on: []
datasets: [swiss roll, chopped torus, ethanol molecular configurations, SDSS galaxy spectra]
---

# Manifold Learning: What, How, and Why

**Meilă, Zhang (2023)**, *arXiv preprint*. [arXiv:2311.03757](https://arxiv.org/abs/2311.03757)

!!! abstract "TL;DR"
    A survey of non-linear dimension reduction written from a statistician's point of view. It fixes the differential-geometric vocabulary (charts, embeddings, Riemannian metric, isometry), groups manifold-learning methods into three paradigms, and then concentrates on embedding algorithms. Its organising claims are that eigendirection methods (Isomap, DM/LE, LTSA) are well understood statistically but suffer the *repeated eigendirection problem*, that attraction-repulsion algorithms (t-SNE, UMAP) avoid it heuristically, and that **all** of them distort geometry — a distortion that can nonetheless be repaired by estimating a push-forward Riemannian metric.

## Problem and motivation

High-dimensional data usually lie near a low-dimensional structure, and the aim of manifold learning (ML, "non-linear dimension reduction") is to recover that structure rather than merely shrink the data (§1). Where PCA fits a linear subspace, ML assumes the data are sampled from a smooth $d$-manifold $M$ embedded in $\R^D$ and looks for a map $F$ that carries the samples to low-dimensional coordinates while preserving intrinsic geometry (§1, §3). The survey answers three questions — what ML is, how it is done, and why the parameters behave as they do — and argues that the "why" is where practitioners most often go wrong: neglecting statistical consistency yields algorithms with no large-$n$ limit, density-induced biases, and *artifacts* (clusters, arms, horseshoes) that have no counterpart in the data (§5).

The setting is stated as the **Manifold Assumption**: $\{x_i\}_{i=1}^n$, $x_i \in \R^D$, are drawn from a distribution supported on or near a $d$-manifold $M \subset \R^D$, and with a few noted exceptions the survey treats the noiseless case where the data lie exactly on $M$ (§3). A valid algorithm should converge, as $n\to\infty$, to a smooth embedding that recovers $M$ whatever its shape (§3).

## Method

The survey is an organising review rather than a new algorithm. Its structure is:

1. **Differential-geometric background** (§2): charts and atlases, the *intrinsic dimension* $d$ (chart-independent), embeddings $F: M \to \R^m$ with $m \ge d$ and $m \ll D$, and the fact that a $d$-manifold embeds in $\R^{2d}$ (Whitney) (§2.2). Geodesic distance, the tangent bundle $TM$, and the Riemannian metric $g$ make arc length, angle and volume intrinsic to $M$ and independent of the ambient space (§2.3).
2. **Three paradigms** (§3, Tab. 1): *linear local approximation*, which represents $x_i$ by coordinates in $\hat{T}_pM$ ($D\to d$, local only); *principal curves and $d$-manifolds*, which denoise to a global $M\subset\R^D$ ($D\to D$, no dimension reduction); and *embedding algorithms*, which give global low-dimensional coordinates ($D\to m\ge d$). All start from a **neighborhood graph** — either a radius graph ($\lVert x_i-x_j\rVert\le r$) or a $k$-NN graph, the latter symmetrized and faster to build but harder to analyse (§3.1).
3. **Embedding algorithms** (§4), split into one-shot methods that read coordinates off eigenvectors of a matrix built from the graph (PCA §4.1, Isomap §4.2.1, Diffusion Maps/Laplacian Eigenmaps §4.2.2, LTSA §4.2.3) and attraction-repulsion methods that iterate from an initial embedding (t-SNE and UMAP §4.3.1).
4. **Statistical basis** (§5): sampling-density bias and how renormalization removes it, the neighborhood scale, intrinsic-dimension estimation, and estimation of the Laplace-Beltrami operator.
5. **Applications** (§6): statistics (regularisers, spectral clustering), visualization, and the physical sciences (astronomy, dynamical systems, chemistry, biology).

## Key equations and definitions

An embedding $F$ is an **isometry** iff it preserves the metric at every point:

$$
\langle v_1, v_2\rangle_{g(p)} = \langle dF_p(v_1), dF_p(v_2)\rangle_{h(F(p))}, \quad \forall p\in M,\ v_1,v_2\in T_pM \qquad \text{(Eq. 1, p. 6)}
$$

The neighborhood graph is weighted by a kernel, almost always Gaussian $K(u)=\exp(-u^2)$, whose width $h$ effectively sets a radius $r\propto h$:

$$
K_{ij} := \begin{cases} K\!\left(\dfrac{\lVert x_i - x_j\rVert}{h}\right), & x_j \in N_i,\\[6pt] 0, & \text{otherwise.}\end{cases} \qquad \text{(Eq. 2, p. 7)}
$$

PCA is the reconstruction-error problem solved by the leading singular vectors of $X$:

$$
\min_{T\in\R^{D\times d},\ T^\top T = I_d} \sum_{i=1}^n \lVert x_i - TT^\top x_i\rVert^2 \;=\; \min_{T\in\R^{D\times d},\ T^\top T = I_d} \lVert X - XTT^\top\rVert_F^2 \qquad \text{(Eq. 4, p. 10)}
$$

t-SNE's objective is a cross-entropy between data similarities $V$ and embedding similarities $W$:

$$
L_{t\text{-SNE}} = -\sum_{i,j} \frac{1}{n} V_{ij} \ln W_{ij} + \ln \sum_{ij} W_{ij} \qquad \text{(Eq. 7, p. 13)}
$$

whose gradient exposes the attraction-repulsion mechanism, with $\rho$ the trade-off parameter:

$$
\frac{\partial L_{t\text{-SNE}}}{\partial y_i} = -\sum_{ij} V_{ij} W_{ij}(y_i - y_j) + \frac{n}{\rho}\sum_{ij} \frac{W_{ij}}{w_{tot}}(y_i - y_j) \qquad \text{(Eq. 8, p. 14)}
$$

Intrinsic dimension via the correlation dimension (slope of $\log k_{i,r}$ against $\log r$):

$$
\hat{d}_C = \lim_{r\to 0} \frac{\log \frac{1}{n}\sum_{i \neq i'} \mathbf{1}_{\lVert x_i - x_{i'}\rVert \le r}}{\log r} \qquad \text{(Eq. 10, p. 20)}
$$

Distortion can be removed by endowing the image with the **push-forward metric** of $F$, using the pseudoinverse $dF_p^{\dagger}$:

$$
\langle u, v\rangle_{\tilde{g}(Fp)} \equiv \langle dF_p^{\dagger}(u),\, dF_p^{\dagger}(v)\rangle_{g(p)}, \qquad \tilde{g}(Fp) \equiv \big((dF_p)^T\big)^{\dagger}\, g(p)\, (dF_p)^{\dagger} \qquad \text{(Eqs. 12–13, p. 23)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $M \subset \R^D$ | data manifold; $D$ ambient, $d$ intrinsic dimension |
| $F: M \to \R^m$ | manifold-learning embedding; $m \ge d$, $m \ll D$ |
| $dF_p$ | differential (Jacobian) of $F$ at $p$ |
| $g$ | Riemannian metric on $M$; $\tilde{g}$ its push-forward on $F(M)$ |
| $T_pM$ | tangent space at $p$ |
| $\Delta_M$ | Laplace-Beltrami operator, $\Delta_M f \equiv \mathrm{div}\,\mathrm{grad}(f)$ |
| $K, L$ | kernel/similarity matrix and graph Laplacian |
| $h, r, k$ | kernel width, radius scale, number of neighbors |
| $q(x)$ | sampling density on $M$ |
| $\rho$ | t-SNE attraction-repulsion trade-off |

## Main results

**A taxonomy and its trade-offs.** One-shot algorithms are the best understood and scalable, but greedy eigenvector selection can collapse a dimension. Because the eigenvectors of a Laplacian are nested harmonics, the scatter plot of $(v_1,v_2)$ on a rectangle or a thin torus traces a parabola: this is the **Repeated Eigendirection Problem (REP)**, a rank drop of the embedding Jacobian, diagnosed visually by the "horseshoe" shape (§4.3). Attraction-repulsion methods are immune to it, but the survey notes that a local relaxation "cannot resolve the rank deficiency globally, and it may become trapped in a local optimum" (§4.3.2).

**Graph construction and density.** Non-uniform sampling contracts dense regions and stretches sparse ones; for the graph Laplacian this bias was computed by Coifman & Lafon and removed by the renormalized Laplacian (Algorithm 3), whose degrees $d'_i$ estimate the local sampling density; $k$-NN graphs show qualitatively the same bias (§5.1, Fig. 6).

**Scale selection has consistency rates.** Singer (2006) gives $h \sim n^{-1/(d+6)}$ for the random-walk Laplacian, and Calder & Trillos (2019) give $k \sim n^{4/(d+4)}(\log n)^{d/(d+4)}$ for $k$-NN Laplacian estimation (§5.2). From the convergence rate $\approx\sqrt{\log n/k}\,(k/n)^{1/d}$ the authors calculate that a 10-fold error decrease needs $n$ to grow by $\approx 10^{(d+4)/3}$ — i.e. non-parametric rates require sample size exponential in the dimension (§5.2).

**Dimension estimation is theoretically possible, empirically hard.** The minimax rate for estimating $d$ lies between $n^{-2n}$ and $n^{-\frac{n}{D+1}}$ (sic; the first exponent renders ambiguously in the PDF), an information-theoretic lower bound; with i.i.d. noise the rate becomes exponential and estimating $d$ has "no satisfactorily robust solutions" (§5.3). Practical estimators use the scaling $\log \#B_r = d\log r + \log n + \text{const}$ (correlation dimension, box-counting, doubling dimension, $k$-NN MLE, local-PCA eigengaps) (§5.3).

**No isometric embedding is practical.** Nash's theorem guarantees isometric embeddings but its proof is not numerically usable; Diffusion Maps is isometric only as $m\to\infty$, which defeats dimension reduction (§5.5). The practical answer is to estimate the per-point distortion $\tilde{h}_i$ (via the push-forward metric, or via $\tfrac12\Delta_M f_{kl,p}$ for products of centered coordinates) and correct distances, angles and volumes afterwards; when the embedding is isometric the $\tilde{g}$ singular values are all $1$, so their deviation measures distortion (Fig. 7) (§5.5).

**Applications** range from manifold regularisation and spectral clustering, through visualization (with the warning that clusters, filaments and holes may be artifacts), to galaxy spectra, molecular collective coordinates and neural recordings (§6).

## Limitations stated by the authors

The survey lists what it omits: **manifold learning in noise**, "perhaps the most important" gap, which biases estimators and slows convergence; the injectivity radius/reach and manifolds with boundary (different rates near borders); and out-of-sample embedding, i.e. mapping a new $x\in\R^D$ onto an existing $F(M)$ (Nyström) and back (§7, "What we omitted"). It also states that UMAP's theory "is still very limited" (§4.3.1), that neighbor-embedding algorithms "remain heuristic for ML" (§4.4), and that the choice of $k$ in t-SNE is an "unsolved issue" (§4.3.1).

## Critical assessment

The survey's value is its insistence on statistical and geometric rigor over recipes: it treats the distortion of an embedding as a measurable quantity rather than a cosmetic issue, and it draws a clean line between algorithms with consistency proofs and heuristics. The strongest sections are the density-bias analysis (§5.1) and the push-forward-metric remedy (§5.5), which give a concrete diagnostic, not just a caveat.

As my own reading: the treatment is uneven. The three-paradigm taxonomy (§3) is a useful frame, but the two non-embedding paradigms get less than a page each, so the actual content is narrower than the title's "what, how, and why" implies. The statistical results are quoted, not proved or unified; the reader cannot tell how tight the recommended rates are, and the authors concede the hidden constants depend on unknown quantities (volume, curvature, injectivity radius). The reported minimax rate for dimension estimation — between $n^{-2n}$ and $n^{-n/(D+1)}$ — is stated without derivation and both exponents are unusual; the first in particular is easy to misread from the PDF text layer, so it should be verified against Kim et al. (2019) before being quoted. Finally, the survey is heavy on spectral methods and light on the reconstruction/autoencoder families (only mentioned via Ting & Jordan), which by 2023 were the dominant practical alternative; and it never measures algorithms on a common benchmark, so its comparative claims rest on figures rather than numbers.

## Connections within the collection

This is the reference survey for the *manifold premise* that most of the collection's latent-geometry papers assume: that high-dimensional data concentrate near a low-dimensional $M$ and that intrinsic geometry should be measured with a Riemannian metric (§2–3). The push-forward metric of §5.5 is the exact counterpart of the **pull-back** metric used to give geometry to latent spaces in [@arvanitidisGeometricallyEnrichedLatent2020] and [@arvanitidisPullingBackInformation2022]: where those papers pull an ambient or Fisher-Rao metric back through a generator, this survey pushes the data metric forward through an embedding, and both use the estimated metric to fix otherwise arbitrary coordinates and distances. The survey covers only Riemannian structure, which is precisely the limitation that [@pouplinIdentifyingLatentDistances2023] addresses by moving to Finslerian geometry for latent distances — a direction the survey does not mention.

The broader differential-geometric background it assembles overlaps with the information-geometry expositions [@nielsenElementaryIntroductionInformation2020] and [@combeExploringInformationGeometry2025], though the survey's manifold is a data manifold without a statistical family attached, whereas those works endow *statistical* manifolds with metric and connection. Its "non-Euclidean data" framing is the statistician's twin of the symmetry-and-geometry programme in [@bronsteinGeometricDeepLearning2017], and its account of eigenvectors that summarise manifold structure is close in spirit to the latent-topology induction of [@fuLatentTopologyInduction2022]. The idea that a neural network can *itself* be used to learn a Riemannian metric ([@halversonMetricFlowsNeural2024]) is the parametric analogue of the non-parametric metric estimation reviewed here, and is a possible bridge between the two literatures.
