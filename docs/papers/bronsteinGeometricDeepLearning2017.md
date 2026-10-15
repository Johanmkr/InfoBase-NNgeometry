---
citekey: bronsteinGeometricDeepLearning2017
title: "Geometric Deep Learning: Going beyond Euclidean Data"
authors: [Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, Pierre Vandergheynst]
year: 2017
venue: IEEE Signal Processing Magazine 34(4), pp. 18–42
doi: 10.1109/MSP.2017.2693418
arxiv: "1611.08097"
url: null
kind: survey
source_type: pdf
source_sha256: 1bee1c15358211e775873fc6c89d05a56e9c14871289dfc2a4495cab281b2395
pages: 22
ingested: 2026-10-15
tags: [geometric-deep-learning, graph-neural-networks, spectral-graph-theory, manifold-learning, survey]
concepts_introduced:
  - geometric deep learning
  - geometric priors
  - spectral graph convolution
  - spatial (charting) convolution
  - spectrum-free graph convolution
concepts_used:
  - graph Laplacian
  - Laplace–Beltrami operator
  - Riemannian manifold
  - heat kernel
  - Fourier analysis on graphs
  - graph neural network
  - convolution theorem
  - wavelet transform
builds_on: []
datasets: []
---

# Geometric Deep Learning: Going beyond Euclidean Data

**Bronstein, Bruna, LeCun, Szlam, Vandergheynst (2017)**, *IEEE Signal Processing Magazine*. [arXiv:1611.08097](https://arxiv.org/abs/1611.08097)

!!! abstract "TL;DR"
    A review that names and systematizes *geometric deep learning*: generalizing convolutional
    neural networks to non-Euclidean domains such as graphs and manifolds. It lays out the
    geometric priors (stationarity, locality, compositionality; invariance/equivariance; stability
    to deformations) that make CNNs work on grids, then organizes the field by *how* a
    convolution-like operation is defined — in the spectral domain, spectrum-free (polynomials of
    the Laplacian), or in the spatial/charting domain — and surveys applications and open problems.

## Problem and motivation

Deep networks succeed on data with an underlying Euclidean or grid-like structure, and their
architectures build the relevant invariances of that structure directly into the model (§I). But
many scientific fields study data whose underlying structure is a non-Euclidean space: social and
sensor networks, regulatory and brain networks, and 3D meshes and surfaces modelled as Riemannian
manifolds (§I). On such domains "there are no such familiar properties as global parameterization,
common system of coordinates, vector space structure, or shift-invariance", so even a convolution
"is not well defined" (§I). The paper sets out to show how the key ingredients of CNNs can be
translated to non-Euclidean data, and to bring order to a fast-growing literature whose
publications "tend to use different terminology and notation", making it hard for a newcomer to
grasp the state of the art (§II).

The review separates geometric learning into two problem classes (§II): *characterizing the
structure* of the data (manifold learning / nonlinear dimensionality reduction, graph
decomposition into motifs), and *analyzing functions defined on a given domain*. It concentrates
on the second class, which splits further into a *fixed* domain (spectral methods on a fixed graph)
and *multiple* domains (charting methods for 3D shapes). The intended setting is a supervised one:
an unknown function $y : L^2(\Omega) \to \mathcal{Y}$ observed on a training set (§III, Eq. 1).

## Method

The paper is a structured survey, not a new architecture. Its method is the taxonomy it imposes.

**Geometric priors (§III).** On a compact domain $\Omega$ it isolates the assumptions that CNNs
exploit: *stationarity* (the target is translation-invariant or equivariant, Eq. 2), *locality*
(filters have compact support, §III), *compositionality* (the multi-resolution grid structure), and
*stability to local deformations* — predicted quantities change little under a smooth deformation
field $\tau$, $|y(L_\tau f) - y(f)| \le C \norm{\tau}$ (Eq. 3, and Eq. 4 for the equivariant case).
Together these give a learning complexity of $O(\log n)$ parameters for inputs of size $n$, avoiding
the curse of dimensionality (§III).

**A common geometry of manifolds and graphs (§IV).** The review develops differential geometry and
graph theory in parallel so the two structures can be treated alike: tangent bundles and Riemannian
metrics for manifolds; weighted graphs with a gradient, divergence and Laplacian for graphs;
discretization of manifolds as graphs or triangular meshes (cotangent weights); and Fourier
analysis via the Laplacian eigenbasis, which generalizes the classical Fourier basis and makes the
graph/manifold Laplacian the central operator (§IV). Heat kernels and diffusion distances provide
intrinsic, perturbation-robust quantities where eigenvectors themselves are unstable (§IV).

**Four families (§V–VIII).**
1. *Spectral methods* (§V): define convolution via the Convolution Theorem, so filters are
   diagonal multipliers in the Laplacian eigenbasis (spectral CNN [52]). Smooth spectral multipliers
   (cubic-spline interpolation of a subsampled spectrum) restore constant parameter count, and graph
   coarsening plays the role of pooling. Their fundamental limitation is domain dependence: filter
   coefficients are basis-dependent and do not transfer between domains.
2. *Spectrum-free methods* (§VI): represent filters as polynomials of the Laplacian, giving
   localized $r$-hop filters with complexity $O(rn)$ and no eigen-decomposition — ChebNet using
   Chebyshev polynomials, the single-parameter GCN of Kipf & Welling, and the general Graph Neural
   Network that learns a pointwise function $\psi((Wf)_i, (\mathcal{L} f)_i)$ of diffusion and
   degree operators (Eq. 51).
3. *Charting (spatial) methods* (§VII): for multiple domains, extract intrinsic local patches
   through position-dependent weighting functions $v_j(x,x')$ and apply a template — geodesic
   patches (polar bins), anisotropic heat kernels (elongated, rotated blobs), and the mixture-model
   network (MoNet) which *learns* the patch weights as Gaussians on pseudo-coordinates and subsumes
   the earlier constructions.
4. *Combined spatial/spectral methods* (§VIII): windowed Fourier transform and wavelets, e.g. the
   localized spectral CNN that uses the WFT as a learnable patch operator and generalizes better
   across near-isometric domains.

## Key equations and definitions

Equations below are transcribed from the PDF's extracted text; the typeset math in this scan
extracts poorly, so a few symbols (noted individually) are a faithful reading of a garbled string
rather than a clean render.

$$
T_v f(x) = f(x - v), \qquad x, v \in \Omega \qquad \text{(Eq. 2, §III)}
$$

The translation operator; stationarity means $y$ commutes with it (invariance) or commutes with it
on the output side (equivariance).

$$
g^l(x) = \xi\!\left(\sum_{l'=1}^{p} (f^{l'} \ast \theta^{l,l'})(x)\right) \qquad \text{(Eq. 6, §III)}
$$

A convolutional layer: a bank of filters $\theta^{l,l'}$ applied to a $p$-dimensional input, summed
and passed through a pointwise nonlinearity $\xi$, producing a $q$-dimensional feature map.

$$
(f \ast \theta)(x) = \int f(x - x')\, \theta(x')\, \mathrm{d}x' \qquad \text{(Eq. 7, §III)}
$$

The standard convolution; it relies on the shift $x - x'$ that does not exist on a manifold or graph.

$$
(\Delta f)_i = \frac{1}{a_i}\sum_{j:(i,j)\in E} w_{ij}\,(f_i - f_j) \qquad \text{(Eq. 25, §IV)}
$$

The graph Laplacian as $\Delta = \operatorname{div}\nabla$: the difference between a vertex value and
its weighted local average, $a_i$ a vertex weight and $w_{ij}$ the edge weight.

$$
L f = A^{-1}(D - W)\, f \qquad \text{(Eq. 26, §IV)}
$$

Matrix form, with $W$ the edge-weight matrix, $A$ the diagonal vertex-weight matrix and
$D$ the degree matrix. $A = I$ gives the unnormalized Laplacian, $A = D$ the random-walk Laplacian.

$$
\Delta \phi_i = \lambda_i \phi_i, \qquad f(x) = \sum_{i \ge 0} \underbrace{\langle f, \phi_i \rangle_{L^2(X)}}_{\hat{f}_i}\, \phi_i(x) \qquad \text{(Eqs. 31–32, §IV)}
$$

The Laplacian eigendecomposition (orthonormal eigenfunctions, non-negative eigenvalues
$0 = \lambda_0 \le \lambda_1 \le \dots$) and the induced Fourier series; the eigenbasis generalizes
the classical Fourier basis to a non-Euclidean domain.

$$
(f \ast g)(x) = \sum_{i \ge 0} \langle f, \phi_i \rangle_{L^2(X)}\, \langle g, \phi_i \rangle_{L^2(X)}\, \phi_i(x) \qquad \text{(Eq. 34, §IV)}
$$

Convolution defined *by* the Convolution Theorem. Unlike the Euclidean case it is not
shift-invariant: the spatial filter is position-dependent, and the operator matrix loses its
circulant structure.

$$
g(\Lambda) = \sum_{j=0}^{r-1} \theta_j \Lambda^{j} \qquad \text{(Eq. 45, §VI)}
$$

A spectrum-free filter as a degree-$(r-1)$ polynomial of the Laplacian, with coefficient vector
$\theta$. Because $\Lambda$ is local, $g(\Lambda)$ acts within $r$ hops of each vertex, so filters
are localized by construction.

$$
D_j(x) f = \int_X f(x')\, v_j(x, x')\, \mathrm{d}x', \qquad (f \ast g)(x) = \sum_j g_j\, D_j(x) f \qquad \text{(Eqs. 52–53, §VII)}
$$

The spatial/charting convolution: an intrinsic patch operator built from weighting functions
$v_1,\dots,v_J$ localized near $x$, then a learned template $g$ applied to the extracted patch.

| Symbol | Meaning (in this paper) |
|---|---|
| $\Omega \subset \R^d$ | compact Euclidean domain $[0,1]^d$ |
| $T_v$ | translation operator |
| $L_\tau$ | deformation operator, $\tau$ a smooth deformation field |
| $X$ | manifold; $T_x X$ tangent space; $T X$ tangent bundle |
| $\langle \cdot, \cdot \rangle_{T_x X}$ | Riemannian metric |
| $f \in L^2(X)$ | scalar field on a manifold; $F \in L^2(T X)$ tangent vector field |
| $V, E, F$ | graph vertices, edges, mesh faces |
| $W, A, D$ | edge-weight, vertex-weight, degree matrices |
| $\Delta$ (or $\mathcal{L}$) | Laplace–Beltrami / graph Laplacian |
| $\phi_i, \lambda_i$ | Laplacian eigenfunctions and eigenvalues |
| $h_t(x, x')$ | heat kernel at diffusion time $t$ |
| $D_j$ | patch operator with weighting function $v_j$ |
| $\ast$ | convolution (redefined per family) |
| $\xi, \psi$ | pointwise nonlinearity; learned node function in the GNN framework |

## Main results

As a survey the paper reports no experiments of its own; its "results" are the taxonomy and the
numbers it relays from the works it reviews.

- **Taxonomy (§VIII, "Dichotomy of geometric deep learning methods").** Spectral CNN [52],
  ChebNet/GCNN [45], GCN [77] and GNN [78] on graphs; Geodesic CNN [47] and Anisotropic CNN [48]
  on meshes; MoNet [54] and the localized spectral CNN [89] on graphs, meshes and point clouds —
  classified by whether their convolution is defined spectrally, as polynomials of the Laplacian,
  via charting, or in a combined spatial/spectral way.
- **Citation-network classification (§IN6).** On the CORA citation network — 2708 vertices (papers),
  5429 edges (citations), 1433-dimensional bag-of-words features, seven classes — a two-layer
  spectral CNN parametrized as in Eq. 50 reaches **81.6%** accuracy, against **75.7%** previous best;
  MoNet improves this slightly to **81.7%**.
- **Recommender systems (§IX).** The Netflix matrix is stated as 480K movies × 18K users
  (8.5B elements), with only **0.011%** known entries, quoted as the setting for geometric matrix
  completion with a multi-graph CNN plus RNN.
- **Applications (§IX).** Network analysis (ranking, community detection, PageRank as a principal
  eigenvector), recommender systems, 3D shape correspondence and retrieval (Fig. 6), particle
  physics and chemistry (graph neural networks at the LHC and IceCube; $N$-body dynamics),
  molecule design, and medical imaging of brain functional networks.

## Limitations stated by the authors

The authors frame most of §X as open problems. *Generalization across domains*: spectral methods
are domain-dependent and cannot be transferred directly, spatial methods lack a principled
construction of low-dimensional local coordinates on general graphs (e.g. anisotropic diffusion),
and spectrum-free methods generalize only in functional form (§X, §V). *Time-varying domains*:
signals over a dynamically changing graph are not handled. *Directed graphs*: their non-symmetric
Laplacians lack an orthogonal eigendecomposition, so spectral constructions do not apply, and
citation networks are often flattened to undirected ones "which obviously may lose important
information" (§X). *Synthesis*: generative modelling on non-Euclidean domains is largely untouched,
with the hard part being reconstruction of a geometric embedding from an intrinsic representation.
*Computation*: mainstream deep-learning frameworks assume grid data; non-grid computation needs
large-scale graph-processing paradigms. The paper also notes (§IV) that "manifold" as used in data
science can be sharply non-smooth (self-intersections, infinite curvature, varying dimension) and
is not a classical smooth manifold.

## Critical assessment

This is a well-timed and, judging by how widely it is cited, field-defining review; its lasting
contribution is the framing rather than any theorem. Its strongest move is to put manifolds and
graphs under one roof by making the Laplacian the common object, so that "spectral vs spatial" and
"single vs multiple domains" become the organizing axes rather than the mathematical machinery.

Several caveats are worth stating. (i) The evidence is secondary: all quantitative results (CORA,
Netflix) are quoted from other papers, with no unified benchmark or head-to-head comparison, so the
paper cannot adjudicate between the families it lists — MoNet is called "currently the
state-of-the-art" on the strength of its authors' own reported numbers. (ii) The "geometric prior"
story is normative: stationarity, locality and deformation stability are motivated as the reasons
CNNs work, but the stability property is only provable for specific wavelet-like constructions and
is otherwise posited (§III). (iii) The review predates the modern message-passing unification of
GNNs, so the treatment of the GNN framework (Eq. 51) as a general form is a 2017 snapshot. (iv) The
manifold-learning and graph-spectral preliminaries are broad but terse, and a reader wanting the
stated goals ("to be re-implementable") will need the cited primary sources. Reproducibility of
the survey itself is a non-issue (no code or data), but the numbers it relays should be traced to
their originals before reuse.

The clearest way to read the paper now is as the 2017 manifesto and taxonomy whose more principled
successor is the same lead authors' later *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics,
and Gauges* blueprint, which replaces the method-by-method survey with a symmetry/scale-separation
framework.

## Connections within the collection

- [@bronsteinGeometricDeepLearning] — the later survey by the same lead authors (Bronstein et al.,
  *Grids, Groups, Graphs, Geodesics, and Gauges*): the direct successor to this paper, recasting the
  same geometric-priors programme as a blueprint rather than a method inventory.
- [@meilaManifoldLearningWhat2023] — the "structure of the domain" problem class of §II
  (manifold learning, nonlinear dimensionality reduction) treated on its own terms; the two overlap
  on the manifold-learning preliminaries but neither cites the other here.
- [@villarFullyCovariantMachine2023] — shares the invariance/equivariance/covariance vocabulary that
  this paper uses as its geometric priors (§III); a principles-level counterpart rather than a
  method.
- [@nielsenElementaryIntroductionInformation2020] and [@combeExploringInformationGeometry2025] —
  information geometry, a different non-Euclidean geometry (the manifold of probability
  distributions) not covered by this review's manifold/graph dichotomy; possibly related only at the
  level of "non-Euclidean geometry for machine learning".
- [@arvanitidisGeometricallyEnrichedLatent2020] and [@pouplinIdentifyingLatentDistances2023] —
  Riemannian/Finslerian geometry of latent spaces; related in spirit (intrinsic, metric-aware
  representations) but a distinct research line from graph/manifold convolutional models;
  possibly related.

No paper in this collection is used as a component or extended by this survey, so `builds_on` is
empty. Several methods central to it — spectral networks (Bruna et al.), ChebNet (Defferrard et al.),
GCN (Kipf & Welling), MoNet and geodesic/anisotropic CNNs (Masci, Boscaini, Monti et al.) — are
outside the collection and have no citekeys here.
