---
citekey: bronsteinGeometricDeepLearning
title: "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"
authors: [Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković]
year: 2021
venue: arXiv preprint
doi: null
arxiv: "2104.13478"
url: null
kind: survey
source_type: pdf
source_sha256: dcf8212ed37db9ac154ac975b8f9c7d36831cf56b75261679b04ab9814985fc2
pages: 160
ingested: 2026-10-15
tags: [geometric-deep-learning, symmetry, equivariance, graph-neural-networks, manifolds, inductive-bias]
concepts_introduced:
  - geometric deep learning blueprint
  - 5G (grids, groups, graphs, geodesics, gauges)
  - deformation stability
  - scale separation prior
  - local equivariant layer
  - global invariant layer
  - local pooling (coarsening)
  - group convolution
  - gauge-equivariant convolution
  - time warping invariance of gated recurrent networks
concepts_used:
  - symmetry group
  - group representation
  - invariance
  - equivariance
  - Riemannian manifold
  - geodesic distance
  - graph Laplacian
  - Fourier transform
  - wavelet transform
  - permutation invariance
builds_on: [bronsteinGeometricDeepLearning2017]
datasets: []
---

# Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges

**Bronstein, Bruna, Cohen, Veličković (2021)**, *arXiv preprint*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)

!!! abstract "TL;DR"
    A 160-page monograph that applies the Erlangen-Programme mindset to deep learning: instead of a zoo of architectures, it derives the main neural network families from three geometric priors — symmetry, deformation stability and scale separation — acting on five geometric domains (the "5G": grids, groups, graphs, geodesics and gauges). The unifying object is a *Geometric Deep Learning blueprint* built from a local equivariant layer, an element-wise nonlinearity, a coarsening operator and a global invariant layer; CNNs, group-equivariant CNNs, GNNs, Deep Sets, Transformers, equivariant message passing, intrinsic mesh CNNs and gated RNNs are all shown to be instances of it (§3.5, p. 29; §5).

## Problem and motivation

The text argues that deep learning, like nineteenth-century geometry before Klein's Erlangen Programme, is a "veritable zoo" of architectures with few unifying principles, so that the same ideas are reinvented and re-branded across application domains (§Preface, p. 2). Learning a generic function on a high-dimensional space is a cursed estimation problem: for the class of 1-Lipschitz functions (and for Sobolev classes $H^s(\Omega^d)$, with minimax rate of order $d/s$) the sample complexity is exponential in the dimension $d$, so global smoothness assumptions only help when $s \gg d$ — unrealistic in practice (§2.2, pp. 8–9). Universal approximation does not remove this: it says nothing about which interpolant is regular, and sparsity-promoting regularisation that does break the curse imposes strong low-dimensional-projection assumptions that real tasks (vision, speech, physics, chemistry) violate (§2.1–2.2, pp. 7–9).

The proposed source of regularity is the *structure of the physical domain* and its symmetries. Following Klein, geometry is recast as "the study of invariants under a symmetry group", and the paper aligns this reading with the earlier Geometric Deep Learning position paper of one of the authors [@bronsteinGeometricDeepLearning2017], recalled in the historic perspective as the origin of the term and of the "new field being born" claim (§7, pp. 125–126). The goal is dual: a common mathematical framework for the most successful architectures, and a constructive recipe for designing future ones from first principles (§1, p. 4).

## Method

**Three geometric priors.** A domain $\Omega$ carries signals $x: \Omega \to \mathcal{C}$, and a symmetry group $G$ acts on them through a representation $\rho$; the hypothesis class is required to be $G$-*invariant* ($f(\rho(g)x) = f(x)$) or $G$-*equivariant* ($f(\rho(g)x) = \rho(g)f(x)$) (§3.1, p. 16). Adding structure (set → topology → smooth → metric) shrinks the symmetry group to a subgroup, an Erlangen-style hierarchy (§3.2, pp. 18–19). Exact symmetry is relaxed to *deformation stability*: a complexity cost $c(\tau)$ measures how far a deformation $\tau \in \mathrm{Diff}(\Omega)$ is from $G$, and $f$ must satisfy Eq. 4 (§3.3, p. 20); the domain itself may also deform, quantified by a distance $d_{\mathcal{D}}$ between domains (graph edit distance, Gromov–Hausdorff) giving Eq. 5 (§3.3, pp. 21–22). Finally, *scale separation* exploits multiscale structure: the Fourier modulus is provably unstable under approximate translations ($\lVert f(\tau x) - f(x)\rVert = O(1)\lVert x\rVert$ however small the deformation), whereas wavelet decompositions are only approximately equivariant ($O(\epsilon)$), and a multiscale coarsening into a hierarchy $\Omega_1, \dots, \Omega_J$ with local pooling yields long-range interactions out of local ones (§3.4, pp. 24–26).

**The blueprint.** The priors do not fix an architecture but an axiomatic construction: a linear $G$-equivariant layer $B$, an element-wise nonlinearity $\rho$, a local pool/coarsening $P$ and a global $G$-invariant layer $A$, composed as in Eq. 13 below (§3.5, p. 29). Because a purely linear invariant is degenerate (it only sees the group average, e.g. the mean RGB colour of an image), richness comes from composing *localised* equivariant maps with nonlinearities; depth enlarges the receptive field and coarsening implements multiresolution analysis (§3.5, pp. 29–30). The text distinguishes a *fixed domain* (images on a grid) from a *varying domain* (graphs, meshes), where stability to domain deformation becomes central (§3.5, p. 30).

**The 5G and eight model families.** Sections 4.1–4.6 develop graphs/sets (permutation group, adjacency $A$, neighbourhood aggregation, Eq. 13), grids/Euclidean spaces (convolutions as circulant matrices, derivation of the discrete Fourier transform from the shift operator), groups/homogeneous spaces (group convolution, whose output lives *on* the group, e.g. spherical convolution on $S^2$ under $SO(3)$), geodesics/manifolds (Riemannian metric, tangent bundle, geodesics, exponential map, parallel transport, isometries; spectral and spatial convolution), gauges/bundles (frames, structure group, gauge-equivalent representations of vector fields), and geometric graphs/meshes (cotangent Laplacian, functional maps, remeshing invariance). Sections 5.1–5.8 then implement the blueprint: CNNs, group-equivariant CNNs, GNNs, Deep Sets/Transformers/latent-graph inference, equivariant message passing, intrinsic mesh CNNs, RNNs and LSTMs/GRUs. A summary table matches each architecture to its domain and symmetry group, e.g. CNN/grid/translation, GNN/graph/permutation, Transformer/complete graph/permutation, LSTM/1-D grid/time warping (§3.5, p. 30).

## Key equations and definitions

The blueprint as a composition of blocks (the printed form $f = A_J B_J P_{J-1}\cdots P_1 \rho_1 B_1$; written here as an explicit composition) (§3.5, p. 29):

$$
f = A_J \circ B_J \circ P_{J-1} \circ \rho_{J-1} \circ \cdots \circ P_1 \circ \rho_1 \circ B_1
$$

Deformation stability (approximate invariance), generalising exact $G$-invariance since $c(\tau) = 0$ for $\tau \in G$ (Eq. 4, p. 20):

$$
\lVert f(\tau \cdot x) - f(x) \rVert \le C\, c(\tau)\, \lVert x \rVert, \qquad \forall x \in X(\Omega)
$$

Stability to domain deformations, with $d_{\mathcal{D}}$ a distance on the space of domains (Eq. 5, p. 22):

$$
\lVert f(x, \Omega) - f(\tilde{x}, \tilde{\Omega}) \rVert \le C \lVert x \rVert\, d_{\mathcal{D}}(\Omega, \tilde{\Omega})
$$

Shift equivariance of convolution, expressed via the shift operator $S$ and a circulant matrix $C(\theta)$; circulant matrices are exactly those commuting with $S$, so "convolution emerges from the first principle of translational symmetry" (§4.2, p. 37):

$$
S C(\theta) x = C(\theta) S x
$$

Permutation-equivariant function on a graph, built by applying one shared permutation-invariant local function $\varphi$ to each node and its neighbourhood features $X_{\mathcal{N}_u}$ (Eq. 13, p. 38):

$$
F(X, A) =
\begin{bmatrix}
\varphi(x_1, X_{\mathcal{N}_1}) \\
\vdots \\
\varphi(x_n, X_{\mathcal{N}_n})
\end{bmatrix}
$$

Group convolution: matching the signal with the $g$-transformed filter, with values on the group $G$ (Eq. 14, p. 41):

$$
(x \star \theta)(g) = \langle x, \rho(g)\theta \rangle = \int_\Omega x(u)\, \theta(g^{-1}u)\, \mathrm{d}u
$$

Spectral convolution on a manifold via a transfer function $\hat{p}$ of the Laplace–Beltrami operator (Eq. 18, p. 54):

$$
(\hat{p}(\Delta) x)(u) = \sum_{k \ge 0} \hat{p}(\lambda_k)\, \langle x, \psi_k \rangle\, \psi_k(u)
$$

Gauge-equivariant convolution on a mesh: messages are parallel-transported by $\rho(g_{v\to u})$ before an anisotropic kernel $\Theta_{\text{neigh}}(\theta_{uv})$ is applied (Eq. 37, p. 89):

$$
h_u = \Theta_{\text{self}}\, x_u + \sum_{v \in \mathcal{N}_u} \Theta_{\text{neigh}}(\theta_{uv})\, \rho(g_{v\to u})\, x_v
$$

Time-warping-invariant recurrence: a learnable gate $\beta$ interpolates between the update $R$ and the previous summary (the gated-RNN form) (Eq. 53, p. 99):

$$
h^{(t+1)} = \beta(z^{(t+1)}, h^{(t)})\, R(z^{(t+1)}, h^{(t)}) + \bigl(1 - \beta(z^{(t+1)}, h^{(t)})\bigr)\, h^{(t)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $\Omega$ | domain (grid, graph, group, manifold, mesh) |
| $X(\Omega, \mathcal{C})$ | space of signals $x: \Omega \to \mathcal{C}$ on the domain |
| $G$, $g$ | symmetry group and its element |
| $\rho(g)$ | representation of $G$ acting on signals (also acts on filters) |
| $\rho$ | group representation (not to be confused with the element-wise nonlinearity, written $\varrho$ in the blueprint) |
| $f$ | function on signals (the learned hypothesis) |
| $B, P, A$ | equivariant layer, coarsening/local-pooling operator, invariant (global-pooling) layer |
| $x$, $x_u$ | signal; node-wise feature vector at node $u$ |
| $X_{\mathcal{N}_u}$ | multiset of neighbourhood features of $u$ |
| $\varphi$ | permutation-invariant local function over a node and its neighbourhood |
| $A$, $\mathcal{N}_u$ | adjacency matrix; 1-hop neighbourhood of $u$ |
| $\Delta$, $\psi_k$, $\lambda_k$ | Laplacian; its eigenfunctions and eigenvalues (the "frequencies") |
| $S$, $C(\theta)$ | shift operator; circulant matrix = discrete convolution |
| $\theta$ | filter / learnable kernel |
| $g_{v\to u}$ | parallel transport (structure-group element) from $v$ to $u$ on a mesh |
| $\tau$, $c(\tau)$ | deformation and its complexity cost |
| $d_{\mathcal{D}}(\Omega, \tilde{\Omega})$ | distance between domains (graph edit, Gromov–Hausdorff) |
| $h^{(t)}$, $\beta$ | RNN summary vector at step $t$; learnable gating function |
| $T = (V, E, F)$ | mesh with nodes $V$, edges $E$, triangular faces $F$ |

## Main results

As a synthesis, the paper's result is the unification itself: it shows that the three priors plus the blueprint reproduce the principal deep architectures, and it makes the domain/symmetry correspondence explicit in a table (§3.5, p. 30). Concrete derivations stand out. Convolution is derived as the unique translation-equivariant linear operator and the discrete Fourier transform is derived from the eigenvectors of the shift operator, with the Convolution Theorem recovered as joint diagonalisation (§4.2, pp. 37–39). The group convolution is proved $G$-equivariant, and the observation that its output is a function on $G$ (not on $\Omega$) explains why a second spherical-convolution layer needs the regular representation on $SO(3)$ (§4.3, pp. 41–43). On manifolds, spectral filters are shown to be geometrically *unstable* under near-isometric perturbation (Figure 12), motivating parametric transfer functions and spatial geodesic-patch convolutions (§4.4, pp. 54–55). GNN layers are classified into three flavours — convolutional (Eq. 33), attentional (Eq. 34) and message passing (Eq. 35) — with the containment "convolution ⊂ attention ⊂ message passing" (§5.3, pp. 78–80). Transformers are obtained as attentional GNNs over a complete graph, with a soft adjacency $a_{uv} = a(x_u, x_v)$ and positional encodings related to the Laplacian eigenvectors of a circular grid (§5.4, pp. 80–82). The derivation of time-warping invariance (below) is the paper's most self-contained theoretical contribution. Section 7 surveys the historic roots (Noether, Weyl gauge theory, Neocognitron, GNNs from computational chemistry, Weisfeiler–Lehman expressiveness): any GNN conforming to the three flavours cannot be more powerful than the 1-WL test, the injective-aggregation variant (GIN) attains it, $k$-WL/$k$-GNN are provably stronger but hard to scale beyond $k = 3$, and with real-valued features injective aggregation needs as many aggregators as the node degree (Borsuk–Ulam argument, §7, pp. 120–121).

Two results are stated as recalled theorems rather than new proofs: shallow "geometric" networks are universal approximators for $G$-invariants (adapting standard universal-approximation arguments, §3.5, p. 29), and the exact characterisations of linear equivariants — two generators on sets, fifteen on graphs, independent of $n$ (§4.1, pp. 33–37). The time-warping analysis (§5.8, pp. 97–101) shows that requiring an RNN class to be invariant to time-warping automorphisms $\tau: \R_+ \to \R_+$ forces the gated form of Eq. 53, in which the gate fits the warping derivative $\mathrm{d}\tau/\mathrm{d}t$; its main difference from an LSTM is that Eq. 53 implies a scalar gate whereas LSTMs vectorise it. From this the authors derive the *chrono initialisation* for the gate bias, $b \sim \log \mathcal{U}([T_l, T_h]^{-1})$, to make the effective forgetting time match a target dependency range.

## Limitations stated by the authors

The text is explicitly "very much a work-in-progress" (§Acknowledgements, p. 126). It does not attempt to cover the whole of Geometric Deep Learning: it focuses on *representation-learning architectures* and sets aside pipelines such as self-supervised learning, generative modelling and reinforcement learning, and optimisers/regularisers such as Adam, dropout and batch normalisation (§1, p. 5). Large parts are deferred: statistical-learning benefits of scale separation, the Fourier transform for general symmetry groups, equivariance characterisations for general domains, and detailed treatment of irregular and regular representations for $E(3)$ equivariant networks are all postponed to "future work" (§3.4, pp. 26; §4.3, p. 43; §5.5, p. 85; §5.3, p. 79). Two-dimensional meshes are treated as a special case of geometric graphs rather than as a standalone domain (§4.6, p. 61), and the "three flavour" GNN taxonomy deliberately excludes higher-order and spectral GNNs (§5.3, p. 80). No experiments or benchmarks are reported; code and data are not released.

## Critical assessment

(My own reading.) This is a *perspective/survey*, not an experimental or theorem-proof paper, and it should be read as a conceptual map rather than a source of precise quantitative results — there are no datasets, baselines or tables of numbers to reproduce. Its strongest content is pedagogical: the derivation of convolution, the DFT and the Laplacian eigenbasis from symmetry, and the three-flavour GNN taxonomy with its representational containment, are unusually clear and are the parts most likely to survive. The "5G"/blueprint framing is genuinely useful as an organising device but its universality claim is programmatic rather than formal: the text shows that known architectures *fit* the blueprint, not that the blueprint *exhausts* the space of useful architectures, and it repeatedly defers the hard questions (expressivity, approximation rates, stability bounds) to future work. The one original derivation — time-warping invariance of gated RNNs — is a sketch that rests on a first-order Taylor expansion and the assumption $\mathrm{d}\tau/\mathrm{d}t \lesssim 1$, and it arrives at gating as a *necessary form* only within that approximation; I would treat Eq. 53 as a suggestive derivation, not a theorem. A pedantic reader will also note notational strain: "convolution" is used for cross-correlation, $x$ denotes both a signal and a filter, and the same glyph family is reused for a group representation and a nonlinearity; the extracted text of several equations (e.g. the blueprint composition, Eq. 13) is typographically mangled in the PDF and had to be reconstructed. Finally, the scope is the geometry of *architectures and domains*; the geometry *inside* a trained network — combinatorial ReLU regions, polyhedral complexes, tropical rational maps, spline decompositions — is outside its frame, which is precisely the gap the rest of this collection fills.

## Connections within the collection

The monograph extends the earlier position paper of the same lead author [@bronsteinGeometricDeepLearning2017], whose term "Geometric Deep Learning" and IEEE SPM framing it adopts and generalises (§7, p. 125). Its viewpoint is complementary to, and largely disjoint in method from, the papers in this collection that study the *combinatorial and algebraic geometry of trained ReLU networks*: the blueprint derives architectures from symmetries of the input domain, whereas [@zhangTropicalGeometryDeep], [@balestrieroSplineTheoryDeep2018] and the region-counting line [@NIPS2014_fa6f2a46; @pascanuNumberResponseRegions2013; @serraBoundingCountingLinear2018] analyse the piecewise-linear/polyhedral structure of the learned function itself. Several threads are "possibly related" rather than directly cited. The mesh/manifold material (§4.4–4.6, §5.6) connects to latent-space and shape geometry work such as [@arvanitidisGeometricallyEnrichedLatent2020] and [@pouplinIdentifyingLatentDistances2023], both of which put a Riemannian/Finslerian metric on a latent manifold in the same spirit. The gauge-theoretic discussion (§4.5) is the neural-network analogue of the information-geometry lineage [@nielsenElementaryIntroductionInformation2020; @combeExploringInformationGeometry2025], and the "manifold hypothesis" motivating the paper's high-dimensional problem setting (§2) is exactly the subject of [@meilaManifoldLearningWhat2023]. The paper's brief mention of topological data analysis for positional embeddings (§7, pp. 120–121) touches the TDA survey [@chazalIntroductionTopologicalData2021] as an "outside-the-frame" technique. Of the surveyed architectures, the GNN and Transformer results are the ones most relevant to the rest of the collection, but the collection contains no GNN, symmetry or equivariance papers to build on directly.

!!! note "Background"
    The Erlangen Programme (Klein, 1872), Noether's Theorem, Weyl's gauge principle, group representations, Riemannian geometry (geodesics, exponential map, parallel transport) and the Fourier/wavelet transforms are standard mathematics used here as background; the paper presents them at a tutorial level rather than assuming them.
