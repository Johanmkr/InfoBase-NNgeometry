---
citekey: pakLecturesDiscretePolyhedral
title: "Lectures on Discrete and Polyhedral Geometry"
authors: [Igor Pak]
year: 2010
venue: null
doi: null
arxiv: null
url: null
kind: tutorial
source_type: pdf
source_sha256: 2fa4730cdbcf16f401e3a708ba306dd5b19c4f85516a60aaa639056f51565879
pages: 442
ingested: 2026-10-15
tags: [discrete-geometry, polyhedral-geometry, convex-polytopes, rigidity, triangulations, spherical-geometry]
concepts_introduced:
  - Fractional Helly theorem
  - Dehn–Sommerville equations
  - Steinitz theorem
  - Delaunay triangulation and flip connectivity
  - polytope algebra and scissor congruence
  - Legendre–Cauchy lemma
  - Cauchy rigidity theorem
  - Gauss–Bonnet theorem for polyhedra
  - Minkowski and Alexandrov theorems
  - bellows conjecture
concepts_used:
  - convex polytope
  - convex hull
  - Euler's formula
  - Morse function
  - spherical geometry
  - simplicial complex
  - triangulation
  - Brunn–Minkowski inequality
  - rigidity and realization space
builds_on: []
datasets: []
---

# Lectures on Discrete and Polyhedral Geometry

**Igor Pak (2010)**, *lecture notes / draft textbook (MIT course notes, 2005–2006; draft dated April 20, 2010)*.

!!! abstract "TL;DR"
    A long, self-contained graduate textbook on discrete and polyhedral geometry, organised in two parts — basic discrete geometry (Helly, Carathéodory, Borsuk, polytope combinatorics, scissor congruence) and the discrete differential geometry of curves and surfaces (the Cauchy rigidity theorem, Dehn's theorem, Alexandrov's theorems, the bellows conjecture, unfoldings). It is written around a small set of reusable tools — variational/Morse arguments, spherical geometry, valuations and local-move connectivity — and gives complete proofs. Nothing in it is about neural networks; the book's role in this collection is as the source of the polyhedral and convex-geometric machinery that the region-counting and ReLU-geometry papers build on.

## Problem and motivation

The subject is discrete geometry and convex polytopes: the combinatorics of finite point, line and polytope configurations, and the geometry of polyhedral surfaces (§ Introduction, p. 3). The author states explicitly that the book is "neither a research monograph nor a comprehensive survey" and is "somewhat biased by [his] personal likes and dislikes" (§ Introduction, p. 3). It began as lecture notes for a course at MIT (Spring 2005, Fall 2006) and was expanded to be largely self-contained (§ Introduction, p. 3). The organising principle is not a list of favourite theorems but "a few favourite tools": topological existence arguments, Morse-theoretic arguments, the variational principle, moduli/realization spaces, geometric and algebraic valuations, local-move connectivity, and spherical geometry (§ Introduction, pp. 4–5).

Part I (Sections 1–20) covers results that are mostly available elsewhere but are reworked into a unified presentation; Part II (Sections 21–40) is "the discrete differential geometry of curves and surfaces" and is "much less readily available", centred on the several proofs of the Cauchy rigidity theorem, the bellows conjecture and Alexandrov's theorems on polyhedral surfaces (§ Introduction, p. 3). Section implications are tabulated on p. 5, and four ready-made course syllabi are suggested on pp. 5–6.

## Method

The book is a teaching text, so the "method" is its expositional strategy. Each section states its main theorems, gives proofs, and closes with exercises, historical remarks and pointers to the literature (§ Introduction, p. 4). Theorems are numbered globally within a section; the author avoids labelling formulas except a few unnamed AMS-TeX symbols (§ Introduction, p. 4). Two running tools recur:

- **Morse functions.** A generic linear functional $\varphi : \R^d \to \R$, non-constant on every edge, orients the graph of a simple polytope; indices of vertices then recover the $h$-vector and the Dehn–Sommerville relations (§8.1), and the same device drives the fractional Helly theorem (§1.3) and the Fáry–Milnor theorem (§24).
- **Variational principle.** Optimising a convex functional (volume, distance, area) constructs the extremal object; used for the Steinitz theorem (§11), the Minkowski and Alexandrov theorems (§§35–37), and the billiard/geodesic existence results (§§9–10).
- **Local-move connectivity.** Flipping triangulations (§14), elementary moves on dissections (§17) and deformations of polygons (§23) prove global statements by checking invariance under a local operation.

Part II repeatedly reduces rigidity questions to a common lemma (the spherical Legendre–Cauchy/arm lemma, §22) and to algebraic relations among squared edge lengths (the Sabitov polynomial relations, §31).

## Key equations and definitions

Helly's theorem in $\R^d$: $n \ge d+1$ convex sets whose every $d+1$ intersect have a common point (§1.1, Thm. 1.2, p. 12). Its proof uses Radon's theorem: any $m \ge d+2$ points in $\R^d$ split into two disjoint subsets with intersecting convex hulls (§1.1, Thm. 1.4, p. 12).

Euler's formula and the Dehn–Sommerville equations for a simple $d$-polytope with $f$-vector $(f_0,\dots,f_d)$ (§8.1, p. 72):

$$
(\heartsuit)\qquad f_0 - f_1 + f_2 - \dots + (-1)^d f_d = 1
$$

$$
\sum_{i=k}^{d} (-1)^i \binom{i}{k} f_i \;=\; \sum_{i=d-k}^{d} (-1)^{d-i} \binom{i}{d-k} f_i \qquad \text{(Thm. 8.1, p. 72)}
$$

with the $h$-vector $h_i$ defined via the Morse orientation, satisfying $F(t)=H(t+1)$ and $h_k = h_{d-k}$ (§8.1, pp. 72–73).

The Legendre–Cauchy lemma (§22.1, Thm. 22.1, p. 207), the engine of the Cauchy rigidity theorem: two convex polygons with equal corresponding edge lengths have either an equal angle sequence or at least four sign changes in $(\alpha_1-\beta_1,\dots,\alpha_n-\beta_n)$; the spherical version (Thm. 22.2) requires both polygons to lie in a hemisphere.

The Cauchy theorem (§26.1, Thm. 26.1, p. 249): two combinatorially equivalent convex polytopes in $\R^3$ with pairwise-equal faces are congruent. Gauss–Bonnet: the curvatures $\omega_i = 2\pi - \alpha_i$ of the vertices of a convex polytope satisfy $\sum_i \omega_i = 4\pi$ (§25.2, Thm. 25.3, p. 242). Minkowski's balance condition and theorem: for a convex polytope with facet areas $a_i$ and unit normals $u_i$, $a_1 u_1 + \dots + a_n u_n = 0$ (§36.1, Prop. 36.1, p. 322), and conversely any positive $a_i$ centred this way are facet areas of a unique polytope (§36.1, Thm. 36.2, p. 323).

The bellows conjecture, proved as a corollary of the integrality of volume: for a simplicial surface $S$ homeomorphic to a sphere, $24\,\mathrm{vol}(S)$ is integral over the ring generated by the squared edge lengths $\ell_{ij}^2$ (§34.1, Lemma 34.1, p. 307); the volume is $\mathrm{vol}(S) = \tfrac{1}{n!}\sum_{(v_1\dots v_d)\in S} \det[v_1,\dots,v_d]$ (§34.1, p. 307).

| Symbol | Meaning (in this book) |
|---|---|
| $P, Q \subset \R^d$ | convex polytopes |
| $f_i$ | number of $i$-dimensional faces |
| $h_i$ | $h$-vector (number of vertices of Morse index $i$) |
| $\omega_i$ | (Gaussian) curvature at vertex $v_i$ |
| $u_i, a_i$ | unit normal and area of facet $F_i$ |
| $\ell_e$, $\theta_e$ | length and dihedral angle of edge $e$ |
| $L : E \to \R^+$ | edge-length function of a plane triangulation |
| $R_\Gamma$ | rigidity matrix of a plane triangulation $\Gamma$ |

## Main results

Part I. Helly/Radon and the fractional Helly theorem: if at least an $\alpha$-fraction of the $(d+1)$-sets of a family of $n$ convex sets in $\R^d$ intersect, then some subfamily of more than $\frac{\alpha}{d+1}n$ sets has a common point (§1.3, Thm. 1.9, p. 15). Carathéodory, Bárány and the colorful Carathéodory theorem for points subordinate to the origin (§2, pp. 20–24). The Borsuk conjecture fails for large $d$ (Kahn–Kalai) but holds for $d=2$ (§3, Thm. 3.2, p. 27) and for smooth bodies (§3.4, Thm. 3.3, p. 28). The Brunn–Minkowski inequality and its equality case are developed because the Minkowski theorem depends on them (§7, p. 61). Simple polytopes have equality-constrained $f$-vectors (Dehn–Sommerville) with $h_0 \le h_1 \le \dots \le h_r$, $r=\lfloor d/2 \rfloor$ (§8.1, Remark 8.2, p. 73); every 3-connected planar graph is the graph of a convex polytope in $\R^3$, in fact with rational coordinates (§11, Thm. 11.1 and Cor. 11.3, pp. 102–103). Scissor congruence: a cube and a regular tetrahedron of equal volume are *not* scissor congruent (Dehn, §15.2, Thm. 15.2, p. 140); every two dissections of a polytope are connected by elementary moves (§17, Thm. 17.1, p. 160), and a polytope tiling space periodically is scissor congruent to a cube (§16, Cor. 16.5, p. 149).

Part II. The four vertex theorem and its many discrete analogues (§21), the Legendre–Cauchy lemma (§22), the Fáry–Milnor theorem for space curves (§24), Gauss–Bonnet (§25). The Cauchy theorem (§26) with two further proofs — Pogorelov's local mean-curvature argument (§28, Lemma 28.5) and the Sen'kin–Zalgaller global argument (§29) — extensions to spherical and higher-dimensional polyhedra (§27). Dehn's theorem in static form (§32, Thm. 32.1, p. 292) and infinitesimal form (§33, Thm. 33.1, p. 302), both equivalent to non-vanishing of the rigidity-matrix determinant. Gluck/Poznjak: almost every length function of a plane triangulation has only finitely many realizations (§31, Thm. 31.1, p. 286), giving rigidity of random triangulated spheres. The bellows conjecture: the volume of any closed polyhedral surface homeomorphic to a sphere is constant under flexing (§31.2, Thm. 31.2, p. 286; proved §34). Alexandrov's curvature theorem, characterising polytopes on given rays by their curvatures (§35, Thm. 35.4, p. 317), the Minkowski theorem (§36), and the Alexandrov existence theorem for arbitrary polyhedral metrics on the sphere (§37). Finally, edge unfoldings: Dürer's conjecture is stated as open (§40, Conjecture 40.1, p. 348) while the source unfolding (§40, Thm. 40.2, p. 349) and the Alexandrov unfolding (Thm. 40.3, p. 349) are proved non-overlapping.

The clearly-stated constants include the kissing numbers: $K_1=2$, $K_2=6$ (§20, Prop. 20.1, p. 185) and $12 \le K_3 \le 14$ (§20, Prop. 20.2, p. 185).

## Limitations stated by the authors

The book is explicitly not comprehensive or exhaustive: hyperbolic geometry is omitted entirely (§ Introduction, p. 5); several major results are cited without proof — the Kahn–Kalai disproof of the Borsuk conjecture (§3.1, p. 26), the Lyusternik–Shnirelman three-geodesics theorem (§10.2, p. 97), the circle-packing proof of Steinitz and the Moore/Thurston extension (§11.3, pp. 104–105), Schramm's theorem (§11.3, p. 104), and the Pogorelov uniqueness theorem (§27, p. 257). Some proofs are given only in low dimension and left as exercises in general (§17.1, p. 160). The material is acknowledged to be at times "completely elementary" and at times "somewhat informal" (§ Introduction, p. 3).

## Critical assessment

As a reference this is excellent exactly where the collection needs it: it proves, from scratch, the convexity and polytope-counting results that ReLU region-counting arguments cite only loosely, and it collects the rigidity material (Cauchy, Dehn, Alexandrov, bellows) that has no counterpart anywhere else in this collection. Two caveats matter for use here. First, coverage is chosen by taste, not by relevance to deep learning: there is **no** chapter on hyperplane arrangements, so Zaslavsky-type region counts [@zaslavskyFacingArrangementsFacecount1975] are not developed here, and the $f$-vector material of §8 applies to *simple polytopes*, not to the complements of arrangements that arise in ReLU networks. Second, the book is a draft: it is dated 2010, several flagged results were later resolved or sharpened, and it carries no DOI or stable URL (the `.bib` entry is `n.d.` and gives no venue; the PDF title page dates it April 20, 2010). There are no datasets, no experiments, and no code — this is pure mathematics. I have not verified the proofs line by line; the statements above are transcribed from the printed text, and the few equations I checked (the Dehn–Sommerville sum, the Gauss–Bonnet sum, the volume formula) are reproduced exactly as printed. The extracted text garbles some symbols (e.g. the "$\heartsuit$" in Euler's formula, and Cyrillic-looking artefacts); I have transcribed the standard mathematical content and note it here as a fidelity caveat.

## Connections within the collection

This book is the collection's general polyhedral-geometry reference rather than a neural-network paper, so the relations run one way: several papers use geometry that this text supplies. The counting of linear regions of ReLU networks [@NIPS2014_fa6f2a46], their upper and lower bounds [@serraBoundingCountingLinear2018; @goujonNumberRegionsPiecewise2023], and the typical-number results [@haninComplexityLinearRegions2019; @haninDeepReLUNetworks2019] all rest on convex-polytope combinatorics of the kind developed here in §8 (f-vectors, Euler-type relations) and on the polyhedral-decomposition viewpoint of §2 and §14. The combinatorial-structure paper [@masdenAlgorithmicDeterminationCombinatorial2022] and the complex-extraction work [@berzinsPolyhedralComplexExtraction2023] work directly with polyhedral complexes and their face lattices, for which §8 and §14 (triangulations, flips, Delaunay) are the relevant background. The persistent-homology treatments of network regions [@carlssonPREIMAGERECTIFIERNETWORK2017; @liuReLUNeuralNetworks2023] use simplicial complexes and subdivisions, where the dissection/valuation machinery of §§16–17 and the triangulation material of §14 apply. The tropical-geometry view of ReLU boundaries [@zhangTropicalGeometryDeep; @alfarraDecisionBoundariesNeural2023] uses Newton polytopes and their subdivisions — the same polyhedral combinatorics. Zaslavsky's arrangement theory [@zaslavskyFacingArrangementsFacecount1975] is the natural companion to §8 for region counting, though (as noted) this book does not itself treat arrangements. The rigidity/realization-space material (§§13, 31–34) is possibly related to the symmetry and equivalence questions for ReLU networks [@grigsbyHiddenSymmetriesReLU; @phuongFunctionalVsParametric2019], but the connection is conceptual — the collection's papers do not cite this book, and I do not claim a direct lineage. Finally, the general convexity tools (Helly, Carathéodory, Radon) of §§1–2 underlie many separation and covering arguments that recur across the collection.
