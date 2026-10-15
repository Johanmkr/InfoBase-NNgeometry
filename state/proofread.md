# Proofreading report 2026-10-15

A (lint): 36 files fixed automatically by `lint_pages.py --fix`, 7 errors fixed by hand
(citation-in-math at 5 sites in `polyhedral-foundations.md`, the math-in-heading at
`information-geometry.md:99`, the `\y` false positive at `relu-network-geometry.md:85`), 0 errors and
0 warnings left. The 105 `eq-label` warnings were resolved by moving every in-math
`\qquad\text{(Eq. N…)}` / `\qquad(\text{Eq. N…})` locator into the sentence introducing its equation.

B (browser): 0 errors (was 113: 74 raw citations, 30 undefined math macros, 9 raw-math blocks); 137
`overflow` warnings left, all of which the checker derives from MathJax's *invisible* assistive-MathML
copies — the rendered width of every display block was measured directly instead (see below) and no
display equation overflows the text column. Two infrastructure files were replaced with the current
scaffold templates (`hooks/citations.py`, `docs/javascripts/mathjax.js`).

C (reading): 148 findings on 17 pages, listed below. They are **not applied** — layer C reports.

## Verification of layer B

`check_site.py` counts a display block up to twelve times: Material re-typesets the page on load, and
each pass leaves a nested `mjx-container` inside `mjx-assistive-mml` (a screen-reader copy, positioned
absolutely and clipped, with inflated metrics). It reports the widths of those copies, so its `overflow`
list is both double-counted and ~6 % too wide. The reader-visible container is the direct child of
`div.arithmatex`.

Measured on the visible containers instead (`/tmp/nnwork/measure_md.py`, which typesets each block's
on-disk TeX inside the built page's own CSS/MathJax/webfont environment, pins the MathJax container
scale to the site's worse case of 127.3 % so the reading is deterministic, and needs no rebuild):

- before this sync's fixes: **46 display blocks wider than the 688 px column, 20 of them ≥ 792 px**
- after: **0**, across every `docs/**/*.md` with a display block (swept in one run)

The widths a single page load reports are not stable on their own: the MathJax container font-size
races with the webfont load and lands on either 127.3 % or 119.5 % of the base, ~6 % apart. Passing a
block therefore requires it to fit at the wider scale.

## Findings

### docs/textbook/polyhedral-foundations.md

- docs/textbook/polyhedral-foundations.md:219 [exercise] the add-the-lines sum is off by one: from $r(A_{m+1})=r(A_m)+m+1$ the closed form comes from $1+\sum_{j=1}^{m}j$, but $\sum_{j=1}^{m}(j+1)$ gives $\frac{m(m+1)}{2}+m+1$ → write $\sum_{j=1}^{m} j$ (or $\sum_{j=0}^{m-1}(j+1)$)
- docs/textbook/polyhedral-foundations.md:416 [exercise] Exercise 3's solution prints the same false chain: $1+\sum_{j=1}^{m}(j+1)=1+\frac{m(m+1)}{2}+m$ does not equal $\frac{m(m-1)}{2}+m+1$ (off by $m$) → change the sum to $\sum_{j=1}^{m} j$ *(verified arithmetically; the error is present in the previous sync's commit and survives in both places)*
- docs/textbook/polyhedral-foundations.md:281 [notation] the maxout bound introduces $n$ for the number of inputs and $m$ for the number of rank-$k$ outputs, though the chapter uses $n_0$ for inputs (line 184) and $m$ already means the number of hyperplanes → use $n_0$ and a distinct symbol for the unit count
- docs/textbook/polyhedral-foundations.md:268 [notation] $i$ changes meaning: line 256 has $f_i$ as a unit's argument and $f_{i,j}$ as its affine pieces, then this line calls an affine piece $f_i$ → keep $i$ for units and use $j$ for the pieces
- docs/textbook/polyhedral-foundations.md:316 [notation] depth is written $k$ here ("$k$ hidden layers of widths $n_1,\dots,n_k$"), then $L$ eleven lines later; the shared notation uses $L$ → use $L$ throughout
- docs/textbook/polyhedral-foundations.md:154 [notation] the arrangement's sign vector is written $s\in\{-,+\}^{n_1}$, but the shared notation (and line 131 of this chapter) write $\mathbf{s}(x)$ → use $\mathbf{s}$
- docs/textbook/polyhedral-foundations.md:120 [term] "the collection of linear regions is the object later chapters call the region complex" conflates regions with the complex, which also contains their lower-dimensional faces → say the regions are its top-dimensional cells
- docs/textbook/polyhedral-foundations.md:330 [repetition] the "maximum is not a measure-zero accident" argument is given in full here and again at counting-regions.md:34 → keep it once and cross-link
- docs/textbook/polyhedral-foundations.md:451 [language] "Adding one point to each cell, the count cannot exceed $k$" does not say why the count is bounded by $k$ → rewrite (each region is selected by one of the $k$ affine pieces)
- docs/textbook/polyhedral-foundations.md:469 [language] "each pair of points $(x)$ and its reflection" — "$(x)$" is not a point → "each point $x$ and its reflection"
- docs/textbook/polyhedral-foundations.md:202 [language] "the $(m)$-th hyperplane" has stray parentheses around $m$ → "the $m$-th hyperplane"

### docs/textbook/relu-network-geometry.md

- docs/textbook/relu-network-geometry.md:34 [notation] the symbol inventory contradicts itself and the shared notation: $F_i$ is given as Liu's layer output, $F_{ij}$ as Masden's node map, $F_1,F_2$ as networks, $S(F)$ as the complex, while notation.md:95 and space-folding.md:108 use $F_i$ for the fold-set → rename the layer-output symbol and keep $F_i$ only for the fold-set
- docs/textbook/relu-network-geometry.md:50 [term] "activation pattern" is defined as the set-valued $S=(S^1,\dots,S^L)$, whereas lines 34, 45 and 114 call the bit vector $\mathbf{s}(x)$ the activation pattern → use one name and symbol for the concept
- docs/textbook/relu-network-geometry.md:114 [notation] $h_i$ for the hidden units of layer $i$ duplicates the shared width symbol $n_l$, and $h$ then clashes with $h^l$ and $h^l(S)$ → use $n_l$
- docs/textbook/relu-network-geometry.md:54 [notation] after fixing $n_0,n_L$ at line 34 the chapter reverts to Sattelberg's $f:\R^{d}\to\R^{o}$ ($d,o$ undefined; also $\R^{d+1}$ at line 199) → translate to $n_0,n_L$
- docs/textbook/relu-network-geometry.md:250 [reference] "the two-input XOR network of §1" (and "Which method from §5" at line 307) point at section numbers the rendered page does not have → refer by section title or link
- docs/textbook/relu-network-geometry.md:116 [language] "find, for instance, 25 polyhedra $[-1,1]^2$ for a $2\text{-}3\text{-}3\text{-}3$ network" is missing a preposition → "25 polyhedra in $[-1,1]^2$"

### docs/textbook/counting-regions.md

- docs/textbook/counting-regions.md:105 [notation] Goujon's $n_1,\dots,n_N$ reuse the width symbol $n_i$ for per-partition region counts, and $N$ already means the total hidden-unit count at line 32 → rename the partition counts and the partition number
- docs/textbook/counting-regions.md:65 [notation] $k$ is the number of hidden layers here and at line 69, but the maxout rank at lines 49 and 67 → use $L$ for layers
- docs/textbook/counting-regions.md:49 [notation] the maxout bound uses $n$ for inputs and $m$ for units, both undefined while the chapter uses $n_0$/$n_l$ → use the shared symbols
- docs/textbook/counting-regions.md:32 [notation] total hidden units is $N$, but relu-network-geometry.md:114 writes $h$ for the same quantity → align one symbol across the textbook
- docs/textbook/counting-regions.md:73 [notation] $\dim(h^l(S))$ uses $h^l$ for the layer-$l$ activations where the shared notation has $a^{(l)}$ → use $a^{(l)}$
- docs/textbook/counting-regions.md:55 [repetition] the identification recursion $N^l_R=\sum_{R'\in P^l_R}N^{l-1}_{R'}$ and the alternating-sign rectifier unit are developed again at length in space-folding.md:59–96 → give the full treatment in one chapter and cross-link
- docs/textbook/counting-regions.md:136 [language] "for any $r,s$ with $rs\ge k=q$" introduces $q$ inside its own defining clause → state the relation between $k$ and $q$ in words

### docs/textbook/space-folding.md

- docs/textbook/space-folding.md:139 [language] the parenthetical is backwards: an angle of $\pi/2$ means the normal is *orthogonal* to the subspace (it folds nothing), while a normal parallel to the subspace is the angle-$0$ "only squashes" case → correct to "(normal orthogonal to the subspace)"
- docs/textbook/space-folding.md:43 [reference] "[next chapter](counting-regions.md)" is wrong: counting-regions is chapter 3 and space-folding chapter 4 → it is the previous chapter
- docs/textbook/space-folding.md:589 [reference] same mistake in Exercise 6: "The next chapter ([Counting linear regions](counting-regions.md))" → it is the previous chapter
- docs/textbook/space-folding.md:25 [reference] "The previous chapter described a ReLU network as a partition … labelled by an activation pattern" points at relu-network-geometry (two chapters back) → re-point or reword
- docs/textbook/space-folding.md:319 [reference] two different results are called "Lemma 1" in this chapter — the 2026 paper's properties lemma here, the 2024 paper's convexity lemma at lines 208 and 400 → qualify each as "Lemma 1 of [key]"
- docs/textbook/space-folding.md:400 [language] "Since Lemma 1 bounds folding to networks with at least two hidden layers" — wrong verb → "restricts folding to"
- docs/textbook/space-folding.md:55 [repetition] this section repeats the identification/folding material already carried in counting-regions.md:55–57 → link to that treatment instead of restating it

### docs/textbook/spline-networks.md

- docs/textbook/spline-networks.md:23 [reference] "The last chapter described …" and line 27 "the previous one", both pointing at [The region complex of a ReLU network], are wrong: the preceding chapter is Space folding and folding neurons (ch. 4) → say "the region-complex chapter"
- docs/textbook/spline-networks.md:121 [reference] "(see [The region complex of a ReLU network](relu-network-geometry.md), §6)" — chapters have unnumbered `##` sections, and the local-linear-map analysis is §7 while §6 is the transition graph → link to the section by title
- docs/textbook/spline-networks.md:41 [term] "CPWL" is used throughout without expansion, while ch. 2 defines "CPWA" for the same class → expand once and use a single abbreviation
- docs/textbook/spline-networks.md:348 [language] malformed page locator "[…, Thm. 1 and §3.1, p. 3, 5]" (also line 287 "p. 4–5") → "pp. 3, 5"

### docs/textbook/tropical-geometry.md

- docs/textbook/tropical-geometry.md:279 [exercise] Exercise 1's solution claims "the constant zero is $-\infty$" and writes $\sigma(x)=x\oplus(-\infty)$; by §1 (line 36) $a\oplus(-\infty)=a$, so this equals $x$, not $\max\{x,0\}$ → the solution should read $\sigma(x)=x\oplus 0$ *(verified against §1; present in the previous sync's commit)*
- docs/textbook/tropical-geometry.md:240 [notation] $\mathrm{Vol}_n=n!\,\mathrm{vol}_n$ is called the "normalised volume", but notation.md defines $\mathrm{Vol}(P)$ as Euclidean volume and $d!\,\mathrm{Vol}(P)$ as the normalised one → use the shared symbol or state the clash
- docs/textbook/tropical-geometry.md:127 [language] "If $f=F\oslash G$ and $\nu=f$, then …" — $\nu$ is introduced nowhere and the clause is unused → delete "and $\nu=f$"
- docs/textbook/tropical-geometry.md:127 [notation] $B$ denotes the decision-boundary set here and at line 141, and also the output weight matrix at line 135 in the same section → rename one
- docs/textbook/tropical-geometry.md:130 [notation] $\ominus$ (tropical subtraction) is used without definition while $\oplus,\odot$ are defined in §1 → define $\ominus$ at first use
- docs/textbook/tropical-geometry.md:54 [notation] the activation is introduced as $\rho^{(l)}$ although lines 24 and 26 promise translation into the shared notation, where the ReLU is $\sigma$ → use $\sigma$ or flag the translation
- docs/textbook/tropical-geometry.md:165 [term] "CPWL" is used (lines 165, 173, 189, 262, 299) without ever being expanded, and ch. 2 spells the same notion "CPWA" → expand once and use one abbreviation across the textbook
- docs/textbook/tropical-geometry.md:232 [repetition] §8 reproduces counting-regions.md's "Depth lower bounds from rational weights" almost verbatim → keep one derivation and link to it
- docs/textbook/tropical-geometry.md:163 [repetition] §6 on decomposition polyhedra repeats the Brandenburg material already in counting-regions.md's "From counting to construction, and to convexity" → link instead of re-deriving

### docs/textbook/region-topology.md

- docs/textbook/region-topology.md:24 [reference] "the previous chapter built that complex" is wrong: the previous chapter is Tropical geometry (ch. 6); the polyhedral complex is built in ch. 2 → name the region-complex chapter
- docs/textbook/region-topology.md:147 [notation] "polyhedra sharing an $(m-1)$-face" uses an undefined $m$; the chapter writes the input dimension $n_0$ → $(n_0-1)$-face
- docs/textbook/region-topology.md:162 [notation] "impractical above $D>8$" uses an undefined $D$ → $n_0>8$
- docs/textbook/region-topology.md:99 [term] "gives the DTEM" is an abbreviation never expanded → "the distance to the empirical measure (DTEM)"
- docs/textbook/region-topology.md:214 [language] the solution phrase "are at bottleneck distance at most $\log$-bounded by the scale" is incoherent → e.g. "differ in scale parameter by at most a logarithmic factor, bounding the bottleneck distance accordingly"
- docs/textbook/region-topology.md:109 [repetition] Masden's sign-sequence cubical complex is explained at length here (lines 109–131) and again in relu-network-geometry.md → give the construction once and link to the other chapter

### docs/textbook/verification-and-preimages.md

- docs/textbook/verification-and-preimages.md:25 [reference] "The [previous chapter](relu-network-geometry.md)" is wrong: the preceding chapter is Topology of regions and of data (ch. 7) → say "the region-complex chapter" and keep the link
- docs/textbook/verification-and-preimages.md:29 [notation] the network is $f$ in the chapter's own notation sentence (line 25) but written $F$ here and at line 69, and the same line reuses $X$ for both the input set and its backward-reachable set → use $f$ throughout and rename the backward set
- docs/textbook/verification-and-preimages.md:33 [language] cross-links to chapter 2 use the raw filename as link text (lines 33, 41, 53, 117, 134), unlike every other cross-link → use "[The region complex of a ReLU network](relu-network-geometry.md)"
- docs/textbook/verification-and-preimages.md:41 [term] "PWA" appears (lines 41, 53, 209) for the piecewise-affine form without ever being introduced → write "(PWA)" at first use
- docs/textbook/verification-and-preimages.md:51 [language] US/UK spelling mixed inside one sentence: "*neighbor constraint*" next to "the neighbour sharing the facet" → use "neighbour" throughout
- docs/textbook/verification-and-preimages.md:216 [term] "MIP" is used (lines 216, 218) without being defined → introduce "(MIP)" at first use

### docs/textbook/information-geometry.md

- docs/textbook/information-geometry.md:24 [notation] the intro promises $\Gamma_{ijk}$ for the Amari–Chentsov cubic tensor, but the chapter uses $\Gamma_{ijk}$ for the connection coefficients (line 72) and $C_{ijk}$ for the tensor (lines 75, 78, 159, 214) → keep $C_{ijk}$ for the tensor and fix the introduced list or notation.md
- docs/textbook/information-geometry.md:24 [notation] declares $\mathcal{M}$ for a statistical manifold, then never uses it: the body writes plain $M$ (lines 28, 33, 35, 41, 78, 101) → write $\mathcal{M}$ throughout
- docs/textbook/information-geometry.md:132 [notation] $\Theta\neq\R^D$ uses $\Theta$ as the parameter domain, but $\Theta$ is nowhere introduced → define it as the domain before use
- docs/textbook/information-geometry.md:148 [notation] introduces $T_{ijm}$ for the Amari–Chentsov tensor without connecting it to the chapter's own $C_{ijk}$ → note that it is the same tensor in Combe's symbol
- docs/textbook/information-geometry.md:83 [fidelity?] "the squared Hellinger divergence at $\alpha=0$" attaches an $\alpha$ label inside a list of $f$-divergences → check Nielsen §3.10 that the Hellinger label is $\alpha$ and not $f$

### docs/textbook/manifold-learning.md

- docs/textbook/manifold-learning.md:177 [notation] $\mathcal{M}$ is used for the BERT representation manifold, but notation.md reserves $\mathcal{M}$ for a statistical manifold and this chapter writes plain $M$ for its own data manifold → use $M$ for the representation manifold
- docs/textbook/manifold-learning.md:38 [notation] $h$ is the target Riemannian metric in the isometry equation, while lines 46 and 56 use $h$ for the Gaussian kernel width → rename one
- docs/textbook/manifold-learning.md:146 [notation] $m$ here is Raghu's input dimension, but $m$ is the embedding dimension earlier (lines 30, 64) → use distinct symbols
- docs/textbook/manifold-learning.md:199 [notation] $N$ is now the number of latent states, while it was the target manifold (lines 30, 33) and the total neuron count (lines 152–154) → rename the state count
- docs/textbook/manifold-learning.md:154 [notation] $U(n,k,m)$ appears in the depth-beats-width chain (and at 218) without ever being defined → define it
- docs/textbook/manifold-learning.md:146 [reference] "the hand-crafted constructions of Montufar et al." is named with no citation although the paper is in the collection → add [@NIPS2014_fa6f2a46]
- docs/textbook/manifold-learning.md:208 [language] "On the reading of the summary, the traversal claim rests on four hand-picked sentences" is a meta-reference to the summary turned textbook prose → state the limitation directly

### docs/textbook/latent-geometry.md

- docs/textbook/latent-geometry.md:73 [notation] the decoder is $\phi$ (lines 24, 30, 104), then the same map is called the generator $g(z)$ (lines 73, 89) → use one symbol
- docs/textbook/latent-geometry.md:101 [notation] the decoder is written $h:Z\to H$ here and $J_h$ at line 104, but it was $\phi$ before → use $\phi$ consistently
- docs/textbook/latent-geometry.md:121 [notation] $\eta_i$ are simplex probabilities, while $\eta$ was the likelihood parameter two paragraphs up → rename the simplex coordinates
- docs/textbook/latent-geometry.md:24 [notation] $\mathcal{M}\subset X$ for the data manifold clashes with notation.md's $\mathcal{M}$ = statistical manifold → use $M$
- docs/textbook/latent-geometry.md:129 [reference] "p. 4–5" (and "p. 9–10" at line 160) should be "pp. 4–5" / "pp. 9–10"; elsewhere the chapter writes `pp.`
- docs/textbook/latent-geometry.md:141 [reference] "The key observation, due to Eklund and Hauberg" attributes a result to a work outside the collection with no citation → cite it or drop the attribution
- docs/textbook/latent-geometry.md:57 [reference] the Background box cites [@pouplinIdentifyingLatentDistances2023] for the Finsler-metric definition, against the rule that Background material is not cited to the collection → drop the citation or move the material out
- docs/textbook/latent-geometry.md:33 [notation] $\langle v_x, v_x\rangle_x$ uses one symbol for the two tangent vectors whose inner product is taken → write $\langle v_x, w_x\rangle_x$
- docs/textbook/latent-geometry.md:185 [reference] "Eq. 7–9" (also at 240) is a plural locator written as `Eq.`; other chapters write `Eqs.` → use `Eqs. 7–9`

### docs/textbook/metric-flows.md

- docs/textbook/metric-flows.md:16 [order] prerequisites list [The region complex of a ReLU network](relu-network-geometry.md), but line 22 says the ReLU-network conventions "are not needed here" → drop the prerequisite or the sentence
- docs/textbook/metric-flows.md:58 [reference] "the ordinary neural tangent kernel of the collection" presupposes an introduction of the NTK, but no other chapter defines it → introduce the NTK here
- docs/textbook/metric-flows.md:128 [notation] $\phi(x)$ is the cosine-activation network function, but $\phi$ is the dilaton in the Perelman functional (lines 153, 159) → rename one
- docs/textbook/metric-flows.md:52 [reference] "The distinguished object appearing in both flow equations (Eq. 2.7, p. 6)" mislabels the locator: the flow equations are 2.5 and 2.8, and 2.7 defines $\Theta$ → move the locator
- docs/textbook/metric-flows.md:137 [term] "a Gaussian NNGP kernel" uses the abbreviation without spelling it out → write "neural network Gaussian process (NNGP)" at first use
- docs/textbook/metric-flows.md:69 [reference] the Background box cites [@halversonMetricFlowsNeural2024] for Hamilton and Perelman, against the Background rule → drop the citation or move the material out

### docs/textbook/symmetry-and-gdl.md

- docs/textbook/symmetry-and-gdl.md:159 [notation] $H^l_i$ is used for a folded/bent hyperplane, but notation.md defines $H_i$ as a region-bounding hyperplane and $\phi^{(l)}$ as the bent hyperplane → use $\phi^{(l)}_i$
- docs/textbook/symmetry-and-gdl.md:153,156,167 [notation] the network map and fold-set are written $F_\theta$, $F(h)$, $h$, $h^{1:l}_i$, while the shared notation writes the map $f_\theta$/$f$ → switch to $f_\theta$/$f$ or state that the source's $F$/$h$ is kept
- docs/textbook/symmetry-and-gdl.md:132 [notation] $L$ denotes the projectile range and $g$ the gravitational acceleration, clashing with $L$ = depth and $g\in G$ a group element → rename (e.g. $R$ for the range, $\mathbf{g}$ for gravity)
- docs/textbook/symmetry-and-gdl.md:134 [notation] $H$ denotes the subgroup fixing a missing object, while $H^l_i$ later denotes a folded hyperplane → rename the subgroup
- docs/textbook/symmetry-and-gdl.md:112 [repetition] the same equation is located twice: the orphan locator "(Eq. 37, §5, p. 89)" here and the full citation after the block at line 118 → keep only the citation after the equation
- docs/textbook/symmetry-and-gdl.md:112 [reference] "acting on the eigenbasis of §IV" is ambiguous: the sentence cites both [@bronsteinGeometricDeepLearning, §4.4] and [@bronsteinGeometricDeepLearning2017, §IV] → say which paper's §IV
- docs/textbook/symmetry-and-gdl.md:66 [language] "Two degenerate cases show why nonlinearity is mandatory" announces two cases, but only one is presented → drop "Two"
- docs/textbook/symmetry-and-gdl.md:42 [language] "a coordinate system, a unit system, a node ordering was chosen" — a three-item list takes a plural verb → "…were chosen"
- docs/textbook/symmetry-and-gdl.md:26 [term] "the geometric deep learning programme" where the chapter and the concept name use "blueprint" → use "blueprint"
- docs/textbook/symmetry-and-gdl.md:26 [term] "the tropical normal fan" is not the term used in the tropical-geometry chapter, which calls the object the dual subdivision of the Newton polytope → align the term or link it
- docs/textbook/symmetry-and-gdl.md:142 [language] "the conjecture that enforcing passive symmetry helps stated as a conjecture" is circular → rephrase
- docs/textbook/symmetry-and-gdl.md:89 [language] "equivariance (a node's output must follow its node)" is unclear → "must permute with its node"
- docs/textbook/symmetry-and-gdl.md:221 [exercise] Exercise 2 asks for one example of each; the solution gives the active example but no concrete passive example, although the chapter supplies units covariance at line 132 → add that example

### docs/textbook/implicit-geometry.md

- docs/textbook/implicit-geometry.md:44 [notation] in $H_i^{(l)}={x\in\R^D:\ W_i^{(l)\top}x^{(l-1)}+b_i^{(l)}=0}$ the symbol $x$ is both the input point and the layer activation, and $x^{(l)}$ contradicts the mapping stated at line 26 → write $a^{(l-1)}$ and use $z$ for the point
- docs/textbook/implicit-geometry.md:44,73,85 [notation] $\R^D$ / $D=2$, $D=3$, $D>8$ use $D$ for the input dimension, but the chapter states the input is $x\in\R^{n_0}$ and never defines $D$ → use $n_0$
- docs/textbook/implicit-geometry.md:47 [notation] $\sigma_i(x)$ for the sign-vector clashes with $\sigma$ the ReLU, defined two lines above; the shared symbol is $\mathbf{s}(x)$ → use $\mathbf{s}(x)$
- docs/textbook/implicit-geometry.md:51 [notation] $\Omega$ is the set of regions here but the shape at line 30 → use a distinct symbol for the collection of regions
- docs/textbook/implicit-geometry.md:51 [notation] $h^L_1$ appears in the boundary formula with no definition → say what it is
- docs/textbook/implicit-geometry.md:95 [notation] $H$ is the entropy of $q$ here, but $H_i^{(l)}$ is the folded hyperplane at line 44 on the same page → rename the entropy
- docs/textbook/implicit-geometry.md:67 [notation] $G$ denotes the GINN generator, while notation.md reserves $G$ for the ReLU transition graph and a symmetry group → rename the generator
- docs/textbook/implicit-geometry.md:171 [exercise] the Exercise 4 solution lists "curvatures" among the vertex-position quantities the piecewise-linear complex can optimise, contradicting line 75 (curvature needs a non-vanishing second derivative) → drop it or qualify it as discrete/combinatorial

### docs/comparison/index.md

- docs/comparison/index.md:146 [reference] concept-map edge `G9 -->|same face-counting machinery| G1` is not in `builds_on`: no worst-case paper builds on `pakLecturesDiscretePolyhedral`, and that summary states the book has no chapter on hyperplane arrangements → drop the edge or re-source it (the face count is Zaslavsky 1975, outside the collection) and correct the label
- docs/comparison/index.md:159 [reference] edge `G7 -->|covariance and units| G8` has no `builds_on` support: `villarFullyCovariantMachine2023` builds on `bronsteinGeometricDeepLearning`, itself in G8 → the arrow must originate within G8 or be removed
- docs/comparison/index.md:108 [fidelity?] "region counts … do not distinguish architectures with identical accuracy [@humayunSplineCamExactVisualization2024, §1]" is not what that summary's §1 says → re-anchor it or cite as outside the collection

### docs/comparison/tables.md

- docs/comparison/tables.md:142 [reference] the "Twelve papers" sentence is wrong: `meilaManifoldLearningWhat2023` is listed as appearing "only in the first and last tables" but has no row in the Assumptions table, and `bronsteinGeometricDeepLearning` is also absent from the second table yet uncounted → recount (thirteen) and name the right set
- docs/comparison/tables.md:161 [notation] $k$ carries four meanings inside the bounds table: number of hidden layers (161–162), maxout rank (166, 167, 171), width (168), number of affine components (181) → rename per row or annotate
- docs/comparison/tables.md:179 [notation] $N$ denotes the hidden-unit count (line 165) and the integer base of the weights ("a prime $p\nmid N$") in the same table → use a distinct symbol
- docs/comparison/tables.md:174 [notation] the input dimension is written $d$ here and $d_{\mathrm{in}}$ on line 175, but $n_0$ in every other row of the same table → use $n_0$
- docs/comparison/tables.md:178 [notation] the bound uses the undefined constants $T$ (also line 176) and $n_{\mathrm{in}}$; neither appears on the notation page → define them in the table caption or add them to notation.md

### docs/comparison/themes.md

- docs/comparison/themes.md:62 [reference] "the same per-region affine parameters … [@humayunSplineCamExactVisualization2024, Lemma 1, Eq. 3]" — the summary gives the pair as Eqs. 3–4 → cite "Lemma 1, Eqs. 3–4"
- docs/comparison/themes.md:192 [notation] $\chi$, $d_\chi$, $\Phi_N$ and $S_N$ (also lines 214–220) are used without ever appearing in notation.md → add the folding measure to the notation page or gloss it at first use
- docs/comparison/themes.md:460 [reference] "[@masdenAlgorithmicDeterminationCombinatorial2022, §5] … one exactly from vertices, one approximately from sampled labels [@liuReLUNeuralNetworks2023, §1]" cites results/motivation where the methods are meant → use masden §4.2 and liuReLU §2.3/§4.2
- docs/comparison/themes.md:128 [notation] the hanin bound uses undefined $T$ and non-shared $n_{\mathrm{in}}$ (same as index.md:79 and tables.md:178) → same fix

### docs/notation.md

- docs/notation.md:72 [notation] the clash note covers $\sigma$, $N$, $G$, $H$ but not $g$, which is the Fisher metric component (line 125), a group element (line 151) and a gravity vector (symmetry chapter) → add $g$ to the clash note
- docs/notation.md:133 [notation] $\rho$ is listed as an alternative for the geodesic distance, but $\rho$ is a representation in the symmetry rows (lines 151–152) and throughout symmetry-and-gdl → note the clash

### docs/papers/dhayalkarDiscreteFunctionalGeometry2025.md

- docs/papers/dhayalkarDiscreteFunctionalGeometry2025.md:73 [notation] the adjacency equation labels regions by $a_i,a_j$, but the symbol table (line 121) defines $a_i$ as the $i$-th bit of a single pattern $a$ → write the region's pattern as $a^{(i)}$ (or $a$, $a'$) and keep $a_i$ for the bit
- docs/papers/dhayalkarDiscreteFunctionalGeometry2025.md:70 [notation] $E$ is defined here as the shared-facet condition, while the symbol table (line 123) and §5.1 (line 63) define $E$ as Hamming-1 adjacency → state one definition and present the facet equivalence as Lem. 1
- docs/papers/dhayalkarDiscreteFunctionalGeometry2025.md:63 [notation] the sampled pattern is written $a(x)\in\{0,1\}^{nL}$, but the table names that count $m=nL$ and writes $a(x)\in\{0,1\}^m$ → use $\{0,1\}^m$
- docs/papers/dhayalkarDiscreteFunctionalGeometry2025.md:155 [repetition] the "three of six validations are tautologies" verdict is already made point by point in the Main-results bullets at lines 135, 137 and 139 → state the assessment once, in the Critical assessment

### docs/papers/serraBoundingCountingLinear2018.md

- docs/papers/serraBoundingCountingLinear2018.md:118 [notation] $\dim(h^l(S)) \le \min\{|S^l|,\ \dim(h^{l-1}(S^0))\}$ uses $S^0$, which is not defined and is inconsistent with the $h^{l-1}(S)$ used two lines above → almost certainly $\dim(h^{l-1}(S))$
- docs/papers/serraBoundingCountingLinear2018.md:132 [notation] $w$ appears in the Thm. 8/9 equation ($(w+1)^{L-1}$) and at line 186, but is in no row of the symbol table → add it
- docs/papers/serraBoundingCountingLinear2018.md:132 [notation] $m$ is used inside the Thm. 8/9 lower bound while the table (line 155) and line 51 define $m$ as the output dimension → reconcile or rename

### docs/papers/haninDeepReLUNetworks2019.md

- docs/papers/haninDeepReLUNetworks2019.md:138 [notation] the weight-rescaling identity uses $d$ (the depth), which is defined neither in the text nor in the symbol table → define it at first use or add it
- docs/papers/haninDeepReLUNetworks2019.md:95 [notation] the bias-density bound uses $\rho_{b_{z_1},\dots,b_{z_k}}$, which is not in the symbol table → name it there

### docs/papers/grigsbyHiddenSymmetriesReLU.md

- docs/papers/grigsbyHiddenSymmetriesReLU.md:136 [notation] the definition of $\dim_{\mathrm{fun}}$ uses $J_{E_Z}(\theta)$, but $J_{E_Z}$ and $E_Z$ are defined nowhere, while the estimator at line 140 writes the Jacobian as stacked $\nabla_\theta F_\theta(z_i)$ → define $E_Z$ or restate in the stacked-gradient form
- docs/papers/grigsbyHiddenSymmetriesReLU.md:118 [notation] $\pi_i$ (coordinate projection) in the pre-activation equation is not defined or tabulated → spell it out once
- docs/papers/grigsbyHiddenSymmetriesReLU.md:216 [fidelity?] the Critical assessment cites "He init, unit-normal inputs, Jacobian rank" as the setup, but the Method (lines 102–104) gives only the weight/bias initialisation, never the input distribution → check against the paper and, if correct, state it in the Method

### docs/textbook/index.md and docs/notation.md — render-level, fixed in this pass

These were confirmed in the built HTML and repaired during the sync rather than only reported; they are
listed here for the record.

- docs/textbook/index.md:15 — the table of contents was stale (chapter 2 omitted `dhayalkar…`, and its goal said "dual graph"): re-rendered from `state/outline.yml`
- docs/notation.md:73–82 — nine bare `@key` mentions in the symbol-clash box were not linkified by `hooks/citations.py` and rendered as literal text: wrapped in `[@key]`
- docs/textbook/symmetry-and-gdl.md:149 — `[$c_l$ in the notation table; @…]` did not match the hook's citation pattern and rendered literally: the math was moved out of the brackets
- docs/comparison/index.md:79 — the Hanin bound had lost its constant $T$, contradicting tables.md:178 and themes.md:128, both of which agreed with the source: restored to $(T\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$
- docs/comparison/tables.md:112,114 — two dataset cells did not match their sources ("300 windows of 200 samples" for three walkers × 100 windows; "8000 contextualized embeddings", a figure no source states): corrected

### Known but not fixed — outside this layer's scope

- **LaTeX accents `\u` and `\v` render literally.** `Meil{\u a}` and `Veli{\v c}kovi{\'c}` appear as
  `Meil\u a` and `Veli\v c` on the references page, the landing page and the comparison page. Both
  `hooks/citations.py:_plain` and the shared `litlib.plain` know only `\'`, `` \` ``, `\"`, `\^`, `\~`;
  `\u` (ă), `\v` (č, an inverted caron) and `\o` (ø, which survives only because the braces are
  dropped) are missing from both tables. The fix belongs in the two shared scripts, not in this repo's
  Markdown, and changing them would alter every label site-wide.
