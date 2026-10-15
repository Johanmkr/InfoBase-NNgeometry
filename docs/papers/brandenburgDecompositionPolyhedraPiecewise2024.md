---
citekey: brandenburgDecompositionPolyhedraPiecewise2024
title: "Decomposition Polyhedra of Piecewise Linear Functions"
authors: [Marie-Charlotte Brandenburg, Moritz Leo Grillo, Christoph Hertrich]
year: 2025
venue: "The Thirteenth International Conference on Learning Representations (ICLR 2025)"
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: 758a68ba9d1d8661b33ee9565236bb3f2ea8d9c8bf10581bef206cd9abdeacf3
pages: 30
ingested: 2026-10-15
tags: [piecewise-linear, polyhedral-geometry, tropical-geometry, neural-network-theory, submodular-optimization]
concepts_introduced:
  - decomposition polyhedron
  - reduced decomposition
  - minimal decomposition
  - coarsening of a convex CPWL function
  - hyperplane function
  - k-th order statistic
concepts_used:
  - continuous piecewise linear function
  - polyhedral complex
  - balanced polyhedral complex
  - tropical geometry
  - convex function
  - submodular function
  - Lovász extension
  - Newton polytope
  - ReLU neural network
builds_on: [zhangTropicalGeometryDeep]
datasets: []
---

# Decomposition Polyhedra of Piecewise Linear Functions

**Brandenburg, Grillo, Hertrich (2025)**, *The Thirteenth International Conference on Learning Representations (ICLR 2025)*.

!!! abstract "TL;DR"
    The paper studies how to write a continuous piecewise linear (CPWL) function $f$
    as a difference $f = g - h$ of two convex CPWL functions with few linear pieces,
    fixing an underlying polyhedral complex $P$ that bounds where $g$ and $h$ may
    bend. The set of such decompositions is a *decomposition polyhedron* $DP(f)$,
    the intersection of two shifted copies of the cone of convex functions; reduced
    decompositions are exactly its bounded faces and every minimal decomposition is a
    vertex. The authors disprove a proposed minimal construction of Tran & Wang (2024)
    beyond two dimensions, give a depth–size tradeoff for ReLU networks representing
    convex CPWL functions, and translate the framework to submodular set functions.

## Problem and motivation

Every CPWL function $f : \R^n \to \R$ can be written as a difference $f = g - h$ of two convex CPWL functions (a folklore fact, §1, p. 1), which underpins difference-of-convex (DC) programming and the analysis of the nonconvexity of ReLU networks. The central question (§1, Problem 1.1, p. 2) is how to find such a decomposition in which $g$ and $h$ have as few linear pieces as possible, since the number of pieces measures complexity and controls how cheaply the two convex parts can be stored and optimized. No finite procedure guaranteeing a minimal decomposition was known, despite the 2-dimensional construction of Tran & Wang (2024), which the authors claim does not extend to higher dimensions (§1.1, p. 2).

The paper's move is to make the problem tractable by *fixing* an underlying polyhedral complex $P$: the possible locus of nonlinearity of $g$ and $h$ is prescribed, and one searches only among decompositions compatible with it. Setting: $P$ is a complete polyhedral complex of $\R^n$, and $\mathcal{V}_P$ is the space of CPWL functions compatible with $P$, quotiented by affine functions (§2, p. 4). The contribution is structural (theory plus proofs), not a practical algorithm; the authors state explicitly that no implementation is provided (§1, Limitations, pp. 2–3).

## Method

The method has four steps.

1. **Parameterize CPWL functions by weights.** Lemma 3.1 (§3, p. 5) observes that the functions compatible with $P$ form a linear subspace $\mathcal{V}_P$; after quotienting affine functions, Lemma 3.2 identifies $\mathcal{V}_P$ with the space $W_P = \{w : P^{n-1} \to \R \mid (P,w) \text{ balanced}\}$ of weight functions satisfying the tropical *balancing condition* on the codimension-1 faces. Each weight $w_f(\sigma)$ measures "how convex or concave" $f$ is across the facet $\sigma$.
2. **Convexity becomes a sign condition.** Proposition 3.3: $f \in \mathcal{V}_P$ is convex iff $w_f \ge 0$, and convex with $P = P_f$ iff $w_f > 0$. Hence the convex functions $\mathcal{V}_P^+$ form a polyhedral cone (Lemma E.2).
3. **Define the decomposition polyhedron.** $DP(f) = \{(g,h) \mid g,h \in \mathcal{V}_P^+,\ f = g-h\}$ (Def. 3.4). Theorem 3.5 shows it is a polyhedron, and under the projection $\pi(g,h)=g$ it is the intersection of two shifted copies of the cone $\mathcal{V}_P^+$.
4. **Read off optimality from faces and vertices.** Reduced decompositions (no convex $\phi$ subtractable from both parts) are exactly the bounded faces (Thm. 3.8); decompositions that admit no non-trivial coarsening are exactly the vertices (Thm. 3.11); every minimal decomposition is a vertex (Thm. 3.13), giving a finite (vertex-enumeration) search within $DP(f)$ and a linear-programming characterization (Thm. 3.18). When $DP(f)$ has a unique vertex, the minimal decomposition is unique (Prop. 3.14), which happens e.g. for hyperplane functions and $k$-th order statistics (§3, pp. 6–7, Examples A.2–A.3).

The framework is then applied: to show pre-existing constructions (hyperplane extension and the local-maxima / lattice decomposition of Kripfganz & Schulze (1987), Appendix B) fail to be vertices even for the $k$-th order statistic; to refute Tran & Wang's higher-dimensional "gluing" procedure via an explicit polytope placement counterexample (Appendix C); to submodular set functions via the Lovász extension (Section 5, Appendix D); and to ReLU-network constructions (Section 6).

## Key equations and definitions

The balancing condition that defines a compatible weight function (§2, p. 4):

$$
\sum_{\substack{\sigma \in P^{n-1}\\ \sigma \supset \tau}} w(\sigma)\cdot e_{\sigma/\tau} = 0 \qquad \text{(§2, p. 4)}
$$

where $e_{\sigma/\tau}$ is the unit normal of the facet $\tau$ pointing into $\sigma$; it holds at every $\tau \in P^{n-2}$.

The weight attached to a function, given affine pieces $f|_P(x) = \langle a_P, x\rangle + b_P$ and $f|_Q(x) = \langle a_Q, x\rangle + b_Q$ on the two full-dimensional polyhedra meeting at $\sigma = P \cap Q$ (Proof of Lemma 3.2, p. 24):

$$
w_f(\sigma) := \langle e_{P/\sigma}, a_P\rangle + \langle e_{Q/\sigma}, a_Q\rangle = \langle e_{P/\sigma}, a_P - a_Q\rangle .
$$

It equals $\lVert a_P - a_Q\rVert_2$ at a convex breakpoint and $-\lVert a_P - a_Q\rVert_2$ at a concave one (p. 24).

The decomposition polyhedron and its projection (Def. 3.4 and Thm. 3.5, pp. 5–6):

$$
DP(f) := \{(g,h) \mid g,h \in \mathcal{V}_P^+,\ f = g - h\}, \qquad
\pi(DP(f)) = \mathcal{V}_P^+ \cap (\mathcal{V}_P^+ + f) = \mathcal{V}_P^+ \cap (f + \mathcal{V}_P^+).
$$

Under the isomorphism $\mathcal{V}_P \cong W_P$ this reads $\{(w_g, w_h) \in W_P^+ \times W_P^+ \mid w_g - w_h = w_f\}$ and $\{w_g \in W_P^+ \mid w_g \ge w_f\}$ (Remark 3.6, p. 6).

Definitions of the two optimality notions:

- *Reduced* (Def. 3.7, p. 6): $(g,h)$ is reduced if no $\phi \in \mathcal{V}_P^+ \setminus \{0\}$ has both $g-\phi$ and $h-\phi$ convex. Equivalent to $(g,h)$ lying in a bounded face of $DP(f)$ (Thm. 3.8).
- *Minimal* (Def. 3.12, p. 6): $(g,h)$ is minimal if no $(g',h') \in DP(f)$ has $g'$ with at most as many pieces as $g$, $h'$ with at most as many pieces as $h$, and one of the two with strictly fewer pieces (Pareto-optimality of the piece counts).

A *hyperplane function* (Def. 3.16, p. 7) — the class computed by a 1-hidden-layer ReLU network:

$$
f(x) = \sum_{i\in[k]} \lambda_i\cdot \max\{\langle x, a_i\rangle + b_i,\ \langle x, c_i\rangle + d_i\}, \qquad a_i, c_i \in \R^n,\ b_i, d_i, \lambda_i \in \R .
$$

The *$k$-th order statistic* (Def. 3.17, p. 7) returns the $k$-th largest entry of $x$; for $k = \lfloor n/2\rfloor$ this is the median.

Submodularity (Def. D.6 / Eq. 2, p. 22) and the base polytope (Def. D.11, p. 23):

$$
F(A) + F(B) \ge F(A\cup B) + F(A\cap B) \ \ \forall A,B \subseteq [n], \qquad
B(F) := \Big\{x \in \R^n \mid \textstyle\sum_{i\in S} x_i \le F(S)\ \forall S\subset[n],\ \sum_{i\in[n]} x_i = F([n])\Big\}.
$$

For submodular $F$, its Lovász extension is the support function of $B(F)$, and $B(F)$ is the Newton polytope $\mathrm{Newt}(f) = \mathrm{conv}(v_1,\dots,v_k)$ of that extension (pp. 17, 23).

| Symbol | Meaning (in this paper) |
|---|---|
| $f$ | CPWL function being decomposed |
| $g, h$ | convex CPWL functions with $f = g-h$ |
| $P$ | fixed complete polyhedral complex; $P^d$ its $d$-dimensional faces |
| $P_f$ | unique coarsest complex compatible with $f$ |
| $\sigma, \tau$ | faces of codimension 1 and 2 |
| $e_{\sigma/\tau}$ | unit normal of $\tau$ pointing into $\sigma$ |
| $w_f$ | balanced weight function representing $f$ |
| $\mathcal{V}_P$, $\mathcal{V}_P^+$ | CPWL functions compatible with $P$; the convex ones |
| $W_P$, $W_P^+$ | weight functions; nonnegative ones |
| $DP(f)$ | decomposition polyhedron, $\pi$ its projection $\pi(g,h)=g$ |
| $q, k$ | number of pieces; number of affine components ($k \le q$) |
| $\mathrm{supp}_P(f)$ | facets $\sigma \in P^{n-1}$ with $w_f(\sigma) \neq 0$ |
| $r, s$ | free depth/size tradeoff parameters with $rs \ge k=q$ |

## Main results

**Structure of $DP(f)$.** $DP(f)$ is a polyhedron = $(\mathcal{V}_P^+ \times \mathcal{V}_P^+) \cap H_f$ with $H_f = \{(g,h) \mid f = g-h\}$ (Thm. 3.5). Reduced decompositions $\leftrightarrow$ bounded faces (Thm. 3.8); no non-trivial coarsening $\leftrightarrow$ vertex of $DP(f)$ $\leftrightarrow$ vertex of $D_Q(f)$ for every compatible $Q$ (Thm. 3.11); minimal $\Rightarrow$ vertex (Thm. 3.13). A decomposition can be certified as the unique minimal one if $\mathrm{supp}^+(f) = \mathrm{supp}_P(g)$ and $\mathrm{supp}^-(f) = \mathrm{supp}_P(h)$, in which case $g,h$ have at most as many pieces as $f$ (Prop. 3.15, p. 7). Reduced decompositions are exactly the optima of an LP over $\pi(DP(f))$ with objective in the interior of the dual cone of $\mathcal{V}_P^+$, e.g. the all-ones functional $u(\sigma)=1$ (Thm. 3.18, p. 7).

**Existing constructions fail to be minimal.** For the $k$-th order statistic both the hyperplane-extension and local-maxima constructions (Construction B.1, B.2) yield decompositions that are not vertices of $DP(f)$ for any regular $P$ (Prop. 4.1, p. 8; Examples A.3, B.4, B.5). Prop. B.3 shows the suboptimality is caused by the underlying complexes, not the construction recipe.

**Tran & Wang (2024) does not generalize.** Their 2-dimensional minimal construction (Thm. C.1) is valid, but their proposed reduction of the $n$-dimensional case to 2-dimensional projections requires "gluing" local polygons, which is not always well-defined. Example C.3 exhibits four polygons (Fig. 4) whose edge-identification linear system has no solution, for the explicit $f(x) = \max\{0,\max_{i\neq j}\min\{x_i, x_j - x_i\}\}$ on $\R^3$. Finding a finite algorithm that guarantees a minimal decomposition *without* fixing a complex is left open (§4, p. 8).

**Submodular functions.** $\Phi(f)(S) = f(\mathbb{1}_S)$ is an isomorphism onto set functions (Prop. D.2) under which submodular $\leftrightarrow$ convex and modular $\leftrightarrow$ affine, so the whole framework transfers: the set of decompositions into submodular functions (modulo modular) is a polyhedron = intersection of two shifted copies of the submodular cone, irreducible $\leftrightarrow$ bounded faces, minimal $\Rightarrow$ vertex (Corollary 5.4, p. 9). For cut functions $F(S) = \sum_{\{u,v\}\in\delta(S)} c(\{u,v\})$ the decomposition into positive- and negative-weight cuts is the unique minimal one (Example D.14, p. 24).

**ReLU constructions.** For convex CPWL $f : \R^n \to \R$ with $k$ affine components, Theorem 6.3 gives networks of depth $\lceil \log_2(n+1)\rceil + \lceil\log_2 r\rceil + 1$ and size $O(rs^{n+1})$ for any $r,s$ with $rs \ge k = q$; $r=1,s=k$ recovers the low-depth bound of Hertrich et al. (2021), $r=k,s=1$ gives depth $O(\log k)$, size $O(k)$, close to Chen et al. (2022). Corollary 6.4 extends this to nonconvex $f$ after a decomposition, using $\tilde q = \lvert P^n\rvert$ full-dimensional pieces of an underlying regular complex.

## Limitations stated by the authors

The work is fundamental theory with no direct practical improvement and no implementation (§1, pp. 2–3). Fixing an underlying compatible complex restricts the search space — the assumption is defended by the structural results it enables but acknowledged as a limitation. For the nonconvex network construction the open question is how much of a restriction regularity of $P$ (and $\tilde q$ vs. $q$) really is (§6, p. 10). Problem 7.1 asks whether $g$ and $h$ can always be polynomial in $n$ and $q$ many pieces; they connect it to the coarsest regular refinement of an arbitrary complete complex, and note a negative answer would be equally interesting (§7, p. 10).

## Critical assessment

The main contribution is conceptual: turning a decomposition search into polyhedral geometry and thereby locating "good" decompositions on faces and vertices of an explicit polyhedron. Theorems 3.8/3.11/3.13 are clean and, once Lemma 3.2 (the weight parameterization) is in place, the arguments are short — the depth here lies in the setup, and the paper is honest that the real difficulty (globally minimal decompositions, hence Problem 1.1) is untouched: minimality is only relative to a fixed $P$, and vertex enumeration of $DP(f)$ may be exponentially expensive. The refutation of Tran & Wang (2024) is convincing and self-contained (an explicit unsolvable linear system, Example C.3), but it refutes only one proposed route, and the gap between "this approach fails" and "no finite algorithm exists" is correctly left as an open problem rather than overstated. The network tradeoff (Thm. 6.3) is a genuine but modest improvement: it interpolates two known constructions and its size bound rests on convexity; for nonconvex functions Corollary 6.4 pays with $\tilde q$, whose relation to $q$ is the paper's own open question.

I would flag two presentation issues. The main text states $k \le q \le k!$ with both inequalities possibly strict (§2, p. 4); the upper bound $q \le k!$ (sic) does not appear consequential later and reads as a typo or an over-strong claim — the arguments that follow use only $k \le q$ and $n < k \le q$. Theorem 6.2 is quoted with a symbol $p$ ($\lceil\log_2 p\rceil$) that is not defined in the surrounding text. There is no code or data, consistent with a pure-theory paper, and no empirical evaluation is claimed. The most valuable follow-up would be an algorithmic instantiation of Theorem 3.18 (linear programming over $DP(f)$) and an answer to Problem 7.1.

## Connections within the collection

The paper leans on the duality between convex CPWL functions and polytopes and on the tropical balancing condition — the machinery behind [@zhangTropicalGeometryDeep]'s tropical-geometry view of ReLU networks, which it cites for the geometry of network functions (§1.2, p. 3). Its account of ReLU complexity (number of pieces and affine components) is the same complexity ladder studied across this collection, e.g. [@NIPS2014_fa6f2a46], [@serraBoundingCountingLinear2018], [@haninComplexityLinearRegions2019] and [@goujonNumberRegionsPiecewise2023], though this paper reasons about exact network *constructions* for a given function rather than counting regions of a fixed architecture; the constructions it improves on (Hertrich et al. 2021; Chen et al. 2022) are not in the collection. The 2-dimensional minimal decomposition it refutes is Tran & Wang (2024), also outside the collection. Its tropical-geometric treatment of decision/breakpoint structure is in the same spirit as [@alfarraDecisionBoundariesNeural2023] (possibly related; the two do not cite each other). The polyhedral-complex view of a CPWL function's linearity regions is shared with the region-complex analyses of [@berzinsPolyhedralComplexExtraction2023], [@masdenAlgorithmicDeterminationCombinatorial2022] and [@liuReLUNeuralNetworks2023], which extract the very complexes that this paper takes as fixed input.
