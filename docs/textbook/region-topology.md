---
chapter: region-topology
title: Topology of regions and of data
outline_hash: db41df084c9f
built_from:
  berzinsGeometryInformedNeuralNetworks2024: a6bc989abbb9
  berzinsPolyhedralComplexExtraction2023: 208bc568db8f
  chazalIntroductionTopologicalData2021: cef80f43dbc4
  liuReLUNeuralNetworks2023: 7069ff27d8f7
  masdenAlgorithmicDeterminationCombinatorial2022: ea8fb21c3649
  sattelbergLocallyLinearAttributes2023: 4e0708abcb09
---

# Topology of regions and of data

!!! abstract "In this chapter"
    - A network's region complex is a polyhedral complex, and a point cloud is a metric space: both become finite combinatorial objects once one builds a complex on them.
    - Simplicial complexes — chiefly the Vietoris–Rips complex — turn a point cloud into a *filtered* family, and persistent homology reads off the homology of that family scale by scale.
    - The persistence diagram is a stable summary: the bottleneck distance to the true diagram is controlled by the Hausdorff, Gromov–Hausdorff or Wasserstein distance to the data.
    - Distance-to-measure replaces the distance function when the data has outliers, and restores exactly that stability with respect to the Wasserstein metric.
    - The *exact* topology of a network's decision boundary is computable from Masden's sign-sequence cubical complex; the persistent homology of the Hamming metric on activation patterns is a cheap, approximate alternative.
    - Prerequisites: [The region complex of a ReLU network](relu-network-geometry.md).

A feedforward ReLU network cuts its input space into a polyhedral complex [@berzinsPolyhedralComplexExtraction2023, §3.1, p. 3], and the previous chapter built that complex. This chapter asks what *shape* it has. Two different objects are in play. The first is the network's own region complex: the cells, their shared faces, and in particular the decision boundary, the set where the network's output is zero [@masdenAlgorithmicDeterminationCombinatorial2022, §4.2, p. 7]. The second is the data the network is used on: a finite point cloud, which carries no topology at all until we put one on it. Both questions are answered with the same toolkit — build a complex, compute its homology, and track how the homology changes as a scale parameter grows — and the second question is answered for the first because that toolkit is *stable*: its output moves no faster than the input does.

This is the entry point into topological data analysis (TDA) for the rest of the textbook, so this chapter develops the TDA vocabulary (simplicial complexes, filtrations, persistence diagrams, distance-to-measure, Mapper) at the level a reader needs, and then applies it to the region complex. I keep the shared [notation](../notation.md): $\mathcal{P}(f)$ is the polyhedral complex of linear regions, $\mathbf{s}(x)$ the activation pattern, $\mathcal{N}(f)$ the region count. For the topology I use $\beta_k$ for Betti numbers, $\mathrm{VR}_t$ for the Vietoris–Rips complex at scale $t$, $\mathrm{dgm}_k(f)$ and $\mathrm{Bar}_k$ for the persistence diagram and barcode, and $d_B$ for the bottleneck distance, following [@chazalIntroductionTopologicalData2021, §5]. Where a paper deviates I say so.

## From a point cloud to a simplicial complex

A finite set of points with a distance on it is, topologically, just a discrete space: every point is its own connected component and there are no loops or voids. The whole art is to enrich it. The survey of [@chazalIntroductionTopologicalData2021] organises the field around a four-step pipeline [@chazalIntroductionTopologicalData2021, §1, pp. 1–2]: (i) take the data as a metric space, with a distance either given a priori or supplied as an application-dependent pairwise matrix; (ii) build a continuous shape on top of it — typically a simplicial complex, or a nested family of them called a filtration; (iii) extract topological or geometric invariants from that shape; and (iv) use those invariants, which are stable, as features for learning. The authors warn at the first step that the choice of metric "may be critical" [@chazalIntroductionTopologicalData2021, §1, p. 2]; the network's own Hamming metric, introduced in the last section of this chapter, is exactly such a non-default choice.

Two complexes are standard on a finite metric space $X$. The *Vietoris–Rips complex* $\mathrm{VR}_\alpha(X)$ contains a simplex whenever every pair of its vertices is at distance at most $\alpha$; the *Čech complex* $\check{C}ech_\alpha(X)$ contains it whenever the closed balls of radius $\alpha$ about its vertices have a common point [@chazalIntroductionTopologicalData2021, §2, p. 5]. The two interleave,

$$
\mathrm{VR}_\alpha(X)\ \subseteq\ \check{C}ech_\alpha(X)\ \subseteq\ \mathrm{VR}_{2\alpha}(X),
$$

so the combinatorially cheaper Rips complex is a two-sided proxy for the geometrically faithful Čech complex [@chazalIntroductionTopologicalData2021, §2, p. 5]. That is the interleaving that later makes Rips complexes usable both for homology inference and for persistence.

What makes the Čech complex the right shape is the *Nerve Theorem*: if the finite intersections of an open cover are empty or contractible, the union of the cover and its nerve — the abstract complex whose vertices are the sets, with a simplex per non-empty intersection — are homotopy equivalent [@chazalIntroductionTopologicalData2021, §2, Thm. 1, p. 6]. Applied to the cover by balls, it says $\check{C}ech_r(X)$ is homotopy equivalent to the union of balls $\bigcup_{x\in X} B(x,r)$ [@chazalIntroductionTopologicalData2021, §2, p. 6]. This single theorem is the reason a *cover* can be replaced by a *combinatorial* object without losing topology, and it is reused verbatim by a ReLU network's region complex: the complex is a polyhedral one, and a polyhedral complex is the more general relative of a simplicial complex — its cells are closed under taking faces and the intersection of two cells is their common face [@berzinsPolyhedralComplexExtraction2023, §3.1, p. 3].

!!! note "Background"
    A *simplicial complex* is a family of simplices closed under taking faces; its homology with $\mathbb{Z}_2$ coefficients is $H_k(K)=Z_k(K)/B_k(K)$ for the $k$-cycles and $k$-boundaries, and the $k$-th *Betti number* is $\beta_k(K)=\dim H_k(K)$ [@chazalIntroductionTopologicalData2021, §4.2, Def. 6, p. 13]. $\beta_0$ counts connected components, $\beta_1$ loops, $\beta_2$ cavities. Betti numbers are invariants of homotopy type: homotopy-equivalent complexes have isomorphic homology and the same Betti numbers [@chazalIntroductionTopologicalData2021, §4.2, p. 13].

## Filtrations and persistent homology

A single scale $r$ is a poor choice — too small and the complex is disconnected, too large and everything is filled in — and the survey's remedy is to keep every scale at once. A *filtration* of a complex $K$ is a nested family $(K_r)_{r\in T}$, $K_r\subseteq K_s$ for $r\le s$, with $K=\bigcup_r K_r$; the Rips and Čech complexes over all radii are the standard examples, and the sublevel sets of a function give another [@chazalIntroductionTopologicalData2021, §5.1, pp. 17–18]. As $r$ grows, connected components appear and merge, loops appear and are filled: homology is not a number but a *history*.

Persistent homology encodes that history. Each homology class is given a birth time $b$ and a death time $d$, and the collection of intervals $(b,d)$ is the *persistence barcode*; representing each interval by the point $(b,d)\in\mathbb{R}^2$ gives the *persistence diagram*, a multiset above the diagonal to which the diagonal itself is added with infinite multiplicity [@chazalIntroductionTopologicalData2021, §5.2, pp. 18–19]. The algebraic definition goes through persistence modules: the homology groups together with the maps induced by inclusions form an indexed family of vector spaces, and when it decomposes into *interval modules* the decomposition is unique up to reordering, so the diagram is well defined [@chazalIntroductionTopologicalData2021, §5.3, Thm. 5, p. 22]. When the module is not decomposable, the weaker $q$-tame condition — every linear map has finite rank — still guarantees a diagram [@chazalIntroductionTopologicalData2021, §5.3, Def. 8, Thm. 6, p. 22].

Diagrams are "highly non linear" and cannot be fed to a learner directly [@chazalIntroductionTopologicalData2021, §6.5, p. 33], so the survey presents two families of numerical summaries. A *persistence landscape* turns each birth–death pair into a tent function

$$
\Lambda_p(t)=
\begin{cases}
t-b, & t\in\bigl[b,\tfrac{b+d}{2}\bigr],\\
d-t, & t\in\bigl(\tfrac{b+d}{2},d\bigr],\\
0, & \text{otherwise},
\end{cases}
\qquad
\lambda_{\mathrm{dgm}}(k,t)=k\text{max}_{p\in\mathrm{dgm}}\,\Lambda_p(t),
$$

and the $k$-th landscape is a $1$-Lipschitz function; the map from diagram to landscape is injective, so no information is lost [@chazalIntroductionTopologicalData2021, §5.4, Eq. 3–4, p. 23]. More generally a *linear representation* is a sum of a vector-valued function over the points of the diagram, $\Psi_k(\mathrm{dgm}_k)=\sum_{p\in\mathrm{dgm}_k} f_k(p)$, with Betti curves, persistence silhouettes and persistence surfaces as special cases [@chazalIntroductionTopologicalData2021, §5.5, Eq. 5, p. 24]. These are the objects that enter a neural network as features. To compare two diagrams there is the *bottleneck distance*

$$
d_B(\mathrm{dgm}_1,\mathrm{dgm}_2)=\inf_{\text{matching } m}\ \max_{(p,q)\in m}\ \lVert p-q\rVert_\infty,
$$

a matching pairing every off-diagonal point of each diagram to a point of the other or to the diagonal, computable as a bipartite perfect matching; the $p$-Wasserstein analogue replaces the max by a sum of $p$-th powers [@chazalIntroductionTopologicalData2021, §5.6, pp. 25–26].

## Stability: why these invariants can be estimated

The reason persistence is usable at all is that the diagram moves slowly when the data moves. The cleanest statement is algebraic: two persistence modules that are $\delta$-interleaved have diagrams at bottleneck distance at most $\delta$ [@chazalIntroductionTopologicalData2021, §5.7, Thm. 7, p. 26]. Every concrete stability statement is a corollary. For two functions $f,g$ on a space, $d_B(\mathrm{dgm}_k(f),\mathrm{dgm}_k(g))\le\lVert f-g\rVert_\infty$ [@chazalIntroductionTopologicalData2021, §5.7, Thm. 8, p. 27]; for the Vietoris–Rips or Čech filtrations of two compact metric spaces,

$$
d_B\bigl(\mathrm{dgm}(\mathrm{Filt}(X)),\mathrm{dgm}(\mathrm{Filt}(Y))\bigr)\le 2\,d_{GH}(X,Y),
$$

where $d_{GH}$ is the Gromov–Hausdorff distance [@chazalIntroductionTopologicalData2021, §5.7, Thm. 9, p. 27] — so topology can be compared even between spaces that are not embedded in the same Euclidean space. Landscapes inherit the same bound, $\lvert\lambda(k,t)-\lambda'(k,t)\rvert\le d_B(\mathrm{dgm},\mathrm{dgm}')$ [@chazalIntroductionTopologicalData2021, §5.7, Prop. 1, p. 27].

Stability turns estimation into a statistical problem. If the sample $X_n$ is drawn i.i.d. from a measure supported on a compact $X_\mu\subset M$ and the measure satisfies the $(a,b)$-standard assumption $\mu(B(x,r))\ge\min(ar^b,1)$ for all $x$ in the support and all $r>0$, then the expected bottleneck distance from the sample diagram to the population diagram is at most $C(\log n/n)^{1/b}$, with $C$ depending only on $a,b$ [@chazalIntroductionTopologicalData2021, §6.1, Eq. 6, Thm. 10, p. 28]. The matching lower bound holds up to a logarithmic term [@chazalIntroductionTopologicalData2021, §6.1, p. 29]. There is a price: stability and discriminative power trade off. Continuity of a linear representation holds only for weight functions that vanish near the diagonal, whereas "in general there is no specific reason to consider that points close to the diagonal are less important than others" [@chazalIntroductionTopologicalData2021, §5.7, pp. 27–28], and the confidence bands the survey builds from the stability theorems are the two it calls the most conservative or least proven — the subsampling band and the bottleneck bootstrap [@chazalIntroductionTopologicalData2021, §6.2, p. 30; §7.1, p. 35].

!!! warning "A clash with the network literature"
    In [@chazalIntroductionTopologicalData2021] and in this section $d_H$ would denote the Hausdorff distance between compact sets, and $d_{GH}$ the Gromov–Hausdorff distance. The [shared notation](../notation.md) reserves $d_H$ for the *Hamming* distance between activation patterns. This chapter writes $d_{\mathrm{Haus}}$ for the Hausdorff distance and keeps $d_H$ for Hamming.

## Geometric inference and distance-to-measure

Persistent homology on a pure point cloud is one route to topology; the *geometric inference* route replaces arbitrary covers by a union of balls of radius $r$ about the sample, and asks when that union has the topology of the unknown support $M$. The regularity needed is a reach-type condition. The $r$-offset $K^r$ is the $r$-sublevel set of the distance function $d_K$; the relevant functions are the *distance-like* ones, proper $\phi$ for which $x\mapsto\lVert x\rVert^2-\phi^2(x)$ is convex [@chazalIntroductionTopologicalData2021, §4.1, Def. 4, p. 11]. Regularity is quantified by the weak feature size $\mathrm{wfs}_\phi(r)$ and the $\alpha$-*reach*, which generalises Federer's reach [@chazalIntroductionTopologicalData2021, §4.1, Def. 5, p. 11]. The reconstruction theorem then states that if $\lVert\phi-\psi\rVert_\infty<\varepsilon$ and the $\alpha$-reach of $\phi$ is at least $R$, the sublevel sets $\psi_r$ and $\phi_\eta$ are homotopy equivalent for $r\in[4\varepsilon/\alpha^2,\,R-3\varepsilon]$ and $\eta\in(0,R)$ provided $\varepsilon\le R/(5+4/\alpha^2)$ [@chazalIntroductionTopologicalData2021, §4.1, Thm. 2, p. 11]. Taking $\phi=d_M$ and $\psi=d_{X_n}$ relates the sample's offsets to the true support. The Betti numbers follow: for a smooth $m$-dimensional submanifold, $\beta_k(\check{C}ech_r(X))=\beta_k(M)$ for $k=0,\dots,m$ [@chazalIntroductionTopologicalData2021, §4.2, Thm. 3, p. 14], and one can avoid the Čech emptiness predicate entirely, since under a weak-feature-size condition $\beta_k(X^\eta)=\mathrm{rk}\bigl(H_k(\mathrm{VR}_r(X))\to H_k(\mathrm{VR}_{4r}(X))\bigr)$ for a whole interval of radii [@chazalIntroductionTopologicalData2021, §4.2, Thm. 4, pp. 14–15].

Three difficulties remain, and the survey states them plainly: the $\alpha$-reach regularity "may be too restrictive"; computing the nerve of a union of balls needs the tricky emptiness-of-a-union-of-balls predicate; and the Betti-number estimate depends on the scale $r$, whose choice "may be a problem" [@chazalIntroductionTopologicalData2021, §4.2, p. 14]. Theorem 4 addresses the first two, not the third.

The fourth and most important gap is robustness to outliers: a single outlier can change the distance function dramatically and with it the topology of every offset [@chazalIntroductionTopologicalData2021, §4.4, p. 16]. The *distance-to-a-measure* repairs this. For a probability measure $P$, the quantile distance is $\delta_{P,u}(x)=\inf\{t>0:P(B(x,t))\ge u\}$, and the DTM with parameter $m\in[0,1]$ and power $r\ge1$ is

$$
d_{P,m,r}(x):=\left(\frac{1}{m}\int_0^m \delta_{P,u}^r(x)\,\mathrm{d}u\right)^{1/r}.
$$

Its defining property is stability in the Wasserstein metric: the map $P\mapsto d_{P,m,r}$ is $m^{-1/r}$-Lipschitz, $\lVert d_{P,m,r}-d_{\tilde P,m,r}\rVert_\infty\le m^{-1/r}W_r(P,\tilde P)$, so measures close in Wasserstein distance have close sublevel sets; and for $r=2$ the square $d_{P,m,2}^2$ is semiconcave, giving the regularity that makes its sublevel sets inferable [@chazalIntroductionTopologicalData2021, §4.4, Eq. 1–2, p. 16]. In practice $P$ is known only through a sample, and plugging in the empirical measure gives the DTEM, computable from pairwise distances alone [@chazalIntroductionTopologicalData2021, §4.4, p. 17].

## Mapper: a graph summary of a point cloud

One further tool from the same toolkit belongs here, because the network literature uses it. Mapper is the Nerve Theorem applied to the *image* of the data. Given a filter (lens) function $f:X\to\mathbb{R}^d$ and a cover $\mathcal{U}$ of $f(X)$, the refined pull-back cover is the collection of connected components of the sets $f^{-1}(U)$, $U\in\mathcal{U}$, and Mapper is its nerve [@chazalIntroductionTopologicalData2021, §3, Def. 2, p. 7]. For well-chosen covers that nerve is a graph, so the output is visualisable; the algorithm clusters each preimage $f^{-1}(U)$ and links clusters whose preimages meet [@chazalIntroductionTopologicalData2021, §3, Algorithm 1, pp. 7–8].

Every ingredient is a user choice. The filter may be a density estimate, coordinates from a non-linear dimension reduction, or one of the parameter-free functions the survey singles out — the centrality function $f(x)=\sum_y d(x,y)$ or the eccentricity function $f(x)=\max_y d(x,y)$ [@chazalIntroductionTopologicalData2021, §3, p. 8]. The cover of a real-valued filter is usually a set of regularly spaced intervals of equal *resolution* $r$ overlapping by a *gain* $g$; if $g<50\%$ every point of the line lies in at most two intervals and the nerve is a graph [@chazalIntroductionTopologicalData2021, §3, p. 9]. And the method is unstable: "the output of the Mapper is very sensitive to the choice of $\mathcal{U}$ and small changes in the resolution and gain parameters may results in very large changes in the output" [@chazalIntroductionTopologicalData2021, §3, p. 9]. The collection's nearest use of the same idea is the clustering of local affine attributes of regions: [@sattelbergLocallyLinearAttributes2023, §2.4.1, p. 8] stacks each output's sampled local weight-bias vectors into a point cloud and applies $k$-means, which is exactly the "cluster the preimages and take the nerve" pattern with a trivial filter.

## The exact topology of a network's region complex

Now to the network's own complex. A ReLU network's decision boundary is the zero set $\{x:f(x)=0\}$, a union of $(n_0-1)$-cells of $\mathcal{P}(f)$, and one can ask for its Betti numbers exactly [@masdenAlgorithmicDeterminationCombinatorial2022, §3.1, p. 3; §4.2, p. 7]. Masden's sign-sequence cubical complex is built for this. Label every cell $C$ of $\mathcal{P}(f)$ — not only the top-dimensional ones — by

$$
s(C)\in\{-1,0,1\}^N,\qquad s_{ij}(C)=\mathrm{sgn}\bigl(F_{ij}(C)\bigr),
$$

with $N$ the number of intermediate neurons and $F_{ij}$ the $(i,j)$-th node map; a zero entry marks exactly the bent hyperplanes the cell lies on [@masdenAlgorithmicDeterminationCombinatorial2022, §3.2, Def. 13, p. 3]. This map is well defined and injective on cells [@masdenAlgorithmicDeterminationCombinatorial2022, §3.2, Thm. 14, p. 3]. For a generic, supertransversal network the number of zeros equals the codimension, so a $k$-cell has exactly $n_0-k$ zeros [@masdenAlgorithmicDeterminationCombinatorial2022, §3.2, Lemma 16, p. 4]; and the image of $s$ is a *pure* $n_0$-dimensional subcomplex $S(F)$ of the hypercube $[-1,1]^N$, geometrically dual to $\mathcal{P}(f)$ in the sense that $k$-cells of $S(F)$ correspond to codimension-$k$ cells of $\mathcal{P}(f)$ and the two face posets are opposite [@masdenAlgorithmicDeterminationCombinatorial2022, §3.2, Thm. 20, p. 4]. Because $S(F)$ is pure, the sign sequences of the *vertices* of $\mathcal{P}(f)$ determine its whole face poset [@masdenAlgorithmicDeterminationCombinatorial2022, §3.2, Corollary, p. 4]. The product

$$
\bigl(S(C)\cdot S(D)\bigr)_{ij}=
\begin{cases}
S(C)_{ij}, & S(C)_{ij}\neq 0,\\
S(D)_{ij}, & \text{otherwise},
\end{cases}
$$

is well defined and makes the sign sequences a semigroup; it yields purely algebraic face tests, $C\le D$ iff $S(C)\cdot S(D)=S(D)$ [@masdenAlgorithmicDeterminationCombinatorial2022, §3.3, Lemmas 18–19, pp. 4–5].

The point of this apparatus for topology is the decision boundary. The cells with last sign $S(C)_N=0$ are precisely those with $F(C)=0$. Restricting the mod-two cellular coboundary of the cubical complex to those cells — whose last sign is also zero — gives a cochain complex dual to the boundary of the decision-boundary cells, and with a single "vertex at infinity" adjoined to the unbounded edges this is the cochain complex of the *one-point compactification* of the decision boundary; its Betti numbers come out of any standard chain-complex software [@masdenAlgorithmicDeterminationCombinatorial2022, §4.2, p. 7]. The compactification makes the bounded and unbounded components separable: their numbers are $\beta_0-1$ and $\beta_{n_0-1}-\beta_0+1$ respectively [@masdenAlgorithmicDeterminationCombinatorial2022, §5.2, p. 8].

Masden runs this on fifty randomly initialised networks per architecture at initialisation, and finds the first exact decision-boundary topology for input dimension above two with more than one hidden layer [@masdenAlgorithmicDeterminationCombinatorial2022, §5, p. 8]. The reported pattern is that shallow networks have nearly constant topology across input dimensions, while deep networks are more variable and shift with width. Bounded components are rare and rarer in higher dimension: $8.1\%$ of $(2,n,1)$ networks have at least one, against $0.05\%$ of $(4,n,1)$ [@masdenAlgorithmicDeterminationCombinatorial2022, §5.2, p. 8]. Unbounded components never exceeded three for shallow networks, with mode one (averages $1.0,1.03,1.04$ for $n_0=2,3,4$), while the deep networks averaged $1.1,1.4,1.4$ and reached maxima of $5,7,12$ [@masdenAlgorithmicDeterminationCombinatorial2022, §5.2, p. 8]. Table 2, for instance, gives $\beta_0=1.06\pm0.034$ and $\beta_{n_0-1}=1.06\pm0.059$ for $(2,5,1)$, against $\beta_0=1.04\pm0.03$ and $\beta_{n_0-1}=1.88\pm0.30$ for $(4,8,8,1)$ [@masdenAlgorithmicDeterminationCombinatorial2022, §5.2, p. 9]. The cost is dimension: the algorithm is exponential in $n_0$, so only low-dimensional slices of a real network's boundary can be studied [@masdenAlgorithmicDeterminationCombinatorial2022, §6, p. 10].

The combinatorial object Masden labels is the one [@berzinsPolyhedralComplexExtraction2023] extracts directly. There the sign-vector of a point records its side of every hyperplane of the arrangement, $\sigma_i(x)\in\{+,0,-\}$ [@berzinsPolyhedralComplexExtraction2023, §3.3, Eq. 1, p. 3], and a cell is the intersection of the closed half-spaces its sign-vector selects, $C=\bigcap_i H_i^{\sigma_i(C)}$ [@berzinsPolyhedralComplexExtraction2023, §3.3, Eq. 2, p. 3]. The network's boundaries are the folded hyperplanes, the zero sets $H_i^{(l)}=\{x:W_i^{(l)\top}x^{(l-1)}+b_i^{(l)}=0\}$ of the pre-activations pulled back to the input [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4]. Edge subdivision refines the 1-skeleton neuron by neuron: find the *splitting edges* whose endpoints differ in sign, interpolate to place the new vertex, and add the *intersecting edges* by recovering the splitting 2-faces via perturbation [@berzinsPolyhedralComplexExtraction2023, §4, p. 5]. The perturbation step is what recovers the whole intersection-poset from the 1-skeleton: flipping one zero of a sign-vector at a time yields the parents of a cell, and on a bounded domain a $k$-cell has $z+2(D-k-z)$ of them, with $z$ the number of zeros among the domain's signs [@berzinsPolyhedralComplexExtraction2023, §4.1, p. 6]. The two papers therefore fit together: Berzins extracts the cells and their combinatorial relations from the geometry, Masden turns the same relations into a chain complex whose homology is the decision-boundary topology.

## Persistent homology of the region decomposition

Exact extraction stops at low dimension. The alternative, developed by [@liuReLUNeuralNetworks2023], is to give up exactness and read the topology *approximately* off a distance the network imposes on its own input space. A ReLU network decomposes input space into convex polyhedra, each carrying a binary vector

$$
s_{i,j}(x)=
\begin{cases}
1 & w_{i,j}F_{i-1}(x)+b_{i,j}>0,\\
0 & w_{i,j}F_{i-1}(x)+b_{i,j}\le 0,
\end{cases}
\qquad
s(x)=[s_1(x)^\top\cdots s_L(x)^\top]^\top,
$$

the binary vector of the point [@liuReLUNeuralNetworks2023, §2.2, Eq. 4–5, p. 2]. The decomposition has a dual graph whose vertices are the polyhedra and whose edges join polyhedra sharing an $(m-1)$-face, and the decisive combinatorial fact — the same one the region-complex chapter uses — is that two polyhedra share a facet *if and only if* their binary vectors differ in exactly one bit [@liuReLUNeuralNetworks2023, §2.3, pp. 3–4]. This dual graph is bipartite [@liuReLUNeuralNetworks2023, §2.3, p. 4; App. B, pp. 11–12].

The Hamming distance $d_H$ between binary vectors is then proposed as the metric: it "is an approximation for the smallest number of polyhedral steps between two polyhedra", i.e. for the shortest path on the dual graph, and the paper is explicit that it is an approximation, not an equality [@liuReLUNeuralNetworks2023, §1, p. 1; §2.3, p. 4]. The pipeline is [@chazalIntroductionTopologicalData2021, §1]'s, with the network supplying both the complex and the metric: sample points, record each point's binary vector, form the pairwise Hamming matrix $D$, threshold it at scale $t$ into a $\{0,1\}$ adjacency matrix $A(t)$ with $A(t)_{i,j}=0$ whenever $D_{i,j}>t$, take the clique complex, and compute Vietoris–Rips persistent homology with Ripser [@liuReLUNeuralNetworks2023, §4.1–4.2, Eq. 13–14, pp. 6–7].

The demonstrations are deliberately small. Sampling a circle in $\mathbb{R}^3$ at twenty points through a $3\text{-}10\text{-}10\text{-}10\text{-}10$ network gives twenty binary vectors, nineteen of them distinct; the $H_0$ barcode connects at Hamming distance four, a spurious short-lived loop appears at three, and the genuine circle signal at four [@liuReLUNeuralNetworks2023, §4.2, Example 4.1, p. 7]. At five hundred samples only $41$ of the bit vectors are distinct; $H_0$ now connects at distance two and a long-lived $H_1$ loop begins there [@liuReLUNeuralNetworks2023, §4.2, Example 4.2, p. 7]. The same network's decomposition has $25$ polyhedra in $[-1,1]^2$ and $27$ in $\mathbb{R}^2$ ($21$ bounded, $6$ unbounded), and the larger model has $1858$ in $[-1,1]^3$ and $3331$ in $\mathbb{R}^3$ [@liuReLUNeuralNetworks2023, §3.2, p. 5].

The interesting test is on a network trained for something else entirely. Running the pre-trained ResNet-50 (input dimension $150528$, more than $6{,}000{,}000$ hidden nodes) on three ImageNet validation images, the $50$ points $\sin\theta A_1+\cos\theta A_2$ produce a strong circle signal in the $50\times50$ Hamming matrix together with "homological noise (or homological dust)" in $H_0$, whereas $A_3+\sin\theta A_1+\cos\theta A_2$ produces the same strong circle signal and no noise [@liuReLUNeuralNetworks2023, §4.2, Example 4.3, pp. 7–8]. Two rotated copies of a sample set can be combined by taking the entrywise maximum or minimum of their Hamming matrices — the Boolean **and** and **or** — and on the noise-free example the max strengthened the signal's length while the min made it start earlier, with the authors unable to say which combination is better [@liuReLUNeuralNetworks2023, §4.2, Examples 4.4–4.5, pp. 7–8].

!!! question "Open question"
    Is the Hamming metric of [@liuReLUNeuralNetworks2023] a *stable* metric in the sense of §"Stability" above — that is, is there a bound relating $d_H$ to the true dual-graph distance under a perturbation of the weights? None is given: the relation is stated only as an approximation [@liuReLUNeuralNetworks2023, §1, p. 1], and the empirical evidence is thin (20, 50, 100 samples, barcodes read visually, no error bars). If such a bound existed, the persistent homology of the Hamming metric would be an estimator of the decision-boundary topology of §"The exact topology" in the same way the persistence diagram is an estimator of the support's topology.

## Two topologies of one network

It is worth stating plainly that the chapter has described two different objects, because papers in the collection disagree about which is the useful one.

The first is *exact and combinatorial*: the region complex carries all the face relations, and its restriction to the boundary cells is a cochain complex whose homology is the decision boundary's, with no sampling and no scale parameter [@masdenAlgorithmicDeterminationCombinatorial2022, §4.2, p. 7; @berzinsPolyhedralComplexExtraction2023, §4, p. 5]. Its weakness is dimension: exponential in $n_0$ [@masdenAlgorithmicDeterminationCombinatorial2022, §6, p. 10], and impractical above $D>8$ even for small networks in the extraction pipeline [@berzinsPolyhedralComplexExtraction2023, §1, p. 2].

The second is *approximate and sampled*: persistent homology of a network-imposed metric on the points one actually has [@liuReLUNeuralNetworks2023, §4.2, p. 7]. It scales to ResNet-50 precisely because it never enumerates the complex, but it is an approximation whose error is unquantified, and its output depends on the scale at which the barcode is read.

Masden takes an explicit position on the trade-off, and it reads as a caution to the second camp: approximation methods based on persistent homology "are dependent on the geometry of the network function", and where the decision boundary has high curvature — exactly the locations where a network is most vulnerable to adversarial examples — the approximation "may lead to inaccuracy between the true topology of a network's decision boundary and the topology which is approximated by persistent homology methods" [@masdenAlgorithmicDeterminationCombinatorial2022, §6, p. 10]. The proposed remedy is to use the exact topology as a ground truth against which the approximate one is compared; the collection does not yet carry out that comparison.

!!! tip "When to use which"
    If you can afford the exact complex — low input dimension, small network, and a question about a specific boundary — compute it, and get Betti numbers with no scale parameter [@masdenAlgorithmicDeterminationCombinatorial2022, §4.2, p. 7]. If the network is large or the input is high-dimensional, the Hamming-metric barcodes of [@liuReLUNeuralNetworks2023] are cheap and demonstrably detect the homology of manifolds sampled in input space, but treat them as a coarse, unquantified proxy [@liuReLUNeuralNetworks2023, §1, p. 1].

## Summary

- TDA's pipeline is metric data to a complex (or filtration) to invariants to features, and the choice of metric is itself a modelling decision that "may be critical" [@chazalIntroductionTopologicalData2021, §1, pp. 1–2].
- Vietoris–Rips and Čech complexes interleave as $\mathrm{VR}_\alpha\subseteq\check{C}ech_\alpha\subseteq\mathrm{VR}_{2\alpha}$; the Nerve Theorem lets the cheaper Rips complex stand in for the union of balls [@chazalIntroductionTopologicalData2021, §2, p. 5].
- Persistent homology turns a filtration into a barcode or diagram, summarised by landscapes or linear representations that live in a function space [@chazalIntroductionTopologicalData2021, §5.2–5.5, pp. 18–24].
- Diagrams are stable: bottleneck distance is bounded by function sup-distance, by $2d_{GH}$ for Rips/Čech filtrations, and by the Wasserstein distance for the DTM, which is what makes them estimators [@chazalIntroductionTopologicalData2021, §5.7, pp. 26–27; §4.4, pp. 16–17].
- Distance-to-measure is $m^{-1/r}$-Lipschitz in the Wasserstein metric and, for $r=2$, semiconcave, so its sublevel sets are robust to outliers [@chazalIntroductionTopologicalData2021, §4.4, Eq. 1–2, p. 16].
- Mapper is the nerve of the refined pull-back cover of a filter, gives a graph for well-chosen covers, and is unstable in its resolution and gain parameters [@chazalIntroductionTopologicalData2021, §3, pp. 7–9].
- Masden's sign-sequence cubical complex is dual to $\mathcal{P}(f)$ and its restriction to boundary cells is a cochain complex computing the decision boundary's Betti numbers exactly; bounded and unbounded components are $\beta_0-1$ and $\beta_{n_0-1}-\beta_0+1$ [@masdenAlgorithmicDeterminationCombinatorial2022, §3.2, p. 4; §4.2, p. 7; §5.2, p. 8].
- The Hamming distance between activation patterns is an unquantified approximation to dual-graph distance, yet persistent homology of the resulting metric recovers loops in manifolds sampled through ResNet-50 [@liuReLUNeuralNetworks2023, §1, p. 1; §4.2, p. 7].

## Further reading

- [@chazalIntroductionTopologicalData2021]: the whole TDA toolchain; read §2 for complexes and the Nerve Theorem, §4 for geometric inference and the DTM, §5 for persistence and its stability, §6 for the statistics.
- [@liuReLUNeuralNetworks2023]: the Hamming metric on activation patterns and the persistent homology of the polyhedral decomposition; read §2–§3 for the binary vectors and active bits, §4.2 for the circle and ResNet-50 examples.
- [@masdenAlgorithmicDeterminationCombinatorial2022]: the sign-sequence cubical complex and the exact decision-boundary topology; read §3.2–§3.3 for the duality and the product, §4.2 for the boundary cochain complex, §5.2 for the Betti-number statistics.
- [@berzinsPolyhedralComplexExtraction2023]: exact extraction of the region complex from its 1-skeleton; read §3.3–§3.4 for sign-vectors and folded hyperplanes and §4 for edge subdivision and the perturbation step that recovers the intersection-poset.
- [@sattelbergLocallyLinearAttributes2023]: for the clustering of local affine attributes, the collection's nearest analogue of the Mapper summary; see §2.4.
- [@berzinsGeometryInformedNeuralNetworks2024]: for a downstream use of persistent homology together with the geometry of a network, in the geometry-informed-network setting.

## Exercises

**Exercise 1** ★ (check). Explain why a finite point cloud, seen only as a metric space, has trivial topology, and what the Vietoris–Rips complex adds to it.

??? success "Solution"
    A finite metric space is a discrete set of points; its only topological features are
    its connected components, one per point, so $\beta_0=n$ and $\beta_k=0$ for $k>0$.
    The Rips complex at scale $\alpha$ fills in a simplex whenever every pair of its
    vertices is at distance at most $\alpha$, so as $\alpha$ grows the points become
    connected and higher-dimensional simplices appear, exactly recording the scale at
    which the data acquire loops and voids [@chazalIntroductionTopologicalData2021,
    §2, p. 5; §5.2, pp. 18–19]. Without a complex on top, the metric alone tells you the
    distances but not the shape.

**Exercise 2** ★ (check). The Čech complex is the geometrically faithful one, yet computations are usually done with Rips complexes. Why is that legitimate?

??? success "Solution"
    They interleave: $\mathrm{VR}_\alpha(X)\subseteq\check{C}ech_\alpha(X)\subseteq
    \mathrm{VR}_{2\alpha}(X)$, because a simplex whose pairwise distances are at most
    $\alpha$ has all its balls of radius $\alpha$ meeting at a common point, and any
    simplex in the Čech complex has pairwise distances at most $2\alpha$
    [@chazalIntroductionTopologicalData2021, §2, p. 5]. This interleaving transfers to
    persistence modules: the diagrams of the two filtrations are at bottleneck distance
    at most $\log$-bounded by the scale, which is what makes the cheaper complex a
    two-sided proxy [@chazalIntroductionTopologicalData2021, §5.7, Thm. 7, p. 26].

**Exercise 3** ★★ (derive). Starting from the stability of persistence modules under interleaving [@chazalIntroductionTopologicalData2021, §5.7, Thm. 7, p. 26], derive the statement that for two filtrations built on compact metric spaces $X,Y$, $d_B(\mathrm{dgm}(\mathrm{Filt}(X)),\mathrm{dgm}(\mathrm{Filt}(Y)))\le 2\,d_{GH}(X,Y)$.

??? success "Solution"
    Let $\delta=d_{GH}(X,Y)$. The Gromov–Hausdorff distance bounds how badly the scale
    parameters of the two filtrations can be shifted: a Rips simplex present at scale
    $r$ in $X$ corresponds to one present at scale $r+\delta$ in $Y$, and conversely,
    because pairwise distances are perturbed by at most $\delta$. Hence the two
    persistence modules are $\delta$-interleaved, and Theorem 7 gives $d_B\le\delta$
    for directly interleaved modules; the factor $2$ in the statement accounts for
    realising the abstract correspondence through a common ambient metric space, which
    doubles the discrepancy in the scale parameter [@chazalIntroductionTopologicalData2021,
    §5.7, p. 27]. The result is why topological summaries can be compared across data
    sets that are not embedded in the same space.

**Exercise 4** ★★ (compare). [@masdenAlgorithmicDeterminationCombinatorial2022] computes decision-boundary Betti numbers exactly, while [@liuReLUNeuralNetworks2023] computes them approximately from persistence. State precisely what each returns and what each assumes, and say which you would trust on a ResNet-50.

??? success "Solution"
    Masden returns the true Betti numbers of the one-point compactification of the
    decision boundary, via a cochain complex built from cells whose last sign is zero,
    with a vertex at infinity for the unbounded edges; it assumes a generic,
    supertransversal network (a full-measure condition) and costs exponential time in
    the input dimension [@masdenAlgorithmicDeterminationCombinatorial2022, §4.2, p. 7;
    §3.1, Lemma 12, p. 3; §6, p. 10]. Liu et al. return the persistence barcodes of a
    Vietoris–Rips filtration in the Hamming metric on activation patterns; they assume
    nothing beyond the network, but the distance is only an approximation to dual-graph
    distance, so the barcodes are a coarse proxy read at a chosen scale
    [@liuReLUNeuralNetworks2023, §1, p. 1; §4.2, p. 7]. On ResNet-50 the exact method is
    out of reach — the input dimension is $150528$ — so the choice is forced: use the
    approximate barcodes, and, per Masden's own caution, be sceptical where the
    boundary is highly curved [@masdenAlgorithmicDeterminationCombinatorial2022, §6,
    p. 10].

**Exercise 5** ★ (check). Why does the distance-to-measure replace the ordinary distance function when the data may contain outliers?

??? success "Solution"
    Adding a single outlier to a point cloud changes the distance function
    dramatically, and with it the topology of every offset [@chazalIntroductionTopologicalData2021,
    §4.4, p. 16]. The DTM averages the quantile distances $\delta_{P,u}$ over the
    lowest-mass fraction $m$ of the measure, so a small number of far-away points
    contributes to a negligible part of the average; formally $P\mapsto d_{P,m,r}$ is
    $m^{-1/r}$-Lipschitz in the Wasserstein metric, so a Wasserstein-small perturbation
    of the measure — which is how outliers enter — moves the DTM's sublevel sets by
    little [@chazalIntroductionTopologicalData2021, §4.4, Eq. 2, p. 16].

**Exercise 6** ★★ (derive). Write down the Vietoris–Rips filtration that [@liuReLUNeuralNetworks2023] builds, starting from a set of sampled inputs and a ReLU network, and say what plays the role of the "scale" parameter.

??? success "Solution"
    For each sampled input $x$ compute its binary vector $s(x)$ from the sign of every
    hidden pre-activation [@liuReLUNeuralNetworks2023, §2.2, Eq. 4–5, p. 2]. Form the
    pairwise Hamming matrix $D_{i,j}=d_H(s(x^{(i)}),s(x^{(j)}))$. For each real scale
    $t$ build the adjacency matrix $A(t)$ with $A(t)_{i,j}=0$ if $D_{i,j}>t$ (or
    $i=j$) and $1$ otherwise, take its clique complex, and let $t$ grow; this is a
    Vietoris–Rips filtration whose scale is the *number of differing activation bits*
    [@liuReLUNeuralNetworks2023, §4.1, Eq. 13, p. 6]. The barcodes are then computed by
    Ripser. The scale parameter therefore counts how many ReLU boundaries must be
    crossed, which is what makes the birth of a long loop an indication of a genuine
    manifold rather than of a single-step flip.

**Exercise 7** ★★ (design). You want a graph summary of a large image data set that you suspect lies near a circle, using a trained network's own geometry as the filter. Describe the Mapper construction you would use and the two choices that could ruin it.

??? success "Solution"
    Take the data set $X$ with a distance (Euclidean, or the Hamming metric of
    [@liuReLUNeuralNetworks2023]), choose a filter $f$ — a centrality or eccentricity
    function avoids any prior knowledge, or coordinates from a dimension reduction —
    and a cover $\mathcal{U}$ of $f(X)$ by intervals; cluster each preimage
    $f^{-1}(U)$ and take the nerve of the refined pull-back cover, which for a
    real-valued filter and gain below $50\%$ is a graph [@chazalIntroductionTopologicalData2021,
    §3, Def. 2 and Algorithm 1, pp. 7–8]. The two choices that can ruin it are the
    resolution and the gain: the output is "very sensitive" to both, and small changes
    can produce very large changes in the graph, so one explores a range of parameters
    and keeps the most informative output [@chazalIntroductionTopologicalData2021, §3,
    p. 9]. A loop in the resulting graph is the candidate circle.
