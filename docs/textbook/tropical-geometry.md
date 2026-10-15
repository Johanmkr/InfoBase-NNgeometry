---
chapter: tropical-geometry
title: Tropical geometry of ReLU networks
outline_hash: 853922cb956f
built_from:
  NIPS2014_fa6f2a46: e11f3b75c4ab
  alfarraDecisionBoundariesNeural2023: a305d52d8025
  averkovEXPRESSIVENESSRATIONALRELU2025: e57bb3498e66
  brandenburgDecompositionPolyhedraPiecewise2024: 307dddfb6bce
  raghuExpressivePowerDeep2017: effafc1ee765
  zhangTropicalGeometryDeep: 56eee67d20f1
---

# Tropical geometry of ReLU networks

!!! abstract "In this chapter"
    - Replace ordinary addition and multiplication by $\max$ and $+$ and a ReLU network becomes a *tropical rational function*, a quotient of two max-affine functions.
    - The Newton polytope of that function carries a *dual subdivision* whose vertices are the network's linear regions; for ReLU weights the polytopes are *zonotopes*.
    - The decision boundaries of a classifier lie inside a tropical hypersurface, and for a shallow network that hypersurface is read from the convex hull of two zonotopes built from the weights.
    - The same difference-of-convex structure organises the *decomposition polyhedron* of a piecewise-linear function and drives pruning and adversarial attacks.
    - A volume invariant of lattice polytopes turns the tropical description into depth lower bounds for networks with rational weights.
    - Prerequisites: [The region complex of a ReLU network](relu-network-geometry.md), and the counting result of [Counting linear regions](counting-regions.md).

The previous chapters took a network apart into regions, activation patterns and bent hyperplanes, and counted them. They used the language of polyhedral geometry and hyperplane arrangements. This chapter uses a different, algebraic language for the same object: *tropical geometry*, the geometry of the semiring in which "addition" is taking a maximum and "multiplication" is ordinary addition. The bridge is the observation that a ReLU is already written in that semiring — $\sigma(t)=\max(0,t)$ is a tropical sum of $t$ and the multiplicative identity [@zhangTropicalGeometryDeep, Def. 2.1, §2] — so a whole ReLU network, built by composing $\max$ with affine maps, is a *tropical rational function*, the tropical analogue of a ratio of two polynomials [@zhangTropicalGeometryDeep, Thm. 5.2, §5]. Once that is true, the region complex, the decision boundaries and the depth of the network all become statements about polytopes.

The payoff is that a network inherits a dictionary. Linear regions correspond to vertices of a polytope, the region count is bounded by a zonotope vertex count, and the decision boundary sits inside a tropical hypersurface whose shape is fixed by the weights [@zhangTropicalGeometryDeep, §3, §6]. The chapter builds the dictionary (Sections 1–4), uses it for decision boundaries, pruning and adversarial examples (Sections 5–7), and then shows how a tropical *volume* — not a count — gives the strongest depth lower bounds known for networks with rational weights [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2]. I translate each paper's symbols into the shared [notation](../notation.md) as they appear and say so once: throughout, $f:\R^{n_0}\to\R^{n_L}$ is the network map, $\sigma$ its ReLU, $W^{(l)},b^{(l)}$ the weights and biases of layer $l$, and $\mathbf{s}(x)$ the activation pattern.

## 1. The max-plus semiring

Tropical arithmetic is ordinary arithmetic with two operations renamed. On the set $\R\cup\{-\infty\}$ define

$$
a\oplus b:=\max\{a,b\},\qquad a\odot b:=a+b,\qquad a\oslash b:=a-b ,
$$

the *tropical sum*, *product* and *quotient* [@zhangTropicalGeometryDeep, Def. 2.1, §2; @alfarraDecisionBoundariesNeural2023, Def. 1, §2, p. 5028]. Then $-\infty$ is the additive identity and $0$ the multiplicative identity, the operations are associative, commutative and distributive, and every element has a multiplicative inverse — so $\R\cup\{-\infty\}$ is a *semifield*, not a ring, because $a$ has no additive inverse [@zhangTropicalGeometryDeep, §2]. Tropical powers collapse to ordinary scaling, $x^{\odot a}=a x$, and are extended to negative exponents by $x^{\odot(-a)}=(x^{\odot a})^{\odot(-1)}$ [@zhangTropicalGeometryDeep, §2].

A *tropical monomial* is $c\odot x^{\alpha}=c+\langle\alpha,x\rangle$ for an exponent vector $\alpha\in\N^{n_0}$, and a *tropical polynomial* is a finite tropical sum of monomials,

$$
p(x)=\bigoplus_i c_i\odot x^{\alpha_i}=\max_i\bigl(c_i+\langle\alpha_i,x\rangle\bigr),
$$

while a *tropical rational function* is a tropical quotient $r=p\oslash q$ of two tropical polynomials [@zhangTropicalGeometryDeep, Defs. 2.2–2.4, §2]. The two facts that will do all the work are immediate from the definitions. First, a tropical polynomial is a *convex* function, being a maximum of affine functions. Second, a tropical rational function is a difference of two convex functions, a DC function [@zhangTropicalGeometryDeep, §2]. A multilayer network is nonconvex, so it cannot be a tropical polynomial; but "most nonconvex functions are a difference of two convex functions", and that is exactly the slot a tropical rational function fills [@zhangTropicalGeometryDeep, §5].

!!! note "Background"
    A semiring is a set with two associative, distributive operations, addition commutative with an identity and multiplication with an identity, but no additive inverses. The max-plus semiring is also called the tropical semiring and is the "tropicalisation" of the ordinary field $\R$ in the sense of algebraic geometry; the standard reference is Maclagan and Sturmfels, *Introduction to Tropical Geometry*. This is outside the collection.

## 2. A ReLU network is a tropical rational function

Fix the standard network setting of the paper: $L$ layers, layer $l$ an affine map with integer weight matrix $W^{(l)}\in\Z^{n_l\times n_{l-1}}$ and real bias $b^{(l)}\in\R^{n_l}$, followed by a threshold activation

$$
\rho^{(l)}(x):=\max\{x,t^{(l)}\},\qquad t^{(l)}\in(\R\cup\{-\infty\})^{n_l},
$$

so that $t=0$ is ReLU and $t=-\infty$ is the identity [@zhangTropicalGeometryDeep, assumptions (a)–(c), §4]. The integer-weight restriction is argued to be no loss of generality: rational weights approximate reals, and clearing denominators and positively rescaling does not change which function is computed [@zhangTropicalGeometryDeep, §4]. The activation class covers leaky ReLU and absolute value, and "with some extra effort" max-pooling and maxout, but not $\tanh$ or sigmoid [@zhangTropicalGeometryDeep, §4].

The reason a network is tropical is the one-layer identity, the observation that a ReLU is already a tropical sum [@zhangTropicalGeometryDeep, §5]. Write $W=W_+-W_-$ with $W_+,W_-$ the entrywise positive and negative parts. For the $i$-th coordinate, with $u=(W_+x+b)_i$, $v=(W_-x)_i\ge 0$ and $Wx+b=u-v$,

$$
\max\{(Wx+b)_i,\,t_i\}=\max\{u-v,\,t_i\}=\max\{u,\,t_i+v\}-v ,
$$

a difference of two max-affine functions of $x$ [@zhangTropicalGeometryDeep, §5]. Applying this decomposition recursively through the layers gives the central theorem.

!!! example "The tropical recursion"
    If each coordinate of the layer-$(l)$ output is a tropical rational $F^{(l)}\oslash G^{(l)}$, then in the non-tropical form of Proposition 5.1 the next layer is

    $$
    \begin{aligned}
    F^{(l+1)}(x)&=\max\{H^{(l+1)}(x),\,G^{(l+1)}(x)+t\},\\
    &\quad G^{(l+1)}(x)=W_+G^{(l)}(x)+W_-F^{(l)}(x),
    \end{aligned}
    $$

    $$
    H^{(l+1)}(x)=W_+F^{(l)}(x)+W_-G^{(l)}(x)+b ,
    $$

    with the output of layer $l+1$ again a tropical rational $F^{(l+1)}\oslash G^{(l+1)}$ [@zhangTropicalGeometryDeep, Prop. 5.1, §5].

**Theorem (tropical characterization).** Every feedforward network satisfying (a)–(c) is a tropical rational map, $f=F\oslash G$ with $F,G$ tropical polynomial maps; conversely, a tropical rational function $f\oslash g$ is realised by a network of depth at most $\max\{\lceil\log_2 r_f\rceil,\lceil\log_2 r_g\rceil\}+2$, where $r_f,r_g$ count the monomials of $f$ and $g$ [@zhangTropicalGeometryDeep, Thm. 5.2 and Thm. 5.4, §5]. Combined with the observation that a continuous piecewise-linear function has integer coefficients exactly when it is tropical rational [@zhangTropicalGeometryDeep, Prop. 5.5, §5], this makes three families the same: tropical rational functions, continuous piecewise-linear functions with integer coefficients, and integer-weight ReLU networks [@zhangTropicalGeometryDeep, §5]. The integer weights reappear here as the *exponents* of the tropical monomials, which is why they, and not just the biases, are needed for the equivalence of families [@zhangTropicalGeometryDeep, §5].

The practical upshot: a network is the tropical analogue of a rational function, and the semifield structure is inherited by the family of computable functions [@zhangTropicalGeometryDeep, §1, §5]. The chapter now reads the geometry off that representation.

## 3. Newton polytopes, tropical hypersurfaces, dual subdivisions

A tropical polynomial is linear on most of its domain, and the set where it stops being linear is a polyhedral object. The *tropical hypersurface* of $p$ is

$$
T(p):=\bigl\{\,x\in\R^{n_0}:\ c_i\odot x^{\alpha_i}=c_j\odot x^{\alpha_j}=p(x)\ \text{for some } \alpha_i\neq\alpha_j\,\bigr\},
$$

the set of points where the maximum is attained by two or more monomials [@zhangTropicalGeometryDeep, Def. 3.1, §3; @alfarraDecisionBoundariesNeural2023, Def. 4, §2, p. 5028]. It divides $\R^{n_0}$ into convex polyhedral cells on which $p$ is a single affine function — cells cut out by inequalities with integer coefficients [@zhangTropicalGeometryDeep, §3].

The hypersurface is dual to a polytope built from the exponents and coefficients. The *Newton polytope* is the convex hull of the exponent vectors, $\mathrm{Newt}(p)=\mathrm{conv}\{\alpha_i : c_i\neq-\infty\}$ [@zhangTropicalGeometryDeep, Def. 3.2, §3; @alfarraDecisionBoundariesNeural2023, Def. 5, §2, p. 5028]. Lift each exponent to $(\alpha_i,c_i)\in\R^{n_0+1}$ and take the convex hull,

$$
P(p):=\mathrm{conv}\{(\alpha_i,c_i)\}_{i},
$$

then project the *upper faces* of $P(p)$ back to $\R^{n_0}$; the result is the *dual subdivision* $\Delta(p)$ of $\mathrm{Newt}(p)$ [@zhangTropicalGeometryDeep, Eq. 1, §3]. (The papers write $\Delta(f)$ or $\delta(f)$ for this subdivision; I use $\Delta(p)$ throughout.) Two consequences drive the rest of the chapter: the tropical hypersurface is exactly the $(n_0-1)$-skeleton of the complex dual to $\Delta(p)$, so each *vertex* of $\Delta(p)$ corresponds to one cell on which $p$ is linear, and the number of vertices of $P(p)$ bounds the number of linear regions [@zhangTropicalGeometryDeep, §3; @alfarraDecisionBoundariesNeural2023, §2, p. 5028].

Monomial operations become polytope operations. For $a\in\N$, $P(p^{\odot a})=aP(p)$; for two tropical polynomials, $P(p\odot q)=P(p)+P(q)$ and $P(p\oplus q)=\mathrm{conv}\bigl(V(P(p))\cup V(P(q))\bigr)$, where $V(\cdot)$ is the vertex set and $+$ is the Minkowski sum [@zhangTropicalGeometryDeep, Props. 3.1–3.2, §3]. These two rules are the reason the geometry propagates cleanly from layer to layer: tropical product is Minkowski addition, and tropical sum takes a convex hull.

Finally, a linear region is a maximal connected set on which the map is affine, and the count is $N(f)$ [@zhangTropicalGeometryDeep, Def. 3.3, §3]. One subtlety matters for everything below. A tropical *polynomial* has convex linear regions; a tropical *rational* function generally has *nonconvex* linear regions [@zhangTropicalGeometryDeep, Def. 3.3, §3]. The convex cells of the dual subdivision are the pieces where one monomial wins; when two adjacent cells carry the same affine map, the maximal linear region is their nonconvex union. The region complex of the previous chapter is the convex refinement, and it is the finer object.

## 4. Zonotopes, and the region count as a vertex count

A *zonotope* is a Minkowski sum of line segments, which in the shared notation is $Z=\sum_i[-v_i,v_i]$; equivalently it is $\{U^\top x : x\in[0,1]^L\}$ for a generator matrix $U$ whose rows are the segment vectors, and it is the Minkowski sum of those segments [@zhangTropicalGeometryDeep, §3.1; @alfarraDecisionBoundariesNeural2023, Def. 6, §2, p. 5029]. Zonotopes are the geometric building blocks of the tropical picture, and the reason is the following layer-wise computation.

**Zonotopes as building blocks.** In the recursion of Section 2, $P(g_i^{(1)})$ and $P(h_i^{(1)})$ are points and $P(f_i^{(1)})$ is a line segment; the two polytopes $P(g_i^{(2)}),P(h_i^{(2)})$ of the second layer are zonotopes; and from there on each $P(g_i^{(l+1)})$ and $P(h_i^{(l+1)})$ is a weighted Minkowski sum of the previous layer's polytopes plus the segment $\{b_i e\}$ [@zhangTropicalGeometryDeep, Lemma 6.2, §6.2]. A one-hidden-layer network is thus *completely characterised by zonotopes*, and deeper networks are built from them by Minkowski addition [@zhangTropicalGeometryDeep, §6.2].

Because the number of linear regions is bounded by the number of vertices of these polytopes, a zonotope vertex bound becomes a region bound. The Gritzmann–Sturmfels theorem gives that the number of vertices of a Minkowski sum of polytopes with $m$ non-parallel edges in total is at most $\sum_{j=0}^{n_0}\binom{m}{j}$, with equality for zonotopes in general position; the projected version for the upper faces of a zonotope is the same count [@zhangTropicalGeometryDeep, Thm. 3.3 and Cor. 3.4, §3.1]. Feeding this through Lemma 6.2 yields the tropical region count.

**Region count.** For an $L$-layer network with threshold $t^{(L)}=-\infty$ and $n_l\ge n_0$ for $l=1,\dots,L-1$, the number of linear regions is at most

$$
\prod_{l=1}^{L-1}\sum_{i=0}^{n_0}\binom{n_l}{i},
$$

which is $O(n^{n_0(L-1)})$ when every hidden width equals $n\ge n_0$ [@zhangTropicalGeometryDeep, Thm. 6.3, §6.3]. The authors are explicit that this does *not* improve on the bound of [@raghuExpressivePowerDeep2017] and is in fact more restrictive, since it applies only to networks satisfying (a)–(c); the point is that tropical geometry *recovers* the same count [@zhangTropicalGeometryDeep, §6.3]. Paired with the lower bound $(n/n_0)^{n_0(L-1)}n^{n_0}$ of [@NIPS2014_fa6f2a46, Cor. 5], it says the region count grows polynomially in the width and exponentially in the depth [@zhangTropicalGeometryDeep, §6.3]. The construction of that trade-off, and its refinements, are the subject of [Counting linear regions](counting-regions.md); here the tropical route is just one more way to reach the same quantity.

## 5. Decision boundaries live in a tropical hypersurface

A classifier cuts its input space by the set where two scores tie, and tropical geometry gives that set a name. Let the network output be consulted through an injective score $s$ with decision threshold $c$, so the decision boundary is $B=\{x:f(x)=s^{-1}(c)\}$ [@zhangTropicalGeometryDeep, §6.1]. If $f=F\oslash G$ and $\nu=f$, then $B$ is contained in the tropical hypersurface of a single tropical polynomial,

$$
B\subseteq T\bigl(s^{-1}(c)\odot G\ominus F\bigr),\qquad s^{-1}(c)\odot G\ominus F=\max\{F,\,G+s^{-1}(c)\},
$$

and $\R^{n_0}$ splits into at most $N(F)$ connected positive regions and at most $N(G)$ negative ones [@zhangTropicalGeometryDeep, Prop. 6.1 and Eq. 3, §6.1]. The inclusion is strict in general, and the reason is the Section 3 subtlety: $s^{-1}(c)\odot G\ominus F$ need not be linear on a whole positive or negative region, so its hypersurface may further split one such region into several linear regions [@zhangTropicalGeometryDeep, §6.1].

The shallow case is the one where the tropical object can be written down completely. Take a bias-free network $f(x)=B\max(Ax+c_1,0)+c_2$ with $A\in\Z^{p\times n}$, $B\in\Z^{2\times p}$, one hidden layer (so this is (Affine, ReLU, Affine)), two logits $f_1=H_1\oslash Q_1$ and $f_2=H_2\oslash Q_2$ [@alfarraDecisionBoundariesNeural2023, §3, p. 5029]. The tie $f_1=f_2$ is the tropical equality $H_1\odot Q_2=H_2\odot Q_1$, which singles out a face of

$$
R(x)=H_1\odot Q_2\oplus H_2\odot Q_1 .
$$

**Theorem (decision boundaries as a polytope).** With $B=\{x\in\R^n:f_1(x)=f_2(x)\}$ the decision boundaries, $B\subseteq T(R(x))$, and the dual subdivision of $R$ is the convex hull of two zonotopes,

$$
\Delta(R(x))=\mathrm{conv}\bigl(Z_{G_1},\,Z_{G_2}\bigr),
$$

where $Z_{G_1}$ is generated by the segments $(B^+(1,j)+B^-(2,j))\,[A^-(j,:),A^+(j,:)]$ with shift $(B^-(1,:)+B^+(2,:))A^-$, and $Z_{G_2}$ by $(B^-(1,j)+B^+(2,j))\,[A^+(j,:),A^-(j,:)]$ with shift $(B^+(1,:)+B^-(2,:))A^-$, using $A^\pm=\max(\pm A,0)$ and the same convention for $B$ [@alfarraDecisionBoundariesNeural2023, Thm. 2, §3, p. 5029]. Since vertex-enumeration algorithms for zonotopes want generators anchored at the origin, a translation converts each segment $[u_{i1},u_{i2}]$ to $[u_{i1}-u_{i2},0]$ plus a shift, and the generators collapse to the diagonal forms

$$
\begin{aligned}
G_1&=\mathrm{Diag}\bigl[(B^+(1,:))+(B^-(2,:))\bigr]A,\\
&\qquad G_2=\mathrm{Diag}\bigl[(B^+(2,:))+(B^-(1,:))\bigr]A
\end{aligned}
$$

[@alfarraDecisionBoundariesNeural2023, Prop. 1 and Cor. 1, §3, pp. 5029–5030]. The paper calls $\Delta(R(x))$ the *decision boundaries polytope*: it is a function of the weights alone, and its edge normals are parallel to $T(R(x))$, hence one-to-one with the hypersurface that contains the boundaries [@alfarraDecisionBoundariesNeural2023, §3, p. 5030; Fig. 2].

This is the cleanest instance of the dictionary, and it comes with a warning. The statement is an *inclusion* — $B\subseteq T(R(x))$ — not an equality, and nothing bounds how much larger the hypersurface is than the boundary set [@alfarraDecisionBoundariesNeural2023, §3, p. 5029]. Everything that follows inherits that slack, because the algorithms steer the polytope rather than the boundary directly. The explicit polytopes are drawn for $n=2$ inputs, where a zonotope's vertex count grows as $O(L^{n-1})$; the paper does not say what replaces the picture in high input dimension [@alfarraDecisionBoundariesNeural2023, §3, p. 5030].

!!! question "Open question"
    Can the inclusion $B\subseteq T(R(x))$ be tightened — is $B$ dense in $T(R(x))$ on the relevant strata, or is the hypersurface generically much larger? No paper in the collection bounds the excess, and the pruning surrogate is justified only by the heuristic that matching the generators matches the boundaries [@alfarraDecisionBoundariesNeural2023, §5, p. 5032].

## 6. Decomposition polyhedra: which difference of convex functions?

Section 2 gave every network a representation $f=g-h$ as a difference of two convex tropical polynomials, but it gave no notion of a *good* such representation. This is the problem [@brandenburgDecompositionPolyhedraPiecewise2024] attacks: given a continuous piecewise-linear (CPWL) function $f$, write it as a difference of two convex CPWL functions $g,h$ with as few linear pieces as possible (Problem 1.1) [@brandenburgDecompositionPolyhedraPiecewise2024, §1]. The tropical and the network pictures meet exactly here, because a *hyperplane function*,

$$
f(x)=\sum_{i\in[k]}\lambda_i\max\{\langle x,a_i\rangle+b_i,\ \langle x,c_i\rangle+d_i\},
$$

is precisely the class computable by a one-hidden-layer ReLU network, and the median function is a canonical example [@brandenburgDecompositionPolyhedraPiecewise2024, Def. 3.16, §3, p. 7].

The move is to *fix* where $g$ and $h$ are allowed to bend. Fix a complete polyhedral complex $P$ covering $\R^n$; the CPWL functions compatible with it form a linear subspace $V_P$, and after quotienting affine functions one can parametrise each function by a weight $w_f$ on its codimension-one faces satisfying the tropical *balancing condition*

$$
\sum_{\substack{\sigma\in P^{n-1}\\ \sigma\supset\tau}} w(\sigma)\,e_{\sigma/\tau}=0 \qquad\text{for every }\tau\in P^{n-2},
$$

where $e_{\sigma/\tau}$ is the unit normal of $\tau$ pointing into $\sigma$ [@brandenburgDecompositionPolyhedraPiecewise2024, §2, p. 4 and Lemma 3.2]. The weight $w_f(\sigma)$ measures how convex or concave $f$ is across the facet $\sigma$, and convexity becomes a sign condition: $f$ is convex iff $w_f\ge 0$, convex with $P=P_f$ iff $w_f>0$ [@brandenburgDecompositionPolyhedraPiecewise2024, Prop. 3.3, §3, p. 5]. So the convex functions form a polyhedral cone $V_P^+$, and the set of decompositions of $f$ is

$$
DP(f):=\{(g,h)\mid g,h\in V_P^+,\ f=g-h\},
$$

the *decomposition polyhedron*, which is the intersection of two shifted copies of $V_P^+$ [@brandenburgDecompositionPolyhedraPiecewise2024, Def. 3.4 and Thm. 3.5, §3, pp. 5–6]. Under the isomorphism $V_P\cong W_P$ this reads $\{(w_g,w_h)\in W_P^+\times W_P^+\mid w_g-w_h=w_f\}$, a polyhedron whose faces carry the answer to the minimisation.

The optimal decompositions are faces and vertices. A decomposition is *reduced* if no convex $\phi\neq 0$ can be subtracted from both parts and stay convex; these are exactly the bounded faces of $DP(f)$ [@brandenburgDecompositionPolyhedraPiecewise2024, Def. 3.7 and Thm. 3.8, §3, p. 6]. A decomposition is *minimal* if no other decomposition has at most as many pieces in each of $g$ and $h$ and strictly fewer in one — Pareto-optimality of the piece counts — and every minimal decomposition is a *vertex* of $DP(f)$ [@brandenburgDecompositionPolyhedraPiecewise2024, Def. 3.12 and Thm. 3.13, §3, p. 6]. The vertex is characterised intrinsically: no non-trivial coarsening exists exactly for the vertices [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 3.11, §3, p. 6]. This makes the search finite but, in general, vertex enumeration; when $DP(f)$ has a unique vertex, the minimal decomposition is unique, which happens for hyperplane functions and for the $k$-th order statistic [@brandenburgDecompositionPolyhedraPiecewise2024, Prop. 3.14 and §3, p. 7].

Two consequences matter for networks. First, this framework lets the authors disprove a proposed minimal construction of Tran and Wang beyond two dimensions and show that the classical hyperplane-extension and local-maxima decompositions of the $k$-th order statistic are not even vertices [@brandenburgDecompositionPolyhedraPiecewise2024, §4, p. 8]. Second, the decomposition feeds a network *construction*: every convex CPWL function $f:\R^n\to\R$ with $k$ affine components is representable by a network of depth $\lceil\log_2(n+1)\rceil+\lceil\log_2 r\rceil+1$ and size $O(r s^{n+1})$, for any $r,s$ with $rs\ge k$; choosing $r=1,s=k$ recovers the low-depth bound and $r=k,s=1$ gives depth $O(\log k)$ and size $O(k)$ [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 6.3, §6, p. 9]. A nonconvex $f$ is handled by first decomposing it with a good vertex of $DP(f)$ and then applying the convex construction [@brandenburgDecompositionPolyhedraPiecewise2024, Cor. 6.4, §6, p. 9]. The paper is theory without implementation, and minimality is only relative to the fixed complex $P$; finding a globally minimal decomposition is left open [@brandenburgDecompositionPolyhedraPiecewise2024, §1, §7].

## 7. Two applications of the decision boundaries polytope

### Pruning by preserving the polytope

If the decision boundaries polytope $\Delta(R(x))$ determines a superset of the boundaries, then preserving the polytope preserves the boundaries, and pruning should try to do that. With the pruned weights $\tilde A,\tilde B$ and their generators $\tilde G_1,\tilde G_2$, the ideal problem is $\min_{\tilde A,\tilde B} d(\Delta(\tilde R(x)),\Delta(R(x)))$ for a distance $d$ between polytopes [@alfarraDecisionBoundariesNeural2023, Problem 1, §5, p. 5031]. It is hard, so the paper replaces the polytope distance by a surrogate on the generator matrices themselves,

$$
\min_{\tilde A,\tilde B}\ \tfrac12\norm{\tilde G_1-G_1}_F^2+\tfrac12\norm{\tilde G_2-G_2}_F^2
+\lambda_1\norm{\tilde G_1}_{2,1}+\lambda_2\norm{\tilde G_2}_{2,1},
$$

where $\norm{C}_{2,1}=\sum_i\norm{C(i,:)}_2$ is the mixed norm that zeroes whole rows, i.e. whole hidden units [@alfarraDecisionBoundariesNeural2023, Problem 2, §5, p. 5032]. Problem 2 separates over the rows of $\tilde A$ and the entries of $\tilde B$, each sub-problem being convex with a closed-form update, so a single alternating iteration is enough in the experiments [@alfarraDecisionBoundariesNeural2023, §5, p. 5032].

The reported results: on AlexNet and VGG16 trained on SVHN, CIFAR-10 and CIFAR-100, more than $90\%$ of the classifier parameters are removed "without affecting the accuracy" [@alfarraDecisionBoundariesNeural2023, §5, p. 5033]. In area under the pruning-accuracy curve the method beats the magnitude-based baselines by $3\%$, $3\%$ and $2\%$ on SVHN, CIFAR-10 and CIFAR-100 for AlexNet, and is comparable on SVHN and CIFAR-10 for VGG16 [@alfarraDecisionBoundariesNeural2023, §5, p. 5033]. On LeNet/MNIST it beats the tropical-network-minimisation baseline of Smyrnis and Maragos at every pruning level, by margins up to $7\%$ [@alfarraDecisionBoundariesNeural2023, §5, p. 5033]. The authors state they do not claim to beat state-of-the-art pruning; the contribution is the geometric perspective [@alfarraDecisionBoundariesNeural2023, §5, p. 5032]. A caution: the headline experiments run outside the theorem's assumptions — the theory is bias-free, shallow and two-output, while AlexNet and VGG16 are handled by a greedy block-wise heuristic [@alfarraDecisionBoundariesNeural2023, §5, p. 5033].

The polytope also reframes the lottery-ticket hypothesis. Training a 2-input, 2-output, 40-node network while pruning and re-initialising four ways — lottery-ticket, Xavier, standard Gaussian, zero-mean Gaussian — the *orientation* of $\Delta(R(x))$ stays close to the original only under the lottery-ticket initialisation, while the others deviate much more and lose accuracy [@alfarraDecisionBoundariesNeural2023, §4, pp. 5030–5031]. The authors stress that this refines the hypothesis: several networks can have similar accuracy with different decision boundaries, so accuracy alone does not pin the boundary down [@alfarraDecisionBoundariesNeural2023, §4, p. 5031].

### Adversarial attacks as moving the polytope

An ordinary targeted attack finds an input perturbation $h$ that carries $x_0$ across a boundary: $\min_h D(h)$ subject to $\mathrm{argmax}_i f_i(x_0+h)=t\neq c$ [@alfarraDecisionBoundariesNeural2023, Eq. 3, §6, p. 5034]. The tropical view is *dual*: instead of moving the input, move the boundary by perturbing the first linear layer $A_1$. The two views coincide when the perturbations satisfy

$$
g\bigl((A_1+\tilde A_1)x_0\bigr)=g(A_1x_0+A_1h)=f(x_0+h),
$$

i.e. when $A_1h=\tilde A_1x_0$ [@alfarraDecisionBoundariesNeural2023, Eq. 4, §6, p. 5034]. Because the normals to $\Delta(R(x))$ are the hypersurface $T(R(x))$, a first-layer perturbation $\tilde A_1$ moves the polytope, and one solves

$$
\min_{h,\tilde A_1}\ D_1(h)+D_2(\tilde A_1)
$$

subject to the loss constraints, the box $x_0+h\in[0,1]^n$, the budgets $\norm{h}_1\le\lambda_1$ and $\norm{\tilde A_1}_{1,1}\le\lambda_2$, and the coupling $A_1h=\tilde A_1x_0$, where

$$
D_2(\tilde A_1)=\frac12\sum_{j=1}^{2}\Bigl(\norm{\mathrm{Diag}(B^+(j,:))\tilde A_1}_F^2+\norm{\mathrm{Diag}(B^-(j,:))\tilde A_1}_F^2\Bigr)
$$

measures how far the two zonotopes move [@alfarraDecisionBoundariesNeural2023, Eq. 5 and Eq. 6, §6, pp. 5034–5035]. It is solved by a penalty method on the equality constraints, each penalty step by ADMM [@alfarraDecisionBoundariesNeural2023, §6, p. 5035].

On MNIST the attack flips digits (for instance $9\to 8,4,3,2,1$) with perturbations as small as $\lambda=0.1$ in the units of $x_0\in[0,1]^n$ [@alfarraDecisionBoundariesNeural2023, §6, p. 5036]. Robust accuracy over 1000 correctly classified MNIST samples is $44.8$ at the baseline ($\lambda_2=0$), $46.1$ at $\lambda_2=1.0$ and $43.8$ at $\lambda_2=2.0$ for $\lambda_1=0.1$, and $12.6$, $11.3$, $11.0$ for $\lambda_1=0.2$ [@alfarraDecisionBoundariesNeural2023, Tab. 1, §6, p. 5036]. Two caveats the authors give themselves: the attack is not claimed to beat state of the art, and its comparisons are internal ablations of $\lambda_2$ rather than benchmarks against strong attacks [@alfarraDecisionBoundariesNeural2023, §6, p. 5036].

## 8. Tropical volume and depth lower bounds

The count $N(f)$ measures a network's complexity by how many regions it carves out, the complexity measure followed by [@zhangTropicalGeometryDeep, §6.3] and [@raghuExpressivePowerDeep2017]. A different question is how *deep* a network must be to represent a given function exactly, and here the tropical dictionary supplies a finer invariant than a count [@averkovEXPRESSIVENESSRATIONALRELU2025, §1, PDF pp. 1–2]. The test function is

$$
F_n=\max\{0,x_1,\dots,x_n\},
$$

which is the support function $h_{\Delta_n}$ of the standard simplex $\Delta_n=\mathrm{conv}(0,e_1,\dots,e_n)$; its normalised volume $\mathrm{Vol}_n=n!\,\mathrm{vol}_n$ equals $1$ on $\Delta_n$ and is a non-negative integer on lattice polytopes [@averkovEXPRESSIVENESSRATIONALRELU2025, §2.1, PDF p. 4]. The conjecture of Hertrich et al. (outside the collection) is that any ReLU network exactly representing $F_n$ needs at least $\lceil\log_2(n+1)\rceil$ hidden layers, matching the known upper bound; it was confirmed only for integer weights [@averkovEXPRESSIVENESSRATIONALRELU2025, §1, PDF pp. 1–2].

The tropical characterisation enters through Hertrich's theorem. With $SU^k$ the $k$-fold *sum-union closure* of a family of polytopes — sums of convex hulls $\mathrm{conv}(A_i\cup B_i)$ — one has

$$
\mathrm{ReLU}_{\R}^{n,0}(k)=\{\,h_A-h_B:\ A,B\in SU^k(\mathcal{P}_0(\R^n))\,\},
$$

so $h_P$ is representable with $k$ hidden layers iff $P+A=B$ for some $A,B$ in the closure [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 8 and Cor. 9, PDF p. 6]. The obstruction is a *modular volume*: for $d=p^t\le n$ a power of a prime, the normalised volume taken mod $p$ is Minkowski additive on $d$-dimensional lattice polytopes, $\mathrm{Vol}_d(\sum_i P_i)\equiv_p\sum_i\mathrm{Vol}_d(P_i)$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Prop. 11, PDF p. 7]. Consequently every $p^k$-dimensional face of a polytope in $SU^k$ has volume divisible by $p$, and if $h_P$ is integer-representable with $k\le t$ hidden layers at dimension $d=p^t$, then $p\mid\mathrm{Vol}_d(P)$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 14 and Thm. 15, PDF p. 8].

To pass from integer to the weights hardware actually uses, clear denominators. If $f$ is representable by a network with $k$ hidden layers and rational weights with common denominator $M$, then $M^{k+1}f$ is representable with integer weights and the same depth [@averkovEXPRESSIVENESSRATIONALRELU2025, Lemma 16, PDF p. 8]. Since scaling the simplex by $N^{t(k+1)}$ gives a polytope with volume $\equiv 1\pmod p$, the divisibility theorem refuses a shallow representation.

**Theorem (depth lower bound).** Let $n,N$ be positive and $p$ a prime not dividing $N$. Every ReLU network with weights in the ring of $N$-ary fractions $z/N^t$ needs at least $\lceil\log_p(n+1)\rceil$ hidden layers to represent $F_n$ exactly [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2, PDF p. 2]. For decimal weights take $p=3$, giving $\lceil\log_3(n+1)\rceil$ hidden layers; this is optimal up to a constant factor [@averkovEXPRESSIVENESSRATIONALRELU2025, Cor. 3, PDF p. 2]. For general $N$ the least prime not dividing $N$ gives a lower bound $C\ln n/\ln\ln N$, the first non-constant depth lower bound for networks with weights of practical precision [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 4, PDF p. 2].

This is the same lattice-polytope machinery that underlies [@brandenburgDecompositionPolyhedraPiecewise2024], used for a different purpose: there a polyhedron of decompositions, here a modular invariant that separates depths. The bound does not resolve the conjecture, because it says nothing about general real weights, and the dependence on $N$ through $\ln\ln N$ is so weak that its content is mainly non-constancy [@averkovEXPRESSIVENESSRATIONALRELU2025, §4, PDF p. 10]. It also gives no control on width or weight magnitude, so it does not separate depth from size [@averkovEXPRESSIVENESSRATIONALRELU2025, §1, PDF p. 2].

## Summary

- A ReLU network under standard assumptions is exactly a *tropical rational function*, a quotient $F\oslash G$ of two convex max-affine maps, and conversely every tropical rational function is a network with depth controlled by its number of monomials [@zhangTropicalGeometryDeep, Thm. 5.2 and Thm. 5.4, §5].
- The tropical hypersurface $T(p)$ is where the maximum is attained twice; its complement's cells are the regions on which $p$ is linear, counted by the vertices of the dual subdivision of the Newton polytope [@zhangTropicalGeometryDeep, Def. 3.1 and §3].
- Zonotopes are the geometric building blocks: layer-2 polytopes are zonotopes and later layers are their Minkowski sums, so a zonotope vertex bound yields the region-count bound $\prod_l\sum_{i\le n_0}\binom{n_l}{i}$ [@zhangTropicalGeometryDeep, Lemma 6.2 and Thm. 6.3, §6].
- The decision boundaries of a classifier lie inside a tropical hypersurface; for a shallow two-output network that hypersurface comes from the convex hull of two zonotopes, the decision boundaries polytope built from the weights [@alfarraDecisionBoundariesNeural2023, Thm. 2 and Cor. 1, §3].
- The decompositions $f=g-h$ into convex CPWL functions form the decomposition polyhedron $DP(f)$; reduced decompositions are its bounded faces and minimal ones are its vertices, linking the tropical structure to network construction [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 3.8 and Thm. 3.13, §3].
- Preserving the decision boundaries polytope drives pruning (mixed-norm row sparsity on the generators, >90% of classifier parameters removed) and a dual adversarial attack that perturbs the first layer to move the polytope [@alfarraDecisionBoundariesNeural2023, Problem 2 and Eq. 5, §5–§6].
- A modular normalised-volume invariant on lattice polytopes gives depth lower bounds $\lceil\log_p(n+1)\rceil$ for $N$-ary-fraction weights, hence $\lceil\log_3(n+1)\rceil$ for decimal weights [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2 and Cor. 3, PDF p. 2].

## Further reading

- [@zhangTropicalGeometryDeep]: the foundational dictionary; read §2–§3 for the algebra and hypersurfaces, §5 for the equivalence with ReLU networks, and §6 for decision boundaries, zonotopes and the region bound.
- [@alfarraDecisionBoundariesNeural2023]: the decision boundaries polytope; read §2–§3 for Thm. 2 and the generators, §4 for the lottery-ticket evidence, §5 for pruning and §6 for the adversarial dual view.
- [@brandenburgDecompositionPolyhedraPiecewise2024]: decomposition polyhedra and the balancing condition; read §3 for $DP(f)$ and its faces and vertices, §4 for the refutation of the 2-dimensional construction, and §6 for the depth-size trade-off.
- [@averkovEXPRESSIVENESSRATIONALRELU2025]: tropical/lattice-polytope methods for depth separation; read §2 for the sum-union closure and §3 for the modular volume obstruction and the main theorems.
- [@raghuExpressivePowerDeep2017]: the region bound the tropical argument recovers, for comparison with [Counting linear regions](counting-regions.md).

## Exercises

**Exercise 1** ★ (check). Starting from the definition $a\oplus b=\max\{a,b\}$, $a\odot b=a+b$, show that a ReLU is a tropical sum of the identity and the multiplicative identity, and explain why a network with a single affine layer followed by ReLU followed by a linear output is a tropical *difference* of two tropical polynomials.

??? success "Solution"
    The identity map is $x\mapsto x=0\odot x$, since $0$ is the tropical multiplicative identity, and the constant zero is $-\infty$. Hence $\sigma(x)=\max\{x,0\}=x\oplus(-\infty)$, a tropical sum of the monomial $0\odot x$ and the constant $-\infty$ [@zhangTropicalGeometryDeep, Def. 2.1 and §4]. For the one-layer network, write $W=W_+-W_-$; then $\max\{Wx+b,0\}=\max\{W_+x+b,\ W_-x\}-W_-x$ coordinate-wise, so the preactivation is a difference $F\oslash G$ of two max-affine (tropical polynomial) maps, and applying the linear output layer preserves the form [@zhangTropicalGeometryDeep, §5].

**Exercise 2** ★ (check). The tropical hypersurface of a polynomial is where two monomials tie for the maximum. Why is it the same object as "the set where the polynomial is not linear", and why does the number of linear regions equal the number of vertices of the dual subdivision?

??? success "Solution"
    On the region where a single monomial $c_i\odot x^{\alpha_i}$ exceeds all others, $p$ equals that one affine function $c_i+\langle\alpha_i,x\rangle$, hence is linear; two monomials tie exactly where the winning affine piece changes [@zhangTropicalGeometryDeep, Def. 3.1, §3]. The tropical hypersurface is the $(n_0-1)$-skeleton of the polyhedral complex dual to the dual subdivision $\Delta(p)$, and each vertex of $\Delta(p)$ corresponds to one cell on which $p$ is linear, so the vertex count bounds (and generically equals) the number of linear regions [@zhangTropicalGeometryDeep, §3].

**Exercise 3** ★★ (derive). Use the polytope rules $P(p\odot q)=P(p)+P(q)$ and $P(p^{\odot a})=aP(p)$ to explain why the layer-2 polytopes of a ReLU network are zonotopes, and why the region count grows as $O(n^{n_0(L-1)})$.

??? success "Solution"
    A first-layer tropical polynomial has monomials whose exponent vectors lie on a line segment, so $P(f_i^{(1)})$ is a segment; the weights of the second layer enter as tropical powers and products, so $P(g_i^{(2)})$ and $P(h_i^{(2)})$ are weighted Minkowski sums of segments — zonotopes — and later layers are Minkowski sums of these plus a point $\{b_ie\}$ [@zhangTropicalGeometryDeep, Lemma 6.2 and Props. 3.1–3.2, §3, §6.2]. A Minkowski sum of $m$ segments has at most $\sum_{j=0}^{n_0}\binom{m}{j}$ vertices [@zhangTropicalGeometryDeep, Thm. 3.3 and Cor. 3.4, §3.1], and propagating this through the $L-1$ hidden layers gives $\prod_{l=1}^{L-1}\sum_{i=0}^{n_0}\binom{n_l}{i}$, which is $O(n^{n_0(L-1)})$ for equal widths $n\ge n_0$ [@zhangTropicalGeometryDeep, Thm. 6.3, §6.3].

**Exercise 4** ★★ (derive). For a bias-free (Affine, ReLU, Affine) network with first weight $A$ and output weight $B$, match the second-layer generators $G_1=\mathrm{Diag}[(B^+(1,:))+(B^-(2,:))]A$ and $G_2=\mathrm{Diag}[(B^+(2,:))+(B^-(1,:))]A$ to the two zonotopes of Theorem 2, and say what a row zero of $G_1$ means for the network.

??? success "Solution"
    $Z_{G_1}$ is generated by the segments $(B^+(1,j)+B^-(2,j))[A^-(j,:),A^+(j,:)]$ and $Z_{G_2}$ by $(B^-(1,j)+B^+(2,j))[A^+(j,:),A^-(j,:)]$, both shifted as in Theorem 2 [@alfarraDecisionBoundariesNeural2023, Thm. 2, §3, p. 5029]; Proposition 1 converts the arbitrary-endpoint segments to origin-anchored ones and Corollary 1 assembles them into the diagonal generator matrices $G_1,G_2$ [@alfarraDecisionBoundariesNeural2023, Prop. 1 and Cor. 1, §3, pp. 5029–5030]. The $j$-th row of $G_1$ is $(B^+(1,j)+B^-(2,j))A(j,:)$; it is zero exactly when that factor vanishes, i.e. when hidden unit $j$ does not influence the first logit in the required direction. Removing such a unit leaves the generator, hence the decision boundaries polytope, unchanged — the motivational insight behind tropical pruning [@alfarraDecisionBoundariesNeural2023, §5, p. 5031].

**Exercise 5** ★★ (compare). Both [@alfarraDecisionBoundariesNeural2023] and [@brandenburgDecompositionPolyhedraPiecewise2024] build a polyhedron from a ReLU network's tropical data. What object does each study, and what does each buy you that the other does not?

??? success "Solution"
    Alfarra et al. study the decision boundaries polytope $\Delta(R(x))=\mathrm{conv}(Z_{G_1},Z_{G_2})$, the dual subdivision of the tropical polynomial whose hypersurface contains the tie set [@alfarraDecisionBoundariesNeural2023, Thm. 2, §3]. It is computable from the weights of a shallow two-output network and directly usable — pruning keeps it, an attack moves it — but it characterises only a superset of the boundaries, with no bound on the excess [@alfarraDecisionBoundariesNeural2023, §5, p. 5032]. Brandenburg et al. study the decomposition polyhedron $DP(f)=\{(g,h):g,h\ \text{convex},\ g-h=f\}$, whose faces and vertices classify all differences of convex functions compatible with a fixed complex $P$ [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 3.5 and Thm. 3.8, §3]. It applies to any CPWL function and any fixed complex and gives a finite minimal-decomposition procedure plus a network construction, but minimality is only relative to $P$ and there is no algorithm for the global problem [@brandenburgDecompositionPolyhedraPiecewise2024, §7]. In short: a concrete weight-level polytope tied to classification versus a general decomposition theory tied to network size.

**Exercise 6** ★★★ (compare / design). [@zhangTropicalGeometryDeep] counts linear regions to argue a deep network is exponentially more expressive than a shallow one; [@averkovEXPRESSIVENESSRATIONALRELU2025] proves a depth lower bound from a volume invariant. Why is a *count* insufficient for a depth lower bound, and what does the volume invariant add?

??? success "Solution"
    A region count is an upper bound on complexity for a fixed architecture and says nothing about whether a *given function* can be realised at a given depth: many shallow networks have large region counts, and a function may be representable shallowly regardless [@zhangTropicalGeometryDeep, Thm. 6.3, §6.3]. The volume bound obstructs a specific function $F_n$: if $h_P$ is representable with $k$ hidden layers then $p\mid\mathrm{Vol}_d(P)$ for $d=p^t$, $k\le t$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 15, PDF p. 8], while the scaled simplex has volume $\equiv 1\pmod p$, so no shallow representation exists [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2, PDF p. 2]. The invariant is a homomorphism on the group of representable functions vanishing below the required depth — a separation certificate, not a size estimate; and it controls neither width nor weight magnitude, so it does not extend to size-bounded separation [@averkovEXPRESSIVENESSRATIONALRELU2025, §4, PDF p. 10].

**Exercise 7** ★★★ (critique). The tropical pruning method reports >90% classifier-parameter reduction at preserved accuracy, yet the theory applies only to shallow, bias-free, two-output networks. Which part of the evidence actually tests the theorem, and what experiment would the theorem predict?

??? success "Solution"
    The theorem is an inclusion $B\subseteq T(R(x))$ for a bias-free (Affine, ReLU, Affine) network [@alfarraDecisionBoundariesNeural2023, Thm. 2, §3], so only the toy (Affine, ReLU, Affine) experiments are within its scope; AlexNet and VGG16 are pruned by a greedy block-wise heuristic with no guarantee [@alfarraDecisionBoundariesNeural2023, §5, p. 5033]. A direct test would keep the theorem's assumptions and check the theorem's own quantity: train a bias-free two-output shallow network, prune with Problem 2, and verify that the decision boundaries polytope — not merely accuracy — stays close, and that the boundaries of the pruned network agree with the original on sampled inputs. That is a check of the claimed mechanism (preserving $\Delta(R(x))$ preserves the boundaries), which the accuracy curves do not isolate; the paper itself notes that different boundaries can give similar accuracy [@alfarraDecisionBoundariesNeural2023, §4, p. 5031].
