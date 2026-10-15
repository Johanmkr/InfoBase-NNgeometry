---
chapter: polyhedral-foundations
title: Polyhedra, arrangements and piecewise-linear maps
outline_hash: e439a201d301
built_from:
  NIPS2014_fa6f2a46: e11f3b75c4ab
  pakLecturesDiscretePolyhedral: b12d340fddd1
  pascanuNumberResponseRegions2013: 3fae3832ff4c
---

# Polyhedra, arrangements and piecewise-linear maps

!!! abstract "In this chapter"
    - Read a piecewise-linear map as a polyhedral object: a finite partition of the domain into
      convex cells on which the map is affine.
    - State what a convex polytope, a polyhedral complex and a hyperplane arrangement are, and how
      the two representations of a polytope (inequalities and convex hull) are used.
    - Count the regions of an arrangement of hyperplanes in general position in low dimension, and
      recognise upper envelopes of affine functions — the maxout picture.
    - See why this vocabulary is the substrate for every later chapter on ReLU region geometry.

    (Prerequisites: none — this is the opening chapter; shared symbols are in the
    [notation](../notation.md) page.)

A rectifier network computes a function that is linear on each of finitely many pieces of its input
space, and the boundaries between the pieces are inherited from the places where individual units
change sign [@pascanuNumberResponseRegions2013, §1]. Everything the rest of this textbook does —
labelling regions by activation patterns, counting them, tracking how deep layers fold space, reading
a network as a spline or as a tropical polynomial — is bookkeeping on that partition. The point of
this chapter is to fix the objects once, carefully, so that later chapters can call them by name.

The mathematics is classical discrete and convex geometry, and the collection contains one textbook
devoted to it [@pakLecturesDiscretePolyhedral]. Pak's lectures are the general reference for convex
polytopes, their combinatorics and their rigidity theory; they are *not* a neural-network text, and
in particular they do not develop hyperplane arrangements. The arrangement side, and the counting of
regions, come from the two papers that founded the region-counting line
[@pascanuNumberResponseRegions2013; @NIPS2014_fa6f2a46]. Read together, they supply the geometric
vocabulary and the first counting theorems; this chapter is that union, arranged so the counting
questions of later chapters have something to attach to.

## Convex polytopes and their two representations

A **convex polytope** is the convex hull of a finite set of points. Pak adopts exactly this reading:
in his lectures "polytope" means the convex hull of a finite number of points, assumed fully
dimensional (not lying in an affine hyperplane), and the word "polyhedron" is reserved for
non-compact intersections of half-spaces and for surfaces that need not be convex
[@pakLecturesDiscretePolyhedral, p. 8]. The convex hull of a set $X\subseteq\R^d$ is written
$\mathrm{conv}(X)$, and a hyperplane $H$ *supports* a polytope $P$ at $x\in\partial P$ if $P$ lies on
one side of $H$ and $x\in H$ [@pakLecturesDiscretePolyhedral, p. 8].

The same polytope can be described in two dual ways, and the distinction matters throughout:

- the **$V$-representation**, as the convex hull of finitely many vertices — the form in which data
  and latent codes are usually given;
- the **$H$-representation**, as the bounded intersection of finitely many half-spaces
  $\{x : Ax\le b\}$ — the form in which a linear region of a network is given, since it is cut out by
  the half-spaces on which each ReLU is active or dead.

For a $d$-dimensional polytope with $f_i$ faces of dimension $i$, the sequence
$(f_0,f_1,\dots,f_d)$ is the **$f$-vector**; $f_d=1$ and $f_0$ is the number of vertices
[@pakLecturesDiscretePolyhedral, §8.1]. Its entries satisfy the generalised **Euler formula**

$$
f_0 - f_1 + f_2 - \dots + (-1)^d f_d = 1 ,
$$

which is the case $k=0$ of the **Dehn–Sommerville equations** for a simple polytope, i.e. one where
every vertex is incident with exactly $d$ edges [@pakLecturesDiscretePolyhedral, §8.1, Thm. 8.1].
Pak proves these by orienting the graph of the polytope with a generic linear functional: the number
of vertices of *Morse index* $i$ — vertices with $i$ outgoing edges — defines the **$h$-vector**
$(h_0,\dots,h_d)$, which satisfies $h_k = h_{d-k}$ and whose generating function is related to the
$f$-vector by $F(t)=H(t+1)$ [@pakLecturesDiscretePolyhedral, §8.1]. This Morse-function device is a
recurring tool: it is what makes "$f$-vector" a computable invariant rather than a list.

Two further operations survive into the neural-network setting. The **Minkowski sum** of sets
$A,B\subseteq\R^d$ is

$$
A+B := \{\,a+b \mid a\in A,\ b\in B\,\},
$$

so that adding a ball of radius $\varepsilon$ thickens a set by $\varepsilon$; Pak uses it to define
surface area via the derivative of $\operatorname{vol}(X+tB)$ and as the statement of the
Brunn–Minkowski inequality [@pakLecturesDiscretePolyhedral, §7.3–7.4]. A polytope all of whose faces
are centrally symmetric is a **zonotope**, and every such polytope is itself a Minkowski sum of
segments [@pakLecturesDiscretePolyhedral, Exercise 7.16]. Zonotopes reappear much later, in the
tropical-geometry treatment of network decision boundaries.

!!! note "Background"
    A bounded polytope has both representations, and they are related by duality: the facets of the
    $H$-representation correspond to the vertices of a dual polytope. The precise dictionary
    (Steinitz' theorem for $d=3$: every $3$-connected planar graph is the graph of a polytope in
    $\R^3$ [@pakLecturesDiscretePolyhedral, Thm. 11.1]; Balinski's theorem in all dimensions: the
    graph of a $d$-polytope is $d$-connected [@pakLecturesDiscretePolyhedral, Thm. 8.8]) is outside
    what the network geometry needs, but it is the reason "compute the local affine map of a region"
    is a well-posed finite problem.

## Polyhedral complexes and piecewise-linear maps

A single polytope is not enough: a piecewise-linear map is affine on *many* cells at once, and those
cells fit together. The right object is a **polyhedral complex**: a finite collection $\Sigma$ of
convex polytopes (the *cells*) such that

1. every face of a cell of $\Sigma$ is again a cell of $\Sigma$, and
2. the intersection of any two cells is a face of each (possibly empty).

Pak keeps the same idea under the names *subdivision* (a face-to-face decomposition of a polytope,
required to be a CW complex) and *triangulation* (a subdivision all of whose cells are simplices),
and notes that a decomposition need only be a CW complex, while a subdivision is required to be
face-to-face [@pakLecturesDiscretePolyhedral, p. 8]. Condition (2) above is precisely the face-to-face
requirement.

With this, the central definition of the whole textbook is short. A continuous map
$f:\R^{n_0}\to\R^{m}$ is **piecewise linear** (or *piecewise affine*) if there is a finite polyhedral
complex $\Sigma$ covering $\R^{n_0}$ such that $f$ agrees with an affine map on every cell; the shared
notation page writes $\mathcal{P}(f)$ for such a complex when it is the one induced by the map itself.
For the functions that arise here, every cell has full dimension $n_0$
[@NIPS2014_fa6f2a46, §2.1]. The maximal connected set on which $f$ is affine is a **linear region**
(or *region of linearity*, or *response region*), and the collection of linear regions is the object
later chapters call the region complex [@pascanuNumberResponseRegions2013, §2].

Why full dimension matters: the linear regions of a piecewise-linear network are the closures of the
open cells of $\mathcal{P}(f)$, and their number, written $\mathcal{N}(f)$, is the counting invariant
of the next chapters. There are two equivalent ways to see that a network's map is piecewise linear.
Directly, a composition of piecewise-linear functions is piecewise linear, and a layered rectifier
MLP is a composition of such functions, so it is piecewise linear [@pascanuNumberResponseRegions2013,
§1]. Concretely, on a region where the sign of every unit is fixed, the network collapses to an affine
map: the ReLU is the identity on the units that are active and zero on the dead ones, and the
composition of the layer matrices with those diagonal sign matrices is again a matrix
[@pascanuNumberResponseRegions2013, §3]. The sign vector that records which units are active on a
region is the **activation pattern** $\mathbf{s}(x)$ of the region's points, the label that
[the region complex](relu-network-geometry.md) uses to index the cells.

!!! tip "Reading a network locally"
    To get the affine map of a region, trace one example through the network recording, at every
    hidden unit, whether it is active or dead; the product of the layer matrices with the
    corresponding diagonal selection of units is the region's affine map
    [@pascanuNumberResponseRegions2013, §3]. This "trace a point, read the pattern, invert the
    selection" recipe is what makes the whole polyhedral picture computational.

## Hyperplane arrangements

A shallow rectifier layer is already interesting. Each hidden unit $i$ has two operational modes,
and the boundary between them is the affine hyperplane

$$
H_i = \{\,x\in\R^{n_0} : W_{i,:}x + b_i = 0\,\},
$$

on one side of which the unit outputs $0$ and on the other a non-zero affine function
[@pascanuNumberResponseRegions2013, §3]. A finite set of hyperplanes in one space is a **hyperplane
arrangement** $A=\{H_i\}$, and its *regions* are the connected components of the complement of their
union, i.e. the sets of the form $\{x : \operatorname{sgn}(Wx+b)=s\}$ for a sign vector
$s\in\{-,+\}^{n_1}$ [@pascanuNumberResponseRegions2013, §3]. The same picture is described for
general activations with a distinguished behaviour at zero in [@NIPS2014_fa6f2a46, §2.2], and it is
recorded on the notation page as "the local model of a piecewise-linear map".

This gives the first clean theorem of the area: **the linear regions of a single rectifier layer are
exactly the regions of its hyperplane arrangement** [@pascanuNumberResponseRegions2013, §3]. The
reason is Lemma 1 of that paper: a vector-valued map $(f_i)$ is affine exactly where all coordinates
are, so its regions of linearity are the non-empty intersections of the regions of its coordinates;
for a layer of rectifier units those coordinate regions are the two half-spaces cut by $H_i$, and
intersecting them reconstructs the arrangement's sign-vector cells
[@pascanuNumberResponseRegions2013, Lemma 1]. Two further reductions, also in the same section,
make the count independent of inessential parameters: the number of *linear output* units does not
affect the maximal region count [@pascanuNumberResponseRegions2013, Lemma 2], and a linear layer
inserted before a rectifier layer can be absorbed into it, because affine maps compose to affine maps
[@pascanuNumberResponseRegions2013, Lemma 3].

The arrangement viewpoint also fixes the geometry of *boundedness*. A region is **relatively
bounded** if its intersection with the span of the hyperplane normals is bounded; the
**essentialization** of $A$ is the arrangement the hyperplanes induce in that span, obtained by
discarding the directions along which they are all parallel
[@pascanuNumberResponseRegions2013, §3]. Since the network map is constant along the orthogonal
complement of the normals, only the essentialization is visible to the network — this is the precise
sense in which a shallow layer cannot "see" directions it never tests.

## Counting the regions of an arrangement

We now count. For an arrangement in **general position** the hyperplanes meet each other as
transversally as possible: any $p\le n_0$ of them meet in dimension $n_0-p$ and any $p>n_0$ have
empty intersection, a situation attained whenever the defining weights and biases are generic
[@pascanuNumberResponseRegions2013, §3]. For such an arrangement the region count is classical. With
$m$ hyperplanes in $\R^{n_0}$, and writing $r(A)$ for the number of regions and $b(A)$ for the
number of relatively bounded ones, the two counts are [@pascanuNumberResponseRegions2013, Prop. 1]:

$$
r(A) = \sum_{s=0}^{n_0}\binom{m}{s},
\qquad
b(A) = \binom{m-1}{n_0}
$$


The two papers we are using state this as the general-position case of Zaslavsky's theorem and cite
it rather than prove it [@pascanuNumberResponseRegions2013, §3; @NIPS2014_fa6f2a46, §2.2]. It is the
one result in this chapter whose proof we place outside the collection.

!!! note "Background"
    **Zaslavsky's theorem.** The number of regions of an arrangement depends only on the intersection
    lattice of the hyperplanes, via the characteristic polynomial. In the special case of an
    arrangement in general position in $\R^{n_0}$ it reduces to $\sum_{s\le n_0}\binom{m}{s}$:
    add the hyperplanes one at a time, and the $(m)$-th hyperplane is cut by the previous $m-1$ into
    $\sum_{s\le n_0-1}\binom{m-1}{s}$ pieces, each splitting one old region. Solving the recursion
    with the boundary value $r(\varnothing)=1$ gives the formula above. The bounded regions are
    counted by $\binom{m-1}{n_0}$.

In low dimension the formula is explicit and worth carrying in one's head. For $n_0=1$ a single
line splits the line into two, and $m$ points split it into $m+1$ intervals: $r = m+1$. For $n_0=2$
the arrangement of $m$ lines in general position satisfies the classical count (Eq. 4) [@pascanuNumberResponseRegions2013, §3]:

$$
r(A_m) = \frac{m(m-1)}{2} + m + 1 = \binom{m}{2} + m + 1
$$

The induction behind this is short and is the prototype for higher dimensions: suppose $m$ lines give
$r(A_m)$ regions and add a new line $L_{m+1}$. In general position it meets each old line at a
distinct point, so the $m$ intersection points divide $L_{m+1}$ into $m+1$ segments; each segment
cuts one old region in two, adding exactly $m+1$ regions. Hence $r(A_{m+1}) = r(A_m)+m+1$, and
starting from $r(A_0)=1$ the sum $\sum_{j=1}^{m}(j+1)$ gives the closed form
[@pascanuNumberResponseRegions2013, §3]. The same "each new hyperplane adds as many regions as it is
cut into" recursion is what underlies the general formula above; in $\R^{n_0}$ the added hyperplane is
cut by the old ones into regions of an $(n_0-1)$-dimensional arrangement, which is why the binomial
sum appears.

The counts for the two dimensions that recur in visualisations are:

| Input dimension $n_0$ | $m$ hyperplanes in general position | Regions |
|---|---|---|
| $1$ | $m$ points | $m+1$ |
| $2$ | $m$ lines | $\binom{m}{2}+m+1$ |
| $n_0$ | $m$ hyperplanes | $\sum_{s=0}^{n_0}\binom{m}{s}$ |

Two sanity checks follow immediately, and both are used later. If $m\le n_0$ then
$\sum_{s=0}^{n_0}\binom{m}{s}=2^m$: with at most as many hyperplanes as dimensions, every sign
pattern is realised, which is the statement that all $2^m$ activation patterns of a narrow layer are
reachable. And if the hyperplanes are *not* in general position — if several are parallel, or three
meet along a common line — then the count is strictly smaller: general position maximises, and any
arrangement can be perturbed by an arbitrarily small perturbation into general position
[@pascanuNumberResponseRegions2013, §3]. For a single rectifier layer with $n_1$ hidden units, the
maximal number of linear regions is therefore (Prop. 2 of [@pascanuNumberResponseRegions2013], and
the same count in [@NIPS2014_fa6f2a46, §2.2]):

$$
R(n_0;n_1;n_y) = \sum_{j=0}^{n_0}\binom{n_1}{j}
$$

independent of the number of outputs by Lemma 2. This is the first of the bounds developed in
[Counting linear regions](counting-regions.md), against which the deep constructions are measured
[@NIPS2014_fa6f2a46, §3].

## Upper envelopes and max-affine functions

A rectifier unit is the simplest of a family. Replace the two-branch function
$g_i(f_i)=\max\{0,f_i\}$ by a **rank-$k$ maxout unit**

$$
g_i(f_i) = \max\{f_{i,1},\dots,f_{i,k}\},
$$

where $f_{i,j}$ are affine functions of the preceding layer's activations; a maxout *layer* computes
one such maximum for each output unit [@NIPS2014_fa6f2a46, §2.1]. A ReLU is the rank-$2$ case in
which one of the two affine functions is the zero function, so everything said about maxout
specialises to rectifiers [@NIPS2014_fa6f2a46, §2.1].

A maximum of affine functions is a **max-affine function**, and it is convex whenever the pieces are
— the maximum of a collection of functions is their **upper envelope**, and a maximum of convex
functions is convex [@NIPS2014_fa6f2a46, §4]. This is why a single maxout *unit* is so simple to
count: if each affine piece $f_{i}$ uniquely maximises the envelope on some open neighbourhood, the
number of linear regions of $\max_i f_i$ is exactly $k$, so **the maximal number of linear regions of
a maxout unit equals its rank** [@NIPS2014_fa6f2a46, §4]. The linear regions of a whole maxout layer
are then the intersections of the regions of its units, and the partition each unit induces is a
lifted **Voronoi diagram**: Voronoi diagrams are upper envelopes of linear functions in one higher
dimension, so maxout layers are "intersections of Voronoi diagrams" [@NIPS2014_fa6f2a46, §4]. Since
intersecting $m$ Voronoi diagrams with $k$ regions each is not easy, the paper settles for bounds
rather than an exact formula for a shallow maxout layer (Prop. 8 of [@NIPS2014_fa6f2a46]):

$$
k^{\min\{n,m\}} \;\le\; \text{regions} \;\le\; \min\Bigl\{\sum_{j=0}^{n}\binom{k^2 m}{j},\ k^{m}\Bigr\},
$$

for $n$ inputs and $m$ rank-$k$ outputs. The trivial upper bound $k^m$ counts all combinations of
piece choices; the first bound refines it with the same convex-geometry count as for arrangements.
The deep construction gives at least $k^{L-1}k^{n_0}$ regions for a maxout network with $L$ layers of
width $n_0$ and rank $k$ [@NIPS2014_fa6f2a46, Thm. 9].

Two warnings belong here. First, the two families are not nested: although some functions computable
by a rank-$2$ maxout layer are computable by a rectifier layer of twice the width, the rectifier
constructions of [@NIPS2014_fa6f2a46, §3] produce functions that maxout networks cannot compute,
except in the rank-$2$ case [@NIPS2014_fa6f2a46, §4]. Comparing their region counts therefore
compares two different geometric mechanisms, not two settings of one dial. Second, an upper envelope
is the natural bridge to the tropical point of view: a max-affine function is a tropical polynomial,
and the boundaries between its pieces form a tropical hypersurface — the subject of
[Tropical geometry of ReLU networks](tropical-geometry.md).

## From local pieces to a global count

The reason this vocabulary is worth the trouble is that it turns "how expressive is this
architecture?" into "how many pieces does this partition have?". Already in
[@pascanuNumberResponseRegions2013, §1] the number of regions is offered as a measure of the
flexibility of the model — "the number of pieces available to the model in order to approximate some
arbitrary nonlinear function" — with the honest caveat that it is an *upper bound*, because the
pieces share parameters and cannot all be given independent slopes.

Above the shallow regime the geometry becomes genuinely hierarchical. Each hidden layer can map
several disjoint neighbourhoods of its input to the *same* output, so whatever the following layers
compute is replicated on all of them; a map $F$ that maps two neighbourhoods $S,T$ to a common image
$F(S)=F(T)$ is said to **identify** them, the canonical example being the absolute value
$g:\R^2\to\R^2$, $(x_1,x_2)\mapsto\bigl(|x_1|,|x_2|\bigr)$, which identifies the four quadrants
[@NIPS2014_fa6f2a46, §2.3, Definition 1 and Example 2]. Iterating identification across layers is
what [Space folding and folding neurons](space-folding.md) calls space folding: each layer folds its
input space, and a partition placed on the final folded space is copied back into every collapsed
subset [@NIPS2014_fa6f2a46, §2.4].

Counting this replication gives the deep lower bounds that [Counting linear regions](counting-regions.md)
develops in full. [@pascanuNumberResponseRegions2013] shows that a rectifier network with $n_0$
inputs and $k$ hidden layers of widths $n_1,\dots,n_k$ can realise (Thm. 1 of [@pascanuNumberResponseRegions2013]):

$$
\prod_{i=1}^{k-1}\left\lfloor\frac{n_i}{n_0}\right\rfloor \cdot \sum_{j=0}^{n_0}\binom{n_k}{j}
$$

regions, by using the first $k-1$ layers to identify many input regions and letting the last layer
place an arrangement in general position — contributing the binomial sum — inside all of them.
[@NIPS2014_fa6f2a46, Thm. 5] sharpens the first factor to
$\prod_{i=1}^{L-1}\lfloor n_i/n_0\rfloor^{n_0}$, giving the asymptotic lower bound
$\Omega\bigl((n/n_0)^{(L-1)n_0}n^{n_0}\bigr)$ for width-$n$ rectifier networks
[@NIPS2014_fa6f2a46, Cor. 6]. The gap between the two is exactly the extra folding of the
intermediate coordinate directions; the constructions differ and the reader should not read the second
as a corollary of the first. In both cases the count is an existence statement about *some* weights:
all regions have finite volume and the parameter-to-function map is continuous, so a parameter choice
attaining a given count is surrounded by an $\varepsilon$-ball of parameter choices attaining at least
as many — the maximal count is not a measure-zero accident, though it is also not a statement about
trained networks [@NIPS2014_fa6f2a46, §2.5].

!!! question "Open question"
    How many linear regions does a *trained* network typically have, as opposed to the maximum over
    parameters? [@NIPS2014_fa6f2a46, §2.5, §5] poses exactly this and leaves it open; the question is
    taken up by the average-case and typical-count bounds of later chapters. The polyhedral objects
    of this chapter are the same in both regimes — what changes is which points of parameter space the
    question samples.

## Summary

- A **convex polytope** is the convex hull of finitely many points, equivalently a bounded
  intersection of half-spaces; the $V$- and $H$-representations are the two forms in which network
  objects are written [@pakLecturesDiscretePolyhedral, p. 8].
- A **polyhedral complex** is a face-to-face collection of polytopes closed under taking faces; a map
  is **piecewise linear** when it is affine on each cell of such a complex
  [@pakLecturesDiscretePolyhedral, p. 8; @NIPS2014_fa6f2a46, §2.1].
- A **hyperplane arrangement** models a single rectifier layer: its regions are exactly the layer's
  linear regions, and they are labelled by sign vectors [@pascanuNumberResponseRegions2013, §3].
- In general position, $m$ hyperplanes in $\R^{n_0}$ cut out $\sum_{s=0}^{n_0}\binom{m}{s}$ regions;
  in the plane that is $\binom{m}{2}+m+1$, derived by the sweep induction
  [@pascanuNumberResponseRegions2013, Prop. 1, Eq. 4].
- A shallow rectifier layer with $n_1$ units has at most $\sum_{j=0}^{n_0}\binom{n_1}{j}$ regions,
  independent of the output dimension [@pascanuNumberResponseRegions2013, Prop. 2, Lemmas 1–2].
- A **maxout unit** is a max-affine function; its region count is its rank, and a maxout layer is an
  intersection of lifted Voronoi diagrams [@NIPS2014_fa6f2a46, §4].
- Deep layers **identify** input neighbourhoods, replicating higher-layer partitions; this yields
  exponential-in-depth lower bounds on the region count [@pascanuNumberResponseRegions2013, Thm. 1;
  @NIPS2014_fa6f2a46, Thm. 5, Cor. 6].
- All these counts are worst-case maxima over parameters, not properties of trained networks
  [@NIPS2014_fa6f2a46, §2.5].

## Further reading

- [@pakLecturesDiscretePolyhedral]: the discrete-geometry reference of the collection — polytopes and their combinatorics (§8, including Euler, Dehn–Sommerville and the $h$-vector), Minkowski sums and geometric inequalities (§7), and Steinitz and rigidity (§11 onward). Read §8.1 and the basic definitions (p. 8) first.
- [@pascanuNumberResponseRegions2013]: response regions of a rectifier MLP. The definition and the single-layer arrangement count are in §2–§3; the replication construction and the main lower bound in §4 (Thm. 1); the efficient $\lvert x\rvert$ construction in §5.
- [@NIPS2014_fa6f2a46]: the general piecewise-linear framework — identification and space folding (§2.3–2.4), the deep rectifier bound (§3, Thm. 5), and the first treatment of maxout networks (§4). Read alongside the previous paper.

## Exercises

**Exercise 1** ★ (check). Show that the number of linear regions of a single hidden rectifier layer
equals the number of regions of the arrangement $\{H_i\}$ of its switch hyperplanes, and explain why
the number of *output* units does not change this number.

??? success "Solution"
    Each unit $i$ has two modes, selected by the sign of $W_{i,:}x+b_i$, with the boundary the
    hyperplane $H_i$ [@pascanuNumberResponseRegions2013, §3]. The layer is a vector-valued map whose
    coordinates are the unit functions; a vector-valued map is affine exactly where all coordinates
    are, so its regions of linearity are the non-empty intersections of the coordinate regions
    [@pascanuNumberResponseRegions2013, Lemma 1]. Intersecting the half-spaces cut by the $H_i$
    reconstructs precisely the sign-vector cells of the arrangement, hence the regions coincide.
    The output dimension is irrelevant because a linear output layer only composes the last hidden
    activation with an affine map: if the output matrix is not orthogonal to any gradient difference
    between neighbouring regions, it preserves every region boundary, and one output row always
    suffices [@pascanuNumberResponseRegions2013, Lemma 2].

**Exercise 2** ★ (check). Verify that a shallow rectifier layer with $n_1\le n_0$ hidden units has at
most $2^{n_1}$ linear regions, and explain the equality case geometrically.

??? success "Solution"
    The count is $\sum_{j=0}^{n_0}\binom{n_1}{j}$ [@pascanuNumberResponseRegions2013, Prop. 2]. Since
    the binomial coefficients vanish for $j>n_1$, and $n_1\le n_0$, the sum is
    $\sum_{j=0}^{n_1}\binom{n_1}{j}=2^{n_1}$. Geometrically, at most as many hyperplanes as the
    ambient dimension are used, so each of the $2^{n_1}$ sign patterns is realised by some open
    region: the hyperplanes can be chosen in general position, and $\sum_{s\le n_0}\binom{m}{s}$ with
    $m=n_1\le n_0$ equals $\sum_{s\le n_1}\binom{n_1}{s}=2^{n_1}$
    [@pascanuNumberResponseRegions2013, §3, Prop. 1]. The bound is the same as the $2^N$ upper bound
    for a rectifier network with $N$ hidden units [@NIPS2014_fa6f2a46, Prop. 4]; the regime
    $n_1>n_0$, where the sum is strictly smaller than $2^{n_1}$, is where the arrangement no longer
    realises every sign pattern.

**Exercise 3** ★★ (derive). Starting from $r(A_0)=1$, derive the closed form
$r(A_m)=\binom{m}{2}+m+1$ for $m$ lines in general position in the plane, and state the analogous
recursion in $\R^{n_0}$.

??? success "Solution"
    Add the lines one at a time [@pascanuNumberResponseRegions2013, §3]. Induction step: if $A_m$ has
    $r(A_m)$ regions, a new line $L_{m+1}$ in general position meets the $m$ old lines at $m$
    distinct points, which split it into $m+1$ segments; each segment lies inside one old region and
    cuts it into two, adding exactly $m+1$ regions. Hence $r(A_{m+1})=r(A_m)+(m+1)$, and summing
    from $m=0$,

    $$
    r(A_m)=1+\sum_{j=1}^{m}(j+1)=1+\frac{m(m+1)}{2}+m=\frac{m(m-1)}{2}+m+1 .
    $$

    In $\R^{n_0}$ the same argument replaces the line by a hyperplane, which is cut by the previous
    $m-1$ hyperplanes into the regions of an $(n_0-1)$-dimensional arrangement, giving the recursion
    $r_{n_0}(m)=r_{n_0}(m-1)+r_{n_0-1}(m-1)$ and, by induction, the general-position count
    $\sum_{s=0}^{n_0}\binom{m}{s}$ [@pascanuNumberResponseRegions2013, Prop. 1].

**Exercise 4** ★★ (compare). Both a shallow rectifier layer and a shallow maxout layer are built from
finitely many affine functions. Compare how the number of linear regions behaves in the two cases,
and say what each representation gives up.

??? success "Solution"
    A rectifier layer's partition is a hyperplane arrangement: its regions are sign-vector cells and
    the count is $\sum_{j=0}^{n_0}\binom{n_1}{j}$ [@pascanuNumberResponseRegions2013, Prop. 2]. A
    rank-$k$ maxout layer instead takes maxima: a single maxout unit has as many regions as its rank
    [@NIPS2014_fa6f2a46, §4], and the layer's partition is an intersection of lifted Voronoi diagrams,
    for which only the bounds $k^{\min\{n,m\}}$ and
    $\min\{\sum_{j=0}^{n}\binom{k^2 m}{j},k^{m}\}$ are available in general
    [@NIPS2014_fa6f2a46, Prop. 8]. A maxout unit *gives up* linearity of the individual pieces in
    exchange for a convex, order-statistic structure — the unit outputs the $k$-th order statistic of
    $k$ affine functions — and its regions are convex cells of a Voronoi-type diagram rather than
    sign cells. The two families are not nested: the rectifier constructions produce functions not
    computable by maxout networks except in the rank-$2$ case [@NIPS2014_fa6f2a46, §4]. In practice
    one prefers whichever mechanism matches the task; the papers explicitly decline to conclude
    which is more powerful [@NIPS2014_fa6f2a46, §5].

**Exercise 5** ★★ (derive). Why does a single rank-$k$ maxout unit have exactly $k$ linear regions,
and why is a ReLU the rank-$2$ special case of this?

??? success "Solution"
    The unit computes $g(z)=\max\{z_1,\dots,z_k\}$ with $z_j$ affine in the layer input
    [@NIPS2014_fa6f2a46, §2.1]. The maximum of affine functions is their upper envelope; each $z_j$
    is a supporting hyperplane of the epigraph, and if each $z_j$ is the unique maximiser on some
    open neighbourhood, the domain splits into $k$ cells on which a different affine function is
    active [@NIPS2014_fa6f2a46, §4]. Adding one point to each cell, the count cannot exceed $k$ (the
    trivial bound $k^m$ with $m=1$), and the parallel-hyperplane example attains it, so the maximal
    count equals the rank [@NIPS2014_fa6f2a46, §4, Prop. 8]. A rectifier $g_i(f_i)=\max\{0,f_i\}$ is
    the case $k=2$ with one branch the zero function, so its region count is $2$
    ($k=2$), matching the fact that a ReLU switches between "dead" and "active"
    [@NIPS2014_fa6f2a46, §2.1; @pascanuNumberResponseRegions2013, §3].

**Exercise 6** ★★ (design / apply). The construction of [@pascanuNumberResponseRegions2013, §5]
uses four rectifiers $f_1=\max\{0,[1,0]^\top x\}$, $f_2=\max\{0,[-1,0]^\top x\}$,
$f_3=\max\{0,[0,1]^\top x\}$, $f_4=\max\{0,[0,-1]^\top x\}$ and the pair-combination
$g=[[1,1,0,0],[0,0,1,1]]\,f$ to mimic $(|x_1|,|x_2|)$. Explain what this writer does geometrically and
how many regions the combined map has.

??? success "Solution"
    On the quadrant $x_1\ge 0$ only $f_1$ of the first pair is active, contributing $x_1$; on
    $x_1<0$ only $f_2$ is active, contributing $-x_1$; so the first output of $g$ is $|x_1|$, and
    likewise the second is $|x_2|$ [@pascanuNumberResponseRegions2013, §5, Eq. 5]. The pair therefore
    implements the absolute-value map $g=(|x_1|,|x_2|)$, which is symmetric about both coordinate
    axes: it **identifies** the four quadrants, each pair of points $(x)$ and its reflection mapping
    to the same activation [@pascanuNumberResponseRegions2013, §5; @NIPS2014_fa6f2a46, §2.3,
    Example 2]. The map $g$ itself has the four quadrants as its linear regions; applied again to the
    image it subdivides each of the four into four more, giving $16$ regions after two such layers
    [@pascanuNumberResponseRegions2013, §5]. This is the recursive folding that multiplies region
    counts with depth.

**Exercise 7** ★★★ (critique). Every quantitative result in this chapter is a bound on the *maximal*
number of linear regions over parameters. What does that leave undetermined, and what did
[@NIPS2014_fa6f2a46] argue about how non-degenerate the maximum is?

??? success "Solution"
    The bounds are existential: [@pascanuNumberResponseRegions2013, Thm. 1] and
    [@NIPS2014_fa6f2a46, Thm. 5] exhibit a choice of weights attaining the count, but say nothing
    about the weights a network learns, so the counts are upper bounds on flexibility rather than
    measurements [@pascanuNumberResponseRegions2013, §1; @NIPS2014_fa6f2a46, §5]. On non-degeneracy,
    [@NIPS2014_fa6f2a46, §2.5] argues that each linear region has finite volume and the
    parameter-to-function map is continuous, so any parameter choice attaining a given count has an
    $\varepsilon$-ball of choices attaining at least as many; the maximum is therefore attained on a
    set of non-zero volume, not just at isolated points. That still does not say training *finds*
    such parameters, and the paper explicitly leaves open how many points of parameter space give
    functions with a given region count, and how many regions typical trained networks have
    [@NIPS2014_fa6f2a46, §2.5, §5].
