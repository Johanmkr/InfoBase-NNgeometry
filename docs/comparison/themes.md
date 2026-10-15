---
title: Themes
built_from:
  NIPS2014_fa6f2a46: e11f3b75c4ab
  alfarraDecisionBoundariesNeural2023: a305d52d8025
  arvanitidisGeometricallyEnrichedLatent2020: 1e269d18ad0c
  arvanitidisPullingBackInformation2022: b6e2d23bcb04
  averkovEXPRESSIVENESSRATIONALRELU2025: e57bb3498e66
  balestrieroSplineTheoryDeep2018: c06113a95fb7
  berzinsGeometryInformedNeuralNetworks2024: 7622243e3191
  berzinsPolyhedralComplexExtraction2023: ce2242f7b108
  brandenburgDecompositionPolyhedraPiecewise2024: 307dddfb6bce
  bronsteinGeometricDeepLearning: 6abf4c1c9edc
  bronsteinGeometricDeepLearning2017: c17967a216bb
  carlssonPREIMAGERECTIFIERNETWORK2017: c2b08537722e
  chazalIntroductionTopologicalData2021: ac2d04605929
  combeExploringInformationGeometry2025: f2e5daaa2fa6
  dhayalkarDiscreteFunctionalGeometry2025: 6c54339045f0
  fuLatentTopologyInduction2022: c2ba4261ab8e
  gagneuxHowImproveExpressivity2025: 3626abad8f47
  goujonNumberRegionsPiecewise2023: fb83a40ca26c
  grigsbyHiddenSymmetriesReLU: 2b3f7998f609
  halversonMetricFlowsNeural2024: 2fd2a25dba82
  haninComplexityLinearRegions2019: 438efc947567
  haninDeepReLUNetworks2019: d172a42e0d06
  humayunSplineCamExactVisualization2024: e413cd7b883c
  keupOrigamiDimensionsHow2022: eb997e66ea80
  lewandowskiExploitingSpaceFolding2026: 79b187d59987
  lewandowskiSpaceFoldsReLU2024: c213be11a02d
  liuIntegratingGeometriesReLU2023a: 74fb72720d6b
  liuReLUNeuralNetworks2023: 0877601e9e2e
  masdenAlgorithmicDeterminationCombinatorial2022: 83d84f985dd7
  meilaManifoldLearningWhat2023: 9d14430d7f11
  nielsenElementaryIntroductionInformation2020: 68ed040d3708
  pakLecturesDiscretePolyhedral: b12d340fddd1
  pascanuNumberResponseRegions2013: 3fae3832ff4c
  phuongFunctionalVsParametric2019: 79b70de62319
  pouplinIdentifyingLatentDistances2023: d11ea5c75a33
  raghuExpressivePowerDeep2017: effafc1ee765
  sattelbergLocallyLinearAttributes2023: 731de219deef
  serraBoundingCountingLinear2018: 9cdce7bede89
  villarFullyCovariantMachine2023: 206ac70ee271
  vincentReachablePolyhedralMarching2021: 7fe473aac8ec
  zhangTropicalGeometryDeep: 56eee67d20f1
---

# Themes

Seven essays on questions that cut across the collection, then the open problems the papers state
themselves. Each essay reports what the papers say with locators, where they **agree**, where they
**contradict** (both sides cited, with the reason when one can be found), and ends with my own
synthesis, worded as such.

## 1. Four exact dictionaries for the same object

The collection contains four formally different descriptions of what a ReLU network is, and they
describe the same thing. **Spline**: a network assembled from affine maps and piecewise-affine convex
nonlinearities is a composition of max-affine spline operators, so conditioned on the input it
computes a signal-dependent affine map $z=A[x]x+B[x]$
[@balestrieroSplineTheoryDeep2018, Thm. 1, Eq. 5], and the same per-region affine parameters
$A^\ell_\omega,b^\ell_\omega$ are what SplineCam computes and back-projects
[@humayunSplineCamExactVisualization2024, Lemma 1, Eq. 3]. **Polyhedral complex**: the cells on
which the network is affine form a complex whose faces can be labelled injectively by the *signs* of
the intermediate node maps, a ternary sequence in $\{-1,0,1\}^N$ dual to a pure cubical complex
[@masdenAlgorithmicDeterminationCombinatorial2022, Thm. 14, Thm. 20]; in the binary version of the
same idea, one bit per hidden unit identifies a cell and a single flipped bit identifies a neighbour
[@berzinsPolyhedralComplexExtraction2023, Eq. 1; @liuIntegratingGeometriesReLU2023a, Eq. 5].
**Tropical**: with integer weights and threshold activations, a network *is* a tropical rational map
$F\oslash G$, and the tropical dictionary translates linear regions into cells of a dual
subdivision of a Newton polytope [@zhangTropicalGeometryDeep, Thm. 5.2, Thm. 5.4].
**Arrangement**: for a shallow layer the partition is simply a hyperplane arrangement, and its region
count is the classical face count $\sum_{j=0}^{n_0}\binom{n_1}{j}$
[@pascanuNumberResponseRegions2013, Prop. 1, Prop. 2].

Where they **agree** is on the local statement: the boundaries of the pieces are the zero sets of the
pre-activations — the bent hyperplanes $H_z=\{x:z(x)=b_z\}$ — and the pieces are the connected
components of their complement [@haninDeepReLUNetworks2019, Lemma 2], which is the same statement as
the "folded hyperplane" of [@berzinsPolyhedralComplexExtraction2023, §3.4] and the "piece-wise
hyperplane" of the canonical representation [@phuongFunctionalVsParametric2019, §5]. Every one of
the four dictionaries therefore implies that a network restricted to a cell is a single affine map,
which is what makes the geometry comparable at all.

Where they **contradict** is on how much the dictionary must record. The binary activation pattern is
provably insufficient: two networks can carry the same set of activation-pattern strings on their
top-dimensional cells and still have non-homeomorphic decision boundaries
[@masdenAlgorithmicDeterminationCombinatorial2022, Thm. 15], so [@liuReLUNeuralNetworks2023],
[@liuIntegratingGeometriesReLU2023a], [@sattelbergLocallyLinearAttributes2023] and
[@lewandowskiSpaceFoldsReLU2024] all work with an object — bits, plus adjacency for the first two —
that the sign-sequence route shows is lossy. The reason is definitional rather than mathematical: the
binary string records which side of each bent hyperplane a point lies on, and adjacency and gluing
are extra data, whereas the ternary string records membership in each hyperplane, which is exactly
what a face poset needs [@masdenAlgorithmicDeterminationCombinatorial2022, Lemma 16]. A second
disagreement is about what "a region" means: the counting tradition treats regions as maximal
connected sets on which the map is affine [@NIPS2014_fa6f2a46, §2.1], while
[@haninDeepReLUNetworks2019, Lemma 3] distinguishes those *linear* regions from the convex
*activation* regions and shows there are always at least as many of the latter; [@goujonNumberRegionsPiecewise2023, §2.2]
notes that the literature uses "projection regions" and "convex regions" interchangeably and fixes
the convex convention. The counts differ because the objects differ, and both papers say so.

My reading is that the four dictionaries are best thought of as different coordinate systems on one
manifold: the tropical and polyhedral views are exact and structurally rich, the spline view is the
one that connects to approximation theory, and the arrangement view is the one that makes counting
tractable. The practical consequence is that a reader should pick the dictionary by the question —
tropical for decision boundaries and pruning [@alfarraDecisionBoundariesNeural2023, Thm. 2],
polyhedral for enumeration and verification [@vincentReachablePolyhedralMarching2021, §IV], spline
for visualisation and for talking about knots along a direction
[@humayunSplineCamExactVisualization2024, §3.2] — and should not expect the counts obtained in one
dictionary to transfer verbatim to another.

## 2. Worst-case expressivity versus what typical networks do

The cleanest factual agreement in the collection is about the maximum. Every counting paper agrees
that a deep, narrow, piecewise-linear network can be made to produce exponentially many linear
regions in its depth and only polynomially many in its width, and the constructions differ
[@pascanuNumberResponseRegions2013, Thm. 1]; [@NIPS2014_fa6f2a46, Thm. 5, Thm. 9];
[@serraBoundingCountingLinear2018, Thm. 8]; [@zhangTropicalGeometryDeep, Thm. 6.3];
[@goujonNumberRegionsPiecewise2023, Cor. 1]. The strongest form of the agreement is
[@raghuExpressivePowerDeep2017, Thm. 1]: an upper bound valid for *all* weight settings matches the
hand-built lower bound, so the worst case is not an artefact of the construction — the maximal
count really is $\Theta(k^{mn})$ for $m$-dimensional inputs. [@goujonNumberRegionsPiecewise2023,
Thm. 2] extends the same two-sided statement to arbitrary continuous piecewise-linear activations,
and [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2] shows that depth is also *necessary* for exact
representation of $\max\{0,x_1,\dots,x_n\}$ once weights are rational with bounded denominator.

The reversal is equally well documented. Along a one-dimensional path, the expected number of
regions is proportional to the number of neurons and independent of the arrangement of layers
[@haninComplexityLinearRegions2019, Thm. 1]; the expected density of activation regions in a cube is
bounded by $(T\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$ with no depth dependence at all
[@haninDeepReLUNetworks2019, Thm. 5]; and for random layers the expected knot density along a path
is at most $D_0^*(\lambda_0 W L)$, so depth contributes only a linear factor
[@goujonNumberRegionsPiecewise2023, Cor. 3]. Experiments agree with the bounds: the distance from a
random point to the nearest region boundary stays within a factor of four of
$1/\#\{\text{neurons}\}$ throughout training [@haninComplexityLinearRegions2019, Fig. 6(a)], and the
region count never grows exponentially even on memorisation tasks
[@haninDeepReLUNetworks2019, §4].

Is this a **contradiction**? The two families measure different things, and the papers say so.
[@goujonNumberRegionsPiecewise2023, §4.3.1] explains that its stochastic assumption — independent
layers — holds at initialisation and not during learning, so its bound describes the measure on
parameters, not the training path; [@haninDeepReLUNetworks2019, §3.1] states its bound is on the
*mean* of a random variable and that the constants $T,C_{\mathrm{grad}},C_{\mathrm{bias}}$ are not
effective; [@NIPS2014_fa6f2a46, §2.5] argues only that the maximal count is attained on an open set
of parameters, which is a statement about the maximum, not about the mean. A sharper tension, though,
is real: [@haninComplexityLinearRegions2019, §6] concludes that deeper networks do not learn more
complex functions than shallow ones *on this measure*, while [@serraBoundingCountingLinear2018,
Cor. 3] shows that for $n_0\ge Ln$ a deep network's maximal count is *below* that of a shallow
network of the same total width — so the quantity that is supposed to certify the value of depth is,
in one regime, maximised by depth and in another by width.

Taken together, I would read the collection as making an operational point rather than a negative
one: the counting measures are measures of the *architecture*, not of the model, and the papers that
use them to compare trained networks do so with the small differences they expect. The practical
divisions are visible where the counting thread meets applications: [@sattelbergLocallyLinearAttributes2023,
Tab. 1] replaces every region's affine map by one of ten cluster centres and retains most of the
accuracy of a convolutional network, and [@vincentReachablePolyhedralMarching2021, §IV-A] uses the
polynomial count of activation patterns at initialisation to justify marching rather than
enumerating. Both use the typical-count picture, not the worst case.

A late 2025 attempt closes the loop by making the *graph itself* the capacity measure: the ReLU
transition graph's node count, degree distribution, diameter and region-volume entropy are proposed
as stand-ins for expressivity, with a VC-dimension bound by the graph diameter as the headline
[@dhayalkarDiscreteFunctionalGeometry2025, Def. 3, Thm. 3]. It **agrees** with the collection's
premise that counting alone is insufficient — the stated motivation is exactly that counting ignores
how regions are organised [@dhayalkarDiscreteFunctionalGeometry2025, §1] — and its size bound is the
Zaslavsky count composed over layers, i.e. the same quantity as
[@NIPS2014_fa6f2a46, Thm. 5], which the paper credits
[@dhayalkarDiscreteFunctionalGeometry2025, §4.1]. It is **weaker** on both counts, and the paper's
own text supplies the evidence: the size proof loosens the correct product form with an inequality
that is false in the direction stated, so the printed numerical column is not an upper bound
[@dhayalkarDiscreteFunctionalGeometry2025, App. A.1]; and the diameter-to-VC-dimension argument is a
sketch whose intermediate count is never related to a shattered set
[@dhayalkarDiscreteFunctionalGeometry2025, App. A.5]. Set against
[@raghuExpressivePowerDeep2017, Thm. 1], an upper bound valid for every weight setting, and against
the depth-independent average-case bounds [@haninDeepReLUNetworks2019, Thm. 5], the RTG claim would
be a genuine capacity statement if it held — a graph invariant is cheaper to estimate than a region
count — but the collection's two established capacity statements remain the counting ones, and this
one is a conjecture with a proof-shaped appendix.

## 3. What folding does and does not buy

Folding enters the collection as the *explanation* of depth: a layer can map several disjoint input
regions onto the same activation, so whatever the following layers compute is replicated in all of
them, which is what turns depth into a multiplicative factor in the region count
[@NIPS2014_fa6f2a46, §2.3, §2.4]. [@keupOrigamiDimensionsHow2022, §3] makes the mechanism concrete
and argues that folding *into unoccupied dimensions* is the efficient route: it is injective, the
exposed fold edge presents a previously interior part of the distribution to a flat readout, an
$N$-dimensional "egg" problem is approximately solved by one layer of $N+1$ neurons, and the
alternative — compressing the data with shear until it separates — needs depth growing with
dimension [@keupOrigamiDimensionsHow2022, App. A]. The collection's two quantitative entries then
turn the picture into a measurement: a straight input segment mapped into Hamming activation space
folds exactly when its distance to the start fails to increase monotonically, giving a bounded
measure $\chi$ [@lewandowskiSpaceFoldsReLU2024, Eq. 4, Lemma 2], which is generalised to any
monotonic activation through equivalence classes of inputs
[@lewandowskiExploitingSpaceFolding2026, Def. 3] and made into a pseudo-metric on inputs
[@lewandowskiExploitingSpaceFolding2026, Prop. 2].

The **agreement** is strong on mechanism: folding is a change of the *combinatorial* object — which
inputs share an activation pattern, i.e. an identification of inputs — rather than a metric
distortion of space. [@phuongFunctionalVsParametric2019, §4] makes the same object the load-bearing
one for a theorem: the fold-set of a general, transparent network is the union of the units'
zero-level sets, and the dependency graph of the resulting piecewise-linear surface determines the
weights up to permutation and scaling [@phuongFunctionalVsParametric2019, Lemma 1, Thm. 1]. The
folding literature and the identifiability literature are describing the same folds from two sides.

The **contradictions** are specific. First, scope: [@lewandowskiSpaceFoldsReLU2024, Lemma 1] proves
that folding effects are *invisible* in a network with a single hidden layer, because a straight path
in input space maps to a shortest path in Hamming space when the hyperplanes cut the whole space,
which the authors themselves flag as an assumption that fails for deeper layers
[@lewandowskiSpaceFoldsReLU2024, §4]. That sits awkwardly next to
[@keupOrigamiDimensionsHow2022, §3], where a single layer of $N+1$ neurons is exactly the
construction that manufactures separability. The two are reconcilable — Keup and Helias need extra
dimensions, Lewandowski et al. keep the dimension fixed and measure a different quantity — but
neither paper does the reconciliation. Second, evidential weight:
[@lewandowskiSpaceFoldsReLU2024, §7] infers that the maximal folding value is associated with
generalisation capacity from one Pearson coefficient (0.987) over architectures of differing depth
at fixed neuron budget for one seed, which the follow-up does not repair — it reports a reversal in
the intra- versus inter-class ordering of $\Phi_N$ as a Remark that motivates a conjecture
[@lewandowskiExploitingSpaceFolding2026, Remark 1, Conjecture 2]. Third, ubiquity:
[@lewandowskiSpaceFoldsReLU2024, §6.1] finds the fraction of folding paths rising from $0.35$ at
$2\times30$ to $0.99$ at $3\times200$, and [@lewandowskiExploitingSpaceFolding2026, §Sparsity] finds
the opposite trend in the sparsity $S_N$ as the neuron count grows. A quantity that saturates is a
poor discriminator, which is the collection's own criticism rather than mine.

My synthesis: folding is best treated as the *combinatorial content* of a ReLU layer — the map from
input regions to activation patterns — rather than as an explanation of generalisation. On that
reading the collection's folding papers and its identifiability papers
([@phuongFunctionalVsParametric2019; @lewandowskiExploitingSpaceFolding2026]) are the direct
continuation of [@NIPS2014_fa6f2a46]'s identification argument, while the fold-and-cut analogy to
universal approximation [@keupOrigamiDimensionsHow2022, §3] remains a picture whose rigorous
$N$-dimensional version the authors explicitly leave open.

## 4. Computing the complex: five exact routes and their cost frontier

Five papers in the collection compute the same object exactly and arrive from five directions.
**Edge subdivision** keeps only the 1-skeleton and splits edges cut by a folded hyperplane, recovering
2-faces by perturbing sign-vectors, because the continuity of the activation means neighbouring
regions share each new vertex [@berzinsPolyhedralComplexExtraction2023, §4]; it runs as tensor
operations on a GPU in time linear in the number of vertices
[@berzinsPolyhedralComplexExtraction2023, §5.1.2]. **Region subdivision with back-projection**
enumerates cells of a 2-D slice, projecting each deeper-layer hyperplane onto every region's tangent
space and finding cycles in the resulting planar graph
[@humayunSplineCamExactVisualization2024, §2.2, Alg. 1–2]. **Marching** starts from one cell and
derives neighbours by flipping terminal neurons, so the reachable set is enumerated without ever
iterating layer by layer [@vincentReachablePolyhedralMarching2021, §IV-B]. **Sign-sequence duality**
computes vertices layer by layer and reads the entire face poset off their signs, without ever
intersecting regions numerically [@masdenAlgorithmicDeterminationCombinatorial2022, §4.1]. And a
**mixed-integer encoding** counts regions of a fixed network by enumerating integer solutions of a
big-$M$ formulation [@serraBoundingCountingLinear2018, §5, Thm. 11].

They **agree** on the algorithmic premise that the network is continuous, and this is where the
engineering comes from: continuity is what lets edge subdivision reuse vertices
[@berzinsPolyhedralComplexExtraction2023, §1], what makes the partition a decomposition rather than
a covering for SplineCam [@humayunSplineCamExactVisualization2024, Lemma 1], and what makes the
per-cell affine map agree on shared faces in the marching and traversal schemes
[@liuReLUNeuralNetworks2023, §2.3; @liuIntegratingGeometriesReLU2023a, §2.4]. They also agree by
construction: [@berzinsPolyhedralComplexExtraction2023, §5.1.2] benchmarks *against* SplineCam,
[@liuIntegratingGeometriesReLU2023a, §2.4] describes its enumeration as aligning with
[@vincentReachablePolyhedralMarching2021], [@masdenAlgorithmicDeterminationCombinatorial2022, §2]
contrasts its route with both, and [@gagneuxHowImproveExpressivity2025, §3] uses the edge-subdivision
routine as a component when checking convexity. This is a rare case in the collection of five papers
that can be cross-validated against each other.

A sixth, cheaper entry sits outside the exactness competition:
[@dhayalkarDiscreteFunctionalGeometry2025, §5.1] does not enumerate the complex, it samples activation
patterns on a grid and joins them at Hamming distance one, producing the ReLU transition graph
[@dhayalkarDiscreteFunctionalGeometry2025, Def. 3] — "the region complex" reduced to its 1-skeleton,
with no vertex coordinates, no face poset and no boundary. It **agrees** with the five exact routes on
the object and, like them, on continuity as the premise: it needs the Hamming-1 rule to be the
adjacency relation, which is the same local statement the edge-subdivision and marching schemes rest on
[@dhayalkarDiscreteFunctionalGeometry2025, §4.2; @berzinsPolyhedralComplexExtraction2023, Eq. 1;
@vincentReachablePolyhedralMarching2021, §IV-B]. It is **weaker** as a description by construction:
the graph is built from the rule it then tests, so its 100/100 adjacency check verifies the code, not
the geometry [@dhayalkarDiscreteFunctionalGeometry2025, §5.3], and at $(L{=}4,n{=}128)$ the sampler
produces 4.40 components on average, an artefact the authors attribute to grid sparsity
[@dhayalkarDiscreteFunctionalGeometry2025, §5.3.1]. Where the exact methods state their input-dimension
limits openly [@berzinsPolyhedralComplexExtraction2023, §1;
@masdenAlgorithmicDeterminationCombinatorial2022, §6], the RTG's own connectivity theorem applies to
the continuous network rather than to its sample, which is exactly the gap the experiment falls into.

Where they **contradict** is on feasibility, and the reason is the domain. SplineCam is exact "up to
machine precision" on 2-D slices only [@humayunSplineCamExactVisualization2024, §2.2], and its
authors say the method is not claimed for the full input space;
[@berzinsPolyhedralComplexExtraction2023, §1] reports that even its own route is impractical for
$D>8$; [@masdenAlgorithmicDeterminationCombinatorial2022, §6] states its algorithm is exponential in
the input dimension; [@serraBoundingCountingLinear2018, §6] can count only networks of two hidden
layers summing to 22 neurons; and [@liuReLUNeuralNetworks2023, §3] concludes bluntly that all exact
decomposition methods are inadequate for the large and deep networks that are commonplace today —
which is why its ResNet-50 experiment deliberately never enumerates anything and uses only the bit
vectors of sampled points. The competing speed claims therefore do not conflict: 20 times faster than
SplineCam on 2-D inputs [@berzinsPolyhedralComplexExtraction2023, §5.1.2], and three orders of
magnitude on unsafe instances for marching [@vincentReachablePolyhedralMarching2021, Tab. I], are
statements about different regimes. The one genuine trade-off both sides acknowledge is anytime
behaviour: marching is markedly faster when a counterexample exists and about twice as slow when the
system is safe [@vincentReachablePolyhedralMarching2021, §VI-B].

I read this as the collection's most mature axis: the object is well defined, five independent
implementations exist, and their limits are stated in terms of the input dimension, the domain
(bounded or not) and the question asked (the complex, the count, a safety certificate, or a picture).
The open engineering question is not which method is fastest but whether the *region-label pipeline*
used at scale — bit vectors of sampled points fed to a Hamming metric
[@liuReLUNeuralNetworks2023, §4.2] — preserves the structure that the exact methods recover, given
that the binary label is known to be lossy on the face poset
[@masdenAlgorithmicDeterminationCombinatorial2022, Thm. 15].

## 5. Symmetry: parametric versus functional equivalence

Two theorems in the collection say essentially the same thing about when a ReLU parameter is
determined by its function, and they cover disjoint architecture classes. For **non-increasing
widths** with a one-dimensional output, there *exists* a network $h$ such that any general network
computing the same function is related to $h$ by permutations of units and positive scalings
[@phuongFunctionalVsParametric2019, Thm. 1]; the proof builds the network so that its fold-set has a
well-connected dependency graph, and reads layer membership off the length of directed paths in it
[@phuongFunctionalVsParametric2019, Lemma 3, Lemma 4]. For **no hidden layer narrower than the
input**, a positive-measure set of parameters has no hidden symmetries at all
[@grigsbyHiddenSymmetriesReLU, Thm. 4.1], proved by constructing parameters whose bent hyperplanes
pairwise intersect transversally and satisfy the Linear Regions Assumption, then using openness to
propagate the conclusion to a neighbourhood [@grigsbyHiddenSymmetriesReLU, §4, Cor. D.13]. The papers
state the complementarity themselves, and both note the case that neither covers — an architecture
with some layers wider and some narrower than the input, e.g. $(n_0,n_1,n_2,n_3,n_4)$ with $n_0<n_1$
and $n_2<n_0$ — as open [@grigsbyHiddenSymmetriesReLU, §2; @phuongFunctionalVsParametric2019, §6].

The two results also **agree** on the mechanism: hidden symmetries arise where the network's
structure degenerates. Four degeneration mechanisms are catalogued — a stably unactivated neuron, two
neurons never coactive, a later layer collapsing earlier regions, and an image of positive
codimension [@grigsbyHiddenSymmetriesReLU, §5] — and each corresponds to a way in which the
dependency graph fails to pin down a hyperplane in the other paper's construction
[@phuongFunctionalVsParametric2019, §5]. The functional view of the same question is taken further by
the folding literature, which defines equivalence of *inputs* rather than of parameters: two inputs
are equivalent under $\chi$ when the walk between them does not fold
[@lewandowskiExploitingSpaceFolding2026, Def. 3], and the symmetrised version is only a pseudo-metric
precisely because that equivalence is not equality [@lewandowskiExploitingSpaceFolding2026, Prop. 2].
[@haninDeepReLUNetworks2019, §4.2, Lemma 15] supplies two further exact symmetries of the same
function: uniform rescaling of all biases leaves the global region count unchanged, and a zero-bias
network is positively scale equivariant, which is why its region count is bounded by that of the same
network with biases [@haninDeepReLUNetworks2019, Lemma 6, Prop. 14]. That parameters cannot be read
as the function for the analogous reason — permuting units or rescaling their incoming and outgoing
weights leaves the mapping unchanged — is the motivation SplineCam gives for computing the geometry
of the partition rather than inspecting weights [@humayunSplineCamExactVisualization2024, §1].

The sharpest **contradiction** in this area is about relevance rather than truth. Both symmetry
theorems are existential and about given parameters: [@phuongFunctionalVsParametric2019, §7] states
that typical networks may not have well-connected dependency graphs and may therefore not be
identifiable, while [@grigsbyHiddenSymmetriesReLU, §6] measures how *often* the good case occurs and
finds that the probability of no hidden symmetries falls steeply with depth (66% at depth 4 and
width 15, 5% at depth 6) — i.e. the architectures where the worst case matters most are the ones
where the theorem's hypothesis is least often satisfied in the measurement. The reason is that the
two statements are about different things: one is about the existence of an identifiable *network*,
the other about the frequency of an identifiable *parameter*, and the second is a probabilistic
statement about initialisation, not about the trained network.

A second tension is conceptual and is stated by the papers: geometric deep learning takes symmetry
as the design principle, deriving architectures from the symmetry of the domain
[@bronsteinGeometricDeepLearning, §3.5], with linear equivariants characterised by two generators on
sets and fifteen on graphs, independently of $n$ [@bronsteinGeometricDeepLearning, §4.1]; while
[@villarFullyCovariantMachine2023, §1, §3] argues that the symmetries induced by the investigator's
choice of coordinates and units are exact and nearly ignored, and that the empirical invariances the
equivariant literature targets are of a different, approximate kind. The collection also contains an
open question that connects the two: whether the success of ReLU is related to its homogeneity, since
a homogeneous nonlinearity acts on quantities with units whereas sigmoid can only act on
dimensionless scalars [@villarFullyCovariantMachine2023, §9].

My synthesis: the collection is converging on the view that "the same network" is a three-level
notion — same function [@phuongFunctionalVsParametric2019], same function up to permutation and
scaling [@grigsbyHiddenSymmetriesReLU], and same identification of inputs
[@lewandowskiExploitingSpaceFolding2026] — and that only the middle level has theorems in two
complementary width regimes. Which of the three is the right notion for reasoning about training is
not settled anywhere in the collection, and the measurements in
[@grigsbyHiddenSymmetriesReLU, §6] suggest the answer depends on depth in a way the existing
statements do not express.

## 6. Latent geometry: ambient, Fisher–Rao and Finsler metrics

Four papers in the collection give a latent space a metric, and they disagree about which metric.
[@arvanitidisGeometricallyEnrichedLatent2020, Eq. 3] pulls an *ambient* Riemannian metric, chosen to
encode domain knowledge, back through the generator Jacobian, so that shortest paths are
identifiability-free and steerable; for a stochastic generator it relaxes the expected metric by
evaluating the ambient metric at the mean output [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 8,
Eq. 9] and argues the relaxation is harmless because the variance term dominates far from the data
[@arvanitidisGeometricallyEnrichedLatent2020, App. D]. [@arvanitidisPullingBackInformation2022,
Prop. 3.2] replaces the ambient metric by the Fisher–Rao metric of the likelihood's parameter space,
which removes the reliance on a location-scale likelihood and makes the construction work for
Bernoulli, Beta, Dirichlet, exponential and von Mises–Fisher decoders
[@arvanitidisPullingBackInformation2022, §3, §4.3]. [@pouplinIdentifyingLatentDistances2023,
Def. 2.5] keeps the data-manifold view but changes the object: the expectation of the metric *tensor*
is not the expectation of the *norm*, and the latter is a Finsler metric, proved to satisfy the
Finsler axioms and to admit a closed form for Gaussian-process decoders
[@pouplinIdentifyingLatentDistances2023, Prop. 2.2, Prop. 2.3]. [@meilaManifoldLearningWhat2023,
§5.5] supplies the mirror image — a push-forward metric that quantifies the distortion of an
embedding rather than repairing a decoder.

The **agreement** is on the premise and on the remedy. All four treat Euclidean distance in a latent
space as meaningless: distances must be built from the model or the data, and geodesics of the
resulting metric are the right interpolation primitive
[@arvanitidisGeometricallyEnrichedLatent2020, §2]; [@arvanitidisPullingBackInformation2022, §1];
[@pouplinIdentifyingLatentDistances2023, §2]. They also
agree that the construction buys invariance to reparametrisation, and
[@arvanitidisPullingBackInformation2022, Prop. 3.3] shows its metric is the Fisher–Rao metric with
the latent variable itself as parameter — the same object whose uniqueness under Markov embeddings
is the background result of [@nielsenElementaryIntroductionInformation2020, Thm. 8]. The
mirror-image question, how much an embedding distorts the geometry it is meant to preserve, is
answered by a push-forward metric whose deviation from the identity quantifies the distortion
[@meilaManifoldLearningWhat2023, §5.5].

The **contradictions** are concrete. First, scope: [@pouplinIdentifyingLatentDistances2023, §1.2]
requires a smooth stochastic immersion, which *excludes* the VAE decoder because its posterior
independence makes the Jacobian covariance a Dirac delta — yet the VAE is exactly the model
[@arvanitidisPullingBackInformation2022, §4.1] handles, by working in the space of distributions
rather than in the data space. The two are not incompatible, but a reader looking for one recipe
finds two scopes. Second, quality: [@arvanitidisPullingBackInformation2022, §6] and
[@arvanitidisGeometricallyEnrichedLatent2020, §3.2, §4] both make their geometry depend on the decoder
being confident where data are sparse, and the former reports a failure case (Bernoulli, where the
regularisation is too weak because parameters near $1/2$ are already highly entropic) and states that
principled calibration is currently lacking; [@pouplinIdentifyingLatentDistances2023, §5] concludes
in the opposite direction by showing the Finsler/Riemannian gap is $O(1/D)$ and therefore that the
simpler Riemannian approximation is good enough in practice. Third, and most instructive,
[@halversonMetricFlowsNeural2024, §3.2] finds that methods with a *frozen* kernel learn an
approximate Calabi–Yau metric far worse than finite networks — no more than a factor of about 5 in
test loss against 10–100 for the finite model — while its own measurement shows the metric neural
tangent kernel of a successful finite network changing measurably during training
[@halversonMetricFlowsNeural2024, §3, Fig. 4]. That is evidence that the learned geometry is
produced by feature learning, not by a fixed kernel, which cuts against the lazy-training picture
that makes kernel approximations attractive in the first place.

My synthesis: the collection's practical consensus is the one
[@pouplinIdentifyingLatentDistances2023, §5] states explicitly — use the Riemannian approximation,
because the deviation from the exact expected-norm geometry shrinks with dimension — and the more
interesting open question is the one the same paper leaves: when is the variance term *not*
negligible. On the evidence here that happens exactly in the low-dimensional, toy-scale regimes where
all four papers demonstrate their methods, and disappears on the image datasets (font, MNIST,
FashionMNIST) where the metric is nearly deterministic.

## 7. Topology: stable invariants of a network's own decomposition

Topological data analysis enters the collection twice: as a toolbox for describing the network's
complex and as a constraint on what the network outputs. The toolchain and its stability theory come
from [@chazalIntroductionTopologicalData2021, §5.7]: persistence diagrams are stable in bottleneck
distance under $\delta$-interleaving, under sup-norm perturbations of a function, and under
Gromov–Hausdorff perturbation of the space up to a factor of two
[@chazalIntroductionTopologicalData2021, Thm. 7, Prop. 1, Thm. 9], and diagram estimation achieves
the minimax rate $(\log n/n)^{1/b}$ under a standard-assumption on the measure
[@chazalIntroductionTopologicalData2021, Thm. 10]. The network-side application uses the ReLU
activation label of each point as a coarse metric: a pairwise Hamming matrix of binary activation
vectors is fed to a Vietoris–Rips filtration, and the circle homology of data sampled through a
trained network is recovered despite the coarseness of the quantisation
[@liuReLUNeuralNetworks2023, §4.2, Example 4.1–4.2], including for ResNet-50 on ImageNet images
where the network was trained for an unrelated purpose [@liuReLUNeuralNetworks2023, Example 4.3].
Independently, [@masdenAlgorithmicDeterminationCombinatorial2022, §4.2] computes the *exact* Betti
numbers of a decision boundary by restricting the mod-two cellular coboundary of the sign-sequence
cubical complex to the cells whose last sign vanishes; bounded components turn out to be rare (8.1%
of shallow networks, 0.05% of four-dimensional-input ones) and deeper networks show more variable
boundary topology [@masdenAlgorithmicDeterminationCombinatorial2022, §5.2]. At the other end,
[@berzinsGeometryInformedNeuralNetworks2024, §4.1, Eq. 24] turns persistent homology into a
*differentiable loss*: the connectedness of a shape is enforced by pushing down all but the most
persistent component at an iso-level, which makes a persistent-homology library usable inside optimisation
without differentiating through it.

The **agreement** is on why topology is the right invariant: the objects being summarised — a
polyhedral complex, a decision boundary, a shape's level set — are stable under the perturbations
that a numeric pipeline introduces, and persistence is the cheapest summary with a continuity
theorem. [@masdenAlgorithmicDeterminationCombinatorial2022, §5] and
[@liuReLUNeuralNetworks2023, §1] are computing invariants of the same complex by different means —
one exactly from vertices, one approximately from sampled labels — and
[@berzinsGeometryInformedNeuralNetworks2024, §4.1] adds the third use, as a target.

The **contradictions** are about what the pipeline guarantees.
[@liuReLUNeuralNetworks2023, §2.3] states outright that the Hamming distance is only an
*approximation* of the shortest path in the dual graph, and illustrates it rather than bounding it;
by contrast, [@chazalIntroductionTopologicalData2021] derives its statistics from the Lipschitz
stability of the diagram map, so the collection contains one pipeline with a proven stability
mechanism and one with none, and only the latter is run at scale. The reason is that the Hamming
metric is not a metric on the underlying space but on labels of a partition, so nothing in the
stability theory applies directly — which is exactly the gap that would need to be closed for the
ImageNet-scale experiment to be a theorem rather than a demonstration. A second disagreement is about
reliability of the summary itself: [@chazalIntroductionTopologicalData2021, §3] calls Mapper very
sensitive to its resolution and gain parameters, so that small changes can produce very large changes
in output, while the same survey's stability results concern filtrations, not Mapper. Within the
collection, Mapper appears only as a survey topic and in a passing conceptual link
[@sattelbergLocallyLinearAttributes2023], so the instability has no consequences here — but a reader
who adopts the "vectorise a diagram and classify" recipe of
[@chazalIntroductionTopologicalData2021, §7] should know that the choice of scale, which the survey
itself flags as unresolved, is not fixed by the theory.

My synthesis: the collection uses topology in two directions that have not yet met. Looking inward —
describing a network's own complex — it has an exact method with real guarantees only in low input
dimension [@masdenAlgorithmicDeterminationCombinatorial2022, §6] and an approximate method that works
at ImageNet scale with no guarantee [@liuReLUNeuralNetworks2023, §4.2].
[@dhayalkarDiscreteFunctionalGeometry2025, Def. 3, §5.1] adds a third and cheaper probe of the same
dual graph — Hamming-1 adjacency of patterns sampled on a grid — which reports its sampling artefact
rather than bounding it [@dhayalkarDiscreteFunctionalGeometry2025, §5.3.1]. Looking outward —
constraining a learned representation — it has a differentiable topological objective with no
guarantee about the optimiser [@berzinsGeometryInformedNeuralNetworks2024, §4.1]. The obvious missing
paper is the one that bounds the Hamming metric of the first by the dual-graph metric, which is what
would make the large-scale topological description of a network a statement instead of a picture.

## 8. Open problems

The problems below are the ones the papers state themselves; the citation points at the statement,
not at my paraphrase.

**Expressivity and depth.**
- The depth lower bound $\lceil\log_2(n+1)\rceil$ for exact representation of $\max\{0,x_1,\dots,x_n\}$
  is open for general real weights; the collection's result covers only rational weights with bounded
  denominator [@averkovEXPRESSIVENESSRATIONALRELU2025, Conjecture 1, §1].
- No effective total region count is known: the results bound the number of regions along a curve or
  the boundary volume, and the total count is believed but not proved to be proportional to the
  number of neurons [@haninComplexityLinearRegions2019, §2.1].
- Whether many regions of *parameter* space correspond to functions with a given number of linear
  regions is left open by the paper that first raised the counting programme
  [@NIPS2014_fa6f2a46, §2.5, §5].
- How the region count relates to accuracy is only initial evidence: a very thin layer can give high
  error despite many regions [@serraBoundingCountingLinear2018, §7].

**Decompositions and their cost.**
- Is a decomposition into two convex pieces with polynomially many pieces always possible, and can a
  minimal one be found without fixing the underlying complex in advance
  [@brandenburgDecompositionPolyhedraPiecewise2024, Problem 7.1, §4]?
- Is there a tight *linear* bound on the maxout knot density, in place of the quadratic bound in the
  number of units [@goujonNumberRegionsPiecewise2023, §4.2]?
- Scaling the convexity penalty of [@gagneuxHowImproveExpressivity2025, Eq. 5] to practical networks
  is called a computational challenge that would open new ways to train more expressive convex
  networks [@gagneuxHowImproveExpressivity2025, §5].
- Exact extraction remains impractical beyond low input dimension, and replacing the sorting step by
  hash tables is named as the route to linear time and memory
  [@berzinsPolyhedralComplexExtraction2023, §5.1, §6].
- The transition-graph construction is only defined and measured for fully connected ReLU MLPs on
  synthetic low-dimensional inputs; extending it to convolutional, residual and attention models and
  to real datasets is stated as future work, as is checking whether the entropy-versus-degree
  inequality survives without the uniform-region-volume assumption
  [@dhayalkarDiscreteFunctionalGeometry2025, §6, §7].

**Symmetry, equivalence and identifiability.**
- The mixed width regime — some hidden layers wider, some narrower than the input — is uncovered by
  both symmetry theorems [@grigsbyHiddenSymmetriesReLU, §2;
  @phuongFunctionalVsParametric2019, §6].
- Whether typical or trained networks have well-connected dependency graphs, and hence are
  identifiable, is explicitly unknown [@phuongFunctionalVsParametric2019, §7].
- The classification of hidden-symmetry mechanisms is partial, and the observed multimodal spacing of
  the functional-dimension distribution is unexplained [@grigsbyHiddenSymmetriesReLU, §5, §6].
- Whether the success of ReLU is connected to its being a homogeneous nonlinearity is offered as an
  open question rather than a finding [@villarFullyCovariantMachine2023, §9].

**Folding and its observables.**
- The fold-and-cut analogy to universal approximation has no rigorous $N$-dimensional version
  [@keupOrigamiDimensionsHow2022, §3], and the efficiency claim for folding without extra dimensions
  rests on a construction rather than a calculation [@keupOrigamiDimensionsHow2022, App. A].
- Whether folding predicts generalisation is a conjecture
  [@lewandowskiExploitingSpaceFolding2026, Conjecture 2, Conjecture 3], and its relation to
  adversarial geometry is explicitly preliminary [@lewandowskiExploitingSpaceFolding2026, §Interaction
  Coefficient].
- The vanishing of folding in single-hidden-layer networks is proved only under an assumption about
  full-space hyperplanes that deeper layers violate [@lewandowskiSpaceFoldsReLU2024, §4].

**Geometry of representations.**
- Principled calibration of decoder uncertainty is lacking, and the pulled-back geometry is only as
  good as that calibration [@arvanitidisPullingBackInformation2022, §6].
- Whether the Finsler correction matters outside Gaussian-process decoders — in particular for VAEs,
  which the derivations exclude — is open [@pouplinIdentifyingLatentDistances2023, §1.2, §5].
- More stable ways of approximating pulled-back metrics are needed, since the KL-based estimate can
  be indefinite near the boundary of the data support
  [@arvanitidisPullingBackInformation2022, §4.4].
- The approximation of the expected metric by the metric at the mean output is argued and checked
  empirically but not derived [@arvanitidisGeometricallyEnrichedLatent2020, §4, App. D], and a
  principled way to choose the mixture weights of several ambient metrics is left open
  [@arvanitidisGeometricallyEnrichedLatent2020, App. E.2].

**Topology and scale.**
- The scale parameter at which to read Betti numbers off a Vietoris–Rips filtration remains a
  practical difficulty, and the confidence regions derived from stability can be very conservative
  [@chazalIntroductionTopologicalData2021, §4.2, §6.2].
- The validity of the bottleneck bootstrap is stated as unproved in the survey's framework
  [@chazalIntroductionTopologicalData2021, §7.1].
- Whether the coarseness of the Hamming quantisation actually helps against noise is a conjecture, and
  the reason for the observed homological noise is unknown
  [@liuReLUNeuralNetworks2023, §4.2, §5].
- The conjecture that deeper networks need more width before reaching a "Gaussian regime" of boundary
  topology is offered without evidence [@masdenAlgorithmicDeterminationCombinatorial2022, §5.2].

**Description versus prediction.** Three papers observe a geometric quantity tracking performance and
each labels the link a hypothesis rather than a result: folding against generalisation
[@lewandowskiSpaceFoldsReLU2024, §7], functional dimension against loss-landscape flatness
[@grigsbyHiddenSymmetriesReLU, §7], and partition density tails near training points as possible
memorisation [@humayunSplineCamExactVisualization2024, App. D.1]. Taken together with the
depth-independent bounds [@haninDeepReLUNetworks2019, Thm. 5] and the failure of frozen-kernel
methods to learn the same geometry [@halversonMetricFlowsNeural2024, §3.2], the collection's most
conspicuous gap is not any single theorem but the absence of a geometric quantity that both predicts
generalisation and has a proof attached.
