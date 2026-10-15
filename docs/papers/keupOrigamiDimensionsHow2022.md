---
citekey: keupOrigamiDimensionsHow2022
title: "Origami in N Dimensions: How Feed-Forward Networks Manufacture Linear Separability"
authors: [Christian Keup, Moritz Helias]
year: 2022
venue: arXiv preprint
doi: "10.48550/arXiv.2203.11355"
arxiv: "2203.11355"
url: null
kind: perspective
source_type: pdf
source_sha256: 5cbcfdaff46b8ce38bf54e9f70aa8e92e66edb232e0763417ac99daa0f642770
pages: 17
ingested: 2026-10-15
tags: [relu-networks, linear-separability, folding, representation-geometry, interpretability]
concepts_introduced:
  - folding in unoccupied dimensions
  - progressive folding / N-dimensional origami
  - separability generation by shear (compressive folding)
  - bimodal tuning curves as a signature of folds
  - representation dimensionality as a fold observable
  - angle between a neuron's normal vector and the data subspace
  - fold-and-cut analogy to universal approximation
concepts_used:
  - ReLU network
  - piecewise-linear function
  - hyperplane arrangement
  - affine transformation
  - convex hull
  - linear separability
  - universal approximation
  - Hartigan dip statistic
builds_on: [balestrieroSplineTheoryDeep2018, carlssonPREIMAGERECTIFIERNETWORK2017, raghuExpressivePowerDeep2017, haninDeepReLUNetworks2019, zhangTropicalGeometryDeep]
datasets: [poker hand (UCI)]
---

# Origami in N Dimensions: How Feed-Forward Networks Manufacture Linear Separability

**Keup, Helias (2022)**, *arXiv preprint*. [arXiv:2203.11355](https://arxiv.org/abs/2203.11355)

!!! abstract "TL;DR"
    A feed-forward classifier must turn linearly non-separable data into a linearly separable
    representation. Keup and Helias argue that the one efficient tool at its disposal is
    *progressive folding* of the data manifold into dimensions the data does not occupy: a ReLU
    hyperplane with a component outside the representation subspace folds the manifold along its
    edge, exposing previously interior regions to a flat readout. Folding scales as one hyperplane
    per fold and one layer of $N+1$ neurons for an $N$-dimensional "egg", whereas the alternative
    shear mechanism needs depth growing with dimension. The mechanistic picture predicts that
    beneficial folds show up as *bimodal* class tuning curves, which are observed in a
    three-layer network trained on the poker hand task.

## Problem and motivation

The paper asks what fundamental *operations* a feed-forward network has available to build its
output, rather than studying the loss landscape or approximation in the infinite-width limit
(§1, p. 1–2). For classification, the last hidden layer must hold a representation in which the
classes are linearly separable, so the network must progressively transform an initially
non-separable distribution into a separable one. The authors focus on ReLU networks because they
are piecewise-linear functions whose piecewise borders are the readout hyperplanes of the
neurons, and argue that the relative arrangement of these hyperplanes is enough to understand the
transformation (§1, p. 2). The stated aim is intuition rather than analysis: the authors explicitly
abstain from most mathematics and hope the picture will guide proofs about trainability and
generalization (§1, p. 2).

## Method

Each layer alternates an affine map with an element-wise nonlinearity (§2, Eq. 1). The affine part
alone cannot change linear separability, since it maps lines to lines; it only rotates, scales and
positions the distribution "on the anvil" so that the nonlinearity can act on it (§2, p. 3). The
ReLU nonlinearity is pictured as $N$ hyperplanes ($\tilde x_i = 0$ for each neuron) that project every
point on the negative side onto its hyperplane, so its combined effect is to press the
representation into the corner of an $N$-dimensional room — a "hammer" making flat dents (§2,
p. 3). Crucially, such a map can only modify the *convex hull* of the representation, and cannot
selectively target interior regions.

The central argument (§3) is that when a layer is wider than the data manifold, some activation
directions are unoccupied. A ReLU hyperplane with a component in such a direction does not squash
data points together; it *folds* the manifold into the unused dimension (injective and nonlinear).
Along the folding edge, an interior region of the distribution becomes exposed to the embedding
space, so a linear readout from the next layer can separate parts that were previously
inaccessible (Fig. 2a,b). The normal vector must have an angle strictly between zero and
$\pi/2$ with the representation subspace to create a fold. In a deep network the folds are applied
progressively to the already-folded object, so the number of edges can grow exponentially in the
total number of folds, giving the origami analogy (§3, p. 5; Fig. 2).

The authors propose three observables to detect folding in trained high-dimensional networks (§4):
incremental PCA of pre- and post-activations to track how many directions acquire nonzero variance
(folding raises dimensionality); class-resolved tuning curves that should become *bimodal* (a dip
near the folding edge, with the central peak present for some classes and absent for others);
and the angle between a neuron's normal vector and the data subspace, which for fold-creating units
should be intermediate. Neurons with smoother nonlinearities are analysed through their
preactivations, since ReLU produces a delta peak at zero in the activation tuning curves. The
validation is a fully connected 3-layer ReLU network on the UCI poker hand task (§4.1).

## Key equations and definitions

The layer update (§2, Eq. 1, p. 2):

$$
x_i^{(l)} = \phi\!\left(\sum_j W^{(l)}_{ij}\, x^{(l-1)}_j + b^{(l)}_i\right) \qquad \text{(Eq. 1, p. 2)}
$$

with the preactivation written separately as an unnumbered equation:

$$
\tilde x^{(l)} = W^{(l)} \tilde x^{(l-1)} + \tilde b^{(l)} \qquad \text{(§2, unnumbered eq., p. 2)}
$$

and the activation

$$
\phi(\tilde x) = \mathrm{ReLU}(\tilde x) = \max(0, \tilde x) \qquad \text{(§2, unnumbered eq., p. 2)}
$$

The ReLU hyperplanes $\tilde x_i = 0$ have normal vectors given by the rows of $W^{(l)}$ and
displacements by the biases $b^{(l)}$ (§2, p. 3; §4, p. 5).

In Appendix B (compressive folding without extra dimensions) the authors state that the
information-losing ("squashing") variant of the operation is limited: (1) any single angle along
the piecewise-linear path of squashing must be smaller than $\pi/2$, and (b) the sum of angles in
any direction along the path is strictly bounded from above by $\pi$ (App. B, p. 13). The upper
bound's symbol is not legible in the extracted text (only "bounded from above by" is printed);
$\pi$ is the value consistent with the surrounding discussion of half-circles versus full circles
and is flagged here as reconstructed.

| Symbol | Meaning (in this paper) |
|---|---|
| $x_i^{(l)}$ | activation of neuron $i$ in layer $l$ |
| $\tilde x^{(l)}$ | preactivation vector of layer $l$ |
| $W^{(l)}, b^{(l)}$ | weights and biases of layer $l$ (rows = normals, entries = displacements) |
| $\phi$ | element-wise activation; $\mathrm{ReLU}$ unless stated otherwise |
| $P_{x^{(0)}}$ | ground-truth data distribution over classes |
| $N$ | layer width / ambient dimensionality of the representation |

## Main results

- **Fold mechanism (§3, p. 4–5).** Folding in unoccupied dimensions is more efficient than shear:
  several folds can be applied by a single layer, and an "$N$-dimensional egg" problem is
  approximately solved by *one* layer with $N+1$ neurons. The exposed folding edge is always a
  hyperplane one dimension below the folded object.
- **Toy "2d-egg" (§3, Fig. 2c,d, p. 4).** Three hidden neurons in a single layer approximately
  solve the problem; recall is 100% for the inner class and 96.8% for the outer class.
- **Fold-and-cut analogy (§3, p. 5).** The fold-and-cut theorem (a flat sheet folded so that one
  straight cut separates arbitrary drawn shapes; Demaine et al. 1998; Bern and Hayes 2011) is
  presented as analogous to universal approximation in ReLU networks, with the flat readout
  hyperplane playing the role of the cut. The authors label this an enticing possibility and leave
  a rigorous generalization to $N$ dimensions to future work.
- **Shear is inefficient (App. A, p. 12–13).** Without extra dimensions, the only method found is
  to shave slices off the outer shell using non-normal hyperplanes, one operation per layer. Even
  the 2d-egg needs 7 layers of width 2 (Fig. 5c–e); in 3d the accumulated mass must trace a spiral
  around the sphere, so depth grows worse than linearly. The deep fixed-width limitation is
  connected to the result that deep fixed-width networks are not universal approximators
  (Johnson 2018, outside the collection).
- **Compressive folding (App. B, p. 13).** Squashing without extra dimensions can straighten one
  angle of a piecewise-linear boundary per hyperplane, $N$ times per layer, but is non-injective;
  a half-circle can be approximated by successive tangential squashes, a full circle cannot.
- **Poker hand validation (§4.1, Fig. 3, p. 6–8).** Networks $10 \to 100 \to 100 \to 10$,
  cross-entropy, momentum 0.9, batch size 500, learning rate 0.1 for 50 epochs then 0.01 for 50
  epochs, $1.2\times10^4$ parameters against $10^6$ training samples (not overparameterized).
  Training accuracy is 100% on all classes except the difficult set $\{5,8,9\}$; the mean intact
  macro-averaged F1 is 0.97. Incremental PCA shows strong folding-related dimensionality increase
  (Fig. 3b); trained preactivation tuning curves are bimodal/multimodal with the central peak
  missing for some classes and present for others (Fig. 4a,c), unlike the near-Gaussian untrained
  curves (Fig. 4b). Hartigan's dip statistic shifts to higher values after training (Fig. 3c).
  Neurons with large dip for at least one class have normals at intermediate angles to the data
  subspace (Fig. 3d), though the angle distribution does not separate large- from small-dip
  neurons. Silencing 10 large-dip neurons causes a considerable accuracy loss, while 10 small-dip
  neurons have little effect (Fig. 3e). No angle is close to zero, so shear plays no role here.

## Limitations stated by the authors

The work is a first step restricted to fully connected, deep feed-forward networks and
classification tasks (Limitations, p. 10). The claim that separability generation without
dimensionality expansion is inefficient is *not* quantified by a rigorous calculation, because a
proof would need assumptions about realistic data distributions. The poker hand task is small
scale, its categorical data make the bimodality analysis less clear-cut, and an extension to
convolutional architectures is needed to validate on large image models (Limitations, p. 10).

## Critical assessment

This is a perspective paper, and it reads as one: the central objects (folding, the "hammer",
origami) are pictures, not theorems, and the authors say so. The strength is that the picture
makes concrete, falsifiable predictions — bimodality, intermediate angles, dimensionality growth —
and tests them on one real task, which is more than such papers usually do. The evidence, however,
is thin in exactly the places that matter: the validation is a single small dataset, the angle
observable does not discriminate large- from small-dip neurons as predicted, and the stronger
"shear plays only a small role in real-world networks" claim rests on the observation that no
angle is near zero in one network rather than on a measurement across architectures. The
fold-and-cut correspondence is asserted as a "remarkably analogous" possibility with no proof, and
the authors correctly flag it as future work; a reader should not take it as a theorem. The claim
that folding "is sufficient to provide the universal approximation property" (§3, p. 4) is
supported only by the appeal to fold-and-cut plus the fixed-width counterexample, so it is a
plausibility argument, not a result. The compressive-folding bound in Appendix B is the most
concrete technical statement and would be worth stating cleanly. No code or data are released.
The most useful thing about the paper is the vocabulary: it suggests experimenters measure
dimensionality expansion, tuning-curve bimodality and normal-to-subspace angles, all of which are
cheap. Whether folding is really *the* mechanism, or whether training merely happens to produce
folds as one of many correlated signatures, is not settled here.

## Connections within the collection

The paper positions itself inside the "ReLU networks are piecewise-linear" literature and cites
several works summarised here as components of its argument. It builds on the affine-spline view
of deep networks of [@balestrieroSplineTheoryDeep2018], which supplies the picture of a network as
a spline operator over a hyperplane arrangement, and on the study of the *preimage* of ReLU
network activities of [@carlssonPREIMAGERECTIFIERNETWORK2017], one of the works it calls closely
related (§Related work, p. 10). It relies on the expressivity analysis of [@raghuExpressivePowerDeep2017]
for the hyperplane-arrangement description of piecewise-linear functions and on
[@haninDeepReLUNetworks2019] for the few-activation-patterns view of deep ReLU geometry. The
tropical-geometry route to ReLU geometry of [@zhangTropicalGeometryDeep] is cited as an alternative
manifold viewpoint (§Related work, p. 10).

Among the papers of the collection that postdate it, the space-fold programme —
[@lewandowskiSpaceFoldsReLU2024] and [@lewandowskiExploitingSpaceFolding2026] — is plainly a
follow-up: it takes up the folding operation and studies it formally, whereas Keup and Helias
introduce it as intuition. The polyhedral-decomposition and region-counting line
([@berzinsPolyhedralComplexExtraction2023], [@liuReLUNeuralNetworks2023],
[@masdenAlgorithmicDeterminationCombinatorial2022], [@brandenburgDecompositionPolyhedraPiecewise2024],
[@goujonNumberRegionsPiecewise2023]) analyses the same hyperplane arrangement that defines the
folds but from a combinatorial rather than a separability-generation standpoint; the tropical
decision-boundary work of [@alfarraDecisionBoundariesNeural2023] is a thematic neighbour. The
observables proposed here (tuning-curve bimodality, normal-to-subspace angle) are complementary to
the value-based symmetry analysis of [@grigsbyHiddenSymmetriesReLU] and the exact visualisation of
[@humayunSplineCamExactVisualization2024] and [@sattelbergLocallyLinearAttributes2023], which
inspect local linearity rather than folding. Johnson (2018), cited for the fixed-width
non-universality link, is outside the collection.
