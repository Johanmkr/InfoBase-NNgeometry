# Textbook

The material of the whole collection in teaching order. Each chapter ends with exercises.

## 1. [Polyhedra, arrangements and piecewise-linear maps](polyhedral-foundations.md)

Read a piecewise-linear map as a polyhedral object: state what a polyhedral complex, a hyperplane arrangement and a max-affine function are, and count the regions of a generic arrangement in low dimension. This is the discrete-geometry background every later chapter uses.

*Sources:* [Pak n.d.](../papers/pakLecturesDiscretePolyhedral.md), [Montúfar et al. 2014](../papers/NIPS2014_fa6f2a46.md), [Pascanu et al. 2013](../papers/pascanuNumberResponseRegions2013.md)

## 2. [The region complex of a ReLU network](relu-network-geometry.md)

Decompose the input space of a ReLU network into linear regions, label each region by its activation pattern, and describe the region complex, its boundaries and its transition graph. The reader can compute the local affine map of any region, state what the transition graph's size, degree and diameter do and do not bound, and say when two parametrisations give the same region structure.

*Sources:* [Liu et al. 2023](../papers/liuIntegratingGeometriesReLU2023a.md), [Sattelberg et al. 2023](../papers/sattelbergLocallyLinearAttributes2023.md), [Masden 2022](../papers/masdenAlgorithmicDeterminationCombinatorial2022.md), [Serra et al. 2018](../papers/serraBoundingCountingLinear2018.md), [Berzins 2023](../papers/berzinsPolyhedralComplexExtraction2023.md), [Dhayalkar 2025](../papers/dhayalkarDiscreteFunctionalGeometry2025.md)

## 3. [Counting linear regions](counting-regions.md)

Derive and compare the bounds on the number of linear regions of a ReLU or maxout network, explain the depth–width trade-off they express, and choose the right bound for a given architecture and input dimension.

*Sources:* [Montúfar et al. 2014](../papers/NIPS2014_fa6f2a46.md), [Pascanu et al. 2013](../papers/pascanuNumberResponseRegions2013.md), [Serra et al. 2018](../papers/serraBoundingCountingLinear2018.md), [Hanin and Rolnick 2019](../papers/haninComplexityLinearRegions2019.md), [Hanin and Rolnick 2019](../papers/haninDeepReLUNetworks2019.md), [Goujon et al. 2023](../papers/goujonNumberRegionsPiecewise2023.md), [Averkov et al. 2025](../papers/averkovEXPRESSIVENESSRATIONALRELU2025.md), [Gagneux et al. 2025](../papers/gagneuxHowImproveExpressivity2025.md), [Brandenburg et al. 2024](../papers/brandenburgDecompositionPolyhedraPiecewise2024.md)

## 4. [Space folding and folding neurons](space-folding.md)

Explain how ReLU units fold space, distinguish folding that creates linear separation from folding that does not, and measure the amount of folding along a path or on a manifold. The reader can identify the neurons responsible for a given fold.

*Sources:* [Keup and Helias 2022](../papers/keupOrigamiDimensionsHow2022.md), [Lewandowski et al. 2024](../papers/lewandowskiSpaceFoldsReLU2024.md), [Lewandowski et al. 2026](../papers/lewandowskiExploitingSpaceFolding2026.md), [Montúfar et al. 2014](../papers/NIPS2014_fa6f2a46.md)

## 5. [Deep networks as splines](spline-networks.md)

Read a deep piecewise-linear network as a continuous spline: state its max-affine form, its knots along a direction, and its local affine templates, and use exact region visualisation to inspect a trained model.

*Sources:* [Balestriero and baraniuk 2018](../papers/balestrieroSplineTheoryDeep2018.md), [Humayun et al. 2024](../papers/humayunSplineCamExactVisualization2024.md), [Sattelberg et al. 2023](../papers/sattelbergLocallyLinearAttributes2023.md)

## 6. [Tropical geometry of ReLU networks](tropical-geometry.md)

Translate a ReLU network into a tropical rational function, read its Newton polytope and dual subdivision as the network's region structure, and use that representation to describe decision boundaries, prune the network and build adversarial examples.

*Sources:* [Zhang et al. n.d.](../papers/zhangTropicalGeometryDeep.md), [Alfarra et al. 2023](../papers/alfarraDecisionBoundariesNeural2023.md), [Brandenburg et al. 2024](../papers/brandenburgDecompositionPolyhedraPiecewise2024.md), [Averkov et al. 2025](../papers/averkovEXPRESSIVENESSRATIONALRELU2025.md)

## 7. [Topology of regions and of data](region-topology.md)

Use simplicial and cubical complexes, persistent homology and distance-to-measure to describe a network's region complex and its training data, and explain why the resulting invariants are stable.

*Sources:* [Chazal and Michel 2021](../papers/chazalIntroductionTopologicalData2021.md), [Liu et al. 2023](../papers/liuReLUNeuralNetworks2023.md), [Masden 2022](../papers/masdenAlgorithmicDeterminationCombinatorial2022.md), [Berzins 2023](../papers/berzinsPolyhedralComplexExtraction2023.md)

## 8. [Exact enumeration, preimages and verification](verification-and-preimages.md)

Enumerate exactly the linear regions a network reaches from a set of input polytopes, compute preimages of layer activities, and use both to verify a network's output on a region of interest.

*Sources:* [Vincent and Schwager 2021](../papers/vincentReachablePolyhedralMarching2021.md), [Serra et al. 2018](../papers/serraBoundingCountingLinear2018.md), [Carlsson et al. 2017](../papers/carlssonPREIMAGERECTIFIERNETWORK2017.md), [Gagneux et al. 2025](../papers/gagneuxHowImproveExpressivity2025.md)

## 9. [Information geometry](information-geometry.md)

Define a statistical manifold, its Fisher–Rao metric, the induced divergences and the α-connections, and derive natural-gradient updates from that geometry. The reader can state which of these objects survive reparametrisation.

*Sources:* [Nielsen 2020](../papers/nielsenElementaryIntroductionInformation2020.md), [Combe et al. 2025](../papers/combeExploringInformationGeometry2025.md)

## 10. [Manifold learning and intrinsic dimension](manifold-learning.md)

State the manifold hypothesis precisely, compare the estimators of intrinsic dimension and the embeddings built on them, and explain what a latent topology induction adds to an unsupervised model.

*Sources:* [Meil\u a and Zhang 2023](../papers/meilaManifoldLearningWhat2023.md), [Fu and Lapata 2022](../papers/fuLatentTopologyInduction2022.md), [Raghu et al. 2017](../papers/raghuExpressivePowerDeep2017.md)

## 11. [Riemannian and Finsler geometry of latent spaces](latent-geometry.md)

Pull a metric back through a decoder, write the geodesic shortest path it defines, and replace the Riemannian metric by an expectation-based Finsler one when the data lie on a manifold. The reader can say how such a metric changes interpolation and clustering in the latent space.

*Sources:* [Arvanitidis et al. 2020](../papers/arvanitidisGeometricallyEnrichedLatent2020.md), [Arvanitidis et al. 2022](../papers/arvanitidisPullingBackInformation2022.md), [Pouplin et al. 2023](../papers/pouplinIdentifyingLatentDistances2023.md)

## 12. [Metric flows with neural networks](metric-flows.md)

Set up a metric evolving under a geometric functional, in the style of Ricci flow, and learn the flow with a neural network. The reader can state what the flow preserves and how it is discretised.

*Sources:* [Halverson and Ruehle 2024](../papers/halversonMetricFlowsNeural2024.md)

## 13. [Symmetry and geometric deep learning](symmetry-and-gdl.md)

Derive the invariance and equivariance that a dataset's symmetry group forces on the architecture, read the geometric deep learning blueprint, and catalog the parameter-space symmetries of ReLU networks and the functional equivalences they produce.

*Sources:* [Bronstein et al. 2017](../papers/bronsteinGeometricDeepLearning2017.md), [Bronstein et al. n.d.](../papers/bronsteinGeometricDeepLearning.md), [Villar et al. 2023](../papers/villarFullyCovariantMachine2023.md), [Grigsby et al. n.d.](../papers/grigsbyHiddenSymmetriesReLU.md), [Phuong and Lampert 2019](../papers/phuongFunctionalVsParametric2019.md)

## 14. [Implicit and geometry-informed representations](implicit-geometry.md)

Represent a shape by the zero level set of a network, train such a representation to meet a geometric objective, and extract the polyhedral complex it induces. The reader can state when an implicit representation is exact and when it only approximates.

*Sources:* [Berzins et al. 2024](../papers/berzinsGeometryInformedNeuralNetworks2024.md), [Berzins 2023](../papers/berzinsPolyhedralComplexExtraction2023.md), [Humayun et al. 2024](../papers/humayunSplineCamExactVisualization2024.md), [Fu and Lapata 2022](../papers/fuLatentTopologyInduction2022.md)
