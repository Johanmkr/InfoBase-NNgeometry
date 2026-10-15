---
title: Comparison tables
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

# Comparison tables

Five tables, each on one axis, so that no single table has to carry both "what the paper does" and
"what it achieved". Symbols follow the [shared notation](../notation.md); a paper keeps its own
symbols in its summary, and the notation page maps them.

!!! info "How to read the tables"
    The paper column carries the citation, so a row is traceable to a summary with locators; in the
    bound tables the locator is attached to the bound itself. "—" means *not reported*, never a
    guess. Where a paper has no experiments at all, it appears only in the first and last table.

## Method family, mechanism and geometric object

| Paper | Family | Key mechanism | Geometric object or deliverable |
|---|---|---|---|
| [@pascanuNumberResponseRegions2013, Thm. 1] | worst-case counting | hyperplane-arrangement count in general position, plus a replication construction that maps several input cells onto the ball | response regions of a rectifier MLP |
| [@NIPS2014_fa6f2a46, Thm. 5] | worst-case counting | *identification* of input neighbourhoods, i.e. deliberate folding of one region onto another | linear regions; Voronoi cells of a maxout layer |
| [@raghuExpressivePowerDeep2017, Thm. 1] | worst-case counting | activation-pattern counting, neuron transitions and arc length along a 1-D trajectory, for all weights | convex-polytope partition of input space |
| [@serraBoundingCountingLinear2018, Thm. 1] | worst-case counting | dimension of the image of a region, decreasing monotonically through the layers; MILP encoding for exact counts | activation-pattern cells in H-representation |
| [@haninComplexityLinearRegions2019, Thm. 3] | average-case geometry | co-area formula applied to the level sets of the pre-activations; boundary volume and distance to the boundary | discontinuity set of the gradient, $B_N$ |
| [@haninDeepReLUNetworks2019, Thm. 5] | average-case geometry | bent hyperplanes as the boundary set; expectation over weights and biases | activation regions versus linear regions |
| [@goujonNumberRegionsPiecewise2023, Cor. 1] | worst-case counting | arrangement of convex partitions, replacing an arrangement of hyperplanes; knot density along a path | convex linear regions of any CPWL network |
| [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2] | depth lower bound | clear denominators to integer weights, then obstruct with the normalised volume of a lattice polytope modulo a prime | support functions, sum-union closure of polytopes |
| [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 3.5] | convex decomposition | balanced weight functions; the decomposition polyhedron as the intersection of two shifted convexity cones | the convex and concave parts $g,h$ with $f=g-h$ |
| [@gagneuxHowImproveExpressivity2025, Thm. 2.3] | convexity of the parametrisation | path-lifting and path-activations; convexity checked only across neighbouring activation regions | frontiers between activation regions, isolated neurons |
| [@zhangTropicalGeometryDeep, Thm. 5.2] | tropical | layer-wise recursion $A=A_+ - A_-$ producing tropical rational functions | tropical hypersurfaces, Newton polytopes, zonotopes |
| [@alfarraDecisionBoundariesNeural2023, Thm. 2] | tropical | dual subdivision of the tie polynomial as the convex hull of two zonotopes; row-wise pruning surrogate | decision-boundary polytope; adversarial perturbation of the first layer |
| [@balestrieroSplineTheoryDeep2018, Thm. 1] | spline | max-affine spline operators; every layer acts as a signal-dependent affine map | spline partition; matched-filter templates of the network |
| [@humayunSplineCamExactVisualization2024, Thm. 1] | exact geometry | region subdivision with back-projection of deeper hyperplanes onto each region, on a 2-D slice | exact input-space partition and decision boundary |
| [@berzinsPolyhedralComplexExtraction2023, §4] | exact geometry | edge subdivision of the 1-skeleton, with 2-faces recovered by perturbation of sign-vectors | polyhedral complex, as GPU tensor rows |
| [@vincentReachablePolyhedralMarching2021, §IV] | exact geometry, verification | marching from cell to cell by flipping type-1/type-2 neurons; constraints propagated to the input layer | reachable polyhedra and explicit PWA representation |
| [@masdenAlgorithmicDeterminationCombinatorial2022, Thm. 20] | combinatorics | injective sign sequences in $\{-1,0,1\}^N$; duality with a pure cubical complex | full face poset of the canonical polyhedral complex |
| [@carlssonPREIMAGERECTIFIERNETWORK2017, §1] | inversion | reciprocal nullspace basis of the layer weights, so that pre-activations become coordinates | preimage of a target layer activity, a piecewise-linear manifold |
| [@liuReLUNeuralNetworks2023, Eq. 13] | topology | binary activation vectors; pairwise Hamming matrix into a Vietoris–Rips filtration | dual graph of the polyhedral decomposition |
| [@liuIntegratingGeometriesReLU2023a, §2.2] | toolbox | traversal-and-pruning enumeration; redundancy LPs for active bits; Chebyshev centre LP | minimal H-representation, neighbour graph, inscribed radius |
| [@sattelbergLocallyLinearAttributes2023, §2.4] | statistics of the complex | sampling region affine maps and treating them as points of a "dual" space | region affine maps, clustered and mapped between networks |
| [@dhayalkarDiscreteFunctionalGeometry2025, Def. 3, §5.1] | region-adjacency graph | activation patterns sampled on a grid become nodes, joined exactly when they differ in one bit; region-volume entropy, degree and diameter are then read off the graph | the ReLU transition graph, its entropy and a pruning-based compression of the function |
| [@keupOrigamiDimensionsHow2022, §3] | folding | folding the data manifold into unoccupied dimensions, with the fold edge exposed to a flat readout | folded data manifold; tuning-curve bimodality as observable |
| [@lewandowskiSpaceFoldsReLU2024, Eq. 4] | folding | mapping a straight input segment into Hamming activation space and comparing range with path length | space folding measure $\chi \in [0,1]$ |
| [@lewandowskiExploitingSpaceFolding2026, Def. 3] | folding | input equivalence classes by activation pattern, so the measure survives non-ReLU monotonic activations | folding pseudo-metric $d_\chi$, global statistic $\Phi_N$ |
| [@phuongFunctionalVsParametric2019, Thm. 1] | symmetry | fold-set, canonical representation and dependency graph of the piecewise-linear surface | the function's own fold-set, read off the end-to-end map |
| [@grigsbyHiddenSymmetriesReLU, Thm. 4.1] | symmetry | transverse pairwise intersection of bent hyperplanes plus the Linear Regions Assumption | bent hyperplane arrangement; functional dimension of a parameter |
| [@bronsteinGeometricDeepLearning2017, §V] | symmetry, representations | spectral, spectrum-free and charting constructions of convolution on graphs and manifolds | Laplacian eigenbasis, local patch operators |
| [@bronsteinGeometricDeepLearning, §3.5] | symmetry, representations | three geometric priors (symmetry, deformation stability, scale separation) instantiated as a blueprint | the five geometric domains, $f = A_J\circ B_J\circ P\circ\rho\circ B$ |
| [@villarFullyCovariantMachine2023, §5] | symmetry | passive versus active symmetry, stated as commuting diagrams; covariance by construction | encodings, unit systems and tensors of the problem |
| [@nielsenElementaryIntroductionInformation2020, §3.2] | information geometry | conjugate connections coupled to a metric; Bregman generators and Legendre–Fenchel duality | statistical manifold $(M,g,\nabla,\nabla^*)$ |
| [@combeExploringInformationGeometry2025, §9.5; §10.2] | information geometry | the Amari–Chentsov tensor as a multiplication making a statistical manifold pre-Frobenius | (pre-)Frobenius manifolds, paracomplex coordinates |
| [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 3] | latent geometry | pulling an *ambient* Riemannian metric back through the generator Jacobian; linear-extrapolation generator | latent metric $M(z)=J_g(z)^\top M_X(g(z)) J_g(z)$ |
| [@arvanitidisPullingBackInformation2022, Prop. 3.2] | latent geometry | decoder viewed as a map into likelihood parameters; curve energy as a sum of KL terms | pullback Fisher–Rao metric on the latent space |
| [@pouplinIdentifyingLatentDistances2023, Def. 2.5] | latent geometry | expectation of the norm rather than of the metric tensor | Finsler norm, its indicatrix and Busemann–Hausdorff volume |
| [@meilaManifoldLearningWhat2023, §5.5] | manifold learning | taxonomy of one-shot and attraction–repulsion embeddings; push-forward metric to measure distortion | embedding $F$ and the distortion of its image |
| [@fuLatentTopologyInduction2022, §3] | representation topology | CRF-VAE over a frozen encoder, whose only learned parameters are state embeddings | latent state transition graph of a representation manifold |
| [@halversonMetricFlowsNeural2024, Eq. 2.8] | metric flow | gradient descent on a network-represented metric, governed by a metric neural tangent kernel | metric flow; Perelman's Ricci-flow functional as a special case |
| [@chazalIntroductionTopologicalData2021, §5.7] | topology | filtrations, persistence modules and their stability under interleaving | persistence diagrams, landscapes, Mapper graphs |
| [@berzinsGeometryInformedNeuralNetworks2024, Eq. 4] | implicit geometry | geometric requirements as differentiable violation functions, solved by an adaptive augmented Lagrangian with a diversity constraint | zero level set of a neural field; its polyhedral complex |
| [@pakLecturesDiscretePolyhedral, §8.1] | background | Morse functions, the variational principle and local-move connectivity, with complete proofs | convex polytopes, their face lattices and rigidity |

## Problems, data and training regime

| Paper | Problem | Data | Architecture or scale | Regime |
|---|---|---|---|---|
| [@arvanitidisGeometricallyEnrichedLatent2020, §5] | interpolate and sample from a generative model without leaving the data manifold | synthetic mixture of Gaussians on a circle; MNIST digits 0-1-2 and 0-1-2-3; CelebA | WGAN and VAE, latent dimension 2 to 5, PCA projection to 10 ambient dimensions | unsupervised |
| [@arvanitidisPullingBackInformation2022, §4] | compute latent shortest paths for non-Gaussian decoders | MNIST digit-1 subset; synthetic noisy circle; CMU motion capture Seq.69_06; MovieLens 25M (60 movies, 10 000 users) | four VAEs on MNIST; products of 26 von Mises–Fisher; Bernoulli VAE | unsupervised |
| [@balestrieroSplineTheoryDeep2018, §6] | interpret the classifier as a matched filter and regularise its templates | MNIST, CIFAR10, CIFAR100, SVHN | largeCNN, smallCNN, ResNet4-4; the printed evidence covers CIFAR100 with largeCNN | supervised |
| [@berzinsGeometryInformedNeuralNetworks2024, §4] | generate diverse feasible shapes with no training data | Gray-Scott 2D reaction–diffusion; Plateau's problem; GE/GrabCAD jet-engine bracket | conditional neural fields with softplus/SIREN/WIRE; 9 shapes with $\dim(z)=2$ for the bracket | data-free, constraint-driven |
| [@chazalIntroductionTopologicalData2021, §7] | topological inference and classification from shape | 14 configurations of maltose-binding protein (370 residues); smartphone accelerometer, 3 walkers, 100 windows of 200 samples each (300 series) | Rips complex up to dimension 2; alpha-complex persistence, 3 landscapes in dimensions 0 and 1, 6000 features, random forest | unsupervised features, supervised classification |
| [@dhayalkarDiscreteFunctionalGeometry2025, §5.1] | describe how a network's linear regions are wired to one another, and compress it by pruning low-degree regions | synthetic 2D uniform grid, 100 000 points in $[-1,1]^2$ | fully connected ReLU MLPs, depths 2–4, widths 4–128, 5 seeds per configuration | post hoc sampling of a randomly initialised network |
| [@fuLatentTopologyInduction2022, §4] | induce the topology of a frozen language model's representation space | 20News dev set (frozen BERT$_{\text{LAST}}$ last-layer outputs) | frozen BERT-base-uncased, 768-dimensional states, $N=2000$ latent states, LSTM decoder | unsupervised (structured VAE probe) |
| [@gagneuxHowImproveExpressivity2025, §4] | learn a convex function without the ICNN constraint | random weight draws; one 2D CPWL target on $[-10,10]^2$ | widths 10k draws for $d=2$; $n=(4,8)$ for the regression; 20K validation points | supervised |
| [@grigsbyHiddenSymmetriesReLU, §6] | how often does a parameter determine its function uniquely | no data; Jacobian rank of unrolled evaluations at $m$ random points | depths 4 and 6, widths 5/10/15, 5000 parameter draws each | none (initialisation study) |
| [@halversonMetricFlowsNeural2024, §3] | learn an approximate Calabi–Yau metric and study how training moves it | Fermat Quintic and Fermat Cubic | finite network of 3 layers, 64 nodes, ReLU, 100k points; cosine-activation kernel network | physics-informed by the sigma loss |
| [@haninComplexityLinearRegions2019, §3] | count regions and boundary volume of a trained network | MNIST, input dimension 784 | fully connected, He-normal weights, biases of variance $10^{-6}$, Adam at $10^{-3}$ | supervised |
| [@haninDeepReLUNetworks2019, §4] | push the region count up by memorisation | MNIST; random 2D points with binary labels | depth-3 networks such as $[20;20;20]$ and width 32 (96 neurons) | supervised, with label corruption |
| [@humayunSplineCamExactVisualization2024, Tab. 1] | exactly partition a slice of input space and locate the decision boundary | MNIST, Fashion-MNIST, CIFAR-10, tinyImageNet-200, MetFaces, Stanford Bunny | MLPs, convolutional networks, VGG11/VGG16 (138M parameters) | supervised, post hoc |
| [@keupOrigamiDimensionsHow2022, §4.1] | explain how a classifier becomes linearly separable | UCI poker hand, $10^6$ training samples | $10 \to 100 \to 100 \to 10$, about $1.2\times10^4$ parameters | supervised |
| [@lewandowskiExploitingSpaceFolding2026, §Sparsity] | measure how much a network folds its input space | MNIST; CIFAR100 | MLPs with varying width and depth, down to 60 neurons in total; $2\times256$ ReLU MLP | supervised |
| [@lewandowskiSpaceFoldsReLU2024, §6.1] | the same measure for ReLU, with depth and width sweeps | CantorNet (fractal benchmark); MNIST | $2\times30 \dots 10\times6$ at 60 neurons fixed; $2\times300$ and $3\times200$ | supervised |
| [@liuIntegratingGeometriesReLU2023a, §4] | integrate the several descriptions of a trained network into one toolbox | toy 2D and 3D quadratics; MNIST; CIFAR-10 | five hidden layers with 300–500 nodes (MNIST); 8 hidden layers of 400 nodes (CIFAR-10, 53.59% test) | supervised, diagnostic |
| [@liuReLUNeuralNetworks2023, §3.2] | is the network's own decomposition fine enough to keep data topology | toy quadratics in $\R^2$ and $\R^3$; three ImageNet images through ResNet-50 | models $2\text{-}3\text{-}3\text{-}3$ and $3\text{-}10\text{-}10\text{-}10\text{-}10$; ResNet-50, input dimension 150528 | unsupervised feature (persistent homology) |
| [@masdenAlgorithmicDeterminationCombinatorial2022, §5.1] | recover and compare decision-boundary topology exactly | no data; 50 random networks per architecture | shallow $(k,n,1)$ and deep $(k,n,n,1)$ with $2\le k\le 4$, width at most 15 | none (initialisation study) |
| [@pouplinIdentifyingLatentDistances2023, §4] | which latent metric should be used when the decoder is stochastic | pinwheel-on-sphere and concentric-circles-on-sphere; font dataset ($D=256$); qPCR single cells ($D=48$); MNIST; FashionMNIST | GPLVM in Pyro, with and without stochastic active sets | unsupervised |
| [@raghuExpressivePowerDeep2017, §3] | measure expressivity for any weight setting, before and after training | MNIST, CIFAR-10 | fully connected ReLU and hard-tanh of varying depth and width; convolutional CIFAR-10 networks | supervised |
| [@sattelbergLocallyLinearAttributes2023, §3] | are the linear regions of different networks similar | MNIST; XOR; circle versus annulus | dense (96%), Conv1 (98.07%), Conv2 (98.04%), Inception (99.08%), ResNet (98.92%) | supervised, post hoc |
| [@serraBoundingCountingLinear2018, §6] | count the true number of regions of a trained network | MNIST, input restricted to $0\le x\le 1$ | ten networks per configuration; two hidden layers summing to 22 neurons plus a 10-unit output; 20 epochs | supervised |
| [@villarFullyCovariantMachine2023, §6] | does enforcing a passive symmetry recover a hidden constant | synthetic black-body radiation from Planck's law; springy double pendulum, $N=500$ initialisations | three 3-layer MLPs with 20 hidden units; three $O(3)$-equivariant models in invariant scalars | supervised regression |
| [@vincentReachablePolyhedralMarching2021, §VI] | certify safety of a network in a control loop | ACAS Xu (45 policy networks); damped pendulum | 5 inputs, 5 outputs, 300 neurons; 12 neurons composed 50 times into 600 neurons | verification, not training |
| [@alfarraDecisionBoundariesNeural2023, §5] | prune and attack by moving one polytope | SVHN, CIFAR-10, CIFAR-100, MNIST | AlexNet $(256,512,10)$ and VGG16 $(512,512,10)$ classifiers; toy networks of 100/200/300 hidden nodes | supervised |
| [@berzinsPolyhedralComplexExtraction2023, §5] | extract the complex exactly, fast, and optimise a shape through it | Stanford bunny as a neural implicit shape; random 4-layer networks | width 10/20/40, input dimension $D=1,\dots,10$, RTX 3090 | post hoc, plus a geometric loss |
| [@bronsteinGeometricDeepLearning2017, §IX] | survey of geometric deep learning and its applications | CORA citation network; Netflix matrix; 3D shapes; particle physics; medical imaging | all numbers relayed from the cited primary sources | survey |

Twelve papers report no experiments of their own and so appear only in the first and last tables:
[@pascanuNumberResponseRegions2013], [@NIPS2014_fa6f2a46], [@goujonNumberRegionsPiecewise2023],
[@averkovEXPRESSIVENESSRATIONALRELU2025], [@zhangTropicalGeometryDeep],
[@brandenburgDecompositionPolyhedraPiecewise2024], [@carlssonPREIMAGERECTIFIERNETWORK2017],
[@phuongFunctionalVsParametric2019], [@nielsenElementaryIntroductionInformation2020],
[@combeExploringInformationGeometry2025], [@meilaManifoldLearningWhat2023] and
[@pakLecturesDiscretePolyhedral]. Two of them say so in the paper:
[@averkovEXPRESSIVENESSRATIONALRELU2025] is purely theoretical, and
[@carlssonPREIMAGERECTIFIERNETWORK2017] is a three-page note that states its empirical evaluation
was still in progress when it was written [@carlssonPREIMAGERECTIFIERNETWORK2017, Abstract].
[@grigsbyHiddenSymmetriesReLU] and [@berzinsGeometryInformedNeuralNetworks2024] are the reverse
case: their tables above cover the parts with data, and the rest of each paper is theory.

## Bounds on the number of linear regions

This is the one axis on which the collection's statements are directly comparable, because the
quantity is the same in every row: the maximal or expected number of pieces into which a
piecewise-linear network can cut its input space. The classic face count for arrangements in general
position, $\sum_{k=0}^{d}\binom{N}{k}$, is due to Zaslavsky (1975), which is outside the collection
and is not summarised here; it is quoted by [@pascanuNumberResponseRegions2013, Prop. 1],
[@serraBoundingCountingLinear2018, §2] and [@goujonNumberRegionsPiecewise2023, Thm. 1].

| Paper | Setting | Bound | Kind |
|---|---|---|---|
| [@pascanuNumberResponseRegions2013, Thm. 1] | $k$ hidden layers of widths $n_1,\dots,n_k$, $n_0$ inputs | $\prod_{i=1}^{k-1}\lfloor n_i/n_0\rfloor\cdot\sum_{j=0}^{n_0}\binom{n_k}{j}$ | lower bound from an explicit construction |
| [@pascanuNumberResponseRegions2013, Thm. 2] | $k$ layers of width exactly $2n_0$ | $2^{(k-1)n_0}\sum_{j=0}^{n_0}\binom{2n_0}{j}$ | lower bound; for $n_0=2$ it gives 44 regions against at most 37 for a shallow model with $4n_0$ units |
| [@NIPS2014_fa6f2a46, Thm. 5] | $L$ hidden layers, $n_i\ge n_0$ rectifiers | $\prod_{i=1}^{L-1}\lfloor n_i/n_0\rfloor^{n_0}\cdot\sum_{j=0}^{n_L}\binom{n_L}{j}$ | lower bound, improving the previous $\Omega((n/n_0)^{L-1}n^{n_0})$ |
| [@NIPS2014_fa6f2a46, Cor. 6] | $L$ layers of width $n\ge n_0$ | $\Omega((n/n_0)^{(L-1)n_0}n^{n_0})$ | asymptotic lower bound |
| [@NIPS2014_fa6f2a46, Prop. 4] | any rectifier network with $N$ hidden units | $2^N$ | upper bound, one region per activation pattern |
| [@NIPS2014_fa6f2a46, Thm. 9] | $L$ maxout layers of width $n_0$, rank $k$ | $k^{L-1}k^{n_0}$ | lower bound, exponential in depth |
| [@NIPS2014_fa6f2a46, Prop. 8] | one maxout layer, $n$ inputs, $m$ outputs, rank $k$ | between $k^{\min\{n,m\}}$ and $\min\{\sum_{j=0}^{n}\binom{k^2m}{j},k^m\}$ | lower and upper bound |
| [@raghuExpressivePowerDeep2017, Thm. 1] | $n$ hidden layers of width $k$, inputs in $\R^m$, any weights | $O(k^{mn})$ for ReLU and $O((2k)^{mn})$ for hard tanh | upper bound over all weight settings, matching the hand-built lower bound |
| [@serraBoundingCountingLinear2018, Thm. 1] | $L$ layers of widths $n_l$, input dimension $n_0$ | $\sum_{(j_1,\dots,j_L)\in J}\prod_{l=1}^{L}\binom{n_l}{j_l}$, $j_l\le\min\{n_0,n_1-j_1,\dots,n_{l-1}-j_{l-1},n_l\}$ | upper bound; the constraint set is what produces the bottleneck effect |
| [@serraBoundingCountingLinear2018, Thm. 7] | $n_0=1$ and $n_l\ge 3$ rectifiers | $\prod_{l=1}^{L}(n_l+1)$ | exact maximum |
| [@serraBoundingCountingLinear2018, Thm. 10] | $L$ layers of $n_l$ rank-$k$ maxout units | $\prod_{l=1}^{L}\sum_{j=0}^{d_l}\binom{k(k-1)n_l/2}{j}$ with $d_l=\min\{n_0,\dots,n_l\}$ | first upper bound for multi-layer maxout |
| [@serraBoundingCountingLinear2018, Cor. 3] | $n_0\ge Ln$ | $R(n_0,n,\dots,n)<R(n_0,Ln)$, and the ratio tends to 0 with $L$ | depth is *worse* than width once the input dimension exceeds the unit count |
| [@zhangTropicalGeometryDeep, Thm. 6.3] | $L$ layers, integer weights, thresholds, $n_l\ge d$ | $\prod_{l=1}^{L-1}\sum_{i=0}^{d}\binom{n_l}{i}$, hence $O(n^{d(L-1)})$ | upper bound from the tropical picture, which the authors state does not improve on [@raghuExpressivePowerDeep2017] and is more restrictive |
| [@dhayalkarDiscreteFunctionalGeometry2025, Thm. 1; App. A.1] | $L$ hidden layers of width $n$, $d$-dimensional input | $\lvert V\rvert\le\left(\sum_{i=0}^{d}\binom{n}{i}\right)^{L}$ | upper bound on the node count of the transition graph, stated as the Zaslavsky arrangement count composed over layers; the proof then loosens it with an inequality that fails in the direction stated, and the numerical column compares the experiments against that invalid form rather than the bound (my reading, not the paper's) |
| [@goujonNumberRegionsPiecewise2023, Cor. 1] | $L$ layers, dimensions $(d_{\mathrm{in}},W,\dots,W,d_{\mathrm{out}})$, activation complexity $\kappa$ | $(\kappa^{\lfloor W/d^*\rfloor})^{Ld^*}\le\kappa_{\max}\le(\kappa W)^{Ld_{\mathrm{in}}}$, $d^*=\min(d_{\mathrm{in}},d_{\mathrm{out}})$ | lower and upper bound for any CPWL network |
| [@goujonNumberRegionsPiecewise2023, Cor. 3] | random layers, any 1-D path | $\E[\lambda]\le D_0^*(\lambda_0WL)$ with $D_0^*=\max(D_0,1)$ | expected knot density; on average depth buys only a linear factor |
| [@haninComplexityLinearRegions2019, Thm. 1] | any piecewise-linear net, random parameters, interval $I$ | $\E[\#\{\text{regions in }I\}]\approx\lvert I\rvert\cdot T\cdot\#\{\text{neurons}\}$ | average case; depth does not enter |
| [@haninDeepReLUNetworks2019, Thm. 5] | ReLU, random parameters, cube $C$, $\#\text{neurons}\ge n_{\mathrm{in}}$ | $\E[\#\text{regions}]/\mathrm{vol}(C)\le(T\#\text{neurons})^{n_{\mathrm{in}}}/n_{\mathrm{in}}!$ | average case; no depth dependence, and it holds throughout training |
| [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2] | exact representation of $\max\{0,x_1,\dots,x_n\}$ with $N$-ary-fraction weights | at least $\lceil\log_p(n+1)\rceil$ hidden layers for a prime $p\nmid N$ | depth lower bound; $\lceil\log_3(n+1)\rceil$ for decimal weights (Cor. 3) |
| [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 4] | the same, in terms of the base $N$ | depth at least $C\ln n/\ln\ln N$ | first non-constant depth lower bound for practically relevant weight precision |
| [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 6.3] | convex CPWL function with $k$ affine components | depth $\lceil\log_2(n+1)\rceil+\lceil\log_2 r\rceil+1$ and size $O(rs^{n+1})$, for any $r,s$ with $rs\ge k$ | a depth–size trade-off, not a region count |

Three readings of this table matter. First, the *kind* column separates existential from typical
statements: every "lower bound" row is realised by one construction, and
[@NIPS2014_fa6f2a46, §2.5] only argues that the maximal count is not a measure-zero accident, not
that training finds it. Second, the upper and lower bounds are not on the same quantity for
different architectures: [@serraBoundingCountingLinear2018, Cor. 3] and
[@NIPS2014_fa6f2a46, Cor. 6] disagree about depth precisely because the first compares $\max$ over
weights at fixed total width while the second compares architectures at fixed width per layer.
Third, one row counts a different thing: [@dhayalkarDiscreteFunctionalGeometry2025, Thm. 1] bounds the
node count of its graph, whose nodes are the regions observed on a sampled grid, so the empirical
$|V|$ is a sampled count and not the true region count.

## Results as reported on the papers' own benchmarks

!!! warning "These numbers are not comparable"
    Every row below is the best figure the authors report for their own setting, on their own data,
    with their own protocol and their own metrics. No two rows share a benchmark, a baseline, a
    model size or a metric definition, and several report no variance at all. Read the table as a
    statement about what each paper claims for itself, never as a ranking.

| Paper | Metric | Reported result | Setting |
|---|---|---|---|
| [@alfarraDecisionBoundariesNeural2023, §5] | area under the pruning–accuracy curve | better than the magnitude-based baselines by 3%, 3% and 2% on SVHN, CIFAR-10 and CIFAR-100 for AlexNet; more than 90% of classifier parameters removed without affecting accuracy | AlexNet, VGG16; baseline test accuracies 92%, 74%, 43% and 92%, 92%, 70% |
| [@alfarraDecisionBoundariesNeural2023, §6, Tab. 1] | robust accuracy over 1000 correctly classified MNIST samples | 44.8 baseline, 46.1 and 43.8 for $\lambda_2=1.0,2.0$ at $\lambda_1=0.1$; 12.6, 11.3, 11.0 at $\lambda_1=0.2$ | tropical adversarial attack on MNIST |
| [@gagneuxHowImproveExpressivity2025, Tab. 1] | mean square error on 20K validation points, mean $\pm$ s.d. over 10 initialisations | unconstrained $0.90\pm0.53$; ICNN $1.71\pm0.90$; convex-regularised $0.84\pm0.73$ | architecture $n=(4,8)$, 2D target on $[-10,10]^2$ |
| [@grigsbyHiddenSymmetriesReLU, Fig. 6] | fraction of networks attaining the maximum functional dimension | depth 4, widths 5/10/15: 25%, 48%, 66%; depth 6: 1%, 3%, 5% | He-initialised ReLU networks, 5000 draws each |
| [@halversonMetricFlowsNeural2024, §1, §3.2] | gain in validation or test loss over the initial metric | baseline finite network gains a factor 10–100; the kernel methods fail to gain more than about a factor 5 on the test loss | Fermat Quintic; 100k points for the finite net, ~1000–2500 for the kernels |
| [@haninComplexityLinearRegions2019, Fig. 6(a)] | distance from a random point to the nearest region boundary | stays between $0.4/\#\{\text{neurons}\}$ and $1.5/\#\{\text{neurons}\}$ throughout training | MNIST, 10 000 sampled points per evaluation |
| [@humayunSplineCamExactVisualization2024, Tab. 1] | average region volume and number of regions on one 2-D slice | MNIST: $1.134\times10^{-5}$ over 8814 regions for the convolutional network against $3.144\times10^{-4}$ over 318 for the MLP | 39 780 and 44 860 parameters respectively |
| [@keupOrigamiDimensionsHow2022, §3, §4.1] | recall on the toy problem; macro-averaged F1 on poker hand | 100% inner class and 96.8% outer class on the 2d-egg; mean intact macro-F1 0.97 | three neurons in one layer; $10\to100\to100\to10$ on UCI poker hand |
| [@lewandowskiSpaceFoldsReLU2024, §6.1] | fraction of paths that exhibit folding | 0.35 and 0.44 at $2\times30$ and $3\times20$; 0.97 and 0.99 at $2\times300$ and $3\times200$ | MNIST, 60 neurons fixed for the first pair |
| [@lewandowskiSpaceFoldsReLU2024, §6.1] | Pearson correlation between depth and aggregated median folding | 0.987 for the best seed, over 100 digit pairs | architectures $2\times30 \dots 10\times6$ at 60 neurons |
| [@lewandowskiExploitingSpaceFolding2026, §Sparsity] | sparsity $S_N$, the fraction of non-folding paths | in $(0.4,0.8)$ for 60-neuron networks and decreasing to 0 as the neuron count grows | MNIST, averaged over 4 architectures per neuron count |
| [@liuIntegratingGeometriesReLU2023a, Tab. 4, Tab. 6] | Hamming distance at fixed Euclidean distance; inscribed radius | at Euclidean distance 0.5: training 52.97 vs FGSM adversarial 75.27; inscribed radii 0.23 (train), 0.21 (test), 0.13 (adversarial) | MNIST, five hidden layers, 1000 samples per set |
| [@masdenAlgorithmicDeterminationCombinatorial2022, Tab. 2, §5.2] | Betti numbers of the compactified decision boundary | $(2,5,1)$: $\beta_0=1.06\pm0.034$, $\beta_{n_0-1}=1.06\pm0.059$; $(4,8,8,1)$: $\beta_0=1.04\pm0.03$, $\beta_{n_0-1}=1.88\pm0.30$ | 50 randomly initialised networks per architecture |
| [@sattelbergLocallyLinearAttributes2023, Tab. 1] | correct test labels out of 10 000 after replacing every region's affine map by one of $k$ cluster centres | Inception: 974 at $k=1$, 9660 at $k=10$ (original 9908); dense network: 8679 at $k=1$ (original 9603) | five networks on MNIST, 10 clusters per digit output |
| [@villarFullyCovariantMachine2023, §6] | test RMSE; angle between learned and true gravity | 6.35 (units covariant, no constant), 0.39 (with a learnable dimensional constant), 0.64 (plain MLP); 0.00016 radians | black-body regression; springy double pendulum |
| [@vincentReachablePolyhedralMarching2021, Tab. I, §VI-A] | wall-clock time; number of reachable regions | 0.01 s against 6.66 s for the face-lattice method on an unsafe ACAS Xu property; 2781 regions in 87 s | ACAS Xu N1,7; damped pendulum over 50 steps |
| [@berzinsPolyhedralComplexExtraction2023, §5.1.2, §5.4.1] | speed-up against region subdivision; parameters after pruning | on average 20 times faster than SplineCam on 2-D inputs; 2751 parameters reduced to 589 | 4-layer width-10 networks on an RTX 3090; the Stanford bunny implicit shape |
| [@fuLatentTopologyInduction2022, §5.1, Tab. 1] | number of aligned latent states and the share of word occurrences they cover | BERT-last: 804 POS-aligned states covering 53.23%, with 40.52% of states not aligned; BERT-zeroth: 673 states covering 65.31%, with 27.69% not aligned | 20News, $N=2000$ states, alignment threshold 90% of occurrences |
| [@arvanitidisPullingBackInformation2022, §4.4] | relative Frobenius error of the approximated metric against the closed form | $5.32\times10^{-4}\pm9.63\times10^{-4}$ for the Normal and $1.73\times10^{-5}\pm1.17\times10^{-5}$ for the Beta decoder | von Mises–Fisher latent space, perturbations of radius 0.1 |
| [@berzinsGeometryInformedNeuralNetworks2024, Tab. 3, Tab. 4] | surface-strain objective; mean diversity | strain 7422 in the base model against 28 115 without the eikonal loss and 22 708 without connectedness; mean diversity 0.21 against 0.01 without the diversity loss | jet-engine bracket; 9 shapes with $\dim(z)=2$ |
| [@bronsteinGeometricDeepLearning2017, §IX] | classification accuracy on CORA | 81.6% for the spectral CNN against 75.7% previous best; 81.7% with MoNet | 2708 papers, 5429 citations, 1433-dimensional bag-of-words features, relayed from the cited works |
| [@dhayalkarDiscreteFunctionalGeometry2025, Tab. 1, §5.3.1] | mean number of transition-graph nodes; number of connected components | $\lvert V\rvert$ rises from 15.5 at $(L{=}2,n{=}4)$ to 7062.5 at $(L{=}4,n{=}128)$, all far below the printed bounds; 1 component for every configuration except $(L{=}4,n{=}128)$, which averages 4.40 components (std 2.06, CI95 1.80) | fully connected ReLU MLPs, depths 2–4, widths 4–128, 100 000 uniform points in $[-1,1]^2$, 5 seeds |

Three patterns are worth naming. Variance is mostly absent: [@gagneuxHowImproveExpressivity2025,
Tab. 1] and [@masdenAlgorithmicDeterminationCombinatorial2022, Tab. 2] report standard deviations,
[@lewandowskiSpaceFoldsReLU2024, §6.1] reports one seed, and
[@alfarraDecisionBoundariesNeural2023, §5] and [@humayunSplineCamExactVisualization2024, Tab. 1]
report no error bars at all. Baselines differ by kind: [@vincentReachablePolyhedralMarching2021,
Tab. I] compares against competing exact verifiers, [@berzinsPolyhedralComplexExtraction2023,
§5.1.2] against one competing exact extractor, and [@villarFullyCovariantMachine2023, §6] against a
single unconstrained MLP. And several headline numbers are *self-defined* metrics with no external
counterpart — folding, sparsity, diversity, surface strain — so they can be compared across rows of
one paper but not across papers.

## Assumptions and guarantees

| Paper | Assumptions | Guarantee |
|---|---|---|
| [@pascanuNumberResponseRegions2013, §6] | rectifier MLPs with linear output units; the last layer in general position | the constructed region count is a lower bound, and the authors state it is only an upper bound on flexibility because the pieces are correlated |
| [@NIPS2014_fa6f2a46, Thm. 5, §2.5] | specific weight constructions; widths at least the input dimension | lower bounds on the maximal number of regions, stable under small parameter perturbations because the region count is attained on an open set |
| [@raghuExpressivePowerDeep2017, Thm. 1, Thm. 3] | none beyond a feedforward architecture for the pattern bound; a non-straight trajectory for the length bound | an upper bound valid for every weight setting, which therefore meets the hand-built lower bounds and closes the worst-case picture |
| [@serraBoundingCountingLinear2018, Thm. 1, Thm. 11] | bounded polyhedral input domain and a sufficiently large big-$M$ for the counting formulation | an upper bound on the maximal region count and an exact count for a given network; the bottleneck effect is proved for the bound, not for the true maximum |
| [@haninComplexityLinearRegions2019, §2.1, §5] | random weights and biases with a density; bounded expected gradient norm | an identity for the expected boundary volume and a lower bound on the expected distance to a boundary, independent of depth |
| [@haninDeepReLUNetworks2019, Thm. 5] | weights with a density, conditional bias densities, bounded pre-activation moments; no tied weights | an upper bound on the expected activation-region density that is independent of depth and survives training, since the conditions are distributional |
| [@goujonNumberRegionsPiecewise2023, Thm. 2, Thm. 3] | independent, identically distributed layer parameters for the stochastic part; pointwise activations for the lower bound | matching upper and lower bounds on the maximal convex-region count, and an expected knot density linear in width and depth |
| [@zhangTropicalGeometryDeep, Thm. 5.2, Thm. 5.4] | integer weights and threshold activations | exact equivalence between ReLU networks and tropical rational maps, and between three families of piecewise-linear functions; no coverage of tanh or sigmoid |
| [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2] | weights restricted to $N$-ary fractions; exact representation of $\max\{0,x_1,\dots,x_n\}$ | a depth lower bound, with no control of width or of weight magnitude, and no statement about general real weights |
| [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 3.13] | the underlying polyhedral complex $P$ is fixed in advance | minimal decompositions are vertices of the decomposition polyhedron, so a finite search exists in principle; minimality is only relative to $P$ and no algorithm or implementation is provided |
| [@masdenAlgorithmicDeterminationCombinatorial2022, Lemma 12, Thm. 20] | the network is generic and supertransversal, and the first layer has at least $n_0$ units | the sign-sequence map is injective and the face poset is recovered exactly from vertex data; the excluded parameter set has measure zero |
| [@phuongFunctionalVsParametric2019, Thm. 1] | non-increasing widths $d_0\ge d_1\ge\dots\ge d_{L-1}\ge 2$ and one output; $h$ general and transparent | for a specially constructed $h$, any general network computing the same function is related to it by a permutation and a positive scaling of units — an existential statement about $h$, not about trained networks |
| [@grigsbyHiddenSymmetriesReLU, Thm. 4.1] | every hidden layer at least as wide as the input | a positive-measure set of parameters has no hidden symmetries; the complementary non-widening case is [@phuongFunctionalVsParametric2019], and the mixed case remains open |
| [@pouplinIdentifyingLatentDistances2023, Prop. 3.1, Prop. 3.3] | the decoder is a smooth stochastic immersion, which excludes VAE decoders | the Finsler norm is sandwiched between two Riemannian norms and differs from the expected-metric norm by $O(1/D)$ in the data dimension |
| [@chazalIntroductionTopologicalData2021, Thm. 7, Thm. 9] | $q$-tame filtrations; compact metric spaces for the Gromov–Hausdorff statement | persistence diagrams are stable in bottleneck distance under interleaving and under $2d_{GH}$; Mapper, by contrast, is explicitly unstable to its resolution and gain parameters |
| [@vincentReachablePolyhedralMarching2021, §V] | the input set is a polyhedron and the network is ReLU | exact forward and backward reachable sets, with an anytime property; the memory advantage is claimed relative to layer-by-layer methods, not in absolute terms |
| [@nielsenElementaryIntroductionInformation2020, Thm. 8] | a parametric family, invariance required only up to a scale constant | the Fisher information metric is the unique invariant metric under Markov embeddings; proofs are omitted throughout, so the statements are pointers into the cited sources |
| [@combeExploringInformationGeometry2025, Thm. 9.5] | a pre-Frobenius manifold with a metric-compatible multiplication | flatness of the pencil of structure connections characterises Frobenius manifolds; the statistical case is exhibited with its ingredients but potentiality is not verified for a concrete family |
| [@villarFullyCovariantMachine2023, §5, §11] | all relevant objects, including unknown constants, must be present among the features | no theorem: the claim that enforcing a passive symmetry improves learning is stated as a conjecture, and the demonstrations are two toy regressions |
| [@halversonMetricFlowsNeural2024, §2.3, §4] | infinite width, a local kernel, no component mixing, and a specific loss | under those assumptions the flow reproduces Perelman's gradient-flow form of Ricci flow; the assumptions are called ad hoc and strong, so Ricci flow is a corner of the space of network metric flows |
| [@carlssonPREIMAGERECTIFIERNETWORK2017, §1] | the layer's weight matrix has full rank | the complete preimage of a target activity is computable from the weights alone, as a cone generated by the reciprocal basis; the multi-layer claim is stated without an account of how pieces compose |
| [@humayunSplineCamExactVisualization2024, §2.2] | the network is built from continuous piecewise-linear pieces and the domain is a 2-D polytope | exactness up to machine precision on that slice; nothing is claimed for the full input space |
| [@liuReLUNeuralNetworks2023, §2.3] | none beyond a feedforward ReLU network | the Hamming distance is an approximation of the dual-graph distance, illustrated rather than bounded |
| [@lewandowskiExploitingSpaceFolding2026, Lemma 1, Prop. 2] | continuous monotonic activations; straight input segments | the folding measure is bounded in $[0,1]$, its zero set is direction-independent, and the symmetrised version is a pseudo-metric; the asymptotic conjectures are labelled as conjectures |
| [@pakLecturesDiscretePolyhedral, §8.1, §26.1] | convex polytopes, simplicial surfaces and plane triangulations as the theorems require | complete proofs of the polytope combinatorics and rigidity results the region-counting arguments cite loosely; hyperplane arrangements are outside the book's scope |
| [@dhayalkarDiscreteFunctionalGeometry2025, Thm. 2, Thm. 3, Thm. 4; §7] | fully connected ReLU MLPs with non-degenerate hyperplanes, sampled on a synthetic low-dimensional grid; a uniform region-volume distribution for the entropy estimate | connectivity of the transition graph when the hyperplanes are non-degenerate, and claimed bounds on the graph diameter versus the network's VC-dimension and on the size of the pruned graph. The connectivity statement is the only one the experiment exercises, and even it fails at the largest configuration; the diameter bound is a proof sketch and the compression theorem's free parameters are never instantiated, so the summary reads neither as established (my reading, not the paper's) |

The one thing this table makes visible is that the collection's strongest guarantees are attached to
the *weakest* claims. The statements that hold for every weight setting
([@raghuExpressivePowerDeep2017, Thm. 1]) or every trained network
([@haninDeepReLUNetworks2019, Thm. 5]) are upper bounds and non-comparability; the statements about
what a network *can* do carry explicit construction, and the statements about what is
*interpretable* mostly rest on assumptions — general position, fixed complexes, smooth decoders,
infinite width — that the same papers are candid about.
