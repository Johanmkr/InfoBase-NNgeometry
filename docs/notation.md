---
title: Notation
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

# Notation

Conventions used throughout the textbook and the comparison pages. The paper summaries
keep each paper's own notation; the last column of every table maps the differences.

!!! info "How to read the tables"
    The *Symbol* column is the notation used in this textbook. The *In the papers* column
    lists the sources whose notation differs, with their form; "same in all" means no
    source in the collection deviates.

## Networks and activations

| Symbol | Meaning | In the papers |
|---|---|---|
| $f:\R^{n_0}\to\R^{n_L}$, $f_\theta$ | the network map, with parameters $\theta$ | $f$ [@zhangTropicalGeometryDeep; @alfarraDecisionBoundariesNeural2023]; $\Phi$ for the Lebesgue-sampling flow [@halversonMetricFlowsNeural2024] |
| $L$ | number of layers (depth) | $L$; $K$ [@serraBoundingCountingLinear2018] |
| $n_l$ | width of layer $l$, $l=0,\dots,L$ | $n_l$; $n_i$ [@NIPS2014_fa6f2a46]; $N_l$ [@masdenAlgorithmicDeterminationCombinatorial2022] |
| $W^{(l)}\in\R^{n_{l}\times n_{l-1}}$, $b^{(l)}\in\R^{n_l}$ | weights and biases of layer $l$ | $(W_l,b_l)$; $\theta^{(l)}$ [@grigsbyHiddenSymmetriesReLU] |
| $z^{(l)}$ | pre-activation of layer $l$, $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$ | $z^{(l)}$; $h^{(l)}$ [@lewandowskiSpaceFoldsReLU2024] |
| $a^{(l)}$ | activation of layer $l$, $a^{(l)}=\sigma(z^{(l)})$, with $a^{(0)}=x$ | $a^{(l)}$; $x^{(l)}$ |
| $\sigma$ | the ReLU activation, $\sigma(t)=\max(0,t)$ | $\sigma$ [@haninDeepReLUNetworks2019]; $\rho$ [@zhangTropicalGeometryDeep]; $\mathrm{ReLU}$ [@lewandowskiSpaceFoldsReLU2024] |
| $x\in\R^{n_0}$ | an input point | $x$; $u$ [@zhangTropicalGeometryDeep] |
| $\odot$ | entrywise (Hadamard) product | $\odot$ [@balestrieroSplineTheoryDeep2018] |
| $\mathbf{1}[\cdot]$ | indicator of a condition | $\mathbf{1}[\cdot]$; $\mathbb{I}[\cdot]$ |

!!! warning "Clash: $\sigma$, $N$, $G$ and $H$"
    $\sigma$ denotes the ReLU activation here. In [@balestrieroSplineTheoryDeep2018] $\sigma$
    is the activation function of a generic spline, and in the symmetry literature $\Sigma$
    is a permutation. For widths this textbook writes $n_l$; [@NIPS2014_fa6f2a46] uses $N$
    for the number of hidden units, and [@lewandowskiExploitingSpaceFolding2026] uses $N$ both
    for the network and for a neuron count. $G$ is the transition graph in the notation of
    [@dhayalkarDiscreteFunctionalGeometry2025] but a group in the symmetry chapter
    ([@bronsteinGeometricDeepLearning2017; @grigsbyHiddenSymmetriesReLU]); where the two could be
    confused the group is written $G_{\text{grp}}$. $H(G)$ is the entropy of
    [@dhayalkarDiscreteFunctionalGeometry2025], $H$ a hidden symmetry in
    [@grigsbyHiddenSymmetriesReLU] and homology in [@liuReLUNeuralNetworks2023].

## Linear regions and activation patterns

| Symbol | Meaning | In the papers |
|---|---|---|
| $\mathbf{s}(x)$ | activation pattern (sign vector) of $x$, one bit per hidden unit | $s(x)$ [@masdenAlgorithmicDeterminationCombinatorial2022]; binary vector $\tilde{x}$ [@sattelbergLocallyLinearAttributes2023]; $q(x)$ [@grigsbyHiddenSymmetriesReLU]; $\mathrm{sign}(\cdot)$ [@haninComplexityLinearRegions2019] |
| $\mathcal{R}$ | a linear region: a maximal connected set on which $f$ is affine | $R$, $Q$ [@serraBoundingCountingLinear2018]; "activation region" [@haninDeepReLUNetworks2019] |
| $\mathcal{N}(f)$ | number of linear regions of $f$ on its input domain | $\mathcal{N}(f)$ [@goujonNumberRegionsPiecewise2023; @serraBoundingCountingLinear2018]; $N$ [@pascanuNumberResponseRegions2013]; region count [@NIPS2014_fa6f2a46] |
| $\mathcal{P}(f)$ | the polyhedral complex of linear regions of $f$ | $P(f)$ [@liuIntegratingGeometriesReLU2023a]; canonical polyhedral complex [@brandenburgDecompositionPolyhedraPiecewise2024] |
| $H_i$ | a hyperplane bounding a region, $H_i=\{x:\langle a_i,x\rangle=b_i\}$ | $H_i$, $\mathcal{H}$ for the arrangement |
| $d(x)$ | distance from $x$ to the nearest region boundary | region distance [@haninComplexityLinearRegions2019] |
| $\phi^{(l)}$ | a bent hyperplane: the preimage, under the map to layer $l$, of a layer-$l$ hyperplane | $H^{(l)}$ [@masdenAlgorithmicDeterminationCombinatorial2022]; bent hyperplane [@grigsbyHiddenSymmetriesReLU] |
| $F_i$ | the fold-set of a ReLU unit: where the unit changes sign, folding space | fold-set $F$ [@lewandowskiSpaceFoldsReLU2024] |
| $d_H$ | Hamming distance on activation patterns | $d_H$ [@masdenAlgorithmicDeterminationCombinatorial2022; @sattelbergLocallyLinearAttributes2023] |
| $G=(V,E)$ | the ReLU transition graph of $f$: nodes are the regions, edges join regions that share a facet, i.e. patterns at Hamming distance one | $G$ [@dhayalkarDiscreteFunctionalGeometry2025]; dual graph [@liuReLUNeuralNetworks2023]; graph of single-bit flips [@liuIntegratingGeometriesReLU2023a] |
| $d_{\mathrm{avg}}$ | average degree of the nodes of $G$ | $d_{\mathrm{avg}}$ [@dhayalkarDiscreteFunctionalGeometry2025] |
| $H(G)$ | entropy of the region-volume distribution $p_v=\operatorname{Vol}(\mathcal{R}_v)/\sum_u\operatorname{Vol}(\mathcal{R}_u)$ | $H(G)$, $p_v$ [@dhayalkarDiscreteFunctionalGeometry2025] |
| $\operatorname{diam}(G)$ | length of the longest shortest path in $G$ | $\operatorname{diam}(G)$ [@dhayalkarDiscreteFunctionalGeometry2025] |
| $\operatorname{VC}(\mathcal{F})$ | VC-dimension of the function class $\mathcal{F}$ realised by the network | $\operatorname{VC}(\mathcal{F})$ [@dhayalkarDiscreteFunctionalGeometry2025] |

## Polyhedral and tropical geometry

| Symbol | Meaning | In the papers |
|---|---|---|
| $[n]$ | the set $\{1,\dots,n\}$ | $[n]$ |
| $\mathrm{conv}(S)$ | convex hull of $S$ | $\mathrm{conv}(\cdot)$ |
| $A+B$ | Minkowski sum of sets or polytopes | $+$ [@zhangTropicalGeometryDeep]; $\oplus$ in the tropical semiring |
| $Z=\sum_i[-v_i,v_i]$ | a zonotope generated by vectors $v_i$ | $Z$ [@zhangTropicalGeometryDeep; @alfarraDecisionBoundariesNeural2023] |
| $\oplus,\ \odot$ | tropical addition $a\oplus b=\max(a,b)$ and multiplication $a\odot b=a+b$ | $\oplus,\odot$ [@zhangTropicalGeometryDeep]; $\max$, $+$, $-\infty$ written out [@alfarraDecisionBoundariesNeural2023] |
| $p(x)=\bigoplus_i c_i\odot x^{\alpha_i}$ | tropical polynomial, i.e. a max-affine function | $p(x)$ [@zhangTropicalGeometryDeep] |
| $r=p\oslash q$ | tropical rational function, a difference of max-affine functions | $r$ [@alfarraDecisionBoundariesNeural2023]; $f-p$ [@zhangTropicalGeometryDeep] |
| $\mathrm{Newt}(p)$ | Newton polytope of $p$: the convex hull of its exponent vectors | $\mathrm{Newt}(p)$ [@zhangTropicalGeometryDeep]; $P_f$ [@alfarraDecisionBoundariesNeural2023] |
| $T(p)$ | tropical hypersurface of $p$: where the maximum is attained twice | $\mathrm{Trop}(p)$ [@zhangTropicalGeometryDeep] |
| $\mathrm{Vol}(P)$ | Euclidean volume; $d!\,\mathrm{Vol}(P)$ is the normalised volume of a lattice polytope | $\mathrm{vol}$ [@zhangTropicalGeometryDeep]; normalised volume [@brandenburgDecompositionPolyhedraPiecewise2024] |
| $H$- and $V$-representation | a polytope as $\{x:Ax\le b\}$ or as $\mathrm{conv}(V)$ | same in all [@pakLecturesDiscretePolyhedral] |

## Information and Riemannian geometry

| Symbol | Meaning | In the papers |
|---|---|---|
| $\mathcal{M}$, $\theta$ | a statistical manifold and its coordinates | $\mathcal{M}$, $\theta$ [@nielsenElementaryIntroductionInformation2020; @combeExploringInformationGeometry2025] |
| $p_\theta$ | a distribution indexed by $\theta$ | $p_\theta$; $p(x\mid\theta)$ |
| $g_{ij}(\theta)$ | Fisher information metric, $g_{ij}=\E[\partial_i\ell\,\partial_j\ell]$, $\ell=\log p_\theta$ | $g$, $\mathcal{I}$ [@nielsenElementaryIntroductionInformation2020]; $\mathbf{F}$ [@arvanitidisPullingBackInformation2022] |
| $\nabla^{(\alpha)}$ | the $\alpha$-connection on a statistical manifold; $\alpha=\pm1$ are the exponential and mixture connections | $\nabla^{(\alpha)}$ [@nielsenElementaryIntroductionInformation2020]; $\nabla^{(-1)},\nabla^{(1)}$ |
| $\Gamma_{ijk}$ | Amari–Chentsov cubic tensor, $\E[\partial_i\ell\,\partial_j\ell\,\partial_k\ell]$; determines the torsion of the $\alpha$-connections | $\Gamma$ [@nielsenElementaryIntroductionInformation2020]; $T$ [@combeExploringInformationGeometry2025] |
| $D_{\mathrm{KL}}(p\,\Vert \,q)$ | Kullback–Leibler divergence; its Hessian is the Fisher metric | $D_{\mathrm{KL}}$; $\mathrm{KL}$ |
| $G(z)$ | metric on the latent space pulled back through a decoder $\phi$: $G(z)=J_\phi(z)^\top G(\phi(z))\,J_\phi(z)$ | $G$ [@arvanitidisGeometricallyEnrichedLatent2020]; $g^E$ [@arvanitidisPullingBackInformation2022] |
| $J_\phi(z)$ | Jacobian of the decoder at $z$ | $J$ [@arvanitidisGeometricallyEnrichedLatent2020] |
| $\gamma:[0,1]\to\mathcal{M}$ | a path, usually a shortest (geodesic) one; $\ell(\gamma)$ its length | $\gamma$; $c$ [@arvanitidisPullingBackInformation2022] |
| $F(x,\dot x)$ | Finsler norm on the tangent space at $x$; $d_F$ its induced distance | $F$, $d_F$ [@pouplinIdentifyingLatentDistances2023] |
| $d(\cdot,\cdot)$ | geodesic (or Finsler) distance from a metric | $d$; $\rho$ |
| $\mathcal{L}$ | learning objective; $\mathcal{L}_{\text{geo}}$ a geometric regulariser | $\mathcal{L}$ [@arvanitidisGeometricallyEnrichedLatent2020]; metric-flow functional $\mathcal{F}$ [@halversonMetricFlowsNeural2024] |

## Topology

| Symbol | Meaning | In the papers |
|---|---|---|
| $\mathrm{VR}_t$ | Vietoris–Rips complex at scale $t$ | $\mathrm{Rips}_t$ [@chazalIntroductionTopologicalData2021] |
| $\mathrm{dgm}_k(f)$ | $k$-th persistence diagram of a filtration; $\mathrm{Bar}_k$ its barcode | $\mathrm{Dgm}_k$, $\mathrm{Bar}_k$ [@chazalIntroductionTopologicalData2021] |
| $\beta_k$ | the $k$-th Betti number, the rank of $H_k$ | $\beta_k$ [@liuReLUNeuralNetworks2023] |
| $d_B$ | bottleneck distance between persistence diagrams | $d_B$ [@chazalIntroductionTopologicalData2021] |
| $d_{\mu,m}(x)$ | distance to the measure $\mu$ with exponent $m$ | $d_{\mu,m}$ [@chazalIntroductionTopologicalData2021] |

## Symmetry

| Symbol | Meaning | In the papers |
|---|---|---|
| $G$ | a group acting on data or on parameter space | $G$; $H$ for a hidden symmetry [@grigsbyHiddenSymmetriesReLU] |
| $g\cdot x$ | the action of $g\in G$ on $x$ | $g\cdot x$; $\pi(g)x$ for a representation $\pi$ |
| $f(g\cdot x)=g'\cdot f(x)$ | equivariance of $f$ under $G$; $f(g\cdot x)=f(x)$ is invariance | same in all [@bronsteinGeometricDeepLearning2017; @villarFullyCovariantMachine2023] |
| $\Pi$ | a permutation of hidden units, a parameter-space symmetry of a ReLU network | $\pi$, $\Pi$ [@grigsbyHiddenSymmetriesReLU] |
| $(c_l)$ | positive per-layer scalings of weights and biases: $W^{(l)}\mapsto c_lW^{(l)}$, $b^{(l)}\mapsto c_lb^{(l)}$, next layer divided by $c_l$ | $\alpha$; $\lambda$ [@grigsbyHiddenSymmetriesReLU] |

## Data, training and evaluation

| Symbol | Meaning | In the papers |
|---|---|---|
| $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^m$ | training data | $\mathcal{D}$; $S$ [@raghuExpressivePowerDeep2017] |
| $m$ | number of samples | $m$; $N$ for sample count [@pouplinIdentifyingLatentDistances2023] |
| $\lambda$ | weight of a regularisation or diversity term | $\lambda$; $\beta$ [@arvanitidisPullingBackInformation2022] |
| $\mathrm{RMSE}$ | root-mean-square error, the regression metric used here | RMSE, MSE |
| $\mathrm{Ecc.}$ | eccentricity of a region, the ratio of its largest to smallest pairwise distance | Ecc. [@humayunSplineCamExactVisualization2024] |
