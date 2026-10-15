---
title: Notation
built_from:
  NIPS2014_fa6f2a46: 74ce28c3ceac
  alfarraDecisionBoundariesNeural2023: ea6d2e0f5af7
  arvanitidisGeometricallyEnrichedLatent2020: 2b5378b36bee
  arvanitidisPullingBackInformation2022: d5884ab455cc
  averkovEXPRESSIVENESSRATIONALRELU2025: 027dbe2ff0ec
  balestrieroSplineTheoryDeep2018: 5784cc65e030
  berzinsGeometryInformedNeuralNetworks2024: a6bc989abbb9
  berzinsPolyhedralComplexExtraction2023: 208bc568db8f
  brandenburgDecompositionPolyhedraPiecewise2024: 9b08305343a9
  bronsteinGeometricDeepLearning: b1bdc405eedd
  bronsteinGeometricDeepLearning2017: 303af41ca265
  carlssonPREIMAGERECTIFIERNETWORK2017: eb5f6b84901f
  chazalIntroductionTopologicalData2021: cef80f43dbc4
  combeExploringInformationGeometry2025: 415b5fcd4364
  fuLatentTopologyInduction2022: fc105aa1f4de
  gagneuxHowImproveExpressivity2025: 22e23b06014b
  goujonNumberRegionsPiecewise2023: 6a2c48d0f7af
  grigsbyHiddenSymmetriesReLU: 75ad4d2ba0ce
  halversonMetricFlowsNeural2024: 699fded4f825
  haninComplexityLinearRegions2019: 5631df3a9bd2
  haninDeepReLUNetworks2019: 3f3cf93f5eda
  humayunSplineCamExactVisualization2024: 9589eff7767b
  keupOrigamiDimensionsHow2022: 8e2b8597f3f1
  lewandowskiExploitingSpaceFolding2026: e8cb3171023a
  lewandowskiSpaceFoldsReLU2024: de2f7efdfad0
  liuIntegratingGeometriesReLU2023a: 3f2e37aa6990
  liuReLUNeuralNetworks2023: 7069ff27d8f7
  masdenAlgorithmicDeterminationCombinatorial2022: ea8fb21c3649
  meilaManifoldLearningWhat2023: 009a25e9883c
  nielsenElementaryIntroductionInformation2020: 84c18077962c
  pakLecturesDiscretePolyhedral: 676ac4afadc7
  pascanuNumberResponseRegions2013: 3fae3832ff4c
  phuongFunctionalVsParametric2019: c7324e1a5cb1
  pouplinIdentifyingLatentDistances2023: 002a4b56f697
  raghuExpressivePowerDeep2017: b62868fb88f1
  sattelbergLocallyLinearAttributes2023: 4e0708abcb09
  serraBoundingCountingLinear2018: e57c5f62b98c
  villarFullyCovariantMachine2023: 20a7c0430350
  vincentReachablePolyhedralMarching2021: c4988156c5b8
  zhangTropicalGeometryDeep: f99703ad8890
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

!!! warning "Clash: $\sigma$ and $N$"
    $\sigma$ denotes the ReLU activation here. In @balestrieroSplineTheoryDeep2018 $\sigma$
    is the activation function of a generic spline, and in the symmetry literature $\Sigma$
    is a permutation. For widths this textbook writes $n_l$; @NIPS2014_fa6f2a46 uses $N$
    for the number of hidden units and @lewandowskiExploitingSpaceFolding2026 uses $N$ both
    for the network and for a neuron count.

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
| $D_{\mathrm{KL}}(p\,\|\,q)$ | Kullback–Leibler divergence; its Hessian is the Fisher metric | $D_{\mathrm{KL}}$; $\mathrm{KL}$ |
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
