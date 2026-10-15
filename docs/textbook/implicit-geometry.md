---
chapter: implicit-geometry
title: Implicit and geometry-informed representations
outline_hash: f5e764161a21
built_from:
  berzinsGeometryInformedNeuralNetworks2024: a6bc989abbb9
  berzinsPolyhedralComplexExtraction2023: 208bc568db8f
  chazalIntroductionTopologicalData2021: cef80f43dbc4
  fuLatentTopologyInduction2022: fc105aa1f4de
  humayunSplineCamExactVisualization2024: 9589eff7767b
---

# Implicit and geometry-informed representations

!!! abstract "In this chapter"
    - An *implicit neural representation* stores a shape not as a mesh but as the level set of a scalar field $f_\theta$; the network *is* the geometry, and the surface is the set where the field vanishes [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 4].
    - When $f_\theta$ is a ReLU network the zero level set is piecewise linear — a sub-complex of the network's own polyhedral complex — and can be extracted exactly [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4].
    - Geometry can enter as the *objective*: differentiable constraints and losses on the level set train a data-free shape generator, with or without a latent variable [@berzinsGeometryInformedNeuralNetworks2024, §3, p. 4; @berzinsPolyhedralComplexExtraction2023, §5.3, p. 8].
    - The probabilistic counterpart is the latent-variable model: a structured variational autoencoder induces the *topology* of a representation manifold as a network of latent states [@fuLatentTopologyInduction2022, §3, p. 3].
    - Prerequisites: [The region complex of a ReLU network](relu-network-geometry.md), [Deep networks as splines](spline-networks.md).

The preceding chapters studied a network from the outside: its input space is cut into linear regions, and along any direction the function is a spline. This chapter turns the picture around and asks a network to *be* a shape. An implicit neural representation (INR) stores a geometry as the level set of a scalar field $f_\theta:X\to\R$ produced by a network, so the object represented is the field's zero set rather than an explicit list of vertices [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 4]. The idea is old in computer graphics but it lands squarely in this textbook, because for a ReLU field the zero set is a piecewise-linear surface carried by exactly the polyhedral complex the earlier chapters taught us to extract [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4].

Three questions organise the chapter. First, what *is* the level set of a network, and what does a field buy over a mesh? Second, how does one train a field against a geometric objective rather than against data? Third, when is an implicit representation exact and when does it merely approximate? The answers come from two complementary lines in the collection: geometry-informed neural networks, which train smooth fields from design requirements alone [@berzinsGeometryInformedNeuralNetworks2024, §3, p. 4], and the exact extraction of the polyhedral complex of a ReLU field, on which one can put a loss directly [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8]. A third line, latent-variable models, replaces the shape by a latent space whose geometry is the object of interest, and is the probabilistic cousin of the same idea [@fuLatentTopologyInduction2022, §3, p. 3].

I keep the shared [notation](../notation.md): $f_\theta$ is the scalar field (a network with parameters $\theta$), $x\in\R^{n_0}$ an input, $\sigma$ the ReLU $\sigma(t)=\max(0,t)$, $W^{(l)}$ and $b^{(l)}$ the weights and biases of layer $l$, and $a^{(l)}$ the activations with $a^{(0)}=x$. Where a paper writes its own symbols — the folded hyperplane $H_i^{(l)}$ of [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4], the sign-vector $\sigma_i(x)$ of the same, the shape $\Omega$ and boundary $\partial\Omega$ of [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 4] — I translate into this notation and say so.

## Shape as the level set of a field

Let $f:X\to\R$ be a continuous scalar function on a domain $X\subset\R^n$. Its sign implicitly defines a *shape* $\Omega=\{x\in X\mid f(x)\le 0\}$ together with its *boundary* $\partial\Omega=\{x\in X\mid f(x)=0\}$ [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 4]. The representation is called *implicit* because the set is never stored: it is recovered on demand by evaluating $f$ and reading its sign. The same picture underlies a *signed distance function* (SDF), the implicit continuous representation of a surface in which the output is the signed distance of an input from the boundary, so that the boundary is again the zero level set [@humayunSplineCamExactVisualization2024, §3.1, p. 5].

!!! note "Background"
    A *level set* of $f$ is $\{x:f(x)=c\}$; a *sub-level set* is $\{x:f(x)\le c\}$. The level set of a signed distance function obeys the eikonal equation $\lVert\nabla f\rVert=1$ away from the *medial axis*, the set of points with more than one nearest boundary point; this is why an SDF can be trained by penalising $(\lVert\nabla f\rVert-1)^2$ [@berzinsGeometryInformedNeuralNetworks2024, §A, p. 28; §2.2, p. 3]. Curvature and normals of the surface follow from the first and second derivatives of $f$ by the standard formulas of differential geometry.

The choice of a field over a mesh is deliberate, and the reasons are stated in the papers. Neural fields are significantly more memory-efficient than discrete representations while providing higher fidelity, continuity and access to automatic derivatives [@berzinsGeometryInformedNeuralNetworks2024, §2.2, p. 3]. Implicit neural shapes add *topological flexibility*: because the shape is a level set, its connectivity can change without any change of the network's architecture, which supports both reconstruction and generation, whereas a mesh must be retriangulated [@berzinsGeometryInformedNeuralNetworks2024, §2.2, p. 3]. In the visualisation literature an INR is simply an MLP trained to map signal coordinates to the intensity of a signal at those coordinates, and the low input dimension is what makes its geometry inspectable [@humayunSplineCamExactVisualization2024, §3, pp. 4–5].

The activation function, and not only the architecture, decides what kind of surface the field can represent. ReLU networks are limited to piecewise-linear surfaces and, combined with gradient descent, are biased toward low frequencies, while Fourier-feature encodings, sine activations and wavelet activations control that frequency bias [@berzinsGeometryInformedNeuralNetworks2024, §2.2, p. 3]. Thus a ReLU field and a smooth field (softplus, SIREN, WIRE) induce geometrically different classes of shapes, and the rest of the chapter shows that each is the right tool for a different task: exact extraction and combinatorial losses on the one hand, smooth curvature-driven design on the other.

## The polyhedral complex induced by a ReLU field

For a ReLU field the level set is governed by the network's own decomposition. Recall from [The region complex of a ReLU network](relu-network-geometry.md) that the $i$-th neuron of layer $l$ induces the *folded hyperplane*
$$
H_i^{(l)}=\bigl\{x\in\R^D:\ W_i^{(l)\top}x^{(l-1)}+b_i^{(l)}=0\bigr\},
$$
the zero level set of that neuron's pre-activation seen from the input [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4]. A finite set of folded hyperplanes forms a folded hyperplane arrangement and induces a polyhedral decomposition of the domain; on each region of that decomposition a folded hyperplane behaves like an ordinary affine hyperplane and does not fold [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4]. The sign-vector $\sigma_i(x)=\operatorname{sgn}\!\bigl(W_i^{(l)\top}x^{(l-1)}+b_i^{(l)}\bigr)$ records on which side of each folded hyperplane a point lies [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4].

The consequence for implicit representations is immediate and is the reason this material sits after the region complex. A ReLU network trained as the SDF of a 3-D shape has a zero level set that is a union of cells of the induced polyhedral complex, and the complex itself can be extracted exactly; [@berzinsPolyhedralComplexExtraction2023, Fig. 1, p. 1] shows precisely this for a ReLU network trained on the signed distance field of the Stanford bunny in $D=3$. Because the activation is continuous, a folded hyperplane stays continuous across its folds, and the whole complex is carried by the 1-skeleton together with the sign-vectors of its vertices; perturbation of the sign-vectors then reconstructs the full intersection-poset, including the regions [@berzinsPolyhedralComplexExtraction2023, §4.1, p. 5]. The extraction algorithm and its GPU implementation are the subject of [The region complex of a ReLU network](relu-network-geometry.md); here the point is that the zero set of a ReLU field is a *known, combinatorial* object rather than an implicit one, and can be handed to any downstream algorithm as a mesh.

The same exactness holds, on a slice, for an arbitrary continuous piecewise-linear (CPWL) field. SplineCam computes the exact input-space partition of a CPWL network layer by layer over a chosen 2-polytope, and for a single-output network the decision boundary — the zero level set of the field — is the union of the back-projected output hyperplanes restricted to their regions, $\bigcup_{\omega\in\Omega}\{\operatorname{proj}_\omega(h^L_1)\cap\omega\}$ [@humayunSplineCamExactVisualization2024, Thm. 1, §2.2, p. 3]. So on a 2-D slice the surface of a CPWL implicit shape is computed exactly up to machine precision, with no sampling [@humayunSplineCamExactVisualization2024, §1, p. 2]. The exact partition is what makes an implicit field inspectable: on two width-256 depth-6 MLPs trained to fit an easy and a convoluted SDF of the same MetFaces image, the harder task develops the finer partition and the higher region density [@humayunSplineCamExactVisualization2024, §3.1, p. 5].

## Training a field against a geometric objective

An implicit shape that only fits data is a regression model; the interesting move in this collection is to train the field when there is no data, using the geometry itself as the target [@berzinsGeometryInformedNeuralNetworks2024, §4, p. 6]. Two formulations appear, and they differ in where the geometric knowledge is placed — in differentiable constraint violations on the field [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 5], or in a loss on the extracted complex [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8].

### Constraints as differentiable violations

A geometry-informed neural network (GINN) starts from a metric space of functions $(d,\mathcal{F})$ and lets the geometric design requirements define a feasible set. Writing the constraints $c_i(f)=0$, $i=1,\dots,m$, and an objective $o:\mathcal{F}\to\R$, one seeks $\min_{f\in K}o(f)$ where $K=\{f\in\mathcal{F}\mid c_i(f)=0\}$ [@berzinsGeometryInformedNeuralNetworks2024, §3, p. 4]. The bridge from geometry to training is that each constraint on the sets $(\Omega,\partial\Omega)$ is rewritten as a differentiable *violation* $c_i:\mathcal{F}\to\R$ on the field $f$, and this choice is explicitly not unique [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 5].

The main experiment uses four geometric constraints [@berzinsGeometryInformedNeuralNetworks2024, Tab. 1, p. 5]. *Design-region containment* $\Omega\subset E$ becomes the penalty $\int_{X\setminus E}[\min(0,f(x))]^2\,dx$, which is nonzero exactly where the shape escapes the allowed region. *Interface attachment* $\partial\Omega\supset I$ becomes $\int_I[f(x)]^2\,dx$, forcing the boundary to pass through the prescribed interface. A *prescribed normal* $n(x)=\bar n(x)$ on the interface becomes $\int_I\lVert \nabla f/\lVert\nabla f\rVert-\bar n\rVert^2\,dx$. *Topology* — for instance connectedness — is handled through persistent homology rather than an algebraic inequality. Because the constraints are differentiable, the constrained problem can be converted to a sequence of unconstrained ones by the *adaptive* augmented Lagrangian method, which keeps one penalty parameter $\mu_i$ per constraint and updates the multipliers $\lambda_i$ during training, so that loss terms with very different scales need no manual weighting [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 6, Eq. 4].

Two of the losses deserve separate mention because they turn discrete or high-order geometry into something differentiable. Connectedness is obtained by discretising the field onto a cubical complex and running persistent homology with $f$ as the filtration function, so that birth and death of each topological feature can be matched to a pair of critical points of $f$ whose values are then adjusted to reach the desired topology; the loss itself reads off the grid cell where a feature dies, which makes it differentiable without differentiating the homology library [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7; §E, pp. 28–29]. Smoothness is a surface integral $\int_{\partial\Omega\setminus I}e(\kappa_1,\kappa_2)\,dx$ of a curvature expression, with the surface-strain energy $E=\kappa_1^2+\kappa_2^2$ (and the variant $E_{\log}=\log(1+E)$) used to obtain visually smooth shapes [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7]. Curvature is a second-order operator, so this objective is expected to be ill-conditioned, which the authors take as motivation for second-order optimisers [@berzinsGeometryInformedNeuralNetworks2024, §4.3, p. 9].

### Generating a diverse solution set

Geometric problems are often under-determined, so GINNs generate a whole set of near-optimal solutions rather than one. The generator $G:z\mapsto f$ maps a latent variable $z\in Z$ to a field through a conditional neural field $f(x)=F(x;z)$, and the solution set is its image $S=\operatorname{Im}G(Z)$; each constraint and the objective are lifted to expectations over the latent distribution, $C_i(S)=\E_{z\sim p_Z}[c_i(G(z))]$ and $O(S)=\E[o(G(z))]$ [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 5]. A naive generative model of this kind collapses: the sampled shapes concentrate, so diversity must be added as an explicit constraint [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6].

Diversity is defined from a shape distance $d$ and an aggregation. GINNs use the *minimal aggregation measure* $\delta(S)=\bigl(\sum_i(\min_{j\neq i}d(f_i,f_j))^{1/2}\bigr)^2$, whose concavity promotes uniform coverage of the feasible set [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6, Eq. 2]. The distance itself is designed to measure shapes rather than fields: it integrates $f_j^2$ only over $\partial\Omega_i$ and $f_i^2$ only over $\partial\Omega_j$, because local changes in geometry can cause global changes in an SDF and the naive $L_2$ function distance would over-penalise them [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6, Eq. 3]. This $d$ is not a metric on functions, but the authors care about the geometries the functions represent; when the fields are SDFs it is closely related to the chamfer discrepancy [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6]. Adding the diversity constraint has a striking side effect: the latent space becomes *organised*, so traversing a 2-D latent space continuously morphs feasible shapes and the model generalises beyond the sampled batch [@berzinsGeometryInformedNeuralNetworks2024, §4.3, p. 9].

### A loss on the extracted complex

The second formulation puts the objective directly on the geometry that the network already carries. Starting from a ReLU network with two hidden layers of 50 neurons each in $D=2$, first trained as a neural implicit representation of a bunny, [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8] extracts the polyhedral complex in each iteration and computes the shape compactness $c=4\pi A/P^2$, normalised so that $c=1$ for the circle. Using $c$ as the loss deforms the bunny into a circle in 100 iterations of a standard Adam optimiser [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8]. This works because edge subdivision is fast enough to run inside the training loop, and because any geometric loss that depends on vertex positions — edge lengths, angles, areas, volumes, curvatures — is differentiable through the extracted complex [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8].

The two formulations sit at opposite ends of the smoothness scale, and the choice is forced by the objective. GINNs require models with a well-defined and non-vanishing second derivative with respect to the input, in order to compute surface normals and curvatures, so ReLU is replaced by softplus, SIREN or WIRE [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7]. Berzins' geometric loss instead exploits the piecewise-linear complex and needs no second derivative at all, but restricts the geometry being optimised to the network's own polyhedral mesh [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8]. A designer who needs a smooth surface with controlled curvature takes the first route; one who needs an exactly triangulated shape whose vertices are optimised combinatorially takes the second.

## When is an implicit representation exact?

"Exact" means two different things here, and the distinction is the practical heart of the chapter.

An implicit representation can be exact *about the network*. For a ReLU or CPWL field the zero level set is a subset of the network's polyhedral complex, and the complex is extracted exactly rather than sampled: [@berzinsPolyhedralComplexExtraction2023, §5.1.1, p. 7] verifies on a fixed four-layer width-10 network that the neurons whose sign-vector entry is zero evaluate to zero at each extracted vertex up to numerical error roughly seven orders of magnitude smaller than the extent of the domain. The authors are careful to call this a necessary but not sufficient condition, since it measures the pre-activation residual at the vertices rather than the geometric error in vertex positions [@berzinsPolyhedralComplexExtraction2023, §5.1.1, p. 7]. On a 2-D slice, SplineCam's partition and boundary are likewise exact up to machine precision, with no sampling or architecture simplification [@humayunSplineCamExactVisualization2024, §1, p. 2].

An implicit representation can only *approximate the target shape*, however. Training an INR as an SDF is a regression task in which a ground-truth distance field is fitted by the model to learn a boundary implicitly, so the recovered surface matches the target only as well as the regression does [@humayunSplineCamExactVisualization2024, §3.1, p. 5]. GINNs solve a constrained optimisation problem rather than a fitting problem, and their shapes satisfy the constraints only up to the residual violation that the augmented Lagrangian drives down, not identically [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 5; §4.1, p. 6]. Even their diversity measure is not a metric on functions, only a proxy for geometry, so "diverse" is a statement about the represented shapes and not a distance identity [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6].

Scalability bounds the exact side. SplineCam is defined only on 2-D domains, chosen for tractability and visualisation, and needs double precision with sparse storage for large convolutional layers [@humayunSplineCamExactVisualization2024, §2.2, p. 3; §B, p. 12]. Edge subdivision is impractical for $D>8$ even for small networks, because the complex grows exponentially in the input dimension, and operating on a bounded domain inserts artificial boundary cells that can dominate the counts [@berzinsPolyhedralComplexExtraction2023, §1, p. 2; §5.2]. And the evidence for the geometry-informed side is thin in the way the authors themselves concede: there are no established baselines, problems or metrics for data-free shape generation, and the engineering case study is a modified version of the original task with no comparison to established topology-optimisation methods [@berzinsGeometryInformedNeuralNetworks2024, §4, p. 6; §5, p. 10].

!!! question "Open question"
    No paper in the collection closes the loop between the two exactness notions. Edge subdivision extracts the exact complex of a *piecewise-linear* field [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8], while GINNs train a *smooth* field precisely because ReLU's second derivative vanishes and curvature cannot be evaluated [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7]. A method that trains a geometry-informed field for a smooth objective and then certifies the exact topology of the resulting zero set — rather than estimating it through persistent homology on a discretisation [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7] — is not available in the collection.

## Latent-variable representations and their geometry

The probabilistic counterpart of an implicit field is a latent-variable model, in which the geometry lives in a learned latent space rather than in the output surface. A variational autoencoder (VAE) posits a latent variable $z$ with a prior and a decoder $p_\theta(x\mid z)$, and trains both by maximising the evidence lower bound; the reparameterisation trick makes the bound differentiable, and the latent space it produces is the object whose geometry one wants to understand [@fuLatentTopologyInduction2022, §3, p. 3]. Two chapters develop that geometry in detail — [Manifold learning and intrinsic dimension](manifold-learning.md) for the manifold hypothesis and latent topology, and [Riemannian and Finsler geometry of latent spaces](latent-geometry.md) for metrics pulled back through a decoder — so here I use the VAE only as the structural bridge to implicit representations.

!!! note "Background"
    The evidence lower bound (ELBO) of a latent-variable model with approximate posterior $q(z\mid x)$ is $\mathcal{L}=\E_{q(z\mid x)}[\log p_\theta(x,z)]+H(q(z\mid x))$ up to sign conventions, with $H$ the entropy of $q$; the reparameterisation trick writes $z=\mu(x)+\varepsilon\odot\varsigma(x)$ with $\varepsilon$ standard normal so that a sample is a differentiable function of the parameters. A $\beta$-ELBO down-weights the entropy or KL term by a coefficient $\beta$, which is the standard knob against posterior collapse in a VAE [@fuLatentTopologyInduction2022, §4, p. 5].

The structured variant in the collection makes the latent variable a *state*, and the geometry it recovers is pure topology. [@fuLatentTopologyInduction2022, §3, p. 3] treats the contextualised word vectors $r_t\in\R^{768}$ of a frozen BERT-base-uncased encoder, all lying in one representation manifold $\mathcal{M}$, and hypothesises a spectrum of $N$ static latent states $s_1,\dots,s_N$ that act as anchors marking neighbours and connectivity in $\mathcal{M}$. Each vector is assigned a latent state index $z_t\in\{1,\dots,N\}$, and the state embeddings $s=[s_1,\dots,s_N]$ are the *only* learnable parameters of the inference model, the language model being frozen; the two relations are a state–word emission potential $\log\phi(x_t,z_t)=r_t^\top s_{z_t}$ and a state–state transition potential $\log\psi(z_{t-1},z_t)=s_{z_{t-1}}^\top s_{z_t}$, whose product is the posterior $q(z\mid x)$ of a linear-chain CRF [@fuLatentTopologyInduction2022, Eq. 1–2, §3, p. 3]. An LSTM decoder reconstructs the sentence on top, so that maximising the likelihood forces the states to be generative — able to predict sentences and their own successors — and the objective is a $\beta$-ELBO, $\mathcal{L}_{\text{ELBO}}=\E_{q_\psi(z\mid x)}[\log p_\theta(x,z)]-\beta H(q_\psi(z\mid x))$ [@fuLatentTopologyInduction2022, Eq. 5, §3, p. 3].

Empirically the states exist and carry meaning. With $N=2000$ states of dimension 768 and a 200-dimensional LSTM decoder on the 20News corpus, a state is said to *align* with a linguistic tag when at least 90% of the word occurrences it covers carry that tag [@fuLatentTopologyInduction2022, §4, p. 5; §5.1, p. 5]. Two controls discipline the reading: a random-embedding baseline (RANDEMB) also aligns 392 part-of-speech states covering 53.02% of occurrences, so alignment alone proves nothing, and the contextualised last layer (BERT$_{\text{LAST}}$) produces *more* aligned states than the static first layer (BERT$_{\text{ZERO}}$, 804 versus 673 part-of-speech-aligned) while covering *fewer* occurrences — the not-aligned share rising from 27.69% to 40.52% [@fuLatentTopologyInduction2022, §5.1, p. 5]. Human annotation of the non-aligned states finds that only about 11–14% of them are uninterpretable, the rest being lexical, syntactic or semantic [@fuLatentTopologyInduction2022, §5.1, p. 5]. The transition structure is the headline: frequent function states act as hubs, and a sentence is a *traversal* of states whose transition chain encodes the syntactic template while the emissions fill in the content [@fuLatentTopologyInduction2022, §6.2, pp. 8–9].

The parallel with implicit representations is the point. GINN's latent variable $z$ indexes a family of shapes, and the diversity constraint organises that latent space so that a path through it morphs one feasible shape continuously into another [@berzinsGeometryInformedNeuralNetworks2024, §4.3, p. 9]; Fu and Lapata's latent states index a family of word roles, and the transition network organises them so that a path through it spells out a sentence [@fuLatentTopologyInduction2022, §6.2, pp. 8–9]. In both, the useful geometry is not in the output alone but in the latent space, and in both the exactness is limited: GINN's latent organisation depends on the diversity setup and is only a qualitative observation [@berzinsGeometryInformedNeuralNetworks2024, §4.3, p. 9], while the CRF-VAE states are inferred, selected by the number of aligned tags during validation, and evaluated against human annotation without a reported agreement statistic [@fuLatentTopologyInduction2022, §5.1, p. 5]. The authors also state the scope: they recover *topological* structure, not the distance structure of the manifold, and local topology more than global [@fuLatentTopologyInduction2022, §7, p. 9].

## Summary

- An implicit neural representation stores a shape as the (sub-)level set of a scalar field $f_\theta$; the boundary is the zero level set, and it can be trained as a signed distance function [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 4; @humayunSplineCamExactVisualization2024, §3.1, p. 5].
- Fields are preferred over meshes for memory efficiency, continuity, differentiability and topological flexibility [@berzinsGeometryInformedNeuralNetworks2024, §2.2, p. 3].
- A ReLU field's zero level set is a union of cells of the folded-hyperplane arrangement, so it is a piecewise-linear surface that can be extracted exactly [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4; Fig. 1, p. 1].
- GINNs train a smooth field from constraints alone, turning each geometric requirement into a differentiable violation and solving the constrained problem by an adaptive augmented Lagrangian [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 5; §4.1, p. 7].
- Diversity as an explicit set constraint prevents mode collapse and organises the latent space, at the cost that the shape distance is a proxy for geometry, not a metric on functions [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6; §4.3, p. 9].
- A geometric loss can instead be put directly on the extracted complex; shape compactness is differentiable through edge subdivision and deforms a bunny into a circle in 100 iterations [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8].
- Exactness about the *network* (extraction to machine precision on a slice) is distinct from exactness about the *target* (regression or constrained optimisation, which only approximates) [@humayunSplineCamExactVisualization2024, §1, p. 2; §3.1, p. 5].
- A structured VAE is the probabilistic counterpart: it induces the topology of a representation manifold as a network of latent states whose transitions encode phrase structure [@fuLatentTopologyInduction2022, §3, p. 3; §6.2, pp. 8–9].

## Further reading

- [@berzinsGeometryInformedNeuralNetworks2024]: the GINN framework; read §3.1–§3.2 for the representation, constraints and diversity, §4.1 for the losses and the augmented Lagrangian, and §4.3 for the engineering case study.
- [@berzinsPolyhedralComplexExtraction2023]: exact extraction and geometric losses on the complex; read §3.4 for folded hyperplanes and sign-vectors, §5.3 for the geometric loss and §5.4 for pruning.
- [@humayunSplineCamExactVisualization2024]: exact partition and boundary of a CPWL field on a 2-D slice; read §3 for implicit neural representations and SDFs and §2.2 for the algorithm.
- [@fuLatentTopologyInduction2022]: inducing latent topology with a structured VAE; read §3 for the CRF-VAE, §5 for state alignment and §6 for transitions and traversals.
- [@chazalIntroductionTopologicalData2021]: the persistent-homology toolbox used by the connectedness loss of GINNs; read §4–§5 of the survey.

## Exercises

**Exercise 1** ★ (check). A GINN represents a shape by the sub-level set $\Omega=\{x:f_\theta(x)\le 0\}$. For the three main constraints of the paper — design-region containment, interface attachment and prescribed normal — write the differentiable violation $c_i(f)$ and say in one line why each integral is zero exactly when the geometric requirement holds.

??? success "Solution"
    Containment $\Omega\subset E$: penalise the field where it is negative outside $E$,
    $c=\int_{X\setminus E}[\min(0,f(x))]^2\,dx$, which vanishes iff $f\ge 0$ on
    $X\setminus E$ [@berzinsGeometryInformedNeuralNetworks2024, Tab. 1, p. 5]. Attachment
    $\partial\Omega\supset I$: $c=\int_I[f]^2\,dx$, zero iff $f=0$ on $I$
    [@berzinsGeometryInformedNeuralNetworks2024, Tab. 1, p. 5]. Prescribed normal:
    $c=\int_I\lVert\nabla f/\lVert\nabla f\rVert-\bar n\rVert^2\,dx$, zero iff the
    normalised gradient equals the required normal on $I$
    [@berzinsGeometryInformedNeuralNetworks2024, Tab. 1, p. 5]. The paper stresses that
    this translation from a constraint on $(\Omega,\partial\Omega)$ to one on $f$ is not
    unique [@berzinsGeometryInformedNeuralNetworks2024, §3.1, p. 5].

**Exercise 2** ★ (check). Why is the zero level set of a ReLU field a piecewise-linear surface, and which object from the earlier chapters carries it? Answer with the folded hyperplane.

??? success "Solution"
    A ReLU network is a composition of ReLU layers
    $x^{(l)}=\mathrm{ReLU}(W^{(l)}x^{(l-1)}+b^{(l)})$
    [@berzinsPolyhedralComplexExtraction2023, Eq. 4, §3.4, p. 4]. Each neuron switches on
    the folded hyperplane $H_i^{(l)}=\{x:W_i^{(l)\top}x^{(l-1)}+b_i^{(l)}=0\}$, and the
    arrangement of all such folded hyperplanes induces a polyhedral decomposition on which
    the field is affine per region [@berzinsPolyhedralComplexExtraction2023, §3.4, p. 4].
    The zero level set of the field is therefore a union of cells of that polyhedral
    complex — a piecewise-linear surface — which is exactly the object extracted by edge
    subdivision [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8].

**Exercise 3** ★★ (derive). GINN defines the shape distance $d(f_i,f_j)=\sqrt{\int_{\partial\Omega_i}f_j^2+\int_{\partial\Omega_j}f_i^2}$. Explain why the plain $L_2$ function distance $\sqrt{\int_X(f_i-f_j)^2}$ is unsuitable, using the example of a disk with its centre point removed.

??? success "Solution"
    $L_2$ compares the *fields*, not the *shapes*, and local geometric changes can cause
    global changes in an SDF. Removing the central point of a disk barely changes the shape
    but changes its SDF over the whole disk, so the $L_2$ distance is large even though the
    geometries are nearly identical [@berzinsGeometryInformedNeuralNetworks2024, §3.2,
    p. 6]. Integrating $f_j^2$ only over $\partial\Omega_i$ (and symmetrically) removes this
    globality, and when the fields are SDFs the expression equals
    $\int_{\partial\Omega_i}\min_{x'\in\partial\Omega_j}\lVert x-x'\rVert^2\,dx$ up to the
    counterpart term, so $d$ is closely related to the chamfer discrepancy
    [@berzinsGeometryInformedNeuralNetworks2024, §3.2, p. 6]. It is still not a metric on
    functions, only a proxy for geometry [@berzinsGeometryInformedNeuralNetworks2024,
    §3.2, p. 6].

**Exercise 4** ★★ (compare). [@berzinsPolyhedralComplexExtraction2023] optimises shape compactness $c=4\pi A/P^2$ on a ReLU field, while [@berzinsGeometryInformedNeuralNetworks2024] minimises a smoothness energy with softplus/SIREN/WIRE fields. State what each requires of the architecture and which geometry each can control.

??? success "Solution"
    The compactness loss needs only the extracted polyhedral complex: the area $A$ and
    perimeter $P$ come from the vertices of the complex, which edge subdivision supplies
    differentiably, so a plain ReLU network suffices and any vertex-position quantity (edge
    lengths, angles, areas, volumes, curvatures) can be optimised
    [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8]. The smoothness energy is a
    surface integral of curvature, which requires first and second derivatives of the field
    to be well defined and non-vanishing, so ReLU is excluded and softplus, SIREN or WIRE is
    used [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7]. In short: the
    piecewise-linear route controls exact combinatorial geometry; the smooth route controls
    curvature but represents the surface only approximately.

**Exercise 5** ★★ (derive). Show that if the field $f_j$ is the exact SDF of $\Omega_j$, then $\int_{\partial\Omega_i}f_j^2\,dx=\int_{\partial\Omega_i}\min_{x'\in\partial\Omega_j}\lVert x-x'\rVert^2\,dx$, and use this to explain the link to the chamfer discrepancy.

??? success "Solution"
    An SDF takes at $x$ the value $\pm d(x,\partial\Omega_j)$; on the boundary
    $\partial\Omega_i$, where the sign is immaterial after squaring,
    $f_j(x)^2=d(x,\partial\Omega_j)^2=\min_{x'\in\partial\Omega_j}\lVert x-x'\rVert^2$
    [@berzinsGeometryInformedNeuralNetworks2024, Eq. 21, p. 28; §3.2, p. 6]. Substituting
    gives the stated equality, and the sum of the two directed terms is a two-sided
    (chamfer) discrepancy between the two boundaries [@berzinsGeometryInformedNeuralNetworks2024,
    §3.2, p. 6]. This is why the paper says $d$ is "closely related" to the chamfer
    discrepancy rather than equal to a metric on functions.

**Exercise 6** ★★ (compare). "Exact" is used in two senses in this chapter: exact about the network and exact about the target. State precisely what is exact in [@humayunSplineCamExactVisualization2024] and in [@berzinsPolyhedralComplexExtraction2023], and what remains approximate in each.

??? success "Solution"
    Both papers are exact about the *network*: SplineCam computes the input-space partition
    and the decision boundary of a CPWL network up to machine precision, with no sampling,
    on a chosen 2-D slice [@humayunSplineCamExactVisualization2024, §1, p. 2], and edge
    subdivision reconstructs the exact polyhedral complex of a ReLU network
    [@berzinsPolyhedralComplexExtraction2023, §4.1, p. 5]. What remains approximate is the
    fit to the *target* shape: training an INR as an SDF is regression to a ground-truth
    distance field [@humayunSplineCamExactVisualization2024, §3.1, p. 5], and the vertex
    check of edge subdivision is only a necessary condition on the pre-activation residual,
    not a bound on geometric error [@berzinsPolyhedralComplexExtraction2023, §5.1.1, p. 7].
    So "exact" always refers to the network's own complex, never to agreement with a
    target geometry.

**Exercise 7** ★★★ (design / apply). You must (a) produce a shape that is exactly a polyhedral mesh with a differentiable loss on its vertices, and (b) produce a family of smooth, diverse, singly connected shapes that fit a design region, without any training data. Choose one method from the chapter for each, state the architecture and the optimisation, and name the principal risk.

??? success "Solution"
    (a) A ReLU implicit shape with a loss on the extracted complex: train a ReLU field to
    the shape, extract the complex with edge subdivision each iteration, and optimise a
    vertex-position objective such as compactness $c=4\pi A/P^2$
    [@berzinsPolyhedralComplexExtraction2023, §5.3, p. 8]. Risk: the method is impractical
    for input dimension $D>8$ because the complex grows exponentially, and the bounded
    domain adds artificial boundary cells [@berzinsPolyhedralComplexExtraction2023, §1,
    p. 2; §5.2]. (b) A GINN: a conditional neural field with softplus/SIREN/WIRE (not ReLU)
    for a non-vanishing second derivative, constraints for containment, attachment and
    connectedness, a diversity constraint on the solution set, all solved by adaptive
    augmented Lagrangian optimisation [@berzinsGeometryInformedNeuralNetworks2024, §4.1,
    p. 6; §4.1, p. 7]. Risk: the evidence base is thin — no baselines or established
    metrics, and the case study is a modified task [@berzinsGeometryInformedNeuralNetworks2024,
    §4, p. 6; §5, p. 10].

**Exercise 8** ★★★ (critique). The connectedness loss of [@berzinsGeometryInformedNeuralNetworks2024] estimates topology by running persistent homology on a discretisation of the field, while edge subdivision extracts topology exactly on the complex. Which of the two is "exact" for a smooth field, and what experiment would settle the question of how well the discrete estimate tracks the true topology?

??? success "Solution"
    For a smooth field neither approach gives the topology directly: the complex is
    polyhedral and defined for CPWA networks, whereas persistent homology is run on a
    cubical discretisation of the smooth field and the loss reads off the grid cell where a
    feature dies [@berzinsGeometryInformedNeuralNetworks2024, §4.1, p. 7; §E, pp. 28–29].
    So the GINN topology loss is a discrete *estimate*, and edge subdivision, though exact,
    applies to the piecewise-linear case [@berzinsPolyhedralComplexExtraction2023, §5.3,
    p. 8]. A settling experiment: train the same GINN and compare the Betti numbers of its
    level set under refinement of the discretisation grid, checking convergence; the paper
    does not report such a refinement study, only the ablation of the connectedness loss
    [@berzinsGeometryInformedNeuralNetworks2024, §C.2; Tab. 3].
