---
citekey: berzinsGeometryInformedNeuralNetworks2024
title: "Geometry-Informed Neural Networks"
authors: [Arturs Berzins, Andreas Radler, Eric Volkmann, Sebastian Sanokowski, Sepp Hochreiter, Johannes Brandstetter]
year: 2024
venue: arXiv preprint
doi: 10.48550/arXiv.2402.14009
arxiv: "2402.14009"
url: null
kind: method
source_type: pdf
source_sha256: 301710ad2e5f2f79ae2ff15240f4fe758f92564a0fa8160f9c91554f48154012
pages: 31
ingested: 2026-10-15
tags: [neural-fields, implicit-neural-shapes, topology-optimization, generative-design, constrained-optimization, data-free-learning]
concepts_introduced:
  - geometry-informed neural network
  - shape-generative neural field
  - diversity constraint
  - minimal aggregation measure
  - geometric shape distance
  - differentiable topology loss
  - surface-strain objective
  - generative physics-informed neural network
concepts_used:
  - neural field
  - implicit neural shape
  - signed distance function
  - eikonal equation
  - persistent homology
  - augmented Lagrangian method
  - topology optimization
  - conditional neural field
  - principal curvature
builds_on: []
datasets: [Gray-Scott 2D reaction-diffusion, Plateau's problem, jet engine bracket (GE/GrabCAD)]
---

# Geometry-Informed Neural Networks

**Berzins, Radler, Volkmann, Sanokowski, Hochreiter, Brandstetter (2024)**, *arXiv preprint*. [arXiv:2402.14009](https://arxiv.org/abs/2402.14009)

!!! abstract "TL;DR"
    Geometry-informed neural networks (GINNs) train a conditional neural field to generate shapes *without any data*, by turning geometric design requirements — containment in a design region, attachment to interfaces, prescribed normals, topology, smoothness — into differentiable constraint violations and solving a constrained optimization problem with an adaptive augmented Lagrangian method. A diversity constraint on the sampled solution set prevents mode-collapse, so a single trained model yields many diverse feasible shapes and an organized latent space. The framework is demonstrated on validation problems in 2D/3D and a realistic 3D engineering design task.

## Problem and motivation

Data-driven shape generation is limited by the scarcity of large shape datasets in computer graphics, design and engineering, while exactly these domains supply formal problem statements as objectives and constraints (§1, p. 2). Citing data-free successes for PDEs, molecular science and combinatorial optimization, the paper asks whether a *shape*-generative model can likewise be trained on objectives and constraints alone (§1, p. 2). The setting is a metric space $(d, \mathcal{F})$ of functions: geometric design requirements define a feasible set $K = \{f \in \mathcal{F} \mid c_i(f) = 0,\ i = 1..m\}$, an objective $o : \mathcal{F} \to \R$ ranks solutions, and one seeks $\min_{f \in K} o(f)$ (§3, p. 4). GINNs are positioned against physics-informed neural networks (PINNs): both represent the solution by a neural field, but PINNs minimize a strong-form PDE residual (with the associated ill numerical properties, [13, 14]), whereas GINNs share the formulation and numerical behaviour of classical topology optimization (§2.1, p. 3). A feature specific to geometric problems is solution multiplicity, arising from under-determinedness, uncertainty or near-optimality, so the framework is asked to generate a whole set $S \subset K$ of diverse near-optimal solutions rather than one optimum (§3, p. 4).

## Method

**Representation.** A shape is the sub-level set $\Omega = \{x \in X \mid f(x) \le 0\}$ of a scalar neural field $f = f_\theta : X \to \R$ on $X \subset \R^n$, with boundary the zero level set $\partial\Omega = \{x \in X \mid f(x) = 0\}$; this "implicit neural shape" is chosen for memory efficiency, continuity and differentiability (§3.1, p. 4). The field is additionally trained to approximate the signed-distance function (SDF) of $\Omega$ — this removes the ambiguity among fields representing the same geometry and aids persistent homology, surface sampling and diversity (§3.1, p. 5). Because the SDF obeys the eikonal equation $\lVert\nabla f\rVert = 1$ except on the medial axis (§E, p. 28), the eikonal constraint is treated like any other design constraint in training (§3.1, p. 5).

**Constraints.** Every requirement is first written as a differentiable violation $c_i : \mathcal{F} \to \R$ by translating a constraint on the sets $(\Omega, \partial\Omega)$ into one on $f$ (Tab. 1, p. 5); the choice of violation is not unique (§3.1, p. 5). Table 1 lists the main-experiment constraints: design-region containment, interface attachment, a prescribed normal, and persistent-homology topology. Two losses receive extra attention: **connectedness**, computed by discretizing $f$ on a cubical complex, running persistent homology (PH) with CRipser [85], and reading off the grid cell where a feature dies, which makes the loss differentiable without differentiating the PH library itself (§4.1, p. 7; §E, pp. 28–29); and **smoothness**, a surface integral $\int_{\partial\Omega \setminus I} e(\kappa_1, \kappa_2)\, dx$ of principal curvatures, computed in closed form from $\nabla_x f$ and $H_x f$ (§4.1, p. 7).

**Generating diversity.** A generator $G : z \mapsto f$ is a conditional neural field $f(x) = F(x; z)$ conditioned by input concatenation; the solution set is $S = \operatorname{Im}G(Z)$, and each constraint and the objective are lifted to expectations over the latent distribution $p_Z$ (§3.2, p. 5). Diversity is added as a constraint on the set, using the *minimal aggregation measure* built from a geometric shape distance (Eq. 2, Eq. 3); its concavity promotes uniform coverage of the feasible set and counters the mode-collapse observed with a naive generative model (§3.2, p. 6; §F, p. 29).

**Optimization.** Constrained problem (Eq. 1) is solved as a sequence of unconstrained problems by the *adaptive* augmented Lagrangian method of Basir and Senocak [55]: one penalty parameter $\mu_i$ per constraint is updated by RMSprop and doubles as the multiplier learning rate, so diverse loss terms with different scales need no manual weighting (§4.1, p. 6; §D, p. 24). Surface points are sampled by projecting envelope samples with Newton iterations and repelling them, after a 500-iteration warm-up; models must have non-vanishing second derivatives, so ReLU is replaced by softplus, SIREN or WIRE (§4.1, pp. 7–8).

## Key equations and definitions

The shape is the sub-level set $\Omega = \{x \in X \mid f(x) \le 0\}$ with $\partial\Omega = \{x \in X \mid f(x) = 0\}$, and the SDF is $f(x) = d(x, \partial\Omega)$ for $x \in \Omega^c$ and $f(x) = -d(x,\partial\Omega)$ for $x \in \Omega$ (Eq. 21, p. 28).

Generative GINN problem — find a diverse near-optimal solution set:

$$
\min_{\substack{S \subset K \\ \delta(S) \ge \delta_{\min}}} O(S) \qquad \text{(Eq. 1, p. 4)}
$$

$O(S)$ aggregates the objectives $o(f)$ over all solutions $f \in S$, and $\delta$ is a set-level diversity (§3.2, p. 5).

Diversity as minimal aggregation (the $p = 1/2$ instance of the general form Eq. 26):

$$
\delta(S) = \left( \sum_i \left( \min_{j \neq i} d(f_i, f_j) \right)^{1/2} \right)^{2} \qquad \text{(Eq. 2, p. 6)}
$$

Geometric shape distance, integrating each field's square only over the other shape's boundary:

$$
d(f_i, f_j) = \sqrt{ \int_{\partial\Omega_i} f_j(x)^2 \, \mathrm{d}x + \int_{\partial\Omega_j} f_i(x)^2 \, \mathrm{d}x } \qquad \text{(Eq. 3, p. 6)}
$$

The distance is closely related to the chamfer discrepancy when the fields are SDFs; it is not a metric on functions, but the paper aims at the geometries they represent (§3.2, p. 6).

Adaptive augmented Lagrangian objective (minimized over $\theta$, maximized over $\lambda$):

$$
L(\theta, \lambda, \mu) := O(S_k(\theta)) + \sum_{i=1}^{m+1} \lambda_i C_i(S_k(\theta)) + \frac{1}{2} \sum_{i=1}^{m+1} \mu_i C_i(S_k(\theta))^2 \qquad \text{(Eq. 4, p. 7)}
$$

with $C_{m+1}(S_k) = \max(\delta(S_k) - \delta_{\min}, 0)$ the diversity constraint, $\mu_i$ updated by RMSprop (Eqs. 18–20, pp. 24–25) and $\lambda_{t,i} \leftarrow \lambda_{t-1,i} + \mu_{t,i} C_i(\theta_t)$ (Eq. 20, Alg. 1, p. 25).

Differentiable connectedness loss, pushing down all but the most persistent component at iso-level $a_0$ (with deaths at grid coordinates $x^{0d_i}, y^{0d_i}, z^{0d_i}$):

$$
L_{cc} = \sum_{i=2}^{M} \left( a_0 - f_\theta(x^{0d_i}, y^{0d_i}, z^{0d_i}) \right)^2 \qquad \text{(Eq. 24, p. 29)}
$$

Surface-strain metric used for the smoothness ablation, with the clip $\kappa_{\max} = 10^6$:

$$
E_{\text{strain}}(\Omega) = \frac{1}{N} \sum_{i \in [N]} \min\!\left( \operatorname{div}^2\!\left( \nabla \frac{f(x_i)}{\lvert f(x) \rvert} \right), \kappa_{\max} \right) \qquad \text{(Eq. 10, p. 22)}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $X \subset \R^n$ | domain of the field (e.g. $[-1,1]\times[-0.5,0.5]$ for the 2D obstacle) |
| $f = f_\theta$ | scalar neural field; implicit representation of the shape |
| $\Omega$, $\partial\Omega$ | shape (sub-level set) and its boundary (zero level set) |
| $E \subseteq X$, $I \subset E$ | design region; interface the shape must attach to |
| $\bar n(x)$ | prescribed outward normal on the interface $I$ |
| $K = \{f \mid c_i(f)=0\}$ | feasible set defined by the geometric constraints |
| $o(f)$, $O(S)$ | objective of one solution; aggregated objective over a set |
| $S$, $S_k$ | solution set; batch of $k$ i.i.d. sampled shapes |
| $G$, $z \in Z$, $F(x;z)$ | generator, latent variable, conditional neural field |
| $d(f_i,f_j)$, $\delta(S)$ | shape distance (Eq. 3); diversity of a set (Eq. 2) |
| $\delta_{\min}$, $C_{m+1}$ | required diversity; diversity constraint violation |
| $c_i$, $C_i(S)$ | per-solution constraint violation; its expected value over $S$ |
| $\lambda_i$, $\mu_i$ | Lagrange multipliers; adaptive penalty parameters |
| $\kappa_1, \kappa_2$, $\kappa_H$ | principal curvatures; mean curvature $\kappa_H = (\kappa_1+\kappa_2)/2$ |
| $E := \kappa_1^2 + \kappa_2^2$, $E_{\log}$ | surface-strain objective and its $\log(1+E)$ variant |
| $b_n$, $H_n(X)$ | $n$-th Betti number; $n$-th homology group |

## Main results

**Validation problems (§4.2, pp. 7–8).** A *generative PINN* applied to an under-determined stationary Gray-Scott reaction-diffusion system (no initial condition; $D_v = 1.2\times10^{-5}$, $D_u = 2D_v$, $\alpha = 0.028$, $\beta = 0.057$) produces diverse Turing patterns ("worms") that morph smoothly under latent interpolation; the authors state this is the first PINN producing 2D Turing patterns in a data-free setting (§4.2, p. 7; §B, pp. 20–21). *Plateau's problem* (minimal surface attached to a prescribed closed curve) is solved qualitatively, agreeing with the known minimal surface (Fig. 5(a), p. 8). The *parabolic mirror* — learning a height function with $f(0)=0$ whose reflected rays meet at $(0,1)$ — approximates the known parabolic solution; it demonstrates that the GINN framework extends beyond implicit shapes to other representations (Fig. 5(b), p. 8). For the 2D *obstacle* (connecting two vertical interfaces around a circular obstacle), the naive generative softplus-MLP mode-collapses and the diversity constraint is required to recover the superposition of 16 distinct solutions (Fig. 5(c), p. 8).

**Engineering case study (§4.3, pp. 9–10).** On the General Electric / GrabCAD jet-engine lifting bracket, the geometric requirements (fit in the design region $E$; attach to the cylindrical interfaces $I$) are kept, but the original objective (minimize volume subject to a linear-elasticity PDE) is replaced by minimizing surface-strain $E$ subject to a connectedness constraint, which avoids a PDE solver in the training loop (§4.3, p. 9). The trained model produces a smooth, singly connected shape inside the design region, and variations with a hole-count constraint and the $E_{\log}$ energy show user control over the result (Fig. 1, p. 1; Fig. 3, p. 5).

**Ablations on the bracket (Tab. 3, p. 23; §C.2, pp. 22–23).** All volumetric constraint metrics ($\text{vol}(\Omega\setminus E)/\text{vol}(X\setminus E)$, $\text{vol}(\Omega\cap\delta E)/\text{vol}(\delta E)$, $\text{vol}(DC(\Omega\cap E))/\text{vol}(E)$, and $CD_1(\Omega,I)$) are 0.00 in every column. Base model: $b_0(\Omega) = 1$, $b_0(\Omega\cap E) = 1$, $CI(\Omega,I)/n_I = 1.00$, $E_{\text{strain}}(\Omega) = 7{,}422$. Ablating the eikonal loss: $b_0(\Omega) = 9$, $b_0(\Omega\cap E) = 7$, $CI/n_I = 0.67$, $E_{\text{strain}} = 28{,}115$. Ablating the connectedness loss: $b_0(\Omega) = 2$, $CI/n_I = 0.83$ (the text notes this is 5 of 6 interfaces, so one is disconnected), $E_{\text{strain}} = 22{,}708$. Ablating the smoothness loss: $E_{\text{strain}} = 8{,}497$; using the $E_{\log}$ variant: $E_{\text{strain}} = 47{,}842$. The authors conclude the eikonal loss improves connectedness and smoothness, the connectedness loss is "crucial" for connected shapes, and the smoothness loss quantitatively improves smoothness (§C.2, pp. 22–23).

**Multi-solution ablation (Tab. 4, p. 24; §C.2, p. 23).** With 9 shapes and $\dim(z)=2$: base model $\delta_{\text{mean}} = 0.21$, $b_0(\Omega) = 1.11$, $CI/n_I = 0.96$; without the diversity loss ($\lambda_{\text{div}}=0$) $\delta_{\text{mean}} = 0.01$, $b_0(\Omega) = 1.00$, $CI/n_I = 1.00$; with fixed $z$ sampling $\delta_{\text{mean}} = 0.23$, $b_0(\Omega) = 3.00$, $CI/n_I = 0.69$. Adding diversity yields a *structured* latent space: sampling the 2D latent space produces continuously morphing feasible shapes (i.e. the model generalizes) whose organization becomes more pronounced for larger $\delta_{\min}$, while over-specifying it harms convergence (§4.3, p. 9; Fig. 6, p. 10; Fig. 8, p. 20).

**Cost (§4.3, p. 9; §A, p. 18).** A single bracket shape takes roughly 10K iterations and 1 hour; the diverse model takes 50K iterations and 72 hours; surface-point sampling accounts for about 50% of runtime, and the diversity and smoothness terms roughly double and quadruple-to-quintuple the iteration count. Single-shape training needs up to ~9 GB GPU memory for the bracket; multi-shape training ~45 GB for 9 bracket shapes and ~7 GB for 16 obstacle shapes.

## Limitations stated by the authors

A modified version of the original bracket task was solved, and no comparison to established topology-optimization methods was made because that would require integrating a PDE solver (§5, p. 10). The adaptive-ALM strategy of location-specific penalties does not transfer to the moving surfaces here, and curvature being a second-order operator is expected to be ill-conditioned, motivating second-order optimizers (§4.3, p. 9). Diversity on infinite sets, especially with manifold structure, is left to future work (§3.2, p. 6), as are alternative shape distances/aggregations, conditioning mechanisms and a broader range of constraints (§5, p. 10). GINNs were studied only in the limit of no data, though partial observations could be integrated (§5, p. 10).

## Critical assessment

My reading: the central contribution — data-free, constraint-driven shape generation with an explicit diversity term — is genuine and the framework is cleanly decomposed, but the evidence is thinner than the framing. There are no baselines and largely no established metrics (§4, p. 6; §5, p. 10, conceded), so "control over geometrical and topological properties" rests on one quantitative case (the bracket) plus self-defined metrics, and several validation problems (Plateau, mirror) are only checked *qualitatively* against known solutions (§4.2, p. 8). The bracket experiment is deliberately weakened (smoothness instead of a PDE-constrained volume objective), so it does not compete with the actual competition solution. Two documentation inconsistencies are worth flagging: §4.3 (p. 9) speaks of "five cylindrical interfaces" while §C.2 (p. 23) assumes six (the reported $CI/n_I = 0.83 \approx 5/6$); and §C.2 (p. 23) states that dropping the diversity loss "halves the diversity $\delta_{\text{mean}}$", whereas Tab. 4 gives $0.01$ versus $0.21$ — a factor ~20, not 2. Equation 10 appears to have a typo ($\lvert f(x)\rvert$ should arguably be $\lvert f(x_i)\rvert$; sic), and the exponent placement in Eq. 2 is hard to read from the printed layout — it is consistent with the $p=1/2$ case of the general form Eq. 26. The "generalization" claim (Fig. 6) is demonstrated by latent-space interpolation of a single trained model, which is continuity rather than generalisation in the supervised sense. Reproducibility is helped by released code (github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks, p. 2), but 72 h for the diverse 3D model and up to 45 GB GPU memory are a practical barrier. The F. derivation (Eq. 28, p. 31) cleanly justifies ignoring the integral off the two boundaries; the observation that the underlying formulation matches classical topology optimization, avoiding PINN ill-conditioning, is a useful conceptual point (§2.1, p. 3).

## Connections within the collection

There is no `builds_on` relation: every foundation GINNs rely on — PINNs, DeepSDF/occupancy networks, SIREN, WIRE, adaptive ALM, CRipser — lies outside this collection (§2, pp. 2–4). The share of the first author with [@berzinsPolyhedralComplexExtraction2023] is the closest link: that earlier paper extracts the polyhedral complex of a ReLU network, whereas GINN deliberately avoids ReLU because its second derivative vanishes everywhere (§A.4, p. 18) and instead requires smooth activations (softplus, SIREN, WIRE, §4.1, p. 7) — the two are complementary views of "geometry" in neural representations rather than dependent results. GINN cites the collection's geometric-deep-learning overview [@bronsteinGeometricDeepLearning] (as reference [8]) as the principled, symmetry-and-scale-separation way to characterize problems, though it does not use group equivariance in the method and this remains related-work context (§2.1, p. 3). The structured latent space the diversity constraint induces (Fig. 6, p. 10) is thematically adjacent to the latent-space geometry studied in [@pouplinIdentifyingLatentDistances2023] and [@arvanitidisGeometricallyEnrichedLatent2020]; GINN neither cites nor uses those, so this is possibly related rather than a documented connection.
