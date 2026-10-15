---
citekey: combeExploringInformationGeometry2025
title: "Exploring Information Geometry: Recent Advances and Connections to Topological Field Theory"
authors: [Noémie C. Combe, Philippe G. Combe, Hanna K. Nencka]
year: 2025
venue: arXiv preprint
doi: "10.48550/arXiv.2502.11188"
arxiv: "2502.11188"
url: null
kind: tutorial
source_type: pdf
source_sha256: d454ee6f6fd4d924b0e086fb89af93c5fbfbe00df4c8322089d3b1c176fae812
pages: 164
ingested: 2026-10-15
tags: [information-geometry, frobenius-manifolds, wdvv-equations, statistical-manifolds, paracomplex-geometry, topological-field-theory]
concepts_introduced:
  - manifold of probability distributions
  - pre-Frobenius manifold
  - Frobenius manifold
  - WDVV equations
  - structure connection
  - semisimple Frobenius structure
  - Amari–Chentsov tensor
  - Gromov–Witten invariants for statistical manifolds
  - m-pairs
  - paracomplex numbers
concepts_used:
  - affine flat connection
  - curvature and torsion
  - fiber bundle
  - Fisher–Rao metric
  - alpha-connection
  - exponential family
  - Markov morphism
  - Klein geometry
  - Monge–Ampère equation
  - Gromov–Witten invariants
builds_on: []
datasets: []
---

# Exploring Information Geometry: Recent Advances and Connections to Topological Field Theory

**Combe, Combe, Nencka (2025)**, *arXiv preprint*. [arXiv:2502.11188](https://arxiv.org/abs/2502.11188)
<!-- authors' affiliation: University of Warsaw (MIMUW), per p. 1 -->

!!! abstract "TL;DR"
    An introductory textbook, grown from a lecture in Göteborg, that builds from point-set topology and differential geometry through measure-theoretic probability to (pre-)Frobenius manifolds, and argues that the manifold of probability distributions of exponential type carries a hidden Frobenius structure. Its distinctive claims are that statistical manifolds admit canonical affine coordinates whose coordinate lines are Ceva lines of the simplex, that the Amari–Chentsov tensor defines a multiplication making a finite-dimensional exponential family into a Frobenius manifold, and that a statistical analogue of Gromov–Witten invariants governs the learning process. The exposition is in three parts with exercises and selected solutions.

## Problem and motivation

The book asks how to endow a probability space $(\Omega, \mathcal{F}, P)$ with geometric data, so that statistics becomes a branch of geometry: the central object is the *manifold of probability distributions*, a family of parametrized distributions on a measurable space that inherits a manifold structure (§0, p. 11). The authors argue that differential geometry supplies the natural language (connections, parallel transport, curvature, flatness) for this structure, and that the payoff is an unexpected link to *Topological Field Theory* (TFT) through the Witten–Dijkgraaf–Verlinde–Verlinde (WDVV) equations and Frobenius manifolds (§0, p. 12). They single out absolutely continuous distributions on intervals of length $2\pi$ (wrapped exponential distributions) after listing the three principal distribution classes (§0, pp. 11–12).

The stated motivation is partly applied: the geometry of information matters for artificial intelligence and machine learning, e.g. language models, so its mathematical underpinnings deserve development (§0, p. 11). The authors are explicit that the text "does not seek to be exhaustive" but offers a pathway into a field "still in its formative stages" (§Abstract, p. 9).

## Method

This is a pedagogical, three-part construction rather than a single theorem-and-proof paper.

**Part 1 (chapters 1–5)** builds the differential-geometric toolkit: general topology (open sets, compactness, connectedness, homotopy) (§1, pp. 11–28); topological and modelled manifolds via charts and atlases (§2, pp. 29–36); differentiability through Gâteaux derivatives, chosen because it survives infinite-dimensional settings without a norm (§3, pp. 39–52); fiber bundles (§4, pp. 53–62); and connections, parallel transport, covariant derivative, curvature and torsion (§5, pp. 63–76).

**Part 2 (chapters 6–8)** treats probability. Standard measure theory (Radon–Nikodym theorem, Thm. 6.1, p. 86) is followed by a self-contained introduction to information geometry (§6.7, pp. 86–90): exponential families, the tangent space of score vectors, the Fisher–Rao metric and the $\alpha$-connections. Chapter 7 recasts Klein's Erlangen programme in probabilistic terms — probability distributions as generalised "Plato–Klein figures", Markov kernels as the transformations that play the role of congruences — and proves that the property of being dominated is invariant under Markov morphisms (§7.4, Thm. 7.5, p. 101), so that dominated families form a complete subcategory of the category `CAP` of measurable spaces with Markov morphisms (Thm. 7.6, p. 102). Chapter 8 makes the geometry explicit: it defines five collections of distributions (`Cap`, `Capd`, `Caph`, `Conh`, `Var`, Def. 8.1, pp. 105–106), shows each carries a manifold atlas (Prop. 8.2, p. 106), and constructs the *canonical affine coordinates* in which a family becomes an exponential family (Def. 8.6, p. 109).

**Part 3 (chapters 9–11)** is the destination. Chapter 9 surveys affine structures, pre-Frobenius and Frobenius manifolds, the WDVV equations and the structure connection. Chapter 10 introduces projective $m$-pairs and the algebra of *paracomplex numbers* ($\epsilon^2 = 1$) and shows that a finite space of probability distributions is a manifold of 0-pairs. Chapter 11 defines Gromov–Witten invariants for statistical manifolds (GWS) and reads the learning process as an intersection problem between geodesics in a tangent bundle over the statistical manifold.

## Key equations and definitions

An exponential family density, the geometric starting point (§6.7, p. 86, and the general form p. 87):

$$
\rho_\theta = \exp\!\left(\sum_{i=1}^{n} \theta^i X_i - \psi(\theta)\right)
\qquad\qquad
\rho_\theta(\omega) = h(\omega)\exp\!\left(\eta(\theta)\cdot T(\omega) - A(\theta)\right)
$$

with $\theta$ the parameters, $X=(X_i)$ the sufficient statistics and $\psi$ (or $A$) the potential/log-partition function.

The Fisher–Rao metric as the covariance of the score vector (§6.7.2, p. 88), and the third-order Amari–Chentsov (skewness) tensor (§6.7.5, p. 89):

$$
g_{ij}(\theta) = E_{P_\theta}\!\left[\partial_i \ell_\theta \, \partial_j \ell_\theta\right],
\qquad\qquad
T_{ijk}(\theta) = E_{P_\theta}\!\left[\partial_i \ell_\theta \, \partial_j \ell_\theta \, \partial_k \ell_\theta\right],
$$

where $\ell_\theta = \ln \rho_\theta$. The $\alpha$-connections interpolate between these objects (§6.7.6, pp. 89–90):

$$
\Gamma^{\alpha}_{ijk} = E\!\left[\left(\partial_i\partial_j \ell + \frac{1-\alpha}{2}\,\partial_i \ell \, \partial_j \ell\right)\partial_k \ell\right]
= \Gamma^{1}_{ijk} + \frac{1-\alpha}{2}\, T_{ijk}.
$$

The metric-compatibility that defines a pre-Frobenius structure and the potentiality axiom (§9.3.3, p. 127; §9.3.4, p. 127):

$$
\begin{aligned}
A(X,Y,Z) &= g(X\circ Y, Z) = g(X, Y\circ Z), \\
g(X\circ Y, Z) &= g(X, Y\circ Z) = (XYZ)\,\Phi .
\end{aligned}
$$

The WDVV (associativity) equations for the potential $\Phi$ (§9.4.1, p. 128), and the structure connection, a pencil depending on a parameter $\lambda$ (§9.4.3, p. 129):

$$
\begin{aligned}
\sum_{ef} \Phi_{abe}\, g^{ef}\, \Phi_{fcd} &= \sum_{ef} \Phi_{bce}\, g^{ef}\, \Phi_{fad}, \\
&\qquad\qquad \nabla_{\lambda,X}(Y) := \nabla_{0,X}(Y) + \lambda\,(X\circ Y).
\end{aligned}
$$

The paracomplex unit and (indefinite) norm, and the entropy potential of a statistical manifold (§10.2, p. 143; Eq. 11.2, p. 150):

$$
\begin{aligned}
z &= x + \epsilon y, \quad \epsilon^2 = 1, \quad \norm{z} = x^2 - y^2, \\
&\qquad\qquad \tilde{\Phi} = \ln\sum_{\omega\in\Omega}\exp\!\Big(-\!\sum_j \beta_j X_j(\omega)\Big).
\end{aligned}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $M$ | smooth manifold; $TM$ its tangent sheaf |
| $g$ | metric compatible with an affine flat structure ($g_{ij}$ the Fisher metric) |
| $A$ | rank-3 symmetric tensor; $T_{ijk}$ its Amari–Chentsov component form |
| $\circ$ | commutative multiplication on $TM$ |
| $\Phi$ | potential function of the (pre-)Frobenius structure |
| $\nabla_\lambda$ | structure connection, pencil parametrised by $\lambda$ |
| $E$ | Euler (affine) vector field of a Frobenius manifold |
| $\rho_\theta,\ \ell_\theta$ | density and log-density ($\ell_\theta=\ln\rho_\theta$) |
| $\theta,\eta(\theta)$ | parameters and natural parameters of an exponential family |
| $\epsilon$ | paracomplex unit, $\epsilon^2=1$ |
| $\mathrm{H}_n$ | space of probability distributions on a finite measurable set |
| $\beta, X_j(\omega)$ | affine parameters and covectors (sufficient statistics) in GWS |

## Main results

The pivotal structural theorem is Manin's characterisation: a pre-Frobenius manifold $(M,g,A)$ is a Frobenius manifold if and only if the pencil of structure connections $\{\nabla_\lambda\}$ is flat (§9.4.3, Thm. 9.5, p. 129). The curvature splits as $\nabla_\lambda^2 = R_1\lambda^2 + R_2\lambda$, and $R_1 = 0$ is equivalent to potentiality while $R_2 = 0$ is equivalent to associativity, so the two Frobenius axioms are exactly the vanishing of these curvature terms (§9.6.5, Thm. 9.9, p. 135). For semisimple structures the picture is made explicit: $(TM,\circ) \cong (\mathcal{O}_M^{\,n},\cdot)$ with idempotents $e_i\circ e_j = \delta_{ij} e_i$, diagonal metric $\eta_i = g(e_i,e_i)$, diagonal cubic tensor $A = \sum_i \eta_i (\epsilon^i)^3$ and co-identity $E = \sum_i \eta_i \epsilon^i$ (Def. 9.5, p. 132). A semisimple pre-Frobenius structure is Frobenius precisely when the $e_i = \partial/\partial u^i$ commute and $\eta_i = e_i\eta$ for some local potential $\eta$, which makes $g$ a Hessian metric $g_{ij} = \partial^2\Phi/\partial u^i\partial u^j$ (§9.6.4, Thm. 9.8, pp. 132–133).

The bridge to statistics is the observation that a finite-dimensional statistical manifold of exponential type carries all the ingredients of a pre-Frobenius manifold: an affine flat structure, the Fisher metric $g$, the Amari–Chentsov tensor $T$, and a multiplication $u\circ v = \bar{T}(u,v)$ with $\bar{T}^k_{ij} = \sum_m g^{km} T_{ijm}$, which is metric-invariant, $g(u, v\circ w) = g(u\circ v, w)$ (§9.5, Thms. 9.6–9.7, pp. 130–131). Whether the full Frobenius conditions hold is handled through the extra potentiality and flatness requirements (p. 131).

Within Part 2, the geometry is pinned down through coordinates: in the *natural* and *canonical affine* coordinates $s$, the density takes the exponential form $\frac{dP_s}{d\mu}(\omega) = p(\omega,0)\exp[\sum_{\alpha=1}^{m-1} s^\alpha g_\alpha(\omega) - \Psi(s)]$, the simplex becomes an affine space, and the coordinate lines are *Ceva lines* with $\dot p_1(t) = p_1(t)[1-p_1(t)]$, i.e. $p_1(t) = 1/(1+e^{-t})$ (§8.0.5–8.0.8, pp. 109–114; Lemma 8.3, p. 111). Consequently a tensor field on the simplex is translation-invariant if and only if its components are constant in canonical coordinates, and there exist translation-invariant Euclidean metrics (§8.1–8.2, Thm. 8.6 and Cor. 8.7, p. 114).

Chapter 10 identifies the space of probability distributions with projective geometry: the space $\mathrm{H}_n$ of distributions on a finite measurable set of dimension $n+1$ is a manifold of 0-pairs (Prop. 10.2, p. 145), isomorphic to the Hermitian projective space over the cone $M_+(2,\mathbb{C})$ (Thm. 10.2, p. 145); the paracomplex decomposition $M = e_+M \oplus e_-M$ splits every such tangent space into two real submodules (§10.3, p. 144). Chapter 11 then defines the statistical Gromov–Witten invariants $\tilde Y_n$ and proves that the entropy function $\tilde\Phi$ is determined by them (Def. 11.1 and Prop. 11.1, pp. 149–150), and reads learning as the convergence $d(\gamma^+, \pi_{E^+}(\gamma^-)) \to 0$ of a geodesic to the orthogonal projection of another, with GWS controlling that evolution (Prop. 11.2, p. 152; Thm. 11.2, p. 153).

## Limitations stated by the authors

The text presents itself as non-exhaustive and introductory, "an invitation to further exploration" rather than a complete account (§Abstract, p. 9). Many foundational questions are declared open, including the complete classification of Frobenius manifolds, the identification of new classes, and the relations among existing classes (§9.1.1, p. 122). Several central claims are deliberately left as exercises rather than proved in the text: the identification of potential pre-Frobenius manifolds with Monge–Ampère domains is stated as Thm. 9.4 and then posed as Exercise 9.2 (pp. 127–128); the flatness characterisation is Exercise 9.3 (p. 129); the curvature-decomposition theorem is Exercise 9.4 (p. 135). The book contains no experiments, numerical results or datasets.

## Critical assessment

My reading, not the authors'. The expository middle is solid and the coordinate constructions in chapter 8 (natural charts, canonical affine coordinates, Ceva lines) are genuinely illuminating, but the load-bearing claims are thin where it matters. Several proofs are one-liners that assert the conclusion — Lemma 7.9 and Lemma 10.1 are dismissed with "the proof is evident", and Prop. 11.1 is argued by restating the definition of $\tilde Y_n$ rather than computing anything. The statistical Frobenius-manifold claim (chapter 9.5) is plausible and structurally motivated, but the text stops just short of demonstrating that a concrete exponential family satisfies potentiality and the flatness of $\nabla_\lambda$; the reader is shown the ingredients and told they match, not shown a worked example with explicit $\Phi$ and $\circ$. Chapter 11 is the weakest: the "learning process" is a geometric metaphor (a distance between two geodesics in $E^\pm$ tending to zero) with no learning rule, no algorithm and no experiment, so the theorem that GWS "determine the evolution of the learning process" (Thm. 11.2) is a reformulation rather than a result. The connection to TFT is asserted early (§0, p. 12) and only made concrete at the level of the WDVV equations, which is arguably the content already present in Dubrovin's and Manin's work rather than a new bridge.

There are also presentation defects a reader should know about: the metric definition (Def. 1.3, p. 21) writes the triangle *equality* $d(x,z)=d(x,y)+d(y,z)$ where the triangle inequality is meant (sic); the preface headers and the first chapter's numbering carry stray artifacts (a section numbered "-1.1", §Foreword, pp. vi–vii); "Kullback–Liebler" appears for Kullback–Leibler (p. 149, sic); and the bibliography entry [D96] lists the year 1993 where 1996 is intended (§Bibliography, p. 155). The Gaussian family is used as a running example but the two-dimensional manifold $\mathrm{H}_2$ is never carried through to an actual GWS computation. The main text has no code or data, so reproducibility is not at issue; the value of the work is expository.

## Connections within the collection

The book shares its core object — the manifold of probability distributions, the Fisher–Rao metric, dual/$\alpha$-connections and exponential families — with the collection's dedicated information-geometry introduction [@nielsenElementaryIntroductionInformation2020], and can be read as a companion that extends that background toward Frobenius manifolds and topological field theory rather than an alternative treatment; the two cover overlapping foundations (score vectors, Fisher metric, $\alpha$-connections) with different destinations.

The same statistical-manifold apparatus is put to generative-modelling use elsewhere in the collection: [@arvanitidisPullingBackInformation2022] pulls the Fisher–Rao metric of the decoder back to a latent space, [@arvanitidisGeometricallyEnrichedLatent2020] equips latent space with a Riemannian metric and geodesics, and [@pouplinIdentifyingLatentDistances2023] generalises latent distances to Finslerian geometry. These use the Fisher/metric machinery that this book develops abstractly, but for representation learning rather than for Frobenius/TFT structure — possibly related rather than directly building on one another.

The categorical Klein-geometry thread (transformations as congruences, equivariant connections, absolute covariance, §7.2, §7.6, pp. 97–104) resonates conceptually with the passive-symmetry and covariance argument of [@villarFullyCovariantMachine2023], though the two never cite each other and the settings differ. [@meilaManifoldLearningWhat2023] uses "manifold" in the data-cloud sense, a different notion from the statistical manifold here, so the overlap is terminological. The remaining collection papers on ReLU/polyhedral and topological geometry of networks do not interact with this text. Works the book leans on most heavily — Chentsov's statistical decision rules, Amari's information geometry, Dubrovin's and Manin's Frobenius manifolds, and the authors' own Combe–Manin and Combe–Combe–Nencka papers — are all outside this collection and are named here without citekeys.
