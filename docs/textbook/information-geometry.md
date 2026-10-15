---
chapter: information-geometry
title: Information geometry
outline_hash: f08837f1cf81
built_from:
  arvanitidisPullingBackInformation2022: b6e2d23bcb04
  combeExploringInformationGeometry2025: f2e5daaa2fa6
  grigsbyHiddenSymmetriesReLU: 2b3f7998f609
  halversonMetricFlowsNeural2024: 2fd2a25dba82
  nielsenElementaryIntroductionInformation2020: 68ed040d3708
---

# Information geometry

!!! abstract "In this chapter"
    - A family of probability distributions is a smooth manifold, and putting a *metric* and an *affine connection* on it — independently — yields a statistical manifold.
    - The Fisher–Rao metric and the Amari–Chentsov cubic tensor are singled out by statistical invariance; the third-order tensor generates the one-parameter family of $\alpha$-connections.
    - A divergence induces a metric and a pair of conjugate connections, and Bregman divergences yield dually flat manifolds with dual coordinates and Pythagorean theorems.
    - Natural-gradient descent is Riemannian steepest descent in the Fisher metric, and it is the one update rule of this chapter that survives reparametrisation.
    - Prerequisites: none; this chapter starts the information-geometric thread. It connects forward to [Manifold learning and intrinsic dimension](manifold-learning.md) and [Riemannian and Finsler geometry of latent spaces](latent-geometry.md).

A statistical model is not one probability distribution but a *family* of them, indexed by a parameter $\theta$. Information geometry takes that family literally: it treats the set of distributions as a smooth manifold and asks what geometric structures it carries [@nielsenElementaryIntroductionInformation2020, §1.1, p. 1]. The manoeuvre is powerful because it converts questions about inference — how far apart are two models, how fast can a parameter be estimated, how should we move during training — into questions about curvature, distance and geodesics. The two surveys behind this chapter develop the answer from two ends: [@nielsenElementaryIntroductionInformation2020] gives the shortest self-contained route through the dualistic core, and [@combeExploringInformationGeometry2025] builds the same objects slowly from topology and measure theory and then pushes them toward Frobenius manifolds and topological field theory [@combeExploringInformationGeometry2025, §Abstract, p. 9].

This chapter is the collection's entry point into that thread, so it stays with the classical picture. It defines the statistical manifold and its Fisher–Rao metric, constructs the metric and the connections from divergences, derives the $\alpha$-connections, and closes with the natural gradient — the piece of information geometry that is used most often in machine learning [@nielsenElementaryIntroductionInformation2020, §4.1, p. 36]. The running theme is the chapter's stated goal: at each step, ask which of these objects is intrinsic and which depends on the parametrisation. I keep the [shared notation](../notation.md): $\mathcal{M}$ for a statistical manifold with coordinates $\theta$, $p_\theta$ for a distribution, $g_{ij}(\theta)$ for the Fisher metric, $\nabla^{(\alpha)}$ for the $\alpha$-connections, and $\Gamma_{ijk}$ for the Amari–Chentsov cubic tensor, following [@nielsenElementaryIntroductionInformation2020, Appendix C].

## Two independent structures on a family of models

The differential geometry needed here is small but precise, and it is worth separating the two structures it puts on a manifold. A smooth $D$-dimensional manifold $M$ locally looks like $\R^D$; at each point $p$ a tangent plane $T_p$ linearises it, and on that plane one can place a *metric tensor* $g$ and an *affine connection* $\nabla$ [@nielsenElementaryIntroductionInformation2020, §2.1, pp. 3–4]. The metric is an inner product on each tangent plane, $g_{ij}:=\langle e_i,e_j\rangle$ in a coordinate basis, so it measures lengths and angles; the connection is a differential operator that defines the covariant derivative $\nabla_XY$, parallel transport of vectors along a curve, and geodesics as the autoparallel curves satisfying $\nabla_{\dot\gamma}\dot\gamma=0$ [@nielsenElementaryIntroductionInformation2020, §2.2, Eq. 4, p. 5; §2.3.1–2.3.3, Eqs. 13–18, pp. 6–8].

The two structures are independent, and this is the point that separates information geometry from ordinary Riemannian geometry. The metric does not determine the connection by itself; a connection is fixed in coordinates by the Christoffel symbols $\Gamma^k_{ij}$ through $\nabla_{\partial_i}\partial_j=\Gamma^k_{ij}\partial_k$, and different choices give different notions of straightness on the same metric space [@nielsenElementaryIntroductionInformation2020, §2.3.1, Eq. 14, p. 6]. Only when one imposes the two extra conditions — no torsion, $\nabla_XY-\nabla_YX=[X,Y]$, and metric compatibility — does the fundamental theorem of Riemannian geometry supply a *unique* connection, the Levi–Civita connection $^{LC}\nabla$, whose Christoffel symbols are $\frac12g^{kl}(\partial_ig_{jl}+\partial_jg_{il}-\partial_lg_{ij})$ [@nielsenElementaryIntroductionInformation2020, Thm. 1, Eq. 35, p. 11].

!!! note "Background"
    A *torsion-free metric-compatible* connection is one whose parallel transport preserves inner products and which is symmetric in its lower indices. The Levi–Civita connection is the unique such connection built from the metric alone, and classical Riemannian geometry is entirely the study of the pair $(M,g)$ with this connection. Information geometry deliberately drops uniqueness: it keeps the metric and allows a *pair* of connections coupled to it, which is what makes the dualistic structure possible.

Information geometry therefore works with the quadruple $(M,g,\nabla,\nabla^*)$ rather than with $(M,g)$ alone [@nielsenElementaryIntroductionInformation2020, §2.5, p. 11]. The two connections are *conjugate* with respect to $g$ when they satisfy, for all smooth vector fields $X,Y,Z$,

$$
X\langle Y,Z\rangle=\langle\nabla_XY,Z\rangle+\langle Y,\nabla^*_XZ\rangle,
$$

which is exactly the condition that parallel transport by $\nabla$ of one vector and by $\nabla^*$ of another preserves their inner product [@nielsenElementaryIntroductionInformation2020, Def. 1, Eq. 40, p. 12]. Conjugation is an involution, $(\nabla^*)^*=\nabla$, and every connection on a metric manifold has a unique conjugate, so the pair is determined by either member [@nielsenElementaryIntroductionInformation2020, Prop. 1–2, p. 13]. The average $\bar\nabla=(\nabla+\nabla^*)/2$ is self-conjugate and coincides with $^{LC}\nabla$, so the classical connection sits in the middle of every conjugate pair [@nielsenElementaryIntroductionInformation2020, Property 3, Eqs. 45–46, p. 13]. The Riemannian manifold is thus the *self-dual* case $\nabla=\nabla^*={}^{LC}\nabla$; what information geometry adds is the freedom to move one connection away from the middle in a statistically meaningful way.

## The Fisher–Rao metric

A parametric family $P=\{p_\theta\}$ over a sample space $X$ becomes a manifold whose coordinates are $\theta$, and its natural metric comes from the sensitivity of the density to the parameter. The *score* of the model is $s_\theta=\nabla_\theta\ell$ with $\ell=\log p_\theta$, and the *Fisher information matrix* is its covariance,

$$
I(\theta)=\E_\theta\!\left[\partial_i\ell\,\partial_j\ell\right]_{ij}\succeq 0,
$$

positive semidefinite and positive definite for a regular model [@nielsenElementaryIntroductionInformation2020, §3.9, Eqs. 84–85, p. 21]. The Fisher information *metric* is the bilinear form this matrix defines on each tangent plane, $g(u,v)=u^\top I(\theta)v$ [@nielsenElementaryIntroductionInformation2020, Eq. 100, p. 24]. In the shared notation this is the single object written $g_{ij}(\theta)=\E[\partial_i\ell\,\partial_j\ell]$, and it is the metric of the Fisher–Rao manifold $(P,g)$ [@combeExploringInformationGeometry2025, §6.7.2, p. 88].

The Fisher metric has a rigidity property that nothing else in this chapter has: it is forced by statistics. Under a *Markov embedding* — coarse-graining followed by embedding into a larger simplex — the metric that is preserved is the Fisher metric, uniquely up to a scale constant [@nielsenElementaryIntroductionInformation2020, Thm. 8, p. 25]. This is the reason the Fisher metric, and not an arbitrary Riemannian metric, is the one attached to a statistical model: it is the unique invariant choice, so a construction that wants to be independent of the sample space has no alternative. The matrix is also *covariant* rather than invariant under reparametrisation of $\theta$: if $\eta=\eta(\theta)$ then $\bar I(\eta)=J^\top I(\theta(\eta))J$ with $J_{ij}=\partial\theta_i/\partial\eta_j$, so the infinitesimal length element $ds^2=I_{ij}d\theta^id\theta^j$ is unchanged [@nielsenElementaryIntroductionInformation2020, Eq. 86, p. 21].

!!! note "Background"
    The Fisher information already appears in classical estimation theory without any geometry: the Cramér–Rao bound states that any unbiased estimator $\hat\theta_n$ from $n$ i.i.d. samples has covariance at least $n^{-1}I^{-1}(\theta)$, in the Löwner order [@nielsenElementaryIntroductionInformation2020, Eq. 93, p. 22]. Information geometry reads the same inequality as a statement about the metric: the attainable precision is the inverse Fisher metric, so the metric controls how well the parameter can be resolved. This is the sense in which a Riemannian metric on the parameter space is *statistical* rather than imposed.

Two families carry the whole classical theory. An *exponential family* is the family of densities $\log p_\theta(x)=\sum_i t_i(x)\theta_i-F(\theta)+k(x)$ with $F$ the strictly convex cumulant function, and its Fisher information matrix is the covariance of the sufficient statistics, $I(\theta)=\mathrm{Cov}[t(x)]=\nabla^2F(\theta)$ [@nielsenElementaryIntroductionInformation2020, §3.9, Example 2, Eqs. 94–95, p. 23] (here the two sources agree, [@combeExploringInformationGeometry2025, §6.7, p. 86] writing the same form with potential $\psi$). A *mixture family* is the family $p_\theta=\sum_i\theta_iF_i(x)+C(x)$ of convex combinations of prescribed components, with $I(\theta)=\int_XF_iF_j/p_\theta\,d\mu$ [@nielsenElementaryIntroductionInformation2020, Example 3, Eqs. 98–99, p. 23]. The probability simplex is both, and both are the families whose Fisher geometry is flat, as the next sections make precise. For the univariate normal family the Fisher information in the $(\mu,\sigma^2)$ parametrisation is $\mathrm{diag}(1/\sigma^2,1/(2\sigma^4))$, the reader's check that the abstract definition reduces to the familiar matrix [@nielsenElementaryIntroductionInformation2020, Example 1, Eq. 89, p. 21].

A metric produces a distance when one integrates its speed along the shortest path. The *Fisher–Rao distance* is the geodesic length

$$
D_\rho(p_{\theta_1},p_{\theta_2})=\inf_\gamma\int_0^1\sqrt{\dot\gamma(t)^\top g_{\gamma(t)}\dot\gamma(t)}\,\mathrm dt,
$$

and it is generally unavailable in closed form, which is why the surveys fall back on bounds and numerical geodesic shooting [@nielsenElementaryIntroductionInformation2020, §2.5, Eqs. 38–39, pp. 11–12; §3.11, p. 27]. Where it is available it is instructive: for the Poisson family $D_\rho(\lambda_1,\lambda_2)=2(\sqrt{\lambda_1}-\sqrt{\lambda_2})$, while no closed form is known for the Gamma family or for two general multivariate Gaussians [@nielsenElementaryIntroductionInformation2020, §3.11, Eqs. 182–183, pp. 34–35]. This is a recurring theme in the chapter's later applications: the geometry is easy to define and hard to compute.

## Divergences and conjugate connections

Distances are not the only way to compare two distributions, and information geometry is built so that a much weaker notion suffices. A *divergence* $D(\theta:\theta')$ is a smooth, generally asymmetric function that is non-negative, vanishes exactly on the diagonal, has vanishing first derivatives there, and has a negative-definite mixed second derivative — the last two conditions being just enough differential structure to define a geometry [@nielsenElementaryIntroductionInformation2020, Def. 5, p. 15]. The Kullback–Leibler divergence $D_{\mathrm{KL}}(p\,\|\,q)=\int p\log(p/q)\,d\mu$ is the canonical example, and reversing the arguments gives the *dual* divergence $D^*(\theta:\theta'):=D(\theta':\theta)$ [@nielsenElementaryIntroductionInformation2020, §3.6, Eqs. 55, 116, pp. 15, 26].

Every divergence induces a full information manifold, and this is the construction that ties metrics and connections together. By Eguchi's recipe the metric is the symmetric part of the mixed second derivative at the diagonal, $g_{ij}=-\partial_i\partial'_jD(\theta:\theta')|_{\theta=\theta'}$, and the two asymmetric third derivatives give a pair of conjugate connections $\Gamma_{ijk}=-\partial_i\partial_j\partial'_kD$ and $\Gamma^*_{ijk}=-\partial_k\partial_i\partial'_jD$ [@nielsenElementaryIntroductionInformation2020, Thm. 4, Eqs. 56–58, p. 15]. The *Amari–Chentsov cubic tensor* is their difference,

$$
C_{ijk}=\Gamma^*_{ijk}-\Gamma_{ijk},
$$

a totally symmetric third-order tensor that measures the failure of the two connections to agree; a *statistical manifold* in the strict sense $(M,g,C)$ is precisely a metric together with such a symmetric cubic tensor [@nielsenElementaryIntroductionInformation2020, §3.3, Eqs. 47–49, Def. 3, pp. 13–14]. For a parametric family with the expected connections, this tensor is the third moment of the score, $C_{ijk}=\E_\theta[\partial_i\ell\,\partial_j\ell\,\partial_k\ell]$, the paper's *skewness tensor* [@nielsenElementaryIntroductionInformation2020, §3.9, Eq. 103, p. 24; @combeExploringInformationGeometry2025, §6.7.5, p. 89].

!!! warning "A sign to watch"
    The extraction of [@nielsenElementaryIntroductionInformation2020] garbles the signs of the $\alpha$-connections in the rendering: the printed Eq. (50) carries a minus and Eq. (51) a plus on the $\tfrac{\alpha}{2}C$ term, consistent with raising an index on a totally symmetric tensor. I follow the printed versions throughout. [@combeExploringInformationGeometry2025, Eq. 6.7.6, pp. 89–90] writes the same family with the opposite sign convention, $\Gamma^\alpha_{ijk}=\Gamma^1_{ijk}+\frac{1-\alpha}{2}T_{ijk}$, which agrees once one fixes the normalisation $\Gamma^1$.

Which divergences deserve to be called statistical is again settled by invariance. A divergence satisfies *information monotonicity* when coarse-graining the sample space — lumping outcomes together — cannot increase it, $D(\theta_{\bar A}:\theta'_{\bar A})\le D(\theta:\theta')$, and the only separable divergences with this property are the $f$-divergences $I_f(\theta:\theta')=\sum_i\theta_if(\theta'_i/\theta_i)$ for a convex generator $f$ [@nielsenElementaryIntroductionInformation2020, §3.10, Eqs. 110–112, p. 25]. The family includes the Kullback–Leibler divergence at $f(u)=-\log u$, the reverse Kullback–Leibler, the squared Hellinger divergence at $\alpha=0$, the Jensen–Shannon divergence and the total variation distance [@nielsenElementaryIntroductionInformation2020, §3.10, Eqs. 115–120, p. 26]. The key infinitesimal fact is that an invariant $f$-divergence reproduces the Fisher metric to leading order, $I_f[p_\theta:p_{\theta+d\theta}]\simeq\frac12g_{ij}d\theta^id\theta^j$, so the metric, the divergences and the connections are not independent structures but faces of one object [@nielsenElementaryIntroductionInformation2020, §3.10, Eqs. 121–122, p. 27].

## Dually flat manifolds and Bregman geometry

The most tractable case is when one chooses the divergence to be a Bregman divergence. Given a strictly convex, smooth generator $F$, the *Bregman divergence* is the gap between the function and its tangent hyperplane,

$$
B_F(\theta:\theta')=F(\theta)-F(\theta')-(\theta-\theta')^\top\nabla F(\theta'),
$$

and it induces the geometry ${}^Fg=\nabla^2F(\theta)$, ${}^FC_{ijk}=\partial_i\partial_j\partial_kF$, and vanishing $\nabla$-Christoffel symbols [@nielsenElementaryIntroductionInformation2020, §3.7, Eqs. 62–65, p. 16]. A manifold with this structure is *dually flat*: the connection $\nabla$ is flat in the Bregman coordinates $\theta$, its conjugate $\nabla^*$ is flat in the dual coordinates $\eta=\nabla F(\theta)$, and the two coordinate systems are exchanged by the Legendre–Fenchel transform $\theta=\nabla F^*(\eta)$ [@nielsenElementaryIntroductionInformation2020, §3.7, Eqs. 63, 66, pp. 16, 18]. Because the dual potentials are convex conjugates, a dually flat space behaves like Euclidean space: it has global affine coordinates, straight primal and dual geodesics, and a Pythagorean theorem $B_F(P:R)=B_F(P:Q)+B_F(Q:R)$ for the dual projection $Q$ of $P$ onto a dual-flat submanifold [@nielsenElementaryIntroductionInformation2020, §3.7, Thm. 6–7, p. 18].

The two families of the previous section are the modelling payoff. Both the exponential family and the mixture family are dually flat: their cumulant function $F$ is a Bregman generator, the Fisher metric is its Hessian $I(\theta)=\nabla^2F(\theta)$, and the two expected connections are exactly the dual pair induced by the Kullback–Leibler divergence [@nielsenElementaryIntroductionInformation2020, §3.9, Eqs. 95, 109, pp. 23–24]. The dictionary is stated cleanly in the companion volume: a finite-dimensional statistical manifold of exponential type carries the ingredients of a (pre-)Frobenius manifold, but this chapter stays with the flat case first [@combeExploringInformationGeometry2025, §9.5, pp. 130–131]. It also fixes intuition: in a dually flat space, "straight line" has two meanings, one per connection, and the two families inherit the Pythagorean decomposition that makes projection and clustering well posed [@nielsenElementaryIntroductionInformation2020, §4.3–4.4, pp. 41–44].

The companion volume gives the flat coordinates a probabilistic reading that is worth carrying forward, because it recurs in the latent-space chapters. On a finite simplex there are *natural coordinates* $t_j(P)=\int f_j(\omega)P(d\omega)$ given by expectations of functionals, and *canonical affine coordinates* $s$ in which the density takes the exponential form $\frac{dP_s}{d\mu}(\omega)=p(\omega,0)\exp[\sum_\alpha s^\alpha g_\alpha(\omega)-\Psi(s)]$ and the simplex becomes an affine space [@combeExploringInformationGeometry2025, §8.0.5, pp. 109–110]. The coordinate lines of this chart are the *Ceva lines*, the segments from a vertex to the opposite face, and along such a line the coordinate obeys $\dot p_1(t)=p_1(t)(1-p_1(t))$, i.e. $p_1(t)=1/(1+e^{-t})$ — a logistic curve, recovering the score of a Bernoulli coordinate [@combeExploringInformationGeometry2025, Lemma 8.3, p. 111]. Consistently, a tensor field on the simplex is translation-invariant exactly when its components are constant in these coordinates [@combeExploringInformationGeometry2025, Thm. 8.6, p. 114].

## The α-connections

One conjugate pair is not enough: from $(M,g,\nabla,\nabla^*)$ one gets a whole circle of geometries. Because the cubic tensor is totally symmetric, so is any scalar multiple $\alpha C$, and the *$\alpha$-connections* are defined by

$$
{}^\alpha\Gamma_{ij,k}=\Gamma^0_{ij,k}-\frac{\alpha}{2}C_{ij,k},
\qquad
g(\nabla^{(\alpha)}_XY,Z)=g({}^{LC}\nabla_XY,Z)+\frac{\alpha}{2}C(X,Y,Z),
$$

with $\nabla^{(0)}$ the Levi–Civita connection and $(\nabla^{(-\alpha)},\nabla^{(\alpha)})$ again conjugate, so $\nabla^{(1)}$ and $\nabla^{(-1)}$ are the original pair [@nielsenElementaryIntroductionInformation2020, §3.4, Eqs. 50–53, Thm. 2, p. 14]. The family is

$$
\Gamma^{(\alpha)}_{ij,k}=\tfrac{1+\alpha}{2}\Gamma_{ij,k}+\tfrac{1-\alpha}{2}\Gamma^*_{ij,k},
$$

and the recommended mental picture is a line through the two connections with the Levi–Civita connection at the centre. In the statistical case the two endpoints have names: $\nabla^{(1)}$ is the expected *exponential* connection and $\nabla^{(-1)}$ the expected *mixture* connection, coupled to the Fisher metric by the theory above [@nielsenElementaryIntroductionInformation2020, §3.9, Eqs. 101–102, p. 24]. The companion volume writes the same objects as $\Gamma^\alpha_{ijk}=\E[(\partial_i\partial_j\ell+\frac{1-\alpha}{2}\partial_i\ell\partial_j\ell)\partial_k\ell]=\Gamma^1_{ijk}+\frac{1-\alpha}{2}T_{ijk}$ [@combeExploringInformationGeometry2025, §6.7.6, pp. 89–90].

Flatness is inherited dually, which is what makes the family computationally useful. The fundamental theorem of information geometry states that a torsion-free connection of constant curvature $\kappa$ has a conjugate torsion-free connection of the same curvature, and its corollary is that a manifold is $\nabla^{(\alpha)}$-flat if and only if it is $\nabla^{(-\alpha)}$-flat; at $\alpha=\pm1$ this is the statement that dual flatness comes in pairs [@nielsenElementaryIntroductionInformation2020, Thm. 3, Cor. 1–2, p. 15; @combeExploringInformationGeometry2025, §7.5.2, p. 103]. The curvature of the expected $\alpha$-connection depends on both $\alpha$ and the model, so "the" geometry of a family is really a pencil of them. Finally, the $\alpha$-connections are also the ones induced by invariant divergences: for an $f$-divergence the induced $\pm1$-connections are the expected $\pm\alpha$-connections with $\alpha=2f'''(1)+3$, so the family is not a free choice but the image of the invariant divergences [@nielsenElementaryIntroductionInformation2020, §3.10, Eq. 124, p. 27].

!!! question "Open question"
    The two surveys disagree in emphasis about how much of the $\alpha$-family is forced. [@nielsenElementaryIntroductionInformation2020] derives the family from an arbitrary conjugate pair and only later asks which members are statistically meaningful, concluding that invariance pins the metric and the connections up to the $\alpha=2f'''(1)+3$ correspondence [@nielsenElementaryIntroductionInformation2020, §3.10, p. 27]. [@combeExploringInformationGeometry2025] instead asks when the third-order tensor turns a statistical manifold into a Frobenius manifold, which is a stronger, algebraic condition; whether a concrete exponential family satisfies it is left as an exercise in that book rather than demonstrated [@combeExploringInformationGeometry2025, §9.5, p. 131; Exercise 9.3, p. 129].

## Natural gradient descent

The application that brought information geometry into machine learning is optimisation in the Fisher metric. Ordinary gradient descent, $\theta_{t+1}=\theta_t-\alpha_t\nabla_\theta L(\theta_t)$, subtracts a *covariant* vector from a *contravariant* position and therefore depends on the parametrisation: minimising a function in Celsius and in Fahrenheit need not follow the same path, and can reach different stationary points [@nielsenElementaryIntroductionInformation2020, §4.1.1, Eqs. 194–195, p. 36]. The fix is to take the steepest direction with respect to the metric. Using the first-order retraction of the Riemannian exponential map, the *natural gradient* and its update are

$$
\mathrm{NG}\nabla L_\theta(\theta)=g_\theta^{-1}(\theta)\nabla_\theta L_\theta(\theta),
\qquad
\theta_{t+1}=\theta_t-\alpha_t\,\mathrm{NG}\nabla L_\theta(\theta_t),
$$

which is the Riemannian steepest-descent vector and is invariant under smooth reparametrisation [@nielsenElementaryIntroductionInformation2020, §4.1.2, Eqs. 201–203, Property 5, pp. 38–39]. The natural gradient is contravariant exactly because $g^{-1}$ raises the index of the ordinary gradient, which is the geometric content of the $g^{-1}$ factor. One caveat: natural-gradient descent does not keep the iterates inside the domain when $\Theta\neq\R^D$, so a constrained parameter needs an explicit projection [@nielsenElementaryIntroductionInformation2020, §4.1.2, p. 39].

On a dually flat space the natural gradient becomes something one can already compute. With $g_\theta=\nabla^2F(\theta)$ and the dual coordinates $\eta=\nabla F(\theta)$, the chain rule gives $^{\mathrm{NG}}\nabla_\theta L_\theta=\nabla_\eta L_\eta$, so *natural-gradient descent in the primal coordinates is ordinary gradient descent in the dual coordinates* [@nielsenElementaryIntroductionInformation2020, §4.1.3, Property 7, Eqs. 208–211, p. 40]. This is the cleanest justification of the method: on an exponential family, where $\eta$ are the natural parameters, the awkward $g^{-1}$ factor disappears. The same statement can be read as mirror descent: the proximal update $\theta_{t+1}=\arg\min_\theta\{\langle\theta,\nabla L(\theta_t)\rangle+\frac1{\alpha_t}B_F(\theta:\theta_t)\}$ with the Bregman divergence as proximity is equivalent to natural-gradient descent on the dual Hessian manifold, which is why Bregman mirror descent and natural gradient are the same algorithm seen from two charts [@nielsenElementaryIntroductionInformation2020, §4.1.3, Eqs. 204–205, Property 6, p. 39].

The practical route to using the natural gradient without differentiating the model is natural evolution strategies. Writing the problem as $\min_\lambda\E_{p_\lambda}[f(x)]$, the search gradient is estimated by $\frac1n\sum_if(x_i)\nabla\log p_\lambda(x_i)$ and the Fisher matrix by the empirical average $\tilde I(\lambda)=\frac1n\sum_i\nabla_\lambda\ell_\lambda(x_i)\nabla_\lambda\ell_\lambda(x_i)^\top$ [@nielsenElementaryIntroductionInformation2020, §4.1.4, Eqs. 212–214, p. 40]. The paper flags a real hazard: the empirical Fisher can be degenerate and need not respect the structure of the true Fisher matrix — for a normal family the expected matrix is diagonal while the sample version generally is not — so steps must be taken to regularise it [@nielsenElementaryIntroductionInformation2020, §4.1.4, pp. 40–41]. This is the point of contact with neural-network training, and it is the one place the tutorial admits it does not develop: parameter-space Fisher information, neuron manifolds and natural-gradient training of networks are listed as applications but not worked out [@nielsenElementaryIntroductionInformation2020, §4, p. 36].

## What survives reparametrisation

The chapter's goal was to say which objects are intrinsic, and the material now assembled gives a clean answer. The divergence itself is *not* intrinsic: $D(\theta:\theta')$ is a function of the chosen coordinates, and only its value on distributions, written $\bar D(p,q)$, has meaning [@nielsenElementaryIntroductionInformation2020, §3.6, Eq. 123, p. 27]. The Fisher metric is *covariant*, not invariant — it transforms with a Jacobian factor — but the induced *length element* $ds^2=g_{ij}d\theta^id\theta^j$, and hence the Fisher–Rao distance between two distributions, is intrinsic, provided the shortest path exists [@nielsenElementaryIntroductionInformation2020, Eqs. 86, 127, pp. 21, 28]. The $\alpha$-connections transform as connections, and their statistical content is the pair's flatness and the family they generate, not their coordinate symbols. And the natural-gradient *trajectory*, not the step vector, is the invariant object: the update is defined so that all parametrisations give the same sequence of distributions.

Taken together, these four facts are the reason the theory is worth the differential-geometric apparatus. A learning algorithm expressed through the Fisher metric is a function of the model family, not of how one happened to write it down — the same principle the collection meets again when a metric is pulled back through a decoder in [Riemannian and Finsler geometry of latent spaces](latent-geometry.md) [@arvanitidisPullingBackInformation2022, §3], and when the parameter-space symmetries of a network are catalogued in [Symmetry and geometric deep learning](symmetry-and-gdl.md) [@grigsbyHiddenSymmetriesReLU, §2].

## Beyond flatness: Frobenius structures and topological field theory

The companion volume takes the same apparatus in a different direction, and its destination is worth stating even though this chapter does not develop it. Its building blocks are *affine flat structures* — charts whose transition maps are affine — which are equivalent to flat, torsion-free connections [@combeExploringInformationGeometry2025, §9.2.1, Def. 9.1, p. 123]. A *pre-Frobenius manifold* adds a metric $g$ and a symmetric rank-three tensor $A$ together with a commutative multiplication $\circ$ on the tangent sheaf, required to be metric-invariant, $A(X,Y,Z)=g(X\circ Y,Z)=g(X,Y\circ Z)$ [@combeExploringInformationGeometry2025, §9.3.3, Def. 9.2, p. 127]. Imposing *potentiality*, $g(X\circ Y,Z)=(XYZ)\Phi$ for a local potential $\Phi$, and *associativity* of $\circ$ yields a *Frobenius manifold*, whose associativity is the WDVV system $\sum_{ef}\Phi_{abe}g^{ef}\Phi_{fcd}=\sum_{ef}\Phi_{bce}g^{ef}\Phi_{fad}$ [@combeExploringInformationGeometry2025, §9.3.4–9.4.1, Def. 9.3, Eqs. 9.3–9.4, pp. 127–128].

The bridge to statistics is that the Amari–Chentsov tensor can play the role of the multiplication. On a finite-dimensional statistical manifold of exponential type, define $\bar T^k_{ij}=\sum_mg^{km}T_{ijm}$ and set $u\circ v=\bar T(u,v)$; the result is metric-invariant, $g(u\circ v,w)=g(u,w\circ v)$, so a statistical manifold carries the ingredients of a pre-Frobenius manifold [@combeExploringInformationGeometry2025, §9.5, Thms. 9.6–9.7, pp. 130–131]. Whether it is a full Frobenius manifold is a separate question, and the book reduces it to a flatness condition on the *structure connection* $\nabla_{\lambda,X}(Y)=\nabla_{0,X}(Y)+\lambda(X\circ Y)$: by Manin's theorem the pencil is flat if and only if the pre-Frobenius manifold is Frobenius, and the two axioms reappear as the vanishing of the $\lambda^2$ and $\lambda$ curvature terms [@combeExploringInformationGeometry2025, §9.4.3, Thm. 9.5, p. 129]. The program connects on to topological field theory through *statistical Gromov–Witten invariants* $\tilde Y_n$, with the entropy potential $\tilde\Phi=\log\sum_\omega\exp(-\sum_j\beta_jX_j(\omega))$ playing the role of the WDVV potential [@combeExploringInformationGeometry2025, §11.1, Def. 11.1, Eqs. 11.1–11.2, Prop. 11.1, pp. 149–150].

!!! warning "How much of this is established"
    The Frobenius programme is presented as an invitation rather than a finished theory. Several load-bearing statements are posed as exercises instead of proved — the flatness characterisation is Exercise 9.3 and the curvature-decomposition theorem Exercise 9.4 — and the text stops just short of exhibiting a concrete exponential family that satisfies potentiality and the flatness of $\nabla_\lambda$ [@combeExploringInformationGeometry2025, §9.5, p. 131; Exercise 9.3, p. 129]. The book also has presentational defects a careful reader should know: the metric definition writes the triangle equality where the inequality is meant, and "Kullback–Liebler" appears for Kullback–Leibler [@combeExploringInformationGeometry2025, §Bibliography, p. 155]. Read chapter 9 as a map of an open programme, not as settled results.

A second thread in the same book reframes the classical constructions categorically. Its starting point is Klein's Erlangen programme read probabilistically: distributions are the "figures", and *Markov morphisms* — transition kernels — are the transformations that play the role of congruences [@combeExploringInformationGeometry2025, §7.2, Def. 7.1, p. 97]. Two families are *congruent* when they factor through a common intermediate space, $P^{(i)}_\theta=\int_E R_\theta(d\epsilon)Q^{(i)}(\epsilon,\cdot)$ [@combeExploringInformationGeometry2025, Thm. 7.1, p. 98]. The categorical fact is stability: being dominated is invariant under Markov morphisms, so dominated families and mutually absolutely continuous families form complete subcategories of the category of measurable spaces [@combeExploringInformationGeometry2025, Thms. 7.5–7.6, pp. 101–102]. This is the invariance that, in the classical language, is what forces the Fisher metric, told as a statement about morphisms rather than about embeddings [@nielsenElementaryIntroductionInformation2020, Thm. 8, p. 25]. A further construction identifies the space of distributions on a finite set with a manifold of projective *0-pairs* and decomposes it using *paracomplex* numbers $z=x+\epsilon y$ with $\epsilon^2=1$ and indefinite norm $x^2-y^2$ [@combeExploringInformationGeometry2025, Prop. 10.2, Thm. 10.2, p. 145; §10.2, p. 143]. These are pointers for the reader, not material on which the rest of this textbook depends.

## Summary

- A statistical manifold is a family of distributions seen as a smooth manifold carrying a metric and a pair of conjugate affine connections; the Levi–Civita connection is the self-dual midpoint of any such pair [@nielsenElementaryIntroductionInformation2020, §2.5, Property 3, pp. 11–13].
- The Fisher–Rao metric $g_{ij}(\theta)=\E[\partial_i\ell\,\partial_j\ell]$ is the unique metric invariant under Markov embeddings; its Hessian form is a log-partition function for exponential families [@nielsenElementaryIntroductionInformation2020, Thm. 8, p. 25; Eqs. 95, 100, pp. 23–24].
- A divergence induces the metric and the conjugate connections through the second and third derivatives at the diagonal; the Amari–Chentsov tensor is their asymmetry and is totally symmetric [@nielsenElementaryIntroductionInformation2020, Thm. 4, Eqs. 47–49, 56–58, pp. 13–15].
- Bregman divergences produce dually flat manifolds with dual Legendre coordinates, straight dual geodesics and asymmetric Pythagorean theorems; exponential and mixture families are the canonical examples [@nielsenElementaryIntroductionInformation2020, §3.7, Thm. 6, pp. 16, 18].
- The $\alpha$-connections interpolate between the exponential and mixture connections and are generated by scaling the cubic tensor; dual flatness is self-dual in $\alpha$ [@nielsenElementaryIntroductionInformation2020, §3.4, Thm. 2–3, Cor. 1–2, pp. 14–15].
- Statistical invariance singles out the Fisher metric and the separable $f$-divergences, and ties them together through $\alpha=2f'''(1)+3$ [@nielsenElementaryIntroductionInformation2020, §3.10, Eqs. 121–124, p. 27].
- Natural-gradient descent is Riemannian steepest descent in the Fisher metric and is reparametrisation-invariant; on a dually flat space it equals ordinary gradient descent in the dual coordinates, and it coincides with Bregman mirror descent [@nielsenElementaryIntroductionInformation2020, §4.1, Properties 5–7, pp. 39–40].
- A statistical manifold of exponential type carries a pre-Frobenius structure with multiplication $u\circ v=\bar T(u,v)$; whether it is Frobenius reduces to flatness of the structure connection, and this is an open programme rather than a settled theory [@combeExploringInformationGeometry2025, §9.5, pp. 130–131; Thm. 9.5, p. 129].

## Further reading

- [@nielsenElementaryIntroductionInformation2020]: the self-contained core; read §2 for metric versus connection, §3.2–3.7 for conjugate connections, divergences and dually flat geometry, §3.10 for the invariance results, and §4.1 for the natural gradient and its relation to mirror descent.
- [@combeExploringInformationGeometry2025]: the slow construction; read §6.7 for the score, Fisher metric and $\alpha$-connections, §7 for Markov morphisms and Klein geometry, §8.0.5 for canonical affine coordinates and Ceva lines, and §9 for the (pre-)Frobenius programme.
- [@arvanitidisPullingBackInformation2022]: the Fisher metric pulled back through a decoder, developed in [Riemannian and Finsler geometry of latent spaces](latent-geometry.md); the direct downstream use of this chapter's machinery.
- [@halversonMetricFlowsNeural2024]: treats the metric as a learned object evolving under a geometric functional rather than as the fixed Fisher metric; see [Metric flows with neural networks](metric-flows.md).

## Exercises

**Exercise 1** ★ (check). A metric and a connection are called independent structures on a manifold. State precisely what each adds, and say what extra conditions make the connection unique.

??? success "Solution"
    The metric $g$ gives an inner product on every tangent plane, hence lengths, angles
    and orthogonality, $g_{ij}=\langle e_i,e_j\rangle$ [@nielsenElementaryIntroductionInformation2020,
    §2.1–2.2, Eq. 4, pp. 3–5]. The connection $\nabla$ gives a covariant derivative,
    parallel transport and geodesics as autoparallel curves, fixed in coordinates by
    the Christoffel symbols $\Gamma^k_{ij}$ [@nielsenElementaryIntroductionInformation2020,
    §2.3.1–2.3.3, Eq. 14, pp. 6–8]. Imposing torsion-freeness and metric
    compatibility singles out the unique Levi–Civita connection built from $g$, whose
    Christoffel symbols are $\frac12g^{kl}(\partial_ig_{jl}+\partial_jg_{il}-\partial_lg_{ij})$
    [@nielsenElementaryIntroductionInformation2020, Thm. 1, Eq. 35, p. 11]. Dropping
    uniqueness and keeping a conjugate pair is what information geometry does.

**Exercise 2** ★ (check). Why is the Fisher information metric — rather than an arbitrary Riemannian metric — the one attached to a statistical model?

??? success "Solution"
    Because it is the only metric preserved by the statistical maps one is forced to
    allow. Under a Markov embedding (coarse-graining followed by embedding into a
    larger simplex) the invariant metric tensor is the Fisher metric, uniquely up to a
    scale constant [@nielsenElementaryIntroductionInformation2020, Thm. 8, p. 25].
    In the categorical language of [@combeExploringInformationGeometry2025] the same
    invariance is the statement that the property of being dominated is preserved by
    Markov morphisms [@combeExploringInformationGeometry2025, Thm. 7.5, p. 101]. A
    construction that must not depend on the sample space therefore has no other
    choice, and the metric also carries the estimation content of the Cramér–Rao
    bound [@nielsenElementaryIntroductionInformation2020, Eq. 93, p. 22].

**Exercise 3** ★★ (derive). Starting from the definition of a divergence, derive the metric and the two conjugate connections it induces, and identify the cubic tensor.

??? success "Solution"
    A divergence $D(\theta:\theta')$ is non-negative and vanishes only on the diagonal,
    with vanishing first derivatives there and negative-definite mixed second
    derivative [@nielsenElementaryIntroductionInformation2020, Def. 5, p. 15]. Setting
    $g_{ij}=-\partial_i\partial'_jD(\theta:\theta')|_{\theta=\theta'}$ gives the metric;
    the third derivatives at the diagonal split into two asymmetric pieces,
    $\Gamma_{ijk}=-\partial_i\partial_j\partial'_kD$ and
    $\Gamma^*_{ijk}=-\partial_k\partial_i\partial'_jD$, which are conjugate with respect
    to $g$ [@nielsenElementaryIntroductionInformation2020, Thm. 4, Eqs. 56–58, p. 15].
    Their difference $C_{ijk}=\Gamma^*_{ijk}-\Gamma_{ijk}$ is the totally symmetric
    Amari–Chentsov tensor; for the expected connections of a parametric family it is
    the third score moment $\E[\partial_i\ell\,\partial_j\ell\,\partial_k\ell]$
    [@nielsenElementaryIntroductionInformation2020, Eqs. 47–49, 103, pp. 14, 24].

**Exercise 4** ★★ (derive). In a dually flat space with potential $F$, derive the identity $^{\mathrm{NG}}\nabla_\theta L_\theta=\nabla_\eta L_\eta$ and explain what it says about the natural gradient.

??? success "Solution"
    Write $g_\theta(\theta)=\nabla^2F(\theta)$ and $\eta=\nabla_\theta F(\theta)$, so
    $\nabla_\theta\eta=\nabla^2F(\theta)=g_\theta$ [@nielsenElementaryIntroductionInformation2020,
    §4.1.3, p. 40]. The chain rule gives
    $\nabla_\theta L_\theta(\theta)=\nabla_\theta(L_\eta(\eta(\theta)))=(\nabla_\theta\eta)(\nabla_\eta L_\eta)$.
    Then $^{\mathrm{NG}}\nabla L_\theta=g_\theta^{-1}\nabla_\theta L_\theta=(\nabla_\theta\eta)^{-1}(\nabla_\theta\eta)\nabla_\eta L_\eta=\nabla_\eta L_\eta$
    [@nielsenElementaryIntroductionInformation2020, Property 7, Eqs. 208–211, p. 40].
    The geometric content is that the contravariant/covariant mismatch that spoils
    ordinary gradient descent is repaired by moving to the dual coordinates: on a
    dually flat manifold, natural-gradient descent on $L_\theta$ is exactly ordinary
    gradient descent on the dually parameterised loss $L_\eta$.

**Exercise 5** ★★ (compare). [@nielsenElementaryIntroductionInformation2020] and [@combeExploringInformationGeometry2025] both construct the Fisher metric and the $\alpha$-connections. What does each do with them, and which should a reader new to the field start from?

??? success "Solution"
    Nielsen treats them as the core of a compact dualistic construction building to
    the natural gradient and two dually flat applications, with the connection to
    neural-network geometry only name-checked [@nielsenElementaryIntroductionInformation2020,
    §4, p. 36]. Combe et al. build the same objects from topology and measure theory
    and then extend them toward Markov-morphism categories, canonical affine
    coordinates and Ceva lines, and the (pre-)Frobenius/WDVV programme
    [@combeExploringInformationGeometry2025, §8.0.5, pp. 109–110; §9, pp. 122–131].
    A newcomer should start from Nielsen: it is shorter, states the definitions and
    theorems in one place, and connects directly to the optimisation use
    [@nielsenElementaryIntroductionInformation2020, §1.2, p. 2]. Combe et al. is the
    modern companion for the algebraic and topological extensions, but its
    load-bearing Frobenius claims are posed as exercises rather than proved
    [@combeExploringInformationGeometry2025, Exercise 9.3, p. 129].

**Exercise 6** ★ (check). Which objects of this chapter survive a change of parametrisation, and which only transform covariantly? Support each with the relevant statement.

??? success "Solution"
    The divergence does not survive as a function of $\theta$; only its value on
    distributions is intrinsic [@nielsenElementaryIntroductionInformation2020, Eq. 123,
    p. 27]. The Fisher metric transforms covariantly, $I(\eta)=J^\top I(\theta(\eta))J$,
    but the length element $ds^2=g_{ij}d\theta^id\theta^j$ and the Fisher–Rao distance
    between two distributions are invariant [@nielsenElementaryIntroductionInformation2020,
    Eqs. 86, 127, pp. 21, 28]. The $\alpha$-connections transform as connections, and
    their statistical content is the dual flatness they carry
    [@nielsenElementaryIntroductionInformation2020, Cor. 1–2, p. 15]. The
    natural-gradient *trajectory* is invariant even though the step vector is not,
    since the update is defined so that every parametrisation gives the same sequence
    of distributions [@nielsenElementaryIntroductionInformation2020, Property 5,
    p. 39].

**Exercise 7** ★★★ (design). You train a probabilistic model whose parameters are constrained to the probability simplex, and you want a reparametrisation-invariant optimiser. Describe the natural-gradient update you would use, which coordinate chart you would work in, and one numerical hazard to guard against.

??? success "Solution"
    Use the natural-gradient update $\theta_{t+1}=\theta_t-\alpha_t g_\theta^{-1}\nabla_\theta L$,
    which is Riemannian steepest descent and is reparametrisation-invariant
    [@nielsenElementaryIntroductionInformation2020, Eqs. 201–203, Property 5, pp. 38–39].
    Because the constraint binds, either work in the canonical affine coordinates $s$
    of the simplex — where the density is exponential, the manifold is affine and the
    coordinate lines are Ceva lines with $\dot p_1=p_1(1-p_1)$ — and project back, or
    equivalently use the dual coordinates $\eta$ where the update is ordinary gradient
    descent [@combeExploringInformationGeometry2025, §8.0.5, Lemma 8.3, pp. 109–111;
    @nielsenElementaryIntroductionInformation2020, Property 7, p. 40]. The hazard is
    degeneracy of an estimated Fisher matrix: the empirical Fisher need not match the
    true structure and may be singular, so one regularises or exploits the known
    diagonal form [@nielsenElementaryIntroductionInformation2020, §4.1.4, pp. 40–41].

**Exercise 8** ★★★ (critique). The claim that a statistical manifold of exponential type is a Frobenius manifold is central to [@combeExploringInformationGeometry2025]. How well is it supported, and what would settle it?

??? success "Solution"
    The book establishes the necessary ingredients — a flat affine structure, the
    Fisher metric, the symmetric Amari–Chentsov tensor, and a metric-invariant
    multiplication $u\circ v=\bar T(u,v)$ — so that a statistical manifold carries a
    *pre*-Frobenius structure [@combeExploringInformationGeometry2025, §9.5,
    Thms. 9.6–9.7, pp. 130–131]. What it does not do is verify potentiality and the
    flatness of the structure connection on a concrete family: the flatness
    characterisation is set as Exercise 9.3 and the curvature decomposition as
    Exercise 9.4, and no worked example is carried through
    [@combeExploringInformationGeometry2025, Exercise 9.3, p. 129; Exercise 9.4, p. 135].
    Settling it would take a concrete exponential family — the book's own Gaussian
    running example, say — for which one computes the WDVV potential and checks that
    $\nabla_\lambda$ is flat, turning the pencil-flatness criterion of Manin's theorem
    into a verified instance rather than a template
    [@combeExploringInformationGeometry2025, Thm. 9.5, p. 129].
