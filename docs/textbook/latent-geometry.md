---
chapter: latent-geometry
title: Riemannian and Finsler geometry of latent spaces
outline_hash: 8bb3c420d703
built_from:
  arvanitidisGeometricallyEnrichedLatent2020: 2b5378b36bee
  arvanitidisPullingBackInformation2022: d5884ab455cc
  combeExploringInformationGeometry2025: 415b5fcd4364
  halversonMetricFlowsNeural2024: 699fded4f825
  meilaManifoldLearningWhat2023: 009a25e9883c
  nielsenElementaryIntroductionInformation2020: 84c18077962c
  pouplinIdentifyingLatentDistances2023: 002a4b56f697
---

# Riemannian and Finsler geometry of latent spaces

!!! abstract "In this chapter"
    - A latent space carries no geometry of its own: straight lines and Euclidean distances depend on the arbitrary parametrisation of the decoder. Pulling a metric back through the decoder gives the latent space a geometry whose lengths are intrinsically meaningful.
    - The metric that is pulled back need not be Euclidean: making it a designed Riemannian metric on the observation space is a way to put domain knowledge — a cost, a class separation, a semantic constraint — directly into the interpolation problem.
    - When the decoder is stochastic, the pullback tensor is random. Replacing it by its expectation gives a Riemannian norm but not the expected length; replacing the norm by its expectation gives a Finsler metric, which coincides with the Riemannian one at rate $O(1/D)$ in the data dimension.
    - A probabilistic decoder is a map into the parameter space of a likelihood, so the natural reference geometry is the Fisher–Rao metric of information geometry; because the Fisher–Rao metric is locally the KL divergence, shortest paths are found by minimising a sum of KL terms and no closed-form metric is needed.
    - Prerequisites: [Manifold learning and intrinsic dimension](manifold-learning.md), [Information geometry](information-geometry.md).

The previous chapters read a network's geometry from the *inside*: the polyhedral complex it induces on its input space, its linear regions, its folds. This chapter turns to the geometry of the *latent space* of a generative model, which is a different object. A generative model assumes the data lie near a low-dimensional manifold $\mathcal{M}\subset X$ of the observation space, and learns a decoder $\phi:Z\to X$ from a lower-dimensional latent space $Z$ that reproduces it [@meilaManifoldLearningWhat2023, §1]. The decoder is not unique, and neither is the latent space: any smooth invertible reparametrisation of $Z$, composed with the decoder, produces exactly the same distribution over observations [@arvanitidisGeometricallyEnrichedLatent2020, §1, p. 1]. Euclidean straight lines in $Z$ therefore have no invariant meaning, and interpolating along them is the source of the familiar artefacts of generative models — images that drift off the data manifold, mixtures that no plausible input produces [@arvanitidisPullingBackInformation2022, §1, p. 1].

The cure is to give $Z$ a metric, and specifically a metric whose curve lengths are measured *in the observation space*, so that a reparametrisation of $Z$ leaves them unchanged. Then the shortest path between two latent codes is a well-defined, dataset-dependent object: the curve that, when decoded, moves through the data with the least "excursion". This chapter develops that idea through the three papers of the collection that build it. [@arvanitidisGeometricallyEnrichedLatent2020] pulls a metric back through the decoder and, crucially, allows the *ambient* metric to encode domain knowledge; [@arvanitidisPullingBackInformation2022] replaces the observation space by the parameter space of the decoder's likelihood and pulls back the Fisher–Rao metric of information geometry; [@pouplinIdentifyingLatentDistances2023] asks what to do when the decoder is stochastic and the pullback tensor itself is a random field, and answers with a Finsler metric. I keep the shared [notation](../notation.md): $\phi$ is the decoder and $J_\phi(z)$ its Jacobian, $G(z)$ the latent (pullback) metric, $\gamma:[0,1]\to\mathcal{M}$ a path and $\ell(\gamma)$ its length, $g_{ij}$ the Fisher–Rao metric of a statistical manifold, and $F(x,\dot x)$ a Finsler norm with induced distance $d_F$. The papers' own symbols ($g$, $M(z)$, $M_X$, $\|\cdot\|_G$, $\|\cdot\|_R$, $\|\cdot\|_F$) are translated into these throughout; I say so once.

## Pulling a metric back through a decoder

Fix a smooth decoder $\phi:Z=\mathbb{R}^{d_Z}\to X=\mathbb{R}^D$ and let $J_\phi(z)\in\mathbb{R}^{D\times d_Z}$ be its Jacobian, which maps a tangent vector $v_z\in T_zZ$ to a tangent vector $J_\phi(z)v_z$ of the image manifold $\phi(Z)$ [@arvanitidisGeometricallyEnrichedLatent2020, §2.1, p. 3]. Suppose the ambient space $X$ carries a Riemannian metric $G_X:X\to\mathbb{R}^{D\times D}_{\succ 0}$, a smoothly varying positive-definite tensor. The ambient inner product of two tangent vectors is then $\langle v_x,v_x\rangle_x = v_x^\top G_X(x) v_x$, and substituting $v_x=J_\phi(z)v_z$ gives the inner product in the latent tangent space:

$$
\langle v_x, v_x\rangle_x = \big\langle v_z,\; J_\phi(z)^\top G_X(\phi(z))\, J_\phi(z)\, v_z\big\rangle = \langle v_z, G(z)v_z\rangle,
$$

so that the latent metric is the *pullback*

$$
G(z) = J_\phi(z)^\top G_X(\phi(z))\, J_\phi(z).
$$

This is the construction of [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 3, §2.1, p. 3]; when $G_X$ is the Euclidean metric it reduces to the classical deterministic pullback $G(z)=J_\phi(z)^\top J_\phi(z)$ [@arvanitidisPullingBackInformation2022, Eq. 1, p. 2]. The map is an *immersion* when $J_\phi$ has full rank everywhere; then $G(z)$ is positive definite and $(Z,G)$ is a Riemannian manifold [@arvanitidisGeometricallyEnrichedLatent2020, footnote 1, p. 3].

With a metric in hand, the standard Riemannian functionals apply. The length of a curve and its energy are

$$
\ell(\gamma)=\int_0^1 \sqrt{\dot\gamma(t)^\top G(\gamma(t))\,\dot\gamma(t)}\,\mathrm{d}t,
\qquad
E(\gamma)=\int_0^1 \dot\gamma(t)^\top G(\gamma(t))\,\dot\gamma(t)\,\mathrm{d}t,
$$

and the curves that minimise the length are the geodesics [@arvanitidisPullingBackInformation2022, Eq. 1–2, p. 2]. Because the integrand of $E$ is quadratic and the endpoints are fixed, stationarity under variation of $\gamma$ is the Euler–Lagrange equation, a second-order ODE in the metric and its first derivatives; solving the boundary-value problem returns the geodesic [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 13, App. A]. In practice this ODE is replaced by a discretisation: the curve is written as a cubic spline with fixed endpoints, its control points are optimised on the energy by automatic differentiation, and the optimisation is initialised from a graph shortest path over a latent grid [@arvanitidisPullingBackInformation2022, §3.2, p. 4].

The essential point is invariance. The length of $\gamma$ in $Z$ equals the length of the decoded curve $\phi\circ\gamma$ on the image manifold, measured with $G_X$ [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 2–3, §2, p. 3]. A smooth invertible change of latent coordinates therefore leaves the length — and hence the geodesics, as unparametrised curves — unchanged. This is exactly the identifiability that the Euclidean latent space lacks, and it is why the pullback metric is the standard cure for the arbitrariness of the latent representation [@arvanitidisPullingBackInformation2022, §3.1, p. 4].

!!! note "Background"
    A *Riemannian metric* on a manifold is a smooth assignment of a positive-definite inner product to each tangent space; it turns each tangent space into a Euclidean space and makes lengths of curves well defined. A *geodesic* is a locally length-minimising curve; its distance $d(x,y)$ is the infimum of the lengths of curves joining $x$ and $y$. A *Finsler metric* relaxes the inner product to a norm: it is a function $F(x,v)$ on the tangent bundle that is positively homogeneous, smooth and strongly convex, but not necessarily bilinear, so it need not come from any inner product [@pouplinIdentifyingLatentDistances2023, Def. 2.6, p. 6]. Every Riemannian metric is Finsler; the converse fails.

## The ambient metric as a place to put knowledge

The pullback of the *Euclidean* metric measures the shape of the learned manifold and nothing else, and [@arvanitidisGeometricallyEnrichedLatent2020] observes that this is a missed opportunity: latent distances are then identifiable but not necessarily useful [@arvanitidisGeometricallyEnrichedLatent2020, §1, p. 1]. Its proposal is to let $X$ itself be a Riemannian manifold whose metric is designed, so that the ambient geometry carries high-level information. Three constructions are given. A metric can be a smooth convex combination of local metrics, $G_X(x)=\sum_k w_k(x) M_k$ with Gaussian weights, each $M_k$ a local linear discriminant or locally learned metric [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 4, §3.1, p. 3]; or it can be the inverse of the local diagonal covariance of the data, a non-parametric unsupervised choice [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 5, §3.1, p. 4]; or it can be the single diagonal form

$$
G_X(x) = (\alpha\, h(x) + \varepsilon)^{-1} I_D,
$$

with $h(x)\to 1$ near the data manifold and $h(x)\to 0$ away from it, $h$ a positive radial-basis-function network, and $\alpha,\varepsilon>0$ scaling factors [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 6, §3.1, p. 4]. Because the metric magnitude is small where $h$ is large, shortest paths under this metric are pulled towards the data [@arvanitidisGeometricallyEnrichedLatent2020, §3.1, p. 4].

The same equation, read without the inversion, gives the *cost* version: a function $h(x)$ that is large where one wants paths to be repelled. This is the paper's semantic control. A pre-trained Progressive GAN on CelebA has its ambient metric built from a classifier for blond hair, so that the shortest path in latent space avoids the region that decodes to blonds; the plain shortest path, which only follows the learned manifold, passes through it, and the straight line does so even more [@arvanitidisGeometricallyEnrichedLatent2020, §5.1, p. 7, Fig. 7]. In a VAE on MNIST digits $0,1,3,2$, an ambient metric that merges $0,1,3$ into one LDA class steers the path around the digit-$2$ region, and adding a local-covariance term pulls it towards chosen points [@arvanitidisGeometricallyEnrichedLatent2020, §5.2, p. 8, Fig. 8]. The ambient metric is thus a modelling interface: any sufficiently smooth positive-definite field the designer can write down is a candidate, and the induced latent geometry respects it.

## Stochastic decoders: the expected metric and its relaxation

A variational autoencoder does not decode to a point. Its generator is $g(z)=\mu(z)+\mathrm{diag}(\varepsilon)\,\sigma(z)$ with $\varepsilon\sim\mathcal{N}(0,I_D)$, so the direct application of the pullback rule gives a *random* metric: the tensor $G_\varepsilon(z)$ depends on the noise draw [@arvanitidisGeometricallyEnrichedLatent2020, §4, p. 5]. Holding $\varepsilon$ fixed across $Z$ to keep the map smooth, the expectation under the noise has a clean form and is the metric used in the earlier latent-space literature:

$$
G(z) = \mathbb{E}_{p(\varepsilon)}\big[J_{g_\varepsilon}(z)^\top J_{g_\varepsilon}(z)\big] = J_\mu(z)^\top J_\mu(z) + J_\sigma(z)^\top J_\sigma(z).
$$

The second term is exactly the decoder's uncertainty, so distances grow where the model is unsure of itself, and shortest paths avoid regions with no latent codes [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 7, §3.2, p. 5]. This is attractive only if the uncertainty is meaningful, which a naively trained variance network is not; modelling the inverse variance $\beta(z)=(\sigma^2(z))^{-1}$ with a positive RBF network makes uncertainty grow as one moves away from the latent codes [@arvanitidisGeometricallyEnrichedLatent2020, §3.2, p. 5].

Pulling back a designed ambient metric $G_X$ through a stochastic decoder gives the random tensor $G_\varepsilon(z)=[J_\mu+J_\sigma\varepsilon]^\top G_X(\mu+\mathrm{diag}(\varepsilon)\sigma)[J_\mu+J_\sigma\varepsilon]$ [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 8, §4, p. 5]. Evaluating the true expectation inside a geodesic solver is prohibitively costly, so the paper relaxes it by evaluating the ambient metric only at the mean, giving the working definition

$$
G(z) \triangleq J_\mu(z)^\top G_X(\mu(z))\, J_\mu(z) + J_\sigma(z)^\top G_X(\mu(z))\, J_\sigma(z).
$$

The justification is a scale-separation argument: near the latent codes $\sigma(z)\to 0$, so the first term dominates and the ambient metric is evaluated where it matters; far from the codes $J_\sigma$ dominates, so the precise form of $G_X$ matters less [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 9 and App. D, §4, p. 6]. This relaxation is a heuristic, checked empirically rather than derived, and the paper says so [@arvanitidisGeometricallyEnrichedLatent2020, §4, App. D, p. 5].

Deterministic decoders — an autoencoder or a GAN — have no second term, so by construction they cannot capture the structure of $\mathcal{M}$ in $Z$ [@arvanitidisGeometricallyEnrichedLatent2020, §3.2, p. 5]. The remedy is to combine a cheap ambient metric with a decoder that *extrapolates meaningfully*: moving outside the prior support should move the decoded manifold away from the data, where the data-support-based ambient metric is large. The proposed architecture is a residual generator

$$
g(z) = f(z) + U\,\mathrm{diag}\big([\sqrt{\lambda_1},\dots,\sqrt{\lambda_d}]\big) z + b,
$$

with $f$ a deep network, $U$ the top $d$ eigenvectors and $\lambda_k$ the eigenvalues of the empirical data covariance, and $b$ the data mean — a probabilistic-PCA-like linear part that dominates far from the prior support and makes the extrapolation predictable [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 10, §4, p. 6]. The immersion condition is still not guaranteed: $J_f$ and the linear map could in principle cancel, and the paper leaves a proper analysis to future work [@arvanitidisGeometricallyEnrichedLatent2020, App. B, p. 13].

## Information geometry: pulling back Fisher–Rao

The constructions so far share a limitation, which [@arvanitidisPullingBackInformation2022] names directly: they pull the Euclidean metric from the *observation* space through the reparametrisation of a Gaussian decoder, and that route "is intrinsically linked with the simple reparametrization of the Gaussian" and covers only location-scale distributions [@arvanitidisPullingBackInformation2022, §2.1, p. 3]. Categorical, Gamma, Beta, Dirichlet and von Mises–Fisher decoders are left without a principled geometry.

The reframing is to notice that the codomain of a probabilistic decoder is *already* the parameter space $H$ of a likelihood: writing the decoder as a map $h:Z\to H$ gives $p(x\mid z)=p(x\mid h(z))$, so the model spans a deterministic manifold $h(Z)$ inside $H$ rather than a random manifold inside $X$ [@arvanitidisPullingBackInformation2022, §3, p. 3]. Equipping $H$ with the Fisher–Rao metric $I_H(\eta)=\int_X[\nabla_\eta\log p(x\mid\eta)\nabla_\eta\log p(x\mid\eta)^\top]\,p(x\mid\eta)\,\mathrm{d}x$ makes it a *statistical manifold* — a modelling decision tied to the likelihood, not to the data [@arvanitidisPullingBackInformation2022, Eq. 7 and Def. 3.1, §3, p. 3]. Pulling that metric back along $h$ gives the latent geometry

$$
G(z) = J_h(z)^\top I_H(h(z))\, J_h(z),
$$

and the paper proves that this equals the Fisher–Rao metric computed with the latent variable $z$ itself as the parameter [@arvanitidisPullingBackInformation2022, Prop. 3.2 and Prop. 3.3, §3.1, p. 4]. Lengths are now measured in the space of distributions, so the construction is invariant to reparametrisations of $Z$ and inherits the identifiability protection of the classical pullback [@arvanitidisPullingBackInformation2022, §3.1, p. 4].

The reason this becomes an algorithm is the local coincidence of the Fisher–Rao metric and the KL divergence,

$$
\mathrm{KL}\big(p(x\mid\eta),\,p(x\mid\eta+\delta\eta)\big) = \tfrac{1}{2}\,\delta\eta^\top I_H(\eta)\,\delta\eta + o(\delta\eta^2),
$$

which lets the curve energy be written as a sum over neighbouring decoded distributions,

$$
E(\gamma) \propto \lim_{N\to\infty}\sum_{n=1}^{N-1} \mathrm{KL}\big(p(x\mid\gamma(t_n)),\,p(x\mid\gamma(t_{n+1}))\big),\qquad t_n=\tfrac{n}{N}.
$$

Minimising this needs nothing beyond a differentiable KL, which is why the authors call the result *black-box random geometry*: when no closed form exists, Monte Carlo estimates through the reparameterisation trick supply gradients [@arvanitidisPullingBackInformation2022, Eq. 8, Eq. 10, and §3.4, p. 5]. If the tensor itself is wanted — for an exponential map, or to fit a distribution on the manifold — it can be recovered from KL values at perturbed points, $G_{ii}\approx 2\,\mathrm{KL}_z(\delta e_i)/\varepsilon^2$ for the diagonal and $\big(\mathrm{KL}_z(\delta e_i+\delta e_j)-\mathrm{KL}_z(\delta e_i)-\mathrm{KL}_z(\delta e_j)\big)/\varepsilon^2$ off it [@arvanitidisPullingBackInformation2022, Prop. 3.4, §3.4, p. 5]. The categorical case is worked out in full: the Fisher–Rao metric on the simplex is $\mathrm{diag}(1/\eta_1,\dots,1/\eta_D)$ and its distance is the spherical distance $\arccos(\sqrt{\eta}^\top\sqrt{\eta'})$ between elementwise square roots [@arvanitidisPullingBackInformation2022, Eq. 11–12, §3.3, p. 5].

A shortest path computed in $H$ can still leave the data support and decode to nonsense, so the decoder has to be designed to extrapolate to *high uncertainty* away from the training codes. The mechanism is a KMeans clustering of the training codes, a distance $D(z)=\min_j\|z-c_j\|$, and a translated sigmoid $\tilde\sigma_\beta(D(z))$ that blends the learned parameters towards a maximum-uncertainty distribution: variance to infinity for a Gaussian, probability to $1/2$ for a Bernoulli, concentration to zero for a von Mises–Fisher [@arvanitidisPullingBackInformation2022, §3.1, p. 4 and Sec. C.2, p. 17]. The experiments confirm that this regularisation, not the choice of metric, is what makes the geometry sensible [@arvanitidisPullingBackInformation2022, §4.1, p. 6].

The evidence is broad but qualitative. Four VAEs on an MNIST subset of the single digit $1$ show the Fisher–Rao pullback "on par" with the Euclidean pullback and both failing without uncertainty regularisation [@arvanitidisPullingBackInformation2022, §4.1, pp. 5–6, Fig. 4]. Noisy circular data in $Z_{\text{toy}}=\mathbb{R}^2$, decoded by untrained MLPs and given metrics pulled from multivariate Normal, Bernoulli, Beta, Dirichlet and Exponential distributions, recover the true circular structure, except for Bernoulli, where some paths fail to converge [@arvanitidisPullingBackInformation2022, §4.2, pp. 6–7, Fig. 5]. A 26-bone CMU motion-capture skeleton, decoded as a product of 26 von Mises–Fisher distributions with a Monte-Carlo KL, has a shortest path that follows the periodicity of walking where the Euclidean straight line decodes to "uncharted territory" [@arvanitidisPullingBackInformation2022, §4.3, p. 7, Fig. 6]. On MovieLens 25M, a Bernoulli VAE's pulled-back path stays among locally similar users while linear interpolation does not [@arvanitidisPullingBackInformation2022, §4.6, p. 8]. An assessment of the approximation itself finds the average error below $\varepsilon^2$ inside and outside the support, growing at the support boundary, with some approximated metrics having negative determinant [@arvanitidisPullingBackInformation2022, §4.4, p. 8, Fig. 7].

## Random manifolds and the Finsler replacement

The expectation $\mathbb{E}[G]$ used above is an approximation chosen for convenience, and [@pouplinIdentifyingLatentDistances2023] asks what it actually approximates. When the decoder $f$ is a *stochastic immersion* — a smooth stochastic process, as in a Gaussian-process latent variable model — the metric tensor $G=J^\top J$ is a matrix-valued random field, so the induced norm $\|u\|_G=\sqrt{u^\top G u}$, and hence every curve length, is a random variable; the manifold is called a *random manifold* [@pouplinIdentifyingLatentDistances2023, Def. 2.1–2.3, §2.1, p. 4–5]. Two deterministic norms can be built from it. The *expected Riemannian* norm uses the expected tensor,

$$
\|u\|_R = \sqrt{u^\top \mathbb{E}[G]\,u},
$$

while the *Finsler* norm takes the expectation of the norm itself,

$$
\|u\|_F = \mathbb{E}\!\left[\sqrt{u^\top G u}\right].
$$

The key observation, due to Eklund and Hauberg and recalled here, is that these are not the same construction: the length computed from the expected metric is not the expected length, and the two energies differ exactly by a variance term,

$$
E_R(\gamma) - E_F(\gamma) = \int_0^1 \mathrm{Var}\!\left[\|\dot\gamma(t)\|_G\right]\mathrm{d}t.
$$

So the Riemannian energy penalises curves that pass through high-variance regions, while the Finsler energy does not [@pouplinIdentifyingLatentDistances2023, §2.3, p. 6].

The central theoretical result is that $\|\cdot\|_F$ is a genuine Finsler metric and *cannot* be Riemannian. It satisfies positive homogeneity, smoothness and strong convexity, as Finsler metrics require, but it is not bilinear, so no inner product induces it; stochastic manifolds are therefore "inherently more complex" than deterministic ones [@pouplinIdentifyingLatentDistances2023, Prop. 2.2, §2.4, p. 7]. For a Gaussian-process decoder, $v^\top G v$ follows a one-dimensional non-central Wishart distribution, whose square root is non-central Nakagami, and the expectation is known in closed form:

$$
\|v\|_F = \sqrt{2}\,\sqrt{v^\top \Sigma v}\;\frac{\Gamma\!\left(\frac{D}{2}+\frac{1}{2}\right)}{\Gamma\!\left(\frac{D}{2}\right)}\;{}_1F_1\!\left(-\frac{1}{2}, \frac{D}{2}, -\frac{\omega}{2}\right),
\qquad \omega = (v^\top\Sigma v)^{-1}\,(v^\top\mathbb{E}[J]^\top\mathbb{E}[J]\,v),
$$

with $\Sigma$ the covariance of the Jacobian and ${}_1F_1$ the confluent hypergeometric function [@pouplinIdentifyingLatentDistances2023, Prop. 2.3, §2.4, p. 7]. This closed form is what makes the comparison quantitative rather than formal.

The comparison has three layers. The *absolute* bounds sandwich the Finsler norm between two Riemannian norms, $\|v\|_{\alpha\Sigma}\le\|v\|_F\le\|v\|_R$ with $\alpha=2\big(\Gamma(\tfrac{D}{2}+\tfrac12)/\Gamma(\tfrac{D}{2})\big)^2$; the upper bound is Jensen's inequality for the concave square root, and the lower bound uses the closed form, and both are inherited by length, energy and the Busemann–Hausdorff volume [@pouplinIdentifyingLatentDistances2023, Prop. 3.1 and Cor. 3.1, §3.2, p. 9]. The *relative* bound for a general stochastic immersion is $0\le(\|v\|_R-\|v\|_F)/\|v\|_R\le \mathrm{Var}[\|v\|_G^2]/(2\,\mathbb{E}[\|v\|_G^2]^2)$, so the two norms coincide exactly when the metric is deterministic, and the Gaussian-process specialisation $1/(D+\omega)+\omega/(D+\omega)^2$ makes the dependence on the data dimension explicit [@pouplinIdentifyingLatentDistances2023, Prop. 3.2–3.3, §3.3, p. 9–10]. The *asymptotic* result is that on a bounded latent manifold the relative differences of length and energy are $O(1/D)$, so in high dimension the Riemannian approximation is asymptotically correct [@pouplinIdentifyingLatentDistances2023, Cor. 3.3, §3.4, p. 11].

!!! question "Open question"
    The printed volume bound in [@pouplinIdentifyingLatentDistances2023, Cor. 3.3, p. 11] is $O(\sqrt{D})$, while the length and energy bounds in the same corollary are $O(1/D)$ and the paper's conclusion is that all functionals converge. A quantity whose ratio grows like $\sqrt{D}$ is not obviously convergent, so the volume case is worth checking against its proof (the hyperspherical-coordinate argument of §B.2) before quoting it. The collection does not resolve this.

The experiments locate where the distinction matters. On synthetic pinwheel and concentric-circle data mapped to a sphere, the Riemannian geodesics avoid high-variance regions and take longer paths in the data space, whereas the Finsler geodesics are "not perturbed by the variance term" and take shorter ones [@pouplinIdentifyingLatentDistances2023, §4, p. 12, Fig. 5]. On data where the posterior variance is low — the font dataset ($D=256$), qPCR single cells ($D=48$), MNIST and FashionMNIST — the two geodesic families agree [@pouplinIdentifyingLatentDistances2023, §4, pp. 13–14, Fig. 6–7]. The practical conclusion is deliberately modest: the expected-metric heuristic that the latent-geometry literature had been using is justified, and the genuinely Finslerian regime is the low-dimensional, high-variance one. The theory has a sharp scope restriction, though: it needs a smooth stochastic process, which excludes VAE decoders, whose posterior independence makes the covariance a Dirac delta [@pouplinIdentifyingLatentDistances2023, footnote 1, p. 4]. So the three papers cover different decoders — GANs and VAEs for the first, any differentiable likelihood for the second, Gaussian processes for the third — and no single construction covers all three.

## What the geometry buys: interpolation, sampling and clustering

It is worth collecting what changes once a latent space has a metric, since the three papers put it to work differently.

For **interpolation**, the payoff is a path that follows the data. Under a pulled-back metric the shortest path in latent space decodes to a gradual, on-manifold sequence, while the Euclidean straight line takes off-manifold "shortcuts" that the metric's length penalises [@arvanitidisGeometricallyEnrichedLatent2020, §5.1, p. 7, Fig. 5; @arvanitidisPullingBackInformation2022, §4.3, p. 7]. With a designed ambient metric the path can additionally be biased towards or away from semantic regions, which is the sense in which the geometry controls interpolation [@arvanitidisGeometricallyEnrichedLatent2020, §5.1, p. 7].

For **sampling and clustering**, the determinant of the metric is a density-like quantity. [@arvanitidisGeometricallyEnrichedLatent2020] defines a latent density $q(z)\propto \tilde p_r(z)(1+\sqrt{|G(z)|})^{-1}$, which is high exactly where the metric is small, i.e. near the data; drawing from it aligns samples with the data manifold and removes the "ghostly" samples in the gaps between disconnected components, and the samples then serve to fit a mixture of locally adaptive normal distributions without a conditional GAN [@arvanitidisGeometricallyEnrichedLatent2020, §5.1, p. 6, Fig. 4]. Kernel density estimation in $Z$ under a combined LDA and cost metric separates the digits and reduces density near high-cost regions [@arvanitidisGeometricallyEnrichedLatent2020, §5.2, p. 8, Fig. 9]. On the information-geometric side, the recovered metric is what supports statistics *on* the manifold: a locally adaptive normal distribution is fitted with the pulled-back metric, and because the shortest paths follow the data, so does the density [@arvanitidisPullingBackInformation2022, §4.5, p. 8].

For **distances**, the metric decides what "close" means. The Finsler norm measures distance as an average over realisations, so it is the expected path length a user of the model would see; the Riemannian norm measures a path in a single averaged geometry, so it is the cheaper and, in high dimension, asymptotically equivalent proxy [@pouplinIdentifyingLatentDistances2023, §2.3–2.4, pp. 6–7; Cor. 3.3, p. 11]. Choosing between them is a question about how much the decoder's variance varies over the region of interest: negligible on MNIST, decisive on low-dimensional synthetic data with a heterogeneous posterior [@pouplinIdentifyingLatentDistances2023, §4, pp. 12–14].

!!! tip "Practical advice from the papers"
    Design the decoder so that its uncertainty grows away from the data before worrying about which metric to pull back. In [@arvanitidisPullingBackInformation2022, §4.1, p. 6] the choice between Euclidean and Fisher–Rao geometry changes little, while switching uncertainty regularisation on or off changes the result qualitatively; [@arvanitidisGeometricallyEnrichedLatent2020, §3.2, p. 5] reaches the same conclusion for the Gaussian-decoder case. Use the Riemannian expected metric unless the posterior variance is large and heterogeneous across the latent space, in which case the Finsler norm is the honest object [@pouplinIdentifyingLatentDistances2023, §4, pp. 12–14].

## Summary

- Latent space geometry is defined by pulling a metric back through the decoder, $G(z)=J_\phi(z)^\top G_X(\phi(z))J_\phi(z)$, which makes curve lengths invariant to reparametrisation of $Z$ and removes the identifiability problem [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 3, §2.1, p. 3].
- Shortest paths are geodesics of that metric, computed in practice by minimising a spline's energy with automatic differentiation rather than solving the Euler–Lagrange ODE [@arvanitidisPullingBackInformation2022, Eq. 1–2, §3.2, p. 4].
- The ambient metric is an interface for domain knowledge: a data-support-based RBF metric pulls paths onto the manifold, while a cost-based metric lets a classifier's output repel them, so interpolation can avoid a semantic region such as "blond" [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 6 and §5.1, p. 7].
- A stochastic decoder's pullback is random; fixing $\varepsilon$ gives the expected metric $J_\mu^\top J_\mu+J_\sigma^\top J_\sigma$, and evaluating the ambient metric at the mean gives the cheap relaxation $J_\mu^\top G_X(\mu)J_\mu+J_\sigma^\top G_X(\mu)J_\sigma$ [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 7–9, §3.2–4, pp. 5–6].
- A probabilistic decoder is a map into the parameter space of its likelihood, so equipping that space with the Fisher–Rao metric and pulling it back gives a distribution-agnostic latent geometry, computable from a differentiable KL alone [@arvanitidisPullingBackInformation2022, Prop. 3.2 and Eq. 10, p. 4].
- When the metric is a random field, the expected tensor gives a Riemannian norm but not the expected length; the expected norm is instead a Finsler metric, with the two differing by the variance of the length [@pouplinIdentifyingLatentDistances2023, Def. 2.4–2.5 and §2.3, pp. 5–6].
- The Finsler norm is sandwiched between two Riemannian norms, and for Gaussian processes it converges to the expected-metric norm at rate $O(1/D)$ in the data dimension — a justification for the widely used expected-metric approximation [@pouplinIdentifyingLatentDistances2023, Prop. 3.1 and Cor. 3.3, pp. 9–11].
- Uncertainty regularisation of the decoder, not the choice of metric, is the decisive ingredient: without a decoder that extrapolates to high uncertainty, neither the Euclidean nor the Fisher–Rao pullback yields a sensible geometry [@arvanitidisPullingBackInformation2022, §4.1, p. 6].

## Further reading

- [@arvanitidisGeometricallyEnrichedLatent2020]: the ambient-metric idea and the meaningful-extrapolation architecture; read §2.1 for the pullback, §3.1 for the three ambient metrics, §4 for the stochastic relaxation Eq. 9 and the linear-extrapolation generator Eq. 10, §5 for the sampling and CelebA demonstrations.
- [@arvanitidisPullingBackInformation2022]: the information-geometric route and the black-box algorithm; read §3.1–3.2 for the Fisher–Rao pullback and the KL-sum energy, §3.3 for the categorical closed form, §3.4 for the metric recovery from KL, §4.3 for the von Mises–Fisher motion-capture experiment.
- [@pouplinIdentifyingLatentDistances2023]: the random-manifold theory and the Finsler alternative; read §2.4 for the Finsler axioms and the closed form, §3.2–3.4 for the bounds and the high-dimensional convergence, §4 for where the two geodesic families differ.
- [@nielsenElementaryIntroductionInformation2020]: the information geometry underlying the Fisher–Rao construction; read §2–§3 for the statistical manifold and the metric.
- [@combeExploringInformationGeometry2025]: another entry to information geometry in this collection, for the wider context of the Fisher–Rao metric and its divergences.
- [@meilaManifoldLearningWhat2023]: the manifold hypothesis and intrinsic-dimension estimation that the whole chapter takes for granted.
- [@halversonMetricFlowsNeural2024]: the next chapter, where a metric evolves under a geometric functional rather than being pulled back once.

## Exercises

**Exercise 1** ★ (check). Explain why Euclidean distances in a latent space are not identifiable, and what exactly the pullback metric $G(z)=J_\phi(z)^\top G_X(\phi(z))J_\phi(z)$ changes about that.

??? success "Solution"
    A latent variable model only determines the distribution over observations: for any
    smooth invertible reparametrisation $g$ of $Z$, the decoder $\phi\circ g^{-1}$ produces
    the same density, so two different latent coordinates are equally valid descriptions of
    the same model [@arvanitidisGeometricallyEnrichedLatent2020, §1, p. 1]. Euclidean
    straight lines are not preserved by such a $g$, so they have no invariant meaning
    [@pouplinIdentifyingLatentDistances2023, §1, p. 1]. The pullback metric measures lengths
    of latent curves in the observation space through the decoder Jacobian
    [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 3, §2.1, p. 3], and a change of latent
    coordinates cancels against the Jacobian of the change, leaving each curve's length
    unchanged; the geodesics are therefore intrinsic to the model, not to the chart
    [@arvanitidisPullingBackInformation2022, §3.1, p. 4].

**Exercise 2** ★★ (derive). Starting from the length functional $\ell(\gamma)=\int_0^1\sqrt{\dot\gamma^\top G(\gamma)\dot\gamma}\,\mathrm{d}t$, show that its minimisers are also minimisers of the energy $E(\gamma)=\int_0^1\dot\gamma^\top G(\gamma)\dot\gamma\,\mathrm{d}t$, and say why this is the functional one actually optimises.

??? success "Solution"
    The length is the energy's square root integrand, and for a fixed parametrisation the
    two integrands differ by the factor $1/\sqrt{\dot\gamma^\top G\dot\gamma}$, so a
    length-critical curve is energy-critical; conversely a length-minimising curve can be
    reparametrised to have constant speed, after which length and energy have the same
    minimisers, up to a factor $2$ [@arvanitidisPullingBackInformation2022, Eq. 1–2, p. 2].
    The energy is the functional of choice because it is quadratic in $\dot\gamma$ and hence
    has no square root to differentiate; its stationarity condition is the Euler–Lagrange
    equation, a second-order ODE in $G$ and its first derivatives
    [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 13, App. A]. In practice the ODE is
    replaced by discretising the curve as a cubic spline and minimising the energy by
    automatic differentiation [@arvanitidisPullingBackInformation2022, §3.2, p. 4].

**Exercise 3** ★★ (compare). [@arvanitidisGeometricallyEnrichedLatent2020] and [@arvanitidisPullingBackInformation2022] both handle stochastic decoders. State precisely which decoders each covers and what each gives up.

??? success "Solution"
    [@arvanitidisGeometricallyEnrichedLatent2020] pulls the ambient metric through the
    Gaussian reparametrisation $g(z)=\mu(z)+\mathrm{diag}(\varepsilon)\sigma(z)$, so its
    expected metric $J_\mu^\top J_\mu+J_\sigma^\top J_\sigma$ and its relaxation
    $J_\mu^\top G_X(\mu)J_\mu+J_\sigma^\top G_X(\mu)J_\sigma$ exist only for location-scale
    distributions [@arvanitidisGeometricallyEnrichedLatent2020, Eq. 7–9, pp. 5–6]. It gives
    up generality, and the relaxation itself is argued rather than derived, with a
    scale-separation heuristic and empirical checks [@arvanitidisGeometricallyEnrichedLatent2020,
    §4, App. D, p. 5]. [@arvanitidisPullingBackInformation2022] instead views the decoder as
    a map $h$ into the parameter space of an arbitrary likelihood and pulls back the
    Fisher–Rao metric, which needs only a differentiable KL and covers Gaussian, Bernoulli,
    Beta, Dirichlet and von Mises–Fisher decoders [@arvanitidisPullingBackInformation2022,
    §3, §3.4, p. 5]. What it gives up is convenience: the metric tensor is usually not
    written in closed form and must be recovered from KL evaluations, at the cost of two
    decoder passes per diagonal and six per off-diagonal entry [@arvanitidisPullingBackInformation2022,
    Prop. 3.4, p. 5]. Both rely on the decoder extrapolating to high uncertainty
    [@arvanitidisGeometricallyEnrichedLatent2020, §3.2, p. 5; @arvanitidisPullingBackInformation2022,
    §4.1, p. 6]. Pick the first when the likelihood is Gaussian and cost matters less than
    a closed form; pick the second when the data type is not location-scale.

**Exercise 4** ★★ (compare). [@arvanitidisPullingBackInformation2022] and
[@pouplinIdentifyingLatentDistances2023] both deal with a random metric tensor. Explain the
two different deterministic norms each uses and why they are not the same object.

??? success "Solution"
    [@arvanitidisPullingBackInformation2022] avoids the randomness by measuring distances in
    the parameter space $H$ of the likelihood, where the geometry is the deterministic
    Fisher–Rao metric; the shortened paths are computed as sums of KL divergences between
    neighbouring distributions [@arvanitidisPullingBackInformation2022, Eq. 10, §3.2, p. 4].
    [@pouplinIdentifyingLatentDistances2023] keeps the random metric $G=J^\top J$ but
    replaces it by a deterministic norm. Its Riemannian norm uses the expected tensor,
    $\|u\|_R=\sqrt{u^\top\mathbb{E}[G]u}$, while its Finsler norm takes the expectation of
    the norm, $\|u\|_F=\mathbb{E}[\sqrt{u^\top G u}]$ [@pouplinIdentifyingLatentDistances2023,
    Def. 2.4–2.5, pp. 5–6]. These are different because the square root is not linear: the
    length from the expected metric is not the expected length, and the two energies differ
    by $\int_0^1\mathrm{Var}[\|\dot\gamma\|_G]\,\mathrm{d}t$ [@pouplinIdentifyingLatentDistances2023,
    §2.3, p. 6]. The Fisher–Rao construction is a third option again: take the expectation in
    distribution space rather than in the data space
    [@pouplinIdentifyingLatentDistances2023, §1.2, p. 4].

**Exercise 5** ★ (check). Why is the expected norm $\|u\|_F=\mathbb{E}[\sqrt{u^\top G u}]$ a Finsler metric and not a Riemannian one, and what does that say about stochastic manifolds?

??? success "Solution"
    A Finsler metric only requires positive homogeneity, smoothness and strong convexity of
    the norm on each tangent space [@pouplinIdentifyingLatentDistances2023, Def. 2.6, p. 6].
    $\|\cdot\|_F$ satisfies all three, but it is not bilinear: $\mathbb{E}[\sqrt{v_1^\top G v_2}]^2$
    fails to be an inner product, so no metric tensor induces it
    [@pouplinIdentifyingLatentDistances2023, Prop. 2.2, §2.4, p. 7]. The consequence is that
    a manifold with a random metric is not merely a Riemannian manifold with a noisy
    parameter: it is "inherently more complex", and the Riemannian expected metric is a
    different object from the expected norm [@pouplinIdentifyingLatentDistances2023, §2.4,
    p. 6; §5, p. 13].

**Exercise 6** ★★ (derive). Using the Gaussian-process closed form, show the upper absolute bound $\|v\|_F\le\|v\|_R$ and identify the inequality that produces it.

??? success "Solution"
    For a Gaussian-process decoder, $v^\top G v$ is a random variable with mean
    $\mathbb{E}[v^\top G v]=v^\top\mathbb{E}[G]v$, so $v^\top G v$ is the square of the random
    norm $\|v\|_G$ [@pouplinIdentifyingLatentDistances2023, Def. 2.4 and Prop. 2.3, pp. 5–7].
    The bound is Jensen's inequality applied to the concave function $x\mapsto\sqrt{x}$:
    $\mathbb{E}[\sqrt{v^\top G v}]\le\sqrt{\mathbb{E}[v^\top G v]}$, which is precisely
    $\|v\|_F\le\|v\|_R$ [@pouplinIdentifyingLatentDistances2023, Prop. 3.1 proof, §3.2, p. 9].
    The lower bound on the other side, $\|v\|_{\alpha\Sigma}\le\|v\|_F$ with
    $\alpha=2(\Gamma(D/2+1/2)/\Gamma(D/2))^2$, is not Jensen but an application of the
    closed form of Prop. 2.3 [@pouplinIdentifyingLatentDistances2023, Prop. 3.1, p. 9].

**Exercise 7** ★★ (design). You fit a VAE to a low-dimensional scientific dataset, and the
posterior variance is large and strongly varying across the latent space. Which construction
from this chapter do you use, and what must be true of your decoder for it to work?

??? success "Solution"
    Use the Fisher–Rao pullback, since the decoder's likelihood need not be Gaussian and the
    geometry is computed from a differentiable KL [@arvanitidisPullingBackInformation2022,
    §3.4, p. 5]. A directly Finslerian treatment is not available, because the Finsler
    results require a smooth stochastic process and VAE decoders are excluded by their
    posterior independence [@pouplinIdentifyingLatentDistances2023, footnote 1, p. 4];
    instead use the Riemannian expected-metric approximation, which is asymptotically
    correct at rate $O(1/D)$ but can differ materially in low dimension with high variance
    [@pouplinIdentifyingLatentDistances2023, Cor. 3.3, p. 11; §4, pp. 12–13]. What must be
    true is that the decoder extrapolates to high uncertainty away from the training codes;
    without this regularisation neither the Fisher–Rao nor the Euclidean pullback gives a
    sensible geometry [@arvanitidisPullingBackInformation2022, §4.1, p. 6]. Practically this
    is the KMeans-plus-sigmoid mechanism of §C.2, or the RBF inverse-variance model of
    [@arvanitidisGeometricallyEnrichedLatent2020, §3.2, p. 5].

**Exercise 8** ★★★ (critique). The confidence of [@pouplinIdentifyingLatentDistances2023] in the Riemannian approximation rests on the $O(1/D)$ bounds. Identify the weakest link in that argument, and the experiment that would settle it.

??? success "Solution"
    The bound is derived for a Gaussian-process decoder on a bounded latent manifold, and
    the paper's own high-dimensional experiment does not retrain a fresh model at each $D$;
    it truncates a sampled Jacobian, so it varies one factor while holding the data
    distribution fixed [@pouplinIdentifyingLatentDistances2023, Cor. 3.3, §3.4, p. 11 and
    §C.5, pp. 33–34]. What is demonstrated numerically is therefore the analytic inequality,
    not the modelling claim that real high-dimensional latent geometry becomes effectively
    Riemannian. In addition, the printed volume ratio is $O(\sqrt{D})$, which sits uneasily
    with the $O(1/D)$ length and energy results [@pouplinIdentifyingLatentDistances2023,
    Cor. 3.3, p. 11]. The experiment that would settle the modelling claim is to train
    separate decoders at a range of data dimensions on data with controlled posterior
    variance and measure the geodesic discrepancy directly, rather than truncating one
    sampled Jacobian. The collection itself does not report such an experiment.
