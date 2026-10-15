---
chapter: manifold-learning
title: Manifold learning and intrinsic dimension
outline_hash: ce511d23fbad
built_from:
  arvanitidisGeometricallyEnrichedLatent2020: 2b5378b36bee
  chazalIntroductionTopologicalData2021: cef80f43dbc4
  fuLatentTopologyInduction2022: fc105aa1f4de
  meilaManifoldLearningWhat2023: 009a25e9883c
  raghuExpressivePowerDeep2017: b62868fb88f1
---

# Manifold learning and intrinsic dimension

!!! abstract "In this chapter"
    - The *manifold hypothesis* says high-dimensional data are sampled on or near a smooth low-dimensional manifold $M\subset\R^D$; manifold learning recovers $M$ and its *intrinsic dimension* $d$ while distorting the geometry as little as possible [@meilaManifoldLearningWhat2023, §3, p. 6].
    - Embedding algorithms come in two kinds: one-shot spectral methods (PCA, Isomap, Diffusion Maps/Laplacian Eigenmaps, LTSA) with statistical guarantees but a *repeated-eigendirection problem*, and attraction–repulsion methods (t-SNE, UMAP) that avoid it heuristically and remain largely untheorised [@meilaManifoldLearningWhat2023, §4.4, p. 16].
    - No practical algorithm is isometric, yet the distortion is measurable and can be repaired with the embedding's *push-forward Riemannian metric* [@meilaManifoldLearningWhat2023, §5.5, pp. 22–23].
    - The same geometry appears inside a trained model: a network stretches a one-dimensional probe trajectory by a factor growing exponentially with depth [@raghuExpressivePowerDeep2017, Thm. 3, pp. 4–5], and a structured variational autoencoder can induce the latent *topology* of a frozen model's representation manifold [@fuLatentTopologyInduction2022, §3, p. 3].
    - Prerequisites: [Counting linear regions](counting-regions.md) and [Topology of regions and of data](region-topology.md) supply the region and activation-pattern vocabulary used in the last two sections.

The first eight chapters studied the geometry a ReLU network imposes on its *own* input space: the polyhedral complex of linear regions [@raghuExpressivePowerDeep2017, Thm. 2, p. 3], its combinatorial face lattice, the tropical polynomials behind it, and the topology of the decision boundary. This chapter turns the question around and asks about the geometry of *data* — and of the representations a network builds from it. The organising premise is the manifold hypothesis: although a data vector may have thousands of coordinates, the plausible vectors usually lie on or near a much lower-dimensional surface, and almost everything interesting about the data is the geometry of that surface [@meilaManifoldLearningWhat2023, §1, p. 1].

The chapter has three movements: classical manifold learning, the subject of the survey by [@meilaManifoldLearningWhat2023]; the geometry a network applies to its input, probed by one-dimensional trajectories [@raghuExpressivePowerDeep2017]; and the latent topology of a frozen model's representation manifold, induced by a structured variational autoencoder [@fuLatentTopologyInduction2022]. One vocabulary warning: the word "dimension" will mean two things. For data it is the intrinsic dimension $d$ of the manifold; for a network it is often the *number of linear regions* (or activation patterns), the count the earlier chapters took as a proxy for expressivity [@raghuExpressivePowerDeep2017, §2.1, p. 3]. I keep them apart. The shared notation — $\mathcal{N}(f)$ for the region count, $\mathbf{s}(x)$ for the activation pattern, $H_i$ for a region boundary — carries over, and for the data manifold I write $M\subset\R^D$ with $D$ ambient and $d$ intrinsic, following the survey [@meilaManifoldLearningWhat2023, §2.1, p. 3].

## The manifold hypothesis

A $d$-dimensional smooth manifold $M$ is a space that near every point $p$ looks like an open set of $\R^d$: there is a *chart* $(U,\varphi)$, a bijection from a neighbourhood $U\ni p$ to $\varphi(U)\subset\R^d$ that is smooth with a smooth inverse, and two overlapping charts must agree smoothly on their overlap [@meilaManifoldLearningWhat2023, §2.2, p. 3]. The dimension is forced to be the same for every chart, which is why it is called *intrinsic*: the sphere and the torus are both 2-manifolds even though they cannot be drawn flat [@meilaManifoldLearningWhat2023, §2.2, pp. 3–4]. A covering family of charts is an *atlas*, and because calculus is defined through charts, geometric and topological quantities are independent of the coordinates chosen — though the coordinates themselves are not, which is the root of the comparability problem below [@meilaManifoldLearningWhat2023, §2.2, pp. 3–4].

An *embedding* is a smooth map $F:M\to N$ whose inverse on its image is also smooth, and this is the object manifold learning computes [@meilaManifoldLearningWhat2023, §2.2, p. 4]. Given data $x_1,\dots,x_n\in\R^D$, one models them as lying on a $d$-manifold $M$ that is a submanifold of $\R^D$ — here $D$ is the *ambient* dimension — and searches for an embedding $F:M\to\R^m$ with $m\ge d$ and $m\ll D$; if $m=d$, $F$ is essentially a global coordinate chart [@meilaManifoldLearningWhat2023, §2.2, p. 4]. Whitney's theorem that every $d$-manifold embeds in $\R^{2d}$ justifies the hope of compression from $D$ to $O(d)$ [@meilaManifoldLearningWhat2023, §2.2, p. 4]. The *manifold hypothesis* itself is the assumption that such an $M$ exists: the $x_i$ are drawn from a distribution $P$ supported on, or close to, a $d$-manifold embedded in $\R^D$, and the survey treats the noiseless case where the data lie exactly on $M$ [@meilaManifoldLearningWhat2023, §3, p. 6].

!!! note "Background"
    A Riemannian metric $g$ assigns to each $p\in M$ an inner product $\langle\cdot,\cdot\rangle_{g(p)}$ on the tangent space $T_pM$, varying smoothly with $p$; it makes lengths, angles and volumes intrinsic to $M$, without reference to the ambient $\R^D$. The tangent spaces form the tangent bundle $TM$, and a smooth map $F$ induces the differential (Jacobian) $dF_p:T_pM\to T_{F(p)}N$ [@meilaManifoldLearningWhat2023, §2.3, pp. 4–5].

With a metric, the right notion of distance is the *geodesic distance* $d_M(x_1,x_2)$, the shortest length of a path in $M$ joining the two points — genuinely different from the ambient Euclidean distance $\lVert x_1-x_2\rVert$ the data matrix hands you [@meilaManifoldLearningWhat2023, §2.3, p. 4]. The survey's motivating example: two configurations of an ethanol molecule may be close in $\R^{27}$, but the straight segment between them passes through physically impossible configurations, so the meaningful distance follows the manifold of realisable configurations [@meilaManifoldLearningWhat2023, §2.3, p. 4]. A smooth map $F:M\to N$ between Riemannian manifolds is an *isometry* iff

$$
\langle v_1,v_2\rangle_{g(p)}=\langle dF_p(v_1),dF_p(v_2)\rangle_{h(F(p))}
\qquad \forall p\in M,\ v_1,v_2\in T_pM,
$$

so that distances, angles and volumes are preserved [@meilaManifoldLearningWhat2023, Eq. 1, p. 6]. Although Nash's theorem guarantees an isometric embedding exists, its constructive proof is not numerically usable, so no known practical algorithm is isometric [@meilaManifoldLearningWhat2023, §2.3, p. 6].

## From a neighbourhood graph to an embedding

Practically every manifold-learning algorithm begins by deciding who each point's neighbours are. The result is the *neighbourhood graph*: nodes are the data points, and an edge joins neighbours — either all points within a radius $r$ (*radius-neighbour graph*) or the $k$ closest points ($k$-NN graph) [@meilaManifoldLearningWhat2023, §3.1, p. 7]. The $k$-NN graph is more regular, is often connected when the radius graph is not, and admits fast approximate construction, but it is asymmetric unless symmetrised and harder to analyse, so fewer consistency results are known for it [@meilaManifoldLearningWhat2023, §3.1, p. 7]. Distances are encoded in a similarity matrix $K$ via a kernel,

$$
K_{ij} :=
\begin{cases}
K\!\left(\dfrac{\lVert x_i-x_j\rVert}{h}\right), & x_j\in N_i,\\[6pt]
0, & \text{otherwise},
\end{cases}
$$

almost universally Gaussian, $K(u)=\exp(-u^2)$, with width $h$ effectively setting a radius $r\propto h$ [@meilaManifoldLearningWhat2023, Eq. 2, p. 7]. This weighted graph is the summary of local geometry the algorithm consumes [@meilaManifoldLearningWhat2023, §3.1, p. 7].

The survey groups the algorithms into three paradigms by what they produce [@meilaManifoldLearningWhat2023, Tab. 1, p. 6]:

| Paradigm | Representation | Dimension change |
|---|---|---|
| Linear local approximation | coordinates in a local tangent space $T_pM$ | $D\to d$, local only |
| Principal curves and surfaces | a global denoised manifold $M\subset\R^D$ | $D\to D$, no reduction |
| Embedding algorithms | global coordinates $y_i\in F(M)\subset\R^m$, $m\ge d$ | $D\to m\ge d$ |

Local linear approximation — weighted ("local") PCA on a neighbourhood — gives honest low-dimensional coordinates but only near a reference point, and stitching the local charts globally is tedious [@meilaManifoldLearningWhat2023, §3.2, pp. 8–9]. Principal curves and $d$-manifolds, the ridge of the data density, produce a global object but do not reduce dimension, and are the only paradigm that assumes noise, which makes them valuable for manifold estimation under noise [@meilaManifoldLearningWhat2023, §3.3, pp. 8–9]. The rest of this section follows the third paradigm.

The one-shot algorithms read coordinates off eigenvectors of a matrix built from the graph, or solve a global convex problem. PCA is the linear special case: find the orthogonal $T\in\R^{D\times d}$ minimising the reconstruction error

$$
\min_{T\in\R^{D\times d},\ T^\top T=I_d}\ \sum_{i=1}^n \lVert x_i-TT^\top x_i\rVert^2
= \min_{T\in\R^{D\times d},\ T^\top T=I_d}\ \lVert X-XTT^\top\rVert_F^2,
$$

whose solution is the leading $d$ singular vectors of the data matrix $X$, with coordinates $Y=XT$ [@meilaManifoldLearningWhat2023, Eq. 4, p. 10]. Isomap generalises multidimensional scaling by replacing Euclidean distance with shortest-path distance in the graph, an approximation to geodesic distance exact in the limit of large $n$ for a flat manifold with convex data support [@meilaManifoldLearningWhat2023, §4.2.1, pp. 10–11]. Diffusion Maps and Laplacian Eigenmaps embed the data by eigenvectors of a *graph Laplacian* built from $K$; the version that removes the sampling-density bias, the *renormalised Laplacian*, normalises the columns of $K$ and then its rows [@meilaManifoldLearningWhat2023, §4.2.2, pp. 11–12]. Local Tangent Space Alignment estimates a tangent coordinate in each neighbourhood and aligns the local affine frames by minimising a global reconstruction error [@meilaManifoldLearningWhat2023, §4.2.3, p. 12].

!!! warning "The repeated-eigendirection problem"
    Eigenvector-based embeddings fail on manifolds of large aspect ratio — a long thin strip, or a thin torus. Because the eigenvectors of a Laplacian are nested harmonics rather than independent directions, the scatter plot of the first two coordinates of a rectangle traces a parabola: a rank drop of the embedding Jacobian the survey calls the *Repeated Eigendirection Problem* [@meilaManifoldLearningWhat2023, §4.3, p. 13]. The visual signature is the "horseshoe", and it is a diagnosis, not a property of the data: one dimension has been collapsed or poorly reflected [@meilaManifoldLearningWhat2023, §4.3, p. 13]. The fix, *Independent Eigendirection Selection*, picks eigenfunctions whose differentials are jointly of rank $d$ everywhere, replacing the rank condition by well-conditioning at the sample points [@meilaManifoldLearningWhat2023, §4.3.2, p. 14].

The attraction–repulsion family — t-SNE and UMAP — avoids the problem by a different route. t-SNE constructs similarity matrices $V$ from the data and $W$ from the embedding and minimises a cross-entropy between them,

$$
L_{t\text{-SNE}} = -\sum_{i,j}\frac{1}{n}V_{ij}\ln W_{ij} + \ln\sum_{ij}W_{ij},
$$

whose gradient

$$
\frac{\partial L_{t\text{-SNE}}}{\partial y_i}
= -\sum_{ij}V_{ij}W_{ij}(y_i-y_j) + \frac{n}{\rho}\sum_{ij}\frac{W_{ij}}{w_{tot}}(y_i-y_j)
$$

exposes the mechanism: the first term attracts graph neighbours, the second repels every pair of embedded points, and $\rho$ tunes the balance [@meilaManifoldLearningWhat2023, Eq. 7, p. 13; Eq. 8, p. 14]. Repulsion is why the family is immune to the repeated-eigendirection problem, and also why it can exaggerate clusters [@meilaManifoldLearningWhat2023, §4.3.1, p. 13; §5.1, p. 18]. The survey is blunt about the price: a local relaxation "cannot resolve the rank deficiency globally, and it may become trapped in a local optimum", UMAP's theory "is still very limited", and the choice of neighbour count $k$ in t-SNE is an unsolved issue [@meilaManifoldLearningWhat2023, §4.3.2, p. 14; §4.3.1, p. 13]. In short, one-shot algorithms are better understood and more scalable but need eigenvector post-processing, while neighbour-embedding algorithms remain heuristic for manifold learning even as they remain useful for visualisation and clustering [@meilaManifoldLearningWhat2023, §4.4, p. 16].

## Estimating the intrinsic dimension

Knowing $d$ matters twice over: it is a descriptive fact about the data, and Isomap, LTSA and every local-PCA method require it as input [@meilaManifoldLearningWhat2023, §5.3, p. 20]. The theory is deceptively encouraging. The minimax rate for estimating $d$ lies between $n^{-2n}$ and $n^{-n/(D+1)}$ for a well-behaved manifold — an information-theoretic bound on what any estimator can achieve [@meilaManifoldLearningWhat2023, §5.3, p. 20]. Once i.i.d. noise is added, the minimax rate becomes exponential, of order $q^n$ with $q<1$, where that quantity is the probability of being wrong [@meilaManifoldLearningWhat2023, §5.3, p. 20]. The survey warns that this rate renders ambiguously in the PDF and should be checked against the original before it is quoted; I pass the warning on.

The practical estimators share one idea: find a local statistic that scales with $d$ by a known law. The volume of a ball of radius $r$ inside $M$ scales like $r^d$, so if $n$ samples are uniform the expected count satisfies

$$
\log \#B_r = d\log r + \log n + \text{constant},
$$

and the slope of $\log\#B_r$ against $\log r$ estimates $d$ [@meilaManifoldLearningWhat2023, Eq. 9, p. 20]. The *correlation dimension* turns this into an estimator from the average number of radius-$r$ neighbours,

$$
\hat{d}_C = \lim_{r\to 0}\frac{\log \frac{1}{n}\sum_{i\neq i'}\mathbf{1}_{\lVert x_i-x_{i'}\rVert\le r}}{\log r},
$$

and *box counting* reads the covering number $\nu(r)$ the same way, becoming the Hausdorff dimension when the cover is by balls [@meilaManifoldLearningWhat2023, Eq. 10, p. 20; Eq. 11, p. 21]. Dropping uniformity gives the *doubling dimension* from $k_{i,2r}/k_{i,r}\approx 2^d$, and modern estimators use the distance to the $k$-th neighbour, the volume of a spherical cap, the Wasserstein distance between two $n$-samples of $M$ (which scales like $n^{-1/d}$), a maximum-likelihood method on the $k$-NN graph, or the largest eigengap of a local covariance [@meilaManifoldLearningWhat2023, §5.3, pp. 20–21]. All converge to $d$ in the noiseless limit but differ under noise and non-uniform sampling [@meilaManifoldLearningWhat2023, §5.3, p. 21].

!!! question "Open question"
    The theory says $d$ is estimable at least as easily as other manifold parameters, but the survey concludes that in practice "no satisfactorily robust solutions have been found yet", with noise the main culprit [@meilaManifoldLearningWhat2023, §5.3, p. 20]. Every estimator needs a scale $r$ or $k$ chosen in a bias–variance trade-off — too large smooths out the scaling law, too small leaves each ball with a single point [@meilaManifoldLearningWhat2023, §5.3, p. 21].

## Consistency, density and the scale of the neighbourhood

The parameters a method forces the user to set — the neighbour count $k$ or kernel width $h$ — are not free. Ignoring them can yield methods with no large-$n$ limit at all, such as unregularised LLE, or milder pathologies such as density-induced biases and *artifacts* — clusters, arms and horseshoes in the embedding that correspond to nothing in the data [@meilaManifoldLearningWhat2023, §5, p. 16]. The cleanest bias is due to sampling density: non-uniform sampling contracts densely sampled regions of $M$ and stretches sparse ones, for one-shot algorithms as well as for t-SNE, and the survey shows this on the ethanol torus, where three high-density regions appear as sharp "corners" [@meilaManifoldLearningWhat2023, §5.1, pp. 18–19]. For the graph Laplacian the bias was computed by Coifman and Lafon and removed asymptotically by the renormalised Laplacian, whose degree values $d'_i$ are themselves estimators of the local sampling density [@meilaManifoldLearningWhat2023, §5.1, p. 18]. A $k$-NN graph with a constant kernel shows qualitatively the same bias, so the graph type does not rescue the analysis [@meilaManifoldLearningWhat2023, §5.1, p. 18].

Where theory does deliver is the rate at which the neighbourhood scale must shrink. For the random-walk Laplacian the optimal bandwidth is $h\sim n^{-1/(d+6)}$, and for a $k$-NN Laplacian estimation the neighbour count must grow at a recommended rate $k\sim n^{4/(d+4)}(\log n)^{d/(d+4)}$ [@meilaManifoldLearningWhat2023, §5.2, p. 18]. The constants hide the manifold volume, curvature and injectivity radius, unknown in practice, and every such result is *non-parametric*: combining the rate $k$ with the convergence rate $\approx\sqrt{\log n/k}\,(k/n)^{1/d}$ shows that a tenfold decrease in error needs $n$ to grow by about $10^{(d+4)/3}$ [@meilaManifoldLearningWhat2023, §5.2, p. 18]. The required sample size is exponential in the dimension, and there is no way around it for a fully non-parametric estimator.

Cross-validation, the usual answer to a tuning problem, does not apply here for lack of a criterion to validate against [@meilaManifoldLearningWhat2023, §5.2, pp. 18–19]. What replaces it are self-consistency criteria: choose $k$ to preserve $k'$-neighbourhoods across the embedding, or choose the radius $r$ for which the tangent space at a point aligns best with the leading local principal subspace [@meilaManifoldLearningWhat2023, §5.2, pp. 18–19]. A diagnostic tuned for one embedding algorithm does not transfer to another, because different embeddings distort neighbourhoods differently [@meilaManifoldLearningWhat2023, §5.2, pp. 18–19].

## Distortion and the push-forward metric

We reach the survey's central geometric claim: the distortions of an embedding are not cosmetic but measurable, and they do not vanish as $n$ grows, as the density becomes uniform, or as the graph is replaced by a consistent Laplacian [@meilaManifoldLearningWhat2023, §5.5, p. 22]. An embedding close to isometric exists in principle, but since Nash's construction is not implementable and Diffusion Maps is isometric only as $m\to\infty$ — which defeats dimension reduction — the practical strategy is to estimate the distortion per point and correct for it afterwards [@meilaManifoldLearningWhat2023, §5.5, pp. 22–23]. The correction is exact. Endow the image $F(M)$ with the *push-forward* metric of $F$, defined on $u,v\in T_{F(p)}F(M)$ by

$$
\langle u,v\rangle_{\tilde{g}(Fp)} \equiv \langle dF_p^{\dagger}(u),\ dF_p^{\dagger}(v)\rangle_{g(p)},
\qquad
\tilde{g}(Fp) \equiv \big((dF_p)^\top\big)^{\dagger}\, g(p)\,(dF_p)^{\dagger},
$$

with $dF_p^{\dagger}$ the pseudoinverse of the differential [@meilaManifoldLearningWhat2023, Eq. 12, p. 23; Eq. 13, p. 23]. With this metric, $(F(M),\tilde{g})$ is isometric to $(M,g)$, so any distance, angle or volume computed with the corrected metric reproduces the intrinsic value up to sampling noise [@meilaManifoldLearningWhat2023, §5.5, p. 23].

Two estimates of $\tilde{g}$ are available: differentiating the embedding directly, or extracting it from the Laplace-Beltrami operator $\Delta_M$ applied to products of centred coordinate functions $f_{kl,p}=(F_k-F_k(p))(F_l-F_l(p))$, for which $\tfrac12\Delta_M f_{kl,p}\big|_p$ is the $(k,l)$ entry of the inverse push-forward metric [@meilaManifoldLearningWhat2023, §5.5, p. 23]. The payoff is a diagnostic. When the embedding is isometric the singular values of $\tilde{g}$ are all $1$, so their deviation from $1$ measures the local distortion $\tilde{h}_i$, and visualising it separates genuine structure from artifact: a "filament" produced by stretching a low-density region shows up as heavy distortion, whereas a true filament does not [@meilaManifoldLearningWhat2023, §5.5, p. 23; §6.2, p. 25]. The same quantity doubles as an objective, defining a "Riemannian relaxation" that improves the conditioning of an embedding iteratively — one more remedy for the repeated-eigendirection problem [@meilaManifoldLearningWhat2023, §5.5, p. 24; §4.3.1, p. 13].

This is where the survey meets the rest of the textbook. The push-forward metric takes a metric on the data manifold and pushes it forward through an embedding $F$; the [next chapter](latent-geometry.md) instead takes an ambient or statistical metric and pulls it *back* through a decoder $\phi$, giving the latent space the metric $G(z)=J_\phi(z)^\top G(\phi(z))J_\phi(z)$ [@arvanitidisGeometricallyEnrichedLatent2020, §3]. The two are inverse constructions — push-forward through the encoder, pull-back through the generator — and both answer the same complaint: an embedding's coordinates are not intrinsic, so attach a metric to them [@meilaManifoldLearningWhat2023, §5.5, pp. 22–23]. The survey itself stops at Riemannian structure and does not touch the Finsler and information-geometric variants that follow [@meilaManifoldLearningWhat2023, §5.5, p. 22].

## Probing a network with trajectories

The complementary question — what geometry does a *network* apply to its input? — is where [@raghuExpressivePowerDeep2017] comes in. Its starting point is that input space $\R^m$ is too large to characterise a network's behaviour over it, so the paper studies one-dimensional *trajectories* $x(t)$, $t\in[0,1]$, between two inputs — in the experiments the circular arc $x(t)=\cos(\tfrac{\pi t}{2})x_0+\sin(\tfrac{\pi t}{2})x_1$ [@raghuExpressivePowerDeep2017, §2, p. 2].

Two combinatorial measures come first. The *activation pattern* $AP(F_A(x;W))$ records, for every neuron, which linear piece of its activation function it is on — a $\{0,1\}^{\text{neurons}}$ string for ReLUs, $\{-1,0,1\}^{\text{neurons}}$ for hard tanh — i.e. the sign vector $\mathbf{s}(x)$ of the shared notation, whose regions are the polyhedra of the earlier chapters [@raghuExpressivePowerDeep2017, §2.1, p. 3]. A *neuron transition* is a change of linear region between nearby inputs, and the counts $A(F_A(x(t);W))$ and $T(F_A(x(t);W))$ record distinct patterns and transitions along the sweep [@raghuExpressivePowerDeep2017, §2.1, p. 3]. Because each distinct pattern corresponds to a different linear map, counting them measures the non-linearity actually available [@raghuExpressivePowerDeep2017, §2.1, p. 3].

The counting theorem is the upper bound matching the hand-crafted constructions of Montufar et al. For a fully connected network with $n$ hidden layers of width $k$ and inputs in $\R^m$, the number of activation patterns over all of $\R^m$ and *all* weight settings is

$$
A(F_{A_{n,k}}(\R^m;W)) = O\!\left(k^{mn}\right)\ \text{(ReLU)},
\qquad
O\!\left((2k)^{mn}\right)\ \text{(hard tanh)},
$$

[@raghuExpressivePowerDeep2017, Thm. 1, p. 3]. This holds for every $W$, not one special construction, and holding the total neuron count $N$ fixed it yields the chain $U(1,N,m)<U(2,\tfrac{N}{2},m)<\cdots<U(n-1,\tfrac{N}{n-1},m)<U(n,k,m)$: at equal parameters, depth buys more patterns than width [@raghuExpressivePowerDeep2017, §2.1, p. 3]. The companion Theorem 2 restates, for both activations, that input space is partitioned into convex polytopes on each of which the network is a different linear function — the same partition the region-complex chapters build [@raghuExpressivePowerDeep2017, Thm. 2, p. 3].

The unification is *trajectory length*: the arc length

$$
\ell(x(t)) = \int \left\lvert \frac{dx(t)}{dt} \right\rvert dt
$$

of the image of the trajectory after a layer, measuring how much the network stretches the curve [@raghuExpressivePowerDeep2017, §2.2, p. 4]. The growth theorem says that for a random ReLU (or hard-tanh) network and a trajectory that is not a straight line,

$$
\mathbb{E}\!\left[\ell(z^{(d)}(t))\right] \gtrsim O\!\left( \left( \frac{\sigma_w\sqrt{k}}{\sqrt{k+1}} \right)^{d} \ell(x(t)) \right)
$$

for ReLUs, with an analogous hard-tanh expression in the bias scale, where $z^{(d)}(t)$ is the trajectory's image after layer $d$ and $\sigma_w,\sigma_b$ the weight and bias initialisation scales [@raghuExpressivePowerDeep2017, Thm. 3, pp. 4–5]. The structural content is that $d$ appears as the exponent while the width $k$ only enters the base: length grows exponentially with depth and polynomially with width, and the bound is tight in the limits of large $k$ and $\sigma_w$ [@raghuExpressivePowerDeep2017, §2.2, pp. 4–5]. A fourth theorem makes transitions proportional to length for hard tanh under a saturation assumption, $T(F_{A_{n,k}}(x(t);W))=O(g(k,\sigma_w,\sigma_b,n))$ with $g=O((\sqrt{k}/\sqrt{1+\sigma_b^2/\sigma_w^2})^n)$ [@raghuExpressivePowerDeep2017, Thm. 4, p. 5].

!!! warning "Reading the growth bound"
    Theorem 3 is *printed* as a lower bound on an expectation set against a big-$O$ — "$\geq O(\cdot)$" — which is mathematically awkward and reads as a typo for an order-of-magnitude growth statement; the paper's own use of it in Theorem 4 treats it that way [@raghuExpressivePowerDeep2017, Thm. 3, pp. 4–5]. Its proof lives in an appendix not present in the saved file, so the argument can only be checked from the main text's sketch [@raghuExpressivePowerDeep2017, §2.2, p. 5].

The empirical consequences tie trajectory length to training. There is a threshold on the initialisation scale below which the exponential regime is not entered, and training pushes the scaled weight variance upward, dragging even badly initialised networks into it [@raghuExpressivePowerDeep2017, §2.2, p. 5]. Because a small change to a lower layer is amplified through all following layers, trained networks are far more sensitive to lower-layer noise — on a CIFAR-10 convolutional network, weight noise of magnitude $0.25$ on layer 1 costs $0.7$ accuracy while the same noise on layer 5 costs about $0.02$ [@raghuExpressivePowerDeep2017, §3, p. 6]. Batch normalisation acts on the same quantity, reducing trajectory length and stabilising the representation, and replacing each batch-norm layer by a layer that directly regularises trajectory length gives comparable CIFAR-10 accuracy without the train/test discrepancy of batch norm [@raghuExpressivePowerDeep2017, §3, p. 7]. A manifold learner asks how much an embedding distorts a data manifold; [@raghuExpressivePowerDeep2017] measures precisely that for the map a network applies, with trajectory length standing in for the push-forward distortion and the count of activation patterns for the complexity of the atlas of affine charts the network overlays on its input [@raghuExpressivePowerDeep2017, §2.2, p. 4; Thm. 1, p. 3].

## Inducing the topology of a representation manifold

The last movement takes the manifold hypothesis literally about a *trained model's* internal representations. [@fuLatentTopologyInduction2022] studies the contextualised word vectors $r_t\in\R^{768}$ produced by a frozen, not fine-tuned, BERT-base-uncased encoder on the 20News corpus, all lying in one *representation manifold* $\mathcal{M}$ [@fuLatentTopologyInduction2022, §1, p. 1; §3, p. 3]. Its target is not the distance structure Meilă's survey measures but the *topology*: it hypothesises a spectrum of $N$ static latent states $s_1,\dots,s_N$ that act as anchors marking neighbours and connectivity in $\mathcal{M}$, described by two relations — state–word (emission) and state–state (transition) [@fuLatentTopologyInduction2022, §1, p. 1; §3, p. 3]. Because the encoder is frozen, any structure recovered is intrinsic to the model rather than imposed by a supervision signal, which is the paper's answer to the complaint that supervised probing cannot separate intrinsic properties from the probe's own bias [@fuLatentTopologyInduction2022, §2, p. 2].

The states are the latent variables of a structured variational autoencoder. Each contextualised vector $r_t$ gets a latent state index $z_t\in\{1,\dots,N\}$, and the state embeddings $s=[s_1,\dots,s_N]$ are the *only* learnable parameters of the inference model; emission and transition potentials are bilinear,

$$
\log\phi(x_t,z_t)=r_t^\top s_{z_t},
\qquad
\log\psi(z_{t-1},z_t)=s_{z_{t-1}}^\top s_{z_t},
$$

forming a linear-chain CRF whose posterior

$$
q(z\mid x)=\frac{1}{Z}\prod_{t=1}^T \psi(z_{t-1},z_t)\,\phi(x_t,z_t)
$$

is the inference network [@fuLatentTopologyInduction2022, Eq. 1, p. 3; Eq. 2, p. 3]. On top sits an LSTM decoder generating $p(x,z)=\prod_t p(x_t\mid z_{1:t})p(z_t\mid z_{1:t-1})$, so that maximising the evidence lower bound

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z\mid x)}\big[\log p(x,z)\big] + H\big(q(z\mid x)\big)
$$

forces the states to be generative; the decoder is dropped after training and only the states are analysed [@fuLatentTopologyInduction2022, Eq. 3–4, p. 3; Eq. 5, p. 3]. The number of states is set to $N=2000$, a "sweet point" between under-clustering at 10K and over-clustering at 100, and the entropy term is reweighted to prevent posterior collapse [@fuLatentTopologyInduction2022, §4, p. 5].

!!! warning "A sign to verify"
    In the saved PDF the entropy term of the ELBO renders ambiguously — the extracted text does not make its printed sign recoverable — and it is transcribed above with a `+`, the standard CRF-VAE form; the printed sign should be checked against the source before it is relied on [@fuLatentTopologyInduction2022, Eq. 5, p. 3].

The induced states are meaningful. Aligning them with five annotation types — POS, ENT, DEP, CCG and BERT Concept Net — gives the paper's Table 1: a random-embedding control (RANDEMB) already aligns 392 POS states covering 53.02% of occurrences, so alignment is not evidence by itself, and BERT$_{\text{LAST}}$ has more aligned states than the zeroth-layer BERT$_{\text{ZERO}}$ (804 vs 673 POS-aligned) but covers fewer occurrences (53.23% vs 65.31%) [@fuLatentTopologyInduction2022, Tab. 1, p. 4; §5.1, p. 5]. The states that do *not* align are still interpretable: annotators class about 85–89% of them as lexical, morphosyntactic or semantic, with only 11–14% uninterpretable [@fuLatentTopologyInduction2022, Tab. 2, p. 4]. This is the conceptual link to the *participation pattern* of the concept graph: what the model encodes is not the state index but the pattern with which the latent units participate in the generative process. Contextualisation is visible directly — before it, function words concentrate around a few head states; afterwards they spread across all states and mix with content, receiving meaning from their neighbours [@fuLatentTopologyInduction2022, §5.2, p. 6].

The transition structure is the paper's backbone. The state–state graph decoded from the validation set has nodes and edges both following a long-tail distribution, with frequent states inter-connected and function states acting as *hubs*: the state encoding "to" attracts content words like free-to, willing-to, go-to, and emits to-buy, to-sell, to-build, while other transitions encode passive voice or time [@fuLatentTopologyInduction2022, §6.1, p. 7; Tab. 5, p. 8]. This yields the headline mechanism: function states correspond to function words, content states to content words of similar meaning, transitions between them encode phrase constructions, and a sentence is a *traversal* of the graph in which the transition chain encodes the syntactic template and the emissions fill in the content, so structurally similar sentences share overlapping paths [@fuLatentTopologyInduction2022, §6.2, pp. 8–9].

The paper is unusually clear about what it has *not* shown, and its limits are this chapter's concerns. The analysis is of topology, not distance: how far apart nodes are in the representation manifold is set aside as an equally important geometric question for future work [@fuLatentTopologyInduction2022, §7, p. 9]. It addresses local topology far more than global topology, and studies only BERT, with evidence that autoregressive transformers have different topologies [@fuLatentTopologyInduction2022, §7, p. 9]. On the reading of the summary, the traversal claim rests on four hand-picked sentences and is illustrative rather than measured, and the alignment evaluation selects the checkpoint on the number of aligned tags, a bias the authors acknowledge [@fuLatentTopologyInduction2022, §5.1, p. 5]. The two open directions the paper names — the distance structure and the global hierarchy of states — are precisely what the metric constructions of the [next chapter](latent-geometry.md) supply.

## Summary

- The manifold hypothesis posits that data are sampled on or near a $d$-manifold $M\subset\R^D$; manifold learning looks for an embedding $F:M\to\R^m$ with $m\ge d$, $m\ll D$, and $d$ is well defined because it is chart-independent [@meilaManifoldLearningWhat2023, §2.2, p. 4; §3, p. 6].
- Algorithms divide into one-shot spectral methods (PCA, Isomap, DM/LE, LTSA), understood statistically but subject to the repeated-eigendirection problem, and attraction–repulsion methods (t-SNE, UMAP), repulsion-robust but heuristic [@meilaManifoldLearningWhat2023, §4.4, p. 16].
- Intrinsic dimension is estimable from local scaling laws (correlation dimension, box counting, doubling dimension, $k$-NN MLE, local-PCA eigengaps), yet noise makes it empirically hard despite favourable minimax rates [@meilaManifoldLearningWhat2023, §5.3, pp. 20–21].
- Neighbourhood scale must shrink with $n$ at non-parametric rates ($h\sim n^{-1/(d+6)}$, $k\sim n^{4/(d+4)}(\log n)^{d/(d+4)}$), so the required sample size grows exponentially in $d$ [@meilaManifoldLearningWhat2023, §5.2, p. 18].
- No practical embedding is isometric; the push-forward metric $\tilde{g}$ repairs distances, angles and volumes pointwise, and its singular-value deviation from $1$ is a distortion diagnostic [@meilaManifoldLearningWhat2023, §5.5, pp. 22–23].
- A network deforms its input along a one-dimensional trajectory by an arc length growing exponentially with depth and only polynomially with width, and the same growth explains its sensitivity to lower-layer weights [@raghuExpressivePowerDeep2017, Thm. 3, pp. 4–5; §3, p. 6].
- The activation-pattern count $O(k^{mn})$ is a tight upper bound over all weight settings and recovers the depth-beats-width chain at fixed parameter count [@raghuExpressivePowerDeep2017, Thm. 1, p. 3].
- The topology of a frozen model's representation manifold can be induced as latent states of a structured VAE, with transitions encoding phrase constructions and sentences as traversals — but only the topology, not the metric [@fuLatentTopologyInduction2022, §6.2, pp. 8–9; §7, p. 9].

## Further reading

- [@meilaManifoldLearningWhat2023]: the reference survey; read §2 for the vocabulary, §3 for the three paradigms, §4 for the algorithms and the repeated-eigendirection problem, §5 for the statistics and the push-forward metric, §6 for the applications and their pitfalls.
- [@fuLatentTopologyInduction2022]: inducing latent topology with a structured VAE; read §3 for the CRF-VAE, §5 for the state–word results, §6 for the transition topology and traversals, §7 for the stated limits.
- [@raghuExpressivePowerDeep2017]: trajectory length and the measures of expressivity; read §2.1 for the tight upper bound, §2.2 for the growth and transitions theorems, §3 for the training-time experiments and trajectory regularisation.
- [@arvanitidisGeometricallyEnrichedLatent2020]: the pull-back counterpart of the push-forward metric, taken up in the [next chapter](latent-geometry.md).
- [@chazalIntroductionTopologicalData2021]: the TDA toolbox for the *global* topology that [@fuLatentTopologyInduction2022] leaves open; see [Topology of regions and of data](region-topology.md).

## Exercises

**Exercise 1** ★ (check). State the manifold hypothesis precisely, and explain why the intrinsic dimension of a manifold is well defined rather than dependent on the chart you picked.

??? success "Solution"
    The manifold hypothesis says the data $\{x_i\}_{i=1}^n$, with $x_i\in\R^D$, are drawn from a distribution $P$ supported on, or close to, a $d$-dimensional manifold $M$ embedded in $\R^D$; $D$ is the ambient dimension and $d$ the intrinsic one [@meilaManifoldLearningWhat2023, §3, p. 6]. A manifold is covered by charts $(U,\varphi)$, smooth bijections onto open subsets of some $\R^k$ with smoothly compatible overlaps [@meilaManifoldLearningWhat2023, §2.2, p. 3]. On an overlap one chart is obtained from the other by a smooth change of variables, and a change of variables cannot alter which Euclidean space a chart lands in, so $k$ is the same for every chart and atlas; it is intrinsic to $M$, not to coordinates [@meilaManifoldLearningWhat2023, §2.2, pp. 3–4].

**Exercise 2** ★ (check). The survey says the $k$-NN graph is often preferred in practice but harder to analyse than the radius-neighbour graph. Explain both halves.

??? success "Solution"
    The radius graph connects $x_i,x_j$ iff $\lVert x_i-x_j\rVert\le r$; the $k$-NN graph connects $x_i$ to its $k$ closest points [@meilaManifoldLearningWhat2023, §3.1, p. 7]. The $k$-NN relation is not symmetric, so it must be symmetrised, making the degree distribution harder to control, whereas the radius graph's neighbour count is proportional to local density and manifold estimation can be analysed as kernel regression [@meilaManifoldLearningWhat2023, §3.1, p. 7]. In practice the $k$-NN graph is more regular, often connected when a radius graph is not, and fast approximate construction is widely available, so it is the default; it carries the same non-uniform-sampling bias as the radius graph [@meilaManifoldLearningWhat2023, §3.1, p. 7; §5.1, p. 18].

**Exercise 3** ★★ (derive). Starting from the scaling law for the number of samples in a ball of radius $r$ inside a $d$-manifold, derive the correlation-dimension estimator and say why its slope estimates $d$.

??? success "Solution"
    For a ball $B_r\subset M$ the volume scales as $r^d$, so if $n$ samples are uniform the expected count scales as $nr^d$: $\log\#B_r=d\log r+\log n+\text{constant}$ [@meilaManifoldLearningWhat2023, Eq. 9, p. 20]. Let $k_{i,r}$ be the number of radius-$r$ neighbours of $x_i$; then $\log k_{i,r}\approx d\log r+\text{constant}$, and averaging over points recovers the same line. Taking $r\to0$ kills the constant and gives the correlation dimension

    $$\hat{d}_C=\lim_{r\to0}\frac{\log\frac{1}{n}\sum_{i\neq i'}\mathbf{1}_{\lVert x_i-x_{i'}\rVert\le r}}{\log r},$$

    i.e. the slope of a line fit to $(\log r,\log\#B_r)$ over a suitable range of scales [@meilaManifoldLearningWhat2023, Eq. 10, p. 20]. It uses the average neighbour count because $\sum_i(k_{i,r}-1)$ counts unordered pairs; box counting is the same idea with a covering number $\nu(r)\sim r^{-d}$ [@meilaManifoldLearningWhat2023, Eq. 11, p. 21].

**Exercise 4** ★★ (derive). The push-forward metric $\tilde{g}$ makes $(F(M),\tilde{g})$ isometric to $(M,g)$. Show that when the embedding is isometric the singular values of $\tilde{g}$ are all $1$, and say what a singular value different from $1$ means.

??? success "Solution"
    An isometry satisfies $\langle v_1,v_2\rangle_{g(p)}=\langle dF_p(v_1),dF_p(v_2)\rangle_{h(F(p))}$ for all tangent vectors [@meilaManifoldLearningWhat2023, Eq. 1, p. 6]. The push-forward metric is $\tilde{g}(Fp)=((dF_p)^\top)^{\dagger}g(p)(dF_p)^{\dagger}$ [@meilaManifoldLearningWhat2023, Eqs. 12–13, p. 23]. When $M\subset\R^D$ inherits the Euclidean metric, $g(p)=I_d$, so on the $d$-dimensional image $\tilde{g}=((dF_p)^\top)^{\dagger}(dF_p)^{\dagger}$ has the same spectrum as the inverse of the squared singular values of the tangent map. For an isometry those singular values are all $1$, so every singular value of $\tilde{g}$ is $1$; a deviation from $1$ is exactly the local distortion, giving the per-point diagnostic that separates a real filament from stretching of a low-density region [@meilaManifoldLearningWhat2023, §5.5, p. 23; §6.2, p. 25].

**Exercise 5** ★★ (compare). This chapter's survey pushes a metric *forward* through an embedding $F$; the [next chapter](latent-geometry.md) pulls a metric *back* through a decoder $\phi$. Contrast what each assumes and what each is for.

??? success "Solution"
    The push-forward transports the intrinsic data metric $g$ on $M$ to the image $F(M)\subset\R^m$ via $\tilde{g}(Fp)=((dF_p)^\top)^{\dagger}g(p)(dF_p)^{\dagger}$, so that a *given* embedding, however distorting, can be used as if isometric: distances, angles and volumes computed with $\tilde{g}$ reproduce the intrinsic values up to sampling noise [@meilaManifoldLearningWhat2023, §5.5, pp. 22–23]. The pull-back transports a metric on the output or ambient space to a generator's latent space through the decoder Jacobian, $G(z)=J_\phi(z)^\top G(\phi(z))J_\phi(z)$, so the latent space *acquires* the geometry of the data it generates [@arvanitidisGeometricallyEnrichedLatent2020, §3]. The survey's construction post-processes an embedding and assumes only a Riemannian $M$; the pull-back presupposes a differentiable generator and a metric to pull back [@meilaManifoldLearningWhat2023, §5.5, p. 22]. They are inverse operations of one idea — an embedding's coordinates are not intrinsic, so attach a metric to them — and the survey stops at the Riemannian case, leaving Finsler and information-geometric variants to the other chapter [@meilaManifoldLearningWhat2023, §5.5, p. 22].

**Exercise 6** ★★ (compare). Both [@raghuExpressivePowerDeep2017] and [@fuLatentTopologyInduction2022] probe the geometry of a trained network's representation. What does each measure, and what does each leave out?

??? success "Solution"
    Raghu et al. probe the *metric* deformation the network induces: they sweep the input along a one-dimensional trajectory and measure its arc length after each layer, which grows exponentially with depth but only polynomially with width, and count the neuron transitions, which record how many linear regions the curve crosses [@raghuExpressivePowerDeep2017, Thm. 3, pp. 4–5; §2.1, p. 3]. This is quantitative, computable for any weight setting, and explains the sensitivity to lower-layer weights [@raghuExpressivePowerDeep2017, §3, p. 6]. What it leaves out is topology: the probe is one-dimensional and local, so it says nothing about global connectivity. Fu and Lapata measure that complement — the *topology* of a frozen encoder's representation manifold, induced as latent states whose transitions form the graph's backbone [@fuLatentTopologyInduction2022, §3, p. 3; §6.1, p. 7]. They defer the distance structure, the quantity Raghu et al. quantify, to future work, and study local topology rather than the global hierarchy [@fuLatentTopologyInduction2022, §7, p. 9].

**Exercise 7** ★★★ (design). You have a frozen encoder mapping sentences to 512-dimensional vectors. You want both the intrinsic dimension of the representation cloud and a trustworthy picture of its global topology. Design a pipeline and name the decisions most likely to mislead you.

??? success "Solution"
    A defensible pipeline: (i) build a symmetrised $k$-NN graph, knowing its bias matches the radius graph's [@meilaManifoldLearningWhat2023, §3.1, p. 7; §5.1, p. 18]; (ii) estimate the intrinsic dimension with more than one estimator, because no estimator is robust in noise [@meilaManifoldLearningWhat2023, §5.3, pp. 20–21]; (iii) choose the neighbourhood scale by a self-consistency criterion, since cross-validation has no criterion and the required $n$ grows exponentially in $d$ [@meilaManifoldLearningWhat2023, §5.2, pp. 18–19]; (iv) for topology, induce the latent state network of a structured VAE rather than reading a t-SNE picture, because t-SNE exaggerates clusters and its features may be artifacts [@fuLatentTopologyInduction2022, §3, p. 3; @meilaManifoldLearningWhat2023, §5.1, p. 18]; (v) validate any feature by re-embedding under different seeds, algorithms and parameters, and estimate the distortion $\tilde{g}$ to separate real structure from stretching [@meilaManifoldLearningWhat2023, §6.2, p. 25]. The likeliest traps are a single dimension estimator under noise, a t-SNE cluster that is a repulsion artifact, and a global-topology claim drawn from local neighbour relations — the gap Fu and Lapata themselves name [@fuLatentTopologyInduction2022, §7, p. 9].
