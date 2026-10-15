---
chapter: counting-regions
title: Counting linear regions
outline_hash: 1931e7f419e7
built_from:
  NIPS2014_fa6f2a46: e11f3b75c4ab
  averkovEXPRESSIVENESSRATIONALRELU2025: e57bb3498e66
  brandenburgDecompositionPolyhedraPiecewise2024: 307dddfb6bce
  gagneuxHowImproveExpressivity2025: 3626abad8f47
  goujonNumberRegionsPiecewise2023: fb83a40ca26c
  haninComplexityLinearRegions2019: 438efc947567
  haninDeepReLUNetworks2019: d172a42e0d06
  pascanuNumberResponseRegions2013: 3fae3832ff4c
  serraBoundingCountingLinear2018: 9cdce7bede89
---

# Counting linear regions

!!! abstract "In this chapter"
    - Why the number of linear regions $\mathcal{N}(f)$ is the standard proxy for the expressivity of a piecewise-linear network, and what it does and does not measure.
    - The exact counts for shallow rectifier and maxout layers, from hyperplane arrangements, and the *replication* construction that makes the count grow exponentially with depth but only polynomially with width.
    - The sharper bounds that add a bottleneck effect, the exact one-dimensional answer, and how the depth–width trade-off is stated for general piecewise-linear activations and for rational-weight networks.
    - Why typical random and trained networks use far fewer regions than the maximum, and how the same region geometry constrains convex networks.
    Prerequisites: [Polyhedra, arrangements and piecewise-linear maps](polyhedral-foundations.md), [The region complex of a ReLU network](relu-network-geometry.md).

A piecewise-linear network partitions its input space into regions on which it computes a different affine function. The previous chapter described that partition as a polyhedral complex; this chapter asks how large it can get. The number of regions is the standard scalar summary of "how complicated a function can this architecture represent" ($\mathcal{N}(f)$ in the [shared notation](../notation.md)), and the whole line of work surveyed here is about how that number depends on depth, width and input dimension [@NIPS2014_fa6f2a46, §1; @pascanuNumberResponseRegions2013, §1].

The results come in two flavours that must not be confused. *Worst-case* bounds are existential: they exhibit one weight assignment that attains a high count [@NIPS2014_fa6f2a46, §2.5]. *Average-case* results ask what a random or trained network actually does, and they tell a very different story [@haninDeepReLUNetworks2019, §1]. Throughout, I translate each paper's notation into the shared one: the widths are $n_l$, the depth $L$, the input dimension $n_0$, and the count is $\mathcal{N}(f)$.

## Counting regions as a complexity measure

Fix a piecewise-linear network $f:\R^{n_0}\to\R^{m}$. A *linear region* is a maximal connected subset of the input space on which $f$ is affine; for the networks considered here every region has full dimension $n_0$ [@NIPS2014_fa6f2a46, §2.1]. For a rectifier network each region is the intersection of half-spaces defined by the hyperplanes $H_i=\{x: W_{i,:}x+b_i=0\}$ of its units, so regions correspond to activation patterns and there are at most $2^N$ of them for $N$ hidden units — a first, architecture-independent upper bound [@NIPS2014_fa6f2a46, Prop. 4; @haninComplexityLinearRegions2019, Eq. 1]. The count can also be as low as $1$ (all weights and biases zero), so only the distribution of parameters decides where on this range a network sits [@haninComplexityLinearRegions2019, §2.1].

Why use this quantity at all? The intuition shared by the early papers is that the linear pieces are the "resolution" available to approximate a curved function, so more pieces mean more flexibility [@pascanuNumberResponseRegions2013, §1]. The same authors are careful to say that the count is only an *upper* bound on flexibility, because the affine pieces are correlated by the shared parameters and a network need not be able to set an independent slope on each [@pascanuNumberResponseRegions2013, §1]. A second, subtler justification is that the maximum is not a measure-zero accident: the map from parameters to functions is continuous, and a region has finite volume, so any parameter choice attaining a given count is surrounded by a ball of parameters attaining at least that count [@NIPS2014_fa6f2a46, §2.5].

## Shallow networks: hyperplane arrangements

For a single rectifier layer the picture is exact. The units' hyperplanes form an arrangement, and the regions of the layer are the regions of that arrangement, so the maximal count for $n_0$ inputs and $n_1$ units is the number of regions of $n_1$ hyperplanes in $\R^{n_0}$ in general position [@pascanuNumberResponseRegions2013, §3; @NIPS2014_fa6f2a46, §2.2].

!!! note "Background"
    The count of regions of an arrangement of $m$ hyperplanes in general position in $\R^{d}$ is
    $\sum_{j=0}^{d}\binom{m}{j}$, and the number of relatively bounded regions is $\binom{m-1}{d}$.
    This is Zaslavsky's theorem (1975), used as a tool by every paper in this chapter but not itself
    part of the collection.

Accordingly the maximal number of linear regions of a shallow rectifier network is
$\sum_{j=0}^{n_0}\binom{n_1}{j}$ [@pascanuNumberResponseRegions2013, Prop. 2], and it is attained generically by shallow rectifier networks [@NIPS2014_fa6f2a46, §2.5]. The formula carries two messages. It grows like $n_1^{n_0}/n_0!$ for large width, i.e. polynomially in width but with the exponent set by the input dimension; and it is only valid for *general position* hyperplanes, a special arrangement that a trained network need not produce [@serraBoundingCountingLinear2018, §2].

Maxout layers have the same flavour of result with a different combinatorial object. A single rank-$k$ maxout unit computes the upper envelope of $k$ affine functions, so it has exactly $k$ linear regions; a layer of $m$ rank-$k$ maxout units therefore has a maximal count between $k^{\min\{n,m\}}$ and $\min\{\sum_{j=0}^{n}\binom{k^2 m}{j},\,k^{m}\}$ [@NIPS2014_fa6f2a46, §4, Prop. 8]. The lower end comes from an arrangement of parallel hyperplanes, the upper end from intersecting the $m$ units' partitions; the gap between them is the price of not knowing how the individual partitions line up [@NIPS2014_fa6f2a46, §4].

## Depth folds space: replication and exponential growth

The shallow count is polynomial in width. The step that made this literature famous is the observation that each extra layer can *replicate* an arrangement throughout the regions created before it, turning a product over layers into a growth that is exponential in depth at fixed width [@pascanuNumberResponseRegions2013, §2; @NIPS2014_fa6f2a46, §2.3].

The mechanism is *identification*. A map $F$ identifies two neighbourhoods $S,T$ if $F(S)=F(T)$; the canonical example is $(x_1,x_2)\mapsto(|x_1|,|x_2|)$, which identifies the four quadrants of $\R^2$ [@NIPS2014_fa6f2a46, Def. 1]. If the first $L-1$ hidden layers map many input neighbourhoods onto the *same* activations, then whatever the last hidden layer computes on those activations is copied into every one of them [@NIPS2014_fa6f2a46, §2.3]. Writing $N^l_R$ for the number of input neighbourhoods mapped onto a neighbourhood $R$ of layer $l$, this gives the recursion $N^l_R=\sum_{R'\in P^l_R}N^{l-1}_{R'}$ with $N^0_R=1$, and the maximal count is at least $\sum_{R\in P^L}N^{L-1}_R$ over neighbourhoods in distinct linear regions of the last hidden layer [@NIPS2014_fa6f2a46, Eq. 2, Lemma 3]. Montúfar et al. describe the same construction visually as *space folding*: a layer folds its input so that distant pieces coincide, and the composition of foldings folds the input space recursively [@NIPS2014_fa6f2a46, §2.4].

The elementary folding unit is a sum of rectifiers with alternating signs, $\tilde h_j(x)=[1,-1,1,\dots,(-1)^{p-1}]\,[h_1(x),\dots,h_p(x)]^\top$ with $h_r(x)=\max\{0,2wx-(r-1)\}$, which creates $p$ linear regions and identifies $p$ unit intervals of one input coordinate [@NIPS2014_fa6f2a46, §3.1, Eq. 3]. Choosing $p=\lfloor n/n_0\rfloor$ such units per coordinate produces $p^{n_0}$ identified hypercubes whose image is a single hypercube, on which the last layer then acts [@NIPS2014_fa6f2a46, §3.1].

Chaining this across layers gives the central lower bound. A rectifier network with $n_0$ inputs and $L$ hidden layers of widths $n_i\ge n_0$ realises at least

$$
\prod_{i=1}^{L-1}\left\lfloor\frac{n_i}{n_0}\right\rfloor^{n_0}\left(\sum_{j=0}^{n_0}\binom{n_L}{j}\right)
$$

linear regions [@NIPS2014_fa6f2a46, Thm. 5], which for equal width $n\ge n_0$ behaves as $\Omega\big((n/n_0)^{(L-1)n_0}\,n^{n_0}\big)$ [@NIPS2014_fa6f2a46, Cor. 6]. The independent-coordinate construction only needs $n_0$ units per layer and gives the cleaner count of at least $2^{(k-1)n_0}\sum_{j=0}^{n_0}\binom{2n_0}{j}$ for $k$ hidden layers of width $2n_0$ [@pascanuNumberResponseRegions2013, Thm. 2]. Its small cases make the depth advantage concrete: for $n_0=2$, a shallow model with $8$ hidden units has at most $37$ regions, while two layers of $4$ units produce $44$; with $12$ hidden units the shallow maximum is $79$ while three layers reach $176$, and a four-layer model with $24$ hidden units computes a function that a shallow model needs at least $137$ units to match [@pascanuNumberResponseRegions2013, §5, p. 13].

The same replication idea works for maxout. A rank-$k$ maxout unit identifies $k$ cones in its input; iterating the construction shows that a maxout network with $L$ layers of width $n_0$ and rank $k$ computes at least $k^{L-1}k^{n_0}$ regions [@NIPS2014_fa6f2a46, Thm. 9]. The two families are not nested: the rectifier constructions of Montúfar et al. are not computable by maxout networks except in the rank-$2$ case, and the authors explicitly leave open which architecture is more powerful by this measure [@NIPS2014_fa6f2a46, §4, §5].

Because a deep and a shallow network can be compared at equal *parameter* count rather than equal unit count, the advantage is also visible per parameter. A $k$-layer width-$n$ model has $O(kn^2)$ parameters and a region-to-parameter ratio of order $(\lfloor n/n_0\rfloor)^{k-1}n^{n_0-2}/k$, against $O(k^{n_0-1}n^{n_0-1})$ for a shallow model with $kn$ units — exponential in depth and polynomially better in width [@pascanuNumberResponseRegions2013, §6, Props. 7–8]. Pascanu et al. note that the ratio is large exactly when $n\gg n_0$, and speculate that data near a low-dimensional manifold is where depth should pay off [@pascanuNumberResponseRegions2013, §6].

## Sharper bounds and the bottleneck effect

The replication bounds are lower bounds from a clever construction; the upper side was improved by tracking the *dimension of each region's image* through the layers. If a region $S$ is linear up to layer $l$, the map is affine on it and $\dim(h^l(S))$ equals the rank of the effective weight matrix, and this dimension can only decrease as later layers partition $S$ further [@serraBoundingCountingLinear2018, §3.1]. Substituting this into Zaslavsky's bound gives the sharpened count at most

$$
\begin{aligned}
&\sum_{(j_1,\dots,j_L)\in J}\prod_{l=1}^{L}\binom{n_l}{j_l}, \\
&J=\big\{(j_l)\in\Z^L: 0\le j_l\le\min\{n_0,\,n_1-j_1,\dots,n_{l-1}-j_{l-1},\,n_l\}\big\}
\end{aligned}
$$

[@serraBoundingCountingLinear2018, Thm. 1]. The index set is the whole point: each $j_l$ is capped by the widths of *all* preceding layers after subtracting the dimensions already used up, so an early narrow layer limits how many regions a later wide layer can create [@serraBoundingCountingLinear2018, §3.1].

This yields the *bottleneck effect*: subtracting a neuron from an earlier layer lowers the bound at least as much as subtracting it from a later one, and for a two-layer network with $n_0>\max\{n_1,n_2\}$ moving a neuron from the first to the second layer strictly decreases the bound [@serraBoundingCountingLinear2018, §3.1, Prop. 2]. One consequence runs against the received "depth always wins" story: for $L\ge2$, $n\ge1$ and $n_0\ge Ln$, the exact maximum satisfies $R(n_0,n,\dots,n)<R(n_0,Ln)$, and the ratio tends to $0$ as $L\to\infty$ — when the input dimension is large relative to the neuron count, a deep network is *worse* than a shallow one of the same total width [@serraBoundingCountingLinear2018, Cor. 3]. So the depth advantage of the previous section is conditional on $n\gg n_0$, consistent with Pascanu et al.'s manifold remark; the two papers do not contradict each other, they delimit when each regime applies [@pascanuNumberResponseRegions2013, §6; @serraBoundingCountingLinear2018, §3.1].

When the input dimension is one, the bookkeeping becomes exact. Serra et al. show that a depth-$L$ rectifier network with $n_l\ge3$ units per layer and a one-dimensional input has maximal count exactly $\prod_{l=1}^{L}(n_l+1)$ [@serraBoundingCountingLinear2018, Thm. 7]. The construction keeps every region's image one-dimensional, so each layer multiplies the count by $n_l+1$ rather than the $n_l$ of the earlier replication argument [@serraBoundingCountingLinear2018, §3.2]. Correspondingly the general lower bounds improve by one region per replicated factor, under the stricter condition $n_l\ge3n_0$ [@serraBoundingCountingLinear2018, Thm. 8]. The maxout upper bound follows from the same technique, applied to the $\tfrac{k(k-1)}{2}n_l$ pairwise-difference hyperplanes that extend each maxout boundary segment: at most $\prod_{l=1}^{L}\sum_{j=0}^{d_l}\binom{\frac{k(k-1)}{2}n_l}{j}$ regions, with $d_l=\min\{n_0,\dots,n_l\}$ [@serraBoundingCountingLinear2018, Thm. 10].

## The depth–width trade-off

Taken together, the growth rates express a clean trade-off. Depth enters as a product of $L$ factors and can therefore grow the maximal count exponentially; width and the complexity of the activation enter only polynomially, with the per-layer degree capped by the input dimension [@goujonNumberRegionsPiecewise2023, §3.3]. The bottleneck effect says the same in reverse: a narrow layer near the input strangles the growth of every later layer, because it caps the dimension of the images that later layers can cut [@serraBoundingCountingLinear2018, §3.1]. This suggests a rule of thumb confirmed on small trained networks — for a fixed total number of units there is an intermediate depth that maximises the bound, and it depends on $n_0$ [@serraBoundingCountingLinear2018, §3.1].

The trade-off is not merely asymptotic. Montúfar et al.'s construction already beats the shallow count at small depth and width, and the margin widens with each extra layer [@NIPS2014_fa6f2a46, Cor. 6]. Conversely, Serra et al.'s counterexample regime shows that adding depth can *lose* regions when the input dimension dominates [@serraBoundingCountingLinear2018, Cor. 3]. So a practitioner choosing an architecture should not read "deep is exponentially more expressive" as unconditional; the correct statement pairs the depth exponent with the condition $n\gg n_0$ [@pascanuNumberResponseRegions2013, §6; @serraBoundingCountingLinear2018, §3.1].

!!! question "Open question"
    All of these are statements about the *maximum* over weight configurations, not about the
    networks that training finds. The gap between the two was left explicitly open by the early
    papers [@NIPS2014_fa6f2a46, §2.5], and closing it is the subject of the average-case work
    below.

## Beyond ReLU: general piecewise-linear activations

The ReLU-specific bounds leave the question of what happens for maxout, GroupSort, deepspline or PWLU activations. Goujon, Etemadi and Unser replace the arrangement of *hyperplanes* with an *arrangement of convex partitions*: given convex partitions $\Pi_1,\dots,\Pi_N$ of $\R^d$, its regions are the non-empty intersections $\bigcap_kP^{(k)}_{m_k}$ [@goujonNumberRegionsPiecewise2023, §2.3]. The maximal cardinality of such an arrangement is the polynomial

$$
\beta_d(n_1,\dots,n_N)=1+\sum_{k=1}^{\min(d,N)}\ \sum_{1\le\ell_1<\dots<\ell_k\le N}\ \prod_{q=1}^{k}(n_{\ell_q}-1),
$$

which reduces to the classical hyperplane count $\sum_{k=0}^{\min(d,N)}\binom{N}{k}$ when every partition has two regions [@goujonNumberRegionsPiecewise2023, Thm. 1, Eq. 4]. Sums and vectorisations of piecewise-linear functions are bounded by $\beta_d$ of their individual region counts, and both bounds are sharp [@goujonNumberRegionsPiecewise2023, Prop. 2]. Composition then multiplies these factors layer by layer, giving

$$
\prod_{\ell=1}^{L}\alpha_{\min(d_1,\dots,d_{L+1})}(\kappa_{\ell,\cdot})\ \le\ \kappa_{\max}\ \le\ \prod_{\ell=1}^{L}\beta_{\min(d_1,\dots,d_\ell)}(\kappa_{\ell,1},\dots,\kappa_{\ell,d_{\ell+1}})
$$

[@goujonNumberRegionsPiecewise2023, Thm. 2, Eq. 15], which for a network of width $W$ and uniform activation complexity $\kappa$ reads $(\kappa^{\lfloor W/d^*\rfloor})^{Ld^*}\le\kappa_{\max}\le(\kappa W)^{Ld^*}$ with $d^*=\min(d_{\text{in}},d_{\text{out}})$ [@goujonNumberRegionsPiecewise2023, Cor. 1, Eq. 17].

This framework explains the earlier results as special cases and marks their limits. For ReLU activations with two regions each, Eq. 15 reproduces the bound of Montúfar et al. — but the authors point out that it is *not* the tightest ReLU bound, precisely because the ReLU image of $(-\infty,0]$ is the singleton $\{0\}$, which reduces the effective dimension for any region that does not fire every neuron; that is the effect Serra et al. exploited [@goujonNumberRegionsPiecewise2023, §3.4]. For maxout, on the other hand, Eq. 15 improves the previously known bound [@goujonNumberRegionsPiecewise2023, §3.4, Eq. 20]. The lesson is that the same combinatorial machinery gives bounds of different tightness in different activation families, and the tightness is limited by how much of the neuron's output range is genuinely used [@goujonNumberRegionsPiecewise2023, §3.4].

## Depth lower bounds from rational weights

Counting regions bounds expressivity *from above*; a different question is how many layers a network *needs* to represent a given function exactly. Averkov, Hojny and Merkert study this for the test function $F_n=\max\{0,x_1,\dots,x_n\}$, which is the support function of the standard simplex [@averkovEXPRESSIVENESSRATIONALRELU2025, §1]. Arora et al. (outside the collection) showed that every continuous piecewise-linear function on $n$ inputs is exactly representable with $\lceil\log_2(n+1)\rceil$ hidden layers, and Hertrich et al. (outside the collection) conjectured that this is also necessary for $F_n$ [@averkovEXPRESSIVENESSRATIONALRELU2025, §1, Conj. 1]. The conjecture was known only for integer weights [@averkovEXPRESSIVENESSRATIONALRELU2025, §1].

The new contribution restricts weights to $N$-ary fractions, the numbers floating-point and quantised networks actually store [@averkovEXPRESSIVENESSRATIONALRELU2025, §1]. Clearing denominators reduces the problem to integer weights; the obstruction is then a modular invariant, the normalised volume of a lattice polytope taken modulo a prime $p$, which is Minkowski-additive in the relevant dimensions [@averkovEXPRESSIVENESSRATIONALRELU2025, §2]. This yields that every ReLU network with $N$-ary-fraction weights needs at least $\lceil\log_p(n+1)\rceil$ hidden layers to represent $F_n$, for any prime $p$ not dividing $N$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2]. For decimal weights the bound is $\lceil\log_3(n+1)\rceil$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Cor. 3], and for general $N$ it degrades to $C\,\ln n/\ln\ln N$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 4]. The authors are candid that this is weak — the target is $\Theta(\log_2 n)$ — but it is the first non-constant depth lower bound for the weight sets that hardware uses [@averkovEXPRESSIVENESSRATIONALRELU2025, §4]. Unlike the counting bounds, it accounts for neither width nor the number of regions; it is a different kind of obstruction, and the two should be read as complementary measures of expressivity [@averkovEXPRESSIVENESSRATIONALRELU2025, §4].

## Average case: typical networks have far fewer regions

Everything above is worst-case. Hanin and Rolnick asked what a random or trained network actually does, and found the answer is governed by the *number of neurons*, not the depth. For a network with input and output dimension one and random weights and biases of bounded mean gradient, the average number of regions along a line segment is proportional to the number of neurons times its length, $\E[\#\{\text{regions in }I\}]\approx|I|\cdot T\cdot\#\{\text{neurons}\}$, where $T$ is the number of breakpoints of the activation ($T=1$ for ReLU) [@haninComplexityLinearRegions2019, Thm. 1]. Depth does not enter: a deep and a shallow network with the same neuron count have the same average complexity along a curve [@haninComplexityLinearRegions2019, §2.1].

The same paper replaces raw counts with the volume of the region boundaries, which controls how far a typical point is from a boundary and hence how stable the function is. Its main theorem is an exact identity, not a bound: the expected $(n_{\rm in}-k)$-dimensional volume of the set where exactly $k$ neurons switch is an integral over the input space of a Jacobian norm times the joint bias density [@haninComplexityLinearRegions2019, Thm. 3, Eq. 8]. Under mild density assumptions this gives the boundary-volume density at most $\binom{\#\{\text{neurons}\}}{k}(C_{\rm grad}C_{\rm bias})^k$ per unit volume [@haninComplexityLinearRegions2019, Cor. 4] and the expected distance to the nearest boundary at least $c/(C_{\rm bias}C_{\rm grad}\#\{\text{neurons}\})$ [@haninComplexityLinearRegions2019, Cor. 5]. Experiments on MNIST confirm the scale: the distance to the nearest boundary stays between $0.4/\#\{\text{neurons}\}$ and $1.5/\#\{\text{neurons}\}$ throughout training, and the average region count stays within a small constant of its initialisation value, dipping slightly then rebounding [@haninComplexityLinearRegions2019, §3.1, §3.2].

The companion paper turns the same surprise into a clean density bound on activation regions. With bent hyperplanes $H_z=\{x:z(x)=b_z\}$, one per neuron, the non-empty activation regions are the connected components left after removing all of them [@haninDeepReLUNetworks2019, Def. 1, Lemma 2]. Under four mild distributional assumptions on weights and biases, the expected number of non-empty activation regions per unit volume inside a cube is at most $(T\,\#\{\text{neurons}\})^{n_{\rm in}}/n_{\rm in}!$, independent of depth [@haninDeepReLUNetworks2019, Thm. 5, Eq. 5]. The bound is sharp to leading order: the same argument in reverse shows the bent hyperplanes of every layer have the same normal-vector statistics under $2/{\rm fan\text{-}in}$ initialisation, so per neuron they contribute equally, and the depth-independent value is just the depth-one hyperplane-arrangement count [@haninDeepReLUNetworks2019, §3.3]. Because only the mild assumptions are needed, the bound holds throughout training, not just at initialisation [@haninDeepReLUNetworks2019, §3.1]. The experiments push the other way: even on memorisation tasks designed to reward more regions, and with random labels, the count never increases by more than a small constant factor from its initial value [@haninDeepReLUNetworks2019, §4]. The conclusion the authors draw is blunt — "deeper networks do not learn more complex functions than shallow networks" by these measures — and they explicitly do not offer a replacement explanation for the success of depth [@haninDeepReLUNetworks2019, §5].

This average-case picture extends beyond ReLU. Goujon et al. bound the expected *knot density* along any one-dimensional path by $D_0^\ast(\lambda_0 W L)$, where $\lambda_0$ bounds the expected density of each layer and $D_0$ its expected directional derivative [@goujonNumberRegionsPiecewise2023, Cor. 3, Eq. 35]. Here depth appears only linearly, through the factor $L$, so on average depth, width and activation complexity play identical roles and no exponential growth survives [@goujonNumberRegionsPiecewise2023, §4.3]. The gap between the worst-case exponential and the average-case linear is the central tension of the whole subject [@haninComplexityLinearRegions2019, §1; @goujonNumberRegionsPiecewise2023, §5].

## From counting to construction, and to convexity

A question in the other direction is: given a piecewise-linear function, how large a network does it need? Brandenburg, Grillo and Hertrich fix an underlying polyhedral complex and study decompositions $f=g-h$ into convex parts; the set of such decompositions is a polyhedron whose bounded faces are the reduced ones and whose vertices include every minimal decomposition [@brandenburgDecompositionPolyhedraPiecewise2024, §3]. The region count reappears as the size measure: a convex function with $k$ affine pieces can be represented with depth $\lceil\log_2(n+1)\rceil+\lceil\log_2 r\rceil+1$ and size $O(rs^{n+1})$ for any $r,s$ with $rs\ge k=q$, which interpolates between the known low-depth construction (depth $\lceil\log_2(n+1)\rceil+1$, size $O(k^{n+1})$) and the low-size one (depth $O(\log k)$, size $O(k)$) [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 6.3, §§6]. The non-convex case needs a regular compatible complex, and whether the number of pieces of $g$ and $h$ can always be polynomial in $n$ and $q$ is left open [@brandenburgDecompositionPolyhedraPiecewise2024, Cor. 6.4, Problem 7.1].

The region geometry also constrains which functions a network can represent *at all*. A convex piecewise-linear function need only be checked across *neighbouring* regions — those sharing a boundary of affine dimension $d-1$ — and this minimal set of conditions becomes, for a ReLU network, a nonnegativity condition on sums of products of weights along the paths through each hidden neuron: $f_\theta$ is convex iff $\min_{a\in\mathfrak{a}^{\nu\to}}\langle a,\Phi^{\nu\to}(\theta)\rangle\ge0$ for every hidden neuron $\nu$ [@gagneuxHowImproveExpressivity2025, §2.3, Thm. 2.3]. Input convex neural networks impose the stronger condition that every hidden weight be nonnegative; already for two hidden layers the two conditions differ, and convex ReLU networks strictly outnumber ICNNs of the same architecture [@gagneuxHowImproveExpressivity2025, §2.4.1, §2.4.2]. Sampling random weights confirms the gap empirically — convex ReLU networks vastly outnumber ICNNs at fixed width and depth, and among convex networks the fraction that are ICNNs drops to zero beyond three hidden layers [@gagneuxHowImproveExpressivity2025, §4.1]. A soft penalty built from the same boundary patterns recovers the accuracy of an unconstrained network (MSE $0.84\pm0.73$) while staying convex, against $1.71\pm0.90$ for the ICNN, on a 2D benchmark [@gagneuxHowImproveExpressivity2025, Eq. 5, Table 1].

## Summary

- The number of linear regions $\mathcal{N}(f)$ measures how many pieces an architecture can cut the input space into; it is an upper bound on flexibility, and the maximum is not a measure-zero accident [@pascanuNumberResponseRegions2013, §1; @NIPS2014_fa6f2a46, §2.5].
- A shallow rectifier layer with $n_1$ units and $n_0$ inputs attains $\sum_{j=0}^{n_0}\binom{n_1}{j}$ regions in general position, a Zaslavsky arrangement count; maxout layers have an analogous, looser bound [@pascanuNumberResponseRegions2013, Prop. 2; @NIPS2014_fa6f2a46, Prop. 8].
- Replication (space folding) makes the maximal count grow exponentially in depth and polynomially in width; the sharpest lower bound is $\Omega((n/n_0)^{(L-1)n_0}n^{n_0})$ [@NIPS2014_fa6f2a46, Cor. 6].
- Tracking the dimension of each region's image adds a bottleneck effect: a narrow early layer caps the count, and for large enough $n_0$ a deep network can be worse than a shallow one of the same total width [@serraBoundingCountingLinear2018, Thm. 1, Cor. 3].
- The depth–width trade-off is universal over piecewise-linear activations: depth enters as a product of $L$ factors, width and activation complexity polynomially [@goujonNumberRegionsPiecewise2023, Thm. 2, Cor. 1].
- Exact-representation depth lower bounds are a different obstruction: $N$-ary-fraction weights force at least $\lceil\log_p(n+1)\rceil$ hidden layers for $\max\{0,x_1,\dots,x_n\}$ [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2].
- Average case, the count is proportional to the number of neurons and independent of depth, both along lines and as a region density $(T\,\#\{\text{neurons}\})^{n_{\rm in}}/n_{\rm in}!$ [@haninComplexityLinearRegions2019, Thm. 1; @haninDeepReLUNetworks2019, Thm. 5].
- The same region geometry governs construction size and convexity constraints, not just the raw count [@brandenburgDecompositionPolyhedraPiecewise2024, Thm. 6.3; @gagneuxHowImproveExpressivity2025, Thm. 2.3].

## Further reading

- [@pascanuNumberResponseRegions2013]: the response-region framework, the replication proof and the small-depth numbers (§§2–5).
- [@NIPS2014_fa6f2a46]: the general folding framework and the sharpened rectifier and maxout lower bounds (§§2–4).
- [@serraBoundingCountingLinear2018]: the dimension bookkeeping, the bottleneck effect, the exact one-dimensional count, and mixed-integer exact counting (§§3–5).
- [@goujonNumberRegionsPiecewise2023]: the arrangement-of-convex-partitions bound for arbitrary piecewise-linear activations and the average-case knot density (§§3–4).
- [@averkovEXPRESSIVENESSRATIONALRELU2025]: depth lower bounds for rational weights via lattice polytopes (§§1–4).
- [@haninComplexityLinearRegions2019; @haninDeepReLUNetworks2019]: the average-case regime, boundary volume and activation-region density (§§2–4).
- [@brandenburgDecompositionPolyhedraPiecewise2024]: convex decompositions and the depth–size trade-off for representing a given function (§3, §6).
- [@gagneuxHowImproveExpressivity2025]: convexity as a region-boundary condition and the empirical rarity of ICNNs (§2, §4).

## Exercises

**Exercise 1** ★ (check). Why is the maximal number of linear regions of a shallow rectifier layer with $n_0$ inputs and $n_1$ units exactly $\sum_{j=0}^{n_0}\binom{n_1}{j}$, and when is this number actually attained?

??? success "Solution"
    The units' hyperplanes $H_i=\{x:W_{i,:}x+b_i=0\}$ form an arrangement, and the layer is
    affine on each region of that arrangement, so the regions of the layer are exactly the
    regions of the arrangement [@NIPS2014_fa6f2a46, §2.2]. The maximum number of regions of
    $n_1$ hyperplanes in $\R^{n_0}$ is $\sum_{j=0}^{n_0}\binom{n_1}{j}$ (Zaslavsky's theorem),
    so this is the maximal shallow count [@pascanuNumberResponseRegions2013, Prop. 2]. It is
    attained if and only if the $n_1$ hyperplanes are in general position; a trained network
    need not produce such a configuration [@pascanuNumberResponseRegions2013, §3; @serraBoundingCountingLinear2018, §2].

**Exercise 2** ★ (check). A rectifier network has a total of $N$ hidden units spread over any number of layers. Explain why it can never have more than $2^N$ linear regions, and why this bound does not see how the units are split between depth and width.

??? success "Solution"
    Each linear region is labelled by an activation pattern — one on/off bit per hidden unit —
    so there are at most $2^N$ distinct patterns and hence at most $2^N$ regions
    [@NIPS2014_fa6f2a46, Prop. 4; @haninComplexityLinearRegions2019, Eq. 1]. The argument
    counts units only, with no reference to how they are arranged, so it is identical for a
    single wide layer and for a deep narrow one with the same total $N$. It is a valid but loose
    upper bound: it is also the value at the *theoretical* extreme, far above what random or
    trained networks reach [@haninDeepReLUNetworks2019, §1].

**Exercise 3** ★ (check). In the rectifier replication construction of Montúfar et al., each of the first $L-1$ layers folds its input using $p=\lfloor n/n_0\rfloor$ units per input coordinate. Why does the resulting count carry a factor $p^{(L-1)n_0}$ rather than $p^{Ln_0}$?

??? success "Solution"
    The first $L-1$ hidden layers do the folding: each of the $n_0$ coordinates is folded by
    $p$ rectifiers of alternating sign, giving $p^{n_0}$ identified hypercubes, and this is
    repeated across the $L-1$ folding layers, producing $p^{(L-1)n_0}$ identified input
    neighbourhoods [@NIPS2014_fa6f2a46, §3.1]. The *last* hidden layer is not folded on top of
    this; it is the one that partitions the single folded space, and it contributes the
    general-position term $\sum_{j=0}^{n_0}\binom{n_L}{j}$ instead of a factor $p^{n_0}$
    [@NIPS2014_fa6f2a46, Thm. 5]. Hence the exponent is $L-1$, not $L$.

**Exercise 4** ★★ (derive). Using the average-case law $\E[\#\{\text{regions in }I\}]\approx|I|\cdot T\cdot\#\{\text{neurons}\}$, compare a deep and a shallow rectifier network with the same total number of neurons, and contrast the conclusion with the worst-case bound $\Omega\big((n/n_0)^{(L-1)n_0}n^{n_0}\big)$.

??? success "Solution"
    The average-case law is linear in the neuron count and contains no depth term, so two
    networks with the same $\#\{\text{neurons}\}$ have the same expected region count along any
    line segment $I$, whatever their depth [@haninComplexityLinearRegions2019, Thm. 1]. The
    worst-case bound, by contrast, grows exponentially in depth $L$ at fixed width $n$ and
    admits a depth advantage already at small $L$ [@NIPS2014_fa6f2a46, Cor. 6]. The two do not
    contradict: the first is about the *average* over random weights, the second about a single
    specially chosen weight assignment. The worst-case construction is numerically fragile, and
    random or trained networks stay near the linear average [@haninComplexityLinearRegions2019, §1].

**Exercise 5** ★★ (derive). For a one-dimensional input, Serra et al. prove the exact maximum $\prod_{l=1}^{L}(n_l+1)$. Take all layers of equal width $n$ and compare with the replication bound $n^{L-1}(n+1)$ obtained from Montúfar et al.; by what factor does the exact answer exceed it, and why?

??? success "Solution"
    With $n_l=n$ the exact maximum is $(n+1)^L$, while Thm. 5 of Montúfar et al. gives
    $\prod_{i=1}^{L-1}n\cdot\sum_{j=0}^{1}\binom{n}{j}=n^{L-1}(n+1)$ when $n_0=1$
    [@serraBoundingCountingLinear2018, Thm. 7; @NIPS2014_fa6f2a46, Thm. 5]. The ratio is
    $(n+1)^{L-1}/n^{L-1}=((n+1)/n)^{L-1}$, so Serra et al. gain one extra region per replicated
    layer. The reason is that their construction keeps the image of every region
    one-dimensional, whereas the coordinate-folding construction produces some regions whose
    image collapses and cannot be split further [@serraBoundingCountingLinear2018, §3.2].

**Exercise 6** ★★ (compare). [@NIPS2014_fa6f2a46] argues that depth is exponentially better than width at producing linear regions, while [@serraBoundingCountingLinear2018] shows a deep network can have strictly fewer regions than a shallow one of the same total width. Reconcile the two statements.

??? success "Solution"
    Montúfar et al. hold the input dimension $n_0$ fixed and let width $n\gg n_0$; in that
    regime the deep count $\Omega((n/n_0)^{(L-1)n_0}n^{n_0})$ grows exponentially in depth and
    beats a shallow network with the same $Ln$ units [@NIPS2014_fa6f2a46, Cor. 6]. Serra et al.
    consider the opposite regime, $n_0\ge Ln$: then the exact maximum satisfies
    $R(n_0,n,\dots,n)<R(n_0,Ln)$ and the ratio tends to $0$ as $L\to\infty$, so depth is worse
    [@serraBoundingCountingLinear2018, Cor. 3]. The reconciliation is the condition $n\gg n_0$
    versus $n_0\gg n$: depth pays only when the widths dominate the input dimension, which is
    also why the depth advantage is expected to be largest when data lie near a low-dimensional
    manifold [@pascanuNumberResponseRegions2013, §6].

**Exercise 7** ★★ (compare). [@averkovEXPRESSIVENESSRATIONALRELU2025] proves that decimal-weight ReLU networks need at least $\lceil\log_3(n+1)\rceil$ hidden layers to exactly represent $\max\{0,x_1,\dots,x_n\}$. Is this the same kind of statement as the region-counting bounds? What does each quantify?

??? success "Solution"
    No. The counting bounds quantify how many linear pieces a *fixed architecture* can realise
    as its weights vary, i.e. an upper bound on representational capacity
    [@NIPS2014_fa6f2a46, §1]. Averkov et al. instead prove a *lower* bound on the depth needed to
    represent one specific function exactly, when weights are restricted to $N$-ary fractions
    [@averkovEXPRESSIVENESSRATIONALRELU2025, Thm. 2, Cor. 3]. The proof is an obstruction via a
    modular invariant of a lattice polytope, not a region count, and it accounts for neither the
    width nor the number of regions of the network [@averkovEXPRESSIVENESSRATIONALRELU2025, §4].
    The two are complementary measures of expressivity, and their incomparability is itself
    instructive: "more regions" and "needs more depth" are different questions.

**Exercise 8** ★★★ (critique). You train a depth-3, width-32 fully-connected ReLU network to memorise random binary labels on 2D points. The worst-case theory says a network this size could have exponentially many regions; the average-case theory predicts about $\#\{\text{neurons}\}^{n_{\rm in}}/n_{\rm in}!$. Which do you expect to observe, and what experiment would settle the question?

??? success "Solution"
    Expect the average-case prediction: for $96$ neurons and $n_{\rm in}=2$,
    $(T\#\{\text{neurons}\})^{n_{\rm in}}/n_{\rm in}!=96^2/2=4608$ regions
    [@haninDeepReLUNetworks2019, Thm. 5]. Hanin and Rolnick train exactly such a network on
    random 2D labels and find the count rises above $4608$ only by a small constant factor,
    never toward the exponential maximum, even as the amount of memorisation grows
    [@haninDeepReLUNetworks2019, §4.2, §5]. A good experiment: count the regions exactly along
    cross-sections before and after training and compare with $4608$; the count should stay
    within a small factor, dipping and rebounding but not growing exponentially
    [@haninComplexityLinearRegions2019, §3.1]. Serra et al. make the same point from the
    optimisation side, reporting only an initial correlation between region count and accuracy
    on 22-neuron MNIST networks [@serraBoundingCountingLinear2018, §6].
