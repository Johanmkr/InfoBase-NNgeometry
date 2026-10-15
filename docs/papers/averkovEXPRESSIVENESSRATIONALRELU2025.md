---
citekey: averkovEXPRESSIVENESSRATIONALRELU2025
title: "On the Expressiveness of Rational ReLU Neural Networks with Bounded Depth"
authors: [Gennadiy Averkov, Christopher Hojny, Maximilian Merkert]
year: 2025
venue: International Conference on Learning Representations (ICLR) 2025
doi: null
arxiv: null
url: null
kind: theory
source_type: pdf
source_sha256: 7d810fc41e69e5d7ffe46eae8dcbb84c79a2a3240bdf045e04b55440adcc80c7
pages: 15
ingested: 2026-10-15
tags: [expressiveness, depth-separation, relu-networks, lattice-polytopes, rational-weights]
concepts_introduced:
  - rational ReLU network
  - N-ary fraction
  - modular volume obstruction for depth separation
  - lattice-polytope depth lower bound
concepts_used:
  - ReLU network
  - exact representation
  - support function
  - lattice polytope
  - normalized volume
  - mixed volume
  - Minkowski sum
  - sum-union closure
builds_on: []
datasets: []
---

# On the Expressiveness of Rational ReLU Neural Networks with Bounded Depth

**Averkov, Hojny, Merkert (2025)**, *International Conference on Learning Representations (ICLR) 2025*.

!!! abstract "TL;DR"
    The paper proves lower bounds on the number of hidden layers a ReLU network needs to *exactly* represent $\max\{0,x_1,\dots,x_n\}$ when its weights are restricted to $N$-ary fractions (the numbers that floating-point and quantised networks actually use). The proof clears denominators to reduce to integer weights, then obstructs shallow representations with a modular invariant: the normalized volume of a lattice polytope modulo a prime $p$. For decimal weights the bound is $\lceil\log_3(n+1)\rceil$ hidden layers; for $N$-ary weights it is $C\ln n/\ln\ln N$, the first non-constant depth lower bound for practically relevant networks.

## Problem and motivation

Understanding which functions a ReLU architecture can *exactly* represent is harder than understanding which it can approximate. Arora et al. (2018) show that the exactly representable functions are precisely the continuous piecewise linear (CPWL) functions, and that every such function on $n$ inputs is representable with $\lceil\log_2(n+1)\rceil$ hidden layers, but for which functions this depth is also necessary is open (§1, PDF p. 1). The key test function is

$$F_n = \max\{0,x_1,\dots,x_n\},$$

and Hertrich et al. (2021) conjecture that any ReLU network exactly representing $F_n$ needs at least $\lceil\log_2(n+1)\rceil$ hidden layers (Conjecture 1, PDF p. 1). Equivalently, if the conjecture holds, the upper bound of Arora et al. (2018) is tight (§1, PDF p. 1). It has been confirmed only for integer-weight networks (Haase et al., ICLR 2023) and for $H$-conforming networks with $n=4$ (Hertrich et al. 2021) (§1, PDF p. 2).

The contribution here is to replace integer weights by *$N$-ary fractions*: rationals of the form $z/N^t$ with $z\in\Z$ and $t\in\Z_+$ (§1, PDF p. 2). The authors stress that this is the practically relevant regime, since floating-point arithmetic (IEEE 754) and post-training quantisation produce exactly such low-precision weights (§1, PDF p. 2). A first lower bound on the depth of realistic ReLU networks is thereby obtained.

## Method

The proof extends the lattice-polytope argument of Haase et al. (2023) from the divisor $2$ to an arbitrary prime $p$, adding mixed volumes and number theory (§2, PDF p. 3).

1. **Clear denominators.** A network with rational weights sharing denominator $M$ can be rescaled so that $M^{k+1}f$ is representable with *integer* weights and the same number of hidden layers (Lemma 16, PDF p. 8). So a bound for integer weights transfers to fractional weights.
2. **Use a polyhedral characterisation.** Hertrich's theorem identifies positively homogeneous ReLU functions with differences of support functions of polytopes built by a "sum-union" closure $SU^k$ (Theorem 8, PDF p. 6). It reduces "is $h_P$ representable with $k$ hidden layers?" to a Minkowski-sum equation $P+A=B$ (Corollary 9, PDF p. 6).
3. **Find a modular invariant.** For $d$ a power of a prime $p$, the normalized volume $\mathrm{Vol}_d$ taken mod $p$ is Minkowski additive on $d$-dimensional lattice polytopes (Proposition 11, PDF p. 7). This yields divisibility: every $p^k$-dimensional face of any $P\in SU^k(\mathcal{P}_0(\Z^n))$ has $\mathrm{Vol}_{p^k}$ divisible by $p$ (Theorem 14, PDF p. 8), and consequently $p\mid \mathrm{Vol}_d(P)$ whenever $h_P$ is integer-representable with $k\le t$ hidden layers and $d=p^t$ (Theorem 15, PDF p. 8).
4. **Apply to the simplex.** Since $F_n$ is the support function of the standard simplex $\Delta_n$, whose normalized volume is $1$, and $1$ is not divisible by $p$, scaling the simplex by $N^{t(k+1)}$ gives a polytope whose volume is $\equiv 1 \pmod p$ yet whose support function would have to be integer-representable with $k$ layers — a contradiction (§3.3, PDF p. 9).

The conceptual message the authors draw is a general "separation strategy": to show $G_k \subsetneq G_{k+1}$ for nested classes of representable functions, find a homomorphism on the ambient group that vanishes on $G_k$ but not on some $f\in G_{k+1}$; here the group is integer-weight positively homogeneous ReLU functions and the invariant is volume mod $p$ (§4, PDF p. 10).

## Key equations and definitions

A rational number is an **$N$-ary fraction** if it equals $z/N^{t}$ for some $z\in\Z$, $t\in\Z_+$ (PDF p. 8). For $N=2$ and $N=10$ these are binary and decimal fractions.

A ReLU network with $k+1$ layers is a chain of affine maps $t^{(i)}(x)=A^{(i)}x+b^{(i)}$ composed with the component-wise ReLU $\sigma(x)=\max\{0,x\}$:

$$
t^{(k+1)}\circ\sigma\circ t^{(k)}\circ\sigma\circ\cdots\circ t^{(2)}\circ\sigma\circ t^{(1)} \qquad \text{(Eq. 2, PDF p. 1)}
$$

and *exactly represents* $f:\R^n\to\R$ when this composition equals $f$ everywhere; its *depth* is $k+1$, its *number of hidden layers* $k$ (PDF p. 1).

The support function of a polytope $P=\mathrm{conv}(p_1,\dots,p_m)$ is

$$
h_P(x)=\max\{\,x^\top y : y\in P\,\}=\max\{p_1^\top x,\dots,p_m^\top x\} \qquad \text{(§2.1, PDF p. 4)}
$$

so that $F_n = h_{\Delta_n}$ for the standard simplex $\Delta_n=\mathrm{conv}(0,e_1,\dots,e_n)$ (PDF p. 4). The normalized volume is $\mathrm{Vol}_n = n!\,\mathrm{vol}_n$, set to $1$ on $\Delta_n$, and takes non-negative integer values on lattice polytopes (PDF p. 4).

The volume of a Minkowski sum expands binomially in mixed volumes:

$$
\mathrm{Vol}_n(A+B)=\sum_{i=0}^{n}\binom{n}{i}\,V(\underbrace{A,\dots,A}_{i},\underbrace{B,\dots,B}_{n-i}) \qquad \text{(Eq. 1, PDF p. 4)}
$$

where the mixed volume $V$ is permutation-invariant, Minkowski-linear in every argument, and equals $\mathrm{Vol}_n$ on the diagonal (Theorem 5, PDF pp. 4–5).

Hertrich's characterisation, extended to a ring $R$: with $\mathcal{P}_0(S)$ the $0$-dimensional polytopes (points) with vertices in $S$, and the *sum-union closure*

$$
SU(\mathcal{X})=\Bigl\{\sum_{i=1}^{m}\mathrm{conv}(A_i\cup B_i) : m\in\N,\ A_i,B_i\in\mathcal{X},\ i\in[m]\Bigr\},
$$

one has $\mathrm{ReLU}_R^{n,0}(k)=\{\,h_A-h_B : A,B\in SU^k(\mathcal{P}_0(R^n))\,\}$ (Theorem 8, PDF p. 6). Equivalently $h_P\in\mathrm{ReLU}_R^{n}(k)$ iff $P+A=B$ for some $A,B\in SU^k(\mathcal{P}_0(R^n))$ (Corollary 9, PDF p. 6).

The modular additivity that drives everything (write $a\equiv_p b$ for "$a-b$ divisible by $p$"): if $d=p^t\le n$ is a prime power and $\sum_i P_i$ is again $d$-dimensional, then

$$
\mathrm{Vol}_d\Bigl(\sum_{i=1}^{m}P_i\Bigr)\equiv_p \sum_{i=1}^{m}\mathrm{Vol}_d(P_i) \qquad \text{(Prop. 11, PDF p. 7).}
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $F_n$ | $\max\{0,x_1,\dots,x_n\}$, the central test function |
| $N$ | base of the fraction ring ($10$ for decimals, $2$ for binaries) |
| $M$ | common denominator of a network's rational weights |
| $\mathrm{ReLU}_R^{n}(k)$ | $n$-variate functions representable with $k$ hidden layers and weights in ring $R$ |
| $\mathrm{ReLU}_R^{n,0}(k)$ | positively homogeneous part (all biases $0$) |
| $SU^k$ | $k$-fold sum-union closure of a polytope family |
| $h_P$ | support function of $P$ |
| $\mathrm{Vol}_d$ | $d$-dimensional normalized volume ($\mathrm{Vol}_d=n!\cdot$Lebesgue on $\R^d$) |
| $V(\cdot,\dots,\cdot)$ | mixed volume |
| $d=p^t$ | dimension equal to a prime power |
| $\equiv_p$ | congruence modulo $p$ |

## Main results

**Theorem 2** (PDF p. 2). Let $n,N$ be positive integers and $p$ a prime not dividing $N$. Every ReLU network with weights in the ring of $N$-ary fractions needs at least $\lceil\log_p(n+1)\rceil$ hidden layers to exactly represent $F_n$; a proof is given by clearing denominators (Lemma 16) and applying the divisibility theorem (Theorem 15) to the scaled simplex (§3.3, PDF p. 9).

**Corollary 3** (PDF p. 2). Every ReLU network with decimal weights needs at least $\lceil\log_3(n+1)\rceil$ hidden layers for $F_n$ (take $p=3$, which does not divide $10$). Since $F_n$ is representable with integral weights and $\lceil\log_2(n+1)\rceil$ hidden layers (Arora et al. 2018), this bound is optimal up to a constant factor (§3.3, PDF p. 9).

**Corollary 17** (PDF p. 9). For each $n\in\N$ and each integer $k\in\{1,\dots,\lceil\log_3 n\rceil\}$, among $n$-variate functions described by decimal-weight ReLU networks there are functions representable with $2k$ but not with $k$ hidden layers. Thus expressiveness with decimal weights grows gradually over depths $k=1,\dots,O(\log n)$.

**Theorem 4** (PDF p. 2). There is a constant $C>0$ such that, for all integers $n,N\ge 3$, every ReLU network with $N$-ary-fraction weights representing $F_n$ has depth at least
$$C\,\frac{\ln n}{\ln\ln N}.$$
The proof picks the least prime $p$ not dividing $N$ and uses the prime number theorem to bound $p\le p_{t+1}\approx t\ln t$ in terms of the number $t$ of distinct prime factors of $N$ (§3.3, PDF pp. 9–10). In particular there is no constant-depth ReLU network exactly representing $F_n$ with rational weights sharing a common denominator $N$ (§1, PDF p. 2).

There are no experiments, benchmarks or datasets: the paper is purely theoretical.

## Limitations stated by the authors

The results do not resolve Conjecture 1 because they say nothing about general real weights (§1, PDF p. 2; §3.3, PDF p. 9). Even for rational weights the achieved bound $C\ln n/\ln\ln N$ is weak: replacing $\ln\ln N$ by a constant would confirm the conjecture for rational weights up to a constant multiple, but $\ln\ln N$ grows extremely slowly (§4, PDF p. 10). The scaling trick used to move from integer to fractional weights is noted not to follow from Haase et al. (2023), whose own discussion (Sec. 1.3) rules out a naive scaling argument (§1, PDF p. 2). The authors also pose **Question 18**: for a subfield $S\subseteq\R$ (in particular $S=\Q$), if $f$ is real-weight representable with $k$ hidden layers and $f$ itself uses weights in $S$, must it also be $S$-weight representable with $k$ layers? A positive answer for $\Q$ would make the rational version of Conjecture 1 equivalent to the original (§4, PDF p. 10).

## Critical assessment

The paper is a crisp, self-contained extension of the integer-weight result of Haase et al. (2023), and its framing is its strongest asset: by restricting weights to the numbers floating-point and quantised hardware actually represent, it gives the first non-constant depth lower bound for architectures as deployed, rather than for an idealised real-weight model. The mechanics are sound and transparent — clear denominators, then obstruct with a Minkowski-additive invariant — and the mixed-volume route to Proposition 11 is genuinely shorter than the parity argument it generalises.

Two caveats temper the significance. First, the gap to Conjecture 1 is large: the target is $\Theta(\log_2 n)$ but Theorem 4 delivers only $\Theta(\ln n/\ln\ln N)$, and the dependence on $N$ is so slow that the bound is meaningful mainly as a non-constancy statement. Second, the reduction to integer weights via Lemma 16 inflates the weights by $N^{t(k+1)}$; the proof is existential and gives no control on width or weight magnitude, so it does not separate depth from width in any size-bounded sense. Vardi & Shamir (2020), cited by the authors, show that hardness of depth separation for polynomial-size networks is a real barrier, which the paper does not attempt to cross. The claim that the decimal bound $\lceil\log_3(n+1)\rceil$ is "optimal up to a constant factor" holds against the integral construction of Arora et al. (2018), not against general real-weight networks, and should be read in that light. The proposed "separation strategy" (§4) is more a template than a theorem: no example beyond the volume-mod-$p$ invariant is exhibited, and its remark that the invariant must be "global" (width-blind) is an informal design heuristic. The paper is fully self-contained, with the deferred lemmas (Lemma 10, Proposition 13, the mixed-volume binomial formula (1)) proved in the appendix, so it is straightforward to verify, though no code or data accompany it (none is expected).

## Connections within the collection

The paper is one entry in the collection's thread on the *combinatorial and polyhedral geometry of ReLU networks*. Its direct ancestors (Arora et al. 2018; Hertrich et al. 2021; Haase et al. 2023; Hertrich 2022) are outside the collection. Within it:

- [@raghuExpressivePowerDeep2017] is cited in the paper's related-work survey as one of the works relating width and depth of ReLU networks (§1, PDF p. 3) — a passing contextual mention, not a component of the proof.
- The lattice-polytope and polyhedral-decomposition machinery is the same subject as [@brandenburgDecompositionPolyhedraPiecewise2024] and [@goujonNumberRegionsPiecewise2023], which study polyhedral decompositions of piecewise linear functions and the number of linear regions; the authors of the former share a co-author with the present line of work, so it is possibly related, though the present paper neither cites nor uses it.
- The paper's subject — exact representability and depth — is a complement to the region-counting literature in this collection ([@NIPS2014_fa6f2a46; @pascanuNumberResponseRegions2013; @serraBoundingCountingLinear2018; @haninDeepReLUNetworks2019; @haninComplexityLinearRegions2019]), which bounds expressiveness by the *number* of linear regions rather than by a depth lower bound; the present modular-volume invariant is a different kind of obstruction and accounts for neither width nor region counts.
- The question of when two networks compute the same function is treated geometrically elsewhere in the collection ([@phuongFunctionalVsParametric2019]); here the relevant equivalence is between a CPWL function and the difference of two support functions.
