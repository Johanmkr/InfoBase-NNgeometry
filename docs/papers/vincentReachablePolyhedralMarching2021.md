---
citekey: vincentReachablePolyhedralMarching2021
title: "Reachable Polyhedral Marching (RPM): A Safety Verification Algorithm for Robotic Systems with Deep Neural Network Components"
authors: [Joseph A. Vincent, Mac Schwager]
year: 2021
venue: "2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 9029–9035"
doi: "10.1109/ICRA48506.2021.9561956"
arxiv: null
url: null
kind: method
source_type: pdf
source_sha256: bc857e70651620a5d633caf9db1e27ea9a51bd73fc976580cb36126d96dfaea8
pages: 7
ingested: 2026-10-15
tags: [neural-network-verification, reachability-analysis, relu-networks, polyhedral-geometry, safety-verification]
concepts_introduced:
  - reachable polyhedral marching
  - neighboring activation pattern
  - neighbor constraint
  - type 1, 2, 3 neurons
  - anytime safety verification
  - forward and backward reachable set computation
concepts_used:
  - piecewise-affine function
  - activation pattern
  - polyhedral complex
  - H-representation
  - essential and redundant constraints
  - polyhedral projection
  - exact reachability
builds_on: [NIPS2014_fa6f2a46, haninDeepReLUNetworks2019, haninComplexityLinearRegions2019]
datasets: [ACAS Xu, damped pendulum]
---

# Reachable Polyhedral Marching (RPM): A Safety Verification Algorithm for Robotic Systems with Deep Neural Network Components

**Vincent, Schwager (2021)**, *ICRA*. [doi:10.1109/ICRA48506.2021.9561956](https://doi.org/10.1109/ICRA48506.2021.9561956)

!!! abstract "TL;DR"
    RPM computes the exact piecewise-affine (PWA) representation of a ReLU network by *marching* over the polyhedral cells of the input space: starting from one cell, it derives each neighbour's activation pattern, and hence its polyhedron, from a local neuron-flipping rule. Because it never iterates layer-by-layer, it is anytime — it can stop the moment an unsafe cell is found — and stores only one activation bit-vector per cell instead of every intermediate polyhedron. On the ACAS Xu benchmark it finds unsafe inputs up to ~600× faster than the fastest exact baseline (0.01 s vs 6.66 s) and certifies safety about twice as slowly, and it solves a 50-step pendulum reachability problem in 87 s.

## Problem and motivation

Exact reachability for a ReLU network — the forward image of an input set, $Y=\{F(x)\mid x\in X\}$, or the backward preimage of an output set, $X=\{x\mid F(x)\in Y\}$ (§III, Eqs. 5–6, p. 2) — is needed to certify safety of learned perception, dynamics or control components in a robot's perception–action loop. Existing *exact* methods ([1]–[3] in the paper) all iterate layer-by-layer, applying affine transformation, intersection and projection at each layer. The authors argue this is limiting for two reasons: (i) the whole reachable set must be computed even when a single unsafe input already settles the verification question, and (ii) every intermediate polyhedron and affine map must be kept in memory, since later neurons may still split them, and the number of regions is hard to predict before runtime (§II, p. 2). RPM is instead a *marching* algorithm: it enumerates one polyhedral cell of the input space at a time, in the spirit of fast marching methods in optimal control and graphics (§I, p. 1). Each cell is found by a sequence of linear programs equal to the number of neurons, independent of network width or depth, and marching continues until the enumerated cells cover the sought reachable set.

## Method

The network is first put in bias-free augmented form, $F(\tilde x):\R^{k_0+1}\to\R^{k_n}$ with $\tilde x=[x_1,\dots,x_n,1]^\top$ (§III, Eq. 3, p. 2). For a fixed activation pattern $AP$ of the hidden neurons, the network collapses to a single affine map $C x + d$, so each pattern corresponds to one cell of a polyhedral complex and each cell yields the pair $(C_i,d_i)$ of the explicit PWA representation (§IV-A, Eqs. 4, 8). The steps are:

1. **Cell from a pattern.** Each hidden neuron gives one halfspace constraint; together they form an H-representation of the cell (§IV-A, Eq. 9). Constraints are the (nonlinear) functions $a_{ij}^\top(x)x \le b_{ij}(x)$ built by propagating the neuron's weights to the input layer (Eq. 10).
2. **Minimal representation.** Constraints are normalised, duplicates removed, and redundant ones stripped: constraint $i$ is redundant if solving $\max a_i^\top x$ s.t. the other constraints $\tilde A x\le\tilde b$ (Eq. 13) yields an optimum $\le b_i$ (§IV-B). A bounding-box heuristic finds about 90% of redundant constraints cheaply; the LP is solved with GLPK/JuMP.
3. **Neighbouring pattern.** For each essential constraint (the *neighbor constraint* defining $c\cap c'$), neurons are classified as **Type 1** (constraint defines $c\cap c'$), **Type 2** (degenerate, $0^\top x\le 0$ on all of $c$) or **Type 3** (strictly satisfied elsewhere) (Def. 5). Algorithm 1 then walks the network layer by layer and sets each Type 1/Type 2 neuron's activation to zero if its new constraint is $0^\top x\le0$, and flips it otherwise. Type 3 neurons never flip. This loop and one logical check per neuron is the entire neighbour-generation cost.
4. **March.** Algorithm 2 keeps a working set of activation patterns; the start cell is evaluated from an arbitrary input point, then each essential constraint spawns a neighbour that is pushed onto the working set unless already visited or queued. Since each cell is connected to another by a shared face, the process enumerates every cell in the input space and returns the explicit PWA representation.

Reachability then reuses this enumeration. For forward reachability each cell's polyhedron is projected through its affine map, $P_{out}=\{y\mid AC^{-1}y\le b+AC^{-1}d\}$ when $C$ is invertible (Eq. 16) and by block elimination otherwise (Eq. 17); for backward reachability the map is applied to the output set, $P_{in}=\{x\mid ACx\le b-Ad\}$ (Eq. 18b) and intersected with the cell (§V-A/B). Multiple backward sets are handled simultaneously at the cost of one extra intersection each. To reach beyond one step, the iterated system $x_{t+1}=F(x_t)$ is replaced by the concatenated network $F^T=F\circ\dots\circ F$ with $TN$ neurons and $TL$ layers, and RPM runs on that (§V, p. 5).

## Key equations and definitions

Affine map of a cell (Eqs. 8a–8b, p. 3) and its governing constraints (Eqs. 9–10, p. 3):

$$
C_i,d_i = \tilde W_n \prod_{i=1}^{n-1}\tilde W^c_i, \qquad \tilde W^c_i(x) = \operatorname{diag}(AP_i(x))\,\tilde W_i
$$

$$
[a_{ij}\ -b_{ij}] = \tilde W_i[j,:]\prod_{l=1}^{i-1}\tilde W_l, \qquad
\begin{cases} a_{ij}^\top x\ge b_{ij} & AP_{ij}=1\\ a_{ij}^\top x\le b_{ij} & AP_{ij}=0\end{cases}
$$

The first line builds the affine map of a cell by zeroing the rows of each hidden weight matrix whose neuron is inactive; the second gives, for a single neuron, the halfspace that bounds its cell. Note the paper writes the constraint as $a_{ij}^\top x \ge b_{ij}$; carrying the sign into $b_{ij}$ gives the form $a_{ij}^\top x\le b_{ij}$ used for the H-representation.

H-representation of a cell (Eq. 11, Def. 2, p. 3):

$$
PH = \{x\in\R^d \mid Ax\le b\}
$$

Each constraint is tested for redundancy by a linear program (Eq. 13, p. 3):

$$
\max_x a_i^\top x \quad \text{subject to } \tilde A x \le \tilde b
$$

Constraint $i$ is redundant if this optimum is $\le b_i$ (Def. 4). The neighbour condition over $c\cap c'$ (Eqs. 14a–14b, p. 4):

$$
a_{ij}^\top(x)\,x = b_{ij}(x)\ \ (ij\in\text{Type 1}), \qquad a_{ij}^\top(x)\,x = b_{ij}(x)\ \ (ij\in\text{Type 2})
$$

| Symbol | Meaning (in this paper) |
|---|---|
| $z_i$ | hidden layer variable of layer $i$; $z_0=x$, $z_n=y$ |
| $\hat z_i$ | preactivation value $W_i z_{i-1}+b_i$ |
| $\tilde W_i$ | augmented (bias-free) weight matrix of layer $i$ |
| $AP_{ij}(x)$ | activation of neuron $j$ in layer $i$: 1 if $\hat z_{ij}>0$, else 0 |
| $c_i$ | polyhedral cell over which the network is affine |
| $C_i, d_i$ | affine map $C_i x+d_i$ of cell $c_i$ |
| $(a_{ij},b_{ij})$ | halfspace constraint contributed by neuron $ij$ |
| Type 1 / 2 / 3 | neurons whose constraint defines $c\cap c'$ / is degenerate on $c$ / is strictly satisfied on $c$ |
| $F^T$ | the $T$-fold composition of $F$, a network with $TN$ neurons |

## Main results

**Pendulum.** A ReLU network with a single hidden layer of 12 neurons approximates the discrete-time damped-pendulum dynamics with a 0.1 s time step, $x_{t+1}=f_{nn}(x_t)$, $x=[\theta,\dot\theta]^\top$ (§VI-A, Eq. 20). Composing it 50 times gives 600 neurons for $x_{t+50}$. Forward reachability from $-90^\circ\le\theta\le90^\circ$, $-90^\circ/\text{s}\le\dot\theta\le90^\circ/\text{s}$ enumerates **2781 regions and takes 87 s** on a 2013 Dell Latitude E6430s laptop (Intel Core i7 3 GHz, 16 GB RAM); the reachable set is contained in the initial set, proving forward invariance over 50 steps (Fig. 4, p. 6). Backward reachability to the target $\lvert\theta\rvert\le5^\circ$, $\lvert\dot\theta\rvert\le2^\circ/\text{s}$ after 50 steps returns only positive initial angles, whereas the true dynamics would be symmetric about zero — the authors read this as a modelling artifact of the learned network (Fig. 5, p. 6).

**ACAS Xu, Property 3.** Each ACAS Xu advisory network has 5 inputs, 5 outputs and 300 neurons (§VI-B). Property 3 fails if the network ever outputs "Clear of Conflict" when the intruder is directly ahead and closing; a nonempty backward reachable set therefore witnesses a safety violation. RPM is compared against the fastest exact method, the face-lattice approach [3], and against Reluplex and Marabou, all single-core (Tab. I, p. 6):

| Network | Result | RPM (s) | Face Lattice (s) | Reluplex (s) | Marabou (s) |
|---|---|---|---|---|---|
| N1,7 | Unsafe | 0.01 | 6.66 | 2.15 | 0.70 |
| N1,8 | Unsafe | 0.01 | 5.45 | 4.32 | 1.49 |
| N3,8 | Safe | 19.72 | 9.35 | 231.28 | 40.13 |
| N5,6 | Safe | 31.92 | 17.28 | 366.39 | 54.07 |

The unsafe cases are settled in 0.01 s by RPM versus 5.45–6.66 s for the face-lattice method (the "many times faster" claim of the abstract); the safe cases cost about twice the face-lattice time, but still 7–12× less than Reluplex and 2× less than Marabou. The paper notes the face-lattice method has a parallel version and that RPM could be parallelised too (§VI-B, future work).

## Limitations stated by the authors

- Memory advantage is stated in relative terms: RPM is claimed to need less memory than layer-by-layer methods, not to have a small absolute footprint, and the number of affine regions remains hard to estimate before runtime (§II, p. 2).
- No parallel implementation is provided; parallelisation is left to future work (§VI-B, §VII).
- When no unsafe input exists, RPM is about 2× slower than the face-lattice method — the anytime advantage only pays off for violating properties (§VI-B).

## Critical assessment

The paper's core claim is a genuine algorithmic inversion — enumerate input-space cells instead of propagating sets through layers — and the two implementations the paper is strongest on (unsafe ACAS Xu properties) show the anytime property is real, three orders of magnitude on N1,7/N1,8. The evidence for the marching machinery itself is thin in places: the "about 90%" redundant-constraint reduction by the bounding-box heuristic (Tab. I paragraph, §IV-B, p. 3) is reported without an ablation or a per-cell measurement, and the claim that no other heuristic improves on it is asserted from the authors' experience rather than shown. The comparison rests on four of the 45 ACAS Xu networks, one property, and a single hardware configuration; a stronger case would report per-property behaviour and the cell counts visited. My reading is that the 87 s pendulum result is the more consequential one for this collection, because it demonstrates that iterating a network through a feedback loop — composing it with itself 50 times — is practical when one enumerates cells rather than layers, and the model artifact it surfaces (broken angle symmetry) is a concrete example of how exact geometry can be used for diagnosis rather than only verification. The code is released (github.com/StanfordMSL/Neural-Network-Reach), which supports the reproduction of Table I. The main open question is scaling: cell enumeration is inherently exponential in the number of regions, so the method's advantage over layer-by-layer exact reachability is a constant-factor (and anytime) one, not a complexity-class one — for high-dimensional inputs like images the enumeration could remain intractable, and the paper's examples are all low-dimensional state spaces.

## Connections within the collection

RPM rests on the fact that a ReLU network is continuous and PWA with a polyhedral partitioning, which the paper cites to [@NIPS2014_fa6f2a46] and related works (§III, p. 2); the explicit PWA representation and the number of regions are exactly the objects studied in this collection by [@serraBoundingCountingLinear2018], [@goujonNumberRegionsPiecewise2023] and [@berzinsPolyhedralComplexExtraction2023]. The practical motivation — that the *number of activation patterns* is much smaller than $2^N$ and better approximated by a polynomial — comes from [@haninDeepReLUNetworks2019] (§IV-A, p. 3), and the layer-by-layer route to an explicit PWA representation, which RPM sets itself against, is that of [@haninComplexityLinearRegions2019] (§II, p. 2). Its marching over neighbouring cells is the discrete analogue of the sign-sequence / canonical-complex enumeration in [@masdenAlgorithmicDeterminationCombinatorial2022], which likewise builds the cell structure from local face adjacencies rather than layer propagation; unlike RPM, that line of work targets the combinatorial topology, while RPM targets exact reachable sets. The explicit-data approach to geometry in [@humayunSplineCamExactVisualization2024; @sattelbergLocallyLinearAttributes2023] solves a related "enumerate the linear regions" problem for visualisation and attribute analysis, and shares RPM's cell-based view of the network, though it does not aim at verification.
