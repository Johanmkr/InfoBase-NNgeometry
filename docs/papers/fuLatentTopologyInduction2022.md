---
citekey: fuLatentTopologyInduction2022
title: "Latent Topology Induction for Understanding Contextualized Representations"
authors: [Yao Fu, Mirella Lapata]
year: 2022
venue: arXiv preprint
doi: 10.48550/arXiv.2206.01512
arxiv: "2206.01512"
url: null
kind: method
source_type: pdf
source_sha256: f77e3d0d388ab75273f2ed5b31e0669c342574a8f2a3855520e8a94ca06d0549
pages: 18
ingested: 2026-10-15
tags: [representation-topology, latent-variable-models, contextual-embeddings, unsupervised-probing, crf-vae]
concepts_introduced:
  - latent state network
  - CRF-VAE topology induction
  - state-word emission potential
  - state-state transition potential
  - function states and content states
  - sentence encoding as traversal
  - mechanism of contextualization
concepts_used:
  - contextualized word representations
  - pretrained language model
  - conditional random field
  - variational autoencoder
  - ELBO
  - Viterbi decoding
  - t-SNE
builds_on: []
datasets: [20News]
---

# Latent Topology Induction for Understanding Contextualized Representations

**Fu, Lapata (2022)**, *arXiv preprint*. [arXiv:2206.01512](https://arxiv.org/abs/2206.01512)

!!! abstract "TL;DR"
    The paper induces, without any linguistic supervision, a network of latent states that
    summarises the topology of a frozen language model's contextualized representation
    space, by fitting a structured (CRF) variational autoencoder whose only learnable
    parameters are the state embeddings. The induced states decompose the space into
    lexical/morphological/syntactic/semantic anchors, and their transitions form a backbone
    over which sentences are represented as traversals: transition chains encode syntactic
    templates, state-word emissions fill in the content.

## Problem and motivation

The paper asks what *intrinsic* topology a contextualized representation space carries — i.e.
how the word vectors of sentences are positioned relative to one another — rather than which
well-defined annotation a probe can recover (§1, p. 1). The authors argue that the mainstream
"Bertology" probing protocol, which fine-tunes a classifier on human annotations such as
part-of-speech, dependency or CCG supertags, cannot separate properties intrinsic to the model
from properties imposed by the supervision signal (§2, p. 2). Their setting is deliberately
minimal: a *fixed, not fine-tuned* pretrained encoder (BERT-base-uncased), so that any
structure discovered lives inside the representation manifold $\mathcal{M}$ of the model
itself (§3, p. 3). They hypothesize that $\mathcal{M}$ contains a spectrum of $N$ static
latent states $s_1,\dots,s_N$ that act as anchors — landmarks that mark neighbours and
connectivity — and that two relations suffice to describe $\mathcal{M}$: state–word
(emission) and state–state (transition). Interpretations are explicitly *not* required to
align with existing annotations, as long as the states are intrinsically meaningful (§1, p. 1).

## Method

**Model.** Each contextualized vector $r_t$ of a token $x_t$ is assigned a latent state index
$z_t \in \{1,\dots,N\}$; the state embeddings $s = [s_1,\dots,s_N]$ are the *only* learnable
parameters of the inference model, everything of the PLM being frozen (§3, p. 3). Emission
and transition potentials are bilinear in the (fixed) representation and the (learned) state
embeddings, forming a linear-chain CRF; its posterior $q(z \mid x)$ is the inference network
of a CRF-VAE. On top sits an LSTM decoder that reconstructs the sentence, so that maximising
the likelihood forces the states to be "generative" — able to predict sentences and their own
successors (§3, p. 3). After training the decoder is discarded and only the inferred states
are analysed.

**Algorithm (as described, §3–§4).**

1. Encode the 20News corpus with frozen BERT-base-uncased; take the last-layer outputs
   (BERT$_{\text{LAST}}$); the same method applies to any PLM (§4, p. 4).
2. Optimise the ELBO (Eq. 5) over the $N$ state embeddings with approximate gradients and
   entropy from Fu et al. (2022), which make the discrete CRF differentiable and
   memory-efficient; tune the entropy regulariser to avoid posterior collapse (§4, p. 5).
3. Decode state sequences with Viterbi; a state is said to *align* with a tag if ≥90% of the
   word occurrences it covers carry that tag (§5.1, p. 5).
4. Compare against POSITIONAL (states induced from positional embeddings), RANDEMB (random
   states with BERT$_{\text{LAST}}$ mean/variance) and BERT$_{\text{ZERO}}$ (zeroth-layer
   static embeddings) to isolate what contextualization adds (§4, p. 5).

**Setting.** 768-dimensional states, decoder hidden size 200, $N=2000$ states, 20News dev
set, Nvidia 2080Ti (§4, pp. 4–5). $N$ is a "sweet point": 10K under-clusters, 100
over-clusters (§4, p. 5).

## Key equations and definitions

$$
\log \phi(x_t, z_t) = r_t^{\top} s_{z_t}, \qquad
\log \psi(z_{t-1}, z_t) = s_{z_{t-1}}^{\top} s_{z_t}
$$

The emission potential scores how well state $z_t$ summarises token $x_t$; the transition
potential scores the affinity between consecutive states (Eq. 1, p. 3). Both use the dot product, "the
common practice of fine-tuning contextualized representations" (§3, p. 3).

$$
q(z \mid x) = \frac{1}{Z} \prod_{t=1}^{T} \psi(z_{t-1}, z_t)\, \phi(x_t, z_t)
$$

Posterior of a state sequence given a sentence (Eq. 2, p. 3); $Z$ is the partition function.

$$
p(x, z) = \prod_{t} p(x_t \mid z_{1:t})\, p(z_t \mid z_{1:t-1}),
\qquad h_t = \mathrm{Dec}(s_{z_{t-1}}, h_{t-1})
$$

$$
p(z_t \mid z_{1:t-1}) = \mathrm{softmax}\big(\mathrm{FF}([s_{z_t}; h_t])\big), \qquad
p(x_t \mid z_{1:t}) = \mathrm{softmax}\big(\mathrm{FF}(h_t)\big)
$$

Generative model (Eqs. 3–4, p. 3); $\mathrm{Dec}$ is an LSTM, $\mathrm{FF}$ a feed-forward network, $\theta$
the decoder parameters (§3, p. 3).

$$
\mathcal{L}_{\text{ELBO}} = \E_{q(z \mid x)}\big[\log p(x, z)\big] + H\big(q(z \mid x)\big)
$$

Training objective (Eq. 5, p. 3); the entropy term $H$ is what is reweighted to prevent posterior collapse
(§4, p. 5). (The sign of the $H$ term is not recoverable from the file's extracted text — see
critical assessment.)

| Symbol | Meaning (in this paper) |
|---|---|
| $x = [x_1;\dots;x_T]$ | input sentence, token sequence |
| $r_t \in \R^{768}$ | contextualized representation of token $t$ |
| $\mathcal{M}$ | representation manifold (space of all $r$) |
| $N$ | number of latent states (set to 2000) |
| $z_t \in \{1,\dots,N\}$ | latent state index assigned to token $t$ |
| $s_i$ | learnable embedding of state $i$ (the only learned parameters) |
| $s = [s_1;\dots;s_N]$ | all state embeddings |
| $\phi(x_t,z_t)$ | emission potential (state–word relation) |
| $\psi(z_{t-1},z_t)$ | transition potential (state–state relation) |
| $q(z\mid x)$ | CRF posterior over state sequences |
| $Z$ | partition function |
| $h_t$ | LSTM decoder hidden state |
| $\mathrm{FF}(\cdot)$ | feed-forward network |
| $\theta$ | decoder parameters |

## Main results

**The states exist and are meaningful.** Aligning the induced states with five annotation
types — POS, ENT, DEP, CCG, BCN (Dalvi et al.'s BERT Concept Net) — gives Table 1 (p. 4).
Isolating the LSTM's own fitting power, RANDEMB aligns 392 POS states covering 53.02% of
occurrences, so a non-trivial alignment is not evidence by itself. BERT$_{\text{LAST}}$ has
*more* aligned states than BERT$_{\text{ZERO}}$ (804 vs 673 POS-aligned) but covers *fewer*
word occurrences (53.23% vs 65.31%); the not-aligned share rises from 27.69%
(BERT$_{\text{ZERO}}$) to 40.52% (BERT$_{\text{LAST}}$). POSITIONAL is near-degenerate (3
POS-aligned states, 0.44%; 1997 not aligned, 99.55%).

**Non-aligned states are still interpretable.** Three annotators (graduate students with
100+ TOEFL scores) labelled non-aligned states, averaged over three annotators (Tab. 2, p. 4):
BERT$_{\text{ZERO}}$ covers more lexical information (LEX 83 states / 30.37% of occurrences;
SYN 54 / 20.96%; SEM 135 / 37.31%; not interpretable 39 / 11.37%), while
BERT$_{\text{LAST}}$ covers more semantic information (LEX 26 / 15.93%; SYN 40 / 26.25%; SEM
56 / 43.50%; N.I. 22 / 14.33%). So only ~11–14% of non-aligned states are uninterpretable,
even though the authors had selected the most-aligned checkpoint over hyperparameters and
seeds (§5.1, p. 5). Example states (Fig. 1, p. 2; Tab. 3, p. 4) include LEX "win/won/wins",
morphosyntax "[s]-suffix plural", "[n't]-suffix negation", and SEM "Religion", "Sports".

**The mechanism of contextualization.** Comparing BERT$_{\text{ZERO}}$ with
BERT$_{\text{LAST}}$ in Fig. 2 (p. 6) and Tab. 4 (p. 5–6): before contextualization function
words cluster around a few head states and states are either function-only or content-only;
afterwards function words spread over the whole distribution and most states mix both,
i.e. function words "receive" meaning from neighbouring content words. Concretely, the symbol
`$` has no meaningful neighbours in BERT$_{\text{ZERO}}$ but encodes *money* in
BERT$_{\text{LAST}}$; `#ing`/`#ed` go from ordinary subwords to tense markers; `be`'s
neighbours change from its morphological variants to linking verbs (§5.2, p. 6).

**Transition topology.** Decoding the dev set gives a state-transition graph (Fig. 3, p. 7)
whose nodes and edges both follow a long-tail distribution: few frequent head states/edges,
many weak tail edges. Frequent states are inter-connected; function states act as *hubs*.
Example transitions with their word bigrams (Tab. 5, p. 8): state pair 886–989 emits
"to-buy/to-sell/to-build" (do something), 1671–1441 emits "free-to/willing-to/happy-to"
(adjective + to), 426–198 emits "caused-by/produced-by/made-by" (passive voice), 355–1201
emits "three-years/five-years/24-hours" (time). Contextualization also spreads function words
off the head transitions, giving BERT$_{\text{LAST}}$ a longer orange tail than
BERT$_{\text{ZERO}}$ (Fig. 4, p. 8).

**Sentences as traversals.** The paper's headline claim (§6.2, pp. 8–9, Fig. 5) is a
four-step mechanism: (1) function states correspond to function words; (2) content states
correspond to content words of similar lexical/syntactic/semantic meaning; (3) transitions
between them encode phrase constructions; (4) a sentence is a traversal of states, where the
transition chain encodes the syntactic template and the emissions fill in the content.
Structurally similar sentences share overlapping paths (four worked examples in Fig. 5).

**Robustness.** Results are searched over three random seeds; the count of not-aligned
BERT$_{\text{LAST}}$ states varies only between 1059 and 1069 (§A.1, p. 13), and the word
clusters, contextualization effects and transitions recur across seeds. Entropy coefficient
$\lambda$ was tuned from $\{0.1,0.01,0.001,10^{-4},10^{-5}\}$ to $\lambda = 0.005$ (§A.1,
p. 12).

## Limitations stated by the authors

- The analysis is about *topological* structure (how nodes are connected), not the *distance*
  structure (how far nodes are), which the authors call an equally important geometric
  property (§7, p. 9).
- It addresses *local* topology (e.g. neighbouring words of a state) far more than *global*
  topology (e.g. how states and words form hierarchies) (§7, p. 9).
- Only BERT (bidirectional) is studied; there is evidence that autoregressive transformers
  (GPT-2, T5 decoder) have different topologies (§7, p. 9).
- The decoder is deliberately light — its role is to help induce states, not to be a strong
  generator (§3, p. 4).

## Critical assessment

This is a tidy, well-motivated use of a structured VAE as an unsupervised probe, and the
function/content split is a genuinely useful lens. Several caveats, in my reading:

- **The alignment evaluation is soft.** "Aligns" is a 90% threshold on *word occurrences*,
  and the model is selected as the checkpoint maximising the number of aligned tags over
  POS∪ENT∪DEP∪CCG∪BCN (§5.1, p. 5) — the authors admit this "introduces a slight bias". The
  main quantitative claims (Tab. 1–2) are therefore partly selected-on-the-test-metric; the
  RANDEMB row is a good sanity control but does not remove the selection effect.
- **Human evaluation is under-specified.** The annotation protocol (Tab. 2), inter-annotator
  agreement, and whether annotators saw the tag sets, are not reported in the main text; only
  three annotators, no agreement statistic.
- **The headline traversal claim is illustrative, not measured.** §6.2 rests on four
  hand-picked sentences in Fig. 5 and qualitative overlaps; there is no quantitative metric
  for "structurally similar sentences share paths", no held-out syntactic task, and no
  comparison against the baselines on this claim.
- **Reproducibility.** Code/data are not referenced in the paper; the PCFG-free pipeline
  depends on an approximate structured-inference implementation (Fu et al. 2022, outside the
  collection) and on multiple seeds, though the reported seed variance is small.
- **Equation 5's sign** could not be recovered from the file's extracted text (the entropy
  term is OCR-garbled); the standard CRF-VAE objective is transcribed here with a `+`, but
  the printed sign should be re-checked against the source if it matters.
- **Scope.** Everything is on 20News, a topic-classification corpus of forum posts; the
  lexical/semantic states (religion, sports, guns, mac/PC) may partly reflect that domain,
  and the paper reports no second dataset or model scale.

Notable for this textbook: although the paper is framed as NLP interpretability, it is a
*representation-geometry* paper — it studies the topology (connectivity) of an embedding
manifold and explicitly defers the distance/metric structure to future work (§7, p. 9), which
is exactly the object of interest in the other papers of this collection.

## Connections within the collection

No other paper in this collection is cited by, or built upon, this one, and the paper cites
no collection entry; `builds_on` is therefore empty. It is nonetheless the collection's only
paper on the geometry of *contextualized NLP* representations, and it is possibly related to
the manifold-learning and topology material: [@meilaManifoldLearningWhat2023] for the
manifold view of representation learning, [@chazalIntroductionTopologicalData2021] for the
topological-data-analysis toolbox that could quantify the "global topology" the authors leave
open (§7, p. 9), and [@nielsenElementaryIntroductionInformation2020] for the
distance/metric structure of a statistical manifold that §7 explicitly sets aside. These are
thematic adjacencies only; the paper itself makes none of them.
