---
layout: post
title: Engram - parameters you look up
date: 2026-10-28
description: How DeepSeek-V4.1-Flash hashes the last few tokens into rows of a 196B-parameter table, gates those rows into the residual stream, and serves and trains that table.
tags: deepseek engram architecture optimization
categories: research
related_posts: false
toc:
  sidebar: left
---

<style>mjx-container[jax="CHTML"][display="true"] { overflow-y: hidden; padding: 0.3em 0; } .footnotes { margin-top: 2.5rem; font-size: 0.9em; } .footnotes::before { content: ""; display: block; width: 20%; margin: 0 auto 1.5rem; border-top: 1px solid currentColor; opacity: 0.3; }</style>

Engram is a conditional memory module introduced by DeepSeek in [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372), and later adopted in DeepSeek-V4.1-Flash. It works like a token embedding table, but for n-grams: the last few tokens are hashed into row numbers, and each row holds a learned vector for the n-grams that land on it. It is parametric and trained end-to-end with the model.

The motivation for Engram is to decouple how much a model memorizes from how much it computes. Its tables are only looked up, never computed over: each token reads a fixed number of rows, so the tables can grow without adding per-token cost.

I first ran into this idea while browsing new PRs in [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)[^speedrun]. [PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201) used a technique I didn't know: hash the previous and current token into a table and add the learned row to the residual stream. It cut about 5 seconds, ~5% of the run, and sparked my interest in n-gram hashing. Then [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360) took it to the extreme: a bigram + trigram table with 84.6M rows, about 65B parameters, attached to a model of roughly 130M parameters. The table is about 500× the size of the model, and it is learned from scratch in a training run that finishes in under 40 seconds. Talk about decoupling memory from compute.

[^speedrun]: modded-nanogpt hosts the NanoGPT speedrun, where people compete to train a GPT-2-small-sized model to a fixed validation loss on 8×H100s as fast as possible.

Recently, Qwen released [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture that will underpin Qwen4, and it includes a similar module: a 51B-parameter n-gram embedding over bigrams and trigrams, next to a 125B-parameter MoE with 6B parameters active per token.

In this short writeup we focus on Engram as it is in DeepSeek-V4.1-Flash. We follow the V4.1 technical report and fill in the implementation details it leaves out from DeepSeek's [Hugging Face implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

#### The big picture

{% include figure.liquid path="assets/img/engram_v41_block.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1. Engram in DeepSeek-V4.1. (a) Two Engram blocks, each with its own table, write into the residual stream just before layers 1 and 14. (b) Inside one block: the n-grams ending at the current token are hashed into 24 table rows, one linear map turns them into a key per residual stream plus a shared value, and each stream adds the value scaled by its gate. Drawn for this post from the V4.1 report and the released code." %}

For every token, an Engram block turns the last few tokens into 24 row numbers: eight for each of the 2-, 3- and 4-grams ending at that token. It reads those rows from its table and compares what it read with the model's hidden state. The better they match, the more of the memory it adds.

V4.1 has two such blocks among its 40 layers, just before layers 1 and 14. Each has its own table of about 384M rows of 256 numbers, and together they hold 196B parameters, about a quarter of the model's 748B. The Engram paper found that the module works best early, once a single layer of attention has given the hidden state enough context to judge the memory, and that two smaller blocks beat one large one.

#### From text to table rows

**Why hashing.** A table with one row per token is easy: the token ID is the row number. Engram wants a row per n-gram, a run of consecutive tokens such as "the Great". With about 100K distinct tokens there are about $10^{10}$ possible 2-grams and $10^{15}$ possible 3-grams, far more than any table can hold. So Engram hashes them. A hash function maps a key of any size to a number from $0$ to $M-1$, and that number is the row. A toy hash, $k \bmod 10$:

```text
key 12  ->  row 2
key 27  ->  row 7
key 32  ->  row 2      collision: two keys, one row
```

Three properties carry over to Engram. The same key always gives the same row, so rows can be computed from the text alone, before the model runs. Different keys can land on the same row, so Engram has to live with collisions. And a good hash spreads keys evenly; $k \bmod 10$ sends every multiple of 10 to row 0.

Engram gets from a token to its rows in four steps.

**Step 1: merge spelling variants.** The tokenizer gives " The", "the" and "THE" different IDs, because it must reproduce text exactly. For memory they are the same word. So Engram first applies **tokenizer compression**: it maps each token to a normalized form, lowercased with accents and surrounding spaces removed, and gives tokens with the same form one shared ID. In V4.1 this turns 129,280 token IDs into 99,092.

```text
token     ID       merged ID
" The"    455      237
"the"     1805     237
" the"    270      237
"THE"     20852    237
```

**Step 2: take the n-grams that end at the token.** At each position Engram looks at the current token and the three before it. For the token "Great" in "Only Alexander the Great could tame":

```text
2-gram                         the   Great
3-gram              Alexander  the   Great
4-gram      Only    Alexander  the   Great
```

Note that V4.1 uses 2-, 3- and 4-grams, while the main models in the original Engram paper use only 2- and 3-grams.

**Step 3: hash each n-gram.** Each position in the n-gram has a fixed, large, odd multiplier: $m_0$ for the current token, $m_1$ for the token before it, and so on. Engram multiplies each token's merged ID by the multiplier of its position and XORs the products:

$$
\begin{aligned}
\phi_2(t) &= c_t\,m_0 \oplus c_{t-1}\,m_1 \\
\phi_3(t) &= c_t\,m_0 \oplus c_{t-1}\,m_1 \oplus c_{t-2}\,m_2 \\
\phi_4(t) &= c_t\,m_0 \oplus c_{t-1}\,m_1 \oplus c_{t-2}\,m_2 \oplus c_{t-3}\,m_3
\end{aligned}
$$

Here $\phi_n(t)$ is the hash of the n-gram of length $n$ that ends at position $t$, an integer below $2^{63}$ that Step 4 turns into rows; $c_t$ is the merged ID of the token at position $t$, and $\oplus$ is XOR.

Nevertheless, XOR has one not-so-subtle drawback: it ignores order, since $a \oplus b = b \oplus a$. The position multipliers are what fix it. Here is a toy example with merged IDs 3 and 5, $m_0 = 7$, $m_1 = 11$, and a table of 13 rows, where the row is the hash mod 13:

```text
"3 5"  current   5 x 7  = 35 = 100011
       previous  3 x 11 = 33 = 100001
       XOR              =  2 = 000010    2 mod 13 = row 2

"5 3"  current   3 x 7  = 21 = 010101
       previous  5 x 11 = 55 = 110111
       XOR              = 34 = 100010   34 mod 13 = row 8
```

Swapping the two tokens moves the n-gram from row 2 to row 8. Without the multipliers, both orders would give 3 XOR 5 = 6. To better understand this step, refer to the example in Appendix A.

**Step 4: read one row from each of eight tables.** The problem is collisions. There are far more n-grams than rows, so different n-grams must share rows, and n-grams that share a row read and train the same vector. Engram limits the damage with **multi-head hashing**: each n-gram length gets eight tables, called heads, whose sizes $P_{n,k}$ are different primes of about 16M. Head $k$ reads row

$$
z_{t,n,k} = \phi_n(t) \bmod P_{n,k}, \qquad k = 1, \dots, 8
$$

Two n-grams with different hashes share a row in head $k$ only if $P_{n,k}$ divides the difference of their hashes. Sharing rows in three heads would need the product of three primes, at least $4 \times 10^{21}$, to divide a difference below $2^{63} \approx 9.2 \times 10^{18}$, which is impossible. So two different hashes share a row in at most two of the eight heads, and no n-gram with a different hash reads the same eight rows, even though each single row is shared. Three n-gram lengths times eight heads makes 24 tables per Engram block, about 384M rows in total, so each token reads 24 rows per block.

**Same n-gram, same rows.** The rows depend only on the n-gram, so an n-gram reads the same rows wherever it appears. Here is "Great" in two sentences:

```text
                            2-gram     3-gram
                            head 1     head 1
Only Alexander the Great    10824761   1864923
He admired the Great        10824761   4475658
```

Both sentences contain the 2-gram "the Great", so they read the same eight 2-gram rows. Their 3-grams differ, so the 3-gram rows tell "Alexander the Great" apart from "admired the Great".

#### Integrating Engram into the model

For each token, an Engram block has now read 24 rows: one from each of the eight heads, for each of the 2-, 3- and 4-gram hashes. V4.1 has two such blocks, before layers 1 and 14 (Figure 1a), each with its own multipliers and tables. Each row holds 256 numbers, and a block concatenates its 24 rows into one memory vector of 6,144 numbers. Writing $\mathbf{E}_{n,k}$ for the table of head $k$ at n-gram length $n$, the memory vector is:

$$
\mathbf{e}_t \;=\; \big\Vert_{n=2}^{4}\, \big\Vert_{k=1}^{8}\; \mathbf{E}_{n,k}\big[z_{t,n,k}\big] \;\in\; \mathbb{R}^{6144}
$$

This vector knows nothing about the sentence around it. Part of it may be collision noise, and the same n-gram can mean different things in different contexts. So Engram does not add it blindly: **context-aware gating** decides how much of it to add.

First, one FP8 linear map turns the memory vector into keys and a value. V4.1 widens the residual stream into four parallel streams, using [mHC]({% post_url 2026-08-27-manifold-constrained-hyper-connections-deepseek-v4 %}). With **multi-branch integration**, the map makes one key per stream and one value shared by all four, each of size $d = 5120$. Separate keys let each stream gate the memory on its own, while sharing the value saves parameters and compute.

$$
\big[\mathbf{k}^{(1)}_t,\, \mathbf{k}^{(2)}_t,\, \mathbf{k}^{(3)}_t,\, \mathbf{k}^{(4)}_t,\, \mathbf{v}_t\big] \;=\; \mathbf{W}_{KV}\,\mathbf{e}_t, \qquad \mathbf{W}_{KV} \in \mathbb{R}^{(5 \cdot 5120) \times 6144}
$$

Each stream $j$ then scores its hidden state $\mathbf{h}^{(j)}_t$, the residual stream as it enters the Engram block (Figure 1b), against its own key. This is attention's scaled dot product, with the hidden state as the query:

$$
s^{(j)}_t \;=\; \frac{\mathrm{RMSNorm}\big(\mathbf{h}^{(j)}_t\big)^{\top}\,\mathrm{RMSNorm}\big(\mathbf{k}^{(j)}_t\big)}{\sqrt{d}}
$$

A sigmoid turns the score into a gate between 0 and 1, and the stream adds the value scaled by its gate:

$$
\alpha^{(j)}_t \;=\; \sigma\Big(\operatorname{sign}\big(s^{(j)}_t\big)\sqrt{\big\lvert s^{(j)}_t\big\rvert}\Big), \qquad
\mathbf{h}^{(j)}_t \;\leftarrow\; \mathbf{h}^{(j)}_t + \alpha^{(j)}_t\,\mathbf{v}_t
$$

When the context agrees with the memory, the score is positive and the gate opens. When it disagrees, the gate closes and the memory is mostly ignored. Normalizing both vectors keeps gradients stable. The signed square root, which compresses large scores before the sigmoid, comes from V4.1's released code.

#### Serving the table

{% include figure.liquid path="assets/img/engram_fig2_system.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2. Where the table lives. (a) In training, the tables are split across GPUs and rows are exchanged between them. (b) At inference, the tables sit in host memory and rows are prefetched while earlier layers compute. V4.1 follows the same split. From the Engram paper, Figure 2." %}

The row numbers depend only on the token IDs, never on anything the model computes. So they are known before the first layer runs, and the tables do not need GPU memory at all. At inference V4.1 keeps them in host memory and copies each token's 48 rows, 24 per block, to the GPU in the background. For the first block, the copy overlaps the computation of layer 0. Per token that is 48 rows of 256 one-byte FP8 numbers plus their scales, a little over 12 KB, however large the table is.

The exception is RL rollouts. There the tables stay in GPU memory, which reduces host memory pressure and avoids out-of-memory failures caused by host memory fragmentation.

#### Training the table

In training, each table is split by rows across a dedicated group of GPUs (Figure 2a). The group size trades memory per GPU against how far each lookup has to travel. As at inference, each pipeline stage starts fetching the rows for its whole batch before it begins its microbatches, and the gradients for those rows go back to their owners after the backward pass. Rows are stored and moved in FP8 and feed straight into the next matrix multiplication.

Adam would substantially increase optimizer memory on 196B parameters, so V4.1 trains the tables with a momentum update followed by Sinkhorn balancing, which keeps only the momentum buffer. Sinkhorn balancing rescales the update's rows and columns so that each has about unit RMS. Each row is one n-gram slot, so every n-gram in the batch gets a step of similar size, whether it is common or rare. The report finds this outperforms Adam. Appendix B shows the update on a small example.

The report places the two blocks at layers 1 and 14 to balance memory across the training pipeline stages. So the placement is a systems choice as well as a modeling one: the Engram paper argues for early blocks on quality grounds, while the reason the V4.1 report gives is memory.

#### What the table buys

The V4.1 report does not measure Engram on its own, so the evidence comes from the Engram paper. It compared a 27B-parameter Engram model with an MoE model of the same size, trained on the same data with the same compute.

- **Better at more than facts.** The Engram model scored higher on knowledge benchmarks such as MMLU, as expected, but also on reasoning, code and math: BBH went from 50.9 to 55.9 and HumanEval from 37.8 to 40.8.
- **The table holds the facts.** Switching the table off at inference cut factual-knowledge benchmarks to 29 to 44% of their score, while reading comprehension kept 81 to 93%. The paper reads this as the table being the main store of factual knowledge, while context-grounded tasks rely mainly on attention.

#### Summary

Engram is more than a new block. It adds a second axis of sparsity, and a different way to think about LLM architecture: MoE makes computation sparse by choosing which experts run, while Engram makes memory sparse by choosing which table rows each token reads. Static knowledge is looked up instead of recomputed.

The Engram paper's mechanistic analysis suggests why it helps. Engram effectively deepens the network: early layers no longer rebuild static patterns, so attention can focus on global context and reasoning.

The strongest evidence that Engram is an important idea is adoption: DeepSeek and Qwen both ship Engram-style memory in their released models. In my opinion, Engram-like modules will be a standard part of future architectures.

<hr style="width: 20%; margin: 2.5rem auto; border: 0; border-top: 1px solid currentColor; opacity: 0.3;">

#### Appendix A: the hashing example in full

These numbers come from running the released code with the released tokenizer, for the Engram block before layer 1. The token is "Great" in "Only Alexander the Great could tame".

```text
token        Only    Alexander    the    Great
merged ID    1134    15695        237    2049
```

The block's four multipliers and the three hashes they produce:

```text
m0 = 76632096046245
m1 =  4839876093313
m2 = 35959672319349
m3 = 73987337458391

2-gram hash = 2049*m0 XOR 237*m1       = 155879025071873224
3-gram hash = 2-gram hash XOR 15695*m2 = 431448553835705555
4-gram hash = 3-gram hash XOR 1134*m3  = 348681755203977393
```

Each head takes its n-gram's hash modulo its own prime. The eight 2-gram heads use the primes 16000057 … 16000183, the 3-gram heads 16000189 … 16000321, and the 4-gram heads 16000339 … 16000463. That gives these 24 rows $z_{t,n,k}$, for the 2-gram "the Great", the 3-gram "Alexander the Great" and the 4-gram "Only Alexander the Great":

```text
head      2-gram      3-gram      4-gram
1       10824761     1864923     1812512
2       14987147    10254891     1391737
3        2303798     6934234     6725617
4       13012933    15323871     9539653
5        5661034    15930218    11470157
6        3366273     8961293    14447633
7       10248185     7418094     3581351
8        1915287     8750646    13656871
```

In the released code, a block's 24 tables are stored end to end in one tensor of 384,006,168 rows, the sum of the 24 primes. A row's position in that tensor is its row number plus the sizes of all the tables before it. The first 3-gram table starts after the eight 2-gram tables, at 128,000,880, so its row 1,864,923 sits at 129,865,803.

The block before layer 14 repeats all of this with different multipliers, the next 24 primes, 16000477 … 16000889, and its own table.

#### Appendix B: the Sinkhorn update, step by step

{% include figure.liquid path="assets/img/dsv41_alg1_sinkhorn.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3. Momentum update with Sinkhorn balancing. For an Engram table, each row of the update matrix is a table row and each column is a feature. From the V4.1 tech report, Algorithm 1." %}

The algorithm treats one table's update as a matrix with $m$ rows, one per table row, and $n$ columns, one per feature. Here $n$ is the number of features, not the n-gram length.

1. **Nesterov momentum** (lines 1–2) gives the update direction $\widehat{G}_t$.
2. **Mask near-zero rows** (lines 3–5). Rows whose norm is below $\tau\bar\rho$, where $\bar\rho$ is the mean row norm and $\tau = 10^{-3}$, are zeroed. The report's reason is numerical stability. A plausible reading: rows the batch barely touched would otherwise be blown up to full size by the next step.
3. **Balance rows and columns** (lines 6–16). Rows and columns are normalized in turn for a fixed $K = 11$ passes, starting and ending with rows. Afterwards every remaining row has about the same norm, and so does every column.
4. **Rescale** (lines 17–19). Multiplying by $\sqrt{n}$ turns a unit row norm into a unit row RMS, and the learning rate is scaled by $\gamma = 0.18$ to match Adam's typical step. Engram's learning rate is also multiplied by 5, as in the paper, and there is no weight decay.

A small example with three rows and two columns. Row 1 belongs to a frequent n-gram, row 3 to a rare one, and row 2 was not touched:

$$
\begin{aligned}
\widehat{G} = \begin{pmatrix} 3 & 4 \\ 0 & 0 \\ 0.3 & 0.4 \end{pmatrix}
&\;\xrightarrow{\text{mask, row-norm}}\;
\begin{pmatrix} 0.6 & 0.8 \\ 0 & 0 \\ 0.6 & 0.8 \end{pmatrix} \\[6pt]
&\;\xrightarrow{\text{col-norm}}\;
\begin{pmatrix} 0.71 & 0.71 \\ 0 & 0 \\ 0.71 & 0.71 \end{pmatrix}
\;\xrightarrow{\times\sqrt{2}}\;
\begin{pmatrix} 1 & 1 \\ 0 & 0 \\ 1 & 1 \end{pmatrix}
\end{aligned}
$$

Here passes 3 to 11 change nothing, so only passes 1 and 2 are shown before the rescale. The rare n-gram gets the same size of step as the frequent one, the untouched row gets none, and neither column dominates.

#### Sources

- Cheng et al., [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372): Sections 2, 4 and 6.1 to 6.3.
- DeepSeek-AI, DeepSeek-V4.1-Flash Technical Report: Sections 2.1, 2.4.2, 2.5, 3.1.3 and 4.2.
- [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on Hugging Face: the inference code and tokenizer. The row numbers, primes and table sizes in this post were computed by running that code with the released tokenizer.
