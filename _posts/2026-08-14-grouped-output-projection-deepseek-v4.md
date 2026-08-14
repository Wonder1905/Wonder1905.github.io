---
layout: post
title: The Grouped Output Projection in DeepSeek-V4
date: 2026-08-14
description: One paragraph in the V4 report halves the attention block's output projection. Where the saving comes from, what it costs, and what the report does not say.
tags: deepseek attention architecture efficiency
categories: research
related_posts: false
toc:
  sidebar: left
---

_All figures below are per attention layer, for the V4-Flash configuration._

While reading the DeepSeek-V4 code on Hugging Face I found an output projection I did not recognise. In the report it gets a name and a single paragraph: _Grouped Output Projection_.

The motivation given is entirely cost. V4's attention emits a very wide vector, and the researchers note that mapping it back to the residual stream with one dense matrix "will impose a substantial computational burden." That sentence is the whole justification — the report says nothing further about the design, and reports no experiment on it.

What they build instead borrows the idea behind grouped convolution and applies it to the attention output's channel dimension. It is not a convolution, and the correspondence is structural rather than literal: there is no kernel, no spatial extent, and nothing here slides over anything. But the saving comes from the same place — each output is wired to a slice of the inputs instead of all of them.

> ##### GROUPED CONVOLUTION, BRIEFLY
>
> A grouped convolution splits input and output channels into $$g$$ matching groups, and computes each output channel from only the $$C_{\text{in}}/g$$ inputs in its own group rather than all $$C_{\text{in}}$$ — cutting parameters and FLOPs by a factor of $$g$$. AlexNet is usually named as its origin, not quite fairly: splitting the network across two GPUs gave the same restricted connectivity, but as a workaround for a card that could not hold the model, not as a proposed layer. It became a deliberate design choice later, once [ResNeXt](https://arxiv.org/abs/1611.05431) found that at matched budget, spending on more groups beat spending on more width or depth.
{: .block-tip }

## 1. What the output projection costs

Notation first, since everything below is arithmetic on five numbers. Write $$d$$ for the hidden size and $$h$$ for the number of attention heads, each emitting a vector of width $$c$$. The report writes $$n_h$$ where this uses $$h$$. In the V4-Flash configuration $$d = 4096$$, $$h = 64$$ and $$c = 512$$, so the attention output is $$hc = 32768$$ wide — eight times the residual stream it has to be written back into.

The attention operation produces $$h$$ head outputs of width $$c$$, flattened into one tensor $$O$$ and mapped back to the residual stream by a single dense matrix:

$$
Y = O W_O^{\top}, \qquad O \in \mathbb{R}^{b \times s \times hc}, \quad W_O \in \mathbb{R}^{d \times hc}, \quad Y \in \mathbb{R}^{b \times s \times d}
$$

| tensor | operation |
| :--- | :--- |
| `[b, s, h, c]` | attention output, per head |
| `[b, s, h·c]` | flatten heads — $$O$$ |
| $$W_O$$ : `[d, h·c]` | **134.2 M parameters** |
| `[b, s, d]` | residual stream — $$Y$$ |

The map is position-wise — the same matrix applied at every $$(b, s)$$, no sequence-length term — so

$$
P_{\text{std}} = d\,hc, \qquad \text{FLOPs/token} = 2\,d\,hc
$$

With V4-Flash's dimensions: $$4096 \times 32768 = 134.2$$ M parameters and 268.4 MFLOP per token.

The cost above is linear in $$hc$$, and V4 raised $$hc$$ sharply. DeepSeek-V2 and V3 use 128 heads with a value width of 128, giving $$hc = 16384$$; both project it with a plain dense `nn.Linear`. V4 quadruples the per-head width to $$c = 512$$. Flash spends that on half the heads — $$h = 64$$, so $$hc$$ doubles to 32768. Pro keeps all 128, and $$hc$$ reaches 65536.

| Model | $$h$$ | $$c$$ | $$hc$$ | $$d$$ | Dense $$W_O$$ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| V2 / V3 | 128 | 128 | 16384 | 7168 | 117.4 M |
| V4-Flash | 64 | 512 | **32768** | 4096 | 134.2 M *if dense* |
| V4-Pro | 128 | 512 | **65536** | 7168 | **469.8 M** *if dense* |

At Flash's width a dense $$W_O$$ would cost 134.2 M — only 14% above V3's, which on its own is not obviously a crisis. The pressure is at Pro: **469.8 M parameters per layer, four times V3's entire output projection**. That is the "substantial computational burden" quoted at the top — the report's characterisation of its own configuration, not a measurement of mine. Flash is the reference configuration throughout because its numbers stay round; Pro appears wherever the scaling is the point.

Either way, the design is not a retrofit of V3's layer. It is the enabling change that made wide heads affordable.

## 2. The obvious fix, and what it costs

The projection is expensive because it is one dense matrix stretched between two large numbers. The standard way to make a big matrix cheaper is not to have one: factor it, and pass through a narrow intermediate. Project $$hc \rightarrow r$$ first, then $$r \rightarrow d$$, with $$r$$ small.

$$
P_{\text{bot}} = r\,hc + d\,r = r(hc + d), \qquad \text{FLOPs/token} = 2\,r(hc + d)
$$

The cost is now linear in $$r$$ instead of in the product $$d\,hc$$, and at Flash's dimensions $$hc + d = 36864$$, so the entire design collapses to choosing one number.

| intermediate $$r$$ | parameters | vs dense | rank of the map |
| :--- | ---: | ---: | ---: |
| none — dense $$W_O$$ | 134.2 M | 1.00× | ≤ 4096 |
| 8192 &nbsp;*a 4× squeeze on* $$hc$$ | **302.0 M** | **2.25×** | ≤ 4096 |
| 3641 &nbsp;*break-even* | 134.2 M | 1.00× | ≤ 3641 |
| 1024 | **37.7 M** | **0.28×** | ≤ 1024 |

Two rows there are worth pausing on. The first is that factoring does not automatically save anything: at $$r = 8192$$ it costs more than twice the matrix it replaced, because the first stage is now wider than the output it feeds. Saving requires

$$
r < r^{*} = \frac{d\,hc}{d + hc} = 3641, \qquad r^{*} < d \;\; \text{always}
$$

and that ceiling sits below $$d$$ for every $$d$$ and $$hc$$, since $$hc/(d + hc)$$ is never 1.

The second is what that implies. Both stages are linear with nothing between them, so the pair is a single matrix of rank at most $$r$$ — the number of independent directions it can produce is capped by the width it was squeezed through. A bottleneck that saves parameters therefore has $$r < d$$, and so cannot reach every direction of the residual stream, for any input, at any setting of its weights. The saving and the rank loss are not two effects to be traded off; they are the same fact, and no choice of $$r$$ separates them.

How much that costs in practice is not settled, but it is not nothing: [Bhojanapalli et al.](http://proceedings.mlr.press/v119/bhojanapalli20a/bhojanapalli20a.pdf) showed that rank caps of exactly this kind restrict which attention maps are expressible at all. What V4 does is keep the factorization and drop the assumption that the intermediate has to be one narrow vector.

## 3. The idea, in one example

Before any of the real dimensions, here is the whole grouping idea at a size you can count on your fingers. Take a 12-element input and project it to a 6-element output, with $$g = 3$$ groups.

<figure style="margin: 2rem 0;">
<div style="background: var(--global-card-bg-color); border: 1px solid var(--global-divider-color); border-radius: 4px; padding: 1.4rem;">
<svg viewBox="0 0 600 372" role="img" aria-label="One 12-element vector, split into 3 groups of 4; each group multiplied by its own 2 by 4 matrix, producing 3 groups of 2 output elements." style="width:100%;height:auto;display:block;">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="currentColor"/>
    </marker>
  </defs>
  <g font-family="ui-monospace, Menlo, monospace" font-size="11">

    <text x="8" y="14" fill="var(--global-text-color-light)" font-size="9.5" letter-spacing="1.3">ONE VECTOR — 12 ELEMENTS</text>

    <rect x="10" y="24" width="520" height="30" fill="var(--global-text-color-light)" fill-opacity=".08" stroke="var(--global-text-color-light)" stroke-width="1.4"/>
    <g stroke="var(--global-divider-color)" stroke-width=".8" opacity=".6">
      <line x1="53.33" y1="24" x2="53.33" y2="54"/>
      <line x1="96.67" y1="24" x2="96.67" y2="54"/>
      <line x1="140.00" y1="24" x2="140.00" y2="54"/>
      <line x1="183.33" y1="24" x2="183.33" y2="54"/>
      <line x1="226.67" y1="24" x2="226.67" y2="54"/>
      <line x1="270.00" y1="24" x2="270.00" y2="54"/>
      <line x1="313.33" y1="24" x2="313.33" y2="54"/>
      <line x1="356.67" y1="24" x2="356.67" y2="54"/>
      <line x1="400.00" y1="24" x2="400.00" y2="54"/>
      <line x1="443.33" y1="24" x2="443.33" y2="54"/>
      <line x1="486.67" y1="24" x2="486.67" y2="54"/>
    </g>
    <g fill="var(--global-text-color)" text-anchor="middle" font-size="10.5">
      <text x="31.67" y="44">i₀</text>
      <text x="75.00" y="44">i₁</text>
      <text x="118.33" y="44">i₂</text>
      <text x="161.67" y="44">i₃</text>
      <text x="205.00" y="44">i₄</text>
      <text x="248.33" y="44">i₅</text>
      <text x="291.67" y="44">i₆</text>
      <text x="335.00" y="44">i₇</text>
      <text x="378.33" y="44">i₈</text>
      <text x="421.67" y="44">i₉</text>
      <text x="465.00" y="44">i₁₀</text>
      <text x="508.33" y="44">i₁₁</text>
    </g>

    <text x="180" y="82" text-anchor="middle" fill="#B8912F" font-size="10">split into g = 3 groups</text>
    <g stroke="var(--global-text-color-light)" stroke-width="1" marker-end="url(#ar2)" color="var(--global-text-color-light)">
      <line x1="90" y1="60" x2="90" y2="96"/><line x1="270" y1="60" x2="270" y2="96"/><line x1="450" y1="60" x2="450" y2="96"/>
    </g>

    <text x="8" y="118" fill="var(--global-text-color-light)" font-size="9.5" letter-spacing="1.3">3 GROUPS OF 4</text>

    <g stroke-width="1.4">
      <rect x="10"  y="128" width="160" height="30" fill="#7B5EA7" fill-opacity=".16" stroke="#7B5EA7"/>
      <rect x="190" y="128" width="160" height="30" fill="#3E7CB1" fill-opacity=".16" stroke="#3E7CB1"/>
      <rect x="370" y="128" width="160" height="30" fill="#1B8A80" fill-opacity=".16" stroke="#1B8A80"/>
    </g>
    <g stroke="var(--global-divider-color)" stroke-width=".8" opacity=".55">
      <line x1="50"  y1="128" x2="50"  y2="158"/><line x1="90"  y1="128" x2="90"  y2="158"/><line x1="130" y1="128" x2="130" y2="158"/>
      <line x1="230" y1="128" x2="230" y2="158"/><line x1="270" y1="128" x2="270" y2="158"/><line x1="310" y1="128" x2="310" y2="158"/>
      <line x1="410" y1="128" x2="410" y2="158"/><line x1="450" y1="128" x2="450" y2="158"/><line x1="490" y1="128" x2="490" y2="158"/>
    </g>
    <g fill="var(--global-text-color)" text-anchor="middle" font-size="10.5">
      <text x="30"  y="148">i₀</text><text x="70"  y="148">i₁</text><text x="110" y="148">i₂</text><text x="150" y="148">i₃</text>
      <text x="210" y="148">i₄</text><text x="250" y="148">i₅</text><text x="290" y="148">i₆</text><text x="330" y="148">i₇</text>
      <text x="390" y="148">i₈</text><text x="430" y="148">i₉</text><text x="470" y="148">i₁₀</text><text x="510" y="148">i₁₁</text>
    </g>
    <g fill="var(--global-text-color-light)" text-anchor="middle" font-size="9.5">
      <text x="90"  y="174">group 0</text><text x="270" y="174">group 1</text><text x="450" y="174">group 2</text>
    </g>

    <g stroke="var(--global-text-color-light)" stroke-width="1" marker-end="url(#ar2)" color="var(--global-text-color-light)">
      <line x1="90" y1="182" x2="90" y2="208"/><line x1="270" y1="182" x2="270" y2="208"/><line x1="450" y1="182" x2="450" y2="208"/>
    </g>

    <g stroke-width="1.4">
      <rect x="40"  y="212" width="100" height="48" rx="2" fill="#7B5EA7" fill-opacity=".12" stroke="#7B5EA7"/>
      <rect x="220" y="212" width="100" height="48" rx="2" fill="#3E7CB1" fill-opacity=".12" stroke="#3E7CB1"/>
      <rect x="400" y="212" width="100" height="48" rx="2" fill="#1B8A80" fill-opacity=".12" stroke="#1B8A80"/>
    </g>
    <g fill="var(--global-text-color)" text-anchor="middle" font-style="italic" font-size="12">
      <text x="90" y="234">W₀</text><text x="270" y="234">W₁</text><text x="450" y="234">W₂</text>
    </g>
    <g fill="var(--global-text-color-light)" text-anchor="middle" font-size="9.5">
      <text x="90" y="251">2 × 4</text><text x="270" y="251">2 × 4</text><text x="450" y="251">2 × 4</text>
    </g>

    <g stroke="var(--global-text-color-light)" stroke-width="1" marker-end="url(#ar2)" color="var(--global-text-color-light)">
      <line x1="90" y1="264" x2="90" y2="290"/><line x1="270" y1="264" x2="270" y2="290"/><line x1="450" y1="264" x2="450" y2="290"/>
    </g>

    <g stroke-width="1.4">
      <rect x="50"  y="294" width="80" height="30" fill="#7B5EA7" fill-opacity=".26" stroke="#7B5EA7"/>
      <rect x="230" y="294" width="80" height="30" fill="#3E7CB1" fill-opacity=".26" stroke="#3E7CB1"/>
      <rect x="410" y="294" width="80" height="30" fill="#1B8A80" fill-opacity=".26" stroke="#1B8A80"/>
    </g>
    <g stroke="var(--global-divider-color)" stroke-width=".8" opacity=".55">
      <line x1="90" y1="294" x2="90" y2="324"/><line x1="270" y1="294" x2="270" y2="324"/><line x1="450" y1="294" x2="450" y2="324"/>
    </g>
    <g fill="var(--global-text-color)" text-anchor="middle" font-size="10.5">
      <text x="70"  y="314">o₀</text><text x="110" y="314">o₁</text>
      <text x="250" y="314">o₂</text><text x="290" y="314">o₃</text>
      <text x="430" y="314">o₄</text><text x="470" y="314">o₅</text>
    </g>
    <text x="8" y="345" fill="var(--global-text-color-light)" font-size="9.5" letter-spacing="1.3">OUTPUT — 6 ELEMENTS</text>

    <text x="8" y="366" fill="#B8912F" font-size="10.5">o₀ never sees i₄.</text>
  </g>
</svg>
</div>
<figcaption style="font-size:.85rem;color:var(--global-text-color-light);margin-top:.7rem;padding-top:.7rem;border-top:1px solid var(--global-divider-color);">
<b>Grouped projection, 12 in and 6 out, g = 3.</b> Each group owns a private 2×4 matrix and reads only its own 4 inputs. A dense projection would instead wire every one of the 6 outputs to all 12 inputs.
</figcaption>
</figure>

Each output is built from its own group only: o₀ and o₁ from i₀..i₃, o₂ and o₃ from i₄..i₇, o₄ and o₅ from i₈..i₁₁. Each of the six outputs is wired to 4 inputs rather than all 12, so weights and FLOPs both drop by exactly 3 — the number of groups. That restricted fan-in is the whole source of the saving. Nothing is approximated, and the sparsity is structural rather than learned: this is three small dense matrices instead of one big one.

The reduction itself lives _inside_ each group: 4 inputs to 2 outputs, exactly as V4's groups go $$4096 \rightarrow 1024$$. How hard to squeeze there is a second and independent choice. §6 prices them separately.

## 4. The grouped output projection

> ##### THE CONSTRUCTION, VERBATIM
>
> To be specific, we first split $$n_h$$ outputs into $$g$$ groups, and then for each group of output $$o^{G}_{t,i} \in \mathbb{R}^{c n_h / g}$$, we project it to a $$d_g$$-dimensional intermediate output $$o^{G'}_{t,i} \in \mathbb{R}^{d_g}$$, where $$d_g < c n_h / g$$. Finally, we project the intermediate output $$[o^{G'}_{t,1}; o^{G'}_{t,2}; \ldots; o^{G'}_{t,g}] \in \mathbb{R}^{d_g g}$$ to the final attention output $$\hat{o}_t \in \mathbb{R}^{d}$$.

Three more symbols, and they are the whole design. $$g$$ is the number of groups — 8 in Flash; $$c_g = hc/g = 4096$$ is the width of one group; and $$d_g = 1024$$ is what each group is squeezed to. The squeeze inside a group is 4×, the same factor that cost more than the dense matrix in §2. What differs is the total: $$d_g g = 8192$$, which is _twice_ $$d$$ rather than a quarter of it.

Formally, two stages. Write $$O^{G}$$ for $$O$$ reshaped to `[b, s, g, c_g]`. **Down-projection**, one independent matrix per group:

$$
Z[b,s,j,:] = A_j\,O^{G}[b,s,j,:], \qquad A_j \in \mathbb{R}^{d_g \times c_g}, \quad j = 1 \ldots g
$$

**Up-projection and mixing**, one shared dense matrix:

$$
Y = Z' B^{\top}, \qquad Z' \in \mathbb{R}^{b \times s \times d_g g}, \quad B \in \mathbb{R}^{d \times d_g g}
$$

Write $$A$$ for the whole down-projection stage — all $$g$$ blocks together — and $$B$$ for the mixing matrix. **Both stages are linear, and there is no nonlinearity between them**, so they do not stack into two layers in any representational sense; they collapse into a single linear map:

$$
Y = O M^{\top}, \qquad M := BA, \quad M \in \mathbb{R}^{d \times hc}
$$

That collapse is the fact to hold on to. It means the pair buys _compression, not depth_ — unlike the grouped-convolution blocks it resembles, which put a normalization and a nonlinearity between the stages and are genuinely deeper for it. It also makes the cost exactly computable rather than approximate, because there is only one matrix to reason about.

What constrains $$M$$ is that it must factor through $$A$$: output group $$j$$ is wired only to input group $$j$$. Stage 1 is, in convolution terms, exactly a grouped 1×1 convolution over the sequence; stage 2 is the pointwise layer that puts the groups back in contact. Two constraints are now in play and it pays to keep them apart: the block-diagonality of $$A$$, and the squeeze $$d_g < c_g$$ inside each block. §6 prices them.

## 5. The code

```python
o = o.view(bsz, seqlen, self.n_local_groups, -1)
wo_a = self.wo_a.weight.view(
    self.n_local_groups, self.o_lora_rank, -1
)

o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
x = self.wo_b(o.flatten(2))
```

The einsum is the whole operation. Relabel the contracted index `k` — the source writes it `d`, which is already the hidden size — and `"bsgk,grk->bsgr"` reads

$$
\text{out}[b,s,g,r] = \sum_{k} o[b,s,g,k] \cdot w[g,r,k]
$$

with `k` ranging over $$c_g$$ and `r` over $$d_g$$. Only the last axis moves: `[b, s, g, c_g]` becomes `[b, s, g, d_g]`.

The load-bearing character is the `g` after the arrow. Nothing sums over it, so group $$j$$'s channels only ever meet $$A_j$$ — that single letter is the block-diagonality. Had the string ended `->bsr`, `g` would fall into the summing loop and every group would accumulate into one shared vector: a dense operation, and a different one.

## 6. The arithmetic

Here $$A$$ is $$g$$ blocks of $$d_g \times c_g$$; $$B$$ is one dense $$d \times d_g g$$.

$$
P_{\text{grp}} = \underbrace{g \cdot d_g \cdot c_g}_{A} + \underbrace{d \cdot d_g g}_{B}
$$

Since $$c_g = hc/g$$, the first term is $$g\,d_g c_g = d_g hc$$: **the $$g$$ cancels**. The block-diagonal down-projection costs what a single _global_ $$d_g$$-wide projection costs, and yields $$g$$ times the intermediate width. Dividing by the dense $$d\,hc$$:

$$
\frac{P_{\text{grp}}}{P_{\text{std}}} = \frac{d_g hc + d\,d_g g}{d\,hc} = \frac{d_g}{d} + \frac{d_g g}{hc} = \frac{d_g}{d} + \frac{d_g}{c_g}
$$

Down-projection plus mixing. The first term is $$A$$'s share: $$d_g hc$$ carries no $$d$$ at all, so against a baseline that grows with $$d$$ it decays like $$1/d$$. The second is $$B$$'s, pinned at $$d_g/c_g = 1/4$$ because $$d\,d_g g$$ scales with $$d$$ exactly as the baseline does. So the ratio falls toward ¼ as models widen.

| | V4-Flash | V4-Pro |
| :--- | ---: | ---: |
| dense $$W_O$$ &nbsp;*$$d\,hc$$* | 134.2 M &nbsp;*1.000* | 469.8 M &nbsp;*1.000* |
| $$A$$ &nbsp;`wo_a` &nbsp;*$$d_g/d$$* | 33.6 M &nbsp;*0.250* | 67.1 M &nbsp;*0.143* |
| $$B$$ &nbsp;`wo_b` &nbsp;*$$d_g/c_g$$* | 33.6 M &nbsp;*0.250* | 117.4 M &nbsp;*0.250* |
| **grouped** | **67.1 M** | **184.5 M** |
| ratio &nbsp;*$$P_{\text{grp}}/P_{\text{std}}$$* | 0.250 + 0.250 = **0.50×** | 0.143 + 0.250 = **0.39×** |

Both forms are position-wise, so FLOPs per token = 2 × parameters and the FLOP ratios are identical. Realized time is not: prefill trades one deep GEMM for $$g$$ shallow ones plus a second kernel and a materialized `[b, s, d_g·g]`.

The $$A$$/$$B$$ split above is worth one more look, because it says where the design runs out. $$A$$'s share contains $$d$$ and shrinks as models widen; $$B$$'s share does not contain $$d$$ and never moves. So the ¼ the ratio converges to **is entirely $$B$$**: asymptotically the grouped down-projection is free and the mixer is the whole cost. That is not a distant limit. At Pro the split is already 67.1 M against 117.4 M — one generation in, the matrix this design _added_ is 64% of what the output projection costs.

## 7. Summary

What the design costs is not read off the parameter count. $$A$$ and $$B$$ hold 67.1 M values between them, but those are not 67.1 M independent choices of $$M$$: for any invertible $$G_j$$ you can send $$A_j \rightarrow G_j A_j$$ and $$B_j \rightarrow B_j G_j^{-1}$$ and the product is unchanged. Each group carries its own $$G_j$$, so $$g\,d_g^{2} = 8.4$$ M of the stored values are re-parameterization rather than capacity. The design reaches a 58.7 M-dimensional family of maps while spending 67.1 M parameters — 43.75% of the dense matrix's reach for 50% of its weights.

| | Standard $$W_O$$ | V4 grouped |
| :--- | :--- | :--- |
| form | one dense $$d \times hc$$ | per-group $$A_j$$, then dense $$B$$ |
| parameters | 134.2 M | **67.1 M** &nbsp;(0.50×) |
| FLOPs / token | 268.4 M | **134.2 M** &nbsp;(0.50×) |
| asymptotic saving, large $$d$$ | — | **→ 4×** |
| distinct maps reachable | 134.2 M dims | 58.7 M dims &nbsp;(43.75%) |

The bet, then, is that head outputs inside a group are redundant enough to survive being read out through a narrow private matrix, and a decade of grouped convolutions suggests the block-diagonal half of that is nearly free. What it leaves open is the problem it created: by §6's split, $$B$$ is the cost from here on, and $$B$$ has to be dense, because putting the groups back in contact is its only job. Whether that matrix has cheap structure of its own — a butterfly, a Monarch factorization, a shuffle — is the next question.

None of it is settled empirically. There is no ablation of the grouped output projection: not against a dense $$W_O$$, not a sweep over $$g$$ or $$d_g$$. The word _ablation_ does not occur in the paper — there are no architecture ablations of any kind. DeepSeek surely ran them. What is published is a cost argument and a construction, and no evidence about what either buys or costs in quality. The paper does not even claim the design is free; that claim has been supplied by its readers.

The absence matters more than it first looks, because of what the operation is. Moving from convolutions to transformers was in large part a campaign to strip out inductive bias — to stop hand-specifying what may talk to what, and let attention learn it. The grouped output projection puts some back. It fixes, before any data arrives, that these channels are read out together and those are not, along a partition that is arbitrary: aligned to heads only because that is how the tensor happens to be laid out.

And if that costs nothing measurable, the interesting conclusion is not about the projection. It is that the dense matrix was never using what it was given — that 134.2 M weights were buying what 58.7 M can buy, and a partition drawn by hand was a good enough guess at which half to throw away. We have no account of which parameters a dense projection wastes, or why, or how to find them without guessing. Until we do, a result like this one is less a design win than a reminder that we are probably overspending nearly everywhere, and only ever finding out by accident.

---

> ##### COLOPHON
>
> Numbers throughout are per attention layer, for the V4-Flash configuration ($$d = 4096$$, $$h = 64$$, $$c = 512$$, $$g = 8$$, $$d_g = 1024$$), with V4-Pro figures ($$d = 7168$$, $$h = 128$$, $$c = 512$$, $$g = 16$$, $$d_g = 1024$$) given where the scaling behaviour is the point. $$W_O$$ runs after attention, so none of this touches the KV cache — the grouped projection and the MLA cache-compression story are orthogonal designs that happen to share a paper.
{: .block-tip }
