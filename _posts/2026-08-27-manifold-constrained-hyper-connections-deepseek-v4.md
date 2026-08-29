---
layout: post
title: Modern residual streams - mHC and Gated Residual
date: 2026-08-27
description: DeepSeek-V4's mHC and Qwen3.8-Flash-Next's Gated Residual widen the residual stream into four dynamically controlled lanes.
tags: deepseek qwen residual architecture
categories: research
related_posts: false
toc:
  sidebar: left
---

The residual stream is the route by which information survives depth: every attention and feed-forward sub-layer reads from it, writes back to it, and relies on it for stable signal propagation. Changing that route touches three things at once: how much cross-layer state the model can carry, whether signals and gradients remain well conditioned, and how much memory traffic each layer pays. mHC and Gated Residual are attempting to widen the residual stream, while not adding a major computational burden.

What draws me to these mechanisms is their use of dynamic parameterization: learned weights use the current token state to generate another set of per-token weights, which decide how each sub-layer reads from and writes to the residual stream. That idea has roots in hypernetworks - a field close to my heart.

We will start with *Manifold-Constrained Hyper-Connections* (mHC), introduced by DeepSeek in [Xie et al., 2025](https://arxiv.org/abs/2512.24880) and used in V4, then use it to frame Qwen3.8-Flash-Next's design.

{% include figure.liquid path="assets/img/mhc.png" class="img-fluid rounded z-depth-1" zoomable=true caption="(a) Residual: one stream, identity skip. (b) Hyper-Connections: $n$ streams, compress into the block, expand out, mix on the skip. (c) mHC: the same topology with the maps constrained. From DeepSeek." %}

#### Why widen the residual

In a standard pre-norm Transformer, every sub-layer reads and updates the same residual stream. A feature written early must compete with everything written after it. Widening the residual gives those features separate paths through depth without changing $\mathcal{F}$.

A wider residual would cost more if that width entered the block: attention and the FFN scale with hidden size. It does not enter. The extra width lives on the skip as $n$ streams of the original hidden size $C$; they are compressed to a single stream in $\mathbb{R}^C$, before the block and expanded back to $n$ after it, so attention and the FFN still run at $C$.

#### Compress, compute, expand

mHC (and Hyper-Connections before it) reconcile the two widths with three maps:

- **Compress** $n \to 1$ entering the block: $H^{\mathrm{pre}}_l$, one scalar weight per stream.
- **Expand** $1 \to n$ leaving it: $H^{\mathrm{post}\top}_l$, one gain per stream.
- **Mix** $n \to n$ along the skip, which never enters the block: $H^{\mathrm{res}}_l$.

$$
\mathbf{x}_{l+1} \;=\; \underbrace{H^{\mathrm{res}}_{l}\,\mathbf{x}_{l}}_{\text{skip: mix streams}} \;+\; \underbrace{H^{\mathrm{post}\top}_{l}\,\mathcal{F}\!\left(H^{\mathrm{pre}}_{l}\,\mathbf{x}_{l},\,W_{l}\right)}_{\text{read}\;\to\;\text{compute}\;\to\;\text{write}}
$$

Within these maps, every channel of a stream gets the same weight: they blend whole streams without mixing channels, so there is no $C \times C$ matrix anywhere.

Reading panel (b) from bottom to top, the topology has one initialization step followed by three moves repeated at every sub-layer.

**Enter.** Before the first layer, the embedding $x_0 \in \mathbb{R}^{1 \times C}$ is copied into $n$ identical streams:

$$
\mathbf{x}_0 = \big(x_0^\top,\dots,x_0^\top\big)^\top \in \mathbb{R}^{n\times C}
$$

This initialization adds no diversity by itself, but it gives the streams room to diverge later. From here the residual remains $n \times C$ per token until the final readout. Throughout, $l$ indexes **sub-layers** - attention, then feed-forward - so there are two updates per Transformer block. $\mathbf{x}_{l+1}$ is the residual after one sub-layer, not after a whole block.

**Compress, $n \to 1$.** The block input is a weighted sum of the streams, one scalar per stream:

$$
\underbrace{h^{\mathrm{in}}_l}_{1\times C}\;=\;\underbrace{H^{\mathrm{pre}}_l}_{1\times n}\,\underbrace{\mathbf{x}_l}_{n\times C}
$$

**Compute, width $C$.** Pre-norm, then the block. This RMSNorm normalizes the block input and is separate from the RMSNorm later used to generate the maps:

$$
\underbrace{h^{\mathrm{out}}_l}_{1\times C}\;=\;\mathcal{F}\big(\mathrm{RMSNorm}(h^{\mathrm{in}}_l)\big)
$$

**Expand, $1 \to n$, and mix.** The block output is fanned back across streams, the saved state is mixed along the skip, and the two are summed - the figure's two branches meeting at $\oplus$:

$$
\underbrace{h^{\mathrm{post}}_l}_{n\times C}=\underbrace{H^{\mathrm{post}\top}_l}_{n\times 1}\,\underbrace{h^{\mathrm{out}}_l}_{1\times C},\qquad \underbrace{h^{\mathrm{res}}_l}_{n\times C}=\underbrace{H^{\mathrm{res}}_l}_{n\times n}\,\underbrace{\mathbf{x}_l}_{n\times C},\qquad \mathbf{x}_{l+1}=h^{\mathrm{res}}_l+h^{\mathrm{post}}_l
$$

Each sub-layer applies this procedure once, so a Transformer block applies it twice. Before the LM head, the $n$ streams are combined into one.

#### The skip has to survive depth

DeepSeek's diagnosis is not that mixing is bad. Unconstrained $H^{\mathrm{res}}$, multiplied across depth, stops behaving like an identity: signals explode or vanish. mHC therefore projects each mix onto $\mathcal{M}$, the doubly stochastic matrices: non-negative, every row and every column summing to $1$. The rows make each output stream a weighted average of the input streams, which a softmax over streams would give you too. The columns are the extra ask: they hold the total across streams fixed, and they are what makes the property survive multiplication. A product of doubly stochastic matrices is still doubly stochastic, so the skip is as well behaved at layer $L$ as it was at layer $1$. The projection is Sinkhorn–Knopp, run for twenty iterations.

#### Hypernetworks are back

A hypernetwork is a net whose weights generate the weights of another map. That is what happens here: learned projection weights transform $\mathbf{x}_l$ into $H^{\mathrm{pre}}_l$, $H^{\mathrm{post}}_l$, and $H^{\mathrm{res}}_l$, so stored weights generate a new set of routing weights for every token. The projection weights and static biases are learned parameters; the three maps themselves are **generated per token** from the current state. Flatten, RMS-normalize, one matmul plus a bias, then constrain. Each map has its own learnable scale - $\alpha_l^{\mathrm{pre}}$, $\alpha_l^{\mathrm{post}}$, and $\alpha_l^{\mathrm{res}}$ - initialized to $0.01$, so the static biases dominate initially and input-dependence phases in:

$$
\bar{\mathbf{x}}_l = \mathrm{vec}(\mathbf{x}_l) \in \mathbb{R}^{1\times nC},\qquad x'_l = \mathrm{RMSNorm}(\bar{\mathbf{x}}_l)
$$

$$
\begin{aligned}
H^{\mathrm{pre}}_l  &= \sigma\!\big(\alpha_l^{\mathrm{pre}}\, x'_l\,W^{\mathrm{pre}}_l + b^{\mathrm{pre}}_l\big) &&\in (0,1)^{1\times n} \\
H^{\mathrm{post}}_l &= 2\sigma\!\big(\alpha_l^{\mathrm{post}}\, x'_l\,W^{\mathrm{post}}_l + b^{\mathrm{post}}_l\big) &&\in (0,2)^{1\times n} \\
H^{\mathrm{res}}_l  &= \mathrm{Sinkhorn}\!\big(\alpha_l^{\mathrm{res}}\, \mathrm{mat}(x'_l\,W^{\mathrm{res}}_l) + b^{\mathrm{res}}_l\big) &&\in \mathcal{M}
\end{aligned}
$$

#### Qwen's Gated Residual
Qwen introduce the **Gated Residual** (GR) in their report, [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf), which keeps mHC's widened residual and drops the mix. Same $n = 4$ streams of width $C$. Residual $\mathbf{x}_l \in \mathbb{R}^{n \times C}$, block input $h^{\mathrm{in}}_l$, block output $h^{\mathrm{out}}_l$.

**Compress.**
GR first uses RMSNorm on each stream separately:

$$
(x'_l)_i = \mathrm{RMSNorm}((\mathbf{x}_l)_i;\,\gamma_i),\qquad i=1,\dots,n
$$

GR's compress is an elementwise gate $G \in \mathbb{R}^{n \times C}$ on the per-stream normalized residual, then a mean:

$$
h^{\mathrm{in}}_l \;=\; \frac{1}{n}\sum_{i=1}^{n} G_i \odot (x'_l)_i
$$

Both contract $n \to 1$ to a $C$-dimensional block input. $H^{\mathrm{pre}}_l$ shares one weight across all $C$ channels of a stream; $G$ assigns a separate weight to every stream and channel. That extra granularity *is* the design: expressiveness moved into the read. Both are hypernetwork output, generated per token from the current state; GR just asks for $nC$ numbers where mHC asked for $n$.

A low-rank bottleneck ($r = C/8$) on the normalized streams emits $G$:

$$
G = \mathrm{unvec}\!\Big(\sigma\big(W_u\,\mathrm{SiLU}(\tfrac{1}{n} W_d\,\mathrm{vec}(x'_l))\big)\Big) \in \mathbb{R}^{n \times C}
$$

with $W_d \in \mathbb{R}^{r \times nC}$ and $W_u \in \mathbb{R}^{nC \times r}$.

**Expand.** $h^{\mathrm{out}}_l = \mathcal{F}(h^{\mathrm{in}}_l)$, no extra RMSNorm. The write is $n$ scalars $s$:

$$
s = 2\sigma\!\big(\tfrac{1}{n} W_w\,\mathrm{vec}(x'_l)\big) \in \mathbb{R}^{n \times 1},\qquad
\mathbf{x}_{l+1} = \mathbf{x}_l + s\, h^{\mathrm{out}}_l
$$

with $W_w \in \mathbb{R}^{n \times nC}$.

Two moves separate GR from mHC. The compress is elementwise ($G \in \mathbb{R}^{n \times C}$ vs mHC's $H^{\mathrm{pre}} \in \mathbb{R}^{1 \times n}$), and there is no mix on the skip: $\mathbf{x}_{l+1} = \mathbf{x}_l + s\, h^{\mathrm{out}}_l$. Per-stream RMSNorm is part of that read, so it replaces the block pre-norm rather than sitting in front of it.

<style>
.hcfig{--ds:#0f6b78;--ds-ink:#0b545f;--ds-soft:#e0f0f2;--ds-edge:#a9d5db;--qw:#9a3364;--qw-ink:#7e2751;--qw-soft:#f9e7ef;--qw-edge:#e4b2cb;color:var(--global-text-color);margin:2.2rem 0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .hcfig{--ds:#57ccda;--ds-ink:#8fe0ea;--ds-soft:#0e2c31;--ds-edge:#1d4d55;--qw:#f184b2;--qw-ink:#f7abca;--qw-soft:#331320;--qw-edge:#5e2440}}
:root[data-theme=dark] .hcfig{--ds:#57ccda;--ds-ink:#8fe0ea;--ds-soft:#0e2c31;--ds-edge:#1d4d55;--qw:#f184b2;--qw-ink:#f7abca;--qw-soft:#331320;--qw-edge:#5e2440}
.hcfig .scroll{overflow-x:auto}
.hcfig svg{display:block;width:100%;min-width:820px;height:auto}
.hcfig figcaption{font-size:.88rem;line-height:1.55;color:var(--global-text-color-light);margin-top:1rem;padding-top:.8rem;border-top:1px solid var(--global-divider-color)}
.hcfig figcaption b{color:var(--global-text-color)}
</style>
<figure class="hcfig">
<div class="scroll">
<svg viewBox="0 0 960 322" role="img" aria-label="Two sub-layers side by side. Both read four residual streams through a read map into the block and write the block output back through four per-stream scalars. mHC additionally passes the four streams through an n by n doubly stochastic mixing matrix; GR leaves the streams unmixed, as straight identity lines.">
<defs>
<marker id="hc-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>
<marker id="hc-arDs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--ds)"/></marker>
<marker id="hc-arQw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--qw)"/></marker>
</defs>
<g font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
<text x="20" y="28" font-family="ui-monospace, Menlo, monospace" font-size="13.5" font-weight="600" fill="var(--ds)" letter-spacing="1.4">DEEPSEEK-V4 mHC</text>
<g stroke="currentColor" stroke-width="1.6" opacity=".55"><line x1="20" y1="70" x2="286" y2="70"/><line x1="20" y1="100" x2="286" y2="100"/><line x1="20" y1="130" x2="286" y2="130"/><line x1="20" y1="160" x2="286" y2="160"/></g>
<g stroke="var(--ds)" stroke-width="1.6"><line x1="338" y1="70" x2="446" y2="70" marker-end="url(#hc-arDs)"/><line x1="338" y1="100" x2="446" y2="100" marker-end="url(#hc-arDs)"/><line x1="338" y1="130" x2="446" y2="130" marker-end="url(#hc-arDs)"/><line x1="338" y1="160" x2="446" y2="160" marker-end="url(#hc-arDs)"/></g>
<g fill="var(--ds)"><circle cx="96" cy="70" r="4.5"/><circle cx="96" cy="100" r="4.5"/><circle cx="96" cy="130" r="4.5"/><circle cx="96" cy="160" r="4.5"/></g>
<g stroke="var(--ds)" stroke-width="1.2" fill="none" opacity=".75"><path d="M96,70 C146,70 146,246 172,252"/><path d="M96,100 C146,100 146,250 172,256"/><path d="M96,130 C146,130 146,256 172,260"/><path d="M96,160 C146,160 146,258 172,264"/></g>
<text x="96" y="50" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="12.5" fill="var(--ds)">Hᵖʳᵉ</text>
<text x="96" y="194" stroke="var(--global-bg-color)" stroke-width="3.5" paint-order="stroke" text-anchor="middle" font-size="12" fill="currentColor" opacity=".65">1 × n scalars</text>
<rect x="172" y="245" width="64" height="28" rx="3" fill="none" stroke="currentColor" stroke-width="1.3" opacity=".7"/>
<text x="204" y="263" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="11.5" fill="currentColor">RMSNorm</text>
<line x1="236" y1="259" x2="252" y2="259" stroke="currentColor" stroke-width="1.3" opacity=".7" marker-end="url(#hc-ar)"/>
<rect x="254" y="240" width="92" height="38" rx="3" fill="var(--ds-soft)" stroke="var(--ds-edge)" stroke-width="1.3"/>
<text x="300" y="264" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--ds-ink)">attn / FFN</text>
<rect x="286" y="54" width="52" height="122" rx="3" fill="var(--ds-soft)" stroke="var(--ds)" stroke-width="1.5"/>
<g stroke="var(--ds)" stroke-width=".9" opacity=".62"><line x1="286" y1="70" x2="338" y2="130"/><line x1="286" y1="100" x2="338" y2="70"/><line x1="286" y1="130" x2="338" y2="160"/><line x1="286" y1="160" x2="338" y2="100"/><line x1="286" y1="70" x2="338" y2="70"/><line x1="286" y1="160" x2="338" y2="160"/></g>
<text x="312" y="42" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="13" font-weight="600" fill="var(--ds)">Hʳᵉˢ</text>
<text x="312" y="198" stroke="var(--global-bg-color)" stroke-width="3.5" paint-order="stroke" text-anchor="middle" font-size="12" fill="var(--ds-ink)" font-weight="600">n × n doubly stochastic</text>
<text x="312" y="214" stroke="var(--global-bg-color)" stroke-width="3.5" paint-order="stroke" text-anchor="middle" font-size="11.5" fill="currentColor" opacity=".6">Sinkhorn, 20 iters</text>
<g stroke="var(--ds)" stroke-width="1.2" fill="none" opacity=".75"><path d="M346,252 C392,248 380,84 398,74"/><path d="M346,256 C394,252 386,110 398,104"/><path d="M346,260 C394,258 388,132 398,134"/><path d="M346,264 C392,262 380,158 398,164"/></g>
<g fill="var(--ds)"><circle cx="398" cy="70" r="4.5"/><circle cx="398" cy="100" r="4.5"/><circle cx="398" cy="130" r="4.5"/><circle cx="398" cy="160" r="4.5"/></g>
<text x="412" y="50" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="12.5" fill="var(--ds)">Hᵖᵒˢᵗ</text>
</g>
<line x1="480" y1="18" x2="480" y2="304" stroke="currentColor" stroke-width="1" opacity=".18" stroke-dasharray="3 5"/>
<g transform="translate(500,0)" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
<text x="20" y="28" font-family="ui-monospace, Menlo, monospace" font-size="13.5" font-weight="600" fill="var(--qw)" letter-spacing="1.4">QWEN3.8-FLASH-NEXT GR</text>
<g stroke="currentColor" stroke-width="1.6" opacity=".55"><line x1="20" y1="70" x2="286" y2="70"/><line x1="20" y1="100" x2="286" y2="100"/><line x1="20" y1="130" x2="286" y2="130"/><line x1="20" y1="160" x2="286" y2="160"/></g>
<g stroke="var(--qw)" stroke-width="1.6"><line x1="286" y1="70" x2="446" y2="70" marker-end="url(#hc-arQw)"/><line x1="286" y1="100" x2="446" y2="100" marker-end="url(#hc-arQw)"/><line x1="286" y1="130" x2="446" y2="130" marker-end="url(#hc-arQw)"/><line x1="286" y1="160" x2="446" y2="160" marker-end="url(#hc-arQw)"/></g>
<g fill="none" stroke="currentColor" stroke-width="1.2" opacity=".55"><rect x="56" y="61" width="17" height="18" rx="2"/><rect x="56" y="91" width="17" height="18" rx="2"/><rect x="56" y="121" width="17" height="18" rx="2"/><rect x="56" y="151" width="17" height="18" rx="2"/></g>
<text x="64" y="50" text-anchor="middle" font-size="11.5" fill="currentColor" opacity=".7">norm</text>
<g fill="var(--qw)"><rect x="104" y="59" width="8" height="22" rx="1.5"/><rect x="104" y="89" width="8" height="22" rx="1.5"/><rect x="104" y="119" width="8" height="22" rx="1.5"/><rect x="104" y="149" width="8" height="22" rx="1.5"/></g>
<g stroke="var(--qw)" stroke-width="1.2" fill="none" opacity=".75"><path d="M112,70 C158,70 158,246 190,252"/><path d="M112,100 C158,100 158,250 190,256"/><path d="M112,130 C158,130 158,256 190,260"/><path d="M112,160 C158,160 158,258 190,264"/></g>
<text x="108" y="50" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="13" font-weight="600" fill="var(--qw)">G</text>
<text x="108" y="194" stroke="var(--global-bg-color)" stroke-width="3.5" paint-order="stroke" text-anchor="middle" font-size="12" fill="var(--qw-ink)" font-weight="600">n × C elementwise</text>
<rect x="190" y="240" width="92" height="38" rx="3" fill="var(--qw-soft)" stroke="var(--qw-edge)" stroke-width="1.3"/>
<text x="236" y="264" text-anchor="middle" font-size="13.5" font-weight="600" fill="var(--qw-ink)">attn / FFN</text>
<text x="236" y="296" stroke="var(--global-bg-color)" stroke-width="3.5" paint-order="stroke" text-anchor="middle" font-size="11.5" fill="currentColor" opacity=".6">no pre-norm; the gate is it</text>
<rect x="286" y="54" width="52" height="122" rx="3" fill="none" stroke="currentColor" stroke-width="1.3" opacity=".28" stroke-dasharray="4 4"/>
<text x="312" y="42" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="13" font-weight="600" fill="currentColor" opacity=".45">no Hʳᵉˢ</text>
<text x="312" y="200" stroke="var(--global-bg-color)" stroke-width="3.5" paint-order="stroke" text-anchor="middle" font-size="12" fill="currentColor" opacity=".62" font-weight="600">no mixing</text>
<g stroke="var(--qw)" stroke-width="1.2" fill="none" opacity=".75"><path d="M282,252 C392,248 380,84 398,74"/><path d="M282,256 C394,252 386,110 398,104"/><path d="M282,260 C394,258 388,132 398,134"/><path d="M282,264 C392,262 380,158 398,164"/></g>
<g fill="var(--qw)"><circle cx="398" cy="70" r="4.5"/><circle cx="398" cy="100" r="4.5"/><circle cx="398" cy="130" r="4.5"/><circle cx="398" cy="160" r="4.5"/></g>
<text x="410" y="50" text-anchor="middle" font-family="ui-monospace, Menlo, monospace" font-size="13" font-weight="600" fill="var(--qw)">s</text>
</g>
</svg>
</div>
<figcaption>Circle: one scalar per stream ($H^{\mathrm{pre}}_l$). Bar: a full $C$-wide gate ($G$). Solid box: mHC's $n \times n$ mix; dashed: where GR has none.</figcaption>
</figure>

Qwen's finding, in their words: once the read and write are expressive enough, $H^{\mathrm{res}}$ adds little. Dropping it removes a source of instability that needed its own constraint, and it removes a full read of the residual on the skip. The hypernetwork spends itself on the read instead - that is the granularity they say mattered. A contribution $s_i h^{\mathrm{out}}_l$ written into stream $i$ remains in that stream; later sub-layers can read it jointly with the other streams, but no $H^{\mathrm{res}}$ moves it directly between them.

#### The importance of dynamic parameterization

The table below summarizes Qwen's Table 5. What is interesting to see are two leaps in performance. The first is static mHC over the baseline, which shows what widening the residual stream buys, even with static maps. The second is between static and dynamic mHC, which shows the importance of dynamic parameterization. Another important outcome from the table is that GR is not drastically superior to mHC. Note that the mHC rows are Qwen's re-implementation.

| Residual scheme | Loss $\downarrow$ | Benchmark avg $\uparrow$ | $n\times n$ mixer |
| :--- | ---: | ---: | :--- |
| Pre-norm baseline | 1.617 | 50.91 | - |
| mHC, static | 1.596 | 52.49 | yes, Sinkhorn |
| mHC, dynamic | 1.594 | 54.47 | yes, Sinkhorn |
| Gated Residual | 1.590 | 54.66 | none |

#### Summary
I think that the main thing to take from Qwen's analysis is not which one is better, mHC or GR, but that Qwen achieved similar results without an explicit $n\times n$ mixer on the skip.

Both designs make the residual path data-dependent, and the table shows what that is worth: turning the static maps dynamic moved the benchmark average more than the widening itself did. In the [previous post]({% post_url 2026-08-14-grouped-output-projection-deepseek-v4 %}) I watched grouped convolution come to life in V4's output projection; here it is hypernetworks, another decade-old idea back inside a frontier LLM. It will be interesting to see how many more operations end up parameterized this way.
