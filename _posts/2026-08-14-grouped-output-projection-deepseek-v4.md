---
layout: post
title: Grouped Output Projection
date: 2026-08-21
description: The Grouped Output Projection in DeepSeek-V4, goals, technique and relations to grouped output attention

---


While reading the DeepSeek-V4 implementation on Hugging Face I found an output projection I did not recognize. In the report it gets a name and a single paragraph: _Grouped Output Projection_.

The motivation for this new block is mainly cost: V4's attention emits a very wide vector, and the researchers note that mapping it back to the residual stream with one dense matrix "will impose a substantial computational burden".
What they build instead borrows the idea behind grouped convolution and applies it to the attention output's channel dimension. What a grouped convolution does is split input and output channels into $g$ matching groups and compute each output channel from only the $C_{\text{in}}/g$ inputs in its own group rather than all $C_{\text{in}}$ - cutting parameters and FLOPs by a factor of $g$.


The attention operation produces $h$ head outputs of width $c$, flattened into one tensor $O$ and mapped back to the residual stream by a single dense matrix. The cost of the output projection is the input dim times the output dim: the input dim is the number of heads, denoted $h$, times the size of each head, denoted $c$; the output dim is the hidden size, denoted $d$. For completeness, batch size and sequence length are denoted $b$ and $s$ respectively. All numbers below are per attention layer, for V4-Flash: $d = 4096$, $h = 64$, $c = 512$, so $hc = 32768$.

Let us formalize the dense projection:

$$
Y = O W_O^{\top}, \qquad O \in \mathbb{R}^{b \times s \times hc}, \quad W_O \in \mathbb{R}^{d \times hc}, \quad Y \in \mathbb{R}^{b \times s \times d}
$$


The arithmetic for every token is simply:

$$
P_{\text{std}} = d\,hc, \qquad \text{FLOPs/token} = 2\,d\,hc
$$

That is $d \times hc = 4096 \times 32768 = 134.2$ M parameters and 268.4 MFLOP per token.

So how can we move from one width to another without paying the full cost? The easy solution is a bottleneck layer: rather than one linear projection, factor it and pass through a narrow intermediate of some low dimension $r$. Project $hc \rightarrow r$ first, then $r \rightarrow d$, with $r$ small.

$$
P_{\text{bot}} = r\,hc + d\,r = r(hc + d), \qquad \text{FLOPs/token} = 2\,r(hc + d)
$$

So take $r = 1024$. The FLOPs per token are now

$$
2\,r(hc + d) = 2 \cdot 1024 \cdot 36864 \approx 75.5 \text{ M}
$$

Just over a quarter of the FLOPs - but there is no free lunch. Factoring the projection forces the transformation through an $r$-dimensional bottleneck, constraining the effective weight matrix to rank at most $r$. Some works suggest that this kind of low-rank bottleneck can hurt pretraining and training dynamics compared with a dense layer ([Wei et al., 2024](https://arxiv.org/abs/2406.16450); [Han et al., 2024](https://arxiv.org/abs/2406.02214); [Godey & Artzi, 2026](https://arxiv.org/abs/2603.10145)).

#### Grouped output projection

V4's block has two parts. Start with the first: a grouped down-projection $A$, on its own. Three symbols describe it: $g$ is the number of groups, $c_g = hc/g$ is the width of one group, and $d_g$ is what each group is squeezed to, with $d_g < c_g$. The $g$ squeezed groups are concatenated, so $A$ maps $hc \rightarrow d_g g$.

A small example, with numbers picked only to keep the picture readable: $hc = 12$ and $g = 3$, so $c_g = 4$; take $d_g = 2$, and the concatenated output is $d_g g = 6$.

<figure style="margin: 2rem 0;">
<div style="background: var(--global-card-bg-color); border: 1px solid var(--global-divider-color); border-radius: 4px; padding: 1.4rem;">
<svg viewBox="0 0 600 460" role="img" aria-label="One 12-element vector, split into 3 groups of 4; each group multiplied by its own 2 by 4 matrix, producing 3 groups of 2 output elements, which are concatenated back into one 6-element vector." style="width:100%;height:auto;display:block;">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="currentColor"/>
    </marker>
  </defs>
  <g font-family="ui-monospace, Menlo, monospace" font-size="11">

    <text x="8" y="14" fill="var(--global-text-color-light)" font-size="9.5" letter-spacing="1.3">ONE VECTOR - 12 ELEMENTS</text>

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
      <text x="90" y="234">A₀</text><text x="270" y="234">A₁</text><text x="450" y="234">A₂</text>
    </g>
    <g fill="var(--global-text-color-light)" text-anchor="middle" font-size="9.5">
      <text x="90" y="251">2 × 4</text><text x="270" y="251">2 × 4</text><text x="450" y="251">2 × 4</text>
    </g>

    <g stroke="var(--global-text-color-light)" stroke-width="1" marker-end="url(#ar2)" color="var(--global-text-color-light)">
      <line x1="90" y1="264" x2="90" y2="290"/><line x1="270" y1="264" x2="270" y2="290"/><line x1="450" y1="264" x2="450" y2="290"/>
    </g>

    <text x="8" y="288" fill="var(--global-text-color-light)" font-size="9.5" letter-spacing="1.3">3 GROUPS OF 2</text>

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

    <text x="430" y="348" fill="#B8912F" font-size="10">concatenate</text>
    <g stroke="var(--global-text-color-light)" stroke-width="1" marker-end="url(#ar2)" color="var(--global-text-color-light)">
      <line x1="90" y1="330" x2="183.33" y2="366"/><line x1="270" y1="330" x2="270" y2="366"/><line x1="450" y1="330" x2="356.67" y2="366"/>
    </g>

    <text x="8" y="384" fill="var(--global-text-color-light)" font-size="9.5" letter-spacing="1.3">ONE VECTOR - 6 ELEMENTS</text>

    <rect x="140" y="394" width="260" height="30" fill="var(--global-text-color-light)" fill-opacity=".08" stroke="var(--global-text-color-light)" stroke-width="1.4"/>
    <g stroke="var(--global-divider-color)" stroke-width=".8" opacity=".6">
      <line x1="183.33" y1="394" x2="183.33" y2="424"/>
      <line x1="226.67" y1="394" x2="226.67" y2="424"/>
      <line x1="270.00" y1="394" x2="270.00" y2="424"/>
      <line x1="313.33" y1="394" x2="313.33" y2="424"/>
      <line x1="356.67" y1="394" x2="356.67" y2="424"/>
    </g>
    <g fill="var(--global-text-color)" text-anchor="middle" font-size="10.5">
      <text x="161.67" y="414">o₀</text>
      <text x="205.00" y="414">o₁</text>
      <text x="248.33" y="414">o₂</text>
      <text x="291.67" y="414">o₃</text>
      <text x="335.00" y="414">o₄</text>
      <text x="378.33" y="414">o₅</text>
    </g>

    <text x="8" y="446" fill="#B8912F" font-size="10.5">o₀ never sees i₄.</text>
  </g>
</svg>
</div>
<figcaption style="font-size:.85rem;color:var(--global-text-color-light);margin-top:.7rem;padding-top:.7rem;border-top:1px solid var(--global-divider-color);">
<b>First part of the grouped output projection - $A$ only, $hc = 12$ in and $d_g g = 6$ out, $g = 3$.</b> Each group owns a private $d_g \times c_g = 2 \times 4$ matrix and reads only its own $c_g = 4$ inputs. A dense projection would instead wire every one of the 6 outputs to all 12 inputs.
</figcaption>
</figure>

That restricted fan-in is the source of the saving: a dense $d_g g \times hc$ map would cost $6 \times 12 = 72$ parameters and 144 FLOPs; the three private $d_g \times c_g$ blocks cost $3 \times 8 = 24$ parameters and 48 FLOPs.

Now the real numbers. On Flash: $g = 8$, $d_g = 1024$, so $c_g = 4096$ and the concatenated intermediate is $d_g g = 8192$ - twice $d$. What differs from a single bottleneck is exactly that total intermediate width: not $d_g$, but $d_g g$.

$g$ independent maps $c_g \to d_g$ cost

$$
P_A = g \cdot d_g \cdot c_g = d_g\, hc, \qquad \text{FLOPs/token} = 2\, d_g\, hc
$$

$$
P_A = 1024 \times 32768 = 33.6 \text{ M}, \qquad \text{FLOPs/token} = 67.1 \text{ M}
$$

The same first-stage bill as the $r = 1024$ bottleneck, but eight times the intermediate width.

Unfortunately, $A$ on its own has a downside: the groups never interact. Grouped convolution has exactly the same problem, and the literature there settled on two answers - ShuffleNet permutes channels across groups so the next layer sees a different mix, while MobileNet follows its depthwise convolution with a pointwise 1×1 convolution that recombines every channel.

DeepSeek takes the second route. The second part of the grouped output projection is an ordinary linear map $B$ that reads all $g$ group outputs at once and puts them back in contact. The construction is two stages. Write $O^{G}$ for $O$ reshaped to `[b, s, g, c_g]`. **Down-projection**, one independent matrix per group:

$$
Z[b,s,j,:] = O^{G}[b,s,j,:]\, A_j^{\top}, \qquad A_j \in \mathbb{R}^{d_g \times c_g}, \quad j = 0,\ldots,g-1
$$

**Mixing**, flatten the group and channel axes of $Z$ into $Z' \in \mathbb{R}^{b \times s \times d_g g}$, then one shared dense matrix that reads all $g$ groups at once:

$$
Y = Z' B^{\top}, \qquad B \in \mathbb{R}^{d \times d_g g}
$$

Note that the entire transformation is still linear.

#### The code

```python
o = o.view(bsz, seqlen, g, -1)
wo_a = self.wo_a.weight.view(g, d_g, -1)

o = torch.einsum("bsgk,guk->bsgu", o, wo_a)
x = self.wo_b(o.flatten(2))
```

The first two lines only change layout. Attention output $O$ arrives as one wide vector of length $hc$; `view` splits it into $g$ chunks of $c_g$. The weights of $A$ are stored as one packed matrix; `view` splits them the same way, into $g$ matrices of shape $d_g \times c_g$. After that, both tensors share a group axis, and group $j$ of the input sits next to $A_j$.

`einsum` is a named matmul: you label every axis, and any letter that appears on both inputs and not on the output is summed out. Here the string is `"bsgk,guk->bsgu"`. Read it as: for each batch item, token, and group, take that group's $c_g$ channels ($k$) and multiply by that group's $d_g \times c_g$ matrix. $k$ is on both sides and gone from the output, so it is the sum - the actual multiply. $u$ is new, so it is the squeezed width $d_g$. $g$ is on both inputs *and* the output, so it is not summed: group $0$ never sees group $1$'s weights. That is the grouped projection, in one string.

`flatten` concatenates the $g$ short vectors into one of length $d_g g$. `wo_b` is $B$: the ordinary linear map that reads all groups at once and writes the residual stream.

#### The cost of the whole block

$A$ is $g$ blocks of $d_g \times c_g$; $B$ is one dense $d \times d_g g$. Their costs add:

$$
P_A = g \cdot d_g \cdot c_g = d_g\, hc, \qquad P_B = d \cdot d_g g, \qquad P_{A+B} = P_A + P_B
$$

$P_A$ is the first-stage bill above: the $g$ cancels, so the grouped down-projection costs what one global $d_g$-wide map costs. Dividing by the dense $W_O$:

$$
\frac{P_A}{P_{\text{std}}} = \frac{d_g}{d}, \qquad \frac{P_B}{P_{\text{std}}} = \frac{d_g}{c_g}, \qquad \frac{P_{A+B}}{P_{\text{std}}} = \frac{d_g}{d} + \frac{d_g}{c_g}
$$

At Flash both fractions are $\tfrac{1}{4}$:

$$
\frac{d_g}{d} = \frac{1024}{4096} = \tfrac14, \qquad \frac{d_g}{c_g} = \frac{1024}{4096} = \tfrac14
$$

so $A$ and $B$ cost 33.6 M each, together **0.50×** $W_O$. FLOPs per token are $2\times$ parameters either way, so the ratios match.

#### Summary

The grouped output projection is how V4 cuts the cost of the attention output layer. The report never ablates this block; I am sure they checked internally that it did not hurt. I see a lot in common with grouped convolution, and I enjoy watching old deep-learning techniques show up again in LLMs.

DeepSeek has a reputation for architectural creativity, and it is earned: Multi-head Latent Attention (MLA) and DeepSeek Sparse Attention (DSA) in the earlier models, Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) elsewhere in V4. The grouped output projection is the smallest of the set, and definitely the least hyped.


