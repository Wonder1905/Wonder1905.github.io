---
layout: post
title: "Mid-Training & Modern LLM Recipes"
date: 2025-07-11
description: What mid-training actually is, and how it shows up in Olmo-2, Phi-4, Yi, Qwen3, Pangu and SmolLM3.
tags: llm mid-training pre-training long-context reasoning
categories: research
related_posts: false
toc:
  sidebar: left
---

> Disclaimers: The definitions in this post are not rigorous and are presented in a way I think it is easier to grasp the main idea.
{: .block-tip }

We all know about the pre and post-training phases of large language models.

Pre-training today typically involves processing (much) more than 10T tokens from web pages, books, and code, providing models with broad linguistic, factual, and reasoning foundations.

Post-training builds upon the pre-trained LLM by using labeled data to guide the model in producing desired outputs for specific instructions and aligning with human preferences.

Recently, several companies have mentioned mid-training in their job postings (OpenAI, xAI) and technical reports (Phi-3.5/4, Olmo, and Yi).

## Mid-training basic characteristics

1. It occurs after pre-training and before post-training.
2. It's shorter in duration than the pre-training phase.
3. It enhances the model's long context understanding and reasoning abilities.

Let's examine three papers that explicitly mention mid-training:

- **[Olmo-2](https://arxiv.org/pdf/2501.00656)** — "In the second stage (Mid-training), we train on the Dolmino Mix 1124. This mix is smaller, but it contains higher-quality text, as well as synthetic data to boost key abilities."
- **[Phi-4](https://arxiv.org/pdf/2412.08905)** — "includes a mid-training stage where the context length is increased from the original 4K to 16K"
- **[Yi](https://arxiv.org/pdf/2412.01253)** — "In the mid-training stage, we focus on enhancing model capabilities and extending context length... upsampling strategy for high-quality data, emphasizing complex reasoning and multilingual capabilities for low-resource languages."

We can categorize mid-training goals into four main areas:

- 🔥 **High Quality Training Data** — This approach improves a model's performance by training it on a smaller, high-quality dataset curated from the best available sources and after multiple filtering techniques. The Olmo-2 model report introduced this concept systematically with its "Dolmino Mix" dataset.
- 🥖 **Domain and Language Extension** — supporting more languages, mostly low-resource languages, or understanding specialized subjects.
- ☄️ **Long Context Extension** — might happen gradually.
- ⌨️ **Scaling Synthetic Data** — This method uses powerful AI models to generate vast amounts of high-quality "synthetic" training data. Note that synthetic data is mostly not generated from scratch, but more by augmenting existing data. The Olmo-2 report details this as a key part of its mid-training, employing techniques like using AI "judge" models to verify data quality.

An intriguing example of "Scaling Synthetic Data" and "High Quality Training Data" can be observed in the [allenai mid-training dataset](https://huggingface.co/datasets/allenai/mid-training-OpenMathReasoning-rewrite-teacher-student-lecture-filtered), released recently. This dataset features content "rewritten" as dialogues between a teacher and a student:

```text
Determine the median and the mode of Jerelyn's scores
from her first six tests, which were 92, 78, 86, 92,
95, and 91.

Teacher: Today, let's talk about two important
         measures...
Student: So, for the median, I always need to sort the
         data first, right?
Teacher: Exactly! Sorting the data is essential so you
         can correctly identify the middle value or
         values. For the mode, you simply look for the
         value or values that occur the most often.
…
Student: First, I'll sort the numbers in ascending
         order. Arranged from least to greatest: $78$,
         $86$, $91$, $92$, $92$, $95$.
Teacher: Excellent. Since there are six numbers, what
         do you do next?
```

## The pre-training phases

Mid-training is not just a trendy label; it's a substantive concept that many teams apply even when they don't explicitly use the term. Consider [Qwen3](https://arxiv.org/pdf/2505.09388):

> "Pre-training went through a three-stage process:
>
> (1) General stage — context window, 30T tokens and 119 languages and dialects.
>
> (2) Reasoning Stage (S2): increase in the proportion of STEM, coding, reasoning, and synthetic data. Still 4k context length.
>
> (3) Long Context Stage: high-quality long context corpora, sequence length of 32k."

Another example is Huawei's [Pangu](https://arxiv.org/pdf/2505.21411). Its pre-training unfolds in three phases: Phase 1 uses a 4K-token context window, while Phases 2 and 3 expand to 32K tokens. Phase 2 focuses on reasoning by "increasing the amount and quality of the reasoning data." Phase 3 becomes even more selective: "In this phase, priority is given to the data with extremely higher quality and difficulty scores."

Sound familiar?

## Pre-training phases and mid-training

A few days ago, Hugging Face released [SmolLM3](https://huggingface.co/blog/smollm3), a 3B parameter model. The model underwent a three-phase pre-training process that gradually shifted its data distribution toward larger code and math segments.

{% include figure.liquid loading="eager" path="assets/img/smollm3-pretraining-phases.png" class="img-fluid rounded z-depth-1" zoomable=true caption="SmolLM3 three-phase pre-training data mixture. Source: huggingface.co/blog/smollm3" %}

Subsequently, its mid-training phases focus on gradually extending the context length from 4k to 128k tokens and enhancing reasoning capabilities.

## Final Thoughts

- **Blurring Boundaries:** The distinction between LLM training phases is increasingly fuzzy. What some labs label as pre-training, others consider mid-training. We're seeing instruction and reasoning data incorporated earlier in the training process.
- **Out of Scope:** This blog post doesn't cover several important technical aspects, including learning rate scheduling during mid-training, specific data mixture compositions, and how mid-training enhances reasoning through reinforcement learning.

If you have questions, want to discuss this topic, or share comments, please feel free to DM us on X: [barak1724](https://x.com/barak1724).
