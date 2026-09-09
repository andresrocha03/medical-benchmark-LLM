# Tabular Foundation Models (TFM)
    - [Post explaining TFMs](https://towardsdatascience.com/tabular-foundation-models/)

    - The idea is to use a pre-trained model to predict labels without having to actually train on your dataset. The training rows become context for predicting text instances.

    - They have some advantages such as not requiring hyperparameter tuning, being robust to outliers, to missing data and to non-informative features.

## TabPFN
    - Prior fitted network.
    - Learns a prior over tabular prediction problems and performs prediction through in-context learning.
    - The key contribution of the TabPFN family is the idea that a neural network can learn a general-purpose tabular learning algorithm during pretraining.

TabPFN belongs to the family of Prior-Fitted Networks (PFNs).

Instead of training a new machine-learning model from scratch for every dataset, TabPFN is pretrained on a very large number of generated tabular prediction tasks.

The model learns to approximate:

[
q_\theta(y_{\text{test}} \mid x_{\text{test}}, D_{\text{train}})
]

where:

[
D_{\text{train}} =
{(x_1,y_1),\dots,(x_n,y_n)}
]

is the labeled training dataset.

At inference time, the training examples are passed directly to the Transformer as context together with the test example.

This means that TabPFN performs in-context learning rather than ordinary dataset-specific optimization.

A crucial characteristic of TabPFN is that it is pretrained largely on synthetic datasets.

However, these datasets are not simply random noise.

They are generated from a designed probability distribution over possible datasets:

[
D \sim p(D)
]

The function (p(D)) represents the model's prior assumptions about the kinds of relationships that may occur in tabular problems.

The synthetic generator can randomly vary: the number of features, feature distributions, relationships between variables, graph structures, nonlinear functions, noise levels, target-generation rules, classification or regression boundaries.
The original TabPFN family uses structural causal models (SCMs) as an important part of this synthetic prior. Later models introduced richer priors including tree-based structures and more sophisticated directed acyclic graphs and mappings.

Therefore:

[
\text{synthetic data}
\neq
\text{pure randomness}
]

Instead:

[
\text{random samples from designed data-generating processes}
]

For example, one synthetic task might approximately follow:

[
x_1 \sim \mathcal N(0,1)
]

[
x_2 = \sin(x_1)+\epsilon
]

[
x_3 = x_1x_2+\epsilon
]

[
y =
\mathbb{1}(x_2+x_3 > c)
]

while another generated task may contain tree-like decision boundaries.

The purpose is to expose the Transformer to a very large variety of statistical relationships.

## Tarte
    - [Tarte](https://arxiv.org/abs/2505.14415)
    - Learns semantic representations of table entries by combining column names, strings and numerical values.

TARTE explicitly models the combination:

[
(\text{column name}, \text{cell value})
]

rather than treating the cell value independently.

TARTE differs strongly from TabPFN in its source of pretraining knowledge.

TabPFN is primarily pretrained on generated machine-learning tasks.

TARTE instead performs what its authors call knowledge pre-training.

The model is pretrained using large relational knowledge sources enriched with numerical attributes, including information derived from Wikidata.


Another important difference is the output of the foundation model.

TabPFN is primarily used as a predictor.

TARTE is designed as a reusable representation backbone.

Conceptually:

[
X
\rightarrow
\text{TARTE}
\rightarrow
Z
]

where (Z) is a learned representation of the table.

These representations can then be used by another predictor:

[
Z
\rightarrow
\text{XGBoost}
\rightarrow
y
]

or:

[
Z
\rightarrow
\text{neural network}
\rightarrow
y
]

TARTE can also be fine-tuned for a downstream task.

The authors emphasize that its pretrained representations can be frozen, fine-tuned, or combined with other machine-learning models.

## TabCLv2
    - [TabICLv2]()
    - Follows the ICL philosophy, but focuses on making this approach more efficient, scalable and robust to long contexts.

abPFNv2 uses a cell-based architecture with alternating row and column attention.

For a table with:

(n) rows;
(m) columns;

the paper describes TabPFNv2-style complexity approximately as:

[
O(n^2m + nm^2)
]

TabICL instead first processes columns and compresses each row into a fixed-dimensional representation.

Dataset-level in-context learning is then performed on these row embeddings.

This gives approximately:

[
O(n^2 + nm^2)
]

complexity.


TabICLv2 introduces repeated feature grouping.

Instead of always representing each feature independently, multiple feature combinations are formed using circular shifts.

For example, the paper uses groups based on offsets such as:

[
(0,1,3)
]

This creates overlapping feature groups while maintaining detailed feature information.

The goal is partly to reduce representation collapse, where different features with similar distributions could otherwise become difficult for the model to distinguish.


abICLv2 also injects information about the target (y_i) early in the representation of training examples.

Instead of treating the label only as a separate token later in the network, a target embedding is added directly to feature representations for training samples.

One of the major problems for Transformer-based in-context learners is that attention behavior changes as the number of context examples grows.

With standard softmax attention:

[
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V
]

as the number of possible keys increases, attention can become increasingly diffuse.

This phenomenon is described in the TabICLv2 paper as attention fading.

The problem is important because a model may be pretrained on relatively short contexts but later be applied to much larger datasets.

TabICLv2 introduces Query-Aware Scalable Softmax (QASSMax) to improve this long-context behavior.

The underlying idea is to adapt the scaling of attention based on context length and the query, allowing the model to preserve sharper attention even as the number of training examples grows.

This improves length generalization.