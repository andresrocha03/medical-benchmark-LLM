# Tabular Foundation Models (TFM)

[Post explaining TFMs](https://towardsdatascience.com/tabular-foundation-models/)

    - The idea is to use a pre-trained model to predict labels without having to actually train on your dataset. The training rows become context for predicting text instances.

    - They have some advantages such as not requiring hyperparameter tuning, being robust to outliers, to missing data and to non-informative features.

## TabPFN

[TabPFN](https://arxiv.org/pdf/2511.08667)

    - Prior fitted network.
    - Learns a prior over tabular prediction problems and performs prediction through in-context learning.
    - The key contribution of the TabPFN family is the idea that a neural network can learn a general-purpose tabular learning algorithm during pretraining.
    - A crucial characteristic of TabPFN is that it is pretrained largely on synthetic datasets. However, these datasets are not simply random noise. They are generated from a designed probability distribution over possible datasets. The synthetic generator can randomly vary: the number of features, feature distributions, relationships between variables, graph structures, nonlinear functions, noise levels, target-generation rules, classification or regression boundaries. The original TabPFN family uses structural causal models (SCMs) as an important part of this synthetic prior. Later models introduced richer priors including tree-based structures and more sophisticated directed acyclic graphs and mappings.
    - The purpose is to expose the Transformer to a very large variety of statistical relationships.

## Tarte

[Tarte](https://arxiv.org/abs/2505.14415)

    - Learns semantic representations of table entries by combining column names, strings and numerical values.
    - TARTE explicitly models the combination column-cell, rather than treating the cell value independently.
    - TARTE differs strongly from TabPFN in its source of pretraining knowledge. TabPFN is primarily pretrained on generated machine-learning tasks. TARTE instead performs what its authors call knowledge pre-training. The model is pretrained using large relational knowledge sources enriched with numerical attributes, including information derived from Wikidata.
    - Another important difference is the output of the foundation model. TabPFN is primarily used as a predictor. TARTE is designed as a reusable representation backbone.
    - TARTE can also be fine-tuned for a downstream task. The authors emphasize that its pretrained representations can be frozen, fine-tuned, or combined with other machine-learning models.

## TabCLv2

[TabICLv2](https://arxiv.org/abs/2602.11139)

    - Follows the ICL philosophy, but focuses on making this approach more efficient, scalable and robust to long contexts.
    - TabICLv2 introduces repeated feature grouping. Instead of always representing each feature independently, multiple feature combinations are formed using circular shifts. This creates overlapping feature groups while maintaining detailed feature information. The goal is partly to reduce representation collapse, where different features with similar distributions could otherwise become difficult for the model to distinguish.
    - TabICLv2 also injects information about the target (y_i) early in the representation of training examples. Instead of treating the label only as a separate token later in the network, a target embedding is added directly to feature representations for training samples.
    - One of the major problems for Transformer-based in-context learners is that attention behavior changes as the number of context examples grows. With standard softmax, as the number of possible keys increases, attention can become increasingly diffuse. This phenomenon is described in the TabICLv2 paper as attention fading.
    - TabICLv2 introduces Query-Aware Scalable Softmax (QASSMax) to improve this long-context behavior. The underlying idea is to adapt the scaling of attention based on context length and the query, allowing the model to preserve sharper attention even as the number of training examples grows. This improves length generalization.