# Third-Party Notices

## LinkAlign (prompts and pipeline mechanism)

`retrievers/upstream/linkalign_prompts.py` is a pinned snapshot of the
LinkAlign upstream prompt set, used by
`retrievers/linkalign_official_retriever.py`. License:
`retrievers/upstream/LinkAlign_LICENSE` (MIT). The files are kept
byte-identical to the audited upstream version.

The public method `LinkAlign-SingleDB-Adapted` retains the upstream
enhancement → second-round retrieval → isolation → extraction mechanism
with **declared replacements** (multilingual MiniLM for BGE, DeepSeek for
Qwen, direct transformers for llama-index) in a single known database. It is
an adaptation, not a reproduction of the original checkpoints.

## KaSLA (relevance/weight functions and knapsack)

`retrievers/upstream/kasla_utils.py` is a pinned snapshot of the KaSLA
upstream utility functions, used by `retrievers/kasla_retriever.py`.
License: `retrievers/upstream/KaSLA_LICENSE` (MIT). Byte-identical to the
audited upstream version.

The public method `KaSLA-Adapted` retains the upstream relevance/weight
functions and hierarchical table/column knapsack DP with **declared
replacements** (DeepSeek with the upstream prompt prefix for the fine-tuned
binary LLM; per-fold L2 logistic classifiers for CodeS probabilities).

## Other dependencies

PyPI packages (pandas, numpy, networkx, jieba, scikit-learn, PyYAML,
sqlglot, matplotlib; optional scipy/torch/transformers/openai and the
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encoder) are
used under their respective licenses; nothing is redistributed inside this
repository.

## Root license

The root repository license is pending the owner's decision. Until a
LICENSE file is added, no reuse rights are granted for the original code
beyond inspection; the third-party MIT components above remain under their
own terms regardless.
