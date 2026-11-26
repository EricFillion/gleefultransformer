# Pretraining with Gleeful Transformer

The [Common Corpus example](chat/common_corpus/) initializes a causal language model from a configuration, packs raw documents into fixed-length token blocks, and trains with a target-token budget. It includes held-out evaluation, checkpoint resume, and a sampling script for the exported model.
