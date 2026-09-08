# 03 - Language Modeling

This tutorial moves from counting words to representing them. First you learn dense word representations — training your own Word2Vec embeddings and reusing pre-trained GloVe vectors to build a sentiment classifier — and then you implement a classic bigram language model and score it with perplexity.

## Notebooks
- `0. Word Embeddings.ipynb` — train Word2Vec on the Brown Corpus, then use GloVe embeddings as features for a tweet sentiment classifier. Fill in every `# TODO` and the `...` placeholders.
- `1. Perplexity.ipynb` — build a bigram model with add-1 smoothing on a toy corpus and compute the perplexity of a test sentence by hand.

## What you will practice
1. **Corpus preprocessing** — tokenizing, lowercasing, deaccenting and removing stopwords with NLTK and gensim
2. **Word2Vec** — training skip-gram embeddings yourself, inspecting a word vector and its nearest neighbours
3. **Pre-trained embeddings** — loading GloVe and turning a sentence into a fixed-length vector by averaging its word vectors
4. **A sentiment classifier** — an MLP in PyTorch over sentence embeddings, with the training loop, BCE loss and test-set evaluation from week 2
5. **N-gram language models** — unigram and bigram counts with `<s>` / `</s>` boundary tokens
6. **Add-1 (Laplace) smoothing** — why unseen bigrams need it, and the smoothed probability formula
7. **Perplexity** — the log-probability formulation and what a lower value actually tells you
8. **MCQs** — to check your understanding

## Before you start
Both notebooks run on CPU; no GPU is required.

The notebooks need `gensim`, `nltk`, `torch` (all part of the project environment, `uv sync`) plus `datasets` and `torchmetrics` for the sentiment classifier:

```bash
uv add datasets torchmetrics
```

Several downloads happen the first time you run the notebooks, so make sure you have a working internet connection and some patience:
- the NLTK `brown` corpus and `stopwords` list (small),
- the `glove-wiki-gigaword-100` vectors (~130 MB),
- the Sentiment140 Twitter dataset from the Hugging Face Hub.

Please give us feedback for this tutorial!

![LS Feedback 3](../QRs/qrf3.png)
