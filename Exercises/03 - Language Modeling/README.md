# 03 - Language Modeling

This tutorial is about turning words into numbers and then measuring how well a model predicts language. You train your own **Word2Vec** embeddings, explore **pre-trained GloVe** vectors, use them as features for a sentiment classifier, and then build a **bigram language model** by hand to compute **perplexity**.

## Notebooks
- `0. Word Embeddings.ipynb` — the first exercise notebook. Work through it in order and fill in every `# TODO`.
- `0. Word Embeddings - Solutions.ipynb` — the worked solutions. Try to finish a part before looking at them.
- `1. Perplexity.ipynb` — the second exercise notebook.
- `1. Perplexity - Solution.ipynb` — its worked solutions.

## What you will practice

**Word Embeddings**
1. **Training Word2Vec** — preprocessing a corpus and training skip-gram embeddings on the Brown corpus
2. **Cosine similarity** — implementing it yourself and checking it against gensim
3. **Pre-trained GloVe** — nearest neighbours, analogies (`king - man + woman`), odd-one-out, and the biases these vectors pick up
4. **A downstream task** — averaging word vectors into sentence features and training a sentiment classifier on 50k tweets
5. **The limits of averaging** — why this representation cannot tell *"good, not bad"* from *"bad, not good"*
6. **MCQs** — to check your understanding

**Perplexity**
1. **Counting** — unigrams and bigrams, with sentence-boundary tokens
2. **Estimating probabilities** — the maximum-likelihood estimate and why it breaks
3. **Add-1 smoothing** — and verifying that the result is still a probability distribution
4. **Perplexity** — computing it, wrapping it in a function, and comparing sentences with it
5. **MCQs** — to check your understanding

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
Both notebooks need the updated project environment (`uv sync`), which now includes `datasets` and `tqdm`.

Two cells in `0. Word Embeddings.ipynb` download data, so make sure you have a working internet connection and start them early:
- the **Brown corpus** and NLTK stopwords (~3 MB),
- the **GloVe vectors** `glove-wiki-gigaword-100` (~130 MB, cached after the first run).

The Sentiment140 tweets are downloaded automatically as well. A GPU is not required — but training the classifier for 30 epochs takes around 10 minutes on a CPU.

`1. Perplexity.ipynb` needs no downloads and runs in a second.

Please give us feedback for this tutorial!

![LS Feedback 3](../QRs/qrf3.png)
