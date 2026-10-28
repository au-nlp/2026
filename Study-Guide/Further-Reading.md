# 📚 Beyond the Slides: Further Reading for NLP

Lectures give you the map. This guide is for when you want to explore the territory.

For every lecture so far, you'll find the papers, blog posts, videos and book chapters that go one step deeper than the slides. Some are the original papers behind the figures you saw in class. Others are beautifully illustrated explainers. A few are papers that argue with each other, which is often where things get interesting.

You don't need to read everything. Pick what you're curious about, what you found confusing in class, or what might help your project.

**Jump to lecture:** [1](#1--intro-to-nlp) · [2](#2--neural-networks-and-word-embeddings) · [3](#3--language-modeling-part-1) · [4](#4--language-modeling-part-2) · [5](#5--seq2seq-attention-and-transformers) · [6](#6--pre-trained-language-models) · [7](#7--in-context-learning-prompting-and-rlhf) · [8](#8--natural-language-generation) · [9](#9--evaluating-generation-question-answering-and-rag)

## 🧭 How to read this guide

| Icon        | Meaning                                                   |
| ----------- | --------------------------------------------------------- |
| ⭐          | **Start here.** The best first read for that topic        |
| 🎨          | Visual explainer or blog post, usually the easiest way in |
| 🎥          | Video                                                     |
| 📖          | Textbook chapter                                          |
| _(no icon)_ | Research paper                                            |

**Tip for reading papers:** you rarely need to read one front to back. Read the abstract, look at the figures, read the introduction and conclusion, and only then dive into the method if you want the details.

## 📖 Your bookshelf

These free books come up again and again below. Bookmark them.

- [Jurafsky & Martin, _Speech and Language Processing_ (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/). The NLP textbook. The draft is updated often, so chapters are referred to by title rather than number.
- [Eisenstein, _Introduction to Natural Language Processing_ (notes)](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf). More mathematical; great if you like derivations.
- [Goodfellow, Bengio & Courville, _Deep Learning_](https://www.deeplearningbook.org/). The deep learning reference.
- [_Dive into Deep Learning_](https://d2l.ai/). Every concept comes with runnable PyTorch code.
- 🎨 [Lena Voita, _NLP Course | For You_](https://lena-voita.github.io/nlp_course.html). Gorgeous illustrated lectures.

---

## 1 · Intro to NLP

_Before neural networks, how did computers make sense of text? Mostly by counting words, cleverly._

> **If you read one thing:** chapter 2 of the IR book below. It is the full story behind the tokenization-to-TF-IDF pipeline from class.

### Cleaning up text: encoding, tokenization, stopwords, stemming

- ⭐ 📖 [Manning, Raghavan & Schütze, _Introduction to Information Retrieval_](https://nlp.stanford.edu/IR-book/), ch. 2 "The term vocabulary and postings lists". Tokenization, stopwords, casefolding, stemming vs. lemmatization, with the same trade-offs we discussed.
- 📖 Jurafsky & Martin, the chapter on words, tokenization and text normalization.
- 🎨 [Spolsky, "The Absolute Minimum Every Software Developer Must Know About Unicode and Character Sets" (2003)](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/). A fun, practical read on why "always write UTF-8" will save your future self.

### Bag of words and TF-IDF

- ⭐ 📖 Manning et al., IR book, ch. 6 "Scoring, term weighting and the vector space model". Where tf-idf, cosine similarity and length normalization come from.
- [Turney & Pantel, "From Frequency to Meaning: Vector Space Models of Semantics" (JAIR 2010)](https://arxiv.org/abs/1003.1141). A tour of the different matrices you can build from text, and a bridge to word embeddings in Lecture 2.
- [Wang & Manning, "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification" (ACL 2012)](https://aclanthology.org/P12-2018/). Shows how hard simple n-gram baselines are to beat.
- [Joulin et al., "Bag of Tricks for Efficient Text Classification" (fastText, 2016)](https://arxiv.org/abs/1607.01759). The "sum the embeddings, then logistic regression" model from the end of the lecture, scaled up and surprisingly strong.

### Zipf's law

- Piantadosi, "Zipf's word frequency law in natural language: A critical review and future directions" (_Psychonomic Bulletin & Review_, 2014). Why do word frequencies follow a power law? Nobody fully knows, and this paper explains why it's so puzzling.

### The four classic tasks

- 📖 Manning et al., IR book, ch. 13 (text classification and Naive Bayes) and ch. 16–17 (clustering). Retrieval, classification and clustering-based topic detection.
- Pang & Lee, "Opinion Mining and Sentiment Analysis" (_Foundations and Trends in Information Retrieval_, 2008). The classic survey on sentiment analysis.
- [Blei, "Probabilistic Topic Models" (_Communications of the ACM_, 2012)](https://www.cs.columbia.edu/~blei/papers/Blei2012.pdf). A very readable introduction to LDA, a probabilistic take on topic detection.

### Setting up a neural NLP model

- ⭐ [Goldberg, "A Primer on Neural Network Models for Natural Language Processing" (JAIR 2016)](https://arxiv.org/abs/1510.00726). Uses the same embeddings → model → prediction framing as the lecture.

---

## 2 · Neural networks and word embeddings

_"You shall know a word by the company it keeps." How that one idea turned words into vectors._

> **If you read one thing:** the two word2vec papers, side by side with Goldberg & Levy's short derivation. Together they explain what the CBOW and skip-gram pictures don't, including the negative-sampling trick that makes training fast.

### Backpropagation and softmax

- ⭐ 🎨 [CS231n notes, "Backpropagation, Intuitions"](https://cs231n.github.io/optimization-2/). The chain rule on computational graphs, worked through by hand.
- 🎨 [Olah, "Calculus on Computational Graphs: Backpropagation" (2015)](https://colah.github.io/posts/2015-08-Backprop/). Why backprop is so efficient, in a few lovely diagrams.
- 📖 [Nielsen, _Neural Networks and Deep Learning_, ch. 2 "How the backpropagation algorithm works"](http://neuralnetworksanddeeplearning.com/chap2.html). A gentle, step-by-step derivation.
- 📖 Goodfellow et al., _Deep Learning_, ch. 6 "Deep Feedforward Networks". Includes softmax outputs and cross-entropy loss. Softmax will keep popping up, so it's worth getting comfortable with it.
- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" (_Nature_, 1986). The original paper, for the history buffs.

### word2vec: CBOW and skip-gram

- ⭐ [Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)](https://arxiv.org/abs/1301.3781). The CBOW and skip-gram paper.
- ⭐ [Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013)](https://arxiv.org/abs/1310.4546). Negative sampling and subsampling: how word2vec avoids a softmax over the whole vocabulary.
- [Goldberg & Levy, "word2vec Explained" (2014)](https://arxiv.org/abs/1402.3722). Four pages that make the negative-sampling objective click.
- [Rong, "word2vec Parameter Learning Explained" (2014)](https://arxiv.org/abs/1411.2738). Every gradient derived. Great practice after the backprop review.
- 🎨 [Alammar, "The Illustrated Word2vec"](https://jalammar.github.io/illustrated-word2vec/). The training loop, visualized.
- 📖 Jurafsky & Martin, "Vector Semantics and Embeddings". TF-IDF → PMI → word2vec in one arc, just like the lecture.
- Harris, "Distributional Structure" (_Word_, 1954). The distributional hypothesis, usually quoted alongside Firth (1957).

### Embeddings as matrix factorization

- Deerwester et al., "Indexing by Latent Semantic Analysis" (_JASIS_, 1990). The original LSA paper.
- Levy, Goldberg & Dagan, "Improving Distributional Similarity with Lessons Learned from Word Embeddings" (_TACL_, 2015). A plot twist: much of word2vec's advantage comes from hyperparameters that help older count-based methods too.

### Beyond one vector per word

- [Bojanowski et al., "Enriching Word Vectors with Subword Information" (fastText, 2017)](https://arxiv.org/abs/1607.04606). Builds word vectors from character n-grams, so rare and unseen words get good vectors too.
- [Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker?" (NeurIPS 2016)](https://arxiv.org/abs/1607.06520). Embeddings learn our biases too. A preview of the Responsible AI week.

### The classical NLP pipeline

- 📖 Jurafsky & Martin, the chapters on part-of-speech tagging and named entities, dependency parsing, constituency parsing, and coreference. One chapter per pipeline stage.
- [Collobert et al., "Natural Language Processing (almost) from Scratch" (JMLR 2011)](https://arxiv.org/abs/1103.0398). One neural network that learns many pipeline tasks at once.

---

## 3 · Language modeling, part 1

_How likely is this sentence? From counting n-grams to the first neural language models._

> **If you read one thing:** Karpathy's RNN blog post. It's where the Shakespeare and Wikipedia samples from class come from, and it's genuinely fun.

### n-gram language models

- ⭐ 📖 Jurafsky & Martin, "N-gram Language Models". Counting, the Markov assumption, perplexity and smoothing, with plenty of worked examples.
- Shannon, "Prediction and Entropy of Printed English" (_Bell System Technical Journal_, 1951). In 1951 Shannon had people guess the next letter of a text. Language modeling started here.

### Perplexity

- [Hugging Face docs, "Perplexity of fixed-length models"](https://huggingface.co/docs/transformers/perplexity). How perplexity is actually computed for real models with a context limit.

### Smoothing

- Chen & Goodman, "An empirical study of smoothing techniques for language modeling" (_Computer Speech & Language_, 1999). The definitive comparison, and the clearest explanation of Kneser-Ney, the method that won.

### Fixed-context neural LMs

- ⭐ [Bengio et al., "A Neural Probabilistic Language Model" (JMLR 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). The original model, with the same W and U matrices you saw in class.
- 🎨 [Voita, NLP Course, "Language Modeling"](https://lena-voita.github.io/nlp_course.html). From n-grams to neural LMs, with great intuition for perplexity.

### Recurrent neural networks

- ⭐ 🎨 [Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" (2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). A tiny character-level RNN that writes Shakespeare, Wikipedia and even LaTeX.
- Elman, "Finding Structure in Time" (_Cognitive Science_, 1990). The Elman network itself.
- Mikolov et al., "Recurrent neural network based language model" (Interspeech 2010). The moment RNNs started beating n-grams.
- 📖 Goodfellow et al., _Deep Learning_, ch. 10 "Sequence Modeling: Recurrent and Recursive Nets". The formal treatment of unrolling and weight sharing.

---

## 4 · Language modeling, part 2

_Why plain RNNs forget, and how a few clever gates fixed it._

> **If you read one thing:** Olah's "Understanding LSTM Networks". It's probably the most-loved explainer in all of deep learning, for good reason.

### Backpropagation through time

- Werbos, "Backpropagation through time: what it does and how to do it" (_Proceedings of the IEEE_, 1990). The original explanation of BPTT.
- [Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (JMLR 2018)](https://arxiv.org/abs/1502.05767). What PyTorch does for you when "you only define the forward pass".

### Vanishing and exploding gradients

- ⭐ [Pascanu, Mikolov & Bengio, "On the difficulty of training Recurrent Neural Networks" (ICML 2013)](https://arxiv.org/abs/1211.5063). The math behind vanishing gradients, plus gradient clipping for the exploding case.
- Bengio, Simard & Frasconi, "Learning long-term dependencies with gradient descent is difficult" (_IEEE Trans. Neural Networks_, 1994). Where the problem was first described.

### LSTMs and GRUs

- ⭐ 🎨 [Olah, "Understanding LSTM Networks" (2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Forget, input and output gates, one diagram at a time.
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (_Neural Computation_, 1997), and Gers, Schmidhuber & Cummins, "Learning to Forget" (2000). The original LSTM, and the paper that added the forget gate.
- [Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014)](https://arxiv.org/abs/1406.1078). Introduces the GRU, and sneak-previews Lecture 5's encoder-decoder.
- [Karpathy, Johnson & Fei-Fei, "Visualizing and Understanding Recurrent Networks" (2015)](https://arxiv.org/abs/1506.02078). The cells that track quotes and indentation, straight from the lecture.

### LSTM or GRU? Or something else?

- [Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)](https://arxiv.org/abs/1412.3555). LSTM vs. GRU, head to head.
- [Greff et al., "LSTM: A Search Space Odyssey" (2017)](https://arxiv.org/abs/1503.04069). What happens when you remove each gate? The forget gate turns out to matter most.
- Jozefowicz, Zaremba & Sutskever, "An Empirical Exploration of Recurrent Network Architectures" (ICML 2015). They searched about 10,000 RNN variants looking for something better.
- [Weiss, Goldberg & Yahav, "On the Practical Computational Power of Finite Precision RNNs for Language Recognition" (ACL 2018)](https://arxiv.org/abs/1805.04908). Where "LSTMs can count, GRUs can't" comes from.

### Deep and bidirectional RNNs

- Schuster & Paliwal, "Bidirectional Recurrent Neural Networks" (_IEEE Trans. Signal Processing_, 1997). The original bidirectional RNN.
- [Graves, "Generating Sequences With Recurrent Neural Networks" (2013)](https://arxiv.org/abs/1308.0850). Deep stacked LSTMs that generate text, and even handwriting.

---

## 5 · Seq2seq, attention and Transformers

_From squeezing a whole sentence into one vector to "Attention Is All You Need"._

> **If you read one thing:** the original Transformer paper. After this lecture you're ready for it, and it's shorter than you'd think. Then build one with Karpathy's video.

You already have _The Annotated Transformer_ and _The Illustrated Transformer_ from the lecture. Here's what comes next.

### Encoder-decoder models

- ⭐ [Sutskever, Vinyals & Le, "Sequence to Sequence Learning with Neural Networks" (2014)](https://arxiv.org/abs/1409.3215). The seq2seq paper, including the odd trick of reversing the input sentence.
- 🎨 [Alammar, "Visualizing A Neural Machine Translation Model"](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). Animated seq2seq, with and without attention.

### Attention

- ⭐ [Bahdanau, Cho & Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)](https://arxiv.org/abs/1409.0473). The paper that introduced attention to fix the bottleneck.
- [Luong, Pham & Manning, "Effective Approaches to Attention-based Neural Machine Translation" (2015)](https://arxiv.org/abs/1508.04025). Dot-product, bilinear and other scoring functions compared.
- 🎨 [Weng, "Attention? Attention!" (2018)](https://lilianweng.github.io/posts/2018-06-24-attention/). A tour of attention variants in one blog post.
- [Jain & Wallace, "Attention is not Explanation" (2019)](https://arxiv.org/abs/1902.10186) and [Wiegreffe & Pinter, "Attention is not not Explanation" (2019)](https://arxiv.org/abs/1908.04626). Can attention weights tell us why a model decided something? Read these two as a debate.

### The Transformer

- ⭐ [Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)](https://arxiv.org/abs/1706.03762). The paper that started it all.
- 🎨 [The Annotated Transformer, 2022 update](https://nlp.seas.harvard.edu/annotated-transformer/). A newer version of the one from class, with refreshed, runnable PyTorch code.
- ⭐ 🎥 [Karpathy, "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY). Builds masked multi-head self-attention, residuals and LayerNorm live, in about two hours. Highly recommended.
- [Phuong & Hutter, "Formal Algorithms for Transformers" (2022)](https://arxiv.org/abs/2207.09238). Precise pseudocode for every component, handy when revising.

### Residual connections and layer normalization

- [He et al., "Deep Residual Learning for Image Recognition" (2015)](https://arxiv.org/abs/1512.03385). Residual connections were invented for images first.
- [Ba, Kiros & Hinton, "Layer Normalization" (2016)](https://arxiv.org/abs/1607.06450). The LayerNorm paper.
- [Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020)](https://arxiv.org/abs/2002.04745). Where you put LayerNorm matters, and why modern models put it first.

### Position embeddings

- [Shaw, Uszkoreit & Vaswani, "Self-Attention with Relative Position Representations" (2018)](https://arxiv.org/abs/1803.02155). Relative instead of absolute positions.
- [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)](https://arxiv.org/abs/2104.09864). RoPE, the scheme most of today's LLMs use.
- [Press, Smith & Lewis, "Train Short, Test Long" (ALiBi, 2021)](https://arxiv.org/abs/2108.12409). Tackles exactly the problem from class: learned positions don't extend to longer inputs. Useful for the position-embedding exercise.

---

## 6 · Pre-trained language models

_Pretrain once, fine-tune everywhere: GPT, ELMo, BERT, BART and T5._

> **If you read one thing:** the BERT paper, with Alammar's "Illustrated BERT" open next to it.

### GPT and the pretrain-then-fine-tune idea

- ⭐ [Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- [Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Less fine-tuning, more zero-shot.
- [Howard & Ruder, "Universal Language Model Fine-tuning for Text Classification" (ULMFiT, 2018)](https://arxiv.org/abs/1801.06146). Showed "pretrain, then fine-tune" working with LSTMs, just before Transformers took over.
- 🎨 [Alammar, "The Illustrated GPT-2"](https://jalammar.github.io/illustrated-gpt2/). Decoder-only Transformers and masked self-attention, visualized.

### ELMo and BERT

- ⭐ [Peters et al., "Deep contextualized word representations" (ELMo, 2018)](https://arxiv.org/abs/1802.05365).
- ⭐ [Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805). Masked language modeling, the [CLS] token, and the ablations from class.
- 🎨 [Alammar, "The Illustrated BERT, ELMo, and co."](https://jalammar.github.io/illustrated-bert/). The whole lecture in pictures.
- [Peters, Ruder & Smith, "To Tune or Not to Tune?" (2019)](https://arxiv.org/abs/1903.05987). When should you fine-tune, and when should you just use the embeddings?

### Benchmarks

- [Wang et al., "GLUE" (2018)](https://arxiv.org/abs/1804.07461) and [Wang et al., "SuperGLUE" (2019)](https://arxiv.org/abs/1905.00537). What's in GLUE, and why a harder version was needed so quickly.

### What does BERT actually learn?

- ⭐ [Clark et al., "What Does BERT Look At? An Analysis of BERT's Attention" (2019)](https://arxiv.org/abs/1906.04341). The attention-head figures from class.
- [Tenney, Das & Pavlick, "BERT Rediscovers the Classical NLP Pipeline" (2019)](https://arxiv.org/abs/1905.05950). BERT's layers roughly follow the POS → parsing → NER → coreference pipeline from Lecture 2. A lovely connection.
- [Rogers, Kovaleva & Rumshisky, "A Primer in BERTology" (2020)](https://arxiv.org/abs/2002.12327). Everything people discovered by poking at BERT, in one survey.

### BERT's relatives

- [Liu et al., "RoBERTa" (2019)](https://arxiv.org/abs/1907.11692), [Sanh et al., "DistilBERT" (2019)](https://arxiv.org/abs/1910.01108), and [Clark et al., "ELECTRA" (2020)](https://arxiv.org/abs/2003.10555). The three variants from class.
- [Joshi et al., "SpanBERT" (2019)](https://arxiv.org/abs/1907.10529). Masking whole spans instead of single tokens, a follow-up to "what should we mask?".

### Encoder-decoder pre-training

- ⭐ [Lewis et al., "BART" (2019)](https://arxiv.org/abs/1910.13461). All the ways to corrupt text before asking a model to repair it.
- ⭐ [Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2019)](https://arxiv.org/abs/1910.10683). Long, but sections 3.1–3.3 are a masterclass in running careful experiments. Worth a look for your project.

### Tokenization: how text becomes tokens

We used subwords all lecture without explaining them. These fill the gap.

- ⭐ [Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2015)](https://arxiv.org/abs/1508.07909). Byte-pair encoding, used by GPT models.
- [Wu et al., "Google's Neural Machine Translation System" (2016)](https://arxiv.org/abs/1609.08144). Describes WordPiece, BERT's tokenizer.
- [Kudo, "Subword Regularization" (2018)](https://arxiv.org/abs/1804.10959) and [Kudo & Richardson, "SentencePiece" (2018)](https://arxiv.org/abs/1808.06226). The unigram tokenizer used by T5.
- 🎥 [Karpathy, "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE). Builds BPE from scratch, and explains why LLMs struggle to spell or count letters.

### Making attention cheaper

Attention costs grow quadratically with sequence length. Here's how people fight that.

- [Tay et al., "Efficient Transformers: A Survey" (2020)](https://arxiv.org/abs/2009.06732). A map of the many "faster attention" ideas.
- [Beltagy et al., "Longformer" (2020)](https://arxiv.org/abs/2004.05150) and [Zaheer et al., "Big Bird" (2020)](https://arxiv.org/abs/2007.14062). Sparse attention; Big Bird is the paper behind the Google blog post from class.
- [Dao et al., "FlashAttention" (2022)](https://arxiv.org/abs/2205.14135). Keeps exact attention but makes it fast with clever memory use. This is the one that won in practice.

---

## 7 · In-context learning, prompting and RLHF

_How GPT-3 became ChatGPT._

> **If you read one thing:** the InstructGPT paper. The three-step "SFT → reward model → RL" figure from class comes from it, and it's very readable.

### Scaling

- ⭐ [Kaplan et al., "Scaling Laws for Neural Language Models" (2020)](https://arxiv.org/abs/2001.08361).
- [Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556). A twist on Kaplan: grow the data as fast as the model. More on this in Week 14.
- [Sevilla et al., "Compute Trends Across Three Eras of Machine Learning" (2022)](https://arxiv.org/abs/2202.05924) and [Cottier et al., "The rising costs of training frontier AI models" (2024)](https://arxiv.org/abs/2405.21015). The data behind the training-cost slide. [Epoch AI's trends page](https://epoch.ai/trends) keeps the numbers up to date.

### Emergence: real or a mirage?

The lecture flagged this as controversial. Read both sides and make up your own mind.

- Anderson, "More Is Different" (_Science_, 1972). The essay behind the quote from class.
- ⭐ [Wei et al., "Emergent Abilities of Large Language Models" (2022)](https://arxiv.org/abs/2206.07682). The case for emergence.
- ⭐ [Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?" (NeurIPS 2023)](https://arxiv.org/abs/2304.15004). The case against: change the metric and the sudden jump disappears.

### In-context learning

- ⭐ [Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)](https://arxiv.org/abs/2005.14165). The TriviaQA and translation results from class.
- [Zhao et al., "Calibrate Before Use" (2021)](https://arxiv.org/abs/2102.09690) and [Lu et al., "Fantastically Ordered Prompts and Where to Find Them" (2021)](https://arxiv.org/abs/2104.08786). Why simply reordering your examples can change accuracy dramatically.
- [Min et al., "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" (2022)](https://arxiv.org/abs/2202.12837). Surprise: the labels in your examples matter much less than you'd expect.
- [Xie et al., "An Explanation of In-context Learning as Implicit Bayesian Inference" (2021)](https://arxiv.org/abs/2111.02080) and [Olsson et al., "In-context Learning and Induction Heads" (2022)](https://arxiv.org/abs/2209.11895). Two answers to "why does in-context learning work at all?": one theoretical, one found by looking inside the model.

### Prompting

- [Liu et al., "Pre-train, Prompt, and Predict" (2021)](https://arxiv.org/abs/2107.13586). The big survey of prompting methods.
- [Shin et al., "AutoPrompt" (2020)](https://arxiv.org/abs/2010.15980). Let gradients search for the best prompt, and it comes back looking like gibberish.
- ⭐ [Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)](https://arxiv.org/abs/2201.11903).
- [Kojima et al., "Large Language Models are Zero-Shot Reasoners" (2022)](https://arxiv.org/abs/2205.11916) and [Wang et al., "Self-Consistency" (2022)](https://arxiv.org/abs/2203.11171). "Let's think step by step", and sampling several reasoning chains then taking a vote. A taste of the Reasoning week.

### Soft prompts and efficient fine-tuning

- ⭐ [Lester, Al-Rfou & Constant, "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)](https://arxiv.org/abs/2104.08691). The paper behind the soft-prompt slides.
- [Li & Liang, "Prefix-Tuning" (2021)](https://arxiv.org/abs/2101.00190). Soft prompts at every layer.
- [Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (adapters, 2019)](https://arxiv.org/abs/1902.00751), [Hu et al., "LoRA" (2021)](https://arxiv.org/abs/2106.09685), and [Dettmers et al., "QLoRA" (2023)](https://arxiv.org/abs/2305.14314). Fine-tune huge models by training only a tiny fraction of the weights. If you fine-tune anything for your project, you'll probably use LoRA.

### Instruction tuning

- [Wei et al., "Finetuned Language Models Are Zero-Shot Learners" (FLAN, 2021)](https://arxiv.org/abs/2109.01652), [Sanh et al., "Multitask Prompted Training Enables Zero-Shot Task Generalization" (T0, 2021)](https://arxiv.org/abs/2110.08207), and [Chung et al., "Scaling Instruction-Finetuned Language Models" (Flan-T5, 2022)](https://arxiv.org/abs/2210.11416). Teaching models to follow instructions by training on many tasks.
- [Zhou et al., "LIMA: Less Is More for Alignment" (2023)](https://arxiv.org/abs/2305.11206). About 1,000 carefully chosen examples go a surprisingly long way.

### RLHF: reward models, REINFORCE and PPO

- ⭐ [Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155).
- [Christiano et al., "Deep reinforcement learning from human preferences" (2017)](https://arxiv.org/abs/1706.03741), [Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019)](https://arxiv.org/abs/1909.08593), and [Stiennon et al., "Learning to summarize from human feedback" (2020)](https://arxiv.org/abs/2009.01325). The road from robots to ChatGPT.
- 🎨 [Lambert et al., "Illustrating RLHF" (Hugging Face blog)](https://huggingface.co/blog/rlhf), then 📖 [Lambert, _Reinforcement Learning from Human Feedback_ (online book)](https://rlhfbook.com/). An accessible overview first, then the full reference.
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (_Machine Learning_, 1992). The original REINFORCE.
- [Schulman et al., "Proximal Policy Optimization Algorithms" (2017)](https://arxiv.org/abs/1707.06347). The clipped objective behind "How does PPO differ?".
- [Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (2024)](https://arxiv.org/abs/2402.14740). Plot twist: good old REINFORCE does as well as PPO for LLMs.
- [Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization" (2022)](https://arxiv.org/abs/2210.10760). What happens when a model learns to game its reward, measured carefully.

### Beyond PPO: DPO and GRPO

- ⭐ [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)](https://arxiv.org/abs/2305.18290).
- [Shao et al., "DeepSeekMath" (2024)](https://arxiv.org/abs/2402.03300) and [DeepSeek-AI, "DeepSeek-R1" (2025)](https://arxiv.org/abs/2501.12948). GRPO, and how it was used to train a reasoning model.
- [Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)](https://arxiv.org/abs/2212.08073). What if the feedback comes from an AI instead of humans?

---

## 8 · Natural language generation

_Your model gives you a probability for every next word. Now what?_

> **If you read one thing:** the Hugging Face "How to generate text" blog post. It shows every decoding method from class with code you can run in a few minutes.

### Decoding at a glance

- ⭐ 🎨 [von Platen, "How to generate text: using different decoding methods for language generation with Transformers" (Hugging Face blog)](https://huggingface.co/blog/how-to-generate). Greedy, beam, top-k, top-p and temperature, side by side.
- [Wiher, Meister & Cotterell, "On Decoding Strategies for Neural Text Generators" (TACL 2022)](https://arxiv.org/abs/2203.15721). The best decoder depends on the task: what works for translation doesn't work for stories.

### When the most likely text is the wrong text

- [Koehn & Knowles, "Six Challenges for Neural Machine Translation" (2017)](https://arxiv.org/abs/1706.03872). Bigger beams can make translations worse.
- [Stahlberg & Byrne, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?" (2019)](https://arxiv.org/abs/1908.10090). Search perfectly for the most likely translation, and you often get… the empty string.
- ⭐ [Meister, Cotterell & Vieira, "If beam search is the answer, what was the question?" (2020)](https://arxiv.org/abs/2010.02650). Beam search works for a reason nobody planned.
- [Vijayakumar et al., "Diverse Beam Search" (2016)](https://arxiv.org/abs/1610.02424). Beam search that doesn't return ten near-identical outputs.

### Repetition

- [Welleck et al., "Neural Text Generation with Unlikelihood Training" (2019)](https://arxiv.org/abs/1908.04319). Fixing repetition during training instead of decoding.

### Sampling beyond top-k and top-p

- [Hewitt, Manning & Liang, "Truncation Sampling as Language Model Desmoothing" (2022)](https://arxiv.org/abs/2210.15191). Why cutting off the tail of the distribution helps at all.
- [Meister et al., "Locally Typical Sampling" (2022)](https://arxiv.org/abs/2202.00666). Pick words that are about as surprising as typical human text.
- [Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs" (2024)](https://arxiv.org/abs/2407.01082). Min-p, a newer method you'll find in many open-source LLM tools.

### Re-balancing and re-ranking

- [Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization" (2022)](https://arxiv.org/abs/2210.15097). Use a small "amateur" model to show the big model what to avoid.
- [Su et al., "A Contrastive Framework for Neural Text Generation" (contrastive search, 2022)](https://arxiv.org/abs/2202.06417). Penalize words that make the text too repetitive.

### "Shouldn't our models just be better calibrated?"

- [Ott et al., "Analyzing Uncertainty in Neural Machine Translation" (2018)](https://arxiv.org/abs/1803.00047). Models spread probability over many low-quality outputs, and that interacts badly with beam search.

### Teacher forcing and exposure bias

- Williams & Zipser, "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks" (_Neural Computation_, 1989). Where teacher forcing comes from.
- [Lamb et al., "Professor Forcing" (2016)](https://arxiv.org/abs/1610.09038). An adversarial alternative to scheduled sampling, with a great name.
- [Schmidt, "Generalization in Generation: A closer look at Exposure Bias" (2019)](https://arxiv.org/abs/1910.00292). Is exposure bias really the problem? A critical second opinion.

---

## 9 · Evaluating generation, question answering and RAG

_How do you grade a model that writes? And how do you stop it from making things up?_

> **If you read one thing:** "How NOT To Evaluate Your Dialogue System". It makes the problems with BLEU-style metrics impossible to forget. For RAG, read "Lost in the Middle".

### Evaluating generated text: the big picture

- ⭐ [Celikyilmaz, Clark & Gao, "Evaluation of Text Generation: A Survey" (2020)](https://arxiv.org/abs/2006.14799). Human, overlap-based and model-based metrics, organized just like the lecture.
- [Gehrmann, Clark & Sellam, "Repairing the Cracked Foundation" (2022)](https://arxiv.org/abs/2202.06935). Common evaluation mistakes in NLG papers. A handy checklist for your project report.

### Overlap metrics: BLEU, ROUGE and friends

- [Papineni et al., "BLEU" (ACL 2002)](https://aclanthology.org/P02-1040/), [Lin, "ROUGE" (2004)](https://aclanthology.org/W04-1013/), [Banerjee & Lavie, "METEOR" (2005)](https://aclanthology.org/W05-0909/), and [Popović, "chrF" (2015)](https://aclanthology.org/W15-3049/). The original papers.
- [Post, "A Call for Clarity in Reporting BLEU Scores" (2018)](https://arxiv.org/abs/1804.08771). Why two papers' BLEU scores often can't be compared, and how sacreBLEU fixes it.
- ⭐ [Liu et al., "How NOT To Evaluate Your Dialogue System" (2016)](https://arxiv.org/abs/1603.08023). Overlap metrics barely agree with humans on dialogue, exactly like the example from class.
- [Reiter, "A Structured Review of the Validity of BLEU" (_Computational Linguistics_, 2018)](https://aclanthology.org/J18-3002/). When does BLEU actually track human judgment?

### Model-based metrics

- [Kusner et al., "From Word Embeddings To Document Distances" (Word Mover's Distance, ICML 2015)](https://proceedings.mlr.press/v37/kusnerb15.html). Comparing texts by how far their words have to "move".
- ⭐ [Zhang et al., "BERTScore" (2019)](https://arxiv.org/abs/1904.09675) and [Zhao et al., "MoverScore" (2019)](https://arxiv.org/abs/1909.02622). Matching meaning with contextual embeddings instead of exact words.
- [Sellam et al., "BLEURT" (2020)](https://arxiv.org/abs/2004.04696), [Rei et al., "COMET" (2020)](https://arxiv.org/abs/2009.09025), and [Yuan et al., "BARTScore" (2021)](https://arxiv.org/abs/2106.11520). Metrics that are trained themselves; COMET is now standard in machine translation.

### LLMs as judges

- ⭐ [Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)](https://arxiv.org/abs/2306.05685). LLM judges mostly agree with humans, but they prefer longer answers, the first answer they see, and their own writing.
- [Liu et al., "G-Eval" (2023)](https://arxiv.org/abs/2303.16634) and [Wang et al., "Large Language Models are not Fair Evaluators" (2023)](https://arxiv.org/abs/2305.17926). A method, and then its failure modes.

### Human evaluation

- [van der Lee et al., "Best practices for the human evaluation of automatically generated text" (INLG 2019)](https://aclanthology.org/W19-8643/). Practical advice if you run a human study.
- [Clark et al., "All That's 'Human' Is Not Gold" (2021)](https://arxiv.org/abs/2107.00061) and [Karpinska, Akoury & Iyyer, "The Perils of Using Mechanical Turk to Evaluate Open-Ended Text Generation" (2021)](https://arxiv.org/abs/2109.06835). Even human evaluation has pitfalls.

### Faithfulness and hallucination

- [Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization" (2020)](https://arxiv.org/abs/2005.00661), [Min et al., "FActScore" (2023)](https://arxiv.org/abs/2305.14251), and [Ji et al., "Survey of Hallucination in Natural Language Generation" (2022)](https://arxiv.org/abs/2202.03629). The "making things up" problem, and why retrieval helps.

### Question answering

- 📖 Jurafsky & Martin, the chapter on information retrieval and question answering (recent drafts also cover RAG). The textbook version of the second half of this lecture.
- [Chen & Yih, "Open-Domain Question Answering" (ACL 2020 tutorial)](https://aclanthology.org/2020.acl-tutorials.8/). From retriever-reader pipelines to dense retrieval.
- Ferrucci et al., "Building Watson: An Overview of the DeepQA Project" (_AI Magazine_, 2010). How the system that won _Jeopardy!_ was built.
- ⭐ [Rajpurkar et al., "SQuAD" (2016)](https://arxiv.org/abs/1606.05250) and [Rajpurkar, Jia & Liang, "Know What You Don't Know" (SQuAD 2.0, 2018)](https://arxiv.org/abs/1806.03822).
- [Seo et al., "Bidirectional Attention Flow for Machine Comprehension" (BiDAF, 2016)](https://arxiv.org/abs/1611.01603). The classic reader that predicts where the answer starts and ends.
- ⭐ [Jia & Liang, "Adversarial Examples for Evaluating Reading Comprehension Systems" (2017)](https://arxiv.org/abs/1707.07328). Add one distracting sentence and "superhuman" QA systems fall apart. The evidence behind "Is reading comprehension solved?".
- [Joshi et al., "TriviaQA" (2017)](https://arxiv.org/abs/1705.03551), [Yang et al., "HotpotQA" (2018)](https://arxiv.org/abs/1809.09600), and [Kwiatkowski et al., "Natural Questions" (TACL 2019)](https://aclanthology.org/Q19-1026/). The other big QA datasets you'll meet.

### Long-form answers, and how to grade them

- [Fan et al., "ELI5: Long Form Question Answering" (2019)](https://arxiv.org/abs/1907.09190). "Explain like I'm five" as a dataset.
- ⭐ [Krishna, Roy & Iyyer, "Hurdles to Progress in Long-form Question Answering" (2021)](https://arxiv.org/abs/2103.06332). ROUGE barely tells good long answers from bad ones.
- [Xu et al., "A Critical Evaluation of Evaluations for Long-form Question Answering" (2023)](https://arxiv.org/abs/2305.18201) and [Kamalloo et al., "Evaluating Open-Domain Question Answering in the Era of Large Language Models" (2023)](https://arxiv.org/abs/2305.06984). Exact match marks plenty of correct LLM answers as wrong.

### What a model knows without looking anything up

- [Petroni et al., "Language Models as Knowledge Bases?" (2019)](https://arxiv.org/abs/1909.01066) and [Roberts, Raffel & Shazeer, "How Much Knowledge Can You Pack Into the Parameters of a Language Model?" (2020)](https://arxiv.org/abs/2002.08910). The "closed book" side of closed-book vs. retrieval.

### Retrieval

- ⭐ Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (_Foundations and Trends in IR_, 2009). BM25 is TF-IDF from Lecture 1, grown up, and it's still the baseline every dense retriever has to beat.
- [Chen et al., "Reading Wikipedia to Answer Open-Domain Questions" (DrQA, 2017)](https://arxiv.org/abs/1704.00051). The retrieve-then-read setup behind "finding the answer in 21M documents".
- [Khattab & Zaharia, "ColBERT" (2020)](https://arxiv.org/abs/2004.12832). A clever middle ground between fast and accurate retrieval.
- [Thakur et al., "BEIR" (2021)](https://arxiv.org/abs/2104.08663). On new domains, fancy dense retrievers often lose to BM25.
- [Johnson, Douze & Jégou, "Billion-scale similarity search with GPUs" (FAISS, 2017)](https://arxiv.org/abs/1702.08734) and [Malkov & Yashunin, "HNSW" (2016)](https://arxiv.org/abs/1603.09320). How you search a billion vectors in milliseconds.
- [Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, 2022)](https://arxiv.org/abs/2212.10496). Have the LLM write a fake answer first, then search for documents like it.

### Retrieval-augmented language models

You've seen DPR, REALM and RAG in class. Here's where the field went next.

- ⭐ [Asai et al., "Retrieval-based Language Models and Applications" (ACL 2023 tutorial)](https://acl2023-retrieval-lm.github.io/). Organizes the whole area around what, how and when to retrieve.
- [Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023)](https://arxiv.org/abs/2312.10997). From simple to modular RAG systems.
- [Izacard & Grave, "Fusion-in-Decoder" (2020)](https://arxiv.org/abs/2007.01282) and [Borgeaud et al., "RETRO" (2021)](https://arxiv.org/abs/2112.04426). Two more ways to wire retrieval into a model.
- [Ram et al., "In-Context Retrieval-Augmented Language Models" (2023)](https://arxiv.org/abs/2302.00083) and [Shi et al., "REPLUG" (2023)](https://arxiv.org/abs/2301.12652). Just paste the retrieved text into the prompt of a frozen LLM. This is how most RAG systems are built today.
- [Asai et al., "Self-RAG" (2023)](https://arxiv.org/abs/2310.11511). The model decides when it needs to look something up, and checks its own answers.
- ⭐ [Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023)](https://arxiv.org/abs/2307.03172). LLMs pay the most attention to the start and end of the context. Where you put retrieved passages matters.
- [Es et al., "Ragas: Automated Evaluation of Retrieval Augmented Generation" (2023)](https://arxiv.org/abs/2309.15217). Bringing the two halves of this lecture together: how to evaluate a RAG system.
- [Zhang et al., "ERNIE: Enhanced Language Representation with Informative Entities" (2019)](https://arxiv.org/abs/1905.07129). The knowledge-graph ERNIE from class (not to be confused with Baidu's ERNIE, a different model).

---

_Found a great resource that isn't here, or a broken link? Tell us on the course forum._ 🙌
