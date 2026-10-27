# NLP Course — Additional Readings (Lectures 1–9)

*Last updated: Oct 27, 2026*

## How to use this list

Each section maps readings to the slide topics of one lecture, and adds only what the decks do not already link. Items marked ⭐ are the ones to read first; everything else is for students who want depth.

- **Excluded because the slides already link them:** GloVe, Levy & Goldberg (2014), Collobert & Weston (2008), sent2vec, Cr5, Paskov et al. (2013), langid.py, the Porter stemmer page, TweetNLP, The Annotated Transformer (2018), The Illustrated Transformer, the Google sparse-attention blog, and the reference lists at the ends of Lectures 8 and 9 (Holtzman et al., Fan et al., DPR, REALM, RAG, Toolformer, ReAct, and others).
- **Textbooks used throughout:** [Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) (chapters are cited by title because the draft renumbers them), [Eisenstein, *Introduction to Natural Language Processing* (notes)](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf), [Goodfellow, Bengio & Courville, *Deep Learning*](https://www.deeplearningbook.org/), [*Dive into Deep Learning*](https://d2l.ai/) (has runnable PyTorch code), and [Lena Voita's NLP Course | For You](https://lena-voita.github.io/nlp_course.html) (illustrated lectures that closely match Lectures 2–5).
- **Verification:** every arXiv ID below was checked against the arXiv API, and every other link was opened, on the date at the top of this page. The one exception is the Stanford IR-book site, whose server refused connections at check time.

## Lecture 1 — Intro to NLP

The best companion to the classical half of this lecture is Manning, Raghavan & Schütze's IR book, which covers the full preprocessing-to-TF-IDF pipeline on the slides.

### Text preprocessing (encoding, tokenization, stopwords, stemming, lemmatization)

- ⭐ [Manning, Raghavan & Schütze, *Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), ch. 2 “The term vocabulary and postings lists”. Tokenization, stopwords, casefolding, stemming vs. lemmatization, with the same trade-offs the slides list.
- Jurafsky & Martin (SLP3), the chapter on words, tokenization and text normalization. Covers word vs. subword tokenization and text normalization.
- [Spolsky, “The Absolute Minimum Every Software Developer Must Know About Unicode and Character Sets” (2003)](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/). A short, practical explanation of the “always write UTF-8” advice.

### Bag of words, TF-IDF, normalization

- ⭐ Manning et al., IR book, ch. 6 “Scoring, term weighting and the vector space model”. Derives tf-idf, cosine similarity, and length normalization (the row-normalization slides).
- [Turney & Pantel, “From Frequency to Meaning: Vector Space Models of Semantics” (JAIR 2010)](https://arxiv.org/abs/1003.1141). Survey of term-document, word-context and pair-pattern matrices; bridges to Lecture 2.
- [Wang & Manning, “Baselines and Bigrams: Simple, Good Sentiment and Topic Classification” (ACL 2012)](https://aclanthology.org/P12-2018/). Shows how strong n-gram bag-of-words baselines are, which supports the “n-grams can beat neural networks” slide.
- [Joulin et al., “Bag of Tricks for Efficient Text Classification” (fastText, 2016)](https://arxiv.org/abs/1607.01759). A bag of n-gram embeddings with a linear classifier: the “sum-pool + logistic regression” model from the end of the lecture, at scale.

### Zipf’s law and sparsity

- Piantadosi, “Zipf’s word frequency law in natural language: A critical review and future directions” (*Psychonomic Bulletin & Review*, 2014). What Zipf’s law does and does not explain.

### The four classical tasks

- Manning et al., IR book, ch. 13 (text classification and Naive Bayes) and ch. 16–17 (flat and hierarchical clustering). Retrieval, classification and clustering-based topic detection.
- Pang & Lee, “Opinion Mining and Sentiment Analysis” (*Foundations and Trends in Information Retrieval*, 2008). The standard survey on sentiment analysis.
- [Blei, “Probabilistic Topic Models” (*Communications of the ACM*, 2012)](https://www.cs.columbia.edu/~blei/papers/Blei2012.pdf). LDA, the probabilistic alternative to the clustering and LSA approaches in the slides.

### Neural NLP setup (embeddings → model → prediction)

- ⭐ [Goldberg, “A Primer on Neural Network Models for Natural Language Processing” (JAIR 2016)](https://arxiv.org/abs/1510.00726). Uses the same three-part framing and covers feature representations, pooling and classifiers.

## Lecture 2 — Neural network primer & word embeddings

Read the two original word2vec papers alongside Goldberg & Levy’s derivation. Together they cover what the CBOW and skip-gram slides show only as pictures, including negative sampling, which the slides skip.

### Neural network basics, softmax, backpropagation

- ⭐ [CS231n notes, “Backpropagation, Intuitions”](https://cs231n.github.io/optimization-2/). Chain rule on computational graphs, worked by hand.
- [Olah, “Calculus on Computational Graphs: Backpropagation” (2015)](https://colah.github.io/posts/2015-08-Backprop/). Why reverse-mode differentiation is efficient.
- [Nielsen, *Neural Networks and Deep Learning*, ch. 2 “How the backpropagation algorithm works”](http://neuralnetworksanddeeplearning.com/chap2.html). A gentler, step-by-step derivation.
- Goodfellow et al., *Deep Learning*, ch. 6 “Deep Feedforward Networks”. Includes the softmax output unit and cross-entropy loss (“softmax will keep popping up”).
- Rumelhart, Hinton & Williams, “Learning representations by back-propagating errors” (*Nature*, 1986). The original paper, for historical context.

### Distributional semantics and word2vec (CBOW, skip-gram)

- ⭐ [Mikolov et al., “Efficient Estimation of Word Representations in Vector Space” (2013)](https://arxiv.org/abs/1301.3781). The CBOW and skip-gram paper.
- ⭐ [Mikolov et al., “Distributed Representations of Words and Phrases and their Compositionality” (NeurIPS 2013)](https://arxiv.org/abs/1310.4546). Negative sampling and subsampling: how the full-vocabulary softmax is avoided in practice.
- [Goldberg & Levy, “word2vec Explained” (2014)](https://arxiv.org/abs/1402.3722). A four-page derivation of the negative-sampling objective.
- [Rong, “word2vec Parameter Learning Explained” (2014)](https://arxiv.org/abs/1411.2738). Full gradient derivations for CBOW and skip-gram; pairs well with the backprop review.
- [Alammar, “The Illustrated Word2vec”](https://jalammar.github.io/illustrated-word2vec/). A visual walkthrough of the training loop.
- Jurafsky & Martin (SLP3), chapter “Vector Semantics and Embeddings”. Covers TF-IDF → PMI → word2vec in one arc, matching the lecture.
- Harris, “Distributional Structure” (*Word*, 1954). The distributional hypothesis, usually quoted alongside Firth (1957).

### Matrix factorization view (LSA, PMI)

- Deerwester et al., “Indexing by Latent Semantic Analysis” (*JASIS*, 1990). The original LSA paper.
- Levy, Goldberg & Dagan, “Improving Distributional Similarity with Lessons Learned from Word Embeddings” (*TACL*, 2015). Shows that much of word2vec’s edge comes from hyperparameters that also help PMI and SVD. A follow-up to the Levy & Goldberg paper the slides link.

### Beyond static word vectors

- [Bojanowski et al., “Enriching Word Vectors with Subword Information” (fastText, 2017)](https://arxiv.org/abs/1607.04606). Character n-gram embeddings that handle rare words and rich morphology.
- [Bolukbasi et al., “Man is to Computer Programmer as Woman is to Homemaker?” (NeurIPS 2016)](https://arxiv.org/abs/1607.06520). Bias in embeddings; a preview of the Responsible AI week.

### The classical NLP pipeline

- Jurafsky & Martin (SLP3), the chapters on part-of-speech tagging and named entities, dependency parsing, constituency parsing, and coreference resolution. One chapter per pipeline stage on the slide.
- [Collobert et al., “Natural Language Processing (almost) from Scratch” (JMLR 2011)](https://arxiv.org/abs/1103.0398). The full journal version of the joint-learning work the slides cite as an “early example”.

## Lecture 3 — Language modeling, part 1

The SLP3 n-gram chapter covers everything on the classical half of these slides, with worked examples. Bengio et al. (2003) is the paper behind the fixed-context neural LM slides.

### n-gram LMs, the chain rule, the Markov assumption

- ⭐ Jurafsky & Martin (SLP3), chapter “N-gram Language Models”. Counting, the Markov assumption, perplexity, and smoothing, with the Shakespeare sparsity example.
- Shannon, “Prediction and Entropy of Printed English” (*Bell System Technical Journal*, 1951). Where next-symbol prediction and entropy estimates for language started.

### Perplexity

- [Hugging Face docs, “Perplexity of fixed-length models”](https://huggingface.co/docs/transformers/perplexity). How perplexity is computed in practice for models with a context limit, including the sliding-window strategy.

### Smoothing (additive, discounting, back-off, interpolation)

- Chen & Goodman, “An empirical study of smoothing techniques for language modeling” (*Computer Speech & Language*, 1999). The definitive comparison, and the clearest description of Kneser-Ney, the method that won.

### Fixed-context neural LMs

- ⭐ [Bengio et al., “A Neural Probabilistic Language Model” (JMLR 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). The original model: the W and U matrices from the slides.
- [Voita, NLP Course, “Language Modeling”](https://lena-voita.github.io/nlp_course.html). Illustrated from n-grams to neural LMs, with good intuition for perplexity.

### Recurrent neural networks

- ⭐ [Karpathy, “The Unreasonable Effectiveness of Recurrent Neural Networks” (2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Source of the Shakespeare and Wikipedia generation examples, with a minimal character-level RNN.
- Elman, “Finding Structure in Time” (*Cognitive Science*, 1990). The Elman network itself.
- Mikolov et al., “Recurrent neural network based language model” (Interspeech 2010). The first strong RNN LM results against n-gram baselines.
- Goodfellow et al., *Deep Learning*, ch. 10 “Sequence Modeling: Recurrent and Recursive Nets”. Unrolling, parameter sharing, and the formal treatment.

## Lecture 4 — Language modeling, part 2

Olah’s LSTM post is the single best companion to the gate-by-gate slides. Pascanu et al. (2013) gives the math behind “vanishing gradients”.

### Backpropagation through time and autodiff

- Werbos, “Backpropagation through time: what it does and how to do it” (*Proceedings of the IEEE*, 1990). The original BPTT exposition.
- [Baydin et al., “Automatic Differentiation in Machine Learning: a Survey” (JMLR 2018)](https://arxiv.org/abs/1502.05767). What PyTorch does when “you only define the forward pass”.

### Vanishing and exploding gradients

- ⭐ [Pascanu, Mikolov & Bengio, “On the difficulty of training Recurrent Neural Networks” (ICML 2013)](https://arxiv.org/abs/1211.5063). Formal analysis via the recurrent Jacobian, plus gradient clipping for the exploding case.
- Bengio, Simard & Frasconi, “Learning long-term dependencies with gradient descent is difficult” (*IEEE Trans. Neural Networks*, 1994). The original statement of the problem.

### LSTMs and GRUs

- ⭐ [Olah, “Understanding LSTM Networks” (2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Forget, input and output gates, diagrammed step by step.
- Hochreiter & Schmidhuber, “Long Short-Term Memory” (*Neural Computation*, 1997), and Gers, Schmidhuber & Cummins, “Learning to Forget” (2000). The original LSTM, and the paper that added the forget gate.
- [Cho et al., “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation” (2014)](https://arxiv.org/abs/1406.1078). Introduces the GRU and also previews Lecture 5’s encoder-decoder.
- [Karpathy, Johnson & Fei-Fei, “Visualizing and Understanding Recurrent Networks” (2015)](https://arxiv.org/abs/1506.02078). The cell-state visualizations (quote and indentation tracking) from the slides.

### Which architecture is better?

- [Chung et al., “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling” (2014)](https://arxiv.org/abs/1412.3555). LSTM vs. GRU head to head.
- [Greff et al., “LSTM: A Search Space Odyssey” (2017)](https://arxiv.org/abs/1503.04069). Ablates each gate; the forget gate and output activation matter most.
- Jozefowicz, Zaremba & Sutskever, “An Empirical Exploration of Recurrent Network Architectures” (ICML 2015). An architecture search over around 10,000 RNN variants, matching the “Optimal architectures?” slide.
- [Weiss, Goldberg & Yahav, “On the Practical Computational Power of Finite Precision RNNs for Language Recognition” (ACL 2018)](https://arxiv.org/abs/1805.04908). The source of “GRUs can’t count, LSTMs can”.

### Deep and bidirectional RNNs

- Schuster & Paliwal, “Bidirectional Recurrent Neural Networks” (*IEEE Trans. Signal Processing*, 1997). The original bidirectional RNN.
- [Graves, “Generating Sequences With Recurrent Neural Networks” (2013)](https://arxiv.org/abs/1308.0850). Deep stacked LSTMs for text generation.

## Lecture 5 — Seq2seq, attention, Transformers

The slides already link The Annotated and The Illustrated Transformer. What follows adds the original papers, the RNN-attention story that comes before them, and the position-embedding work behind the “poor generalization to longer sequences” question.

### Encoder-decoder models

- ⭐ [Sutskever, Vinyals & Le, “Sequence to Sequence Learning with Neural Networks” (2014)](https://arxiv.org/abs/1409.3215). The seq2seq paper, including the source-reversal trick.
- [Alammar, “Visualizing A Neural Machine Translation Model”](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). Animated seq2seq with and without attention, matching the toy “Je m’appelle” example.

### Attention

- ⭐ [Bahdanau, Cho & Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate” (2014)](https://arxiv.org/abs/1409.0473). The attention paper that fixed the temporal bottleneck.
- [Luong, Pham & Manning, “Effective Approaches to Attention-based Neural Machine Translation” (2015)](https://arxiv.org/abs/1508.04025). Dot, general (bilinear) and concat scoring functions, the “many possible functions” slide.
- [Weng, “Attention? Attention!” (2018)](https://lilianweng.github.io/posts/2018-06-24-attention/). A survey-style blog post comparing attention variants.
- [Jain & Wallace, “Attention is not Explanation” (2019)](https://arxiv.org/abs/1902.10186) and [Wiegreffe & Pinter, “Attention is not not Explanation” (2019)](https://arxiv.org/abs/1908.04626). Read as a pair; they complicate the “Interpretability?” slide.

### The Transformer and self-attention

- ⭐ [Vaswani et al., “Attention Is All You Need” (NeurIPS 2017)](https://arxiv.org/abs/1706.03762). The original paper; short and readable after this lecture.
- [The Annotated Transformer, 2022 update](https://nlp.seas.harvard.edu/annotated-transformer/). The slides link the 2018 version; this one has refreshed, runnable PyTorch code.
- [Karpathy, “Let’s build GPT: from scratch, in code, spelled out” (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY). Builds masked multi-head self-attention, residuals and LayerNorm live in about two hours.
- [Phuong & Hutter, “Formal Algorithms for Transformers” (2022)](https://arxiv.org/abs/2207.09238). Precise pseudocode for every component, useful for exam revision.

### Residual connections and layer normalization

- [He et al., “Deep Residual Learning for Image Recognition” (2015)](https://arxiv.org/abs/1512.03385). Where residual connections come from.
- [Ba, Kiros & Hinton, “Layer Normalization” (2016)](https://arxiv.org/abs/1607.06450). The LayerNorm paper.
- [Xiong et al., “On Layer Normalization in the Transformer Architecture” (2020)](https://arxiv.org/abs/2002.04745). Pre-LN vs. post-LN placement, and why modern models use pre-LN.

### Position embeddings

- [Shaw, Uszkoreit & Vaswani, “Self-Attention with Relative Position Representations” (2018)](https://arxiv.org/abs/1803.02155). Relative instead of absolute positions.
- [Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding” (2021)](https://arxiv.org/abs/2104.09864). RoPE, the scheme most current LLMs use.
- [Press, Smith & Lewis, “Train Short, Test Long” (ALiBi, 2021)](https://arxiv.org/abs/2108.12409). Directly addresses the slide’s answer: learned positions do not extrapolate to longer inputs. Useful for the Lecture 6 position-embedding exercise.

## Lecture 6 — Pre-trained LMs (GPT, ELMo, BERT, BART, T5)

Every model named on the slides has a primary paper, listed here. Two things the lecture uses but never explains get their own subsections: subword tokenization, needed for the whole-word-masking slide, and work on what BERT learns.

### GPT, GPT-2, and the transfer-learning paradigm

- ⭐ [Radford et al., “Improving Language Understanding by Generative Pre-Training” (GPT, 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- [Radford et al., “Language Models are Unsupervised Multitask Learners” (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). The move from fine-tuning to zero-shot prediction.
- [Howard & Ruder, “Universal Language Model Fine-tuning for Text Classification” (ULMFiT, 2018)](https://arxiv.org/abs/1801.06146). The LSTM-era paper that set up “pretrain, then fine-tune”, which the transfer-learning slides depict.
- [Alammar, “The Illustrated GPT-2”](https://jalammar.github.io/illustrated-gpt2/). Decoder-only blocks and masked self-attention, visualized.

### ELMo and BERT

- ⭐ [Peters et al., “Deep contextualized word representations” (ELMo, 2018)](https://arxiv.org/abs/1802.05365).
- ⭐ [Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” (2018)](https://arxiv.org/abs/1810.04805). MLM, \[CLS\], and the ablations shown on the slides.
- [Alammar, “The Illustrated BERT, ELMo, and co.”](https://jalammar.github.io/illustrated-bert/). Visual companion for the whole lecture.
- [Peters, Ruder & Smith, “To Tune or Not to Tune?” (2019)](https://arxiv.org/abs/1903.05987). When to fine-tune versus extract features, the “Fine-tuning vs. Embeddings” slide.

### Benchmarks

- [Wang et al., “GLUE” (2018)](https://arxiv.org/abs/1804.07461) and [Wang et al., “SuperGLUE” (2019)](https://arxiv.org/abs/1905.00537). GLUE’s tasks, and why it saturated quickly.

### What does BERT learn?

- ⭐ [Clark et al., “What Does BERT Look At? An Analysis of BERT’s Attention” (2019)](https://arxiv.org/abs/1906.04341). The attention-head figures on the “Demystifying BERT” slides.
- [Tenney, Das & Pavlick, “BERT Rediscovers the Classical NLP Pipeline” (2019)](https://arxiv.org/abs/1905.05950). BERT’s layers map onto POS → parsing → NER → coreference, linking back to Lecture 2’s pipeline.
- [Rogers, Kovaleva & Rumshisky, “A Primer in BERTology” (2020)](https://arxiv.org/abs/2002.12327). A survey of 150+ studies of BERT.

### BERT variants

- [Liu et al., “RoBERTa” (2019)](https://arxiv.org/abs/1907.11692), [Sanh et al., “DistilBERT” (2019)](https://arxiv.org/abs/1910.01108), and [Clark et al., “ELECTRA” (2020)](https://arxiv.org/abs/2003.10555). The three variants on the slides.
- [Joshi et al., “SpanBERT” (2019)](https://arxiv.org/abs/1907.10529). Masking contiguous spans, a follow-up to the “What to mask?” slide.

### Encoder-decoder pre-training

- ⭐ [Lewis et al., “BART” (2019)](https://arxiv.org/abs/1910.13461). The noising functions compared on the “which encoder-side corruption?” slide.
- ⭐ [Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer” (T5, 2019)](https://arxiv.org/abs/1910.10683). Long, but sections 3.1–3.3 are a model of how to run ablations, useful for the course project.

### Subword tokenization (used but not covered in the slides)

- ⭐ [Sennrich, Haddow & Birch, “Neural Machine Translation of Rare Words with Subword Units” (BPE, 2015)](https://arxiv.org/abs/1508.07909). BPE, used by GPT models.
- [Wu et al., “Google’s Neural Machine Translation System” (2016)](https://arxiv.org/abs/1609.08144). Describes WordPiece, BERT’s tokenizer.
- [Kudo, “Subword Regularization” (2018)](https://arxiv.org/abs/1804.10959) and [Kudo & Richardson, “SentencePiece” (2018)](https://arxiv.org/abs/1808.06226). The unigram LM tokenizer, used by T5.
- [Karpathy, “Let’s build the GPT Tokenizer” (video)](https://www.youtube.com/watch?v=zduSFxRajkE). Implements BPE from scratch and shows the tokenization quirks behind many LLM failures.

### Quadratic attention cost (the “disadvantage of Transformers” slide)

- [Tay et al., “Efficient Transformers: A Survey” (2020)](https://arxiv.org/abs/2009.06732). A taxonomy of sub-quadratic attention variants.
- [Beltagy et al., “Longformer” (2020)](https://arxiv.org/abs/2004.05150) and [Zaheer et al., “Big Bird” (2020)](https://arxiv.org/abs/2007.14062). Sparse-attention models; Big Bird is the paper behind the Google blog post the slides link.
- [Dao et al., “FlashAttention” (2022)](https://arxiv.org/abs/2205.14135). Exact attention made fast through memory-aware kernels: the approach that won in practice.

## Lecture 7 — In-context learning, prompting, RLHF

Read the GPT-3 and InstructGPT papers first; most slides in this lecture are figures taken from them. The “Controversial topic!!” slide on emergence has a clear pro/con pair below.

### Scaling

- ⭐ [Kaplan et al., “Scaling Laws for Neural Language Models” (2020)](https://arxiv.org/abs/2001.08361).
- [Hoffmann et al., “Training Compute-Optimal Large Language Models” (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556). Revises Kaplan: scale data and parameters equally. A preview of Week 14.
- [Sevilla et al., “Compute Trends Across Three Eras of Machine Learning” (2022)](https://arxiv.org/abs/2202.05924) and [Cottier et al., “The rising costs of training frontier AI models” (2024)](https://arxiv.org/abs/2405.21015). Data behind the training-cost growth slide. [Epoch AI’s trends page](https://epoch.ai/trends) keeps the numbers current.

### Emergence (both sides)

- Anderson, “More Is Different” (*Science*, 1972). The source of the quote on the slide.
- ⭐ [Wei et al., “Emergent Abilities of Large Language Models” (2022)](https://arxiv.org/abs/2206.07682). The case for emergence.
- ⭐ [Schaeffer, Miranda & Koyejo, “Are Emergent Abilities of Large Language Models a Mirage?” (NeurIPS 2023)](https://arxiv.org/abs/2304.15004). The case against: emergence can be an artifact of discontinuous metrics.

### In-context learning

- ⭐ [Brown et al., “Language Models are Few-Shot Learners” (GPT-3, 2020)](https://arxiv.org/abs/2005.14165). The TriviaQA and translation results on the slides.
- [Zhao et al., “Calibrate Before Use” (2021)](https://arxiv.org/abs/2102.09690) and [Lu et al., “Fantastically Ordered Prompts and Where to Find Them” (2021)](https://arxiv.org/abs/2104.08786). Why “ICL is sensitive” to example choice, order and format.
- [Min et al., “Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?” (2022)](https://arxiv.org/abs/2202.12837). Labels in demonstrations matter surprisingly little.
- [Xie et al., “An Explanation of In-context Learning as Implicit Bayesian Inference” (2021)](https://arxiv.org/abs/2111.02080) and [Olsson et al., “In-context Learning and Induction Heads” (2022)](https://arxiv.org/abs/2209.11895). Two answers to “Why does in-context learning work?”, one theoretical and one mechanistic.

### Prompting and prompt search

- [Liu et al., “Pre-train, Prompt, and Predict” (2021)](https://arxiv.org/abs/2107.13586). A comprehensive survey of prompting methods.
- [Shin et al., “AutoPrompt” (2020)](https://arxiv.org/abs/2010.15980). Gradient-searched discrete prompts that look like random tokens, the “optimal prompt search” slide.
- ⭐ [Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models” (2022)](https://arxiv.org/abs/2201.11903).
- [Kojima et al., “Large Language Models are Zero-Shot Reasoners” (2022)](https://arxiv.org/abs/2205.11916) (“Let’s think step by step”) and [Wang et al., “Self-Consistency” (2022)](https://arxiv.org/abs/2203.11171). Zero-shot CoT, and sampling several chains then majority-voting. A bridge to the Reasoning week.

### Soft prompts and parameter-efficient fine-tuning

- ⭐ [Lester, Al-Rfou & Constant, “The Power of Scale for Parameter-Efficient Prompt Tuning” (2021)](https://arxiv.org/abs/2104.08691). The prompt-tuning paper behind the soft-prompt slides.
- [Li & Liang, “Prefix-Tuning” (2021)](https://arxiv.org/abs/2101.00190). Soft prompts at every layer.
- [Houlsby et al., “Parameter-Efficient Transfer Learning for NLP” (adapters, 2019)](https://arxiv.org/abs/1902.00751), [Hu et al., “LoRA” (2021)](https://arxiv.org/abs/2106.09685), and [Dettmers et al., “QLoRA” (2023)](https://arxiv.org/abs/2305.14314). PEFT is on the syllabus but not in the slides; LoRA is the method students are most likely to use in projects.

### Instruction tuning

- [Wei et al., “Finetuned Language Models Are Zero-Shot Learners” (FLAN, 2021)](https://arxiv.org/abs/2109.01652), [Sanh et al., “Multitask Prompted Training Enables Zero-Shot Task Generalization” (T0, 2021)](https://arxiv.org/abs/2110.08207), and [Chung et al., “Scaling Instruction-Finetuned Language Models” (Flan-T5, 2022)](https://arxiv.org/abs/2210.11416). SFT on many tasks, before RLHF.
- [Zhou et al., “LIMA: Less Is More for Alignment” (2023)](https://arxiv.org/abs/2305.11206). About 1,000 curated examples can go a long way, a counterpoint to “expensive to collect lots of data”.

### RLHF: reward models, REINFORCE, PPO

- ⭐ [Ouyang et al., “Training language models to follow instructions with human feedback” (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155). The three-step SFT → RM → PPO figure is from this paper.
- [Christiano et al., “Deep reinforcement learning from human preferences” (2017)](https://arxiv.org/abs/1706.03741), [Ziegler et al., “Fine-Tuning Language Models from Human Preferences” (2019)](https://arxiv.org/abs/1909.08593), and [Stiennon et al., “Learning to summarize from human feedback” (2020)](https://arxiv.org/abs/2009.01325). The lineage leading to InstructGPT.
- [Lambert et al., “Illustrating RLHF” (Hugging Face blog)](https://huggingface.co/blog/rlhf) and [Lambert, *Reinforcement Learning from Human Feedback* (online book)](https://rlhfbook.com/). An accessible overview, then a full reference.
- Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning” (*Machine Learning*, 1992). The original REINFORCE.
- [Schulman et al., “Proximal Policy Optimization Algorithms” (2017)](https://arxiv.org/abs/1707.06347). The clipped objective on the “How does PPO differ?” slides.
- [Ahmadian et al., “Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs” (2024)](https://arxiv.org/abs/2402.14740). Argues that plain REINFORCE with a baseline matches PPO for LLMs, directly on the REINFORCE vs. PPO comparison.
- [Gao, Schulman & Hilton, “Scaling Laws for Reward Model Overoptimization” (2022)](https://arxiv.org/abs/2210.10760). Reward hacking, measured.

### Beyond PPO: DPO and GRPO

- ⭐ [Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model” (2023)](https://arxiv.org/abs/2305.18290).
- [Shao et al., “DeepSeekMath” (2024)](https://arxiv.org/abs/2402.03300) and [DeepSeek-AI, “DeepSeek-R1” (2025)](https://arxiv.org/abs/2501.12948). GRPO (on the syllabus, not in the slides) and its use for training reasoning models.
- [Bai et al., “Constitutional AI: Harmlessness from AI Feedback” (2022)](https://arxiv.org/abs/2212.08073). Replaces human preference labels with AI feedback (RLAIF).

## Lecture 8 — Natural language generation

The slides already cite Holtzman et al. (nucleus sampling), Fan et al. (top-k), scheduled sampling, and the RL-for-generation papers. This section adds why beam search fails, decoding methods newer than top-p, and the calibration question that closes the decoding part.

### Overview

- ⭐ [von Platen, “How to generate text: using different decoding methods for language generation with Transformers” (Hugging Face blog)](https://huggingface.co/blog/how-to-generate). Greedy, beam, top-k, top-p and temperature, each with code and outputs. The quickest way to try the lecture yourself.
- [Wiher, Meister & Cotterell, “On Decoding Strategies for Neural Text Generators” (TACL 2022)](https://arxiv.org/abs/2203.15721). Compares decoders across tasks and finds that the best one depends on the task, the “works for MT, not for open-ended generation” point.

### Why maximum-likelihood decoding goes wrong

- [Koehn & Knowles, “Six Challenges for Neural Machine Translation” (2017)](https://arxiv.org/abs/1706.03872). Shows that larger beams make translations worse.
- [Stahlberg & Byrne, “On NMT Search Errors and Model Errors: Cat Got Your Tongue?” (2019)](https://arxiv.org/abs/1908.10090). Exact search finds that the most likely translation is often the empty string, a striking answer to “what do maximum likelihood sequences look like?”.
- ⭐ [Meister, Cotterell & Vieira, “If beam search is the answer, what was the question?” (2020)](https://arxiv.org/abs/2010.02650). Beam search works because of a hidden bias toward uniform information density.
- [Vijayakumar et al., “Diverse Beam Search” (2016)](https://arxiv.org/abs/1610.02424). A beam-search variant that yields diverse outputs.

### Repetition

- [Welleck et al., “Neural Text Generation with Unlikelihood Training” (2019)](https://arxiv.org/abs/1908.04319). A training-time fix for the repetition problem.

### Sampling beyond top-k and top-p

- [Hewitt, Manning & Liang, “Truncation Sampling as Language Model Desmoothing” (2022)](https://arxiv.org/abs/2210.15191). Why truncation helps, plus η-sampling.
- [Meister et al., “Locally Typical Sampling” (2022)](https://arxiv.org/abs/2202.00666). Samples tokens whose information content is close to the expected value.
- [Nguyen et al., “Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs” (2024)](https://arxiv.org/abs/2407.01082). Min-p, widely used in open-source inference stacks today.

### Re-balancing and re-ranking

- [Li et al., “Contrastive Decoding: Open-ended Text Generation as Optimization” (2022)](https://arxiv.org/abs/2210.15097). Re-balances an expert LM’s distribution against a small amateur LM.
- [Su et al., “A Contrastive Framework for Neural Text Generation” (contrastive search, 2022)](https://arxiv.org/abs/2202.06417). Penalizes candidates too similar to the context.

### “Shouldn’t our models just be better calibrated?”

- [Ott et al., “Analyzing Uncertainty in Neural Machine Translation” (2018)](https://arxiv.org/abs/1803.00047). Models spread probability mass over many low-quality hypotheses, and how that interacts with beam search.

### Teacher forcing and exposure bias

- Williams & Zipser, “A Learning Algorithm for Continually Running Fully Recurrent Neural Networks” (*Neural Computation*, 1989). The origin of teacher forcing.
- [Lamb et al., “Professor Forcing” (2016)](https://arxiv.org/abs/1610.09038). An adversarial alternative to scheduled sampling.
- [Schmidt, “Generalization in Generation: A closer look at Exposure Bias” (2019)](https://arxiv.org/abs/1910.00292). Argues that exposure bias is really a generalization problem, a useful critical view.

## Lecture 9 — NLG evaluation, question answering, RAG

The slides’ reference list covers RAG and agents (DPR, REALM, RAG, Toolformer, ReAct, and surveys) but has nothing on evaluation or QA. This section concentrates there and adds RAG work from after the cited papers.

### NLG evaluation: overview

- ⭐ [Celikyilmaz, Clark & Gao, “Evaluation of Text Generation: A Survey” (2020)](https://arxiv.org/abs/2006.14799). Human, overlap-based and model-based metrics, organized the same way as the lecture.
- [Gehrmann, Clark & Sellam, “Repairing the Cracked Foundation” (2022)](https://arxiv.org/abs/2202.06935). Common evaluation mistakes in NLG papers; a good checklist for project reports.

### Content-overlap metrics

- [Papineni et al., “BLEU” (ACL 2002)](https://aclanthology.org/P02-1040/), [Lin, “ROUGE” (2004)](https://aclanthology.org/W04-1013/), [Banerjee & Lavie, “METEOR” (2005)](https://aclanthology.org/W05-0909/), and [Popović, “chrF” (2015)](https://aclanthology.org/W15-3049/). The original metric papers.
- [Post, “A Call for Clarity in Reporting BLEU Scores” (2018)](https://arxiv.org/abs/1804.08771). Why BLEU numbers are not comparable across papers, and sacreBLEU.
- ⭐ [Liu et al., “How NOT To Evaluate Your Dialogue System” (2016)](https://arxiv.org/abs/1603.08023). Overlap metrics barely correlate with human judgment for dialogue, the “simple failure case” slide.
- [Reiter, “A Structured Review of the Validity of BLEU” (*Computational Linguistics*, 2018)](https://aclanthology.org/J18-3002/). A meta-review of when BLEU tracks human judgment.

### Model-based metrics

- [Kusner et al., “From Word Embeddings To Document Distances” (Word Mover’s Distance, ICML 2015)](https://proceedings.mlr.press/v37/kusnerb15.html). The “word distance functions” slides.
- ⭐ [Zhang et al., “BERTScore” (2019)](https://arxiv.org/abs/1904.09675) and [Zhao et al., “MoverScore” (2019)](https://arxiv.org/abs/1909.02622). Contextual-embedding matching.
- [Sellam et al., “BLEURT” (2020)](https://arxiv.org/abs/2004.04696), [Rei et al., “COMET” (2020)](https://arxiv.org/abs/2009.09025), and [Yuan et al., “BARTScore” (2021)](https://arxiv.org/abs/2106.11520). Learned metrics; COMET is the current standard in MT.

### LLMs as judges

- ⭐ [Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena” (2023)](https://arxiv.org/abs/2306.05685). Agreement with humans, plus position, verbosity and self-preference biases.
- [Liu et al., “G-Eval” (2023)](https://arxiv.org/abs/2303.16634) and [Wang et al., “Large Language Models are not Fair Evaluators” (2023)](https://arxiv.org/abs/2305.17926). A method, then its failure modes.

### Human evaluation

- [van der Lee et al., “Best practices for the human evaluation of automatically generated text” (INLG 2019)](https://aclanthology.org/W19-8643/). Practical guidelines for designing a human study.
- [Clark et al., “All That’s ‘Human’ Is Not Gold” (2021)](https://arxiv.org/abs/2107.00061) and [Karpinska, Akoury & Iyyer, “The Perils of Using Mechanical Turk to Evaluate Open-Ended Text Generation” (2021)](https://arxiv.org/abs/2109.06835). The issues on the “Do we have anything to worry about?” slides.

### Faithfulness and factuality

- [Maynez et al., “On Faithfulness and Factuality in Abstractive Summarization” (2020)](https://arxiv.org/abs/2005.00661), [Min et al., “FActScore” (2023)](https://arxiv.org/abs/2305.14251), and [Ji et al., “Survey of Hallucination in Natural Language Generation” (2022)](https://arxiv.org/abs/2202.03629). The hallucination problem that motivates RAG.

### Question answering: classical and extractive

- Jurafsky & Martin (SLP3), chapter on information retrieval and question answering (including RAG in recent drafts). Textbook coverage of the whole second half of the lecture.
- [Chen & Yih, “Open-Domain Question Answering” (ACL 2020 tutorial)](https://aclanthology.org/2020.acl-tutorials.8/). From retriever-reader pipelines to dense retrieval.
- Ferrucci et al., “Building Watson: An Overview of the DeepQA Project” (*AI Magazine*, 2010). A canonical classical QA pipeline.
- ⭐ [Rajpurkar et al., “SQuAD” (2016)](https://arxiv.org/abs/1606.05250) and [Rajpurkar, Jia & Liang, “Know What You Don’t Know” (SQuAD 2.0, 2018)](https://arxiv.org/abs/1806.03822).
- [Seo et al., “Bidirectional Attention Flow for Machine Comprehension” (BiDAF, 2016)](https://arxiv.org/abs/1611.01603). The classic pre-BERT extractive reader, predicting start and end spans.
- ⭐ [Jia & Liang, “Adversarial Examples for Evaluating Reading Comprehension Systems” (2017)](https://arxiv.org/abs/1707.07328). The evidence behind “Is reading comprehension solved?”.
- [Joshi et al., “TriviaQA” (2017)](https://arxiv.org/abs/1705.03551), [Yang et al., “HotpotQA” (2018)](https://arxiv.org/abs/1809.09600), and [Kwiatkowski et al., “Natural Questions” (TACL 2019)](https://aclanthology.org/Q19-1026/). The other major QA datasets.

### Generative and long-form QA, and its evaluation

- [Fan et al., “ELI5: Long Form Question Answering” (2019)](https://arxiv.org/abs/1907.09190).
- ⭐ [Krishna, Roy & Iyyer, “Hurdles to Progress in Long-form Question Answering” (2021)](https://arxiv.org/abs/2103.06332). Shows that ROUGE is uninformative for long-form QA.
- [Xu et al., “A Critical Evaluation of Evaluations for Long-form Question Answering” (2023)](https://arxiv.org/abs/2305.18201) and [Kamalloo et al., “Evaluating Open-Domain Question Answering in the Era of Large Language Models” (2023)](https://arxiv.org/abs/2305.06984). Exact match and F1 undercount correct LLM answers, the “QA systems in the LLM era” slides.

### Closed-book QA: knowledge in parameters

- [Petroni et al., “Language Models as Knowledge Bases?” (2019)](https://arxiv.org/abs/1909.01066) and [Roberts, Raffel & Shazeer, “How Much Knowledge Can You Pack Into the Parameters of a Language Model?” (2020)](https://arxiv.org/abs/2002.08910). The “closed book” side of the “closed book vs. retrieval” slide.

### Retrieval

- ⭐ Robertson & Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond” (*Foundations and Trends in IR*, 2009). BM25, the sparse baseline DPR is compared against, is TF-IDF from Lecture 1 refined.
- [Chen et al., “Reading Wikipedia to Answer Open-Domain Questions” (DrQA, 2017)](https://arxiv.org/abs/1704.00051). The retriever-reader setup behind “finding the answer in 21M documents”.
- [Khattab & Zaharia, “ColBERT” (2020)](https://arxiv.org/abs/2004.12832). Late interaction, a middle ground between DPR and cross-encoders.
- [Thakur et al., “BEIR” (2021)](https://arxiv.org/abs/2104.08663). Dense retrievers often lose to BM25 out of domain.
- [Johnson, Douze & Jégou, “Billion-scale similarity search with GPUs” (FAISS, 2017)](https://arxiv.org/abs/1702.08734) and [Malkov & Yashunin, “HNSW” (2016)](https://arxiv.org/abs/1603.09320). How dense retrieval is made fast in practice.
- [Gao et al., “Precise Zero-Shot Dense Retrieval without Relevance Labels” (HyDE, 2022)](https://arxiv.org/abs/2212.10496). Retrieval using an LLM-generated hypothetical answer.

### Retrieval-augmented LMs: beyond the cited papers

- ⭐ [Asai et al., “Retrieval-based Language Models and Applications” (ACL 2023 tutorial)](https://acl2023-retrieval-lm.github.io/). Organizes the field along the same axes as the “landscape” slide: what, how and when to retrieve.
- [Gao et al., “Retrieval-Augmented Generation for Large Language Models: A Survey” (2023)](https://arxiv.org/abs/2312.10997). Naive, advanced and modular RAG.
- [Izacard & Grave, “Fusion-in-Decoder” (2020)](https://arxiv.org/abs/2007.01282) and [Borgeaud et al., “RETRO” (2021)](https://arxiv.org/abs/2112.04426). Two more architectures for the “landscape” slide.
- [Ram et al., “In-Context Retrieval-Augmented Language Models” (2023)](https://arxiv.org/abs/2302.00083) and [Shi et al., “REPLUG” (2023)](https://arxiv.org/abs/2301.12652). RAG with a frozen, black-box LM: how most RAG systems are built today.
- [Asai et al., “Self-RAG” (2023)](https://arxiv.org/abs/2310.11511). The model decides when to retrieve and critiques its own output.
- ⭐ [Liu et al., “Lost in the Middle: How Language Models Use Long Contexts” (2023)](https://arxiv.org/abs/2307.03172). Where you place retrieved passages in the prompt matters.
- [Es et al., “Ragas: Automated Evaluation of Retrieval Augmented Generation” (2023)](https://arxiv.org/abs/2309.15217). Ties RAG back to the evaluation half of the lecture.
- [Zhang et al., “ERNIE: Enhanced Language Representation with Informative Entities” (2019)](https://arxiv.org/abs/1905.07129). The knowledge-graph ERNIE on the slides. A different model, Baidu’s ERNIE, shares the name.
