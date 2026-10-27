# 09 - QA and Evaluation

This tutorial is about measuring how good generated text is, and about answering questions. You implement the classic overlap metrics **BLEU** and **ROUGE** and the model-based **BERTScore**, find out where they fail, and check how well they agree with **human judgements** of summaries. You then build **extractive question answering** by hand with a BERT model fine-tuned on SQuAD, evaluate it with **exact match** and **F1**, teach it to **abstain** on unanswerable questions, and compare it with a small chat model that **generates** its answers. Finally, you build a **retrieval-augmented generation** (RAG) system: **BM25** and **dense retrieval** to find the right passages, an LLM to answer from them, and an **LLM judge** to grade the answers.

## Notebooks
- `0. Evaluation Metrics.ipynb`: the first exercise notebook. You implement corpus BLEU and ROUGE-N/ROUGE-L and check them against `sacrebleu` and `rouge-score`, test them on paraphrases and negations, implement BERTScore with DistilBERT, and correlate all metrics with human ratings on SummEval. Runs on a CPU.
- `0. Evaluation Metrics - Solutions.ipynb`: the worked solutions. Try to finish a part before looking at them.
- `1. Question Answering.ipynb`: the main exercise notebook. You decode answer spans from the start and end logits of `deepset/bert-base-cased-squad2`, implement the SQuAD metrics, tune a no-answer threshold on SQuAD 2.0, and compare extractive QA with `Qwen/Qwen2.5-0.5B-Instruct`, open-book and closed-book. Runs on a CPU.
- `1. Question Answering - Solutions.ipynb`: the worked solutions for the QA notebook.
- `2. Retrieval-Augmented Generation.ipynb`: the follow-up notebook. You implement BM25 with an inverted index, dense retrieval with mean pooling and reciprocal rank fusion, evaluate them with Recall@k on the SQuAD paragraphs, and prompt a hosted LLM through the free Groq API with retrieved passages. No GPU needed, but you need your free Groq account from week 7.
- `2. Retrieval-Augmented Generation - Solutions.ipynb`: the worked solutions for the RAG notebook.

## What you will practice
In `0. Evaluation Metrics.ipynb`:
1. **BLEU**: clipped n-gram precision, the brevity penalty, and why BLEU is a corpus-level metric
2. **ROUGE**: n-gram precision and recall, and the longest common subsequence
3. **Where overlap metrics fail**: paraphrases, negations and word salad
4. **BERTScore**: greedy matching of contextual token embeddings, and why its scores are always high
5. **Can we trust a metric?**: summary-level Kendall's τ with human ratings of relevance and consistency on SummEval, against a length baseline
6. **MCQs**: on evaluation metrics

In `1. Question Answering.ipynb`:
1. **SQuAD**: contexts, questions, and answers stored as character positions
2. **Extractive QA by hand**: start and end logits, offset mappings, and the best valid span
3. **Evaluating QA**: answer normalisation, exact match and F1, and why the two differ
4. **Questions without an answer**: SQuAD 2.0, the no-answer score, and the trade-off of the threshold
5. **Generative QA**: a chat model with and without the passage, answer containment, and hallucinations
6. **MCQs**: on question answering

In `2. Retrieval-Augmented Generation.ipynb`:
1. **Why retrieval?**: an LLM that does not know the answer, and one that is given the documents
2. **A retrieval benchmark**: the SQuAD paragraphs as a collection, and Recall@k
3. **BM25**: idf, term-frequency saturation, length normalisation, and an inverted index
4. **Dense retrieval**: mean pooling with MiniLM, cosine similarity, and reciprocal rank fusion
5. **RAG**: closed-book, BM25, hybrid and oracle passages, and whether the retriever or the generator is to blame
6. **LLM-as-a-judge**: grading answers with an LLM, and its biases
7. **MCQs**: on retrieval and RAG

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks need the updated project environment (`uv sync`), which now includes `rouge-score`.

`0. Evaluation Metrics.ipynb` downloads **`distilbert/distilbert-base-uncased`** (~270 MB, still cached if you did week 7) and the **SummEval** dataset `mteb/summeval` (~1 MB). Scoring the 1,600 SummEval summaries in Part 5 takes about a minute on a CPU; everything else takes seconds.

`1. Question Answering.ipynb` downloads **`deepset/bert-base-cased-squad2`** (~430 MB), **`Qwen/Qwen2.5-0.5B-Instruct`** (~1 GB), and the **SQuAD** and **SQuAD 2.0** validation sets (~20 MB). Nothing is trained. On a CPU, the evaluations in Parts 3 and 4 (200 questions each) take about a minute each, and the generative comparison in Part 5 takes a few minutes, so start these cells and read ahead while they run.

`2. Retrieval-Augmented Generation.ipynb` calls the **Groq API**, like `1. Prompting.ipynb` in week 7. Have your API key ready; no payment details are needed. Never paste your API key into a notebook cell. The notebook also downloads **`sentence-transformers/all-MiniLM-L6-v2`** (~90 MB) and the SQuAD validation set. Embedding the 2,067 paragraphs takes about a minute on a CPU. The LLM experiments use roughly 50,000 tokens of the free daily budget (as of October 2026: 200,000 tokens per day for the model we use) and take a few minutes because of the per-minute limit. Replies are cached in `groq_cache.json`, so re-running a cell costs nothing.

## Further reading
- [BLEU](https://aclanthology.org/P02-1040/) (Papineni et al., 2002), [ROUGE](https://aclanthology.org/W04-1013/) (Lin, 2004), [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) (Post, 2018), [BERTScore](https://arxiv.org/abs/1904.09675) (Zhang et al., 2020) and [SummEval](https://arxiv.org/abs/2007.12626) (Fabbri et al., 2021)
- [Question answering](https://huggingface.co/docs/transformers/tasks/question_answering) in the Hugging Face documentation, and [SQuAD](https://arxiv.org/abs/1606.05250) (Rajpurkar et al., 2016) and [SQuAD 2.0](https://arxiv.org/abs/1806.03822) (Rajpurkar et al., 2018)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020), [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020), [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (Robertson & Zaragoza, 2009) and [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2024)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023)

Please give us feedback for this tutorial!

![LS Feedback 9](../QRs/qrf9.png)
