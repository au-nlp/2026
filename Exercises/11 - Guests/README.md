# 11 - Guests

This week has three guest lectures, and this tutorial has one short, introductory notebook for each. The notebooks are independent, so you can do them in any order. They do not cover everything in the lectures: each one lets you try the core ideas with your own hands.

- **Agentic design patterns** (Aman Chadha): instead of one big prompt or a fully autonomous agent, you combine model calls in structures you design: **prompt chaining** with checks in between, **routing**, **parallelization** and **majority voting**, **reflection** with feedback from a checker, and **orchestrator–workers**.
- **Looking inside a language model** (Martin Tutek): you read GPT-2's intermediate predictions with the **logit lens**, train **probes** that tell true from false statements from its hidden states, find the hidden states that *cause* a prediction with **activation patching**, and test whether a small model's **chain of thought** is faithful by cutting it short or planting mistakes in it (Lanham et al., 2023).
- **Autoformalization with Lean** (André Freitas): you write and prove small theorems in the proof assistant **Lean** with the **Mathlib** library, see formalizations that compile but mean something else, and build a loop in which a language model formalizes statements and repairs them with the **prover's error messages**, judged by a second model for **faithfulness**.

## Notebooks
- `0. Agentic Design Patterns.ipynb`: the five patterns in plain Python. No GPU needed with the Groq API (default); a local model needs a GPU runtime (Colab or Kaggle).
- `1. Inside a Language Model.ipynb`: logit lens, probes, activation patching and CoT faithfulness. **No GPU and no API key needed**: it runs GPT-2 small and Qwen2.5-0.5B-Instruct on your computer.
- `2. Autoformalization with Lean.ipynb`: Lean, Mathlib and an LLM in the loop. No GPU needed, but Lean and Mathlib take about **10 GB of disk** and 5–10 minutes to install, so we recommend **Google Colab** (and on Windows, Colab is the easiest option).
- Each notebook has a `- Solution` version with worked solutions. Try to finish a part before looking at them.

## What you will practice
In `0. Agentic Design Patterns.ipynb`:
1. **Prompt chaining**: extracting JSON, a gate in code, and a second step
2. **Routing**: classifying requests and sending them to specialised prompts
3. **Parallelization**: concurrent calls, and majority voting (self-consistency)
4. **Reflection**: a rule checker as critic, and a revise loop
5. **Orchestrator–workers**: planning subtasks, running workers in parallel, and combining their results
6. **MCQs**: on design patterns

In `1. Inside a Language Model.ipynb`:
1. **The residual stream** of GPT-2 and its hidden states
2. **The logit lens**: the prediction after every layer
3. **Probing**: linear probes for truth, with a control
4. **Activation patching**: forward hooks, clean and corrupted runs, and the logit difference
5. **CoT faithfulness**: early answering and adding mistakes
6. **MCQs**: on interpretability and faithfulness

In `2. Autoformalization with Lean.ipynb`:
1. **Lean basics**: theorems, goals, tactics and `sorry`
2. **Formalizing by hand**: statements and proofs about even and odd numbers
3. **Faithfulness**: changed assumptions, vacuous statements and truncated subtraction
4. **LLM autoformalization**: zero-shot, and refinement with prover feedback
5. **LLM as a judge**: checking semantic fidelity
6. **MCQs**: on autoformalization

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
**LLM backend.** Notebooks 0 and 2 call a language model. At the top of Part 1 (notebook 0) or Part 5 (notebook 2) you choose the backend:
- `BACKEND = "groq"` (default) uses the **Groq API**, like weeks 7 and 10. Have your API key ready, and never paste it into a notebook cell. Notebook 0 uses at most about 25,000 tokens and notebook 2 at most about 20,000 tokens of the free daily budget, so both fit into one day.
- `BACKEND = "local"` runs `Qwen/Qwen2.5-1.5B-Instruct` with `transformers` instead: no account, but use a GPU runtime (Colab: *Runtime → Change runtime type → T4 GPU*). The local model is much smaller, so expect weaker results.

Replies are cached in `llm_cache_*.json`, so re-running a cell costs nothing.

**Running on Colab.** Upload the notebook to [Google Colab](https://colab.research.google.com) (*File → Upload notebook*). Everything else is installed or downloaded by the notebooks themselves; for notebook 0 and 2 with Groq, run `!pip install groq` first if Colab does not have it.

**Notebook 1** downloads GPT-2 small (about 500 MB) and Qwen2.5-0.5B-Instruct (about 1 GB) from the Hugging Face Hub. Parts 1–4 take seconds; Part 5 generates text and takes about 5 minutes on a laptop CPU.

**Notebook 2** installs Lean with [elan](https://github.com/leanprover/elan) and downloads the compiled Mathlib library into `~/lean_autoformalization` (about 10 GB). Each Lean check takes 10–30 seconds. When you are done, you can delete that folder and `~/.elan` to free the space.

## Further reading
- [Agentic Design Patterns](https://aman.ai/primers/agentic-design-patterns/) (Aman Chadha), [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024), [Self-Refine](https://arxiv.org/abs/2303.17651) (Madaan et al., 2023) and [Self-Consistency](https://arxiv.org/abs/2203.11171) (Wang et al., 2023)
- [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (nostalgebraist, 2020), [Probing Classifiers](https://arxiv.org/abs/2102.12452) (Belinkov, 2022), [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) (Heimersheim and Nanda, 2024), [Interpretability in the Wild](https://arxiv.org/abs/2211.00593) (Wang et al., 2022) and [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702) (Lanham et al., 2023)
- [Activation Oracles](https://arxiv.org/abs/2512.15674) (Karvonen et al., 2025), [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) (Anthropic, 2026) and [Emergent Introspective Awareness in Large Language Models](https://arxiv.org/abs/2601.01828) (Lindsey, 2026)
- [MASA: LLM-Driven Multi-Agent Systems for Autoformalization](https://aclanthology.org/2025.emnlp-demos.44/) (Zhang, Valentino and Freitas, 2025) and its [tutorial notebook](https://github.com/lanzhang128/multi_agent_autoformalization/blob/main/tutorial-demo-autoformalization.ipynb), [Autoformalization with Large Language Models](https://arxiv.org/abs/2205.12615) (Wu et al., 2022), the [Natural Number Game](https://adam.math.hhu.de/#/g/leanprover-community/nng4) and [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/)

Please give us feedback for this tutorial!

![LS Feedback 11](../QRs/qrf11.png)
