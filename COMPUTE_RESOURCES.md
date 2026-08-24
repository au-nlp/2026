# Compute Resources for Natural Language Processing

This guide explains which free compute options you can use for your Natural Language Processing project at Aarhus University, and exactly how to get access to each one.

> **Disclaimer:** all quotas, limits, and information below are **as of August 2026**.

## What do I need?

| Option                                                                  | Best for                                                  | GPU?                | How you get access                | Cost                |
| ----------------------------------------------------------------------- | --------------------------------------------------------- | ------------------- | --------------------------------- | ------------------- |
| 1. UCloud (DeiC Interactive HPC)                                        | Course exercises (CPU-only)                               | No                  | Automatic with university login   | Free                |
| 2. Free GPU notebooks (Kaggle, Colab, Lightning)                        | GPU work: training and fine-tuning for your project       | Yes (T4/P100 class) | Sign up yourself, use immediately | Free                |
| 3. Free LLM APIs (AI Studio, OpenRouter, Groq, Cerebras, GitHub Models) | Projects that prompt/evaluate LLMs rather than train them | N/A (hosted)        | Create an API key yourself        | Free (rate-limited) |
| 4. Cloud credits & free VMs (Azure, AWS, Google, Oracle, Student Pack)  | Hosting a demo app or inference API                       | Restricted          | Redeem with your university email | Free credits        |

---

## 1. UCloud / DeiC Interactive HPC

UCloud is Denmark's national interactive HPC service, run by SDU, Aarhus University, and Aalborg University (~22,000 users). Includes apps like JupyterLab, Coder/VS Code, RStudio, and a terminal run directly in your browser.

### How to get access

1. Go to [cloud.sdu.dk](https://cloud.sdu.dk) and log in with your Aarhus University credentials (WAYF login).
2. On first login you get a personal **My Workspace** with a free CPU-only allocation and storage.

Docs live at [docs.cloud.sdu.dk](https://docs.cloud.sdu.dk); service info at [interactivehpc.dk](https://interactivehpc.dk) and [DeiC Interactive HPC](https://www.deic.dk/en/supercomputing-hpc/tjenester/deic-interactive-hpc).

---

## 2. Free GPU notebooks

Hosted Jupyter environments with a free GPU attached. Rule of thumb: a 16 GB T4/P100 handles fine-tuning BERT-class models and LoRA/QLoRA fine-tuning of quantized ~1–8B-parameter LLMs.

### How to get access

1. **Kaggle Notebooks**: sign up at [www.kaggle.com](https://www.kaggle.com). Free T4/P100 GPUs and TPUs: ~30 GPU-hours + ~20 TPU-hours per week, sessions up to ~12 hours, no credit card.
2. **Google Colab (free tier)**: sign in with a Google account at [colab.research.google.com](https://colab.research.google.com). Free T4, but the allocation is dynamic and unpublished (~15–30 h/week in practice).
3. **Lightning AI (free tier)**: sign up at [lightning.ai](https://lightning.ai). 15 credits/month (~22 T4-hours).

---

## 3. Free LLM APIs

If your project _uses_ LLMs (prompting, evaluation, RAG, agents) rather than training them, it's often easier to use inference providers instead of GPUs.

### How to get access

1. **Google AI Studio**: sign in at [aistudio.google.com](https://aistudio.google.com) and create a key. Free Gemini API tier with generous daily request limits.
2. **OpenRouter**: sign up at [openrouter.ai](https://openrouter.ai) for ~25 free models behind one API key.
3. **Groq**: ([console.groq.com](https://console.groq.com)) and **Cerebras** ([cloud.cerebras.ai](https://cloud.cerebras.ai)) offer free tiers serving open-weights models.
4. **GitHub Models**: free access to OpenAI and other models at [github.com/marketplace/models](https://github.com/marketplace/models). Rate limits are tied to your Copilot tier. Students get Copilot Pro free via the GitHub Student Developer Pack (see Option 4).

---

## 4. Cloud credits & free VMs

For spinning up a real VM to host an inference API or demo web app. CPU VMs work well on student credits, while GPU VMs are often restricted.

### How to get access

1. **GitHub Student Developer Pack**: claim it at [education.github.com/pack](https://education.github.com/pack) with your university email. It bundles many dev tools, and unlocks Copilot Pro.
2. **Azure for Students**: claim $100 credit at [azure.microsoft.com/free/students](https://azure.microsoft.com/free/students) with your university email.
3. **AWS Educate**: apply at [aws.amazon.com/education/awseducate](https://aws.amazon.com/education/awseducate/): up to $100 in credits in a sandboxed Starter Account.
4. **Google Cloud for Students**: see [cloud.google.com/edu/students](https://cloud.google.com/edu/students). The page offers free Google Skills credits for hands-on labs and skill badges. Additionally, any new Google Cloud account gets a standard $300 free trial.
5. **Oracle Cloud Always Free**: sign up at [www.oracle.com/cloud/free](https://www.oracle.com/cloud/free/). Permanent free ARM VM allowance.

---

## Rules & etiquette

- **Never pay for compute for this course.** Every service above has a paid tier and will happily upsell you. The free tiers are sufficient for all exercises and projects, and under no circumstances do we think it necessary for you to spend your own money. If you feel you've hit a wall that only money can solve, talk to us first.
- **Protect yourself from accidental billing.** Some signups ask for a credit card for identity verification (Oracle and the Google Cloud free trial). If you redeem any cloud credits, set a billing budget + alert on day one, and shut down VMs when not in use. It's your own responsibility to stay within the free credits. Credits also expire so redeem them when you actually start working towards your project.
- **Stop your jobs when you finish working.** Idle sessions burn your core-hours (UCloud) and GPU quota (notebooks).
- **Plan around free-tier limits.** Free tiers are rate-limited and sessions can disconnect (Colab after ~90 min idle) so save checkpoints often, and don't leave your only training run to the night before the deadline.
- **No sensitive or personal data on free API tiers**, assume anything you send may be logged or used for training.
