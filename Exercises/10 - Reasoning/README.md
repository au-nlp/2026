# 10 - Reasoning

This tutorial is about getting language models to solve problems that need more than one pass of text generation. You start with the **Game of 24**, a puzzle whose answers are easy to check but hard to find, and compare **few-shot prompting** and **chain-of-thought** with two frameworks that let the model explore: **ReAct**, where an environment checks every step, and **Tree of Thoughts**, where the model proposes and evaluates partial solutions inside a breadth-first search. You then turn a model into an **agent** with native **function calling**: you describe tools with JSON schemas, execute the calls the model asks for, and run the loop until the task is done, with tools for exchange rates, the weather, symbolic math and the Game of 24. Finally, you see what happens when tools fail, or return text written by an attacker.

## Notebooks
- `0. Reasoning Frameworks.ipynb`: the main exercise notebook. You write an exact verifier and a brute-force solver for the Game of 24, prompt a hosted LLM with few-shot and chain-of-thought prompts, and build ReAct and Tree of Thoughts (BFS) around it. No GPU needed, but you need your free Groq account from week 7.
- `0. Reasoning Frameworks - Solution.ipynb`: the worked solutions. Try to finish a part before looking at them.
- `1. Agents and Tools.ipynb`: the follow-up notebook. You write tool schemas, execute tool calls, build an agent loop, and connect it to free public APIs and to the Game of 24. No GPU needed; it uses the Groq API and two public APIs that need no account.
- `1. Agents and Tools - Solution.ipynb`: the worked solutions for the agents notebook.

## What you will practice
In `0. Reasoning Frameworks.ipynb`:
1. **The Game of 24**: exact checking with fractions, states and steps, and depth-first search
2. **Baselines**: few-shot (IO) and chain-of-thought prompts, and extracting the answer from a reply
3. **ReAct**: thoughts, actions and observations, with your code as the environment
4. **Tree of Thoughts**: a thought generator, a state evaluator, and breadth-first search
5. **Comparison**: accuracy against tokens, and where reasoning models fit in
6. **MCQs**: on reasoning frameworks

In `1. Agents and Tools.ipynb`:
1. **Tool schemas**: describing a Python function with JSON Schema, and what the model sees of it
2. **Tool calls**: executing them safely, and sending back results and errors
3. **The agent loop**: several and parallel tool calls, and a step limit
4. **Real tools**: exchange rates, the weather and symbolic math
5. **Back to the Game of 24**: an agent that checks its own answers
6. **When tools go wrong**: errors, plausible but wrong results, and indirect prompt injection
7. **MCQs**: on agents and tool use

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks run in the project environment (`uv sync`) and need no new packages. Both call the **Groq API**, like `1. Prompting.ipynb` in week 7. Have your API key ready; no payment details are needed. Never paste your API key into a notebook cell.

`0. Reasoning Frameworks.ipynb` reads the puzzles from `dataset_game24.csv.gz` in this folder. Part 1 runs entirely on your computer. Tree of Thoughts makes many requests per puzzle, so the experiments use roughly 110,000 tokens of the free daily budget (as of October 2026: 200,000 tokens per day for the model we use) and take about 15 minutes because of the per-minute limit. Start the long cells and read ahead while they run. Replies are cached in `groq_cache.json`, so re-running a cell costs nothing.

`1. Agents and Tools.ipynb` also calls [Frankfurter](https://frankfurter.dev) (exchange rates) and [Open-Meteo](https://open-meteo.com) (weather), which are free and need no account. It uses roughly 40,000 tokens. Both notebooks together fit into the free daily budget, but without much room to spare: if you want to raise the sizes of the experiments, do it on another day.

## Further reading
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei et al., 2022), [Self-Consistency](https://arxiv.org/abs/2203.11171) (Wang et al., 2023), [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2023) and [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al., 2023)
- [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023), [Graph of Thoughts](https://arxiv.org/abs/2308.09687) (Besta et al., 2024), [Language Agent Tree Search](https://arxiv.org/abs/2310.04406) (Zhou et al., 2024) and [Fleet of Agents](https://arxiv.org/abs/2405.06691) (Klein et al., 2024)
- [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., 2023), [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024), [tool use](https://console.groq.com/docs/tool-use) in the Groq documentation, and the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction)
- [Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) (Greshake et al., 2023)

Please give us feedback for this tutorial!

![LS Feedback 10](../QRs/qrf10.png)
