# Results so far

All numbers come from the gym in this repo. Each task is a real bug report from an open-source Python project, solved in a network-less sandbox. A task counts as solved only when the project's hidden tests pass. Beam was run through the Hugging Face router with a 40-turn, 60-tool-call budget per task.

## Task set

129 tasks from fastapi, rich, requests and httpx. For 125 of them, the known-good fix flips at least one hidden test, which is our usability bar. These were split by repository into 98 training tasks and 27 held-out tasks (`splits/v1.json`). The held-out tasks are never used for training.

## Beam on held-out tasks (n = 27)

| reasoning effort | solved | rate | $ / task | mean turns | hit the 40-turn cap |
|---|---|---|---|---|---|
| low | 18 | 66.7% | 0.019 | 22.7 | 2 |
| medium | 18 | 66.7% | 0.033 | 32.1 | 11 |
| high | 17 | 63.0% | 0.046 | 37.1 | 16 |

By repository (low / medium / high): fastapi 9 / 9 / 7 of 14, rich 6 / 6 / 7 of 10, requests 3 / 3 / 3 of 3.

The efforts don't solve the same tasks. Low solves four fastapi tasks that high misses, and high solves three tasks (two fastapi, one rich) that low misses. Across all three runs, 21 of the 27 tasks are solved at least once.

### More thinking spends the turn budget

A higher effort did not raise the solve rate. What it changed was how long Beam worked: mean turns went from 23 to 37, and the share of tasks that ran out of turns before submitting went from 2 to 16 of 27. Earlier community studies found that higher effort can spend the *token* budget and return an empty answer. This is the agent version of that: under a fixed step budget, higher effort spends the *turn* budget. For an agent loop with a step limit, `low` was the best setting here. It was also the cheapest, at about 40% of the cost of `high`.

We ran each setting once on 27 tasks, so a one-task difference is noise. The turn-cap pattern is the robust part.

## Beam on training tasks (n = 98)

| run | attempted | solved |
|---|---|---|
| low effort | 98 | 51 (52%) |
| medium effort, only on the 47 low missed | 47 | 2 |

That gives 53 training tasks with at least one test-verified Beam session. Those 53 sessions are the fine-tuning data for the student model.

## Cost

Everything above cost about $6.50 of inference: the probe, the smoke test, 145 training episodes and 81 held-out episodes. Prompt caching does most of the work here. In a typical 20-turn session, around 90% of input tokens are served from cache.

## Untuned Gemma 4 E4B

In a one-task pilot, the untuned 4B model looked around the repo, asked twice for the problem statement even though it was in the prompt, and submitted a patch that changed nothing useful. The full held-out run for the untuned and fine-tuned students comes next.
