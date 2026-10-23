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

## Gym fixes found by watching a small model fail

The first student run exposed two ways `edit_file` rejected edits that were right in substance:

- **Wrong leading whitespace.** The agent quotes the correct lines but miscounts the indentation.
- **Copied JSON escapes.** Tool results reach the model as JSON, so file text shows up as `\n` and `\t`. Smaller models copy those characters literally into `old_string`.

On the strict tool, Beam failed 8 of 22 edits on held-out tasks and the 4B student failed 51 of 54. Since gym v1.2, `edit_file` falls back to a unique line-by-line match that ignores surrounding whitespace, then to a match after decoding escapes. It re-indents the replacement to fit. Full disclosure: these fixes came from studying held-out failures, so every model was re-run on v1.2 and the numbers below all come from v1.2.

Beam's held-out solve rate across three runs at `low` (v1.0, v1.1, v1.2) was 18, 16 and 17 of 27. Treat Beam as roughly 63% on this set, plus or minus about a task.

## Students on held-out tasks (gym v1.2, n = 27)

| model | solved | mean turns | ran out of turns |
|---|---|---|---|
| Beam 501B, `low` | 17 (63%) | 22.1 | 1 |
| Gemma 4 E4B, untuned | 5 (18.5%) | 11.6 | 0 |
| Gemma 4 E4B + LoRA on 53 Beam sessions (v1) | 0 | 34.4 | 17 |

Both Gemma models were served the same way: Q8_0 GGUF through llama-server, a 64k context, and thinking enabled and kept in history.

**The first fine-tune made things worse.** Student v1 picked up Beam's way of working: grep, read, edit, run tests. It did not pick up the judgement that makes that pay off. It explored for longer, kept retrying, sometimes dug through `git log` for a fix that does not exist in the snapshot, and ran out of turns on 17 of 27 tasks. The untuned model is blunt. It submits within about 12 turns, and on the easier tasks that is enough.

Likely reasons, and what the next run changes:

- **Most training windows were mid-exploration.** Long sessions were cut into 8k-token windows, and only about 29% of windows ended with a submit. The 12B run uses 16k windows.
- **Too little data for a 4B model to learn when to stop.** There were 53 sessions. The next run adds two more Beam passes over the training tasks.
- **Capacity.** The next student is Gemma 4 12B, compared against its own untuned baseline.
