# Apprentice

Teaching a small model to fix real bugs by letting a big one think out loud first.

## The idea

Beam ([reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B)) is a very good engineer that never stops thinking. Ask it to fix a bug and it reasons before every move: what the issue is really asking, which file to open next, which test proves the fix. That reasoning is the most valuable thing it produces, and most of it is thrown away once the answer comes back.

Apprentice keeps it. We put Beam to work on real bug reports from real open-source Python projects (fastapi, rich, requests, httpx), inside a sandbox where it can read code, run commands, edit files and submit a patch. Then we run the project's own hidden tests against that patch. If the tests pass, the whole session gets saved: every thought, every tool call, every dead end it backed out of. If they fail, we throw it away.

What's left is a set of worked examples where the reasoning is known to lead somewhere correct. We use those to fine-tune a 4B Gemma model, so a laptop-sized model can pick up some of the habits of a 500B one. Beam's Apache-2.0 license allows exactly this kind of distillation.

So there are three things here, built in this order:

1. **The gym.** A sandboxed environment for coding agents on real repository tasks. Patches are checked by hidden tests, and edits to test files are stripped out before checking, so an agent can't pass by rewriting the exam.
2. **The traces.** Beam's sessions in that gym, with only the test-passing ones kept as training data.
3. **The apprentice.** A small Gemma model fine-tuned on those traces, compared against its untrained self and against Beam on tasks neither of them saw during training.

## Why bother

Most of what people have measured about Beam so far is how well it answers questions. We wanted to see whether its way of working on a codebase can be taught. A coding assistant that runs offline on ordinary hardware is useful to a lot of people, and the cheapest route there might be a strong open model acting as a teacher.

The other reason is honesty. A patch either passes the tests or it doesn't. That makes this a good place to check whether reasoning actually helps, instead of trusting it because it sounds convincing.

## Working with Beam's quirks

Beam has two habits that break naive agent loops, and the client in this repo handles both:

- Its thinking counts against the output budget. At high effort it can spend the whole budget thinking and return an empty answer. We detect that, retry with more room, and never treat an empty reply as a real one.
- In multi-turn conversations it expects its earlier reasoning to be sent back exactly as it was. Drop it and the next request fails. We keep it verbatim.

We also keep a spending ledger. This project runs on about $25 of inference credit, so every call is priced before it goes out and the run stops cleanly when the cap is reached.

## Status

Early and moving fast. The gym and the Beam baseline come first, then the trace dataset, then the fine-tuned model and a demo Space where you can watch Beam and the apprentice work through the same bug side by side. Results will land here as they come in, sample sizes included, along with anything that didn't work.

## Running it

Setup instructions are coming once the sandbox is stable. You'll need Docker, Python 3.12, [uv](https://docs.astral.sh/uv/), a Hugging Face token with inference credit, and a local copy of the task file and repository snapshots.

## License

Apache-2.0
