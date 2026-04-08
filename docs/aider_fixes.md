# Aider Local Fixes

This file documents the local Aider patch made while debugging benchmark runs against local models.

The goals were:

- prevent a trailing background summarization thread from outliving a benchmark exercise
- raise the benchmark path's minimum chat-history budget for local models that otherwise default too low
- preserve enough detail to reproduce the patch later or turn it into an upstream PR

## Problem Summary

Observed behavior during Aider benchmark runs:

- benchmark output was produced successfully for a model
- at the very end of `run.stdout.log`, Aider printed:
  - `Summarization failed for model ...: cannot schedule new futures after shutdown`
  - `summarizer unexpectedly failed for all models`
- some models still exited cleanly, but slower ones could remain alive long enough for the outer benchmark timeout to fire

Working theory:

- Aider starts chat-history summarization in a background thread
- that thread is normally joined before the next prompt is built
- but the benchmark path did not obviously force a final join before the exercise returned
- this could leave a trailing summarization attempt running during teardown

## Change: Force Final Summarization Join In Local Aider

File changed:

- `benchmark/benchmark.py`

Change made:

- wrapped the main body of `run_test_real()` in a `try/finally`
- added a final call to:
  - `coder.summarize_end()`
- kept the cleanup defensive with a small exception guard

Intent:

- ensure any background summarization thread is joined before the benchmark exercise exits
- avoid leaving trailing summarization work alive after results are already written

Why this location:

- the benchmark path creates one fresh `Coder` per exercise
- earlier summarization is normally joined before future prompts
- the suspicious gap was at exercise shutdown, where `run_test_real()` could return without an obvious final `summarize_end()`

## Patch Sketch

Diff-style summary:

```diff
--- a/benchmark/benchmark.py
+++ b/benchmark/benchmark.py
@@
-    coder.show_announcements()
-    coder.get_file_mentions = lambda x: set()
-    ...
-    return results
+    try:
+        coder.show_announcements()
+        coder.get_file_mentions = lambda x: set()
+        ...
+        return results
+    finally:
+        try:
+            coder.summarize_end()
+        except Exception:
+            if verbose:
+                print("Failed to finalize chat summarization")
+                traceback.print_exc()
```

Minimal conceptual change:

- wrap the main body of `run_test_real()` in `try/finally`
- call `coder.summarize_end()` during cleanup
- keep cleanup logging defensive rather than failing the benchmark on cleanup-only issues

## Why This Change Was Chosen

This patch is aimed at the most plausible failure mode:

- summarization normally gets joined before later prompts
- but the benchmark exercise could return without an obvious final join
- that leaves room for a trailing summarization attempt to survive into teardown

## Change: Raise Benchmark Minimum Chat History Tokens

File changed:

- `benchmark/benchmark.py`

Change made:

- added a benchmark-local constant:
  - `MIN_CHAT_HISTORY_TOKENS = 4096`
- after constructing `main_model`, clamp:
  - `main_model.max_chat_history_tokens = max(main_model.max_chat_history_tokens, MIN_CHAT_HISTORY_TOKENS)`

Intent:

- keep benchmark chat-history summarization from starting too aggressively for local models with missing or weak metadata
- avoid the benchmark path inheriting Aider's default floor of `1024` when the real serving context is substantially larger

Why this location:

- the benchmark CLI does not expose Aider's regular `--max-chat-history-tokens` option
- `run_test_real()` is where the benchmark creates the `Model` used for each exercise
- changing `main_model.max_chat_history_tokens` there affects the default `ChatSummary` created by `Coder`

Patch sketch:

```diff
--- a/benchmark/benchmark.py
+++ b/benchmark/benchmark.py
@@
+MIN_CHAT_HISTORY_TOKENS = 4096
@@
     main_model = models.Model(...)
+    main_model.max_chat_history_tokens = max(
+        main_model.max_chat_history_tokens,
+        MIN_CHAT_HISTORY_TOKENS,
+    )
```

Minimal conceptual change:

- keep benchmark CLI behavior unchanged externally
- raise only the benchmark path's minimum summarization threshold
- leave models with larger native values unchanged

Why this change was chosen:

- the benchmark was using model names like `openai/<gguf-slug>` that may not have precise model metadata
- in those cases, Aider could fall back to `1024` chat-history tokens
- a `4096` floor is a safer local default for the larger-context GGUF models being benchmarked here

## Reproduce From Scratch

If rebuilding this setup later, re-apply:

1. In `aider/benchmark/benchmark.py`
   - ensure `run_test_real()` always calls `coder.summarize_end()` in a `finally` block before returning
2. In `aider/benchmark/benchmark.py`
   - define `MIN_CHAT_HISTORY_TOKENS = 4096`
   - after `main_model = models.Model(...)`, set `main_model.max_chat_history_tokens = max(main_model.max_chat_history_tokens, MIN_CHAT_HISTORY_TOKENS)`

## Suggested Validation

After reapplying the patches, run a short Aider benchmark and check:

- the end of `aider/run.stdout.log` no longer shows the summarization failure
- the outer benchmark no longer marks the run failed due to a timeout that happens after benchmark results were already printed
- `main_model.max_chat_history_tokens` in `aider/run.stdout.log` is at least `4096` for local GGUF-backed benchmark models that previously resolved to `1024`

## Notes For A Possible Aider PR

The `benchmark.py` lifecycle fix is the more plausible upstream candidate:

- small change
- benchmark-scoped
- low conceptual risk
- directly addresses the possibility of a trailing summarization thread surviving past exercise completion

Possible PR framing:

- "Ensure benchmark exercises finalize chat summarization before exit"
- "Join pending summarization thread in `run_test_real()` cleanup"

## Local Scope

This change was applied locally only:

- `benchmark/benchmark.py`

It is not assumed to exist in a fresh checkout unless re-applied.

## Change: Deterministic exercise order (`--test-seed`)

**Problem:** Aider `benchmark.py` builds a sorted list of exercise paths, then calls `random.shuffle` before slicing to `--num-tests`. With no seed, limited runs (`--aider-limit` / `--num-tests N`) get a **different subset and order every run**.

**Change (in `benchmark/benchmark.py`):**

- Added optional CLI flag `--test-seed N`.
- If `--test-seed` is set: `random.Random(N).shuffle(test_dnames)` (reproducible).
- If omitted: `random.shuffle(test_dnames)` (previous behavior).

**Integration in `local-llm`:**

- `benchmark_performance.py` exposes `--aider-test-seed`; when set, `suite_aider.py` appends `--test-seed` to the container command.
- `run-meta.txt` records `aider_test_seed=<N>` when the flag is set (empty when omitted).

**Docs:** See [README.md](../README.md) (Aider notes and tuning section).

**Suggested validation:** Run twice with the same `--aider-limit`, `--aider-test-seed`, and model config; compare exercise order lines in `raw/<model>/aider/run.stdout.log` (should match).
