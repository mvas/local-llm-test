### Step 4: Manual prompt smoke test

Use a small fixed prompt pack of 10 to 20 prompts that reflect real usage:

- explain an unfamiliar code snippet
- write a small utility function
- refactor a function for readability
- diagnose a failing test from error text
- summarize a long technical document
- generate a shell command from a task description

Record:

- usefulness
- hallucination rate
- verbosity fit
- whether responses are fast enough to feel interactive

This matters because benchmark leaders are not always the best day-to-day assistants.

## Suggested Prompt Packs

Create three local prompt packs and keep them fixed for every run.

### Pack A: General assistant

Use 20 to 30 prompts:

- summarization
- factual QA
- planning
- extraction
- instruction following
- math and logic

### Pack B: Coding assistant

Use 20 to 30 prompts:

- write functions
- explain code
- repair bugs from stack traces
- generate tests
- refactor code
- regex and shell tasks

### Pack C: Agent tasks

Use 10 to 20 tasks:

- inspect repo structure
- identify bug source from logs
- patch a broken function
- update tests after a spec change
- trace a failing command

Score each item with a simple rubric:

- `0` = failed
- `1` = partly useful
- `2` = correct and useful

This gives a lightweight benchmark that reflects your real workflows better than public suites alone.