# Decisions

## Native model tool calls

Use one public `ModelClient.run_with_tools(prompt, tools, max_rounds)` entry point with provider-neutral `ToolSpec` values. Provider adapters own their native Chat Completions, Responses, or LiteLLM continuation mechanics; the runner supplies Python callbacks and receives one normalized `ModelResult` containing the final answer, aggregate usage, raw provider-call audit records, and tool traces.

This keeps provider message formats out of the experiment runner while preserving actual function calls. A stateful public session API was rejected because it exposes lifecycle details and is easy to misuse. A public turn-by-turn transcript API was rejected because it makes Responses continuation state awkward and pushes provider translation into callers. Existing `generate(prompt)` remains unchanged for injected answers and grading.
