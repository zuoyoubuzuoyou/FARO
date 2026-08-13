# Unclosed LLM Code Fence Recovery Design

## Context

`OpenAIModel.chat()` asks the planning model to return executable Python in a fenced Markdown block. Local Qwen can emit an opening ` ```python ` fence without a closing fence. The current `_extract_code()` implementation assumes that every opening fence has a matching closing fence and reads past the end of the response, raising `IndexError` and terminating the evaluation episode.

The observed malformed response also contains the agent's final decision, such as `{{no||...}}`, after the Python source. Blindly treating the entire remainder as Python would therefore produce invalid code.

## Chosen Behavior

Keep the existing `_extract_code()` return type and make extraction tolerant of an unclosed final block:

1. Extract normally closed code blocks exactly as before.
2. If a block is not closed, look for the first standalone final-decision line beginning with `{{yes` or `{{no` after the opening fence.
3. Treat that decision line as the end of the candidate code. If no decision line exists, treat the end of the response as the candidate end.
4. For a Python block, compile the candidate without executing it. Accept the recovered block only when compilation succeeds.
5. If recovery is unsafe or invalid, ignore only that malformed block and return any previously completed blocks. Never raise an indexing error.

This is a virtual repair performed during parsing; the original model response and chat history remain unchanged.

## Data Flow

`OpenAIModel.chat()` receives the model response and passes its content to `_extract_code()`. A valid or safely recovered Python block is returned to the existing interpreter path. The interpreter result is then sent back to the model as it is today. If no block can be recovered, `chat()` follows its existing non-code branch and returns the raw response so downstream decision parsing can still inspect `{{yes}}` or `{{no||...}}`.

## Error Handling

- Empty or missing response content yields no code blocks.
- An unmatched opening fence never indexes beyond the response.
- An unclosed Python block containing invalid Python is not executed.
- A warning identifies an ignored malformed block without terminating the episode.
- Non-Python unclosed blocks are not repaired because this path cannot validate their syntax safely.

## Tests

Add focused unit tests for `_extract_code()` covering:

- a normally closed Python block;
- an unclosed valid Python block ending at end-of-response;
- an unclosed valid Python block followed by `{{no||...}}`;
- an unclosed invalid Python block, which is ignored without raising;
- empty or missing content.

The regression test based on the observed Qwen shape must fail with the current implementation because it raises `IndexError`, then pass after the recovery logic is implemented.

## Scope

The change is limited to code-block extraction and its tests. It does not change prompts, model settings, task assignment, interpreter behavior, or the user's existing `multi_llm_policy.py` modification.
