# Structured general detection and recovery prompts

## Intent and scope

Apply the user's supplied teacher feedback: fixed general instructions for multi-agent embodied execution, followed by structured, per-run problem, trajectory and code inputs. No benchmark names or task-specific repair recipes in the instructions. Detection considers a possible error; triggered recovery checks the reported problem and supplies updated code.

This is a prompt-text and explanatory-template change. Preserve existing JSON paths, binary/patch output contracts, model/runtime APIs, source permissions and trajectory selection. Preserve all pre-existing worktree changes. No model calls, simulator runs, commits or new dependencies.

## Approach

The user's requested general-instruction/template structure is fixed. For the trajectory choice, retain the existing recent-window filter and lossless source-input deduplication as the implementation default. Document its defaults, current-state exception path and lack of a total token budget. Full-history input is unnecessarily long; first-only input misses late errors; new event retrieval needs an experiment before replacing the baseline. These are implementation judgments, not a claim the user selected an optimal filter.

Use existing JSON fields as the template destinations: `initial_context`/`task` for problem; `recent_frames`/`runtime` for trajectory; `code` for source and edit policy; plans, history, interface, real errors and previous recovery results in their existing positions. Document placeholders explicitly without adding duplicate payload sections or a second renderer.

## Changes and verification

Revise only prompt constants in detection, source recovery and compatible extension recovery; bump the detector prompt version. Shared monitoring rules must accept source recovery's current frame and actual exception, not only standalone `recent_frames`. Add a complete Chinese template/input-selection guide and link it from current detection and recovery docs.

Run existing transport-fake detection/recovery tests. Compare Python ASTs excluding prompt-only assignments and reconstruct representative requests before/after to verify unchanged user payload and request options. Independently review task neutrality, detection assumptions, causal evidence, field mapping and documentation accuracy. These checks establish compatibility, not live-model accuracy.

## Follow-up: one instruction for detection and code recovery

The user explicitly requests one prompt rather than separately worded detection and recovery prompts. Replace the default source prompt's concatenated roles and repeated introductions with one general instruction covering judgment and conditional code updates. Present this unified instruction and its structured inputs first in the template guide.

The same source instruction is shared with the older `generate()` interface; only its output-format adapter differs because its existing callers require patches, entrypoint and explanation. Preserve both parsers, controller routing, input fields, runtime boundaries and the main three-outcome response union. Independent offline detection and extension mode remain existing compatibility interfaces, not extra stages in the default online flow. Observed system errors need not have a known code cause; uncertain or disallowed repairs must still use the positive-without-patches outcome.
