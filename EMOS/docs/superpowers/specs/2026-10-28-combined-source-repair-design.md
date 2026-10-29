# Combined detection and source repair

The user requests one LLM interaction containing current observations and original runtime source: determine whether a system-level execution problem exists, then output repaired source for that problem. They explicitly selected validation followed by automatic installation and in-place resume. Keep the standalone boolean detector and legacy extension recovery compatible.

## Architecture

- In source mode with the default LLM monitor, a sampled full-window check sends one combined request. It includes task/constraints, current and recent observations, plans/history, actual exception or execution feedback, current readable source with SHA256 and class/process metadata, and the permitted edit boundary. A previous model verdict must not be supplied as the answer to a fresh check.
- Negative response: `{"system_error": false}`. Positive response carries `system_error=true`, structured method patches (`path`, `target`, `base_sha256`, complete `code`) and an entrypoint. No separate detector request or second repair request follows a positive combined response.
- A detected problem without a safe repair may return empty patches and a null entrypoint; record it without installation rather than forcing a fabricated repair or negative prediction.
- Separate querying from proposal application. Reuse original-file transactions, parent/worker installation, loaded-class checks, Resume on the existing policy, progress assessments, rollback and cleanup.
- Keep sample/window/check cadence. Collect observations during cooldown/budget exhaustion but do not request an unusable repair. Normal windows create no source version and start no cooldown. Failed or positive repair attempts are bounded by the existing repair iteration budget.
- A real policy exception before the next simulator step may trigger the same combined call immediately, including the real step-0 observation. A negative/unrepairable response returns immediately; never loop forever because no version was created. Retry only through the existing checkpoint and same-decision boundary.

## Prompt and code contract

Share the general monitoring policy with the standalone detector, while using a distinct combined output contract. The repair prompt describes execution/scheduling/argument/error-handling defects generally, without an episode-specific control recipe. Preserve task meaning, success/reward definitions, capability limits, genuine observations, model identity/history and existing physical state. Do not generate an answer/goal shortcut or introduce extra simulation steps, resets or teleports.

Readable source is not permission to edit every visible method. Automatic source proposals use a restricted validation profile in both preparation and runtime installation: protect environment/task/simulator lifecycle and observation/evaluation producers; reject new forbidden references including aliases/computed calls, protected-state writes, extra stepping/teleports, dynamic execution, external IO and unbounded recovery loops. Preserve unchanged trusted calls only where the restricted profile can justify them; otherwise reject the candidate. The source review must describe actual guard coverage honestly: static checks are not a general arbitrary-Python security sandbox or a proof of semantic correctness.

## Compatibility and verification

Existing injected detectors/generators and team-progress source mode retain their explicit routes; the default source+LLM path combines calls. Existing rollback and consistency behavior remains fail-closed. The source snapshot refreshes after every installed revision. Recovery remains disabled unless explicitly enabled in evaluation configuration.

Use fake network transports with real source transactions and policy/model objects. Test normal/no-mutation/one-call behavior, positive installation from that exact response, safe inability to repair, malformed/error responses, refreshed sources, exception step0, finite retries, cadence/cooldown/budget/terminal handling, physical/history/object identity preservation, worker validation and rollback. Add adversarial guard tests for scoring/goals/observations/physical-state changes, extra steps, callable aliases, dynamic calls and infinite loops. No live API call, real Habitat run, credentials change or commit is part of this implementation task.
