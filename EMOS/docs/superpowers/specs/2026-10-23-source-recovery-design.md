# Original source recovery and live Resume

The user explicitly requested completing the original proposal after learning that the previous implementation only evolved a recovery extension. This revision must generate and apply edits to original execution, model-call, control and simulation methods. A newly generated Resume method must actually be invoked on the existing policy instance. Subsequent revisions can modify original methods and add error handling; no predefined command list constitutes the source implementation.

## Mechanism

The LLM returns structured method patches: repository-relative path, actual `Class.method`, exact current file SHA256 and complete method code. A patch can replace an existing method with its existing signature or add a method. It also selects a generated parent-policy method `(self, context)` to invoke after installation. Source snapshots are refreshed for each revision and include original complete files, hashes, process roles and available class names. The fixed prompt retains task, planner plans, recent ten full turns, diagnostics, current state, previous source revisions and measured outcomes.

The patch engine validates the complete batch before mutation, compiles changed files and methods, preserves class identities and binds methods using existing module globals. New method bodies become effective on the next call; no running Python frame is retroactively changed. Therefore a fixed host edit in the evaluator provides a safe boundary and bounded retry around policy action generation. `env.step` is outside this retry and is never repeated after an uncertain simulator failure.

Parent and worker methods are installed in their existing processes. Worker RPC catches patch errors instead of terminating the VectorEnv worker. Installation and original-file writes form a rollback transaction: late failures restore earlier files and method descriptors. Hash checks prevent overwriting concurrent user edits. Original source before/after, unified diffs, raw generation and validation/application outcomes remain as versioned artifacts. Normal evaluation cleanup restores run-local patches by default; explicit persistence retains source changes for later runs.

The first policy exception can recover from the captured initial state. Before an attempt, a logical checkpoint captures existing policies, skill counters, history and recurrent tensors without copying clients or simulator objects. Failed attempts restore these logical changes before generation/retry. Resume preserves physical pose, held objects, episode identity, initialized agents and models. New attempts and repeated exceptions share finite revision/retry budgets.

## Boundaries and verification

Only allowlisted method targets in original runtime files are editable. Recovery coordinator/validator, success metrics, imports, class definitions and destructive lifecycle functions are excluded. Static checks and transactional runtime handling are safeguards, not a general security sandbox or proof of physical safety. A source patch cannot revive a dead process or retroactively undo an uncertain physics transition. Terminal frames already reset by VectorEnv do not receive stale recovery actions.

Acceptance tests must show original file contents and live method behavior change; a policy lacking Resume receives a generated method that resumes the same instance; a second revision repairs another original method; parent/worker partial failure rolls back; zero-argument super retains the original class; and a failed decision is retried without a second reset or duplicate env.step. Existing extension mode remains explicitly selectable for compatibility. No real recovery success rate is claimed without a real experiment.
