# Execution repair scope implementation

1. Add a shared execution-method validator with explicit method metadata and AST-region checks. Write and observe failing acceptance/rejection tests first, then implement each supported edit region.
2. Register navigation action source targets; version the restricted profile and dispatch the same execution validator from both proposal preparation and runtime installation. Preserve the parser validator and all existing protected targets.
3. Extend Resume's typed bounded iteration to existing child skills, allowing only zeroing transient counters and clearing previous-running flags. Exercise actual control objects and reject arguments/goal/physical-state changes.
4. Enable the existing parent/worker transaction route only for declared execution targets; add RPC tests for same-instance installation, failure rollback, source restoration and forbidden worker manifests.
5. Update the model-facing edit policy and runtime context to describe the actual parent and worker boundaries without changing the general prompts or response schema. Update operator documentation.
6. Review the combined guard and transaction changes independently. Run focused tests, the recovery suite and full available discovery; report pre-existing missing dependencies separately. Verify original production source methods are restored after integration checks.

Review focus: invariant motion-call locations; completion predicates cannot become constant true or inflate thresholds; forbidden manifest/profile downgrade; no physical transition during installation; nested Resume loops remain bounded and cannot mutate model choices.

Completed: shared guard, both validation boundaries, bounded Resume cleanup, restricted worker transactions, model-facing metadata and documentation. Twenty new cases pass; full discovery executes 509 passing cases with no case failures and 30 pre-existing import errors (pytest/mock unavailable). Independent review rejected 83 attack probes. Hydra and fake-transport context checks pass; no live model request or Habitat episode was run. Evidence: `analysis_results/recovery/execution_scope_20261028/verification.json`.
