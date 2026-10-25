# Source recovery execution plan

Implement the user's explicitly requested completion in the current workspace, preserving all earlier recovery work.

1. Source patch transaction (parallel): failing tests, then original-file method edits, hashes, compilation, original-class live binding, added Resume, super closure and transaction rollback; artifacts include source/diff/manifests.
2. Live runtime (parallel): failing tests, then worker patch RPC, coordinated parent/worker installation, logical checkpoints and bounded policy-only retries in the original evaluator. Do not retry env.step.
3. Generator/controller (root): failing tests, then fixed source-edit prompt, refreshed source snapshots, original-source revision controller, exception-triggered regeneration and budgets. Source mode becomes the primary enabled mode; extension compatibility is explicit.
4. Integration: test generated Resume on a class with no recovery helper, later patch an original failing method, verify same simulator/held-object/history, run existing recovery/detection/trajectory regressions, independent review and usage documentation.

Shared protocol: proposal `{patches:[{path,target,base_sha256,code}],entrypoint:{path,target},explanation}`. `SourcePatchManager.prepare` returns a transaction with `.manifest(process)`, `.commit_files()`, `.rollback_files()`. `RuntimeSourcePatcher.install/rollback` preserve original class identity. `SourceRuntime.apply(transaction,policy,context)` coordinates installations and invokes the generated entrypoint. `SourceRecoveryController.handle_execution_error(frame,policy,error)` can regenerate without advancing the observation clock. `OnlineRecovery.run_policy` owns logical checkpoint/restore and retries only policy calls.

Review focus: no unused added Resume, actual original-method changes on existing objects, no cached-source prompts, worker rollback without process death, logical state after partial agent failure, no duplicate physical transitions, concurrent source conflict handling, method descriptors/closures, terminal isolation and accurate evidence claims.
