# Remove Model Environment Selection

**Requirement:** The user explicitly rejects `EMOS_LLM_MODEL`. The task model must come from `habitat_baselines.llm_model`, defaulting to `gpt-5.6-sol`. A leftover environment variable must have no effect, and the implementation must not use it as an internal configuration bridge.

**Design:** Register a concrete default string. `MultiLLMPolicy.from_config` retains that string and passes it explicitly to `group_discussion` (leader/robot models) and `CrabAgent.init_agent` (planning/execution model). Remove the entry-point environment context. Keep standalone explicit `OpenAIModel(model=...)` and its default. The stability worker passes its existing configured model through the same Hydra option; its experiment settings remain unchanged.

**Constraints:** Preserve the current uncommitted trajectory work, remain in this checkout, and do not run simulations, API calls, dependency installation or commits. Appended optional arguments preserve existing callers. Keep temperature/seed behavior unchanged.

- [x] Model owner: reproduce environment-ignore assertions failing, then remove environment lookup; six tests pass.
- [x] Root: reproduce CLI environment mutation/default failures and missing stability override. Replace the bridge with explicit configuration arguments.
- [x] Root: validate real Hydra/main/policy/discussion/agent paths using seven captured requests covering leader, robot reflections, planning and execution. Twenty-four focused model/CLI/stability tests pass. Existing `group_discussion(save_chat_history=False)` has an unrelated local-directory initialization bug; the integrated fixture uses its normal enabled-history path with a temporary directory rather than changing that behavior here.
- [x] Root: related regression, actual CLI config composition, documentation and independent review.

**Evidence:** `/tmp/emos-step-trajectory-20261025/no-env-{cli-red,worker-red,model-green}.log`. CLI RED: five failures; worker RED: missing model override. Model owner: three failing environment-selection assertions before removal. No runtime `EMOS_LLM_MODEL` references remain in `habitat-baselines`, `habitat-mas` or `scripts`; tests retain the old variable only to prove it is ignored.

**Final verification:** 115 related model/CLI/trajectory/stability/recovery/config tests passed in 5.342 seconds (`no-env-regression.log`). The actual HSSD `--cfg job` command with a deliberately conflicting legacy environment value still resolved `llm_model: gpt-5.6-sol` (`no-env-default-hssd.yaml`). Independent scoped review passed with no Important findings. Runtime-source search confirmed zero legacy variable references; `git diff --check` passed. No real API calls or simulations were started.
