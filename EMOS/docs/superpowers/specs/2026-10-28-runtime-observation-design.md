# Runtime observation diagnostics

The user approved adding the logging items discussed in this conversation and connecting them to the existing heuristic and LLM detectors. This change records current execution evidence without changing EMOS planning, physical execution, termination policy, tool Success placeholders, rewards, or task success definitions.

## Scope and approach

Extend the existing V3 recorder and its readers. Reuse returned observations, existing controller calculations, and read-only robot transforms. Do not add another simulator step, sensor-suite evaluation, skill termination evaluation, or live PDDL cache evaluation for logging. Missing or stale evidence remains null with an explicit reason. Old recordings retain their original semantics.

The alternatives were to continue estimating from base positions (known ambiguity near the native thresholds), or change the planner's feedback (a behavioral intervention). The approved scope is richer observation and detector consumption; it preserves the execution baseline. Full camera imagery on every step is unnecessary: retain existing decision images and add available camera/config metadata without rendering new images.

## Shared data contract

All new state is detached, finite JSON. `state.current_goal` contains `env_step`, `timing: post_env_step`, `source`, `any_at` and `is_detected` maps. Keys are stable PDDL entity names, including `any_targets|0` and `TARGET_any_targets|0`. Values are bool or null. Any-at evidence uses the task's native `robot.base_transformation.inverted().transform_point(target_position)` norm and configured threshold, never ground-base distance. Detection evidence uses the already returned detected-object observations and semantic image pixel counts, with the configured strict pixel threshold. No historical stage value is substituted for a current predicate.

Each agent can carry `goal_coverage` with `env_step`, timing, reference, distance_threshold, and `targets` mapping entity names to distance/at_goal; `detection` retains the existing available/eligible/sensor/pixel_threshold/detected_object_ids/target_detections/target_pixel_counts schema with timing, step and missing reasons. Sensor and target semantic-ID metadata are recorded where available. Root/base poses remain distinguishable. Optional camera metadata is read without advancing sensors.

Navigation action producers publish detached `_navigation_diagnostics` using already computed variables: step, pre_control timing, active/available, requested target and proxy point, path_found, distance/angle/thresholds and satisfaction flags, controller command and skill_done. A nonselected controller or old-step diagnostic cannot appear as fresh active execution. Diagnostic collection failures do not change original control results or errors.

The recorder links an action's controller termination to its last actual returned observation (`action_outcome`), preserving both signal step and observation step. It also links the next actual model request and the placeholder messages it contains to the preceding action outcome (`execution_feedback`). Neither event edits model messages or labels controller termination as task success. Existing decision/action IDs, communication queue/consume events, source configuration and exceptions remain intact.

Readers map current native predicates to metrics `current_goal.any_at.<entity>` and `current_goal.is_detected.<entity>`, carry the source/timing information and per-agent diagnostics through normalization, and expose them to the LLM whitelist. The heuristic prefers explicitly recorded native current booleans, including null (unknown), over its old approximation; legacy recordings without the new fields retain their behavior. Timings and step numbers must prevent stale or future evidence from entering a current frame.

## Verification

Use failing tests first. Cover native reference offsets and threshold edges, instantaneous detection loss, unavailable sensors, real target-ID mapping, no live cache/sensor/control side effects, fresh versus stale navigation, action/outcome/request correlation, unchanged Success/request content, and capture-to-V3-to-reader-to-detector integration. Verify existing trajectory, detection, recovery and model tests. No paid LLM request or real simulation is needed for this implementation.
