# Boolean LLM detection response

The user requests that the model return only whether a system error exists. The model wire response is exactly `{"system_error": true}` or `{"system_error": false}`. Generic monitoring policy and existing observation input remain in place.

## Contract

Accept one JSON object with exactly one `system_error` boolean field. Reject coercions, null, extra fields and the previous rich status schema. False means no system error detected in the current window, not proof of normal operation; insufficient observations must not manufacture an error. Request, parse and budget failures remain incomplete predictions without a false boolean.

Normalize valid replies to `system_error` plus an internal `status` for existing consumers. The program adds its observed `step` and detector identity; it does not fabricate agents, reason, evidence or confidence. Preserve input sanitation, sampling, first-alarm behavior, raw response auditing, usage reporting and recovery behavior. Bump the prompt version to isolate caches and retain historical data.

## Verification

Boolean positive/negative tests must fail against the old parser before implementation. Retain causal input and transport/cache/error tests, replace obsolete rich-output validation tests, and exercise runner scoring and real recovery consumers with minimal alarms. No new model requests, simulator runs or credential changes are needed.
