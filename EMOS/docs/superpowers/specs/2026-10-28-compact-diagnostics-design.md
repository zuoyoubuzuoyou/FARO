# Lossless diagnostic trajectory compaction

The user approved the previously proposed reduction: store static configuration once, preserve changing values each step, reference unchanged values, and retain complete diagnostic evidence at action/decision boundaries. The 4000-step episode61 recorded at `trajectory/2026-10-28/perception_diagnostics/run_20261028_020031` is the acceptance dataset: its steps file is 79,918,613 bytes. No simulation, model request, lossy sampling or source-data overwrite is authorized or necessary.

## Approach

Improve the existing V3 storage layer, not the execution/diagnostic collectors. Reuse content-addressed payload references for static and repeated state subtrees. Keep small current values and step numbers inline; avoid making unique per-frame wrapper objects into thousands of payload files. Reference metrics already present on the same row rather than copying them into state. Deduplicate the common event envelope against the same row's explicit identity, preserving directly inspectable event_type/agent_id/sequence and unique event IDs, timestamps and ordering. Expanded events, states, actual observations, model requests, diagnostics, unknowns and summaries must be exactly equal to the originals.

Considered alternatives were deleting diagnostic fields (loses the evidence just requested), only gzipping the file (small but does not address redundant representation), and new chained delta snapshots (adds seek/read complexity and risks quadratic history reads). Selected approach is bounded, self-contained V3 references plus content-addressed repeated blocks, using the current reader interfaces.

## Compatibility and safety

- Retain one committed row per environment step and the V3 identity envelope; new files advertise an encoding feature if required.
- Existing V1/V2/V3 files and escaped literal `$trace` data keep their meaning. New references require exact shape, valid row context and identity checks; no arbitrary path access or future-step reconstruction.
- `iter_trace_events`, `_Codec.decode` and `read_trace_summary` fully expand references. Recorder callbacks stay expanded and synchronous at the original times; pending evidence remains independently readable before a commit.
- Small records remain simple; repeated camera configuration and unchanged diagnostic dictionaries are shared. Unique timestamps, frame steps, IDs and actual sensor changes remain distinct.
- Bound codec caches if needed, and keep corruption/missing reference validation. Never retain an unbounded episode history merely to compress a new row.
- A read-only-input compaction command writes to a fresh output directory, copies required config/media and referenced payloads safely, verifies expanded events/summary and source hashes, and reports steps-file and whole-directory sizes. Original 76.22MiB recording remains intact; no source file is deleted.

## Acceptance

Meaningful RED/GREEN tests cover repeated/static state, changed values, arbitrary literal control-shaped JSON, event identity reconstruction, partial pending and truncated commit, corrupt/missing/cyclic references, summary reconstruction and unchanged detector/recorder consumption. Repack all 4000 real rows offline, compare every expanded event and normalized frame and the final summary, and measure total size including payloads. Count files so a smaller JSONL cannot hide an explosion in payload metadata. Report exact bytes and reductions, verification limits and output path.
