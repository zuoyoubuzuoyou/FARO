# Readable Trajectory Directories

**Requirement:** Replace UUID-looking trajectory folder names with recognizable run times and episode IDs, preserving repeated runs and trajectory data. Apply the change to future recordings and the user's completed run `trajectory/2026-10-26/perception/73d2e999-97f0-4fe7-915b-83f19aab6170`.

**Design:** New runs use `run_YYYYMMDD_HHMMSS` in Asia/Shanghai, with `_02`, `_03` collision suffixes allocated using atomic mkdir. Episode ID 2 uses `episodes/episode_0002`; repeated or sanitized-colliding IDs use `_attempt_02`, `_attempt_03`. Internal UUID identities remain unchanged. Directory depth and episode-relative references remain unchanged; config's episodes[].directory records actual names.

**Migration:** Derive the completed run's time from its recorded config timestamp (`2026-10-25T17:58:31.858339+00:00` -> `20261026_015831` in Asia/Shanghai). Rename the specified run and episode only, update config's relative directory index, preserve a reversal mapping, and verify hashes/expanded events/summary before and after. Search external manifests for references before changing paths. Other recorded runs remain untouched.

**Constraints:** Keep previous uncommitted changes. No API calls, simulations, dependency installs, commits or irreversible deletion. An existing result is eligible only when its episode summary is complete and its files remain unchanged during preflight. Existing payload/media/step contents must remain byte-identical.

- [x] Recorder owner: failing tests for readable names, time-zone formatting, repeated/parallel collisions and safe episode IDs; implement minimal allocation change and verify.
- [x] Root: inspect consumers and reference paths, document format, run relevant regression and independent review.
- [x] Root: migrate the specified completed run with a reversible mapping and verify preserved content and reader output; report the new path.

**Preflight evidence:** Specified run has 109 files, 178 completed steps, status completed, total 5,965,156 bytes. References use paths relative to the episode directory; consumers use metadata identities and glob patterns, rather than parsing UUID directory names.

**Naming evidence:** Four new tests reproduced UUID directory names before implementation. Numeric IDs longer than 255 characters separately reproduced an unsafe length regression; normalized numeric labels now retain the existing 80-character cap. A focused set of 144 trajectory/detection/recovery/stability tests passed (`readable-directories-regression.log`). Scoped naming review passed with no Important findings. External JSON/JSONL/CSV/YAML path-reference search in trajectory and analysis_results found no references to the specified run path.

**Migration preflight finding:** Unified expansion of this real recording failed before any path mutation: step 1 contains a blob-encoded canonical replay_context and local references into nested model_memory/model_config. The decoder traversed encoded containers directly. A focused compatible reader repair is required before migration verification; it must decode the canonical root first, traverse decoded data and verify the target hash without rewriting stored content. This is a verified existing reader bug, independent of naming. Storage owner assigned reproducing tests and minimal repair; no user trajectory files have changed during the failed preflight.

**Resolved finding:** Two RED regressions reproduced the nested-reference issue for event local references and summary step references. The reader now decodes the allowlisted canonical root once before walking the remaining path, preserving hash/cycle validation and escaped literal data. All 21 storage tests and the final 146-test relevant suite passed; independent scoped reviews of naming and the reader fix found no Important issues.

**Completed migration:** The specified run is now `trajectory/2026-10-26/perception/run_20261026_015831/episodes/episode_0002/`. Only config's episode directory index changed; all other 108 original files are byte-identical. Expanded events (1,867), normalized frames (178) and summary compare equal before/after moving. Internal run/evaluation UUIDs remain unchanged. `directory_rename.json` in the new run records old/new paths and original config hash for reversal. Other existing runs were not moved. Evidence: `/tmp/emos-step-trajectory-20261025/readable-directories-migration.log` and `readable-directories-regression-final.log`. No new simulation/API requests were started.
