# Method source input implementation

1. TDD generic source extraction and adapter selection rules, full method boundaries/decorators, full-file hashes and compact source indexes.
2. TDD exact runtime/agent/static-field deduplication with a lossless local expansion check.
3. Add an idempotent model-view preparer at the source generator request boundary; explain method views and input_layout in general repair instructions.
4. Archive full combined inputs locally before requests, record their references in audits, and retain original expanded contexts for Resume and legacy integrations.
5. Exercise the real generator request/response path with fake transport, including first/second views, negative/failure audits, and unchanged patch validation.
6. Independently review selection and fact preservation. Run relevant and full available tests, validate Hydra defaults, and export an offline before/after comparison from the saved real request.

Completed: method views, exact deduplication, request-boundary integration, full gzip audit and compatibility checks. Offline saved input: 510,421 to 323,406 characters (-36.64%), 54 complete methods, all full-file hashes and runtime facts verified. New tests: 36 passing; full discovery: 547 executed passing cases, no case failures, 30 existing pytest/mock import errors. Independent selection and integration reviews found no blockers for the one-attempt production path. No live model request or Habitat episode. Evidence: `analysis_results/recovery/method_input_20261028/`.
