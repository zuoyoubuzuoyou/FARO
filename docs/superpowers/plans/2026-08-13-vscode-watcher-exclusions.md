# VS Code Watcher Exclusions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a workspace configuration that stops VS Code from watching the two large dataset directories.

**Architecture:** A single workspace-scoped `.vscode/settings.json` file defines two `files.watcherExclude` glob entries. No visibility, search, application runtime, or system-wide settings are changed.

**Tech Stack:** VS Code workspace settings, JSON

## Global Constraints

- Exclude only `partnr-planner/data` and `EMOS/data` from file watching.
- Keep both directories visible, searchable, readable, and editable.
- Do not modify project runtime configuration.

---

### Task 1: Add and verify workspace watcher exclusions

**Files:**
- Create: `.vscode/settings.json`
- Reference: `docs/superpowers/specs/2026-08-13-vscode-watcher-exclusions-design.md`

**Interfaces:**
- Consumes: VS Code's `files.watcherExclude` workspace setting.
- Produces: Two boolean glob mappings that disable watching beneath the selected directories.

- [ ] **Step 1: Verify the configuration does not already exist**

Run:

```bash
test ! -e .vscode/settings.json
```

Expected: exit status 0.

- [ ] **Step 2: Create the minimal workspace configuration**

Create `.vscode/settings.json` with exactly:

```json
{
  "files.watcherExclude": {
    "**/partnr-planner/data/**": true,
    "**/EMOS/data/**": true
  }
}
```

- [ ] **Step 3: Validate JSON syntax and exact values**

Run:

```bash
python3 -m json.tool .vscode/settings.json
python3 -c 'import json; p=".vscode/settings.json"; d=json.load(open(p)); assert d == {"files.watcherExclude": {"**/partnr-planner/data/**": True, "**/EMOS/data/**": True}}'
```

Expected: formatted JSON output followed by exit status 0.

- [ ] **Step 4: Check the final diff**

Run:

```bash
git diff --check
git diff -- .vscode/settings.json
```

Expected: no whitespace errors; the diff contains only the new watcher exclusions.

- [ ] **Step 5: Commit the configuration**

```bash
git add .vscode/settings.json docs/superpowers/plans/2026-08-13-vscode-watcher-exclusions.md
git commit -m "chore: exclude dataset directories from VS Code watcher"
```
