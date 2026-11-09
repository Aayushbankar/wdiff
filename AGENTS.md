# AGENTS.md - Agent Operating Instructions for wdiff

## Worklog Protocol (MANDATORY)

**Rule: After each milestone completion, the worklog MUST be APPENDED with a new entry. The worklog is NEVER replaced or edited — only appended to.**

### Worklog Format
Each worklog entry MUST be appended to `.agent/worklog/wrk_<timestamp>_<id>.yaml` with:

```yaml
work_record:
  id: wrk_<timestamp>_<id>
  timestamp: '<ISO8601>'
  intent:
    task: "<milestone description>"
    phase: "<phase name>"
    persona: <persona>
    model: <model used>
    complexity_triggers: []
  execution:
    steps: [...]
    ...
  artifacts:
    - { type: file, path: <path>, verified: true }
    ...
  learned:
    patterns: [...]
    dead_ends: [...]
    promoted_to_skill_kb: []
  next:
    - "<next steps>"
  verification:
    claims:
      - claim: "<claim>"
        evidence: "<evidence>"
        method: <STATIC|RUNTIME>
        status: VERIFIED
```

### Key Rules
1. **ALWAYS APPEND** - Never overwrite or replace existing worklog entries
2. **One entry per milestone** - Each major milestone gets its own worklog entry
3. **Include verification** - Every entry must have verification claims with evidence
4. **Timestamp format** - Use ISO8601 UTC timestamps
4. **Reference prior work** - Link to previous worklog entries when relevant

## Project-Specific Rules

### wdiff Development Rules
- **Memory bound**: All streaming operations must stay under 500MB peak RAM
- **Testing**: All new features must have tests; `pytest tests/ -q` must pass
- **Linting**: `ruff check src/ tests/` and `ruff format --check src/ tests/` must pass
- **No external network dependencies** in HTML reports (self-contained)

### Verification Gates
Every phase must produce a verification record with:
- **STATIC** evidence (file existence, structure)
- **RUNTIME** evidence (test passes, CLI output)
- **Claims** with explicit evidence and method