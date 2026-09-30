---
name: contextos
description: Use ContextOS to manage persistent agent context, execute 6-pass token allocations, recall durable engineering decisions, and record session traces.
---

# ContextOS Agent Workflow

ContextOS provides deterministic, budget-bounded context management for autonomous AI workflows.

## CLI Quick Access

When working in this repository:

```bash
# Resume latest branch state, uncommitted changes, and active decisions
./bin/ctx resume -repo .

# Plan minimum-sufficient context under a strict token budget (e.g., 2000 tokens)
./bin/ctx plan -repo . -task "Fix issue" -budget 2000

# Remember key decisions or bug fixes
./bin/ctx remember -repo . -kind decision -authority user -content "Decision..."

# View allocation traces and cache token savings
./bin/ctx stats -repo .
```
