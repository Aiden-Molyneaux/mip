"""Phase 4: register a new project into the multi-project registry.

Sketch:
  uv run python scripts/register_project.py \
      --name <slug> \
      --repo <local path or git URL> \
      --collection <qdrant collection> \
      [--system-prompt <path>] \
      [--default-model <model>]

  - clone if URL, otherwise validate local path
  - insert into projects table (state/sqlite.py)
  - ensure Qdrant collection exists
  - kick off scripts/bootstrap.py for the initial index

Implement when Phase 4 lands.
"""
