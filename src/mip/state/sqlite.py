"""Phase 3-4: project registry + last-indexed-sha tracking in a small SQLite DB.

Schema sketch:
  projects(name TEXT PRIMARY KEY, repo_path TEXT NOT NULL, qdrant_collection TEXT NOT NULL,
           system_prompt_path TEXT, default_model TEXT, created_at TEXT)
  sync_state(project_name TEXT PRIMARY KEY REFERENCES projects(name),
             last_indexed_sha TEXT, last_sync_at TEXT, last_error TEXT)

Use stdlib `sqlite3`. Path comes from `settings().mip_state_db`.
Implement when Phase 4 (multi-project) lands.
"""
