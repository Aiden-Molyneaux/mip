"""Phase 3: APScheduler job that polls `git fetch` and triggers incremental reindex.

Sketch:
  start_poller(repo_path, collection, interval_minutes=5):
    - APScheduler BackgroundScheduler
    - Every <interval_minutes>: `git -C <repo_path> fetch origin main`
    - diff against last_indexed_sha (state/sqlite.py)
    - For each modified/added file: delete_by_path → re-chunk → embed → upsert
    - For each deleted file: delete_by_path
    - Update last_indexed_sha
    - Surface failures via /admin/status (last_error, commits_behind)

Implement after Phase 2 retrieval is verified end-to-end.
"""
