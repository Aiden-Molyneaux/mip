"""Phase 3: parse `git log <sha>..HEAD --name-status` into added/modified/deleted lists.

Sketch:
  diff_since(repo_path: Path, sha: str) -> dict[str, list[Path]]:
    - subprocess.run(["git", "-C", str(repo_path), "log", f"{sha}..HEAD", "--name-status", "--no-renames"])
    - parse lines starting with A/M/D, group by status
    - return {"added": [...], "modified": [...], "deleted": [...]}

Implement after Phase 2.
"""
