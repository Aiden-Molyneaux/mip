"""Qdrant collection management.

One collection per project. Payload schema (set by `bootstrap.py`):
  text          str   the chunk body
  path          str   repo-relative POSIX path
  kind          str   "code" | "doc" | "schema"
  lang          str   "ts" | "tsx" | "md" | "sql"
  module        str         server module name or "components"/"pages"/"lib" (optional)
  section       str         markdown H2 heading (optional)
  symbols       list[str]   top-level export names extracted from TS/TSX files (optional)
  migration     int         migration number for SQL (optional)
  chunk_index   int         sub-chunk index when a chunk was split by oversized-splitter (optional)
  line_start    int
  line_end      int
"""
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from mip.config import settings

EMBED_DIM = 768  # nomic-embed-text v1.5

_INDEXED_FIELDS = ("path", "filename", "kind", "lang", "module")


def get_client() -> QdrantClient:
    return QdrantClient(url=settings().qdrant_url)


def ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    for field_name in _INDEXED_FIELDS:
        try:
            client.create_payload_index(name, field_name=field_name, field_schema="keyword")
        except UnexpectedResponse:
            pass


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    points: list[tuple[dict, list[float], str]],
) -> None:
    """points: list of (payload, vector, point_id_uuid)."""
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=pid, vector=vec, payload=payload) for payload, vec, pid in points],
    )


def delete_by_path(client: QdrantClient, collection: str, path: str) -> None:
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="path", match=MatchValue(value=path))]
        ),
    )


def search(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int = 10,
    filter_kind: str | None = None,
) -> list[dict]:
    flt: Filter | None = None
    if filter_kind:
        flt = Filter(must=[FieldCondition(key="kind", match=MatchValue(value=filter_kind))])
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        query_filter=flt,
    )
    return [{"score": r.score, "payload": r.payload or {}} for r in response.points]


def search_by_filename(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    filename: str,
    limit: int = 3,
) -> list[dict]:
    """Search restricted to chunks whose `filename` payload equals `filename`.
    Used when the user's query literally mentions a file name like 'routes.ts'
    — pure embedding similarity often fails to surface the canonical files,
    but an exact match on basename always finds them. The vector still orders
    within the filename-matched subset so we get the most relevant chunk
    when multiple files share the basename (every module has its own routes.ts).
    """
    flt = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))])
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        query_filter=flt,
    )
    return [{"score": r.score, "payload": r.payload or {}} for r in response.points]
