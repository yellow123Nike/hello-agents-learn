"""SQLite 文档存储实现：创建 database/表（集合）、增删改查、全文/关键词检索。"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from hello_agents.core.storage.base_store import BaseStore


class SQLiteDocumentStore(BaseStore):
    """
    基于 SQLite 的文档存储：database 为单文件路径，collection 为表名。
    表结构：id TEXT PRIMARY KEY, content TEXT, metadata TEXT (JSON)。
    检索为 LIKE/content 包含或 FTS5（若启用）。
    """

    def __init__(self, db_path: str = "./storage.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def create_database(self, **kwargs: Any) -> None:
        path = kwargs.get("path") or self.db_path
        self.db_path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.create_database()
        assert self._conn is not None
        return self._conn

    def create_collection(self, name: str, **kwargs: Any) -> None:
        conn = self._ensure_conn()
        table = self._table_name(name)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{{}}'
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_content ON {table}(content)"
        )
        conn.commit()

    def _table_name(self, collection: str) -> str:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in collection)
        return f"doc_{safe}" if safe else "doc_default"

    def _add_impl(
        self,
        collection: str,
        doc: Dict[str, Any],
        *,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        import uuid
        conn = self._ensure_conn()
        self.create_collection(collection)
        table = self._table_name(collection)
        id_ = doc_id or doc.get("id") or str(uuid.uuid4())
        content = doc.get("content", "")
        meta = doc.get("metadata")
        metadata_str = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "{}"
        conn.execute(
            f"INSERT OR REPLACE INTO {table} (id, content, metadata) VALUES (?, ?, ?)",
            (id_, content, metadata_str),
        )
        conn.commit()
        return id_

    def get(self, collection: str, id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        conn = self._ensure_conn()
        table = self._table_name(collection)
        cur = conn.execute(
            f"SELECT id, content, metadata FROM {table} WHERE id = ?",
            (id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return {
            "id": row["id"],
            "content": row["content"] or "",
            "metadata": meta or {},
        }

    def _update_impl(
        self,
        collection: str,
        id: str,
        doc: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        conn = self._ensure_conn()
        table = self._table_name(collection)
        content = doc.get("content", "")
        meta = doc.get("metadata")
        metadata_str = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "{}"
        cur = conn.execute(
            f"UPDATE {table} SET content = ?, metadata = ? WHERE id = ?",
            (content, metadata_str, id),
        )
        conn.commit()
        return cur.rowcount > 0

    def delete(
        self,
        collection: str,
        id: Optional[str] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> int:
        conn = self._ensure_conn()
        table = self._table_name(collection)
        to_delete: List[str] = []
        if id is not None:
            to_delete.append(id)
        if ids:
            to_delete.extend(ids)
        if not to_delete:
            return 0
        placeholders = ",".join("?" * len(to_delete))
        cur = conn.execute(
            f"DELETE FROM {table} WHERE id IN ({placeholders})",
            to_delete,
        )
        conn.commit()
        return cur.rowcount

    def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_conn()
        self.create_collection(collection)
        table = self._table_name(collection)
        # LIKE 检索（简单实现；可扩展 FTS5）
        pattern = f"%{query}%"
        cur = conn.execute(
            f"SELECT id, content, metadata FROM {table} WHERE content LIKE ? LIMIT ?",
            (pattern, limit * 2 if metadata_filter else limit),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if metadata_filter and meta:
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            out.append({
                "id": row["id"],
                "content": row["content"] or "",
                "metadata": meta or {},
            })
            if len(out) >= limit:
                break
        return out
