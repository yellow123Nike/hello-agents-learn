"""Neo4j 图存储实现：创建 database/集合、增删改查、检索（按标签 + 文本/属性过滤）。"""

import json
from typing import Any, Dict, List, Optional

from hello_agents.core.storage.base_store import BaseStore


class Neo4jStore(BaseStore):
    """
    基于 Neo4j 的存储：database 对应 Neo4j 的 database 名，collection 对应节点标签（如 Document_xxx）。
    每个文档存为一个节点，属性：id, content, metadata（JSON 字符串）。
    检索为标签 + 文本 CONTAINS 或属性过滤。
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    def create_database(self, **kwargs: Any) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("请安装 neo4j: pip install neo4j")
        uri = kwargs.get("uri") or self.uri
        user = kwargs.get("user") or self.user
        password = kwargs.get("password") or self.password
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        # 可选：创建 database（Neo4j 4+ 多库）
        db_name = kwargs.get("database") or self.database
        self.database = db_name

    def _ensure_driver(self) -> Any:
        if self._driver is None:
            self.create_database()
        return self._driver

    def _label(self, collection: str) -> str:
        """将 collection 名转为合法标签（去掉非法字符）。"""
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in collection)
        return f"Document_{safe}" if safe else "Document"

    def create_collection(self, name: str, **kwargs: Any) -> None:
        driver = self._ensure_driver()
        label = self._label(name)
        with driver.session(database=self.database) as session:
            try:
                # Neo4j 5.x
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)"
                )
            except Exception:
                try:
                    # Neo4j 4.x
                    session.run(f"CREATE INDEX ON :{label}(id)")
                except Exception:
                    pass

    def _add_impl(
        self,
        collection: str,
        doc: Dict[str, Any],
        *,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        import uuid
        driver = self._ensure_driver()
        label = self._label(collection)
        id_ = doc_id or doc.get("id") or str(uuid.uuid4())
        content = doc.get("content", "")
        meta = doc.get("metadata")
        metadata_str = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "{}"
        with driver.session(database=self.database) as session:
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n.content = $content, n.metadata = $metadata",
                id=id_,
                content=content,
                metadata=metadata_str,
            )
        return id_

    def get(self, collection: str, id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        driver = self._ensure_driver()
        label = self._label(collection)
        with driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{label} {{id: $id}}) RETURN n.content AS content, n.metadata AS metadata",
                id=id,
            )
            record = result.single()
        if record is None:
            return None
        meta = record["metadata"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return {
            "id": id,
            "content": record["content"] or "",
            "metadata": meta or {},
        }

    def _update_impl(
        self,
        collection: str,
        id: str,
        doc: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        driver = self._ensure_driver()
        label = self._label(collection)
        content = doc.get("content", "")
        meta = doc.get("metadata")
        metadata_str = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "{}"
        with driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{label} {{id: $id}}) SET n.content = $content, n.metadata = $metadata RETURN n",
                id=id,
                content=content,
                metadata=metadata_str,
            )
            if result.single() is None:
                return False
        return True

    def delete(
        self,
        collection: str,
        id: Optional[str] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> int:
        driver = self._ensure_driver()
        label = self._label(collection)
        to_delete: List[str] = []
        if id is not None:
            to_delete.append(id)
        if ids:
            to_delete.extend(ids)
        if not to_delete:
            return 0
        with driver.session(database=self.database) as session:
            session.run(
                f"MATCH (n:{label}) WHERE n.id IN $ids DELETE n",
                ids=to_delete,
            )
        return len(to_delete)

    def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        driver = self._ensure_driver()
        label = self._label(collection)
        # 简单按 content CONTAINS 检索；metadata_filter 用 metadata JSON 包含键值（需实现可再扩展）
        with driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{label}) WHERE n.content CONTAINS $q RETURN n.id AS id, n.content AS content, n.metadata AS metadata LIMIT $limit",
                q=query,
                limit=limit * 2 if metadata_filter else limit,
            )
            rows = list(result)
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = r["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if metadata_filter and meta:
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            out.append({
                "id": r["id"],
                "content": r["content"] or "",
                "metadata": meta or {},
            })
            if len(out) >= limit:
                break
        return out
