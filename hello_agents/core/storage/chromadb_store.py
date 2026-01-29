"""ChromaDB 向量存储实现：创建 database/集合、增删改查、向量检索。

"""

import uuid
from typing import Any, Dict, List, Optional

from hello_agents.core.llm.message import Message
from hello_agents.core.storage.base_store import BaseStore, message_to_doc
from hello_agents.core.storage.document_model import DocumentModel


class ChromaDBStore(BaseStore):
    """
    基于 ChromaDB 的存储：持久化目录为 database，collection 对应 Chroma 的 collection。
    检索为向量相似度（使用 Chroma 默认 embedding 或传入的 embedding_function）。

    使用本地 vLLM（类 OpenAI）embedding 示例：
        from hello_agents.core.llm.llm_schema import LLMParams
        from hello_agents.core.llm.other_llm_model import LLMClient_OpenAI, VLLMChromaEmbeddingFunction
        from hello_agents.core.storage import ChromaDBStore

        params = LLMParams(model_name="your-embed-model", api_key=".", base_url="http://vllm-host:8000/v1", ...)
        llm = LLMClient_OpenAI(params, "openai")
        emb_fn = VLLMChromaEmbeddingFunction(llm, dimensions=1024)
        store = ChromaDBStore(persist_directory="./chroma_db", embedding_function=emb_fn)
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_function: Any = None,
    ):
        """
        Args:
            persist_directory: 持久化目录，即「database」路径
            embedding_function: 用于将 text 转为向量的可调用对象；ChromaDB 会调用 __call__(input: List[str]) -> List[List[float]]。
                               None 则使用 Chroma 默认；本地 vLLM 可用 hello_agents.core.llm.other_llm_model.VLLMChromaEmbeddingFunction。
        """
        self.persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._client = None
        self._collections: Dict[str, Any] = {}

    def create_database(self, path: Optional[str] = None, **kwargs: Any) -> None:
        import chromadb
        from chromadb.config import Settings

        path = path or kwargs.get("path") or self.persist_directory
        self.persist_directory = path
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        if self._embedding_function is not None:
            self._client._embedding_function = self._embedding_function

    def _ensure_client(self) -> Any:
        if self._client is None:
            self.create_database()
        return self._client

    def create_collection(self, name: str, **kwargs: Any) -> None:
        client = self._ensure_client()
        if name not in self._collections:
            self._collections[name] = client.get_or_create_collection(
                name=name,
                embedding_function=self._embedding_function,
                metadata=kwargs.get("metadata"),
            )

    def _get_coll(self, collection: str) -> Any:
        self.create_collection(collection)
        return self._collections[collection]

    def _add_impl(
        self,
        collection: str,
        doc: Dict[str, Any],
        *,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        coll = self._get_coll(collection)
        content = doc.get("content", "")
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        metadata_flat = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        id_ = doc_id or doc.get("id") or str(uuid.uuid4())
        coll.add(
            ids=[id_],
            documents=[content],
            metadatas=[metadata_flat],
        )
        return id_

    def add_many(
        self,
        collection: str,
        docs: List[Any],
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        normalized = []
        for i, d in enumerate(docs):
            if isinstance(d, Message):
                normalized.append(message_to_doc(d, doc_id=ids[i] if ids and i < len(ids) else None))
            elif isinstance(d, DocumentModel):
                normalized.append(d.to_dict())
            else:
                normalized.append(d)
        coll = self._get_coll(collection)
        doc_ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for i, doc in enumerate(normalized):
            content = doc.get("content", "")
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            metadata_flat = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            id_ = (ids[i] if ids and i < len(ids) else None) or doc.get("id") or str(uuid.uuid4())
            doc_ids.append(id_)
            documents.append(content)
            metadatas.append(metadata_flat)
        coll.add(ids=doc_ids, documents=documents, metadatas=metadatas)
        return doc_ids

    def get(self, collection: str, id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        coll = self._get_coll(collection)
        try:
            r = coll.get(ids=[id], include=["documents", "metadatas"])
            if not r["ids"]:
                return None
            return {
                "id": r["ids"][0],
                "content": r["documents"][0] if r["documents"] else "",
                "metadata": r["metadatas"][0] if r["metadatas"] else {},
            }
        except Exception:
            return None

    def _update_impl(
        self,
        collection: str,
        id: str,
        doc: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        coll = self._get_coll(collection)
        try:
            content = doc.get("content", "")
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            metadata_flat = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            coll.update(ids=[id], documents=[content], metadatas=[metadata_flat])
            return True
        except Exception:
            return False

    def delete(
        self,
        collection: str,
        id: Optional[str] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> int:
        coll = self._get_coll(collection)
        to_delete = []
        if id is not None:
            to_delete.append(id)
        if ids:
            to_delete.extend(ids)
        if not to_delete:
            return 0
        coll.delete(ids=to_delete)
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
        coll = self._get_coll(collection)
        where = metadata_filter  # Chroma 支持 where 条件
        try:
            r = coll.query(
                query_texts=[query],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            r = coll.query(
                query_texts=[query],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        out: List[Dict[str, Any]] = []
        if not r["ids"] or not r["ids"][0]:
            return out
        for i, id_ in enumerate(r["ids"][0]):
            doc = {
                "id": id_,
                "content": r["documents"][0][i] if r["documents"] and r["documents"][0] else "",
                "metadata": r["metadatas"][0][i] if r["metadatas"] and r["metadatas"][0] else {},
            }
            if r.get("distances") and r["distances"][0]:
                doc["score"] = float(r["distances"][0][i])  # 距离，越小越相似
            out.append(doc)
        return out
