"""
存储基类：文档形态抽象为 DocumentModel（BaseModel），与 Message 互转。
子类可实现：ChromaDB（向量）、Neo4j（图）、SQLite（文档表）等。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from hello_agents.core.llm.message import Message
from hello_agents.core.storage.document_model import DocumentModel


# -------------------------------------------------------------------------
# 兼容：Message / dict 与 DocumentModel 互转（对外保留原名）
# -------------------------------------------------------------------------

def message_to_doc(msg: Message, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """将 Message 转为存储用 dict（委托 DocumentModel.from_message）。"""
    return DocumentModel.from_message(msg, doc_id=doc_id).to_dict()


def doc_to_message(doc: Union[DocumentModel, Dict[str, Any]]) -> Message:
    """将文档（DocumentModel 或 dict）转回 Message。"""
    if isinstance(doc, DocumentModel):
        return doc.to_message()
    return DocumentModel.from_dict(doc).to_message()


# -------------------------------------------------------------------------
# 基类
# -------------------------------------------------------------------------

class BaseStore(ABC):
    """
    存储基类：创建 database、创建集合、增删改查、检索。

    约定：
    - 文档形态：抽象为 DocumentModel（Pydantic BaseModel），可与 Message 互转；内部存为 id、content、metadata。
    - database：底层存储的根（如 DB 文件、Neo4j 库、Chroma 持久化目录）。
    - collection：逻辑分组（表名 / 集合名 / 图标签等），用于隔离不同业务数据。
    """

    # -------------------------------------------------------------------------
    # 库与集合
    # -------------------------------------------------------------------------
    @abstractmethod
    def create_database(self, **kwargs: Any) -> None:
        """
        创建或连接底层 database（若已存在则复用）。
        调用后应保证后续 create_collection / add / get 等可正常执行。
        """
        pass

    @abstractmethod
    def create_collection(self, name: str, **kwargs: Any) -> None:
        """
        创建或确保集合（表/namespace/标签等）存在。
        name: 集合名称，用于 add/get/update/delete/search 的 collection 参数。
        """
        pass

    # -------------------------------------------------------------------------
    # 增删改查（入参支持 Message 或 dict）
    # -------------------------------------------------------------------------
    def _normalize_doc(
        self,
        doc: Union[Message, DocumentModel, Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> DocumentModel:
        """统一转为 DocumentModel。"""
        if isinstance(doc, Message):
            return DocumentModel.from_message(doc, doc_id=doc_id)
        if isinstance(doc, DocumentModel):
            if doc_id is not None:
                return DocumentModel(
                    id=doc_id, content=doc.content, metadata=doc.metadata, score=doc.score
                )
            return doc
        d = dict(doc)
        if doc_id is not None:
            d["id"] = doc_id
        return DocumentModel.from_dict(d)

    def add(
        self,
        collection: str,
        doc: Union[Message, DocumentModel, Dict[str, Any]],
        *,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        插入单条文档。doc 传 Message、DocumentModel 或 dict（至少含 content，可选 id、metadata）。
        Args:
            collection: 集合名
            doc: 文档，推荐 Message 或 DocumentModel；或 dict
            doc_id: 显式指定 id（覆盖 doc 中的 id）

        Returns:
            文档 id
        """
        document = self._normalize_doc(doc, doc_id=doc_id)
        return self._add_impl(
            collection, document.to_dict(), doc_id=document.id, **kwargs
        )

    @abstractmethod
    def _add_impl(
        self,
        collection: str,
        doc: Dict[str, Any],
        *,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """子类实现：插入单条文档（已为 dict 形态）。"""
        pass

    def add_many(
        self,
        collection: str,
        docs: List[Union[Message, DocumentModel, Dict[str, Any]]],
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        批量插入。doc 可为 Message、DocumentModel 或 dict；默认逐条调用 add，子类可重写为批量。
        """
        result_ids: List[str] = []
        for i, doc in enumerate(docs):
            doc_id = ids[i] if ids and i < len(ids) else None
            if isinstance(doc, dict):
                doc_id = doc_id or doc.get("id")
            result_ids.append(self.add(collection, doc, doc_id=doc_id, **kwargs))
        return result_ids

    @abstractmethod
    def get(self, collection: str, id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """按 id 获取单条文档（dict：id、content、metadata），不存在返回 None。"""
        pass

    def get_as_document(self, collection: str, id: str, **kwargs: Any) -> Optional[DocumentModel]:
        """按 id 获取单条文档并转为 DocumentModel，不存在返回 None。"""
        doc = self.get(collection, id, **kwargs)
        return DocumentModel.from_dict(doc) if doc else None

    def get_as_message(self, collection: str, id: str, **kwargs: Any) -> Optional[Message]:
        """按 id 获取单条文档并转为 Message，不存在返回 None。"""
        document = self.get_as_document(collection, id, **kwargs)
        return document.to_message() if document else None

    def update(
        self,
        collection: str,
        id: str,
        doc: Union[Message, DocumentModel, Dict[str, Any]],
        **kwargs: Any,
    ) -> bool:
        """
        按 id 更新文档。doc 传 Message、DocumentModel 或 dict（含 content、可选 metadata）。
        """
        document = self._normalize_doc(doc, doc_id=id)
        return self._update_impl(collection, id, document.to_dict(), **kwargs)

    @abstractmethod
    def _update_impl(
        self,
        collection: str,
        id: str,
        doc: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """子类实现：按 id 更新文档（已为 dict 形态）。"""
        pass

    @abstractmethod
    def delete(
        self,
        collection: str,
        id: Optional[str] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> int:
        """
        删除文档。传 id 删单条；传 ids 删多条；id 与 ids 均为 None 时可由 kwargs 指定条件删除，返回删除条数。
        """
        pass

    # -------------------------------------------------------------------------
    # 检索
    # -------------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        检索：按 query 在 collection 中查找相关文档（语义或关键词由实现决定）。
        Returns: 文档列表，每项至少含 id、content、metadata；可含 score。
        """
        pass

    def search_as_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[DocumentModel]:
        """检索并转为 DocumentModel 列表。"""
        docs = self.search(
            collection, query, limit=limit,
            metadata_filter=metadata_filter, **kwargs
        )
        return [DocumentModel.from_dict(d) for d in docs]

    def search_as_messages(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Message]:
        """检索并转为 Message 列表。"""
        documents = self.search_as_documents(
            collection, query, limit=limit,
            metadata_filter=metadata_filter, **kwargs
        )
        return [d.to_message() for d in documents]

    def retrieve_by_ids(
        self,
        collection: str,
        ids: List[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """按 id 列表批量取回文档（dict），顺序不保证，缺失的跳过。"""
        out: List[Dict[str, Any]] = []
        for id in ids:
            doc = self.get(collection, id, **kwargs)
            if doc is not None:
                out.append(doc)
        return out

    def retrieve_by_ids_as_documents(
        self,
        collection: str,
        ids: List[str],
        **kwargs: Any,
    ) -> List[DocumentModel]:
        """按 id 列表批量取回并转为 DocumentModel 列表。"""
        docs = self.retrieve_by_ids(collection, ids, **kwargs)
        return [DocumentModel.from_dict(d) for d in docs]

    def retrieve_by_ids_as_messages(
        self,
        collection: str,
        ids: List[str],
        **kwargs: Any,
    ) -> List[Message]:
        """按 id 列表批量取回并转为 Message 列表。"""
        documents = self.retrieve_by_ids_as_documents(collection, ids, **kwargs)
        return [d.to_message() for d in documents]
