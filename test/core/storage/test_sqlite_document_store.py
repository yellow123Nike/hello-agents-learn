# SQLiteDocumentStore tests
import pytest
from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.llm.message import Message
from hello_agents.core.storage import DocumentModel, SQLiteDocumentStore


def test_test_store(tmp_path):
    db_path = str(tmp_path / "test_storage.db")
    s = SQLiteDocumentStore(db_path=db_path)
    s.create_database()
    yield s

def test_create_database_and_collection(store):
    store.create_collection("docs")
    id_ = store.add("docs", {"content": "hello", "metadata": {}})
    assert id_
    doc = store.get("docs", id_)
    assert doc is not None
    assert doc["content"] == "hello"

def test_add_returns_id(store):
    store.create_collection("c1")
    id1 = store.add("c1", {"content": "a", "metadata": {"k": "v"}})
    assert isinstance(id1, str) and len(id1) > 0
    id2 = store.add("c1", {"content": "b"}, doc_id="custom-id")
    assert id2 == "custom-id"

def test_get_existing(store):
    store.create_collection("c1")
    id_ = store.add("c1", {"content": "content", "metadata": {"type": "test"}})
    doc = store.get("c1", id_)
    assert doc is not None
    assert doc["id"] == id_
    assert doc["content"] == "content"
    assert doc["metadata"] == {"type": "test"}

def test_get_missing_returns_none(store):
    store.create_collection("c1")
    assert store.get("c1", "non-existent-id") is None

def test_update_existing(store):
    store.create_collection("c1")
    id_ = store.add("c1", {"content": "old", "metadata": {}})
    ok = store.update("c1", id_, {"content": "new", "metadata": {"v": 1}})
    assert ok is True
    doc = store.get("c1", id_)
    assert doc["content"] == "new"
    assert doc["metadata"] == {"v": 1}

def test_update_missing_returns_false(store):
    store.create_collection("c1")
    ok = store.update("c1", "no-such-id", {"content": "x", "metadata": {}})
    assert ok is False

def test_delete_by_id(store):
    store.create_collection("c1")
    id_ = store.add("c1", {"content": "x", "metadata": {}})
    n = store.delete("c1", id=id_)
    assert n == 1
    assert store.get("c1", id_) is None

def test_delete_by_ids(store):
    store.create_collection("c1")
    id1 = store.add("c1", {"content": "a", "metadata": {}})
    id2 = store.add("c1", {"content": "b", "metadata": {}})
    n = store.delete("c1", ids=[id1, id2])
    assert n == 2
    assert store.get("c1", id1) is None
    assert store.get("c1", id2) is None

def test_search_by_content(store):
    store.create_collection("c1")
    store.add("c1", {"content": "user likes short answers", "metadata": {"type": "pref"}})
    store.add("c1", {"content": "other content", "metadata": {}})
    results = store.search("c1", "short", limit=5)
    assert len(results) >= 1
    assert any("short" in d["content"] for d in results)

def test_retrieve_by_ids(store):
    store.create_collection("c1")
    id1 = store.add("c1", {"content": "a", "metadata": {}})
    id2 = store.add("c1", {"content": "b", "metadata": {}})
    docs = store.retrieve_by_ids("c1", [id1, id2])
    assert len(docs) == 2
    ids = {d["id"] for d in docs}
    assert id1 in ids and id2 in ids

def test_add_many(store):
    store.create_collection("c1")
    docs = [{"content": "first", "metadata": {}}, {"content": "second", "metadata": {"i": 2}}]
    ids = store.add_many("c1", docs)
    assert len(ids) == 2
    assert store.get("c1", ids[0])["content"] == "first"
    assert store.get("c1", ids[1])["content"] == "second"


def test_add_and_get_as_message(store):
    """文档形态传入 Message，存后再用 get_as_message 取回为 Message。"""
    store.create_collection("c1")
    msg = Message.user_message("用户说：你好")
    id_ = store.add("c1", msg)
    assert id_
    got = store.get_as_message("c1", id_)
    assert got is not None
    assert got.content == "用户说：你好"
    assert got.role == RoleType.USER


def test_add_and_get_as_document(store):
    """文档形态传入 DocumentModel，用 get_as_document 取回为 DocumentModel。"""
    store.create_collection("c1")
    doc = DocumentModel(id="doc-1", content="存储正文", metadata={"role": "user"})
    id_ = store.add("c1", doc)
    assert id_ == "doc-1"
    got = store.get_as_document("c1", id_)
    assert got is not None
    assert got.id == "doc-1"
    assert got.content == "存储正文"
    assert got.metadata.get("role") == "user"
