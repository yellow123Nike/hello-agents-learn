# ChromaDBStore tests
import os
import pytest

pytest.importorskip("chromadb")

from hello_agents.core.llm.llm_schema import LLMParams, RoleType
from hello_agents.core.llm.message import Message
from hello_agents.core.llm.other_llm_model import LLMClient_OpenAI, VLLMChromaEmbeddingFunction
from hello_agents.core.storage import ChromaDBStore, DocumentModel


@pytest.fixture
def store(tmp_path):
    """使用 ChromaDB 默认 embedding（无需外部 API）。"""
    persist_dir = str(tmp_path / "chroma_test")
    s = ChromaDBStore(persist_directory=persist_dir)
    s.create_database()
    yield s


def test_store_creation(tmp_path):
    """单独测试：创建 store 并建库，可被 pytest 收集并执行。"""
    persist_dir = str(tmp_path / "chroma_test")
    s = ChromaDBStore(persist_directory=persist_dir)
    s.create_database()
    s.create_collection("docs")
    id_ = s.add("docs", {"content": "hello", "metadata": {}})
    assert id_
    doc = s.get("docs", id_)
    assert doc is not None
    assert doc["content"] == "hello"


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


def test_search_by_query(store):
    """向量检索：插入几条文档，用相似 query 搜索。"""
    store.create_collection("c1")
    store.add("c1", {"content": "user likes short answers", "metadata": {"type": "pref"}})
    store.add("c1", {"content": "other content here", "metadata": {}})
    results = store.search("c1", "short answers", limit=5)
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
    docs = [
        {"content": "first", "metadata": {}},
        {"content": "second", "metadata": {"i": 2}},
    ]
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


# ---------- 可选：使用本地 vLLM embedding 的测试（需配置环境变量才运行） ----------
def _vllm_embedding_store(tmp_path):
    """当 EMBEDDING_BASE_URL 等已配置时，返回使用 VLLMChromaEmbeddingFunction 的 store。"""
    base_url = os.getenv("EMBEDDING_BASE_URL","http://192.168.88.235:18001/v1") 
    if not base_url:
        return None
    params = LLMParams(
        model_name=os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B"),
        api_key=os.getenv("LLM_API_KEY", "sk-"),
        base_url=base_url,
        temperature=0.0,
        max_tokens=8024,
        is_claude=False,
    )
    llm = LLMClient_OpenAI(params, "openai")
    emb_fn = VLLMChromaEmbeddingFunction(llm, dimensions=1024)
    persist_dir = str(tmp_path / "chroma_vllm_test")
    s = ChromaDBStore(persist_directory=persist_dir, embedding_function=emb_fn)
    s.create_database()
    return s


@pytest.fixture
def store_vllm(tmp_path):
    """可选：使用本地 vLLM embedding；未配置 EMBEDDING_BASE_URL/LLM_BASE_URL 时跳过。"""
    s = _vllm_embedding_store(tmp_path)
    if s is None:
        pytest.skip("未设置 EMBEDDING_BASE_URL 或 LLM_BASE_URL，跳过 vLLM embedding 测试")
    yield s


def test_search_with_vllm_embedding(store_vllm):
    """使用 VLLMChromaEmbeddingFunction 时向量检索可返回结果。"""
    store_vllm.create_collection("c1")
    store_vllm.add("c1", {"content": "用户喜欢简短回答", "metadata": {}})
    results = store_vllm.search("c1", "简短", limit=5)
    assert len(results) >= 1
