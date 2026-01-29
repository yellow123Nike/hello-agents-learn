"""Neo4jStore 测试：创建 database/集合、增删改查、检索。无 Neo4j 服务时跳过。"""

import pytest

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from hello_agents.core.storage import Neo4jStore


def _neo4j_connectible():
    """检测默认本机 Neo4j 是否可连。"""
    if not NEO4J_AVAILABLE:
        return False
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


neo4j_required = pytest.mark.skipif(
    not NEO4J_AVAILABLE or not _neo4j_connectible(),
    reason="Neo4j 未安装或本地 Neo4j 服务不可用（默认 bolt://localhost:7687, neo4j/password）",
)


@pytest.fixture
def store():
    """使用默认本机 Neo4j；若不可用则用例被 skip。"""
    s = Neo4jStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
    )
    s.create_database()
    yield s
    # 清理：删除测试集合中的全部节点（使用唯一集合名避免影响其他数据）
    try:
        s.delete("pytest_neo4j_store_test", ids=[])  # 无 id 时 to_delete 为空，不删
        # 若需清空整个集合，可在此执行 Cypher DELETE；此处仅做占位
    except Exception:
        pass


@pytest.fixture
def collection(store):
    """使用带 pytest 前缀的集合名，便于区分测试数据。"""
    name = "pytest_neo4j_store_test"
    store.create_collection(name)
    return name


@neo4j_required
def test_create_database_and_collection(store, collection):
    id_ = store.add(collection, {"content": "hello", "metadata": {}})
    assert id_
    doc = store.get(collection, id_)
    assert doc is not None
    assert doc["content"] == "hello"


@neo4j_required
def test_add_returns_id(store, collection):
    id1 = store.add(collection, {"content": "a", "metadata": {"k": "v"}})
    assert isinstance(id1, str) and len(id1) > 0
    id2 = store.add(collection, {"content": "b"}, doc_id="custom-neo4j-id")
    assert id2 == "custom-neo4j-id"


@neo4j_required
def test_get_existing(store, collection):
    id_ = store.add(collection, {"content": "内容", "metadata": {"type": "test"}})
    doc = store.get(collection, id_)
    assert doc is not None
    assert doc["id"] == id_
    assert doc["content"] == "内容"
    assert doc["metadata"] == {"type": "test"}


@neo4j_required
def test_get_missing_returns_none(store, collection):
    assert store.get(collection, "non-existent-id") is None


@neo4j_required
def test_update_existing(store, collection):
    id_ = store.add(collection, {"content": "old", "metadata": {}})
    ok = store.update(collection, id_, {"content": "new", "metadata": {"v": 1}})
    assert ok is True
    doc = store.get(collection, id_)
    assert doc["content"] == "new"
    assert doc["metadata"] == {"v": 1}


@neo4j_required
def test_update_missing_returns_false(store, collection):
    ok = store.update(collection, "no-such-id", {"content": "x", "metadata": {}})
    assert ok is False


@neo4j_required
def test_delete_by_id(store, collection):
    id_ = store.add(collection, {"content": "x", "metadata": {}})
    n = store.delete(collection, id=id_)
    assert n == 1
    assert store.get(collection, id_) is None


@neo4j_required
def test_delete_by_ids(store, collection):
    id1 = store.add(collection, {"content": "a", "metadata": {}})
    id2 = store.add(collection, {"content": "b", "metadata": {}})
    n = store.delete(collection, ids=[id1, id2])
    assert n == 2
    assert store.get(collection, id1) is None
    assert store.get(collection, id2) is None


@neo4j_required
def test_search_by_content(store, collection):
    store.add(collection, {"content": "用户偏好简洁回答", "metadata": {"type": "pref"}})
    store.add(collection, {"content": "无关内容", "metadata": {}})
    results = store.search(collection, "简洁", limit=5)
    assert isinstance(results, list)
    assert any("简洁" in d["content"] for d in results)


@neo4j_required
def test_add_many(store, collection):
    docs = [
        {"content": "first", "metadata": {}},
        {"content": "second", "metadata": {"i": 2}},
    ]
    ids = store.add_many(collection, docs)
    assert len(ids) == 2
    assert store.get(collection, ids[0])["content"] == "first"
    assert store.get(collection, ids[1])["content"] == "second"


@neo4j_required
def test_retrieve_by_ids(store, collection):
    id1 = store.add(collection, {"content": "a", "metadata": {}})
    id2 = store.add(collection, {"content": "b", "metadata": {}})
    docs = store.retrieve_by_ids(collection, [id1, id2])
    assert len(docs) == 2
    ids = {d["id"] for d in docs}
    assert id1 in ids and id2 in ids
