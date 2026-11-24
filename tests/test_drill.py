"""补传演练测试：切块、规划器（紧额度）、漏帧补传、Flask 全流程。"""
import hashlib
import io

import pytest

import app as web
from planner import (CHUNK_SIZE, MAX_CHUNKS, apply_round, plan_round,
                     reassemble, split_chunks)


# ---------- 切块 ----------

def test_split_and_reassemble_roundtrip():
    data = bytes(range(256)) * 35 + b"tail"  # 8964 字节 → 3 块
    chunks, size = split_chunks(data)
    assert len(chunks) == 3
    assert all(len(c) == CHUNK_SIZE for c in chunks)
    assert reassemble(chunks, size) == data


def test_split_limits():
    chunks, _ = split_chunks(b"x" * (CHUNK_SIZE * MAX_CHUNKS))
    assert len(chunks) == MAX_CHUNKS
    with pytest.raises(ValueError):
        split_chunks(b"x" * (CHUNK_SIZE * MAX_CHUNKS + 1))
    with pytest.raises(ValueError):
        split_chunks(b"")


# ---------- 规划器 ----------

def test_plan_xor_saves_frame_when_quota_allows():
    # T0 缺块3，T1 缺块2：一帧异或即可双双解回，但各需 1 次解码额度
    owned = [0b011, 0b101]
    plan = plan_round(owned, [1, 1], frame_limit=1, n_chunks=3)
    assert plan == [("xor", 1, 2)]


def test_plan_tight_quota_no_solution_then_exact():
    owned = [0b011, 0b101]
    # 解码额度为 0、只许 1 帧：无解
    assert plan_round(owned, [0, 0], frame_limit=1, n_chunks=3) is None
    # 额度为 0、放宽到 2 帧：退化为两帧单块
    plan = plan_round(owned, [0, 0], frame_limit=2, n_chunks=3)
    assert plan is not None and len(plan) == 2
    assert all(fr[0] == "single" for fr in plan)


def test_plan_tight_frame_limit():
    # 三台各缺两块，最少 3 帧（三块单播）
    owned = [0b001, 0b010, 0b100]
    assert plan_round(owned, [8, 8, 8], frame_limit=2, n_chunks=3) is None
    plan = plan_round(owned, [8, 8, 8], frame_limit=3, n_chunks=3)
    assert plan is not None and len(plan) == 3


def test_plan_already_complete():
    assert plan_round([0b111, 0b111], [0, 0], frame_limit=4, n_chunks=3) == []


# ---------- 轮次执行 ----------

def test_apply_round_fault_injection():
    frames = [("xor", 1, 2)]
    res = apply_round([0b011, 0b101], [1, 1], frames, fault=(0, 1))
    assert res[0]["mask"] == 0b011 and res[0]["decodes"] == 0  # 漏收，原地踏步
    assert res[1]["mask"] == 0b111 and res[1]["decodes"] == 1  # 解回块2
    assert res[1]["decoded"] == [(1, 1)]


def test_apply_round_quota_exhausted_no_decode():
    frames = [("xor", 1, 2)]
    res = apply_round([0b011, 0b101], [0, 0], frames)
    assert res[0]["mask"] == 0b011 and res[1]["mask"] == 0b101


# ---------- Flask 全流程 ----------

ORIGINAL = (b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 34 + b"demo")  # 3 块


@pytest.fixture
def client():
    with web.LOCK:
        web.STATE.clear()
        web.STATE.update(web._fresh_state())
    web.app.config.update(TESTING=True)
    return web.app.test_client()


def _upload(client, data=ORIGINAL, name="new.png"):
    return client.post("/upload", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", follow_redirects=True)


def _setup_two_terminals(client):
    # T1 首播收到 #1#2（缺#3），T2 收到 #1#3（缺#2）
    return client.post("/terminals", data={
        "enable_0": "on", "name_0": "T1", "recv_0": ["1", "2"],
        "enable_1": "on", "name_1": "T2", "recv_1": ["1", "3"],
    }, follow_redirects=True)


def test_full_flow_xor_round_completes_all(client):
    assert b"3" in _upload(client).data  # 切为 3 块
    _setup_two_terminals(client)
    r = client.post("/round", data={"frame_limit": "1", "quota_0": "1", "quota_1": "1"},
                    follow_redirects=True)
    assert "已发送 1 帧".encode() in r.data
    with web.LOCK:
        terms = list(web.STATE["terminals"])
        rounds = list(web.STATE["rounds"])
    assert all(t["completed"] for t in terms)
    assert rounds[0]["frames"] == ["异或 #2⊕#3"]
    # 两台都通过校验并替换旧图，可下载且与新图一致
    for tid in (0, 1):
        dl = client.get(f"/terminal/{tid}/download")
        assert dl.status_code == 200
        assert dl.data == ORIGINAL
        assert hashlib.sha256(dl.data).hexdigest() == web.STATE["image"]["sha256"]


def test_no_solution_keeps_state(client):
    _upload(client)
    _setup_two_terminals(client)
    r = client.post("/round", data={"frame_limit": "1", "quota_0": "0", "quota_1": "0"},
                    follow_redirects=True)
    assert "无方案".encode() in r.data
    with web.LOCK:
        assert web.STATE["rounds"] == []
        assert web.STATE["terminals"][0]["mask"] == 0b011
        assert web.STATE["terminals"][1]["mask"] == 0b101


def test_missed_frame_then_next_round_repair(client):
    _upload(client)
    _setup_two_terminals(client)
    # 第 1 轮：T1 漏收唯一的异或帧
    r = client.post("/round", data={
        "frame_limit": "1", "quota_0": "1", "quota_1": "1",
        "fault_terminal": "0", "fault_frame": "1",
    }, follow_redirects=True)
    assert "漏收第1帧".encode() in r.data
    with web.LOCK:
        t1, t2 = web.STATE["terminals"]
        assert not t1["completed"] and t1["mask"] == 0b011  # 仍缺 #3
        assert t2["completed"]
    dl = client.get("/terminal/0/download")
    assert dl.data != ORIGINAL  # 旧图未被替换
    # 第 2 轮：按实收状态补传，单块 #3 即可，无需解码额度
    r = client.post("/round", data={"frame_limit": "2", "quota_0": "0", "quota_1": "0"},
                    follow_redirects=True)
    assert "已发送 1 帧".encode() in r.data
    with web.LOCK:
        assert web.STATE["rounds"][1]["frames"] == ["单块 #3"]
        assert web.STATE["terminals"][0]["completed"]
    assert client.get("/terminal/0/download").data == ORIGINAL


def test_upload_oversize_rejected(client):
    big = b"y" * (CHUNK_SIZE * MAX_CHUNKS + 1)
    r = _upload(client, data=big, name="big.bin")
    assert "超过".encode() in r.data or "上限".encode() in r.data
    with web.LOCK:
        assert web.STATE["image"] is None
