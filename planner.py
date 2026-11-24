"""补传演练核心逻辑：切块、异或帧规划（BFS）、轮次执行。

约定：
- 图被切成 4 KiB 块，至多 8 块，末块补零，重组后按原始长度截断。
- 一帧要么是单块 ("single", c, None)，要么是两块异或 ("xor", a, b)。
- 终端实收异或帧时，恰缺两块之一且本轮解码额度未用完，才解回缺块（计 1 次解码）。
- 规划按"本轮无丢帧"推算：所有终端都收到所有帧。
"""
from collections import deque
from itertools import combinations

CHUNK_SIZE = 4096
MAX_CHUNKS = 8
MAX_FRAME_LIMIT = 16      # 单轮帧数上限的硬顶，防止 BFS 失控
SEARCH_NODE_CAP = 300_000  # BFS 状态数上限


def split_chunks(data: bytes):
    """切成 4 KiB 块（末块补零）。返回 (chunks, 原始长度)。"""
    if not data:
        raise ValueError("空文件，无法切块")
    n = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    if n > MAX_CHUNKS:
        raise ValueError(
            f"图共 {n} 块，超过 {MAX_CHUNKS} 块上限（需 ≤ {CHUNK_SIZE * MAX_CHUNKS} 字节）"
        )
    chunks = [
        data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].ljust(CHUNK_SIZE, b"\0")
        for i in range(n)
    ]
    return chunks, len(data)


def reassemble(chunks, size: int) -> bytes:
    """按原始长度截断重组。"""
    return b"".join(chunks)[:size]


def _candidate_frames(missing_union: int, n_chunks: int):
    """候选帧：缺失并集里的单块，以及至少含一块缺失块的异或对。"""
    frames = []
    for c in range(n_chunks):
        if (missing_union >> c) & 1:
            frames.append(("single", c, None))
    for a, b in combinations(range(n_chunks), 2):
        if ((missing_union >> a) & 1) or ((missing_union >> b) & 1):
            frames.append(("xor", a, b))
    return frames


def _step(masks, decs, quotas, frame):
    """全体终端收一帧后的联合状态；无人能受益则返回 None。"""
    kind, a, b = frame
    new_masks = list(masks)
    new_decs = list(decs)
    changed = False
    for i, m in enumerate(masks):
        if kind == "single":
            if not (m >> a) & 1:
                new_masks[i] = m | (1 << a)
                changed = True
        else:
            ha = (m >> a) & 1
            hb = (m >> b) & 1
            if ha != hb and decs[i] < quotas[i]:
                miss = b if ha else a
                new_masks[i] = m | (1 << miss)
                new_decs[i] = decs[i] + 1
                changed = True
    if not changed:
        return None
    return tuple(new_masks), tuple(new_decs)


def plan_round(owned_masks, quotas, frame_limit, n_chunks):
    """BFS 求一套帧序列：帧数 ≤ frame_limit，各终端解码 ≤ quotas，且
    按无丢帧推算所有终端收齐。返回帧列表；无解返回 None；已收齐返回 []。"""
    full = (1 << n_chunks) - 1
    owned_masks = tuple(owned_masks)
    if all(m == full for m in owned_masks):
        return []
    frame_limit = max(1, min(int(frame_limit), MAX_FRAME_LIMIT))
    missing_union = 0
    for m in owned_masks:
        missing_union |= full ^ m
    frames = _candidate_frames(missing_union, n_chunks)

    start = (owned_masks, tuple(0 for _ in owned_masks))
    parent = {start: None}
    queue = deque([(start, 0)])
    goal = None
    while queue and goal is None:
        (masks, decs), depth = queue.popleft()
        if depth >= frame_limit:
            continue
        for fr in frames:
            nxt = _step(masks, decs, quotas, fr)
            if nxt is None or nxt in parent:
                continue
            parent[nxt] = ((masks, decs), fr)
            if all(m == full for m in nxt[0]):
                goal = nxt
                break
            if len(parent) >= SEARCH_NODE_CAP:
                return None  # 搜索超限，按无方案处理
            queue.append((nxt, depth + 1))
    if goal is None:
        return None
    path = []
    s = goal
    while parent[s] is not None:
        prev, fr = parent[s]
        path.append(fr)
        s = prev
    path.reverse()
    return path


def apply_round(owned_masks, quotas, frames, fault=None):
    """实发一轮。fault=(终端序号, 帧号1基) 表示该终端漏收该帧。
    返回每终端结果：[{mask, decodes, decoded, received, missed_frame}]。"""
    results = []
    for i, m0 in enumerate(owned_masks):
        m = m0
        decs = 0
        decoded = []    # [(帧号, 解回的块)]
        received = []   # 实收帧号
        missed = fault[1] if fault and fault[0] == i else None
        for no, (kind, a, b) in enumerate(frames, 1):
            if no == missed:
                continue
            received.append(no)
            if kind == "single":
                m |= 1 << a
            else:
                ha = (m >> a) & 1
                hb = (m >> b) & 1
                if ha != hb and decs < quotas[i]:
                    miss = b if ha else a
                    m |= 1 << miss
                    decs += 1
                    decoded.append((no, miss))
        results.append({
            "mask": m,
            "decodes": decs,
            "decoded": decoded,
            "received": received,
            "missed_frame": missed,
        })
    return results
