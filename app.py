"""站台终端窄带组播补传演练网页。

流程：上传新图（≤32 KiB，切 4 KiB 块，至多 8 块）→ 登记 2~4 台终端首播
实收块号 → 逐轮规划补传（限帧数、限各终端异或解码次数，可注入某终端漏收
某帧）→ 各终端独立拼图，SHA-256 与新图相符才替换可下载的旧图。
"""
import hashlib
import io
import threading

from flask import (Flask, flash, redirect, render_template, request,
                   send_file, url_for)

from planner import (CHUNK_SIZE, MAX_CHUNKS, MAX_FRAME_LIMIT, apply_round,
                     plan_round, reassemble, split_chunks)

app = Flask(__name__)
app.secret_key = "multicast-repair-drill-dev-key"

LOCK = threading.Lock()
MIN_TERMINALS = 2
MAX_TERMINALS = 4


def _fresh_state():
    return {"image": None, "terminals": [], "rounds": []}


STATE = _fresh_state()


def _mask_of(indices):
    m = 0
    for c in indices:
        m |= 1 << c
    return m


def _missing_list(mask, n_chunks):
    """缺块号，1 基，供页面显示。"""
    return [c + 1 for c in range(n_chunks) if not (mask >> c) & 1]


def _owned_list(mask, n_chunks):
    return [c + 1 for c in range(n_chunks) if (mask >> c) & 1]


def _fmt_frame(fr):
    kind, a, b = fr
    if kind == "single":
        return f"单块 #{a + 1}"
    return f"异或 #{a + 1}⊕#{b + 1}"


def _terminal_view(t, n_chunks):
    return {
        "name": t["name"],
        "owned": _owned_list(t["mask"], n_chunks),
        "missing": _missing_list(t["mask"], n_chunks),
        "completed": t["completed"],
        "total_decodes": t["total_decodes"],
    }


def _view_state():
    img = STATE["image"]
    n = img["n_chunks"] if img else 0
    return {
        "image": img,
        "terminals": [_terminal_view(t, n) for t in STATE["terminals"]],
        "rounds": STATE["rounds"],
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "max_frame_limit": MAX_FRAME_LIMIT,
        "min_terminals": MIN_TERMINALS,
        "max_terminals": MAX_TERMINALS,
        "term_rows": range(MAX_TERMINALS),
    }


@app.get("/")
def index():
    with LOCK:
        return render_template("index.html", **_view_state())


@app.post("/upload")
def upload():
    f = request.files.get("image")
    if not f or not f.filename:
        flash("未选择图片文件")
        return redirect(url_for("index"))
    data = f.read()
    try:
        chunks, size = split_chunks(data)
    except ValueError as e:
        flash(f"上传被拒：{e}")
        return redirect(url_for("index"))
    with LOCK:
        STATE.clear()
        STATE.update(_fresh_state())
        STATE["image"] = {
            "name": f.filename,
            "size": size,
            "sha256": hashlib.sha256(data).hexdigest(),
            "n_chunks": len(chunks),
            "chunks": chunks,
            "data": data,
        }
    flash(f"已载入新图：{f.filename}，{size} 字节，切为 {len(chunks)} 块。请登记终端首播实收块号。")
    return redirect(url_for("index"))


@app.post("/terminals")
def setup_terminals():
    with LOCK:
        img = STATE["image"]
        if not img:
            flash("请先上传新图")
            return redirect(url_for("index"))
        n = img["n_chunks"]
        terminals = []
        for i in range(MAX_TERMINALS):
            if not request.form.get(f"enable_{i}"):
                continue
            got = set()
            for v in request.form.getlist(f"recv_{i}"):
                c = int(v)
                if 1 <= c <= n:
                    got.add(c - 1)
            name = request.form.get(f"name_{i}", "").strip() or f"T{len(terminals) + 1}"
            terminals.append({
                "name": name,
                "mask": _mask_of(got),
                "completed": False,
                "total_decodes": 0,
                "old": f"旧图占位：终端 {name} 尚未收齐新图。\n".encode(),
                "ext": ".txt",
            })
        if not (MIN_TERMINALS <= len(terminals) <= MAX_TERMINALS):
            flash(f"启用的终端数须为 {MIN_TERMINALS}~{MAX_TERMINALS} 台，当前 {len(terminals)} 台")
            return redirect(url_for("index"))
        STATE["terminals"] = terminals
        STATE["rounds"] = []
        # 首播即收齐的终端立即校验替换
        full = (1 << n) - 1
        for t in terminals:
            if t["mask"] == full:
                _try_finalize(t)
    flash(f"已登记 {len(terminals)} 台终端首播实收情况。")
    return redirect(url_for("index"))


def _try_finalize(t):
    """收齐后重组，SHA-256 与新图相符才替换可下载旧图。"""
    img = STATE["image"]
    data = reassemble(img["chunks"], img["size"])
    digest = hashlib.sha256(data).hexdigest()
    if digest == img["sha256"]:
        t["old"] = data
        t["ext"] = _guess_ext(img["name"])
        t["completed"] = True
        return True
    return False


def _guess_ext(filename):
    if "." in filename:
        return "." + filename.rsplit(".", 1)[1]
    return ".bin"


@app.post("/round")
def run_round():
    with LOCK:
        img = STATE["image"]
        terms = STATE["terminals"]
        if not img or not terms:
            flash("请先上传新图并登记终端")
            return redirect(url_for("index"))
        n = img["n_chunks"]
        try:
            frame_limit = int(request.form.get("frame_limit", ""))
        except ValueError:
            flash("帧数限额须为整数")
            return redirect(url_for("index"))
        if not (1 <= frame_limit <= MAX_FRAME_LIMIT):
            flash(f"帧数限额须在 1~{MAX_FRAME_LIMIT} 之间")
            return redirect(url_for("index"))
        quotas = []
        for i in range(len(terms)):
            try:
                q = int(request.form.get(f"quota_{i}", "0"))
            except ValueError:
                q = -1
            if not (0 <= q <= MAX_CHUNKS):
                flash(f"终端 {terms[i]['name']} 的解码额度须为 0~{MAX_CHUNKS} 的整数")
                return redirect(url_for("index"))
            quotas.append(q)
        fault = None
        ft = request.form.get("fault_terminal", "")
        ff = request.form.get("fault_frame", "").strip()
        if ft != "" and ff:
            try:
                fi, fn = int(ft), int(ff)
            except ValueError:
                fi, fn = -1, 0
            if 0 <= fi < len(terms) and fn >= 1:
                fault = (fi, fn)

        masks = [t["mask"] for t in terms]
        frames = plan_round(masks, quotas, frame_limit, n)
        if frames is None:
            flash("无方案：本轮限额内无法让各站收齐，已保留各站已收块，未发送任何帧。")
            return redirect(url_for("index"))
        if frames == []:
            flash("各站均已收齐，无需补传。")
            return redirect(url_for("index"))
        if fault and fault[1] > len(frames):
            fault = None  # 指定的漏收帧号超出实发帧数，忽略

        results = apply_round(masks, quotas, frames, fault)
        full = (1 << n) - 1
        rec_terminals = []
        for i, (t, r) in enumerate(zip(terms, results)):
            before = _missing_list(t["mask"], n)
            t["mask"] = r["mask"]
            t["total_decodes"] += r["decodes"]
            just_completed = False
            if t["mask"] == full and not t["completed"]:
                just_completed = _try_finalize(t)
            rec_terminals.append({
                "name": t["name"],
                "before_missing": before,
                "after_missing": _missing_list(t["mask"], n),
                "received": r["received"],
                "missed_frame": r["missed_frame"],
                "decoded": [(no, c + 1) for no, c in r["decoded"]],
                "decodes": r["decodes"],
                "quota": quotas[i],
                "completed": t["completed"],
                "just_completed": just_completed,
            })
        STATE["rounds"].append({
            "no": len(STATE["rounds"]) + 1,
            "frame_limit": frame_limit,
            "frames": [_fmt_frame(fr) for fr in frames],
            "fault": (terms[fault[0]]["name"], fault[1]) if fault else None,
            "terminals": rec_terminals,
        })
        done = sum(1 for t in terms if t["completed"])
        flash(f"第 {len(STATE['rounds'])} 轮已发送 {len(frames)} 帧；{done}/{len(terms)} 台已收齐并校验替换。")
    return redirect(url_for("index"))


@app.post("/reset")
def reset():
    with LOCK:
        STATE.clear()
        STATE.update(_fresh_state())
    flash("演练已重置。")
    return redirect(url_for("index"))


@app.get("/terminal/<int:tid>/download")
def download(tid):
    with LOCK:
        if not (0 <= tid < len(STATE["terminals"])):
            flash("终端不存在")
            return redirect(url_for("index"))
        t = STATE["terminals"][tid]
        return send_file(
            io.BytesIO(t["old"]),
            as_attachment=True,
            download_name=f"{t['name']}_current{t['ext']}",
        )


@app.get("/image/download")
def download_new():
    with LOCK:
        img = STATE["image"]
        if not img:
            flash("尚未上传新图")
            return redirect(url_for("index"))
        return send_file(io.BytesIO(img["data"]), as_attachment=True,
                         download_name=img["name"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
