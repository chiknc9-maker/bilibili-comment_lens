"""pipeline.py — LangGraph 编排的分析管线

节点图（DAG）：
    prepare
      ├─→ stance_map ──────────────────────────────┐
      ├─→ pick_junk ──→ autopsy ─┬─→ best_args ─────┤
      │                           └─→ insights ─────┤
      └─→ suspect_facts ────────────────────────────┤
                                                     ▼
                                                  compose
（N3 串行在解剖后：须避开解剖名单，避免同评论既被解剖又被表扬）

节点都是纯函数 (state: dict) -> dict，只写自己负责的字段。
模型调用统一走 analyze.llm.DeepSeekClient（JSON mode 结构化输出）。
"""
from __future__ import annotations

import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import nodes
from .llm import DeepSeekClient
from .preprocess import build_corpus_text, load_threads


class PipelineState(TypedDict, total=False):
    meta: dict
    threads: list               # Thread dataclass 列表（入口传入，prepare 消费）
    corpus_text: str
    threads_mini: list[dict]    # 结构化 thread（供原文引用，不失真）
    threads_total: int
    stance_map: dict
    picks: list[dict]
    autopsy: dict
    insights: dict
    best_args: dict
    suspect_facts: dict
    report_md: str


def make_graph(llm: DeepSeekClient):
    def prepare(state: PipelineState) -> dict:
        # 脱敏后的树结构（仅 like/text/replies），供引用与文本块组装
        def mini(node: dict) -> dict:
            return {"like": node.get("like", 0), "text": node["text"],
                    "replies": [mini(k) for k in node.get("replies") or []]}

        threads_mini = [{"text": t.text,
                         "replies": [mini(r) for r in t.replies]}
                        for t in state["threads"]]
        return {
            "meta": state["meta"],
            "corpus_text": state["corpus_text"],
            "threads_mini": threads_mini,
            "threads_total": len(state["threads"]),
        }

    def stance_map(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_stance_map(llm, state["corpus_text"])
        print(f"  [N1 立场地图] {time.time()-t0:.1f}s 阵营数: {len(r.get('stances', []))}")
        return {"stance_map": r}

    def pick_junk(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_pick_junk(llm, state["corpus_text"])
        picks = r.get("picks", [])
        print(f"  [N2a 候选挑选] {time.time()-t0:.1f}s 选中 {len(picks)} 处")
        return {"picks": picks}

    def _resolve_quote(state: PipelineState, thread_id: str) -> tuple[str, int]:
        """按路径 ['T5', 'T5_回复2', 'T5_回复2_回复3', …] 任意深度解析回原文，
        返回 (文本, 抓取时刻点赞数)；找不到返回 ("", 0)。"""
        path = _parse_tid(thread_id)
        if not path:
            return "", 0
        idx = path[0]
        if idx >= len(state["threads_mini"]):
            return "", 0
        node: dict = state["threads_mini"][idx]
        for step in path[1:]:
            kids = node.get("replies") or []
            if step >= len(kids):
                return "", 0
            node = kids[step]
        return node.get("text", ""), node.get("like", 0)

    def autopsy(state: PipelineState) -> dict:
        # 输入：每个 pick 对应"整楼文本"（含主楼与回复，回复带 [Tx_回复N] 标记），
        # 从结构化 threads_mini 组装——不依赖语料文本的格式
        items = []
        for p in state.get("picks", []):
            tid = p["thread_id"]
            block = _thread_block_text(state, tid)
            if block:
                items.append((tid, block))
        if not items:
            return {"autopsy": {"items": []}}
        t0 = time.time()
        r = nodes.node_autopsy(llm, items)
        # 关键一步：把解剖结果与原文对齐（注入 quote 与赞数，供报告引用与定位）
        for it in r.get("items", []):
            it["quote"], it["quote_like"] = _resolve_quote(
                state, it.get("thread_id", ""))
        print(f"  [N2b 垃圾解剖] {time.time()-t0:.1f}s 解剖 {len(r.get('items', []))} 处")
        return {"autopsy": r}

    def insights(state: PipelineState) -> dict:
        t0 = time.time()
        items = (state.get("autopsy") or {}).get("items", [])
        r = nodes.node_insights(llm, state["corpus_text"], items)
        print(f"  [N5 洞察] {time.time()-t0:.1f}s "
              f"观察 {len(r.get('readings', []))} 条 / 开放问题 {len(r.get('open_questions', []))} 个")
        return {"insights": r}

    def best_args(state: PipelineState) -> dict:
        # 排除解剖过的主楼（N3 串行在 autopsy 之后）。双保险：
        # 1) 提示词告知（仅语义提醒）  2) 语料物理删除解剖整楼——模型看不到就不会选
        exclude = [it.get("thread_id")
                   for it in (state.get("autopsy") or {}).get("items", [])
                   if "_回复" not in it.get("thread_id", "")] or None
        corpus = (filter_out_threads(state["corpus_text"], exclude)
                  if exclude else state["corpus_text"])
        t0 = time.time()
        r = nodes.node_best_args(llm, corpus, exclude)
        # 范本同样附原文
        for it in r.get("items", []):
            it["quote"], it["quote_like"] = _resolve_quote(
                state, it.get("thread_id", ""))
        print(f"  [N3 最佳论证] {time.time()-t0:.1f}s 范本 {len(r.get('items', []))} 条")
        return {"best_args": r}

    def suspect_facts(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_suspect_facts(llm, state["corpus_text"])
        print(f"  [N4 可疑事实] {time.time()-t0:.1f}s 断言 {len(r.get('facts', []))} 条")
        return {"suspect_facts": r}

    def compose(state: PipelineState) -> dict:
        # 数据清洗 1：解剖对象与范本撞车时（独立调用不一致的产物），
        # 主楼解剖优先，同名范本剔除；回复级解剖不影响范本。
        autopsy_main = {it.get("thread_id")
                        for it in (state.get("autopsy") or {}).get("items", [])
                        if "_回复" not in it.get("thread_id", "")}
        best = (state.get("best_args") or {}).copy()
        if autopsy_main:
            best["items"] = [b for b in best.get("items", [])
                             if b.get("thread_id") not in autopsy_main]
        # 注意：render 必须使用清洗后的状态（局部合并，勿直接读 state）
        cleaned = {**state, "best_args": best}
        return {"best_args": best, "report_md": render_report(cleaned)}

    g = StateGraph(PipelineState)
    g.add_node("prepare", prepare)
    g.add_node("stance_map", stance_map)
    g.add_node("pick_junk", pick_junk)
    g.add_node("autopsy", autopsy)
    g.add_node("insights", insights)
    g.add_node("best_args", best_args)
    g.add_node("suspect_facts", suspect_facts)
    g.add_node("compose", compose)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "stance_map")
    g.add_edge("prepare", "pick_junk")
    g.add_edge("prepare", "suspect_facts")
    g.add_edge("pick_junk", "autopsy")
    # N3 串行在解剖之后（需避开解剖名单）；N5 与 N3 并行
    g.add_edge("autopsy", "best_args")
    g.add_edge("autopsy", "insights")
    # fan-in：全部链都完成后 compose 才执行
    g.add_edge("stance_map", "compose")
    g.add_edge("best_args", "compose")
    g.add_edge("insights", "compose")
    g.add_edge("suspect_facts", "compose")
    g.add_edge("compose", END)
    return g.compile()


def filter_out_threads(corpus_text: str, exclude_ids: list[str]) -> str:
    """从语料文本中物理移除指定主楼（[T5] 开头直到下一个 [T 块的整块）。

    用于 N3：解剖过的整楼从输入中消失，模型不可能再选它做范本。
    仅回复被解剖的楼（如 T13_回复8）不在排除名单时，整楼保留。
    """
    blocks = []
    cur = None
    for line in corpus_text.split("\n"):
        if line.startswith("[T") and "]" in line[:6]:
            if cur is not None:
                blocks.append("\n".join(cur))
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append("\n".join(cur))

    excluded = {tid for tid in exclude_ids if "_回复" not in tid}
    kept = []
    for blk in blocks:
        m = re.match(r"\[T(\d+)\]", blk)
        if m and f"T{int(m.group(1))}" in excluded:
            continue  # 解剖过的整楼，跳过
        kept.append(blk)
    return "\n\n".join(kept)


def _parse_tid(tid: str) -> list[int] | None:
    """'T5' → [5]；'T5_回复2' → [5, 1]；'T5_回复2_回复3' → [5, 1, 2]。
    返回从主楼起的 0-based 路径；解析失败返回 None。"""
    m = re.fullmatch(r"T(\d+)(?:(?:_回复(\d+))*)$", tid or "")
    if not m:
        return None
    path = [int(m.group(1))]
    for g in re.findall(r"_回复(\d+)", tid):
        path.append(int(g) - 1)
    return path


def _thread_block_text(state: PipelineState, tid: str) -> str | None:
    """取某一楼完整文本块（主楼 + 任意深度的对话树，每节点带 [Tx_回复…] 标记），
    供解剖节点选择任一层级。tid 指向该楼即可（主楼或任一后代皆可）。"""
    parsed = _parse_tid(tid)
    if not parsed:
        return None
    idx = parsed[0]
    if idx >= len(state["threads_mini"]):
        return None
    t = state["threads_mini"][idx]
    lines = [f"[T{idx}]（主楼）{t['text']}"]

    def render(node: dict, segs: list[str], depth: int):
        for n, kid in enumerate(node.get("replies") or [], 1):
            seg = f"_回复{n}" if segs else f"回复{n}"
            tag = f"[T{idx}_{'_'.join(segs + [seg])}]"
            indent = "  " * (depth + 1)
            lines.append(f"{indent}{tag}（{kid['like']}赞）{kid['text']}")
            render(kid, segs + [seg], depth + 1)

    render({"replies": t["replies"]}, [], 0)
    return "\n".join(lines)


def _q(quote: str, like: int = 0) -> str:
    """引用原文（多行压成一行、去回复缩进）并转成引用块，
    附抓取时刻的赞数供原评论区定位。"""
    if not quote:
        return ""
    one = " ".join(quote.split())
    if len(one) > 150:
        one = one[:150] + "…"
    like_note = f"（抓取时 {like} 赞）" if like else ""
    return f"\n> {one}{like_note}"


def render_report(s: PipelineState) -> str:
    """把各节点 JSON 结果组装成最终 Markdown 报告。"""
    meta = s["meta"]
    st = s.get("stance_map") or {}
    au = s.get("autopsy") or {}
    ins = s.get("insights") or {}
    ba = s.get("best_args") or {}
    sf = s.get("suspect_facts") or {}

    L = []
    L.append(f"# 评论区思辨报告\n")
    L.append(f"**视频**：{meta.get('title', '?')}")
    L.append(f"**链接**：https://www.bilibili.com/video/{meta.get('bvid', '')}")
    L.append(f"**分析范围**：热评前 {s.get('threads_total', '?')} 条主评论（含楼中楼）")
    L.append(f"**抓取时间**：{meta.get('fetched_at', '?')}（UTC）\n")
    L.append("> **引用定位**：编号按抓取时刻的 B站「按热度」排序生成——"
             "`[T0]` = 热度第 1 条主评论，`[T12]` = 从上往下第 13 条；"
             "`T12_回复3` = 点开 T12 后楼中楼第 3 条（热度序）。"
             "抓取后热度会随点赞变化重排，位置对不上时，按引用旁的原文和"
             "「抓取时 N 赞」在评论区对照查找。\n")
    L.append("> 本报告分析**论证质量**，不评判任何立场对错。评论一律匿名。\n")

    # ── 一、争论地图 ──
    L.append("\n## 一、争论地图：这场争论到底在争什么\n")
    L.append("### 立场阵营\n")
    for i, stc in enumerate(st.get("stances", []), 1):
        L.append(f"{i}. **{stc.get('name')}**（{stc.get('size', '?')} 条）：{stc.get('summary')}")
    L.append("\n### 核心分歧\n")
    disputes = st.get("core_disputes") or []
    if disputes:
        for d in disputes:
            L.append(f"- **{d.get('question')}**〔{d.get('dimension')}〕")
            for side in d.get("sides", []):
                L.append(f"  - {side}")
    else:
        L.append("- 未识别出真正的命题冲突")
    L.append('\n### 错位检测：谁在「各说各话」\n')
    mm = st.get("mismatches") or []
    if mm:
        for m in mm:
            L.append(f"- {m.get('description')}（相关：{', '.join(m.get('thread_ids', []))}）")
    else:
        L.append("- 未发现明显错位")

    # ── 二、垃圾解剖 ──
    L.append("\n## 二、垃圾解剖：烂论证烂在哪\n")
    L.append("> 以下是评论中**高赞/高传播但论证质量低**的代表。注意：分析论证不评"
             "立场；每条都标注了其中值得回应的内核。\n")
    for item in au.get("items", []):
        L.append(f"### 解剖对象 {item.get('thread_id')}"
                 f"{_q(item.get('quote', ''), item.get('quote_like', 0))}\n")
        layers = [
            ("论据", item.get("evidence_problems")),
            ("推理", item.get("reasoning_problems")),
            ("情绪", item.get("emotion_problems")),
            ("修辞", item.get("rhetoric_problems")),
        ]
        for name, probs in layers:
            if probs:
                L.append(f"- **{name}层**：" + "；".join(probs))
        L.append(f"- **为什么有说服力**：{item.get('why_convincing')}")
        L.append(f"- **值得回应的内核**：{item.get('kernel')}")
        L.append(f"- **示范性回应**：{item.get('reply_demo')}\n")

    # ── 三、最佳论证 ──
    L.append("\n## 三、值得学习的论证范本\n")
    for i, b in enumerate(ba.get("items", []), 1):
        quote = _q(b.get("quote", ""), b.get("quote_like", 0))
        L.append(f"{i}. **{b.get('thread_id')}**（{b.get('stance', '?')}）：{b.get('why_good')}"
                 f"{quote}\n")

    # ── 四、可疑事实 ──
    L.append("\n## 四、可疑事实清单（待核实，不代表证伪）\n")
    facts = sf.get("facts") or []
    if facts:
        for i, f in enumerate(facts, 1):
            L.append(f"{i}. {f.get('claim')}")
            L.append(f"   - 出现于：{', '.join(f.get('appears_in', []))}｜{f.get('why_suspicious')}")
    else:
        L.append("- 无")

    # ── 五、垃圾的社会学观察 ──
    L.append("\n## 五、垃圾的社会学观察：低质评论在反映什么\n")
    L.append("> 论证质量低 ≠ 信息价值低。把情绪、偏见、玩梗当社会认知的窗口来读。\n")
    for r in (ins.get("readings") or []):
        L.append(f"### {r.get('phenomenon')}")
        L.append(f"- **证据**：{r.get('evidence')}")
        L.append(f"- **解读**：{r.get('reflection')}\n")

    # ── 六、评论区没有解决的问题 ──
    L.append("\n## 六、评论区没有解决的问题\n")
    for q in (ins.get("open_questions") or []):
        L.append(f"- **{q.get('question')}**")
        L.append(f"  - 为什么难解决：{q.get('why_hard')}")

    L.append("\n---\n*由评论透镜生成。报告只评论证不评人，请勿据此攻击任何评论者。*\n")
    return "\n".join(L)
