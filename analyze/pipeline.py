"""pipeline.py — LangGraph 编排的分析管线（v2 内核）

节点图（全串行单链，每节点仅一个直接前驱）：
    prepare → extract_claims → position_map → debate_relations → pick_diagnosis
           → argument_diagnosis → nonargument_autopsy → best_args
           → suspect_facts → cognitive_insights → compose

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
    threads_mini: list[dict]    # 脱敏树结构（供原文引用，不失真）
    threads_total: int
    claims: dict
    position_map: dict
    debate_relations: dict
    picks: list[dict]
    diagnosis: dict
    nonargument: dict          # 非论证表达解剖（第六节）
    best_args: dict
    suspect_facts: dict
    insights: dict
    timings: dict              # 各节点耗时（compose 写入，供运行日志落盘）
    report_md: str


def make_graph(llm: DeepSeekClient):
    timings: dict = {}   # 各节点耗时（compose 返回，供运行日志落盘）
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

    def extract_claims(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_extract_claims(llm, state["corpus_text"])
        timings["M1 命题提取"] = time.time()-t0
        print(f"  [M1 命题提取] {time.time()-t0:.1f}s "
              f"命题 {len(r.get('claims', []))} 条 / "
              f"非论证 {len(r.get('non_argument_comments', []))} 条")
        return {"claims": r}

    def position_map(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_position_map(llm, state.get("claims", {}).get("claims", []),
                                    state["corpus_text"])
        timings["M2 观点地图"] = time.time()-t0
        print(f"  [M2 观点地图] {time.time()-t0:.1f}s "
              f"观点簇 {len(r.get('positions', []))} / "
              f"分歧 {len(r.get('core_disputes', []))}")
        return {"position_map": r}

    def debate_relations(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_debate_relations(
            llm, state.get("claims", {}).get("claims", []),
            state.get("position_map", {}), state["corpus_text"])
        timings["M3 论辩关系"] = time.time()-t0
        print(f"  [M3 论辩关系] {time.time()-t0:.1f}s "
              f"关系 {len(r.get('debate_relations', []))} 条 / "
              f"模式 {len(r.get('debate_patterns', []))}")
        return {"debate_relations": r}

    def pick_diagnosis(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_pick_diagnosis(llm, state["corpus_text"])
        picks = r.get("picks", [])
        timings["M4a 诊断候选"] = time.time()-t0
        print(f"  [M4a 诊断候选] {time.time()-t0:.1f}s 选中 {len(picks)} 处")
        return {"picks": picks}

    def _resolve_quote(state: PipelineState, thread_id: str) -> tuple[str, int]:
        """按路径 ['T5', 'T5_回复2', …] 任意深度解析回原文，返回 (文本, 赞数)。"""
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

    def argument_diagnosis(state: PipelineState) -> dict:
        # 输入：每个 pick 对应"整楼文本"（含主楼与回复，回复带 [Tx_回复N] 标记）
        items = []
        for p in state.get("picks", []):
            tid = p["thread_id"]
            block = _thread_block_text(state, tid)
            if block:
                items.append((tid, block))
        if not items:
            return {"diagnosis": {"items": []}}
        t0 = time.time()
        r = nodes.node_argument_diagnosis(llm, items)
        # 诊断结果与原文对齐（注入 quote 与赞数，供报告引用与定位）
        for it in r.get("items", []):
            it["quote"], it["quote_like"] = _resolve_quote(
                state, it.get("thread_id", ""))
        timings["M4b 论证诊断"] = time.time()-t0
        print(f"  [M4b 论证诊断] {time.time()-t0:.1f}s 诊断 {len(r.get('items', []))} 处")
        return {"diagnosis": r}

    def nonargument_autopsy(state: PipelineState) -> dict:
        """非论证表达解剖：玩梗/情绪/提问在争论里实际起了什么作用。

        素材来自命题提取给出的非论证索引；编号统一走 _resolve_quote 解析，
        解析不到的编号直接丢弃——模型写错编号也不能让报告出现指不到原文的引用。
        """
        idx = (state.get("claims") or {}).get("non_argument_comments") or []
        items = []
        for nc in idx:
            tid = nc.get("thread_id", "")
            text, like = _resolve_quote(state, tid)
            if text:
                items.append((tid, text, like))
        if not items:
            print("  [M5b 非论证解剖] 无非论证表达，跳过")
            return {"nonargument": {"clusters": [], "summary": {}}}
        t0 = time.time()
        r = nodes.node_nonargument_autopsy(llm, items)
        # 素材披露：模型登记的条数 vs 真正能读到正文的条数
        # （纯 "@某人" 之类的回复剥离前缀后没有正文，只能跳过——不说清楚，
        #   读者会把 4 条素材得出的结论当成 30 条的结论）
        r["material"] = {"indexed": len(idx), "used": len(items)}
        for c in r.get("clusters", []):
            c["thread_ids"] = [x for x in (c.get("thread_ids") or [])
                               if _resolve_quote(state, x)[0]]
        timings["M5b 非论证解剖"] = time.time()-t0
        print(f"  [M5b 非论证解剖] {time.time()-t0:.1f}s "
              f"聚类 {len(r.get('clusters', []))} 个 / 素材 {len(items)} 条")
        return {"nonargument": r}

    def best_args(state: PipelineState) -> dict:
        # 排除诊断过的主楼（N6 串行在诊断后）。双保险：
        # 1) 提示词告知  2) 语料物理删除诊断整楼——模型看不到就不会选
        exclude = [it.get("thread_id")
                   for it in (state.get("diagnosis") or {}).get("items", [])
                   if "_回复" not in it.get("thread_id", "")] or None
        corpus = (filter_out_threads(state["corpus_text"], exclude)
                  if exclude else state["corpus_text"])
        t0 = time.time()
        r = nodes.node_best_args(llm, corpus, exclude)
        for it in r.get("items", []):
            it["quote"], it["quote_like"] = _resolve_quote(
                state, it.get("thread_id", ""))
        timings["M6 最佳论证"] = time.time()-t0
        print(f"  [M6 最佳论证] {time.time()-t0:.1f}s 范本 {len(r.get('items', []))} 条")
        return {"best_args": r}

    def suspect_facts(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_suspect_facts(llm, state["corpus_text"])
        timings["M7 可疑事实"] = time.time()-t0
        print(f"  [M7 可疑事实] {time.time()-t0:.1f}s 断言 {len(r.get('facts', []))} 条")
        return {"suspect_facts": r}

    def cognitive_insights(state: PipelineState) -> dict:
        t0 = time.time()
        r = nodes.node_cognitive_insights(
            llm, state["corpus_text"],
            state.get("claims", {}).get("claims", []),
            state.get("position_map", {}),
            state.get("debate_relations", {}),
            (state.get("diagnosis") or {}).get("items", []))
        timings["M5 认知观察"] = time.time()-t0
        print(f"  [M5 认知观察] {time.time()-t0:.1f}s "
              f"观察 {len(r.get('readings', []))} 条 / "
              f"开放问题 {len(r.get('open_questions', []))} 个")
        return {"insights": r}

    def compose(state: PipelineState) -> dict:
        return {"report_md": render_report(state), "timings": timings}

    g = StateGraph(PipelineState)
    g.add_node("prepare", prepare)
    g.add_node("extract_claims", extract_claims)
    g.add_node("position_map", position_map)
    g.add_node("debate_relations", debate_relations)
    g.add_node("pick_diagnosis", pick_diagnosis)
    g.add_node("argument_diagnosis", argument_diagnosis)
    g.add_node("nonargument_autopsy", nonargument_autopsy)
    g.add_node("best_args", best_args)
    g.add_node("suspect_facts", suspect_facts)
    g.add_node("cognitive_insights", cognitive_insights)
    g.add_node("compose", compose)

    # 全串行单链：每个节点只有一个直接前驱。
    # 实测本版本 LangGraph 多入边节点会重复执行（模型调用翻倍），
    # 串行化后总耗时仅增加约 15 秒（M1 命题提取是绝对瓶颈）。
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "extract_claims")
    g.add_edge("extract_claims", "position_map")
    g.add_edge("position_map", "debate_relations")
    g.add_edge("debate_relations", "pick_diagnosis")
    g.add_edge("pick_diagnosis", "argument_diagnosis")
    g.add_edge("argument_diagnosis", "nonargument_autopsy")
    g.add_edge("nonargument_autopsy", "best_args")
    g.add_edge("best_args", "suspect_facts")
    g.add_edge("suspect_facts", "cognitive_insights")
    g.add_edge("cognitive_insights", "compose")
    g.add_edge("compose", END)
    return g.compile()


# ────────────────────────── 工具函数 ──────────────────────────

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
    供诊断节点选择任一层级。tid 指向该楼即可。"""
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


def filter_out_threads(corpus_text: str, exclude_ids: list[str]) -> str:
    """从语料文本中物理移除指定主楼（[T5] 开头直到下一个 [T 块的整块）。

    用于 M6：诊断过的整楼从输入中消失，模型不可能再选它做范本。
    """
    blocks, cur = [], None
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
            continue
        kept.append(blk)
    return "\n\n".join(kept)


def _q(quote: str, like: int = 0) -> str:
    """引用原文（多行压成一行）并转成引用块，附抓取时刻赞数供定位。"""
    if not quote:
        return ""
    one = " ".join(quote.split())
    if len(one) > 150:
        one = one[:150] + "…"
    return f"\n> {one}（抓取时 {like} 赞）"


# ────────────────────────── M6 报告渲染 ──────────────────────────

def render_report(s: PipelineState) -> str:
    """九段式报告：概览 / 命题还原 / 观点地图 / 论辩关系 / 论证诊断 /
    非论证表达解剖 / 值得学习的论证 / 可疑事实 / 认知与社会观察。"""
    meta = s["meta"]
    cl = s.get("claims") or {}
    pm = s.get("position_map") or {}
    dr = s.get("debate_relations") or {}
    dg = s.get("diagnosis") or {}
    na = s.get("nonargument") or {}
    ba = s.get("best_args") or {}
    sf = s.get("suspect_facts") or {}
    ins = s.get("insights") or {}

    L = []
    L.append("# 评论区思辨报告\n")
    L.append(f"**视频**：{meta.get('title', '?')}")
    L.append(f"**链接**：https://www.bilibili.com/video/{meta.get('bvid', '')}")
    L.append(f"**分析范围**：热评前 {s.get('threads_total', '?')} 条主评论（含楼中楼）")
    L.append(f"**抓取时间**：{meta.get('fetched_at', '?')}（UTC）\n")
    L.append("> **引用定位**：编号按抓取时刻的 B站「按热度」排序生成——"
             "`[T0]` = 热度第 1 条主评论，`[T12]` = 从上往下第 13 条；"
             "`T12_回复3` = 点开 T12 后楼中楼第 3 条（热度序）。"
             "抓取后热度会随点赞变化重排，位置对不上时，按引用旁的原文和"
             "「抓取时 N 赞」在评论区对照查找。\n")
    L.append("> 本报告分析**论证质量与讨论结构**，不评判任何立场对错。"
             "评论一律匿名，不分析评论者本人，不做评分榜。"
             "对玩梗、反问、脏话一类非论证表达，本报告**不删除、不排序、不折叠**，"
             "而是逐条解剖它们在争论中起的作用——见第六节。\n")

    # ── 一、讨论概览 ──
    claims = cl.get("claims") or []
    non_arg = cl.get("non_argument_comments") or []
    L.append("\n## 一、讨论概览\n")
    L.append(f"本次分析从 {s.get('threads_total', '?')} 条主评论及其回复中，"
             f"提取出 {len(claims)} 条有实质内容的命题，"
             f"{len(non_arg)} 条被归为非论证表达（玩梗/提问/纯情绪）"
             f"——它们的解剖见第六节。")
    L.append("样本说明：仅覆盖热评前若干条及其高赞回复链，不代表全部评论。\n")

    # ── 二、命题还原 ──
    L.append("\n## 二、命题还原：评论者到底提出了什么\n")
    L.append("> 只还原命题，不评价对错。「隐含前提」是论证依赖但未明说的假设。\n")
    shown = 0
    for c in claims:
        if shown >= 12:
            break
        L.append(f"- **[{c.get('claim_id')}]（{c.get('thread_id')}）** {c.get('main_claim')}")
        types = c.get("claim_types") or []
        if types:
            L.append(f"  - 类型：{'、'.join(types)}")
        for a in (c.get("implicit_assumptions") or [])[:2]:
            L.append(f"  - 隐含前提〔{a.get('confidence', '?')}〕：{a.get('content')}")
        for u in (c.get("uncertainties") or [])[:2]:
            L.append(f"  - 不确定处：{u}")
        shown += 1
    if len(claims) > shown:
        L.append(f"- ……（其余 {len(claims)-shown} 条命题见结构化数据）")

    # ── 三、观点地图 ──
    L.append("\n## 三、观点地图：真正的分歧在哪里\n")
    L.append("### 观点簇\n")
    for p in pm.get("positions", []):
        L.append(f"- **{p.get('position_id')} · {p.get('name')}**：{p.get('core_claim')}")
        for r in (p.get("main_reasons") or [])[:2]:
            L.append(f"  - 主要理由：{r}")
        for a in (p.get("key_assumptions") or [])[:2]:
            L.append(f"  - 关键前提：{a}")
        for v in (p.get("internal_variations") or [])[:2]:
            L.append(f"  - 簇内分歧：{v}")
    L.append("\n### 观点之间的关系\n")
    for r in pm.get("relationships", []):
        rel = {"conflict": "冲突", "partial_conflict": "部分冲突",
               "apparent_conflict": "表面冲突（可能可兼容）",
               "compatible": "可兼容"}.get(r.get("relationship"), r.get("relationship"))
        L.append(f"- **{r.get('position_a')} ↔ {r.get('position_b')}**：{rel}。{r.get('reason')}")
    L.append("\n### 核心分歧\n")
    for d in pm.get("core_disputes", []):
        L.append(f"- **{d.get('question')}**〔{d.get('dimension')}〕")
        for side in d.get("sides", []):
            L.append(f"  - {side}")

    # ── 四、论辩关系 ──
    L.append("\n## 四、论辩关系：他们真的在互相回应吗\n")
    rel_name = {
        "direct_response": "直接回应", "partial_response": "部分回应",
        "misunderstanding": "误解", "talking_past": "各说各话",
        "topic_shift": "转移话题", "escalation": "升级对抗",
        "productive_progress": "有效推进",
    }
    for r in (dr.get("debate_relations") or [])[:12]:
        L.append(f"- **{r.get('target_comment_id')} → {r.get('source_comment_id')}**"
                 f"〔{rel_name.get(r.get('relation_type'), r.get('relation_type'))}"
                 f"，{r.get('confidence', '?')}〕")
        L.append(f"  - 被回应的命题：{r.get('responded_to_claim')}")
        L.append(f"  - 实际回应了：{r.get('actual_response')}")
        L.append(f"  - 判定依据：{r.get('analysis')}")
    L.append("\n### 反复出现的论辩模式\n")
    for p in dr.get("debate_patterns", []):
        L.append(f"- **{p.get('pattern')}**：{p.get('description')}"
                 f"（代表：{', '.join(p.get('representative_threads', []))}）")
    sm = dr.get("summary") or {}
    if sm:
        L.append("\n### 数据概览\n")
        for k, label in [("direct_response_ratio", "直接回应"),
                         ("talking_past_ratio", "各说各话"),
                         ("misunderstanding_ratio", "误解"),
                         ("escalation_ratio", "升级对抗"),
                         ("productive_progress_ratio", "有效推进")]:
            if sm.get(k):
                L.append(f"- {label}：{sm[k]}")
        if sm.get("data_limitations"):
            L.append(f"- 数据局限：{sm['data_limitations']}")

    # ── 五、论证诊断 ──
    L.append("\n## 五、论证诊断：有问题的表达为什么有传播力\n")
    L.append("> 诊断的是**论证过程**，不是评论者本人；论证质量低不等于没有"
             "分析价值。每条都附带其中值得认真对待的问题。\n")
    for it in dg.get("items", []):
        L.append(f"### {it.get('thread_id')}{_q(it.get('quote', ''), it.get('quote_like', 0))}\n")
        st = it.get("argument_structure") or {}
        if st.get("conclusion"):
            L.append(f"- **试图让人接受**：{st['conclusion']}")
            for r in (st.get("reasons") or [])[:2]:
                L.append(f"  - 理由：{r}")
            if st.get("reasoning_path"):
                L.append(f"  - 推理路径：{st['reasoning_path']}")
        for d in it.get("diagnosis", []):
            conf = f"〔{d.get('confidence', '?')}〕" if d.get("confidence") else ""
            L.append(f"- **问题**{conf}：{d.get('observed_problem')}")
            if d.get("possible_logical_issue"):
                L.append(f"  - 可能的逻辑问题：{d['possible_logical_issue']}")
        for w in (it.get("why_this_pattern_may_appear") or [])[:2]:
            L.append(f"- **为什么常见**：{w}")
        av = it.get("analytical_value") or {}
        if av.get("valuable_question") and av.get("valuable_question") != "无":
            L.append(f"- **值得认真对待的问题**〔{av.get('level', '?')}〕：{av.get('valuable_question')}")
            if av.get("explanation"):
                L.append(f"  - {av['explanation']}")
        if it.get("better_response"):
            L.append(f"- **示范性回应**：{it['better_response']}\n")

    # ── 六、非论证表达解剖 ──
    L.append("\n## 六、非论证表达解剖：玩梗、情绪与提问在做什么\n")
    L.append("> 这些表达没有被丢掉，也不等于论证缺陷。解剖它们是为了看清它们在争论里"
             "实际承担了什么工作（有时比正经论证更有效），以及其中有没有被埋掉的真问题。\n")
    clusters = na.get("clusters") or []
    if clusters:
        for c in clusters:
            kinds = "、".join(c.get("kinds") or []) or "未标注"
            L.append(f"- **{c.get('label')}**〔{kinds}，{c.get('confidence', '?')}〕")
            if c.get("literal_reading"):
                L.append(f"  - 字面：{c['literal_reading']}")
            if c.get("what_it_does"):
                L.append(f"  - 在争论里：{c['what_it_does']}")
            kernel = (c.get("argument_kernel") or "").strip()
            if kernel and kernel != "无":
                L.append(f"  - 值得认真对待的内核：{kernel}")
            else:
                L.append("  - 值得认真对待的内核：无（已确认过，不是没找）")
            if c.get("thread_ids"):
                L.append(f"  - 出现于：{', '.join(c['thread_ids'])}")
        mat = na.get("material") or {}
        if mat.get("used") is not None:
            idx_n, used_n = mat.get("indexed", 0), mat.get("used", 0)
            extra = "（其余无正文或编号无效，已跳过）" if idx_n > used_n else ""
            L.append(f"\n- 解剖素材：{used_n} 条" +
                     (f"，命题提取共登记 {idx_n} 条{extra}" if idx_n > used_n else ""))
        sm2 = na.get("summary") or {}
        if sm2.get("share_of_sample"):
            L.append(f"\n- 占比：{sm2['share_of_sample']}")
        if sm2.get("data_limitations"):
            L.append(f"- 数据局限：{sm2['data_limitations']}")
    else:
        L.append("- 本次样本里没有可归类的非论证表达。")

    # ── 七、值得学习的论证 ──
    L.append("\n## 七、值得学习的论证\n")
    for i, b in enumerate(ba.get("items", []), 1):
        L.append(f"{i}. **{b.get('thread_id')}**（{b.get('stance', '?')}）："
                 f"{b.get('why_good')}{_q(b.get('quote', ''), b.get('quote_like', 0))}\n")

    # ── 八、可疑事实清单 ──
    L.append("\n## 八、可疑事实清单（待核实，不代表证伪）\n")
    facts = sf.get("facts") or []
    if facts:
        for i, f in enumerate(facts, 1):
            L.append(f"{i}. {f.get('claim')}")
            L.append(f"   - 出现于：{', '.join(f.get('appears_in', []))}｜{f.get('why_suspicious')}")
    else:
        L.append("- 无")

    # ── 九、认知与社会观察 ──
    L.append("\n## 九、认知与社会观察\n")
    for r in (ins.get("readings") or []):
        L.append(f"### {r.get('phenomenon')}")
        L.append(f"- **证据**：{r.get('evidence')}")
        L.append(f"- **解读**：{r.get('reflection')}〔{r.get('confidence', '?')}〕\n")
    L.append("\n### 评论区没有解决的问题\n")
    for q in (ins.get("open_questions") or []):
        L.append(f"- **{q.get('question')}**")
        L.append(f"  - 为什么难解决：{q.get('why_hard')}")

    L.append("\n---\n*由评论透镜生成。报告分析讨论结构，不评判立场，不评价个人，"
             "请勿据此攻击任何评论者。*\n")
    return "\n".join(L)
