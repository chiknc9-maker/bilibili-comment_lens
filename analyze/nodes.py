"""nodes.py — M3 分析节点（v2 内核）：命题提取 / 观点地图 / 论辩关系 /
论证诊断 / 认知与社会观察 / 辅助节点（最佳论证、可疑事实）

每个节点是纯函数 (ctx...) -> dict，只负责自己的字段。

═══ 产品价值观（写进每个分析节点提示词的公共原则）═══
1. 分析论证与讨论结构，不评判立场对错：任何阵营接受同一套标准。
2. 报告匿名化：输出永远不出现用户名，引用用 [T序号]。
3. 只描述问题，不贴标签：优先说"论证在哪一环断了"，而不是扣"稻草人"帽子。
4. 不推测评论者身份、心理、动机——只分析表达本身。
5. 论证质量低 ≠ 信息价值低：低质表达是公共讨论的观察窗口。
6. 拿不准的事实用条件式表达（"如果……则可能……"），不伪精确。
"""
from __future__ import annotations

import json

SYSTEM_CORE = """你是一名网络评论分析员，正在分析一个B站争议视频的评论区。
铁律：
- 你的任务是分析"论证质量"和"讨论结构"，绝不评判任何一方的立场对错。
- 不要带入你自己的政治/价值立场，不要表现出对任何一方的倾向。
- 输出中不得出现用户名（uname），引用评论时用 [T序号] 标注即可。
- 涉及历史的讨论只区分"事实断言"与"观点"，不为任何事实断言背书或证伪。
- 语境铁律：中文互联网评论大量使用反讽、玩梗与亚文化黑话（圈子梗、双关、
  抽象话术），字面意思不等于真实意图。遇到用词突兀、语气夸张、逻辑反常、
  或读起来不像认真发言的评论，先怀疑是在玩梗或反讽，按上下文解读，不要
  按字面意思归类、解剖或引用。对拿不准是否为梗的表达，如实标注
  "疑似圈子梗/反讽，含义未确认"，不要自信地给出字面解读。
- 所有输出必须是合法的 JSON 对象。"""

# ────────────────────────── M1 命题提取器 ──────────────────────────

def node_extract_claims(llm, corpus_text: str, max_claims: int = 50) -> dict:
    """M1：从评论中提取命题（claims）。只还原命题，不评价对错。

    输出规模受控：聚焦有实质论证或立场表达的评论，最多 max_claims 条；
    无明确命题的评论（纯玩梗/灌水/纯提问）不生成 claim 条目。
    """
    system = SYSTEM_CORE + """
执行"命题提取"：把评论还原为它们实际提出的命题。这是后续所有分析的基础，
所以只做还原、绝不评价。

对每条有实质内容的评论输出一条 claim：
{
  "claims": [
    {
      "claim_id": "C1",
      "thread_id": "T3 或 T3_回复2（原样标注）",
      "main_claim": "该评论的核心命题，用完整句子表述（不含立场评价）",
      "supporting_claims": ["支撑核心命题的次级命题"],
      "reasons": ["给出的理由"],
      "implicit_assumptions": [
        {"content": "未明说但论证依赖的前提", "confidence": "high/medium/low"}
      ],
      "claim_types": ["factual/causal/conceptual/evaluative/normative/preference/mixed 中选"],
      "uncertainties": ["该命题自身承认或暴露的不确定之处"]
    }
  ],
  "non_argument_comments": [
    {"thread_id": "T7", "kind": "playful_meme/question/emotion_only/other",
     "note": "一句话说明为何归为非论证"}
  ]
}

原则：
- 严格区分三层：文本明确表达的 / 从文本合理推断的（进 implicit_assumptions）/
  无法确定的（不写）。绝不推测评论者身份、心理、真实动机。
- 不贴政治或意识形态标签。
- 玩梗、反讽的表达：标注其字面命题的同时，在 uncertainties 里注明
  "疑似反讽/圈子梗，真实立场无法确定"。
- claims 最多 """ + str(max_claims) + """ 条，优先提取有实质论证的；
  其余归入 non_argument_comments（也控制在 30 条内）。"""
    return llm.chat_json(system, f"评论区内容：\n\n{corpus_text}",
                         max_tokens=8000)


def claims_brief(claims: list[dict], limit: int = 50) -> str:
    """claims 压缩摘要（供下游节点输入，控制 token）。"""
    brief = [{
        "claim_id": c.get("claim_id"),
        "thread_id": c.get("thread_id"),
        "main_claim": (c.get("main_claim") or "")[:80],
        "types": c.get("claim_types"),
        "assumptions": [(a or {}).get("content", "")[:40]
                        for a in (c.get("implicit_assumptions") or [])][:2],
    } for c in (claims or [])[:limit]]
    return json.dumps(brief, ensure_ascii=False)


# ────────────────────────── M2 观点地图构建器 ──────────────────────────

def node_position_map(llm, claims: list[dict], corpus_text: str) -> dict:
    """M2：把命题聚类为观点（positions），找出真正的分歧。

    关键升级（相对旧 stance_map）：
    - 不按"支持/反对"二分，按命题簇聚类
    - 区分前提分歧/概念分歧/价值分歧
    - 标记"表面冲突但实际可兼容"的伪对立（apparent_conflict）
    """
    system = SYSTEM_CORE + """
基于"命题提取"的结果构建观点地图。

positions 是命题簇，不是阵营。两条评论立场相反但共享同一前提时，
要单独标出它们共享的前提；表面在吵、实则用词/定义不同而可兼容的，
标记为伪对立。

输出 JSON：
{
  "positions": [
    {
      "position_id": "P1",
      "name": "中性命名，描述主张本身（如'艺术鉴赏需要专门训练'），不用贬褒义词",
      "core_claim": "该观点簇的核心命题（完整句）",
      "main_reasons": ["该簇常见理由"],
      "key_assumptions": ["该簇依赖的未言明前提"],
      "representative_comment_ids": ["C1", "C5"],
      "internal_variations": ["簇内成员之间的分歧点"]
    }
  ],
  "relationships": [
    {
      "position_a": "P1", "position_b": "P2",
      "relationship": "conflict/partial_conflict/apparent_conflict/compatible",
      "reason": "为什么这么判定（apparent_conflict 必须说明是定义不同还是层面错位）"
    }
  ],
  "core_disputes": [
    {
      "question": "双方实际在争的命题（写清主语与谓语）",
      "dimension": "事实层/定义层/价值层/规范层/可行层",
      "sides": ["一方论证要点", "另一方论证要点"]
    }
  ]
}
要求：positions 给 2~6 个；core_disputes 给 1~4 个；relationships 覆盖主要
position 对（不必穷举）。命名全部中性。"""
    return llm.chat_json(
        system,
        f"命题提取结果：\n{claims_brief(claims)}\n\n评论区原文（辅助核对语境）：\n\n{corpus_text[:6000]}",
        max_tokens=5000)


# ────────────────────────── M3 论辩关系分析器 ──────────────────────────

def node_debate_relations(llm, claims: list[dict], positions: dict,
                          corpus_text: str) -> dict:
    """M3：分析谁真正回应了谁、谁在各说各话。

    铁律：A 回复了 B 不等于 A 回应了 B 的命题——必须对比父评论核心命题
    与子评论实际回应内容。比例无法可靠计算时如实说"样本不足/粗略估计"。
    """
    positions_brief = json.dumps(
        [{"id": p.get("position_id"), "name": p.get("name"),
          "core": (p.get("core_claim") or "")[:60]}
         for p in (positions or {}).get("positions", [])],
        ensure_ascii=False)
    system = SYSTEM_CORE + """
执行"论辩关系分析"：评论区里的回复链，哪些是真正的回应，哪些在
各说各话、误解、转移话题。

回复层级信息来自语料缩进与编号（T5_回复2 是 T5 的回复，T5_回复2_回复3
是 T5_回复2 的回复）。逐条对比：父评论的核心命题 vs 子评论实际处理的内容。

输出 JSON：
{
  "debate_relations": [
    {
      "source_comment_id": "发起方 T编号（被回复者）",
      "target_comment_id": "回复方 T编号",
      "relation_type": "direct_response/partial_response/misunderstanding/talking_past/topic_shift/escalation/productive_progress",
      "responded_to_claim": "父评论被回复的那个命题",
      "actual_response": "子评论实际在说什么",
      "analysis": "为什么这么判定（50字内）",
      "confidence": "high/medium/low"
    }
  ],
  "debate_patterns": [
    {"pattern": "反复出现的模式名（如'用个人经验替代对方主张'）",
     "description": "模式说明", "representative_threads": ["T3", ...]}
  ],
  "summary": {
    "direct_response_ratio": "粗略比例或'样本不足'（不得编造精确百分比）",
    "talking_past_ratio": "同上",
    "misunderstanding_ratio": "同上",
    "escalation_ratio": "同上",
    "productive_progress_ratio": "同上",
    "data_limitations": "数据局限说明（如只分析高赞楼、深树采样不完整）"
  }
}
要求：debate_relations 最多 15 条，挑最有代表性的攻防链；
比例只能给"多数/少数/约一半/样本不足"这类粒度。"""
    return llm.chat_json(
        system,
        f"命题提取结果：\n{claims_brief(claims)}\n\n观点地图：\n{positions_brief}"
        f"\n\n评论区原文（含回复缩进层级）：\n\n{corpus_text[:8000]}",
        max_tokens=6000)


# ────────────────────────── M4 论证诊断器 ──────────────────────────

def node_pick_diagnosis(llm, corpus_text: str, n: int = 5) -> dict:
    """M4a 候选挑选：选出"论证问题典型、有教学价值"的评论用于诊断。

    注意：挑选标准是论证质量，与立场无关——任何立场的坏论证都可能被选中。
    主楼与回复分别评估。
    """
    system = SYSTEM_CORE + """
从评论中挑选 """ + str(n) + """ 处"高互动/高传播但论证存在典型问题"的评论，
用于论证诊断（教学目的）。优先挑"看着有理、细想有问题"或"话术典型"的，
如：轻佻化严肃议题、传闻当论据、个案代表整体、类比替代论证、情绪替代论证、
信口开河的数字/时间线。

不要挑：纯提问、纯脏话、有实质论据的评论、无实质文字内容的图文评论、
纯玩梗附和、单纯纠正他人错误且纠正正确的评论（无论语气多嘲讽）。
输出 JSON：
{
  "picks": [
    {"thread_id": "T序号（回复用 T序号_回复N）",
     "reason": "为什么值得诊断（30字内，说明问题类型）"}
  ]
}"""
    return llm.chat_json(system, f"评论区内容：\n\n{corpus_text}")


def node_argument_diagnosis(llm, threads_text: list[tuple[str, str]]) -> dict:
    """M4b 逐条诊断。克制原则：
    - 描述问题（observed_problem），不直接贴"稻草人"等标签；标签只作为
      "可能的逻辑问题"（possible_logical_issue）条件式给出。
    - 区分观点立场/事实准确性/论证质量/表达质量/分析价值五个维度。
    - 论证质量低 ≠ 信息价值低：analytical_value 是必答项。
    - 不评价评论者本人。
    """
    item_prompt = "\n\n".join(f"{tid}:\n{text}" for tid, text in threads_text)
    system = SYSTEM_CORE + """
对给出的每个"争论串"（含主楼与回复）做"论证诊断"。注意控制数量：
- 每个争论串必须诊断其主楼，输出 1 条
- 回复只在"论证问题典型、有教学价值"时才诊断，每个串最多 2 条回复

诊断用语要求（重要）：
- observed_problem 用描述性语言写"这个论证哪里断了"（如"结论依赖一个
  未被证明的因果联系"），不要直接贴"稻草人""滑坡"等术语标签。
- possible_logical_issue 可以用术语，但必须条件式：
  "如果 X 确实依赖前提 Y，则可能构成 Z（confidence 表示把握）"。
- 合理提问、讽刺、情绪表达本身不是论证缺陷，只有"用它们替代论证"才算。

输出 JSON：
{
  "items": [
    {
      "thread_id": "T序号（回复用 T序号_回复N）",
      "argument_structure": {
        "conclusion": "该评论试图让人接受的结论",
        "reasons": ["给出的理由"],
        "evidence": ["引用的证据（如'评论区里说是X'）"],
        "reasoning_path": "从理由到结论的推理路径（一句话）"
      },
      "diagnosis": [
        {
          "observed_problem": "描述性说明论证哪里断了",
          "possible_logical_issue": "条件式指出可能对应的逻辑问题",
          "confidence": "high/medium/low"
        }
      ],
      "why_this_pattern_may_appear": ["这种论证方式为什么常见/有传播力（认知或语境角度，不指责个人）"],
      "analytical_value": {
        "level": "high/medium/low",
        "valuable_question": "这条评论背后值得认真对待的问题",
        "explanation": "为什么即使论证有问题，这个问题本身有分析价值"
      },
      "better_response": "若正面回应，一句示范（具体、有信息量、不嘲讽；禁止万能空话）"
    }
  ]
}
要求：diagnosis 每层没有发现就空数组，绝不凑数；items 总数不得超过
输入争论串数量的 3 倍。"""
    return llm.chat_json(system, f"待诊断评论（[T序号]为一个争论串，含主楼与回复）：\n\n{item_prompt}",
                         max_tokens=6000)


# ────────────────────────── M5 认知与社会观察器 ──────────────────────────

def node_cognitive_insights(llm, corpus_text: str, claims: list[dict],
                            positions: dict, relations: dict,
                            diagnosis_items: list[dict]) -> dict:
    """M5：基于前序模块的认知与社会观察。

    只分析表达模式、认知路径、讨论结构——不分析单个评论者心理。
    """
    claims_note = claims_brief(claims, 30)
    pos_note = json.dumps(
        [{"name": p.get("name"), "core": (p.get("core_claim") or "")[:60]}
         for p in (positions or {}).get("positions", [])], ensure_ascii=False)
    rel_note = json.dumps(
        [(r.get("source_comment_id"), r.get("target_comment_id"),
          r.get("relation_type")) for r in (relations or {}).get("debate_relations", [])],
        ensure_ascii=False)
    diag_note = "\n".join(
        f"- {it.get('thread_id')}: {it.get('argument_structure', {}).get('conclusion', '')[:40]}"
        for it in (diagnosis_items or [])[:10])
    system = SYSTEM_CORE + """
做一层"元观察"：基于前序分析（命题、观点、论辩关系、论证诊断），总结
评论区**整体**呈现的认知模式、表达模式与讨论结构问题。
- 只分析表达模式与讨论结构，不分析单个评论者心理
- 每条观察必须有评论区证据（引用评论原文片段）
- 解读用"可能反映""或与……有关"等克制表达，不下断语
输出 JSON：
{
  "readings": [
    {"phenomenon": "现象名（如'用个人经验替代对方主张'）",
     "evidence": "评论原文片段（1~2条）",
     "reflection": "它可能反映的认知模式/社会语境（60~100字）",
     "confidence": "high/medium/low"}
  ],
  "open_questions": [
    {"question": "评论区认真讨论了但最终没有解决的问题",
     "why_hard": "为什么难解决（信息不足/价值冲突/立场预设/问题本身无解…）"}
  ]
}
readings 给 2~4 条，open_questions 给 2~3 条。"""
    return llm.chat_json(
        system,
        f"命题摘要：\n{claims_note}\n\n观点簇：\n{pos_note}\n\n"
        f"论辩关系（来源→回复→类型）：\n{rel_note}\n\n"
        f"诊断对象结论：\n{diag_note}\n\n评论区原文：\n\n{corpus_text[:6000]}",
        max_tokens=3072)


# ────────────────────────── 辅助节点 ──────────────────────────

def node_best_args(llm, corpus_text: str, exclude_ids: list[str] | None = None) -> dict:
    """N6 值得学习的论证：为各观点簇挑论证质量高的范本，避开诊断对象。"""
    ex = exclude_ids or []
    system = SYSTEM_CORE + """
从评论中挑出论证质量最高的几条作为"值得学习的论证范本"（覆盖不同观点簇，
不偏袒任何一方）。标准：有事实依据或推理清晰、认真回应了对方、不人身攻击。

重要：以下评论（主楼级）已被选为"论证诊断"对象，范本必须避开，不要挑它们
（其楼内未被诊断的回复仍可入选）：
""" + ("、".join(ex) if ex else "（本轮无诊断对象）") + """
输出 JSON：
{
  "items": [
    {"thread_id": "T序号", "stance": "它代表的观点（中性命名）",
     "why_good": "为什么这条论证是好的（50字内，说明好在论据/推理/态度）"}
  ]
}"""
    return llm.chat_json(system, f"评论区内容：\n\n{corpus_text}")


def node_suspect_facts(llm, corpus_text: str) -> dict:
    """N7 可疑事实清单：评论区高频出现、需要核实的事实性断言。

    硬约束：最多 10 条、按"若为假对讨论误导程度"排序取前 10、不得重复列出。
    """
    system = SYSTEM_CORE + """
提取评论区中出现的"事实性断言"（数字、统计、历史事件经过、作品内容、
人物背景、引语等可被验证对错的内容）。这些断言是争论的火药桶。

筛选铁律：
- 重要性：只列"一旦为假会显著误导讨论"的公共断言。排除：纯玩笑/反讽/
  明显不真、私人八卦、无法验证的个人主观、纯比喻。
- 同一断言多人重复时必须合并为一条，绝不重复列出。
- 不为任何断言下"对/错"结论，只标"待核实"。
- 最多输出 10 条，按重要性从高到低排列。
输出 JSON：
{
  "facts": [
    {"claim": "被断言的事实",
     "appears_in": ["T序号", ...],
     "why_suspicious": "为什么值得核实（50字内）"}
  ]
}
没有符合条件的事实断言时 facts 给空数组。"""
    return llm.chat_json(system, f"评论区内容：\n\n{corpus_text}", max_tokens=2048)
