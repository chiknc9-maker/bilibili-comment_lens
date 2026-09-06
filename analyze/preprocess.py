"""preprocess.py — 把抓取的评论 JSON 变成适合送入 LLM 的紧凑文本

成本控制（M1-v1 的取舍，后续可加 thread 摘要层进一步压缩）：
- 主评论全部保留（热评本身有立场代表性）
- 楼中楼只保留点赞 >= MIN_SUB_LIKE 的回复，每条主评最多 MAX_SUB_PER_THREAD 条
- 单条文本截断 MAX_TEXT_LEN 字符（保留开头，通常信息密度最高）
- 超长文本在截断处标注「…」，分析时不会误当作完整引用
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MIN_SUB_LIKE = 0       # 楼中楼点赞门槛（0=全保留：低赞回复常藏高质量反驳）
MAX_SUB_PER_THREAD = 10  # 每条主评论最多保留多少楼中楼（按点赞降序）
MAX_D3_PER_SUB = 5     # 每条二级回复最多保留多少"回复的回复"（三级）
MAX_TEXT_LEN = 300     # 单条评论截断长度（字符）


@dataclass
class Thread:
    """一条主评论及其楼中楼（分析单元）。

    replies 元素为 dict：{"like", "text", "replies": [{"like","text"}...]}，
    其中内层 replies 是三级评论（回复的回复）。
    """
    rpid: int
    uname: str
    like: int
    text: str
    replies: list[dict] = field(default_factory=list)

    def to_text(self, thread_index: int) -> str:
        lines = [f"[T{thread_index}]（主楼，{self.like}赞）{self.text}"]
        for i, r in enumerate(self.replies, 1):
            lines.append(
                f"  [T{thread_index}_回复{i}]（回复{i}，{r['like']}赞）{r['text']}")
            for j, d3 in enumerate(r.get("replies") or [], 1):
                lines.append(f"    [T{thread_index}_回复{i}_回复{j}]"
                             f"（回复{i}的回复{j}，{d3['like']}赞）{d3['text']}")
        return "\n".join(lines)


def load_threads(json_path: Path) -> tuple[dict, list[Thread]]:
    """读取抓取 JSON，返回 (meta, threads)；按点赞数降序。"""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    threads = []
    for t in raw["threads"]:
        subs = []
        for r in t.get("replies") or []:
            if r.get("like", 0) < MIN_SUB_LIKE:
                continue
            if len(subs) >= MAX_SUB_PER_THREAD:
                break
            d3 = [{"like": x.get("like", 0), "text": x["text"][:MAX_TEXT_LEN]}
                  for x in (r.get("replies") or [])][:MAX_D3_PER_SUB]
            subs.append({"like": r.get("like", 0),
                         "text": r["text"][:MAX_TEXT_LEN],
                         "replies": d3})
        threads.append(Thread(
            rpid=t["rpid"], uname=t["uname"], like=t.get("like", 0),
            text=t["text"][:MAX_TEXT_LEN], replies=subs,
        ))
    threads.sort(key=lambda t: t.like, reverse=True)
    return raw["meta"], threads


def build_corpus_text(threads: list[Thread]) -> str:
    """所有 thread 拼成一个带编号的文本块（给立场聚类用）。"""
    return "\n\n".join(t.to_text(i) for i, t in enumerate(threads))


def estimate_chars(threads: list[Thread]) -> int:
    return sum(len(t.text) + sum(
        len(r["text"]) + sum(len(d3["text"]) for d3 in r.get("replies") or [])
        for r in t.replies) for t in threads)
