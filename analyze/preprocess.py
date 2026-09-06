"""preprocess.py — 把抓取的评论 JSON 变成适合送入 LLM 的紧凑文本

支持任意深度的对话树（主楼 → 回复 → 回复的回复 → …）。
成本与隐私控制：
- 每节点最多保留 MAX_PER_NODE 条直接回复（按点赞降序）
- 单条文本截断 MAX_TEXT_LEN
- 匿名化：剥离"回复 @用户名 ："前缀（正文里嵌的他人用户名不进入语料）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_PER_NODE = 5      # 树中每个节点最多保留多少条直接回复
MAX_TEXT_LEN = 300     # 单条评论截断长度（字符）
MAX_DEPTH = 8          # 渲染时的防御性深度上限

_REPLY_PREFIX = re.compile(r"^\s*回复\s*@[^:：]+[:：]\s*")
_AT_PREFIX = re.compile(r"^\s*@[^:：]+[:：]\s*")


def clean_text(raw: str) -> str:
    """清洗：去回复引用前缀（含被 @ 的用户名），截断。"""
    t = _REPLY_PREFIX.sub("", raw or "")
    t = _AT_PREFIX.sub("", t)
    return t[:MAX_TEXT_LEN]


@dataclass
class Thread:
    """一条主评论及其对话树（分析单元）。

    树节点均为 dict：{"like", "text", "replies": [子节点…]}。
    """
    rpid: int
    uname: str
    like: int
    text: str
    replies: list[dict] = field(default_factory=list)

    def to_text(self, thread_index: int) -> str:
        lines = [f"[T{thread_index}]（主楼，{self.like}赞）{self.text}"]

        def render(node: dict, path: str, depth: int):
            if depth > MAX_DEPTH:
                return
            for n, kid in enumerate(node.get("replies") or [], 1):
                kid_path = f"{path}_回复{n}" if path else f"回复{n}"
                tag = (f"[T{thread_index}_{kid_path}]" if path
                       else f"[T{thread_index}_回复{n}]")
                indent = "  " * (depth + 1)
                lines.append(f"{indent}{tag}（{kid['like']}赞）{kid['text']}")
                render(kid, kid_path, depth + 1)

        render({"replies": self.replies}, "", 0)
        return "\n".join(lines)


def load_threads(json_path: Path) -> tuple[dict, list[Thread]]:
    """读取抓取 JSON，返回 (meta, threads)；按点赞数降序。"""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    threads = []
    for t in raw["threads"]:
        subs = [_clean_node(r) for r in (t.get("replies") or [])]
        threads.append(Thread(
            rpid=t["rpid"], uname=t["uname"], like=t.get("like", 0),
            text=clean_text(t["text"]), replies=subs,
        ))
    threads.sort(key=lambda t: t.like, reverse=True)
    return raw["meta"], threads


def _clean_node(node: dict, depth: int = 0) -> dict:
    """递归清洗/截断树节点；每层最多 MAX_PER_NODE 子。"""
    if depth > MAX_DEPTH:
        return {"like": 0, "text": "", "replies": []}
    kids = [_clean_node(k, depth + 1)
            for k in (node.get("replies") or [])][:MAX_PER_NODE]
    return {"like": node.get("like", 0), "text": clean_text(node.get("text", "")),
            "replies": kids}


def build_corpus_text(threads: list[Thread]) -> str:
    """所有 thread 拼成一个带编号的文本块（给立场聚类用）。"""
    return "\n\n".join(t.to_text(i) for i, t in enumerate(threads))


def estimate_chars(threads: list[Thread]) -> int:
    def node_chars(node: dict) -> int:
        return len(node["text"]) + sum(node_chars(k)
                                       for k in node.get("replies") or [])
    return sum(len(t.text) + sum(node_chars(s) for s in t.replies)
               for t in threads)
