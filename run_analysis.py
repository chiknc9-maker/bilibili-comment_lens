"""run_analysis.py — M1 分析管线入口

用法:
    python run_analysis.py data/BV1roE9zUEF5_comments.json
    python run_analysis.py data/BV1roE9zUEF5_comments.json --dry-run   # 只算成本预算，不调模型
    python run_analysis.py data/BV1roE9zUEF5_comments.json --out reports

先确保 .env 里有 DEEPSEEK_API_KEY（见 .env.example）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from analyze.llm import DeepSeekClient
from analyze.pipeline import make_graph
from analyze.preprocess import build_corpus_text, estimate_chars, load_threads

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# DeepSeek 价格（2026-09 快照，仅供预算提示，以官方为准）
PRICE_PER_M_IN = 2.0   # 元/百万输入 token
PRICE_PER_M_OUT = 8.0  # 元/百万输出 token


def analyze_comments(json_path: Path, use_cache: bool = True) -> dict:
    """分析评论 JSON（预处理→LangGraph 管线），返回结果 dict：
    {meta, report_md, 及各节点结构化结果}。供 CLI 与网页服务共用。"""
    meta, threads = load_threads(json_path)
    corpus = build_corpus_text(threads)
    chars = estimate_chars(threads)
    print(f"  视频: {meta.get('title', '?')}")
    print(f"  参与分析: {len(threads)} 条主评论，约 {chars} 字符 ≈ "
          f"{chars//2}~{chars//1} token（中文约 1 token/字）")

    llm = DeepSeekClient(cache_enabled=use_cache)
    print(f"  模型应答: {llm.ping()}（key 有效）")
    graph = make_graph(llm)
    t0 = time.time()
    result = graph.invoke({
        "meta": meta,
        "threads": threads,
        "corpus_text": corpus,
        "threads_total": len(threads),
    })
    result["usage"] = {"prompt_tokens": llm.total_usage.get("prompt_tokens", 0),
                       "completion_tokens": llm.total_usage.get("completion_tokens", 0),
                       "total_tokens": llm.total_usage.get("total_tokens", 0),
                       "calls": llm.call_count,
                       "cache_hits": llm.cache_hits}
    print(f"  管线完成，耗时 {time.time()-t0:.0f}s（{llm.call_count} 次调用，"
          f"{llm.cache_hits} 次命中缓存，"
          f"{llm.total_usage.get('total_tokens', 0)} tokens）")
    return result


def save_report(result: dict, out_dir: Path) -> Path:
    """把分析结果落盘（报告 md + 结构化 json），返回报告路径。"""
    out_dir.mkdir(exist_ok=True)
    meta = result.get("meta") or {}
    stem = meta.get("bvid", "video")
    report_path = out_dir / f"{stem}_report.md"
    report_path.write_text(result.get("report_md", ""), encoding="utf-8")
    structured = {k: v for k, v in result.items()
                  if k not in ("meta", "threads", "corpus_text", "report_md",
                               "threads_mini", "timings", "usage")}
    (out_dir / f"{stem}_structure.json").write_text(
        json.dumps(structured, ensure_ascii=False, indent=1), encoding="utf-8")

    # 运行日志：每次分析追加一行（节点耗时 / token / 估算成本），供事后对照与成本校准
    usage = result.get("usage") or {}
    entry = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bvid": meta.get("bvid"), "title": meta.get("title", ""),
        "nodes": result.get("timings") or {},
        "tokens": {k: usage.get(k, 0) for k in
                   ("prompt_tokens", "completion_tokens", "total_tokens", "calls")},
        "cache_hits": usage.get("cache_hits", 0),
        "cost_cny_est": round(
            ((usage.get("prompt_tokens", 0) * PRICE_PER_M_IN
              + usage.get("completion_tokens", 0) * PRICE_PER_M_OUT) / 1e6), 4),
        "degraded": bool((result.get("claims") or {}).get("degraded")),
    }
    with (out_dir / "_run_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return report_path


def main():
    ap = argparse.ArgumentParser(description="运行评论区思辨分析管线")
    ap.add_argument("json_path", help="抓取脚本输出的评论 JSON")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports"),
                    help="报告输出目录（默认 reports/）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做预处理并估算 token/成本，不调用模型")
    ap.add_argument("--no-cache", action="store_true",
                    help="禁用本地 LLM 缓存，强制重新调用模型")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise SystemExit(f"找不到评论 JSON: {json_path}")

    print("① 读取并预处理评论…")
    if args.dry_run:
        meta, threads = load_threads(json_path)
        corpus = build_corpus_text(threads)
        chars = estimate_chars(threads)
        print(f"  视频: {meta.get('title', '?')}")
        print(f"  参与分析: {len(threads)} 条主评论，约 {chars} 字符 ≈ "
              f"{chars//2}~{chars//1} token（中文约 1 token/字）")
        est_in = chars * 0.8 * 4   # 9 次调用里 4 次喂全篇语料（中文约 0.8 token/字）
        est_out = 30000
        cost = (est_in * PRICE_PER_M_IN + est_out * PRICE_PER_M_OUT) / 1e6
        print(f"\n  --dry-run：当前管线共 9 次模型调用（其中 4 次喂全篇语料），")
        print(f"  粗估总成本约 ¥{cost*0.7:.2f}~¥{cost*1.5:.2f}；"
              f"真实用量以 reports/_run_log.jsonl 的实跑记录为准。")
        print("  同一数据文件重复分析会命中本地缓存（data/llm_cache/），几乎零成本；")
        print("  强制重新调用模型：加 --no-cache。")
        return

    print("② 运行分析管线（约 1~3 分钟）…")
    result = analyze_comments(json_path, use_cache=not args.no_cache)
    path = save_report(result, Path(args.out))
    print(f"✅ 报告已保存: {path}")


if __name__ == "__main__":
    main()
