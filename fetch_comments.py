"""fetch_comments.py — M0：抓取 B站视频的完整楼中楼评论，保存为 JSON

用法:
    python fetch_comments.py BV1RhtV6iEt3
    python fetch_comments.py "https://www.bilibili.com/video/BV1RhtV6iEt3/"
    python fetch_comments.py BV1RhtV6iEt3 --main-total 20 --no-subs   # 小规模调试

抓取范围（默认）:
    热评排序前 100 条主评论（每页 20 条），每条主评论抓取其楼中楼前 20 条回复。
    子评论下更深层（三级对话）暂不抓取，仅保留 rcount 数值作为"争论深度"信号。

风控说明:
    匿名可抓取，但高频请求会触发风控(-412)。如遇风控，在 .env 中填入自己的
    SESSDATA（获取方法见 README），并加大 --delay。
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import dotenv_values

API = "https://api.bilibili.com"
TYPE_VIDEO = "1"  # 视频评论的 type 值
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
OUT_DIR = Path(__file__).parent / "data"

sys.stdout.reconfigure(encoding="utf-8")


def parse_bvid(raw: str) -> str:
    """接受 BV 号或完整视频链接，返回 BV 号。解析失败抛 ValueError。"""
    m = re.search(r"(BV[0-9A-Za-z]{10})", raw)
    if m:
        return m.group(1)
    raise ValueError(f"无法从输入中解析出 BV 号: {raw!r}（应形如 BV1xx...）")


def parse_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def build_headers() -> dict:
    """构造请求头，保持"身份一致"：
    - 匿名时只带 UA（实测：无 cookie 却带 Referer 会触发 B站 412 风控）
    - 提供 cookie 时才带 Referer + Cookie，模拟完整浏览器身份
    """
    env = dotenv_values(Path(__file__).parent / ".env")
    cookies = []
    if env.get("SESSDATA"):
        cookies.append(f"SESSDATA={env['SESSDATA']}")
    if env.get("BUVID3"):
        cookies.append(f"buvid3={env['BUVID3']}")
    headers = {"User-Agent": UA}
    if cookies:
        headers["Referer"] = "https://www.bilibili.com/"
        headers["Cookie"] = "; ".join(cookies)
    return headers


def api_get(client: httpx.Client, path: str, params: dict) -> dict:
    """请求接口并校验 code，带一次重试。风控时给出友好提示。"""
    for attempt in range(2):
        try:
            resp = client.get(API + path, params=params)
        except httpx.HTTPError as e:
            print(f"  [网络错误] {path} {e}，重试…")
            time.sleep(1.5)
            continue
        try:
            j = resp.json()
        except ValueError:
            print(f"  [非 JSON 响应] status={resp.status_code} "
                  f"body 前 120 字: {resp.text[:120]!r}")
            if resp.status_code == 412:
                raise SystemExit(
                    "B站风控 (412)：请降低 --delay 或检查网络出口。"
                    "若使用代理，B站需走国内直连（境外 IP 会被风控）。"
                )
            time.sleep(1.5)
            continue
        code = j.get("code")
        if code == 0:
            return j["data"]
        if code in (-412, -352, -101):
            raise SystemExit(
                f"接口风控/鉴权失败 (code={code})：请求太快或需要登录。\n"
                f"  解决：在 .env 中填入 SESSDATA（见 README），或加大 --delay。\n"
                f"  路径: {path} params: {params}"
            )
        print(f"  [code={code}] {j.get('message')} 路径 {path}，重试…")
        time.sleep(1.5)
    raise RuntimeError(f"请求多次失败: {path} {params}")


def normalize_reply(rep: dict) -> dict:
    """把接口返回的一条评论精简为我们要的字段。"""
    member = rep.get("member") or {}
    content = rep.get("content") or {}
    return {
        "rpid": rep["rpid"],
        "mid": member.get("mid"),
        "uname": member.get("uname", ""),
        "like": rep.get("like", 0),
        "ctime": parse_ts(rep.get("ctime", 0)),
        "rcount": rep.get("rcount", 0),  # 楼中楼评论总数（更深层未抓的信号）
        "text": content.get("message", ""),
    }


def fetch_replies(client: httpx.Client, aid: int, root_rpid: int,
                  limit: int, delay: float, pn_start: int = 1) -> list:
    """抓取 root_rpid 下的直接回复（按热度，最多 limit 条）。"""
    out, pn = [], pn_start
    while len(out) < limit:
        d = api_get(client, "/x/v2/reply/reply",
                    {"type": TYPE_VIDEO, "oid": aid, "root": root_rpid,
                     "pn": pn, "ps": 20, "mode": 3})
        batch = d.get("replies") or []
        if not batch:
            break
        out.extend(normalize_reply(r) for r in batch)
        pn += 1
        time.sleep(delay)
    return out[:limit]


DEPTH3_FLOOR = 10      # 主评论楼中楼总数达到此值才下钻三级（控制请求量）


def fetch_thread_subs(client: httpx.Client, aid: int, root_rpid: int,
                      sub_limit: int, delay: float,
                      depth3_limit: int = 3, main_rcount: int = 0,
                      depth3_floor: int = DEPTH3_FLOOR) -> list:
    """抓取一条主评论的楼中楼回复（最多 sub_limit 条，按热度）。

    重要机制：B站 reply 接口已不再返回二级评论的 rcount（恒 0），但
    root=<二级rpid> 会返回以该二级为根的对话树多层展开（扁平列表，
    实测单次请求可含至五六级的后代，每项带 parent 指针）。
    因此对"争论深"（主评论 rcount >= floor）的楼，逐个请求二级，
    用 fetch_subtree 把返回数据本地重组装成完整嵌套树。
    """
    subs = fetch_replies(client, aid, root_rpid, sub_limit, delay)
    if depth3_limit > 0 and main_rcount >= depth3_floor:
        for s in subs:
            s["replies"] = fetch_subtree(client, aid, s["rpid"],
                                         depth3_limit, delay)
    return subs


def fetch_subtree(client: httpx.Client, aid: int, sub_rpid: int,
                  child_limit: int, delay: float) -> list:
    """下钻二级评论并本地重组装其对话树（直到接口返回的叶子）。

    接口一次返回以 sub_rpid 为根的多层展开（含其自身与各级后代，可能
    混入同主楼其他分支）。重组装规则：从 sub_rpid 自身出发，凡 parent
    落在已收集集合中的项都收为对应父节点的子级；无 parent 可达的项
    （其他分支）丢弃。每层按点赞数截断 child_limit 条。
    """
    d = api_get(client, "/x/v2/reply/reply",
                {"type": TYPE_VIDEO, "oid": aid, "root": sub_rpid,
                 "pn": 1, "ps": 20, "mode": 3})
    batch = d.get("replies") or []
    items = {r["rpid"]: r for r in batch if r.get("rpid")}
    root = items.get(sub_rpid)
    time.sleep(delay)
    if root is None:
        # 根自身未出现在返回中（罕见）：退化为取 parent 指向自身的直接子
        return [normalize_reply(r) for r in batch
                if r.get("parent") == sub_rpid][:child_limit]

    # 收集集合：从根开始，把 parent 在集合内的项逐轮并入
    collected = {root["rpid"]}
    changed = True
    while changed:
        changed = False
        for r in batch:
            rp = r.get("rpid")
            if rp in collected or not r.get("parent"):
                continue
            if r["parent"] in collected:
                collected.add(rp)
                changed = True

    # 组装嵌套：children[rpid] -> [子项]
    def build(node: dict) -> dict:
        kids = [r for r in batch if r.get("parent") == node["rpid"]
                and r["rpid"] != node["rpid"] and r["rpid"] in collected]
        kids.sort(key=lambda r: r.get("like", 0), reverse=True)
        node["replies"] = [build(r) for r in kids[:child_limit]]
        return node

    built = build(root)
    return built["replies"]


def fetch_video(bvid: str, main_total: int = 100, no_subs: bool = False,
                sub_limit: int = 20, depth3_limit: int = 5,
                depth3_floor: int = 10, delay: float = 0.4,
                proxy: str | None = None) -> dict:
    """抓取单个视频的评论（含深树对话），返回完整数据 dict（含 meta）。

    供 CLI 与网页服务共用。headers 在内部按 cookie 配置构建。
    """
    bvid = parse_bvid(bvid)
    headers = build_headers()

    # trust_env=False：忽略系统代理（B站是国内站，直连更稳，且部分系统的
    # NO_PROXY 写法会触发 httpx 解析 bug）。需要代理时显式传入。
    with httpx.Client(headers=headers, timeout=20,
                      trust_env=False, proxy=proxy) as client:
        # 1) 视频信息
        view = api_get(client, "/x/web-interface/view", {"bvid": bvid})
        aid = view["aid"]
        stat = view.get("stat") or {}
        print(f"视频: {view['title']}  (aid={aid})")
        print(f"  播放 {stat.get('view', '?')} / 弹幕 {stat.get('danmaku', '?')} "
              f"/ 评论 {stat.get('reply', '?')} / 硬币 {stat.get('coin', '?')}")

        # 2) 主评论翻页（mode=3 热度排序）
        threads, next_p = [], 1
        while next_p and (main_total == 0 or len(threads) < main_total):
            d = api_get(client, "/x/v2/reply/main",
                        {"type": TYPE_VIDEO, "oid": aid, "mode": 3, "next": next_p})
            cursor = d.get("cursor") or {}
            for rep in d.get("replies") or []:
                t = normalize_reply(rep)
                if not t["text"].strip():
                    continue
                if t["rcount"] > 0 and not no_subs:
                    t["replies"] = fetch_thread_subs(
                        client, aid, t["rpid"], sub_limit, delay,
                        depth3_limit, t["rcount"], depth3_floor)
                else:
                    t["replies"] = []
                threads.append(t)
                if main_total and len(threads) >= main_total:
                    break
            if cursor.get("is_end"):
                break
            next_p = cursor.get("next")
            time.sleep(delay)

        if not threads:
            raise ValueError("没有抓到任何评论（视频可能关闭了评论区）。")

        return {
            "meta": {
                "bvid": bvid,
                "aid": aid,
                "title": view["title"],
                "owner": (view.get("owner") or {}).get("name", ""),
                "stat": {k: stat.get(k) for k in ("view", "danmaku", "reply", "like")},
                "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
                "config": {"main_total_cap": main_total,
                           "sub_limit": sub_limit,
                           "depth3_limit": depth3_limit,
                           "depth3_floor": depth3_floor,
                           "sort": "hot"},
            },
            "threads": threads,
        }


def main():
    ap = argparse.ArgumentParser(description="抓取 B站视频评论（含楼中楼）")
    ap.add_argument("bvid", help="BV 号或视频链接")
    ap.add_argument("--main-total", type=int, default=100,
                    help="抓取多少条主评论（默认 100，0=不限）")
    ap.add_argument("--no-subs", action="store_true",
                    help="不抓楼中楼，只抓主评论")
    ap.add_argument("--sub-limit", type=int, default=20,
                    help="每条主评论最多抓多少条楼中楼回复（默认 20）")
    ap.add_argument("--depth3-limit", type=int, default=5,
                    help="深树下钻时每个节点最多保留多少条直接回复（默认 5，"
                         "树深度由接口返回决定，可达五六级；0=不下钻）")
    ap.add_argument("--depth3-floor", type=int, default=10,
                    help="主评论楼中楼总数达到此值才下钻三级（默认 10，"
                         "控制请求量）")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="请求间隔秒数（默认 0.4，风控时加大）")
    ap.add_argument("--proxy", default=None,
                    help="手动指定代理，如 http://127.0.0.1:7892")
    args = ap.parse_args()

    out = fetch_video(args.bvid, main_total=args.main_total,
                      no_subs=args.no_subs, sub_limit=args.sub_limit,
                      depth3_limit=args.depth3_limit,
                      depth3_floor=args.depth3_floor,
                      delay=args.delay, proxy=args.proxy)

    def tree_count(items) -> int:
        return sum(1 + tree_count(x.get("replies") or []) for x in items)

    threads = out["threads"]
    n_subs = sum(len(t["replies"]) for t in threads)
    n_tree = sum(tree_count(t["replies"]) for t in threads)
    n_deep = sum(1 for t in threads if t["rcount"] > len(t["replies"]))
    print(f"\n完成: 主评论 {len(threads)} 条, 直接回复 {n_subs} 条, "
          f"深树对话共 {n_tree} 条")
    print(f"仍有更深层争论的楼 {n_deep} 个（可调大 depth3_floor 覆盖更多楼）")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{out['meta']['bvid']}_comments.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
