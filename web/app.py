"""M2 网页服务：粘贴 B站链接 → 后台抓取+分析 → 返回思辨报告

启动:
    .venv\\Scripts\\python.exe -m uvicorn web.app:app --port 8000
然后浏览器打开 http://127.0.0.1:8000/

任务状态在内存里（MVP 简化）：单机自用足够；重启服务即清空历史任务。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

def _app_dir() -> Path:
    """打包后取 exe 所在目录（数据/报告写在旁边），开发时取项目根。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _app_dir()
sys.path.insert(0, str(PROJECT_ROOT))

import fetch_comments as fc        # noqa: E402
import run_analysis as ra          # noqa: E402
from analyze.llm import ENV as LLM_ENV  # noqa: E402

app = FastAPI(title="评论透镜")

# 允许任意来源（本机自用）：兼容用户直接双击 index.html（file:// 源）打开
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tasks: dict[str, dict] = {}
_lock = threading.Lock()

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"


class AnalyzeRequest(BaseModel):
    url: str
    # 可选：用户自带模型配置（不填则用本机 .env 的默认配置）。
    # api_key 只在本进程内存中使用，绝不写入任务状态或日志。
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    price_in: float | None = None
    price_out: float | None = None


def _set_status(task_id: str, **fields):
    with _lock:
        _tasks[task_id].update(fields)


MAX_ATTEMPTS = 3
FETCH_RETRY_DELAYS = [90, 180]    # 抓取失败（风控/网络）需要冷却，等久一点
ANALYSIS_RETRY_DELAYS = [15, 30]  # 分析失败很快能再试（本地缓存让重试几乎免费）


def _retryable(e: BaseException) -> bool:
    """哪些失败值得自动重试：缺 key / 鉴权失败 / 非法输入是确定性的，不重试。"""
    msg = str(e)
    if ("DEEPSEEK_API_KEY" in msg or "API key" in msg
            or "'401" in msg or "'403" in msg):
        return False
    if isinstance(e, ValueError):
        return False
    return True


def _fetch_and_save(bvid: str) -> Path:
    """抓取并落盘。fetch 内部用 SystemExit 表达风控中断，
    转成普通异常，否则这里会漏捕、任务永远卡在抓取中。"""
    try:
        data = fc.fetch_video(bvid, main_total=100, delay=0.35)
    except SystemExit as e:
        raise RuntimeError(f"抓取被中断：{e}") from e
    DATA_DIR.mkdir(exist_ok=True)
    json_path = DATA_DIR / f"{bvid}_comments.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return json_path


def _run_task(task_id: str, bvid: str, cfg: dict | None = None):
    """后台执行一个分析任务；失败自动重试（最多 MAX_ATTEMPTS 次）。

    抓取失败（B站风控/网络）与中途分析失败都会重试，风控类等待更久；
    确定性错误（缺 key、鉴权失败）不重试，直接报错并给出原因。
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        stage = "fetch"
        try:
            _set_status(task_id, status="fetching",
                        message="正在抓取评论区（含深树对话，约 1~5 分钟）…")
            json_path = _fetch_and_save(bvid)
            stage = "analysis"

            _set_status(task_id, status="analyzing",
                        message="正在分析（命题还原 / 观点地图 / 论辩关系 / 论证诊断 / "
                                "非论证解剖…，约 1~3 分钟）…")
            result = ra.analyze_comments(json_path, llm_config=cfg)
            ra.save_report(result, REPORT_DIR)

            claims_meta = result.get("claims") or {}
            tier = claims_meta.get("claim_tier", 50)
            _set_status(task_id, status="done", message="完成",
                        report_md=result.get("report_md", ""),
                        title=(result.get("meta") or {}).get("title", bvid),
                        degraded=bool(claims_meta.get("degraded")) or tier < 50,
                        claim_tier=tier)
            return
        except BaseException as e:  # 含 SystemExit：任何中断都转为任务状态，绝不悬挂
            if attempt >= MAX_ATTEMPTS or not _retryable(e):
                _set_status(task_id, status="error",
                            message=f"失败（第 {attempt}/{MAX_ATTEMPTS} 次尝试）：{e}")
                return
            delays = FETCH_RETRY_DELAYS if stage == "fetch" else ANALYSIS_RETRY_DELAYS
            wait = delays[attempt - 1]
            _set_status(task_id, status="retrying",
                        message=f"第 {attempt}/{MAX_ATTEMPTS} 次失败：{str(e)[:120]}；"
                                f"{wait} 秒后自动重试…")
            time.sleep(wait)


@app.post("/api/analyze")
def start_analyze(req: AnalyzeRequest):
    try:
        bvid = fc.parse_bvid(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # 缺 key 是确定性错误：立刻挡掉，别让用户白等一两分钟抓取
    if not ((req.api_key or "").strip()
            or (LLM_ENV.get("DEEPSEEK_API_KEY") or "").strip()):
        raise HTTPException(
            status_code=400,
            detail="还没有 API Key：请展开页面上的「高级设置」填写自己的 key"
                   "（DeepSeek 或其他 OpenAI 兼容服务），或在 .env 里配置 DEEPSEEK_API_KEY。",
        )
    task_id = uuid.uuid4().hex[:12]
    with _lock:
        _tasks[task_id] = {"status": "queued", "message": "排队中…"}
    cfg = {k: v for k, v in {"api_key": req.api_key, "base_url": req.base_url,
                             "model": req.model, "price_in": req.price_in,
                             "price_out": req.price_out}.items() if v is not None}
    threading.Thread(target=_run_task, args=(task_id, bvid, cfg), daemon=True).start()
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    with _lock:
        t = _tasks.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在（服务重启后历史任务清空）")
    return {"task_id": task_id,
            "status": t.get("status"),
            "message": t.get("message", ""),
            "title": t.get("title"),
            "has_report": "report_md" in t,
            "degraded": bool(t.get("degraded")),
            "claim_tier": t.get("claim_tier")}


@app.get("/api/tasks/{task_id}/report")
def get_report(task_id: str):
    with _lock:
        t = _tasks.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t.get("status") != "done":
        raise HTTPException(status_code=409, detail="任务未完成")
    return {"title": t.get("title", ""), "report_md": t["report_md"]}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static",
                           html=True), name="static")
