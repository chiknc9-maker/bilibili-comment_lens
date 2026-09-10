"""M2 网页服务：粘贴 B站链接 → 后台抓取+分析 → 返回思辨报告

启动:
    .venv\\Scripts\\python.exe -m uvicorn web.app:app --port 8000
然后浏览器打开 http://127.0.0.1:8000/

任务状态在内存里（MVP 简化）：单机自用足够；重启服务即清空历史任务。
"""
from __future__ import annotations

import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fetch_comments as fc        # noqa: E402
import run_analysis as ra          # noqa: E402

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


def _set_status(task_id: str, **fields):
    with _lock:
        _tasks[task_id].update(fields)


def _run_task(task_id: str, bvid: str):
    try:
        _set_status(task_id, status="fetching",
                    message="正在抓取评论区（含深树对话，约 1~5 分钟）…")
        data = fc.fetch_video(bvid, main_total=100, delay=0.35)

        DATA_DIR.mkdir(exist_ok=True)
        json_path = DATA_DIR / f"{bvid}_comments.json"
        json_path.write_text(
            __import__("json").dumps(data, ensure_ascii=False, indent=1),
            encoding="utf-8")

        _set_status(task_id, status="analyzing",
                    message="正在分析（立场地图 / 垃圾解剖 / 社会学观察，约 30~60 秒）…")
        result = ra.analyze_comments(json_path)
        ra.save_report(result, REPORT_DIR)

        _set_status(task_id, status="done", message="完成",
                    report_md=result.get("report_md", ""),
                    title=(result.get("meta") or {}).get("title", bvid))
    except Exception as e:  # 后台线程兜底：任何失败都转为任务错误
        _set_status(task_id, status="error", message=f"失败：{e}")


@app.post("/api/analyze")
def start_analyze(req: AnalyzeRequest):
    try:
        bvid = fc.parse_bvid(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    task_id = uuid.uuid4().hex[:12]
    with _lock:
        _tasks[task_id] = {"status": "queued", "message": "排队中…"}
    threading.Thread(target=_run_task, args=(task_id, bvid), daemon=True).start()
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
            "has_report": "report_md" in t}


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
