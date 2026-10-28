"""打包版入口：双击启动本地服务并自动打开浏览器。

数据（data/）与报告（reports/）写在程序所在目录；
关闭这个黑窗口即停止服务。
"""
from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn

from web.app import app


def free_port() -> int:
    for port in (8000, 8001, 8002, 8003, 8004, 8005):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 8006


def main():
    port = free_port()
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    print(f"评论透镜已启动：http://127.0.0.1:{port}/（浏览器应已自动打开；关闭本窗口即停止）")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
