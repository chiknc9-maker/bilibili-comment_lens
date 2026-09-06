"""llm.py — DeepSeek API 客户端（httpx 直连，零框架依赖）

DeepSeek 兼容 OpenAI 协议。M1-v1 用 JSON mode 做结构化输出：
POST /chat/completions + response_format={"type":"json_object"}，
系统提示词里必须出现 "json" 字样并给出期望的 JSON 形状。

key 从项目根 .env 读取（DEEPSEEK_API_KEY），不落任何日志。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV = dotenv_values(PROJECT_ROOT / ".env")


def _strip_json(text: str) -> str:
    """兼容模型偶尔在 JSON 外包 ```json 代码块的情况。"""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


class LLMError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self):
        api_key = ENV.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise LLMError(
                "未找到 DEEPSEEK_API_KEY：请复制 .env.example 为 .env，"
                "填入 platform.deepseek.com 申请的 API key（只存在本地）"
            )
        self.api_key = api_key
        self.base_url = ENV.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = ENV.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
            trust_env=False,  # 同抓取脚本：忽略系统代理（DeepSeek 国内直连）
        )

    def chat_json(self, system: str, user: str,
                  temperature: float = 0.2, max_tokens: int = 4096,
                  retries: int = 2) -> dict:
        """调用模型并要求返回 JSON 对象。失败自动重试并解析。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(_strip_json(content))
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"模型调用失败（重试 {retries} 次）: {last_err}")

    def ping(self) -> str:
        """连通性测试：让模型回一句简短文本。"""
        resp = self.client.post("/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "user", "content": "只回复两个字：正常"}],
            "max_tokens": 16,
            "stream": False,
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
