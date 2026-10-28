"""llm.py — DeepSeek API 客户端（httpx 直连，零框架依赖）

DeepSeek 兼容 OpenAI 协议。M1-v1 用 JSON mode 做结构化输出：
POST /chat/completions + response_format={"type":"json_object"}，
系统提示词里必须出现 "json" 字样并给出期望的 JSON 形状。

key 从项目根 .env 读取（DEEPSEEK_API_KEY），不落任何日志。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV = dotenv_values(PROJECT_ROOT / ".env")
CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache"   # 本地 LLM 缓存目录（已 gitignore）


def _strip_json(text: str) -> str:
    """兼容模型偶尔在 JSON 外包 ```json 代码块的情况。"""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def _scan_objects(text: str, start: int) -> list[str]:
    """从 `[` 之后开始扫描，收集其中**已完整**的顶层 `{...}` 对象文本。

    只做机械扫描、不猜内容：跟踪字符串/转义状态与花括号深度，
    深度回到 0 即视为一个完整对象；遇到不完整的尾巴就停下丢弃。
    """
    objs: list[str] = []
    depth = 0
    buf: list[str] = []
    in_str = False
    esc = False
    for ch in text[start:]:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                objs.append("".join(buf))
                buf = []
        elif depth > 0:
            buf.append(ch)
    return objs


def salvage_partial_json(raw: str,
                         keys: tuple[str, ...] = ("claims", "non_argument_comments")) -> dict:
    """从被 max_tokens 截断的 JSON 里抢救出已完整的数组条目。

    命题提取的输出一旦超限就被拦腰砍断，整份丢弃的代价太大（一个视频白跑几分钟）。
    这里把完整的前缀条目捞回来让管线继续，覆盖会变窄——调用方有责任把这件事
    告诉使用者，而不是假装报告是完整的。
    """
    out: dict = {}
    for k in keys:
        m = re.search(r'"%s"\s*:\s*\[' % re.escape(k), raw or "")
        if not m:
            continue
        items = []
        for obj in _scan_objects(raw, m.end()):
            try:
                items.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
        if items:
            out[k] = items
    return out


class LLMError(RuntimeError):
    pass


class LLMTruncatedError(LLMError):
    """输出被 max_tokens 截断（finish_reason=length），JSON 不完整。

    raw 保留不完整原文，调用方可以据此降档重试，或抢救已完整的条目。
    """

    def __init__(self, message: str, raw: str = "", completion_tokens: int = 0):
        super().__init__(message)
        self.raw = raw
        self.completion_tokens = completion_tokens


class DeepSeekClient:
    def __init__(self, cache_enabled: bool | None = None):
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
        # 最近一次调用的服务端信息（供运行日志与诊断读取，不参与业务逻辑）
        self.last_usage: dict = {}
        self.last_finish_reason: str | None = None
        # 运行期累计（供运行日志/成本统计，跨所有 chat_json 调用）
        self.total_usage: dict = {}
        self.call_count: int = 0
        # 本地 LLM 缓存：同 prompt 同参数直接复用结果，重复分析不再花钱
        # 默认开；LLM_CACHE=0 或构造参数 cache_enabled=False 关闭
        self.cache_enabled = (
            ENV.get("LLM_CACHE", "1").strip().lower() not in ("0", "false", "off")
            if cache_enabled is None else cache_enabled)
        self.cache_hits = 0

    @staticmethod
    def _cache_key(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            .encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / (key + ".json")

    def _cache_get(self, key: str):
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("result")
        except (json.JSONDecodeError, OSError):
            return None

    def _cache_put(self, key: str, result: dict, usage: dict):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {"result": result, "usage": usage,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            tmp = self._cache_path(key).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._cache_path(key))   # 原子替换，并发安全
        except OSError:
            pass   # 缓存写失败不致命，照常返回模型结果

    def chat_json(self, system: str, user: str,
                  temperature: float = 0.2, max_tokens: int = 4096,
                  retries: int = 2) -> dict:
        """调用模型并要求返回 JSON 对象。失败自动重试并解析。

        截断（finish_reason=length）单独识别：这类失败是"要得太多"造成的，
        原样重试收益极低，所以最多试 2 次就抛 LLMTruncatedError（带 raw），
        由调用方决定降档重试还是抢救已完整的条目；
        其他错误（网络、JSON 畸形）仍按 retries 原样重试。
        """
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
        key = self._cache_key(payload)
        # 命中缓存：直接返回，不占 token 不计成本
        if self.cache_enabled:
            hit = self._cache_get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit

        last_err = None
        trunc_tries = 0
        for attempt in range(retries + 1):
            try:
                resp = self.client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                j = resp.json()
                choice = j["choices"][0]
                content = choice.get("message", {}).get("content") or ""
                self.last_usage = j.get("usage") or {}
                self.last_finish_reason = choice.get("finish_reason")
                self.call_count += 1
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.total_usage[k] = self.total_usage.get(k, 0) + (self.last_usage.get(k) or 0)
                if self.last_finish_reason == "length":
                    trunc_tries += 1
                    done = self.last_usage.get("completion_tokens") or 0
                    last_err = LLMTruncatedError(
                        "输出被 max_tokens=%d 截断（completion_tokens=%s）" % (max_tokens, done),
                        raw=content, completion_tokens=done)
                    if trunc_tries >= 2:   # 边界情况下第二次常能过，但不再多试
                        raise last_err
                    time.sleep(1.5 * (attempt + 1))
                    continue
                result = json.loads(_strip_json(content))
                if self.cache_enabled:
                    self._cache_put(key, result, self.last_usage)
                return result
            except LLMTruncatedError:
                raise
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
