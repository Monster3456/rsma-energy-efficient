"""
Qoder Cloud Agents HTTP API 封装。

提供 Agent 管理、Session 管理、消息收发、SSE 流解析等功能，
替代 qodercli 命令行调用，支持多智能体协调（Managed Agents）。
"""

import json
import os
import time
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# ===========================================================================
# 常量
# ===========================================================================

DEFAULT_BASE_URL = "https://api.qoder.com/api/v1/cloud"
SSE_TIMEOUT = 600  # SSE 流超时 10 分钟（Cantus max effort 推理较慢）
HEARTBEAT_INTERVAL = 15  # CAS 心跳间隔


# ===========================================================================
# 数据类
# ===========================================================================

@dataclass
class AgentInfo:
    id: str
    name: str
    version: int = 1


# ===========================================================================
# 异常
# ===========================================================================

class CloudAgentsError(Exception):
    pass


class CloudAgentsAuthError(CloudAgentsError):
    pass


class CloudAgentsTimeoutError(CloudAgentsError):
    pass


class CloudAgentsQuotaError(CloudAgentsError):
    """额度/配额耗尽（403 credit limit / permission_error），属暂时性外部错误。"""
    pass


# ===========================================================================
# 客户端
# ===========================================================================

class CloudAgentsClient:
    """Qoder Cloud Agents API 客户端。"""

    def __init__(self, pat: Optional[str] = None, base_url: str = DEFAULT_BASE_URL):
        self.pat = pat or os.environ.get("QODER_PAT", "")
        if not self.pat:
            raise CloudAgentsAuthError(
                "QODER_PAT 未设置。请在 ~/.zshrc 中添加: export QODER_PAT=\"your-token\""
            )
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # 配置连接重试适配器（解决轮询时 RemoteDisconnected 问题）
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {self.pat}",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _post(self, path: str, json_data: dict = None) -> dict:
        """POST 请求，返回 JSON。"""
        url = f"{self.base_url}{path}"
        r = self.session.post(url, json=json_data, timeout=30)
        if r.status_code == 401:
            raise CloudAgentsAuthError(f"认证失败 (401): {r.text}")
        if r.status_code == 403 and any(
            k in r.text for k in ("credit", "quota", "permission_error", "limit has been reached")
        ):
            raise CloudAgentsQuotaError(f"额度耗尽 (403): {r.text[:200]}")
        if r.status_code >= 400:
            raise CloudAgentsError(f"POST {path} 失败 ({r.status_code}): {r.text}")
        return r.json()

    def _get(self, path: str, params: dict = None) -> dict:
        """GET 请求，返回 JSON。"""
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=params, timeout=30)
        if r.status_code == 401:
            raise CloudAgentsAuthError(f"认证失败 (401): {r.text}")
        if r.status_code >= 400:
            raise CloudAgentsError(f"GET {path} 失败 ({r.status_code}): {r.text}")
        return r.json()

    # ------------------------------------------------------------------
    # 环境管理
    # ------------------------------------------------------------------

    def get_environments(self) -> List[dict]:
        data = self._get("/environments")
        return data.get("data", [])

    def get_or_create_environment(self, name: str = "Default") -> str:
        """获取或创建环境，返回 ENV_ID。"""
        envs = self.get_environments()
        for env in envs:
            if env.get("name") == name:
                return env["id"]
        # 创建新环境
        data = self._post("/environments", {
            "name": name,
            "config": {"type": "cloud", "networking": {"type": "unrestricted"}}
        })
        return data["id"]

    # ------------------------------------------------------------------
    # Agent 管理
    # ------------------------------------------------------------------

    def list_agents(self) -> List[dict]:
        data = self._get("/agents", {"limit": 50})
        return data.get("data", [])

    def create_agent(self, name: str, model: dict, system: str,
                     tools: List[dict] = None,
                     multiagent: dict = None,
                     metadata: dict = None) -> AgentInfo:
        """创建 Agent，返回 AgentInfo。"""
        body: Dict[str, Any] = {
            "name": name,
            "model": model,
            "system": system,
        }
        if tools:
            body["tools"] = tools
        if multiagent:
            body["multiagent"] = multiagent
        if metadata:
            body["metadata"] = metadata
        data = self._post("/agents", body)
        return AgentInfo(id=data["id"], name=data["name"], version=data.get("version", 1))

    def get_agent(self, agent_id: str) -> dict:
        return self._get(f"/agents/{agent_id}")

    # ------------------------------------------------------------------
    # Session 管理
    # ------------------------------------------------------------------

    def create_session(self, agent_id: str, env_id: str,
                       title: str = None, metadata: dict = None) -> str:
        """创建 Session，返回 session_id。"""
        body: Dict[str, Any] = {
            "agent": agent_id,
            "environment_id": env_id,
        }
        if title:
            body["title"] = title
        if metadata:
            body["metadata"] = metadata
        data = self._post("/sessions", body)
        return data["id"]

    def get_session(self, session_id: str) -> dict:
        return self._get(f"/sessions/{session_id}")

    def get_session_status(self, session_id: str) -> str:
        data = self.get_session(session_id)
        return data.get("status", "unknown")

    def cancel_session(self, session_id: str) -> None:
        self._post(f"/sessions/{session_id}/cancel")

    def archive_session(self, session_id: str) -> None:
        self._post(f"/sessions/{session_id}/archive")

    def delete_session(self, session_id: str) -> None:
        url = f"{self.base_url}/sessions/{session_id}"
        self.session.delete(url, timeout=30)

    # ------------------------------------------------------------------
    # 消息收发
    # ------------------------------------------------------------------

    def send_message(self, session_id: str, text: str) -> None:
        """向 Session 发送 user.message 事件。"""
        body = {
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": text}]
                }
            ]
        }
        self._post(f"/sessions/{session_id}/events", body)

    def send_compress_instruction(self, session_id: str) -> None:
        """发送上下文压缩指令，让 Coordinator 总结历史。"""
        text = (
            "【系统指令】上下文窗口接近上限，请执行以下操作：\n"
            "1. 总结到目前为止所有轮次的优化历史（每轮1行：iter-N, decision, score, accepted/rejected）\n"
            "2. 总结失败库和成功库的关键方向\n"
            "3. 总结当前 best score 和瓶颈\n"
            "4. 丢弃旧对话详情，只保留上述摘要\n"
            "请在下一轮回复中直接使用这些摘要作为上下文。"
        )
        self.send_message(session_id, text)

    # ------------------------------------------------------------------
    # 响应等待（异步轮询优先，SSE 流备选）
    # ------------------------------------------------------------------

    def wait_for_response_polling(self, session_id: str,
                                  timeout: int = 900,
                                  poll_interval: int = 15) -> str:
        """异步轮询等待 Agent 响应。不依赖 SSE 流，彻底避免编码/超时问题。

        发送消息后定期轮询 Session 状态，status=idle 后读取事件列表获取 agent.message。
        """
        deadline = time.time() + timeout
        start = time.time()

        while time.time() < deadline:
            try:
                data = self._get(f"/sessions/{session_id}")
                status = data.get("status", "unknown")
                elapsed = int(time.time() - start)

                if status == "idle":
                    # Session 完成，读取事件列表获取 agent.message
                    events_data = self._get(
                        f"/sessions/{session_id}/events",
                        {"limit": 50, "order": "desc"}
                    )
                    events = events_data.get("data", [])
                    messages = []
                    for evt in reversed(events):
                        evt_type = evt.get("type", "")
                        if evt_type == "agent.message":
                            content = evt.get("content", [])
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    messages.append(block.get("text", ""))
                        elif evt_type == "session.error":
                            err = evt.get("error", {})
                            raise CloudAgentsError(
                                f"Session 错误: {err.get('message', '未知错误')}"
                            )
                    if messages:
                        print(f"[CA] 轮询完成 ({elapsed}s)，收到 {len(messages)} 条消息")
                        return "\n".join(messages)
                    # idle 但无 message — 可能是子线程完成，继续等待
                    if elapsed > 30:
                        print(f"[CA] Session idle 但无消息 ({elapsed}s)，可能需要重发")
                        return ""
                elif status == "running":
                    if elapsed % 60 == 0:
                        print(f"[CA] 等待中... ({elapsed}s, status=running)")
                elif status == "terminated":
                    raise CloudAgentsError(f"Session 已终止")
            except CloudAgentsError:
                raise
            except Exception as e:
                print(f"[CA] 轮询异常（非致命）: {e}")

            time.sleep(poll_interval)

        raise CloudAgentsTimeoutError(
            f"轮询超时 ({timeout}s)，Session 仍未完成"
        )

    def wait_for_response(self, session_id: str,
                          timeout: int = SSE_TIMEOUT,
                          last_event_id: str = None) -> str:
        """SSE 流等待 Agent 响应（备选方案）。

        已修复编码 bug（显式设置 utf-8）和 session.error 处理。
        推荐优先使用 wait_for_response_polling。
        """
        url = f"{self.base_url}/sessions/{session_id}/events/stream"
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id

        messages: List[str] = []
        current_event_id = last_event_id or ""
        deadline = time.time() + timeout

        try:
            with self.session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    raise CloudAgentsError(
                        f"SSE 流连接失败 ({resp.status_code}): {resp.text[:200]}"
                    )

                # 修复编码 bug: requests 默认 Latin-1，必须显式设为 utf-8
                resp.encoding = "utf-8"

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if time.time() > deadline:
                        raise CloudAgentsTimeoutError(
                            f"SSE 流超时 ({timeout}s)，已收集 {len(messages)} 条消息"
                        )

                    if line.startswith("id:"):
                        current_event_id = line[3:].strip()
                    elif line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        raw_data = line[5:].strip()
                        if not raw_data:
                            continue
                        try:
                            data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            continue

                        msg_type = data.get("type", event_type if 'event_type' in dir() else "")

                        if msg_type == "heartbeat":
                            continue

                        # 处理 session.error 事件
                        if msg_type == "session.error":
                            err = data.get("error", {})
                            raise CloudAgentsError(
                                f"Session 错误: {err.get('message', '未知')}"
                            )

                        if msg_type == "agent.message":
                            content = data.get("content", [])
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    messages.append(block.get("text", ""))
                            continue

                        if msg_type == "session.status_idle":
                            if messages:
                                return "\n".join(messages)
                            continue

                        if msg_type == "session.status_running":
                            continue

                        if "thread" in msg_type:
                            continue

        except requests.exceptions.Timeout:
            raise CloudAgentsTimeoutError(
                f"SSE 流连接超时 ({timeout}s)，已收集 {len(messages)} 条消息"
            )
        except requests.exceptions.ConnectionError as e:
            if messages:
                return "\n".join(messages)
            raise CloudAgentsError(f"SSE 流连接中断: {e}")

        if messages:
            return "\n".join(messages)
        raise CloudAgentsTimeoutError(
            f"SSE 流结束但无 agent.message 事件，最后 event_id={current_event_id}"
        )

    # ------------------------------------------------------------------
    # 上下文估算
    # ------------------------------------------------------------------

    def estimate_context_tokens(self, session_id: str) -> int:
        """粗略估算 Session 当前上下文 token 用量。

        通过读取历史事件列表，累加 user.message + agent.message 的字符数，
        按 1 token ≈ 3.5 字符 估算。
        """
        try:
            data = self._get(f"/sessions/{session_id}/events", {"limit": 100, "order": "desc"})
            events = data.get("data", [])
            total_chars = 0
            for evt in events:
                evt_type = evt.get("type", "")
                if evt_type in ("user.message", "agent.message"):
                    content = evt.get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total_chars += len(block.get("text", ""))
                elif "thread" in evt_type:
                    # 子线程消息也计入
                    content = evt.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                total_chars += len(block.get("text", ""))
            return int(total_chars / 3.5)  # 粗略估算
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 高级接口
    # ------------------------------------------------------------------

    def send_and_wait(self, session_id: str, text: str,
                      timeout: int = 900) -> str:
        """发送消息并等待 Agent 响应。

        优先使用异步轮询（不受 SSE 编码/超时影响），SSE 流作为备选。
        - 409（Session running）：等待后重试
        - All models failed（模型端间歇性故障）：指数退避重试整个发送+轮询
        """
        model_fail_retries = 3
        for mf_attempt in range(model_fail_retries):
            # 发送消息（含 409 重试）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.send_message(session_id, text)
                    break
                except CloudAgentsError as e:
                    if "409" in str(e) and attempt < max_retries - 1:
                        print(f"[CA] Session running，等待 15s 后重试 ({attempt+1}/{max_retries})")
                        time.sleep(15)
                        continue
                    raise

            # 优先异步轮询（彻底避免 SSE 编码/超时问题）
            try:
                return self.wait_for_response_polling(session_id, timeout=timeout)
            except CloudAgentsTimeoutError:
                print("[CA] 轮询超时，尝试 SSE 流备选...")
                return self.wait_for_response(session_id, timeout=timeout)
            except CloudAgentsError as e:
                # 模型端间歇性故障：退避后重试（非额度、非终止）
                if "All models failed" in str(e) and mf_attempt < model_fail_retries - 1:
                    wait = 30 * (mf_attempt + 1)
                    print(f"[CA] All models failed（模型端间歇故障），等 {wait}s 重试 "
                          f"({mf_attempt+1}/{model_fail_retries})")
                    time.sleep(wait)
                    continue
                raise
        # 不应到达
        raise CloudAgentsError("send_and_wait 重试耗尽")

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def list_session_threads(self, session_id: str) -> List[dict]:
        """列出 Session 中的所有线程（managed agents）。"""
        data = self._get(f"/sessions/{session_id}/threads")
        return data.get("data", [])

    def list_session_events(self, session_id: str, limit: int = 20) -> List[dict]:
        """列出 Session 事件（历史）。"""
        data = self._get(f"/sessions/{session_id}/events", {"limit": limit, "order": "desc"})
        return data.get("data", [])
