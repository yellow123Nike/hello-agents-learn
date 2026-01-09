import threading
import logging
from typing import Dict, Optional
import requests
import httpx
from abc import ABC, abstractmethod


class OkHttpUtil:
    _http_session: Optional[requests.Session] = None
    _sse_client: Optional[httpx.Client] = None
    _lock = threading.Lock()

    CONNECT_TIMEOUT = 240
    READ_TIMEOUT = 240

    # 连接池与资源复用
    @classmethod
    def get_http_client(cls) -> requests.Session:
        if cls._http_session is None:
            with cls._lock:
                if cls._http_session is None:
                    session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=1000,
                        pool_maxsize=1000,
                        max_retries=0,
                    )
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                    cls._http_session = session
        return cls._http_session

    @classmethod
    def get_sse_client(cls) -> httpx.Client:
        if cls._sse_client is None:
            with cls._lock:
                if cls._sse_client is None:
                    cls._sse_client = httpx.Client(timeout=None)
        return cls._sse_client

    @staticmethod
    def post_new(url: str, headers: Dict[str, str], json_body: str) -> requests.Response:
        client = OkHttpUtil.get_http_client()
        return client.post(
            url,
            data=json_body,
            headers=headers,
            timeout=(OkHttpUtil.CONNECT_TIMEOUT, OkHttpUtil.READ_TIMEOUT),
        )

    # 强约束调用（失败即错误）
    @staticmethod
    def post_json_body(url: str, headers: Dict[str, str], json_body: str, printer) -> str:
        printer.send("POST %s payload=%s headers=%s", url, json_body, headers)
        response = OkHttpUtil.post_new(url, headers, json_body)
        if not response.ok:
            raise RuntimeError(f"调用接口 {url} 失败: {response.text}")
        return response.text

    # 弱约束调用（失败可接受）
    @staticmethod
    def post_json(
        url: str,
        json_params: str,
        headers: Optional[Dict[str, str]],
        timeout: int,
    ) -> Optional[str]:
        client = OkHttpUtil.get_http_client()
        response = client.post(
            url,
            data=json_params,
            headers=headers,
            timeout=(timeout, timeout),
        )
        return response.text if response.ok else None

    class SseEventListener(ABC):
        @abstractmethod
        def on_event(self, event: str): ...

        @abstractmethod
        def on_complete(self): ...

        @abstractmethod
        def on_error(self, e: Exception): ...

    # 同步 SSE
    @staticmethod
    def request_sse(
        url: str,
        json_body: str,
        headers: Optional[Dict[str, str]],
        event_listener: "OkHttpUtil.SseEventListener",
        printer

    ):
        headers = headers or {}
        printer.send("SSE POST %s payload=%s headers=%s",
                     url, json_body, headers)

        client = OkHttpUtil.get_sse_client()
        try:
            with client.stream(
                "POST",
                url,
                headers={
                    **headers,
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                content=json_body,
            ) as response:
                for chunk in response.iter_text():
                    if not chunk:
                        continue
                    for line in chunk.splitlines():
                        if not line:
                            continue
                        line = line.strip()   # 🔥 关键一行
                        if not line.startswith("data"):
                            continue
                        # 兼容 data:xxxx / data: xxxx
                        _, _, payload = line.partition(":")
                        payload = payload.strip()

                        event_listener.on_event(payload)

                event_listener.on_complete()
        except Exception as e:
            event_listener.on_error(e)
