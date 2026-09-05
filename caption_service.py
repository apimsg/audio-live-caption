"""字幕共享服务：浏览器端（服务端）与桌面悬浮窗（跟随端）共用同一份识别结果。

设计：
- Streamlit 进程内跑唯一的 CaptionEngine + 翻译线程，把每条字幕写进 CaptionStore；
- 同进程起一个只监听 127.0.0.1 的 HTTP 服务，暴露 /captions?since=N 与 /status；
- desktop_caption.py --follow <url> 只做"拉取 + 渲染"，不再加载模型、不再占声卡。

这样两个界面天然双语同步（同一条 caption 的原文/译文绑定在一个对象上）。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class CaptionStore:
    """字幕表 + 增量协议。

    关键设计：**修订号 rev**。任何一条字幕的任何变化（新原文、译文回填）
    都会分配全局递增的 rev，客户端按 rev 拉增量——这样"先发出原文、
    后补上译文"的更新也能被跟随端看到（只按字幕编号过滤会漏掉译文）。

    **纪元 epoch**：新一轮识别（clear）时 epoch+1，跟随端据此丢弃旧缓存重拉。
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._lock = threading.Lock()
        self._items: dict[int, dict] = {}
        self._maxlen = maxlen
        self._rev = 0
        self.epoch = 1
        self.status: dict = {"running": False, "hint": "", "error": "", "epoch": 1}

    def clear(self) -> None:
        """开始新一轮识别：清空条目、epoch+1（rev 永不清零，单调递增）。"""
        with self._lock:
            self._items = {}
            self.epoch += 1
            self.status["epoch"] = self.epoch

    def put(self, index: int, *, src: str = "", trans: str | None = None,
            start: float = 0.0, end: float = 0.0, wall: float = 0.0) -> None:
        with self._lock:
            item = self._items.get(index)
            changed = False
            if item is None:
                if not src:
                    return  # 原文不存在：不接受"孤悬译文"
                item = {"i": index, "src": src, "trans": "", "rev": 0,
                        "start": start, "end": end, "wall": wall}
                self._items[index] = item
                changed = True
                if len(self._items) > self._maxlen:
                    oldest = min(self._items)
                    self._items.pop(oldest, None)
            elif src and item["src"] != src:
                # 原位修订（半句缝合把后条并了进来）：旧译文是"半句"的，必须作废，
                # 否则屏幕上会出现英文整句 + 中文半句的错配。
                item["src"] = src
                item["trans"] = ""
                changed = True
            if trans is not None and item.get("trans") != trans and trans != item["src"]:
                item["trans"] = trans
                changed = True
            if changed:
                self._rev += 1
                item["rev"] = self._rev

    def since(self, rev: int) -> list[dict]:
        with self._lock:
            out = [dict(v) for v in self._items.values() if v["rev"] > rev]
        out.sort(key=lambda d: d["i"])
        return out

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._items.values()]


class _Handler(BaseHTTPRequestHandler):
    store: CaptionStore  # set on the subclass by CaptionServer

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in ("/captions", "/status"):
            self.send_error(404)
            return
        qs = parse_qs(parsed.query)
        since = int(qs.get("since", ["-1"])[0] or -1)
        body = {
            "items": self.store.since(since),
            "status": self.store.status,
        }
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):  # 静音访问日志
        pass


class CaptionServer:
    """127.0.0.1 上的极简 JSON 服务。port=0 自动挑空闲端口。"""

    def __init__(self, store: CaptionStore, host: str = "127.0.0.1", port: int = 0) -> None:
        handler = type("BoundHandler", (_Handler,), {"store": store})
        self._srv = ThreadingHTTPServer((host, port), handler)
        self._srv.daemon_threads = True
        self.host, self.port = self._srv.server_address[:2]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "CaptionServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._srv.shutdown()
            self._srv.server_close()
        except Exception:  # noqa: BLE001
            pass


class CaptionClient:
    """跟随端 HTTP 拉取（桌面窗线程里用，异常由调用方兜底）。"""

    def __init__(self, base_url: str, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(self, since: int = -1) -> dict:
        import urllib.request

        url = f"{self.base_url}/captions?since={since}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
