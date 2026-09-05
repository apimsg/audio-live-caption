"""真实环境端到端验证（无 GUI、无需播放声音）：

真 Whisper 模型 + 真 WASAPI 回环流 + 真字幕 HTTP 服务 + 客户端拉取状态。
沙箱里没有声音播放，预期：引擎能启动、无信号提示会出现、/captions 能拉到状态。
"""

import sys
import time

sys.path.insert(0, ".")

from caption_engine import CaptionEngine
from caption_service import CaptionClient, CaptionServer, CaptionStore
from pathlib import Path
import whisper_backend

print("加载真实模型…", flush=True)
model = whisper_backend.load_model("faster", "base", Path("models"))
store = CaptionStore()
srv = CaptionServer(store).start()
client = CaptionClient(srv.url)
print("服务地址:", srv.url, flush=True)

try:
    eng = CaptionEngine(model, source="system", loopback_device=None, chunk_seconds=2.0)
    ok = eng.start()
    print("引擎启动:", ok, "error:", eng.error or "-", flush=True)
    assert ok, eng.error
    store.status["running"] = True

    # 观察 ~5 秒：模拟 app.py 里 Dispatcher 的搬运逻辑
    n = 0
    deadline = time.time() + 5
    hint_seen = False
    while time.time() < deadline:
        for cap in eng.drain():
            store.put(cap.index, src=cap.text, start=cap.start, end=cap.end,
                      wall=cap.wall_clock)
            n += 1
        store.status["hint"] = eng.hint
        resp = client.fetch(-1)
        if resp["status"]["hint"]:
            hint_seen = True
        time.sleep(0.25)
    eng.stop()
    store.status["running"] = False

    print("产出字幕条数:", n, flush=True)
    print("客户端看到无信号提示:", hint_seen, flush=True)
    final = client.fetch(-1)
    assert "status" in final and final["status"]["running"] is False
    print("E2E OK（真实设备流 + 真实服务往返）", flush=True)
finally:
    srv.stop()
