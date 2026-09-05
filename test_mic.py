"""真机麦克风冒烟：调用 engine.start() 打开物理设备并确认不报错。

不校验识别内容（环境静音即可），只验证设备打开、线程存活、可干净停止。
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from caption_engine import CaptionEngine  # noqa: E402

import sounddevice as sd  # noqa: E402


class _StubModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, *a, **k):
        self.calls += 1
        return {"text": f"seg{self.calls}"}


inputs = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
print("input devices:", inputs)

failures = []
for dev in inputs[:4]:
    stub = _StubModel()
    eng = CaptionEngine(stub, device=dev, chunk_seconds=0.5)
    ok = eng.start()
    if not ok:
        failures.append((dev, eng.error))
        print(f"dev {dev}: FAILED -> {eng.error}")
        continue
    time.sleep(1.2)
    alive = eng.running
    caps = eng.drain()
    eng.stop()
    print(
        f"dev {dev}: started={ok} alive={alive} caps={len(caps)} "
        f"rms={eng.last_rms:.5f} calls={stub.calls}"
    )
    assert alive, f"dev {dev} 线程未存活"
    assert eng.error == "", f"dev {dev} 运行报错: {eng.error}"
    # 停止后线程应退出
    assert not eng.running, f"dev {dev} 停止后仍在运行"

# 默认设备路径（device=None）也不应崩溃
eng = CaptionEngine(_StubModel(), device=None, chunk_seconds=0.5)
ok = eng.start()
print("default device start:", ok, eng.error or "")
if ok:
    time.sleep(0.8)
    eng.stop()

print("MIC SMOKE DONE", "failures:", failures)
