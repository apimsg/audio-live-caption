"""端到端验证桌面字幕管线（不需要真实声音/窗口）：

48k 立体声回环回调 → CaptionEngine 重采样切窗 → 识别 → TranslatorWorker 翻译 → 译文输出。
同时校验 make_translator 的回退与缓存、以及 tkinter 是否可用。
"""

import sys
import time

sys.path.insert(0, ".")

import numpy as np

from caption_engine import Caption, CaptionEngine
from translate_engine import (
    NoneTranslator,
    TranslatorsProvider,
    TranslatorWorker,
    make_translator,
)


# --- 1. 系统声音回环回调 -> 引擎 -> 识别 ---
class _Model:
    def __init__(self):
        self.n = 0

    def transcribe(self, audio, *a, **k):
        self.n += 1
        # 引擎应已把 48k 立体声降混并重的到 16k 单声道窗口
        assert audio.ndim == 1 and audio.shape[0] == 16_000, audio.shape
        return {"text": f"Hello world {self.n}"}


eng = CaptionEngine(_Model(), source="system", chunk_seconds=1.0, vad_probe=False)
eng._pa_mod = type("M", (), {"paComplete": 1, "paContinue": 0})
eng._lb_channels = 2
eng.source_sr = 48_000
import threading

eng.started_at = time.time()
eng._thread = threading.Thread(target=eng._run, daemon=True)
eng._thread.start()

try:
    stereo = (np.ones(48_000 * 2, dtype=np.float32) * 0.2).reshape(-1, 2)
    for _ in range(3):
        eng._on_loopback(stereo.tobytes(), 48_000, None, None)
    caps = []
    deadline = time.time() + 5
    while len(caps) < 2 and time.time() < deadline:
        caps.extend(eng.drain())
        time.sleep(0.1)
    assert len(caps) >= 2, caps
    assert caps[0].text == "Hello world 1", caps[0].text
    print("loopback->engine OK:", [c.text for c in caps])
finally:
    eng._stop.set()
    eng._thread.join(timeout=3)


# --- 2. TranslatorWorker 异步翻译 ---
class _FakeTranslator:
    last_error = ""

    def translate(self, text, context=""):
        assert context in ("", "Hello world 1"), context
        return "你好世界"


worker = TranslatorWorker(_FakeTranslator())
worker.start()
worker.submit(caps[0], True)
cap, translated = worker.out.get(timeout=3)
assert translated == "你好世界", translated
worker.stop()
print("translator worker OK:", translated)


# --- 3. 翻译后端选择与回退 ---
none_t = make_translator("none")
assert isinstance(none_t, NoneTranslator)
assert none_t.translate("abc") == "abc"

openai_missing_key = make_translator("openai", api_key="")
assert isinstance(openai_missing_key, NoneTranslator), "缺 api_key 应退化为不翻译"

baidu_t = make_translator("baidu", appid="x", secret="y")
assert baidu_t.translate("") == ""

# 缓存：同一文本第二次不再调用底层
calls = {"n": 0}


class _CountTranslator(TranslatorsProvider):
    def _do_translate(self, text, context=""):
        calls["n"] += 1
        return "T:" + text


ct = _CountTranslator()
ct.translate("hi"); ct.translate("hi"); ct.translate("hi")
assert calls["n"] == 1, calls
print("translator fallback + cache OK")


# --- 4. tkinter 真实字体测量/绘制（隐藏窗口，不闪现到桌面） ---
import tkinter as tk
import tkinter.font as tkfont

root = tk.Tk()
root.withdraw()  # 从不映射到屏幕

f = tkfont.Font(font=("Microsoft YaHei UI", 20, "bold"))
w1 = f.measure("短")
w2 = f.measure("短" * 40)
assert w1 > 0 and w2 > w1 * 10, (w1, w2)

# Canvas.create_text 在未映射窗口上可用
c = tk.Canvas(root, width=300, height=60, bg="#010203")
item = c.create_text(10, 10, anchor="w", text="测试字幕", font=f)
assert c.coords(item) == [10, 10]

# 复刻 desktop_caption 的折行逻辑，确认长句被拆开且每段不超宽
import desktop_caption as dc  # noqa: E402

max_px = 400
cache = {}


def measure_cached(text):
    font = cache.get(20)
    if font is None:
        font = cache[20] = f
    return font.measure(text)


out, line = [], ""
for ch in "这是一段很长很长的中文句子用来验证按像素宽度自动折行的功能是否正常工作的测试文本。":
    probe = line + ch
    if measure_cached(probe) > max_px:
        out.append(line)
        line = ch
    else:
        line = probe
if line:
    out.append(line)
assert len(out) >= 3, out
for seg in out:
    assert f.measure(seg) <= max_px or len(seg) == 1

root.destroy()
print("tkinter measure/canvas OK:", tk.TkVersion)
# --- 5. 黄色提醒画在字幕正上方 + 悬浮窗记住上次显示位置 ---
import os

import desktop_caption as dc

hints, rows = dc.split_hint_lines([("src", "第一条"), ("hint", "提醒A"),
                                   ("trans", "译文"), ("hint", "提醒B")])
assert [t for _k, t in hints] == ["提醒A", "提醒B"], hints
assert [t for _k, t in rows] == ["第一条", "译文"], rows     # 字幕行顺序不变
default = dc.resolve_origin(None, 1100, 200, 1920, 1080, 140)
assert default == ((1920 - 1100) // 2, 1080 - 200 - 140), default
assert dc.resolve_origin((555, 666), 1100, 200, 1920, 1080, 140) == (555, 666)
# 拔掉一台显示器后记忆坐标可能落在不存在的屏幕上 → 必须回默认而不是看不见
assert dc.resolve_origin((99999, 99999), 1100, 200, 1920, 1080, 140) == default
assert dc.resolve_origin((-32000, -32000), 1100, 200, 1920, 1080, 140) == default
os.makedirs(".tmp", exist_ok=True)
pf = os.path.join(".tmp", "overlay_pos_test.json")
dc.save_pos(pf, 123, 456)
assert dc.load_pos(pf) == (123, 456), dc.load_pos(pf)
assert dc.load_pos(os.path.join(".tmp", "nope.json")) is None      # 没记过也要能开
open(pf, "w", encoding="utf-8").write("{坏 json")
assert dc.load_pos(pf) is None, "位置文件损坏时不能崩"
os.remove(pf)
print("overlay hint-above + position memory OK:", default)

print("DESKTOP PIPELINE OK")