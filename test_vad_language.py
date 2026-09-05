"""②③ 专项测试（无麦克风/无网络/无窗口）：

1. whisper_backend 适配器归一化（faster / openai 两路，含 no_speech 过滤）
2. VAD 驱动的早触发：BGM 持续高电平下，"探测说停"才 flush、"探测说在说"不 flush
   （RMS 全程参与不了，证明判定来源真的换成了语音段）
3. 语种投票（对应用户报的"中文错乱 / 日语不准"）：
   静音幻觉文本不投票；假名硬证据 > 模型自评，"汉字无假名"在自评<0.90 时纠回 zh；
   冷启动 2 票一致才钉；钉定后正式窗用钉定语种、探测窗仍自由检测；
   漂移需连续票（普通语种对 2 票，中↔日 3 票），票不连续立刻清零。
"""
import sys
import threading
import time

sys.path.insert(0, ".")

import numpy as np

from caption_engine import CaptionEngine
from whisper_backend import FasterAdapter, OpenAIAdapter

SR = 16_000


# --- 1. 适配器归一化 ---
class _Seg:
    def __init__(self, start, end, text, nsp):
        self.start, self.end, self.text, self.no_speech_prob = start, end, text, nsp


class _Info:
    language, language_probability = "ja", 0.99


class _FakeFaster:
    def __init__(self, segs, info):
        self._segs, self._info = segs, info
        self.kwargs = None

    def transcribe(self, audio, **kw):
        self.kwargs = kw
        return iter(self._segs), self._info


raw = _FakeFaster([
    _Seg(0.0, 0.6, "你好", 0.10),
    _Seg(0.8, 1.1, "幻觉", 0.90),      # 高 no_speech_prob → 必须被过滤
    _Seg(1.1, 1.3, "世界", 0.20),
], _Info())
fa = FasterAdapter(raw)
r = fa.transcribe(np.zeros(SR, dtype=np.float32), language=None)
assert r["language"] == "ja" and r["language_probability"] == 0.99
assert r["speech_end"] == 1.3 and r["has_segments"]
assert r["text"] == "你好 世界" and "幻觉" not in r["text"], r
assert raw.kwargs.get("vad_filter") is True        # VAD 真的开了


class _FakeOld:
    def transcribe(self, audio, **kw):
        return {"text": " hi uh", "language": "en", "segments": [
            {"start": 0.0, "end": 0.8, "text": " hi", "no_speech_prob": 0.1},
            {"start": 1.0, "end": 1.2, "text": " uh", "no_speech_prob": 0.9},
        ]}


r2 = OpenAIAdapter(_FakeOld()).transcribe(np.zeros(2 * SR, dtype=np.float32))
assert r2["text"] == "hi" and r2["speech_end"] == 0.8, r2
assert r2["language_probability"] == 1.0
print("adapters normalized OK")


# --- 2. VAD 早触发（真实 _run 线程 + 全程高电平"BGM"） ---
class _VadModel:
    """probe 与正式窗都走它；用返回的 speech_end 模拟 VAD 判定。"""

    def __init__(self):
        self.mode = "active"          # active=语音持续, paused=语音已停
        self.formal = []              # 正式窗口 (长度, 文本判定)
        self.probes = 0

    def transcribe(self, audio, *a, **k):
        dur = audio.shape[0] / SR
        lang = k.get("language")
        if lang is None:              # 探测窗（未钉定前始终 None）
            self.probes += 1
            end = dur - 0.1 if self.mode == "active" else 0.2
            # 探测窗必须给出**有效文本**：新规则下空文本/静音幻觉不参与语种投票，
            # 语种钉不下来，下面的正式窗就会一直以 None 送检（老写法是踩着 bug 过的）。
            return {"text": "hello there", "language": "en", "language_probability": 0.95,
                    "speech_end": end, "has_segments": True}
        self.formal.append(int(audio.shape[0]))
        return {"text": "句子内容", "language": lang, "language_probability": 0.95,
                "speech_end": dur - 0.1 if self.mode == "active" else 0.2,
                "has_segments": True}


vm = _VadModel()
eng = CaptionEngine(vm, chunk_seconds=6.0, vad_probe=True,
                    probe_interval=0.2, probe_len=1.0, flush_after_silence=0.6)
eng.started_at = time.time()
eng._thread = threading.Thread(target=eng._run, daemon=True)
eng._thread.start()
try:
    loud = (np.ones(SR // 2, dtype=np.float32) * 0.2)   # 0.5s 高电平块（BGM）
    # A) 语音持续（VAD active）+ 缓冲 < 窗口：绝不早触发
    for _ in range(7):                                   # 3.5s
        eng._audio_q.put((loud.copy(), SR))
        time.sleep(0.05)
    time.sleep(0.8)                                      # 留出探针时间
    assert eng.drain() == [] and vm.formal == [], (vm.formal, "active 时不该 flush")
    assert vm.probes > 0, "探针没跑"
    # B) VAD 判定"语音已停"（电平依旧很高）→ 必须早触发，窗口远不满 6s
    vm.mode = "paused"
    deadline = time.time() + 3
    while not vm.formal and time.time() < deadline:
        time.sleep(0.1)
    assert vm.formal, "VAD 判停后没有早触发"
    assert vm.formal[0] < 6 * SR, vm.formal              # 是早触发不是等满窗
    print("VAD early-flush under loud BGM OK:", vm.formal[0], "samples")
finally:
    eng._stop.set()
    eng._thread.join(timeout=3)

# --- 3. 语种投票：冷启动 2 票 / 脚本优先 / 幻觉不投票 / 中↔日 3 票 ---
class _Spy:
    """按脚本喂探测结果，并记录每次送进模型的 language。"""

    def __init__(self):
        self.calls = []
        self.reply = ("hello there", "en", 0.95)

    def transcribe(self, audio, *a, **k):
        self.calls.append(k.get("language"))
        text, lang, prob = self.reply
        return {"text": text, "language": lang, "language_probability": prob}


def _feed(replies, engine=None):
    """依次把 (text, declared, prob) 当探测窗结果喂进去，返回 (engine, spy)。"""
    spy = _Spy()
    e = engine or CaptionEngine(spy, vad_probe=False)
    loud = np.ones(SR, dtype=np.float32) * 0.2
    for i, (text, lang, prob) in enumerate(replies):
        spy.reply = (text, lang, prob)
        e._transcribe(loud, float(i), float(i + 1), probe=True)
    return e, spy


# 3.1 静音幻觉：中文音频被钉成日语的经典路径
e, _ = _feed([("日本語字幕", "ja", 0.99)] * 4)
assert e.language is None, "静音幻觉文本不得参与语种投票"
e, _ = _feed([("", "ja", 0.99)] * 4)
assert e.language is None, "空文本（没解出东西）不得投票"

# 3.2 冷启动要 2 票一致（不再首见即钉）
e, spy = _feed([("hello there", "en", 0.95)])
assert e.language is None, "单票不得钉语种"
e, spy = _feed([("hello there", "en", 0.95)] * 2)
assert e.language == "en" and e.language_switches == 0

# 3.3 真日语低置信也要能钉上（旧门槛 0.7 会把正确的票整张丢掉）
e, _ = _feed([("会議は三時です", "ja", 0.52)] * 2)
assert e.language == "ja", e.language

# 3.4 假名是硬证据：模型自报 zh 0.98 也判日语
e, _ = _feed([("会議は三時です", "zh", 0.98)] * 2)
assert e.language == "ja", e.language

# 3.5 汉字无假名：模型没到 0.90 时脚本纠回中文，到了就留给模型
e, _ = _feed([("我们明天见", "ja", 0.85)] * 2)
assert e.language == "zh", e.language
e, _ = _feed([("明日会議全体", "ja", 0.95)] * 2)
assert e.language == "ja", e.language

# 3.6 钉定后：正式窗用钉定语种，探测窗仍然自由检测（漂移雷达）
e, spy = _feed([("hello there", "en", 0.95)] * 2)
e._transcribe(np.ones(SR, dtype=np.float32) * 0.2, 9.0, 10.0)      # 正式窗
assert spy.calls[-1] == "en", spy.calls
e._transcribe(np.ones(SR, dtype=np.float32) * 0.2, 10.0, 11.0, probe=True)
assert spy.calls[-1] is None, spy.calls

# 3.7 漂移：en→ja 两票即切（非 CJK 对）并给 notice
spy.reply = ("会議は三時です", "ja", 0.95)
loud = np.ones(SR, dtype=np.float32) * 0.2
e._transcribe(loud, 11.0, 12.0, probe=True)
assert e.language == "en", "第 1 票不该切"
e._transcribe(loud, 12.0, 13.0, probe=True)
assert e.language == "ja" and e.language_switches == 1 and "ja" in e.notice

# 3.8 中文↔日语这一对要 3 票，且必须连续（中间被别的语种打断就重数）
spy.reply = ("我们明天见", "zh", 0.9)
e._transcribe(loud, 13.0, 14.0, probe=True)          # 第 1 票
e._transcribe(loud, 14.0, 15.0, probe=True)          # 第 2 票
assert e.language == "ja", "中↔日 2 票不得切换"
spy.reply = ("hello there", "en", 0.99)
e._transcribe(loud, 15.0, 16.0, probe=True)          # 插一刀：票要重数
spy.reply = ("我们明天见", "zh", 0.9)
e._transcribe(loud, 16.0, 17.0, probe=True)
e._transcribe(loud, 17.0, 18.0, probe=True)
assert e.language == "ja", "非连续的票不得切换"
e._transcribe(loud, 18.0, 19.0, probe=True)
assert e.language == "zh" and e.language_switches == 2, (e.language, e.language_switches)

# 3.9 用户显式选定语种：全程钉死，投票完全忽略
spy2 = _Spy()
spy2.reply = ("会議は三時です", "ja", 0.99)
e2 = CaptionEngine(spy2, language="en", vad_probe=False)
e2._transcribe(loud, 0.0, 1.0)
e2._transcribe(loud, 1.0, 2.0, probe=True)
assert e2.language == "en" and spy2.calls == ["en", "en"], spy2.calls
print("language rules OK")
print("VAD LANGUAGE OK")
