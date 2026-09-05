"""端到端测试：走真实线程路径，只有物理麦克风被替换为注入的音频块。

验证 PortAudio 回调 → 重采样 → 窗口切分 → 静音门限 → 识别 → 结果队列 全链路。
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from caption_engine import CaptionEngine  # noqa: E402


class _ScriptModel:
    """按调用次序返回固定文本，并记录收到的音频长度。"""

    def __init__(self, texts):
        self.texts = list(texts)
        self.lengths: list[int] = []

    def transcribe(self, audio, *a, **k):
        self.lengths.append(int(audio.shape[0]))
        return {"text": self.texts.pop(0) if self.texts else ""}


def make_block(seconds: float, sr: int, amp: float = 0.3) -> np.ndarray:
    """模拟 PortAudio 传入的 (frames, channels) float32 块。"""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    mono = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return mono.reshape(n, 1)


SOURCE_SR = 48_000

model = _ScriptModel(["第一句", "第二句"])
# vad_probe=False：本测试严格校验"每 1s 窗口→一次 transcribe"的时序，
# VAD 探针会额外调用模型打乱计数（探针自身逻辑在 test_vad_language.py 覆盖）。
eng = CaptionEngine(model, device=None, chunk_seconds=1.0, silence_rms=0.005,
                    vad_probe=False)
eng.source_sr = SOURCE_SR

# 启动采集线程，但不打开真实 InputStream：直接喂回调。
import threading  # noqa: E402

eng._stop.clear()
eng.started_at = time.time()
eng._thread = threading.Thread(target=eng._run, daemon=True)
eng._thread.start()

try:
    # 1) 一段静音：不应产生字幕
    eng._on_audio(make_block(1.2, SOURCE_SR, amp=0.0), 0, None, None)
    time.sleep(0.4)
    assert eng.drain() == [], "静音不应产出字幕"

    # 2) 两段有声：跨 48k→16k 重采样后应各切出一个 1s 窗口
    eng._on_audio(make_block(1.2, SOURCE_SR), 0, None, None)
    time.sleep(0.6)
    eng._on_audio(make_block(1.2, SOURCE_SR), 0, None, None)

    caps = []
    deadline = time.time() + 5
    while len(caps) < 2 and time.time() < deadline:
        caps.extend(eng.drain())
        time.sleep(0.1)

    assert [c.text for c in caps] == ["第一句", "第二句"], caps
    assert all(length == 16_000 for length in model.lengths), model.lengths
    assert eng.processed_chunks == 2, eng.processed_chunks
    assert eng.backlog_seconds < 1.5, eng.backlog_seconds
    assert eng.last_rms > 0.01, eng.last_rms
    # 时间轴应连续推进
    assert caps[1].start >= caps[0].end - 1e-6, (caps[0].end, caps[1].start)
    print("captions:", [(c.index, round(c.start, 2), round(c.end, 2), c.text) for c in caps])
    print("window lengths:", model.lengths)
finally:
    eng._stop.set()
    eng._thread.join(timeout=3)

# --- 5. 单窗异常绝不能杀死识别线程（"出了两条就卡住"就是这种静默死法） ---
class _FlakyModel:
    """第 3 窗给怪 speech_end（必须被就地兜住），第 4 窗给非字符串 text
    （会在模型 try 之外炸，靠线程外层保险挡住）。"""

    def __init__(self):
        self.n = 0

    def transcribe(self, audio, *a, **k):
        self.n += 1
        if self.n == 3:
            return {"text": "坏类型探针", "speech_end": "不是数字"}
        if self.n == 4:
            # 置信度是个怪东西：_vote_language 里 float() 会炸，必须只丢这一窗
            return {"text": "第4句中文", "language": "en",
                    "language_probability": "很自信"}
        return {"text": f"第{self.n}句中文"}


flaky = _FlakyModel()
eng5 = CaptionEngine(flaky, device=None, chunk_seconds=1.0, silence_rms=0.005,
                     vad_probe=False)
eng5.source_sr = SOURCE_SR
eng5._stop.clear()
eng5.started_at = time.time()
eng5._thread = threading.Thread(target=eng5._run, daemon=True)
eng5._thread.start()
caps5 = []
try:
    for _ in range(6):
        eng5._on_audio(make_block(1.2, SOURCE_SR), 0, None, None)
        deadline = time.time() + 2
        while time.time() < deadline:
            caps5.extend(eng5.drain())
            if len(caps5) >= 5:
                break
            time.sleep(0.05)
    time.sleep(0.4)
    caps5.extend(eng5.drain())
    assert eng5.thread_alive(), "识别线程被单窗异常杀死了（字幕就会永远停在最后一条）"
    assert eng5.thread_errors == 1, eng5.thread_errors        # 只有第 4 窗该计入异常
    assert "识别线程异常" in eng5.error, eng5.error
    texts5 = [c.text for c in caps5]
    assert len(caps5) >= 5, caps5
    # 坏窗（第 4 窗）没有字幕，但之后的窗必须照常出 —— 这才叫没卡住
    assert not any("第4句" in t for t in texts5), texts5
    assert any("第6句" in t for t in texts5) and any("第7句" in t for t in texts5), texts5
    assert "坏类型探针" in texts5, texts5              # 怪 speech_end 被就地兜住，字幕没丢
    # 栈尾必须留在界面上能看到的地方：线程死了不报错是最难查的故障
    assert eng5.thread_error_log and "识别" in eng5.thread_error_log[0]
    assert "_vote_language" in eng5.thread_error_log[0], eng5.thread_error_log[0]
    assert "ValueError" in eng5.thread_error_log[0], eng5.thread_error_log[0]
    print("thread survives bad window OK:", texts5)
finally:
    eng5._stop.set()
    eng5._thread.join(timeout=3)

# --- 6. 分发轮必须真的发布并返回 True（曾经的低级错：整段逻辑被留在
#        `return False` 那个 if 里，引擎在跑时函数落到末尾返回 None，
#        分发线程"秒退且零异常"：识别出 4 段而字幕 0 条、悬浮窗全白） ---
import importlib
from caption_engine import Caption
from caption_service import CaptionStore
_rt = importlib.import_module("app").Runtime

class _OkEngine:
    """一个健康引擎的最小替身。"""
    running = True
    processed_chunks = 4
    hint = error = notice = ""
    last_rms = 0.1
    language = detected_language = "zh"
    thread_errors = 0

    def __init__(self, caps):
        self._caps = caps

    def drain(self):
        caps, self._caps = self._caps, []
        return caps

    def thread_alive(self):    # ensure_dispatcher 用的就是这个口径
        return self.running


stub = _rt.__new__(_rt)            # 不跑 __init__ 的真 Runtime：连 _publish 都是真的
stub.lock = threading.RLock()
stub.store = CaptionStore()
stub.translating = False
stub.worker = None
stub.disp_errors = 0
stub.disp_restarts = 0
stub.worker = None
stub.proc = None
stub.model_dev = "cuda/float16"
stub.engine = _OkEngine([
    Caption(index=1, start=0.0, end=1.0, text="我们明天开会", wall_clock=time.time()),
    Caption(index=2, start=1.0, end=2.0, text="记得带合同", wall_clock=time.time()),
])
stub.captions, stub.trans, stub._pos, stub.text_of = [], {}, {}, {}
assert _rt._dispatch_once(stub) is True, "分发轮没返回 True：线程会被立刻收工"
assert [c.text for c in stub.captions] == ["我们明天开会", "记得带合同"], stub.captions
assert stub.store._rev > 0, "没写进字幕服务，悬浮窗永远空白"
assert stub.store.status["chunks"] == 4
stub.engine.running = False            # 只有引擎真停了这轮才该收工
assert _rt._dispatch_once(stub) is False
print("dispatch publishes OK: rev =", stub.store._rev)

# 分发线程掉了必须自动续上（不自愈就得靠人重启，用户看到的就是"永久卡住"）
stub.engine = _OkEngine([])          # 引擎活着，这一轮没有新字幕
stub._disp_stop = threading.Event()
stub._disp = None
stub.ensure_dispatcher()
assert stub._disp is not None and stub._disp.is_alive(), "分发线程没被续上"
assert stub.disp_restarts == 1, stub.disp_restarts
time.sleep(0.5)
assert stub._disp.is_alive(), "续上的分发线程没能持续工作"
stub._disp_stop.set()
stub._disp.join(timeout=3)
assert not stub._disp.is_alive()
print("dispatcher self-heal OK")

# 同一份快照就是界面上「卡住时自查」显示的东西：键必须齐、值必须真
_dg = stub.diagnostics()
for _k in ("引擎在跑", "分发线程在跑", "翻译线程在跑", "栈尾", "语种票", "待识别秒数",
           "电平", "距上一条字幕", "服务rev", "服务epoch", "模型", "录音留档", "分发自愈次数"):
    assert _k in _dg, _k
assert _dg["分发自愈次数"] == 1 and _dg["服务rev"] > 0, _dg
assert _dg["引擎在跑"] is True and _dg["模型"] == "cuda/float16", _dg
print("diagnostics snapshot OK:", _dg["模型"], "|", len(_dg), "keys")

# --- 7. 解码挂死必须被看门狗发现（CT2/CUDA 挂住时不抛异常，try 拦不住） ---
hold = threading.Event()


class _HangModel:
    def __init__(self):
        self.n = 0

    def transcribe(self, audio, *a, **k):
        self.n += 1
        if self.n == 1:
            assert hold.wait(5), "测试没放行"
        return {"text": f"第{self.n}句中文"}


eng7 = CaptionEngine(_HangModel(), device=None, chunk_seconds=1.0, silence_rms=0.005,
                     vad_probe=False)
eng7.source_sr = SOURCE_SR
eng7._stop.clear()
eng7.started_at = time.time()
eng7._thread = threading.Thread(target=eng7._run, daemon=True)
eng7._thread.start()
try:
    eng7._on_audio(make_block(1.2, SOURCE_SR), 0, None, None)
    deadline = time.time() + 5
    while time.time() < deadline and not eng7.decoding_at:
        time.sleep(0.05)
    assert eng7.decoding_at, "没记下「正在解哪一窗」这个标记"
    time.sleep(0.3)
    assert eng7.decoding_age() > 0.2, eng7.decoding_age()
    assert eng7.stall_seconds(0.05) > 0, "挂死程度没被量化"
    assert eng7.thread_alive(), "看门狗期间线程不该死"
    hold.set()
    deadline = time.time() + 5
    while time.time() < deadline and eng7.decoding_at:
        time.sleep(0.05)
    assert eng7.decoding_at == 0.0, "解完没清掉在飞标记"
    assert eng7.stall_seconds() == 0.0
    print("decode hang detected OK:", eng7.decoding_span, eng7.processed_chunks)
finally:
    hold.set()
    eng7._stop.set()
    eng7._thread.join(timeout=3)

# --- 8. 「有音量却零字」必须触发非语音过滤自救（真实故障：243s 音频只出 16 字） ---
from caption_engine import TARGET_SR


class _VadTrapModel:
    """前 3 窗返回空文本（就是 Silero/no_speech 把语音吃光的样子）；
    一旦引擎关掉过滤，后面立刻有字。"""
    supports_vad_toggle = True

    def __init__(self):
        self.calls = 0
        self.empty_until = 3
        self.resets = 0

    def set_non_speech_filter(self, on):
        self.resets += 1
        if not on:
            self.empty_until = 0

    def transcribe(self, audio, *a, **k):
        self.calls += 1
        if self.calls <= self.empty_until:
            return {"text": "", "language": "zh", "language_probability": 0.99}
        return {"text": f"第{self.calls}句中文", "language": "zh",
                "language_probability": 0.99}


m8 = _VadTrapModel()
eng8 = CaptionEngine(m8, device=None, chunk_seconds=1.0, silence_rms=0.005,
                     vad_probe=False)
loud = np.full(int(TARGET_SR), 0.2, dtype=np.float32)   # 有音量的窗
for _ in range(5):
    eng8._transcribe(loud, 0.0, 1.0)
assert eng8.filter_disabled is True, "连续零字没触发非语音过滤自救"
assert m8.resets == 1 and eng8.processed_chunks >= 2, (m8.resets, eng8.processed_chunks)
assert "非语音过滤" in eng8.notice, eng8.notice
# 用户明确不要自救时，绝不能自作主张改模型设置
m8b = _VadTrapModel()
eng8b = CaptionEngine(m8b, device=None, chunk_seconds=1.0, silence_rms=0.005,
                      vad_probe=False, auto_unfilter=False)
for _ in range(6):
    eng8b._transcribe(loud, 0.0, 1.0)
assert eng8b.filter_disabled is False and m8b.resets == 0, (eng8b.filter_disabled, m8b.resets)
# 真静音（电平不过门限）不该触发自救 —— 否则安静环境里会被误关
m8c = _VadTrapModel()
eng8c = CaptionEngine(m8c, device=None, chunk_seconds=1.0, silence_rms=0.05,
                      vad_probe=False)
quiet = np.zeros(int(TARGET_SR), dtype=np.float32)
for _ in range(6):
    eng8c._transcribe(quiet, 0.0, 1.0)
assert eng8c.filter_disabled is False and eng8c.empty_windows == 0, eng8c.empty_windows
print("non-speech filter self-rescue OK:", eng8.processed_chunks, eng8.notice[:18])

# --- 9. 中文强制简体（后端给繁体也要转；日文同形字绝不能动） ---
import han_convert

assert len(han_convert._TABLE) >= 150, len(han_convert._TABLE)
assert han_convert.to_simplified('應該') == '应该'   # 曾漏字导致整表错位，钉住
assert han_convert.to_simplified('我們在這裡說話') == '我们在这里说话'


class _TradModel:
    def __init__(self):
        self.kwargs = []

    def transcribe(self, audio, *a, **k):
        self.kwargs.append(k)
        return {"text": "我們在這裡說話，應該都會說中文", "language": "zh",
                "language_probability": 0.99}


m9 = _TradModel()
eng9 = CaptionEngine(m9, device=None, chunk_seconds=1.0, silence_rms=0.005, vad_probe=False)
eng9._transcribe(loud, 0.0, 1.0)     # loud 来自 §8
assert m9.kwargs and m9.kwargs[0].get("simplify") is True, m9.kwargs
got9 = eng9.drain()
assert got9 and got9[0].text == "我们在这里说话，应该都会说中文", got9
# 显式关掉就不该转
eng9b = CaptionEngine(_TradModel(), device=None, chunk_seconds=1.0, silence_rms=0.005,
                      vad_probe=False, simplify=False)
eng9b._transcribe(loud, 0.0, 1.0)
got9b = eng9b.drain()
assert got9b and got9b[0].text.startswith("我們在這裡"), got9b
# 日文：同形字转简体会改坏用字，必须原样


class _JaModel:
    def transcribe(self, audio, *a, **k):
        return {"text": "會議はもう始まった。手紙を書く。", "language": "ja",
                "language_probability": 0.99}


eng9c = CaptionEngine(_JaModel(), device=None, chunk_seconds=1.0, silence_rms=0.005,
                      vad_probe=False)
eng9c._transcribe(loud, 0.0, 1.0)
got9c = eng9c.drain()
assert got9c and got9c[0].text == "會議はもう始まった。手紙を書く。", got9c
assert han_convert.to_simplified("手紙を書く") != "手紙を書く"
# 保险：混了假名（其实是日文）的译文不许转，纯中文的才转
assert han_convert.to_simplified_if_chinese("会議は三時です") == "会議は三時です"
assert han_convert.to_simplified_if_chinese("會議是三時") == "会议是三时"
assert han_convert.to_simplified_if_chinese("") == ""
print("simplified-Chinese enforcement OK:", got9[0].text, "|", han_convert.backend_name())

print("LIVE PIPELINE OK")
