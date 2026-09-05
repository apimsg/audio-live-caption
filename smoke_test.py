"""Headless smoke test for the captioning app (no browser, no mic needed)."""

import os
import sys

from streamlit.testing.v1 import AppTest

sys.path.insert(0, ".")

at = AppTest.from_file("app.py", default_timeout=60)
at.run()

print("exception count:", len(at.exception))
for exc in at.exception:
    print("EXC:", exc.value)

print("title:", [t.value for t in at.title])
print("sidebar headers:", [h.value for h in at.sidebar.header])
print("buttons:", [b.label for b in at.sidebar.button])
print("metrics:", [(m.label, m.value) for m in at.sidebar.metric])
print("select_sliders:", [s.label for s in at.sidebar.select_slider])
print("selectboxes:", [s.label for s in at.sidebar.selectbox])
print("sliders:", [s.label for s in at.sidebar.slider])
print("errors:", [e.value for e in at.error])
print("infos:", [i.value for i in at.info])

assert not at.exception, "app raised an exception"

# 「本地视频 → 16k WAV」卡片：填一个真媒体文件，整页重渲染 + 真点一次提取
import time
import wave

src_sample = os.path.join("models", "jfk.flac")
if os.path.isfile(src_sample):
    out_dir = os.path.join(".tmp", "smoke-extract")
    os.makedirs(out_dir, exist_ok=True)
    for f0 in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f0))
    at.session_state["x_src"] = src_sample
    at.session_state["x_out"] = out_dir
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    caps = [c.value for c in at.caption]
    assert any("轨道" in c and "分钟" in c for c in caps), "没显示探测到的时长/轨道"

    # 语种护栏：语言选「日本語」却喂纯汉字热词 → 必须提醒（会把模型往中文带）
    sb = [i for i, s in enumerate(at.sidebar.selectbox) if s.label == "语言"]
    ti = [i for i, t in enumerate(at.sidebar.text_input) if "热词" in t.label]
    assert sb and ti, ([s.label for s in at.sidebar.selectbox],
                       [t.label for t in at.sidebar.text_input])
    at.sidebar.selectbox[sb[0]].set_value("日本語")
    at.sidebar.text_input[ti[0]].set_value("会议 合同 预算")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("片假名" in c.value for c in at.caption), "日语配汉字热词没提示"
    at.sidebar.text_input[ti[0]].set_value("サーバー, 負荷試験")
    at.run()
    assert not any("片假名" in c.value for c in at.caption), "写了片假名还提示"
    at.sidebar.selectbox[sb[0]].set_value("自动检测")
    at.sidebar.text_input[ti[0]].set_value("")
    at.run()
    # 「浏览」按钮必须走 on_click 回调：早前写成 handler 分支体，点一次就抛
    # WidgetAlreadyInstantiatedError（同一轮里 text_input 已经实例化了）。
    os.environ["ALC_PICK"] = src_sample        # 无桌面会话下不弹真对话框
    bi = [i for i, b in enumerate(at.button) if "浏览" in b.label]
    assert bi, [b.label for b in at.button]
    at.button[bi[0]].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["x_src"] == src_sample, at.session_state["x_src"]
    del os.environ["ALC_PICK"]
    labels = [b.label for b in at.button]
    gi = [i for i, b in enumerate(labels) if "提取 16k WAV" in b]
    assert gi and any("本机识别出双语字幕" in x for x in labels), labels
    at.button[gi[0]].click()
    at.run()
    for _ in range(60):                       # 提取在后台线程里跑，等它收尾
        time.sleep(0.5)
        if not at.session_state["extract"]["busy"]:
            break
    at.session_state["x_src"] = src_sample    # 线程结束时页面已重渲染，再点一次同名按钮
    at.run()
    made = sorted(x for x in os.listdir(out_dir) if x.endswith(".wav"))
    assert made, "没产出 WAV：  "
    with wave.open(os.path.join(out_dir, made[0])) as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)
        assert wf.getnframes() > 16000
    assert not any("没音轨" in (e.value or "") for e in at.error), [e.value for e in at.error]
    print("extract card OK:", made)

# 引擎单元测试（不依赖麦克风/模型）
import numpy as np
from caption_engine import Caption, CaptionEngine, to_srt, resample_to_16k

a = np.sin(np.linspace(0, 100, 48_000)).astype(np.float32)
r = resample_to_16k(a, 48_000)
assert abs(r.shape[0] - 16_000) <= 1, r.shape
assert resample_to_16k(a, 16_000).shape == a.shape

caps = [
    Caption(1, 0.0, 4.0, "你好，世界", 1_700_000_000),
    Caption(2, 4.0, 8.5, "hello there", 1_700_000_004),
]
srt = to_srt(caps)
assert "00:00:00,000 --> 00:00:04,000" in srt, srt
assert "00:00:04,000 --> 00:00:08,500" in srt, srt
assert srt.count("\n\n") >= 1
assert to_srt([]) == ""

# 静音 / 太短的块不应调用模型（_transcribe 会吞异常，用 error 是否为空来判断）
class _FakeModel:
    def transcribe(self, *a, **k):
        raise AssertionError("should not transcribe silence")

eng = CaptionEngine(_FakeModel(), chunk_seconds=4.0)
eng._transcribe(np.zeros(16_000, dtype=np.float32), 0.0, 1.0)      # 静音
eng._transcribe(np.ones(4_000, dtype=np.float32) * 0.2, 0.0, 0.25)  # 太短
assert eng.processed_chunks == 0
assert eng.error == "", f"model was called: {eng.error}"
assert eng.drain() == []

# filler 文本被过滤
class _FillerModel:
    def transcribe(self, *a, **k):
        return {"text": "谢谢观看"}

eng2 = CaptionEngine(_FillerModel(), chunk_seconds=1.0)
eng2._transcribe(np.ones(32_000, dtype=np.float32) * 0.2, 0.0, 2.0)
assert eng2.processed_chunks == 0, "filler should be filtered"
assert eng2.error == ""

# 正常文本应产出字幕
class _OkModel:
    def transcribe(self, *a, **k):
        return {"text": " 今天天气不错 "}

eng3 = CaptionEngine(_OkModel(), chunk_seconds=1.0)
eng3._transcribe(np.ones(32_000, dtype=np.float32) * 0.2, 0.0, 2.0)
out = eng3.drain()
assert len(out) == 1 and out[0].text == "今天天气不错", out
assert eng3.processed_chunks == 1

# 缓冲切分：连续喂 3 段 16k 样本，chunk=2s 时应切出 1 个完整窗口
class _CountModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, *a, **k):
        self.calls += 1
        assert audio.shape[0] == 32_000, audio.shape[0]
        return {"text": f"seg{self.calls}"}

m = _CountModel()
eng4 = CaptionEngine(m, chunk_seconds=2.0)
eng4._buffer = [np.ones(16_000, dtype=np.float32) * 0.2]
eng4._buffered_samples = 16_000
eng4._drain_buffer(force=False)
assert m.calls == 0, "不完整窗口不应识别"
eng4._buffer.append(np.ones(24_000, dtype=np.float32) * 0.2)
eng4._buffered_samples += 24_000
eng4._drain_buffer(force=False)
assert m.calls == 1, m.calls
assert eng4._buffered_samples == 8_000, eng4._buffered_samples

# 背压：积压超过上限时丢弃最旧音频
eng5 = CaptionEngine(_FakeModel(), chunk_seconds=2.0, max_buffered_seconds=2.0)
eng5._buffer = [np.ones(16_000, dtype=np.float32) * 0.2]
eng5._buffered_samples = 16_000
with eng5._lock:
    eng5._drop_oldest_locked()
assert eng5._buffered_samples == 16_000, eng5._buffered_samples   # 未超限，不丢
assert eng5.dropped_samples == 0

eng5._buffer.append(np.ones(48_000, dtype=np.float32) * 0.2)
eng5._buffered_samples += 48_000
with eng5._lock:
    eng5._drop_oldest_locked()
assert eng5._buffered_samples == 32_000, eng5._buffered_samples
assert eng5.dropped_samples == 32_000, eng5.dropped_samples

# 系统声音回环：立体声 48k 回调块应被降为单声道并识别
class _LbModel:
    def __init__(self):
        self.shapes = []

    def transcribe(self, audio, *a, **k):
        self.shapes.append(audio.shape)
        return {"text": "视频里的话"}

eng6 = CaptionEngine(_LbModel(), source="system", chunk_seconds=1.0)
eng6._pa_mod = type("M", (), {"paComplete": 1, "paContinue": 0})
eng6._lb_channels = 2
eng6.source_sr = 48_000
stereo = (np.ones(48_000 * 2, dtype=np.float32) * 0.2).reshape(-1, 2)
eng6._on_loopback(stereo.tobytes(), 48_000, None, None)
assert eng6._audio_q.qsize() == 1
block, sr = eng6._audio_q.get_nowait()
assert block.ndim == 2 and block.shape[1] == 2, block.shape
assert sr == 48_000, sr
# --- 解码档位：默认必须是官方档 balanced(beam_size=5) ---------------------
# 贪心（beam_size=1）在"近音词"上没有回头路：the females 会被听成 the funeral。
# 本机实测 beam=5 与贪心延迟相同，所以把默认从贪心改回 beam=5，并留档位可调。
import whisper_backend as wb

assert wb.decode_options("balanced") == {"beam_size": 5}, wb.decode_options("balanced")
assert wb.decode_options("fast") == {"beam_size": 1}
assert wb.decode_options("accurate")["beam_size"] == 10
assert wb.decode_options("accurate")["patience"] == 2.0
assert wb.decode_options("不存在") == wb.decode_options(None) == wb.decode_options("balanced")
# 白名单：引擎塞来的 fp16/verbose 之类绝不能透传给推理后端
assert wb.decode_options("fast", {"beam_size": 7, "verbose": None, "fp16": False}) == {"beam_size": 7}
assert wb.collapse_ws(" They are  fighting\nfor access ") == "They are fighting for access"


class _KwSpy:
    """假 WhisperModel：记录 faster-whisper 真正收到的解码参数。"""

    def __init__(self):
        self.kwargs = {}

    def transcribe(self, audio, **kw):
        self.kwargs = kw

        class _S:
            text = " hello "
            start = 0.0
            end = 0.5
            no_speech_prob = 0.0

        info = type("I", (), {"language": "en", "language_probability": 0.99})()
        return iter([_S()]), info


spy = _KwSpy()
ad = wb.FasterAdapter(spy)
assert ad.quality == "balanced", ad.quality
# 引擎的实际调用形状（fp16/verbose 是 openai-whisper 专属参数）
ad.transcribe(np.zeros(16_000, dtype=np.float32), language="en", fp16=False,
              verbose=None, condition_on_previous_text=False, initial_prompt=None)
assert spy.kwargs["beam_size"] == 5, spy.kwargs
assert "fp16" not in spy.kwargs and "verbose" not in spy.kwargs, spy.kwargs
assert spy.kwargs["vad_filter"] is True
ad.set_quality("fast")
ad.transcribe(np.zeros(8, dtype=np.float32))
assert spy.kwargs["beam_size"] == 1 and "patience" not in spy.kwargs, spy.kwargs
ad.set_quality("accurate")
ad.transcribe(np.zeros(8, dtype=np.float32))
assert spy.kwargs["beam_size"] == 10 and spy.kwargs["patience"] == 2.0, spy.kwargs
try:
    ad.set_quality("nope")
    raise AssertionError("未知档位必须报错")
except ValueError:
    pass
assert ad.transcribe(np.zeros(8, dtype=np.float32))["text"] == "hello"

# 一条字幕 = 一行原文：换行/连续空白必须压掉；prompt 超长要粗截
class _NlModel:
    def transcribe(self, *a, **k):
        return {"text": "They are fighting\n for access to the   females."}


eng_nl = CaptionEngine(_NlModel(), chunk_seconds=1.0,
                       initial_prompt="females, " + "x" * 400)
assert eng_nl.initial_prompt and len(eng_nl.initial_prompt) <= 200, len(eng_nl.initial_prompt)
eng_nl._transcribe(np.ones(32_000, dtype=np.float32) * 0.2, 0.0, 2.0)
out_nl = eng_nl.drain()
assert len(out_nl) == 1, out_nl
assert out_nl[0].text == "They are fighting for access to the females.", out_nl[0].text
# 单条字幕恰好 3 行（序号/时间轴/一行原文）：换行没被压掉就会多出一行
assert len(to_srt(out_nl).splitlines()) == 3, repr(to_srt(out_nl))

# --- 推理设备：GPU 不可用必须安静回退 CPU，并把原因写进 load_note -------
import whisper_backend as wb

assert wb.cuda_present() in (True, False)
assert wb.SUPPORTED_DEVICES == ("auto", "cpu", "cuda")
# 登记 CUDA 目录必须是幂等的（os.add_dll_directory 每次会建一个句柄）；
# 具体返回几个目录取决于本机装了什么，所以只断言"第二次为空"。
assert isinstance(wb.cuda_runtime_dirs(), list)
assert all(os.path.isdir(d) for d in wb.cuda_runtime_dirs())
wb.add_cuda_dll_dirs()
assert wb.add_cuda_dll_dirs() == [], "第二次不该再登记一遍"
# has_cuda_runtime() 必须是"文件级真相"。两个坑都踩过：pip 把 cuBLAS 和 cuDNN
# 装在不同目录（nvidia/cublas/bin、nvidia/cudnn/bin），而判定要求同一目录两样都有；
# 且文件名写成了 cudnn_64_9.dll（真实名是 cudnn64_9.dll）—— 于是本机永远 False，
# 界面误报"GPU 不可用"，而实际加载模型时 CUDA 好使。这里用 glob 独立算一遍对照。
import glob as _glob

_dlls = [os.path.basename(p)
         for d in wb.cuda_runtime_dirs()
         for p in _glob.glob(os.path.join(d, "*.dll"))]
_expect_rt = ("cublas64_12.dll" in _dlls
              and any(x in _dlls for x in ("cudnn64_9.dll", "cudnn_ops64_9.dll")))
assert wb.has_cuda_runtime() == _expect_rt, (wb.has_cuda_runtime(), _expect_rt)
if _expect_rt:
    assert wb.cuda_present() and wb._LOADED, "有运行库却没预加载进任何一个"
print("cuda runtime detection OK:", wb.has_cuda_runtime(), len(_dlls), "dll")
_real_load = wb._load_faster
_CUDA_ERR = "Library cublas64_12.dll is not found or cannot be loaded"


def _ctor_fail(size, dest, device, ct):
    """构造就缺库（老版本 CT2 的姿势）。"""
    if device == "cuda":
        raise RuntimeError(_CUDA_ERR)
    return _real_load(size, dest, "cpu", ct)


class _StubRaw:
    """构造成功、第一次推理才炸 —— CT2 4.8 的真实姿势，必须探测才发现。"""

    def transcribe(self, *a, **k):
        raise RuntimeError("Library cudnn_ops64_9.dll is not found or cannot be loaded")


def _probe_fail(size, dest, device, ct):
    if device == "cuda":
        return _StubRaw()
    return _real_load(size, dest, "cpu", ct)


wb._load_faster = _ctor_fail
wb._CUDA_FAIL = ""
try:
    mm = wb.load_model("faster", "base", "models", device="auto")
    assert mm.device == "cpu" and mm.compute_type == "int8", (mm.device, mm.compute_type)
    assert "GPU 不可用" in mm.load_note and "cublas64_12" in mm.load_note, mm.load_note
    assert wb._CUDA_FAIL, "失败原因要被记住，下一轮不再慢重试"
    mm2 = wb.load_model("faster", "base", "models", device="cuda")
    assert mm2.device == "cpu" and "上次 GPU 加载失败" in mm2.load_note, mm2.load_note
    mm3 = wb.load_model("faster", "base", "models", device="cpu")
    assert mm3.device == "cpu" and mm3.load_note == "", mm3.load_note
    try:
        wb.load_model("faster", "base", "models", device="gpu")
        raise AssertionError("未知设备必须报错")
    except RuntimeError as exc:
        assert "未知推理设备" in str(exc), exc

    # 探测路径：模型能构造，但第一次 encode 才报缺 cuDNN —— 也必须回退 CPU
    wb._CUDA_FAIL = ""
    wb._load_faster = _probe_fail
    mp = wb.load_model("faster", "base", "models", device="auto")
    assert mp.device == "cpu", mp.device
    assert "cudnn_ops64_9" in mp.load_note, mp.load_note

    # 后端压根没装（ImportError）不许被悄悄当成"GPU 不可用"回退掉
    wb._CUDA_FAIL = ""
    def _no_backend(size, dest, device, ct):
        raise ImportError("no module named faster_whisper")
    wb._load_faster = _no_backend
    try:
        wb.load_model("faster", "base", "models", device="cpu")
        raise AssertionError("没装后端必须报错")
    except RuntimeError as exc:
        assert "未安装 faster-whisper" in str(exc), exc
finally:
    wb._load_faster = _real_load
    wb._CUDA_FAIL = ""
print("DEVICE FALLBACK OK")

print("DECODE QUALITY OK")

# 状态快照 diagnostics() 的覆盖放在 test_live_pipeline §6（那里有真 Runtime 实例 + 假引擎）：
# 在这里再 import 一份 app 会重复起一个 Runtime/服务端口，会把 smoke 进程整个带走。

# 沙箱环境里 AppTest 建的临时目录删不干净（退出阶段的 PermissionError 会把退出码
# 变成非 0，看着像测试失败）。真实用户机器上不会这样，这里只把收尾清理提前跑掉。
import shutil
import tempfile


def _sweep_temp() -> None:
    for cls in (tempfile.TemporaryDirectory, tempfile._TemporaryFileCloser,):
        names = getattr(cls, "_all_names", None)
        if names is None:
            continue
        for name in list(names):
            path = name[0] if isinstance(name, tuple) else name
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
        try:
            names.clear()
        except AttributeError:
            pass


_sweep_temp()


print("SMOKE OK")

