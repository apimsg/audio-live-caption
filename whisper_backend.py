"""统一 Whisper 后端：openai-whisper / faster-whisper (CT2 int8)。

faster-whisper 在 CPU 上 int8 量化通常比 openai-whisper 快 3~5 倍，
是把字幕延迟压进 ~1 秒档的关键；两个后端经适配器归一化为同一返回契约：

    transcribe(audio16k, *, language=None, initial_prompt=None,
               condition_on_previous_text=False, **kw)
      -> {"text": str,
          "language": str|None,
          "language_probability": float,
          "speech_end": float|None,   # 最后一个"像语音"的段结束时刻（秒，相对本窗口）
          "has_segments": bool}

speech_end 是引擎做 VAD 早触发/语种漂移检测的依据（②③ 的基础）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from han_convert import is_chinese_lang, to_simplified

_NO_SPEECH_MAX = 0.6   # 段的 no_speech_prob 高于此视为非语音

# ------------------------------------------------------------------ 解码档位
# 为什么要管 beam_size：贪心解码（beam_size=1）一旦在某个 token 上选中了错词，
# 就没有回头路——"the females" 被听成 "the funeral" 这类**近音/先验词**错误正是
# 它的典型翻车姿势（Whisper 论文里的 DogeCoin 例子同源）。
# 本机实测（12 核 CPU / int8 / base / jfk 素材）：
#     beam_size=1  RTF 0.06    beam_size=5  RTF 0.06    beam_size=10 RTF 0.07
# 即"官方默认 beam=5 相比贪心几乎不加延迟"，却把搜索余地找补回来了。
# 所以默认档改为 balanced（=faster-whisper 官方默认），只有算力真紧时才用 fast。
DECODE_PROFILES: dict[str, dict] = {
    "fast": {"beam_size": 1},                        # 最低延迟（易听错近音词）
    "balanced": {"beam_size": 5},                    # Whisper 官方默认，推荐
    "accurate": {"beam_size": 10, "patience": 2.0},  # 搜索更充分，约慢 15~25%
}
DEFAULT_QUALITY = "balanced"

# 只把这些键透传给推理后端：引擎那边还会塞 verbose / fp16 这类 openai-whisper
# 专属参数，faster-whisper 不认识，必须过滤（否则 TypeError）。
_DECODE_KEYS = ("beam_size", "best_of", "patience", "temperature",
                "length_penalty", "repetition_penalty", "no_repeat_ngram_size")


def decode_options(quality: str | None, overrides: dict | None = None) -> dict:
    """取某一档的解码参数。未知档位回落默认档；overrides 只接受白名单键。"""
    q = (quality or DEFAULT_QUALITY).strip().lower()
    opts = dict(DECODE_PROFILES.get(q) or DECODE_PROFILES[DEFAULT_QUALITY])
    for k, v in (overrides or {}).items():
        if k in _DECODE_KEYS and v is not None:
            opts[k] = v
    return opts


def collapse_ws(text: str) -> str:
    """把识别文本压成单行：模型偶尔在段内吐 \\n/连续空白，下游（悬浮窗按
    宽度换行、SRT 行号）都假定"一条字幕 = 一行原文"。"""
    return " ".join((text or "").split())


class _QualityMixin:
    """运行期可改的解码档位（模型对象被 st.cache_resource 复用，所以档位
    不放在构造参数里，由 Runtime 每轮开始 set 一次）。"""

    quality = DEFAULT_QUALITY

    def set_quality(self, quality: str | None) -> str:
        q = (quality or DEFAULT_QUALITY).strip().lower()
        if q not in DECODE_PROFILES:
            raise ValueError(
                f"未知解码档位 {quality!r}，可选：{' / '.join(DECODE_PROFILES)}")
        self.quality = q
        return q


# ------------------------------------------------------------------ 推理设备
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")

# GPU 推理要的是 CUDA 12 运行库（cuBLAS 12 + cuDNN 9）。装了 NVIDIA 驱动只代表
# "设备看得见"，不代表这些 DLL 在 DLL 搜索路径里 —— 缺哪个都在加载模型那一刻
# 抛 RuntimeError，我们捕获后回退 CPU 并把原因原样告诉用户。
GPU_SETUP_HINT = (
    "启用 GPU 需 CUDA 12 运行库：pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
    "（约 850MB）；机器上别处已有的话，把它们的目录（分号分隔）写进环境变量 "
    "ALC_CUDA_BIN 也能直接用。"
)

_CUDA_FAIL = ""          # 记住上一次的失败原因，避免每轮都重试一次慢失败


def cuda_present() -> bool:
    """CTranslate2 是否"看得见"CUDA 设备（有驱动即真，能否用另说）。"""
    try:
        import ctranslate2
        fn = getattr(ctranslate2, "get_cuda_device_count", None)
        return bool(fn and fn() > 0)
    except Exception:  # noqa: BLE001
        return False


_DLL_DIRS_DONE: set[str] = set()


def _pip_nvidia_bin_dirs() -> list[str]:
    """pip 装的 nvidia-*-cu12 把 DLL 放在 site-packages/nvidia/<pkg>/bin。

    CTranslate2 只按 DLL 搜索路径找它们，所以自动注册这些目录 —— 用户不必自己
    配环境变量（配了 ALC_CUDA_BIN 也照样优先，两条路都通）。
    """
    out: list[str] = []
    try:
        import sysconfig

        root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        for sub in sorted(root.glob("*/bin")):
            if sub.is_dir():
                out.append(str(sub))
    except Exception:  # noqa: BLE001
        return []
    return out


def cuda_runtime_dirs() -> list[str]:
    """候选的 CUDA 运行库目录：环境变量 ALC_CUDA_BIN 优先，其次是 pip 的 nvidia/*/bin。"""
    dirs: list[str] = []
    for d in filter(None, (p.strip() for p in
                           os.environ.get("ALC_CUDA_BIN", "").split(os.pathsep))):
        try:
            d = os.path.abspath(d)        # 必须是绝对路径
        except Exception:  # noqa: BLE001
            continue
        if os.path.isdir(d):
            dirs.append(d)
    for d in _pip_nvidia_bin_dirs():
        if d not in dirs:
            dirs.append(d)
    return dirs


# 关键坑（实测踩到）：CTranslate2 是用普通 LoadLibrary **按文件名**找 cuBLAS/cuDNN
# 的，而 AddDllDirectory 登记的目录只对带 LOAD_LIBRARY_SEARCH_USER_DIRS 标志的加载
# 生效 —— 结果 pip 明明装好了 cublas64_12.dll，CT2 依旧报 "not found"。解法是先按
# 绝对路径把它加载进进程：模块名一旦存在，后面按名字加载就会命中同一模块。
_CUDA_DLLS = ("cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_graph64_9.dll",
              "cudnn_cnn64_9.dll", "cudnn_heuristic64_9.dll",
              "cublasLt64_12.dll", "cublas64_12.dll")
_LOADED: dict[str, object] = {}          # 持有句柄，别被 GC 掉


def has_cuda_runtime() -> bool:
    """CUDA 12 运行库在不在（只查文件，不加载）。

    页面渲染时想知道"该不该提示用户可以换大模型"，用这个就够了 —— 真去 ctypes
    加载 cuBLAS/cuDNN 会把 1GB 级 DLL 映射进界面进程，那事该留给加载模型时做。

    注意 pip 是把它们**拆在不同目录**里的（nvidia/cublas/bin、nvidia/cudnn/bin），
    所以必须跨目录汇总判断。早先按"同一目录里既有 cublas 又有 cudnn"来查，在本机
    永远是 False，界面就误报 GPU 不可用（而加载模型时其实能正常跑 CUDA）。
    """
    have_cublas = have_cudnn = False
    for d in cuda_runtime_dirs():
        if os.path.isfile(os.path.join(d, "cublas64_12.dll")):
            have_cublas = True
        # 文件名要盯准：cudnn64_9.dll / cudnn_ops64_9.dll（不是 cudnn_64_9.dll ——
        # 早期写成 f"cudnn{n}_64_9.dll"，于是这个函数永远返回 False，界面误报 GPU 不可用）
        if any(os.path.isfile(os.path.join(d, "cudnn" + n + "64_9.dll"))
               for n in ("", "_ops")):
            have_cudnn = True
        if have_cublas and have_cudnn:
            return True
    return False


def preload_cuda_runtime(dirs: list[str] | None = None) -> list[str]:
    """按绝对路径预加载 CUDA 运行库；返回本次真正加载成功的路径。找不到就返回空。"""
    import ctypes

    loaded: list[str] = []
    for name in _CUDA_DLLS:
        if name in _LOADED:
            continue
        for d in (dirs if dirs is not None else cuda_runtime_dirs()):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            try:
                # 0x8 = LOAD_WITH_ALTERED_SEARCH_PATH（依赖项在同目录里找）
                _LOADED[name] = ctypes.CDLL(p, winmode=0x8)
                loaded.append(p)
                break
            except Exception:  # noqa: BLE001：版本不匹配等，交给后面的探测判活
                continue
    return loaded


def add_cuda_dll_dirs() -> list[str]:
    """登记 CUDA 运行库目录（幂等）+ 预加载。返回本次新登记的目录。"""
    dirs = cuda_runtime_dirs()
    added: list[str] = []
    for d in dirs:
        if d in _DLL_DIRS_DONE:
            continue
        try:
            os.add_dll_directory(d)
            _DLL_DIRS_DONE.add(d)
            added.append(d)
        except Exception:  # noqa: BLE001
            pass
    preload_cuda_runtime(dirs)
    return added


def _probe_infer(adapter) -> None:
    """真跑一次极短推理，验证 GPU 通路真的能用。

    两个坑叠在一起：① CTranslate2 构造 CUDA 模型**不会**报错，缺 cuBLAS/cuDNN 要
    等第一次 encode 才炸；② 用静音探测会被 vad_filter 提前短路（连 encode 都不跑，
    于是"探测通过"是假的）。所以这里关 VAD、喂一个非零小信号，并把生成器消费掉。
    """
    raw = getattr(adapter, "_m", None)
    if raw is None:
        return
    t = np.arange(8_000, dtype=np.float64)
    tone = (0.02 * np.sin(2 * np.pi * 220 * t / 16_000)).astype(np.float32)
    if adapter.backend == "faster":
        segs, _info = raw.transcribe(tone, language="en", beam_size=1, task="transcribe",
                                     vad_filter=False, without_timestamps=True,
                                     condition_on_previous_text=False)
        list(segs)                       # 不消费生成器就不会真的算
    else:
        out = raw.transcribe(tone, language="en", beam_size=1, fp16=adapter.device == "cuda",
                             word_timestamps=False, verbose=None)
        list((out or {}).get("segments") or [])


def _load_faster(size: str, dest: Path, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type=compute_type,
                        download_root=str(dest / "faster"))


def _load_openai(size: str, dest: Path, device: str):
    import whisper

    return whisper.load_model(size, device=device, download_root=str(dest))


def _hf_env_defaults() -> None:
    """huggingface 下载的工程化默认值（只设默认，尊重用户显式配置）。

    - HF_ENDPOINT：huggingface.co 直连不通的环境（国内常见）走 hf-mirror 镜像；
    - HF_HUB_DISABLE_SYMLINKS_ENV：Windows 无"开发者模式"时 HF 缓存用符号链接
      会失败，禁用后改为直接拷贝（普通用户机器也会踩这个坑）；
    - HF_HUB_DISABLE_XET：绕开 xet 大文件协议在受限环境的日志/缓存目录问题。
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class OpenAIAdapter(_QualityMixin):
    backend = "openai"

    def __init__(self, model, device: str = "cpu") -> None:
        self._m = model
        self.device = device
        self.compute_type = "fp16" if device == "cuda" else "fp32"

    def transcribe(self, audio, *, language=None, initial_prompt=None, simplify=False,
                   condition_on_previous_text=False, **kw) -> dict:
        opts = decode_options(self.quality, kw)
        r = self._m.transcribe(
            audio,
            language=language,
            fp16=self.device == "cuda",
            verbose=None,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=condition_on_previous_text,
            beam_size=opts.get("beam_size"),
            patience=opts.get("patience"),
        )
        segs = []
        for s in r.get("segments") or []:
            if float(s.get("no_speech_prob", 0.0) or 0.0) >= _NO_SPEECH_MAX:
                continue
            t = (s.get("text") or "").strip()
            if t:
                segs.append((float(s.get("start", 0.0)), float(s.get("end", 0.0)), t))
        text = (r.get("text") or "").strip() or " ".join(t for _a, _b, t in segs)
        # 抗幻觉：全量 text 会把高 no_speech_prob 的幻觉段也拼进来；
        # 只要模型给了 segments，就以"过滤后的语音段文本"为准。
        if simplify and is_chinese_lang(r.get("language")):
            segs = [(a, b, to_simplified(t)) for a, b, t in segs]
        if r.get("segments"):
            text = " ".join(t for _a, _b, t in segs)
        text = collapse_ws(text)
        if simplify and is_chinese_lang(r.get("language")):
            text = to_simplified(text)
        speech_end = max((b for _a, b, _t in segs), default=None)
        return {"text": text.strip(), "language": r.get("language"),
                "language_probability": 1.0 if r.get("language") else 0.0,
                "speech_end": speech_end, "has_segments": bool(segs),
                "segments": segs}          # 离线精修要用逐段时间轴


class FasterAdapter(_QualityMixin):
    backend = "faster"
    # 引擎在「有音量却连续零字」时会拨掉这个开关（BGM 盖住语音时的自救）
    supports_vad_toggle = True

    def __init__(self, model, device: str = "cpu", compute_type: str = "int8") -> None:
        self._m = model
        self.device = device
        self.compute_type = compute_type
        self.filter_non_speech = True

    def set_non_speech_filter(self, on: bool) -> None:
        """开/关 Silero VAD + no_speech_prob 两道非语音过滤。"""
        self.filter_non_speech = bool(on)

    def transcribe(self, audio, *, language=None, initial_prompt=None,
                   condition_on_previous_text=False, simplify=False, **kw) -> dict:
        filt = self.filter_non_speech
        segments, info = self._m.transcribe(
            audio,
            language=language,
            vad_filter=filt,                  # Silero VAD 过滤非语音（BGM/噪声）
            vad_parameters={"min_silence_duration_ms": 300},
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=condition_on_previous_text,
            **decode_options(self.quality, kw),   # 默认 beam_size=5（官方档）
        )
        segs = []
        for s in segments:                    # 生成器在此消费完
            if filt and float(getattr(s, "no_speech_prob", 0.0) or 0.0) >= _NO_SPEECH_MAX:
                continue
            t = (s.text or "").strip()
            if t:
                segs.append((float(s.start), float(s.end), t))
        if simplify and is_chinese_lang(getattr(info, "language", None)):
            # 中文一律简体（逐词/逐字表；日文绝不进这里）
            segs = [(a, b, to_simplified(t)) for a, b, t in segs]
        text = collapse_ws(" ".join(t for _a, _b, t in segs))
        speech_end = max((b for _a, b, _t in segs), default=None)
        return {"text": text, "language": getattr(info, "language", None),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "speech_end": speech_end, "has_segments": bool(segs),
                "segments": segs}          # 离线精修要用逐段时间轴


def load_model(backend: str, size: str, models_dir: str | Path,
               compute_type: str | None = None, quality: str = DEFAULT_QUALITY,
               device: str = "auto"):
    """加载归一化模型（返回的适配器带 .device / .compute_type / .load_note）。

    device 取 "auto"（看得见 CUDA 就用，失败自动回退 CPU）、"cpu" 或 "cuda"。
    quality 只是初值：模型对象被 st.cache_resource 复用，运行期由 Runtime 调
    set_quality 改档，所以档位不进缓存键。
    """
    global _CUDA_FAIL
    dest = Path(models_dir)
    dest.mkdir(parents=True, exist_ok=True)
    backend = (backend or "faster").strip().lower()
    if backend not in ("faster", "openai"):
        raise RuntimeError(f"未知后端 {backend}，可选：faster / openai")

    want = (device or "auto").strip().lower()
    if want not in SUPPORTED_DEVICES:
        raise RuntimeError(f"未知推理设备 {device}，可选：{' / '.join(SUPPORTED_DEVICES)}")
    add_cuda_dll_dirs()
    note = ""
    use = want
    if want == "auto":
        use = "cuda" if (cuda_present() and not _CUDA_FAIL) else "cpu"
    elif want == "cuda" and _CUDA_FAIL:
        use, note = "cpu", f"上次 GPU 加载失败（{_CUDA_FAIL}），本次直接用 CPU。"

    def _make(dev: str):
        ct = compute_type or ("float16" if dev == "cuda" else "int8")
        if backend == "faster":
            _hf_env_defaults()
            try:
                return FasterAdapter(_load_faster(size, dest, dev, ct),
                                     device=dev, compute_type=ct)
            except ImportError as exc:      # 后端没装是硬错误，不该悄悄回退
                raise RuntimeError(
                    f"未安装 faster-whisper：pip install faster-whisper（{exc}）。"
                    "或改用 openai 后端") from None
        return OpenAIAdapter(_load_openai(size, dest, dev), device=dev)

    try:
        adapter = _make(use)
        if use == "cuda":
            _probe_infer(adapter)
    except Exception as exc:  # noqa: BLE001
        if use == "cpu":
            raise
        _CUDA_FAIL = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:180]
        note = (f"GPU 不可用（{_CUDA_FAIL}）→ 已自动回退 CPU。{GPU_SETUP_HINT}")
        adapter = _make("cpu")
        use = "cpu"
    adapter.load_note = note
    adapter.set_quality(quality)
    return adapter

