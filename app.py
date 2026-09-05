"""音频实时转字幕 — 统一控制中心（浏览器版）。

一个进程内只跑一份：采集引擎（caption_engine）+ Whisper + 翻译线程；
通过本机 HTTP（caption_service）把同一份字幕同时喂给：
  · 本页面（滚动双语字幕、导出 SRT/TXT）
  · 桌面悬浮窗（desktop_caption.py --follow，置顶/透明/可拖动）
运行： streamlit run app.py   或   run.bat → 选 2
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import han_convert
import streamlit as st

from caption_engine import (
    Caption,
    CaptionEngine,
    list_loopback_devices,
    pick_loud_loopback_device,
    to_srt,
)
from caption_service import CaptionServer, CaptionStore
from desktop_caption import COLORS
from media_extract import MEDIA_SUFFIXES
from translate_engine import TranslatorWorker, is_chinese_text, make_translator

st.set_page_config(page_title="音频实时转字幕", page_icon=":material/mic:", layout="wide")

MAX_CAPTIONS = 400
RENDER_CAPS = 40
LANG_MAP = {"中文": "zh", "English": "en", "日本語": "ja"}
PROVIDER_MAP = {"不翻译": "none", "在线免费（无需 key）": "translators",
                "百度翻译": "baidu", "OpenAI 兼容大模型": "openai"}
# 解码档位 → whisper_backend.DECODE_PROFILES 的键（本机实测 beam=5 与贪心同延迟）
QUALITY_MAP = {"均衡（推荐）": "balanced",
               "速度优先（极端卡时才用）": "fast",
               "高精度（搜索更充分，稍慢）": "accurate"}
# 「整篇精修重识别」可选模型；第一项 = 沿用实时那份（不额外下载权重）
POLISH_SIZES = ["与实时相同", "tiny", "base", "small", "medium", "large-v3"]


# --------------------------------------------------------------------- 缓存
@st.cache_resource(show_spinner="正在加载模型…（faster-whisper 首次会自动下载量化权重）")
def load_model(backend: str, model_size: str, device: str = "auto"):
    from pathlib import Path as _P

    import whisper_backend

    dest = _P(__file__).parent / "models"
    dest.mkdir(exist_ok=True)
    # device 必须进缓存键：CPU 版和 GPU 版是两个不同的模型实例
    return whisper_backend.load_model(backend, model_size, dest, device=device)


@st.cache_resource
def cuda_visible() -> bool:
    import whisper_backend

    return whisper_backend.cuda_present()


@st.cache_resource
def gpu_ready() -> bool:
    """设备看得见 + 运行库文件在位（能不能真跑由加载模型时的探测判活）。"""
    import whisper_backend

    return whisper_backend.cuda_present() and whisper_backend.has_cuda_runtime()


def available_backends() -> list[str]:
    out: list[str] = []
    try:
        import faster_whisper  # noqa: F401
        out.append("faster")
    except Exception:  # noqa: BLE001
        pass
    try:
        import whisper  # noqa: F401
        out.append("openai")
    except Exception:  # noqa: BLE001
        pass
    return out


@st.cache_data(show_spinner=False)
def list_input_devices() -> dict[int, str]:
    import sounddevice as sd

    out: dict[int, str] = {}
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) > 0:
                out[idx] = dev["name"]
    except Exception:  # noqa: BLE001
        return {}
    return out


@st.cache_data(show_spinner=False)
def cached_loopback_devices() -> dict[int, str]:
    return list_loopback_devices()


# ------------------------------------------------------------------ 运行时
class Runtime:
    """进程级单例：唯一引擎 + 唯一翻译线程 + 字幕服务 + 悬浮窗进程管理。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.store = CaptionStore()
        self.server = CaptionServer(self.store).start()
        self.engine: CaptionEngine | None = None
        self.worker: TranslatorWorker | None = None
        self.translating = False
        self.disp_errors = 0      # 分发线程被异常打断的次数（字幕"卡住"时看得见）
        self.disp_restarts = 0    # 分发线程被自愈重启的次数（>0 就说明它偷偷退出过）
        self.captions: list[Caption] = []
        self.trans: dict[int, str] = {}
        # 条目号 → 在 self.captions 里的下标 / 当前原文：同一 index 再次到达就是
        # "原位修订"（半句缝合），修订后旧碎片的译文必须丢弃。
        self._pos: dict[int, int] = {}
        self.text_of: dict[int, str] = {}
        self.proc: subprocess.Popen | None = None
        self.overlay_fp: tuple | None = None
        self.decode_quality = ""   # 当前解码档位（状态行显示用）
        self.model_dev = ""        # 实际推理设备/精度（GPU 回退时必须让用户看得见）
        self.load_note = ""        # 模型加载的非致命说明（如缺 CUDA 库回退原因）
        # 整篇精修重识别（离线）：用本轮录音把整段音频重跑一遍，出更准的字幕。
        self.record_path = ""      # 本轮录音 WAV（由引擎给出；"" = 没录）
        self.polish_src = ""       # 精修实际吃的那个文件（本轮录音或用户挑的视频）
        self.session_t0 = 0.0      # 本轮起始时刻（精修条目的墙钟基准）
        self.polish_rows: list = []          # [(Caption, 译文)]
        self.polish_busy = False
        self.polish_done = 0
        self.polish_total = 0
        self.polish_status = ""
        self.polish_error = ""
        self.polish_secs = 0.0
        self.polish_started_at = 0.0
        self._disp_stop = threading.Event()
        self._disp: threading.Thread | None = None

    # ---- 采集/识别 ------------------------------------------------------
    def running(self) -> bool:
        return self.engine is not None and self.engine.running

    def start(self, model, *, provider="none", appid="", secret="",
              base_url="", api_key="", model_name="",
              decode_quality="balanced", **engine_kw) -> str:
        with self.lock:
            if self.running():
                return ""
            # 解码档位是「每次转写」的参数，改它不用重新加载模型；模型对象被
            # st.cache_resource 复用，所以每轮开始都显式设一次。
            setq = getattr(model, "set_quality", None)
            if callable(setq):
                setq(decode_quality)
            # 模型实例是跨轮复用的：上一轮因 BGM 关过的非语音过滤必须每轮复位，
            # 否则"停止再开始"恢复不了过滤，界面上那句话就是假的。
            unf = getattr(model, "set_non_speech_filter", None)
            if callable(unf):
                unf(True)
            self.decode_quality = getattr(model, "quality", decode_quality)
            self.model_dev = (f"{getattr(model, 'device', '?')}"
                              f"/{getattr(model, 'compute_type', '?')}")
            self.load_note = getattr(model, "load_note", "")
            engine = CaptionEngine(model, **engine_kw)
            self.session_t0 = time.time()
            self.record_path = ""
            if not engine.start():
                return engine.error or "无法启动采集"
            self.engine = engine
            self.store.clear()          # 新一轮：epoch+1，跟随端据此清空重拉
            self.worker = TranslatorWorker(make_translator(
                provider, target="zh-CN",
                appid=appid, secret=secret,
                base_url=base_url, api_key=api_key, model=model_name,
            ))
            self.translating = provider != "none"
            self.worker.start()
            self.captions = []
            self.trans = {}
            self._pos = {}
            self.text_of = {}
            self.store.status.update({"running": True, "hint": "", "error": "",
                                      "source": engine.source})
            self._disp_stop.clear()
            self._disp = threading.Thread(target=self._dispatch, daemon=True)
            self._disp.start()
            return ""

    def _publish(self, cap: Caption) -> None:
        """按条目号落库/落页面：同一 index 再来 = 原位修订（半句缝合走这条路）。"""
        with self.lock:
            pos = self._pos.get(cap.index)
            if pos is None:
                self.captions.append(cap)
                self._pos[cap.index] = len(self.captions) - 1
                if len(self.captions) > MAX_CAPTIONS:
                    del self.captions[:-MAX_CAPTIONS]
                    self._pos = {c.index: n for n, c in enumerate(self.captions)}
                self.trans.pop(cap.index, None)
            elif self.text_of.get(cap.index) != cap.text:
                self.captions[pos] = cap
                self.trans.pop(cap.index, None)   # 旧半句的译文作废
            else:
                self.captions[pos] = cap
            self.text_of[cap.index] = cap.text
        self.store.put(cap.index, src=cap.text, start=cap.start,
                       end=cap.end, wall=cap.wall_clock)

    def diagnostics(self) -> dict:
        """一份"现在到底卡在哪"的状态快照：界面上自查用，也是反馈问题时最有用的一屏。

        绝不抛异常 —— 越是要排查的时候越不能因为排查本身炸掉。
        """
        def g(obj, name, default=""):
            try:
                return getattr(obj, name, default)
            except Exception:  # noqa: BLE001
                return default

        eng, wk = self.engine, self.worker
        mod = g(eng, "model", None)      # 引擎没建时是 None，别拿 "" 当模型
        out: dict = {
            "引擎在跑": bool(eng is not None and eng.thread_alive()),
            "分发线程在跑": bool(self._disp is not None and self._disp.is_alive()),
            "翻译线程在跑": bool(wk is not None and g(wk, "_thread").is_alive()),
            "悬浮窗进程在跑": bool(self.proc is not None and self.proc.poll() is None),
            "已识别段数": g(eng, "processed_chunks", 0),
            "收到音频块": g(eng, "rx_blocks", 0),
            "待识别秒数": round(float(g(eng, "backlog_seconds", 0.0) or 0.0), 1),
            "丢弃样本": g(eng, "dropped_samples", 0),
            "电平": round(float(g(eng, "last_rms", 0.0) or 0.0), 4),
            "钉定语种": g(eng, "language") or "（未钉）",
            "最近检测": g(eng, "detected_language"),
            "语种票": list(g(eng, "_lang_votes", []) or []),
            "切换次数": g(eng, "language_switches", 0),
            "非语音过滤": "已自动关掉（BGM 自救）" if g(eng, "filter_disabled") else "开",
            "连续空窗": g(eng, "empty_windows", 0),   # 有音量却零字的窗数（>3 会触发自救）
            "中文简体化": ("开（" + han_convert.backend_name() + "）"
                          if g(eng, "simplify", True) else "关"),
            "识别线程异常": g(eng, "thread_errors", 0),
            "分发轮异常": self.disp_errors,
            "栈尾": list(g(eng, "thread_error_log", []) or [])[-2:],   # 出错在哪一行
            "分发自愈次数": self.disp_restarts,
            "模型": (f'{self.model_dev or g(mod, "device", "?")} · '
                     f'{g(mod, "compute_type", "?")} · {g(mod, "quality", "?")}'
                     + (f' · {mod.model_size}' if getattr(mod, "model_size", "") else ""))
                  if mod is not None else (self.model_dev or "未加载"),
            "录音留档": (f'{g(eng, "record_path", "") or "无"} · '
                        f'{round(g(eng, "recorded_samples", 0) / 16000.0, 1)}s'
                        + (f' · {g(eng, "record_error")}' if g(eng, "record_error") else "")),
            "字幕条数": len(self.captions),
            "最新条目号": max(self.text_of) if self.text_of else -1,
            "服务epoch": g(self.store, "epoch", "?"),
            "服务rev": g(self.store, "_rev", "?"),   # 单调递增的全局修订号
            "界面错误": g(self.store, "status", {}).get("error", "") if isinstance(
                g(self.store, "status", {}), dict) else "",
        }
        try:
            ages = [time.time() - c.wall_clock for c in self.captions if c.wall_clock]
            out["距上一条字幕"] = f"{min(ages[-1:], default=0.0):.1f}s" if ages else "无"
        except Exception:  # noqa: BLE001
            out["距上一条字幕"] = "?"
        try:
            la = g(eng, "last_audio_at", 0.0) or g(eng, "started_at", 0.0)
            out["采集块龄"] = "%.1fs" % (time.time() - la) if la else "无"
        except Exception:  # noqa: BLE001
            out["采集块龄"] = "?"
        try:
            if g(eng, "decoding_at", 0.0):
                a, b = g(eng, "decoding_span", (0.0, 0.0))
                out["正在解码"] = "%.1f-%.1fs · 已 %.1fs" % (
                    a, b, eng.decoding_age())
            else:
                out["正在解码"] = "空闲"
        except Exception:  # noqa: BLE001
            out["正在解码"] = "?"
        return out

    def ensure_dispatcher(self) -> None:
        """识别线程还活着、分发线程却没了 → 悄悄续上。

        不自愈的话，分发线程任何一次提前退出都会让字幕**永久停在最后一条**，
        而识别还在正常跑（快照里就是 引擎在跑=true / 分发线程在跑=false / 字幕条数不涨）。
        """
        eng = self.engine
        if eng is None or not eng.thread_alive() or self._disp_stop.is_set():
            return
        if self._disp is not None and self._disp.is_alive():
            return
        self.disp_restarts += 1
        self._disp = threading.Thread(target=self._dispatch, daemon=True)
        self._disp.start()

    def threads_ok(self) -> bool:
        """识别线程 + 分发线程都在跑吗（False 就是"字幕卡住"的直接原因）。"""
        eng = self.engine
        if eng is None:
            return False
        return bool(eng.thread_alive() and (self._disp is None or self._disp.is_alive()))

    def _dispatch(self) -> None:
        """后台分发循环：关浏览器标签也不影响悬浮窗继续出字幕。

        外层必须兜异常：这条线程一旦静默死掉，字幕就停在最后一条**而且界面毫无提示**
        （"播中文视频、出了两条就卡住"就是这类症状）。兜住 → 报出来 → 继续跑。
        """
        while not self._disp_stop.is_set():
            try:
                if not self._dispatch_once():
                    return
            except Exception as exc:  # noqa: BLE001
                self.disp_errors += 1
                self.store.status["error"] = ("字幕分发异常（已跳过该轮，继续工作）："
                                              f"{type(exc).__name__}: {exc}")
                eng = self.engine
                if eng is not None:
                    eng.record_thread_error("分发", exc)
                time.sleep(0.3)

    def _dispatch_once(self) -> bool:
        """跑一轮分发；返回 False = 引擎已停，线程可以收工。"""
        engine, worker = self.engine, self.worker
        if engine is None or not engine.running:
            return False                        # 引擎已停：这条线程可以收工
        pinned = engine.language or engine.detected_language or ""
        for cap in engine.drain():
            self._publish(cap)
            if worker is not None:
                # 带上语种：原文已经是中文的句子在 worker 里直接跳过翻译
                worker.submit(cap, self.translating, pinned)
        if worker is not None:
            import queue as _q
            try:
                while True:
                    cap, translated = worker.out.get_nowait()
                    if translated and translated != cap.text:
                        with self.lock:
                            if self.text_of.get(cap.index) != cap.text:
                                continue   # 该条已被修订，旧半句的译文丢掉
                            self.trans[cap.index] = translated
                        self.store.put(cap.index, trans=translated)
            except _q.Empty:
                pass
        self.store.status["hint"] = engine.hint
        self.store.status["error"] = engine.error
        self.store.status["chunks"] = engine.processed_chunks
        self.store.status["level"] = engine.last_rms
        self.store.status["lang"] = engine.language or engine.detected_language
        self.store.status["notice"] = engine.notice
        self.store.status["thread_errors"] = engine.thread_errors
        # "活着但没进展"（挂死）也得看得见：塞进状态，网页与悬浮窗共用
        self.store.status["audio_age"] = round(time.time() - (
            getattr(engine, "last_audio_at", 0.0) or getattr(engine, "started_at", 0.0)), 1)
        self.store.status["decoding"] = (
            "" if not getattr(engine, "decoding_at", 0.0)
            else "%.1f-%.1fs" % engine.decoding_span)
        self.store.status["stall"] = round(
            getattr(engine, "stall_seconds", lambda: 0.0)(), 1)
        time.sleep(0.25)
        return True

    def stop(self) -> None:
        self._disp_stop.set()
        if self._disp is not None:
            self._disp.join(timeout=3)
        engine = self.engine
        if engine is not None:
            engine.stop()
            # 录音写的是"送进模型之前"那份 16k 单声道，不受静音门/背压丢弃影响，
            # 所以精修还能看到实时链路为了追进度而丢掉的音频。
            self.record_path = getattr(engine, "record_path", "") or ""
            if getattr(engine, "record_error", ""):
                self.store.status["hint"] = engine.record_error
            for cap in engine.drain():
                self._publish(cap)
        if self.worker is not None:
            self.worker.stop()
        self.store.status.update({"running": False, "hint": "", "error": ""})
        self.overlay(False)

    # ---- 整篇精修重识别（离线） -----------------------------------------
    def start_polish(self, model, *, language=None, initial_prompt=None,
                     translate=True, src=None) -> str:
        """后台线程跑整篇重识别。返回错误信息（"" 表示已启动）。

        src 不给就用本轮录音；给的话可以是任意本地视频/音频（mp4/mkv/mp3…）——
        "网课视频直接出双语字幕"走的就是这条路。

        只允许在**停止识别之后**跑：模型实例全进程共享，精修要把解码档位临时调到
        "高精度"（离线不在乎慢），和实时链路同时用会互相打架。
        """
        if self.polish_busy:
            return "精修已经在跑了，等它结束。"
        if self.running():
            return "请先点「停止」再做整篇精修（共用一个模型实例，档位不能同时两样）。"
        # src 可以是任意本地视频/音频（读音频统一在 media_extract 里做）
        src = str(src or self.record_path or "")
        if not src or not os.path.isfile(src):
            return ("文件不存在：" + src) if src else \
                "没找到本轮录音：开始前勾选「保留本次录音」，或直接挑一个视频文件。"
        self.polish_src = src
        translator = None
        if translate and self.translating and self.worker is not None:
            translator = self.worker.translator
        self.polish_rows = []
        self.polish_busy = True
        self.polish_error = ""
        self.polish_done = self.polish_total = 0
        self.polish_status = "正在整篇重识别（不切窗 + 跨句上下文 + 高精度解码）…"
        self.polish_started_at = time.time()
        threading.Thread(target=self._polish_run, daemon=True,
                         args=(model, src, language,
                               initial_prompt, translator)).start()
        return ""

    def _polish_step(self, done: int, total: int, cap) -> None:
        self.polish_done, self.polish_total = done, total
        if total > 1:
            self.polish_status = f"精修中：第 {done}/{total} 句（翻译）…"

    def _polish_run(self, model, wav_path, language, initial_prompt, translator) -> None:
        import polish

        prev_quality = getattr(model, "quality", None)
        setq = getattr(model, "set_quality", None)
        t0 = time.time()
        try:
            if callable(setq):
                setq("accurate")        # 离线：beam 开大，用时间换准确率
            self.polish_rows = polish.polish(
                model, wav_path, language=language, initial_prompt=initial_prompt,
                translator=translator, on_step=self._polish_step, t0=self.session_t0)
            if not self.polish_rows and not self.polish_error:
                self.polish_error = "精修没出任何字幕（录音里可能没有语音）。"
        except Exception as exc:  # noqa: BLE001
            self.polish_error = f"精修失败：{exc}"
        finally:
            if callable(setq) and prev_quality:
                try:
                    setq(prev_quality)  # 恢复档位，别影响下一轮实时识别
                except Exception:  # noqa: BLE001
                    pass
            self.polish_secs = time.time() - t0
            self.polish_status = ""
            self.polish_busy = False

    # ---- 悬浮窗进程 -----------------------------------------------------
    def overlay(self, enable: bool, fp: tuple | None = None, argv: list[str] | None = None):
        if not enable:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            self.proc, self.overlay_fp = None, None
            return
        if self.overlay_fp == fp and self.proc is not None and self.proc.poll() is None:
            return  # 配置没变且活着，什么都不做
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "desktop_caption.py")] + (argv or []),
            creationflags=flags,
        )
        self.overlay_fp = fp


@st.cache_resource
def runtime() -> Runtime:
    return Runtime()


RT = runtime()

# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.header(":material/tune: 设置")

    deps_ok = True
    backends = available_backends()
    if not backends:
        deps_ok = False
        st.error("未安装任何 Whisper 后端，请运行：`pip install faster-whisper`（推荐）"
                 "或 `pip install openai-whisper`")

    running = RT.running()

    st.subheader(":material/graphic_eq: 采集")
    source = st.segmented_control(
        "采集源", ["麦克风（说话）", "系统声音（播放视频）"],
        default="系统声音（播放视频）", disabled=running,
        help="选「系统声音」即可给正在播放的视频/音乐实时出字幕（WASAPI 回环）。",
    )
    is_system = source == "系统声音（播放视频）"

    BACKEND_LABELS = {"faster": "faster-whisper（推荐：CPU 快 3~5 倍，延迟 ~1 秒）",
                      "openai": "openai-whisper（原始实现，较慢）"}
    be_opts = [b for b in ("faster", "openai") if b in backends] or ["faster"]
    be_pick = st.selectbox("推理后端", [BACKEND_LABELS[b] for b in be_opts],
                           index=0, disabled=running,
                           help="faster-whisper 用 int8 量化，4 秒窗口推理约 0.1~0.2 秒，"
                                "日语等长句内容也追得上实时；首次使用自动下载量化权重。")
    backend = be_opts[[BACKEND_LABELS[b] for b in be_opts].index(be_pick)]

    DEV_MAP = {"自动（有可用 GPU 就用，否则 CPU）": "auto", "只用 CPU（int8）": "cpu"}
    if cuda_visible():
        DEV_MAP["强制 GPU（CUDA，需已装 cuBLAS12 + cuDNN9）"] = "cuda"
    dev_label = st.selectbox("推理设备", list(DEV_MAP), index=0, disabled=running,
                             help="有 NVIDIA 显卡时可切 GPU。GPU 的意义不是让 base 更快"
                                  "（CPU 实测已是 8~10 倍实时），而是让 medium / large-v3"
                                  "这类更准的模型追得上实时。缺 CUDA 12 运行库会自动回退"
                                  "CPU 并在页面写明原因；机器上别处有现成 DLL 的话，把目录"
                                  "（分号分隔）写进环境变量 ALC_CUDA_BIN 即可。")
    device = DEV_MAP[dev_label]
    if device != "cpu" and gpu_ready():
        st.caption("GPU 运行库已就绪：把下面的模型升到 small / medium 才是真的提准确率"
                   "（本机实测 4 秒窗 small 0.12s、medium 0.27s，都远快于实时；CPU 上"
                   " medium 一个窗要 4.2s，追不上）。")
    model_size = st.select_slider(
        "识别模型", options=["tiny", "base", "small", "medium", "large-v3"], value="base",
        disabled=running, help="越小越快越省资源；首次使用自动下载到 ./models。听不准就"
                               "先试 small，再试 medium；large-v3 建议配 GPU（CPU 上追不上实时）。")
    chunk_seconds = st.slider("每段最长（秒）", 2.0, 10.0, 4.0, 0.5, disabled=running,
                              help="检测到语句停顿会提前送检，实际延迟通常小于此值。")
    vad_on = st.checkbox("按语音段（VAD）判断停顿", value=False,
                         disabled=running,
                         help="用探测级解码判断句子说完没，BGM/音效下 RMS 判不动时才需要。"
                              "实测多数场景关掉延迟更低（VAD 探针会额外占用算力），"
                              "若发现某段背景音很吵、字幕迟迟不出，再来勾选它。")

    unfilter_on = st.checkbox("背景音乐严重时自动关掉「非语音过滤」", value=True, disabled=running,
                              help="默认开。Silero VAD + no_speech_prob 会把唱歌/噪声段过滤掉，"
                                   "但 BGM 盖住人声时它会把整段语音判成非语音 —— 表现就是"
                                   "「电平正常、声音一直在、字幕就是不出」。触发后本会话保持关闭；"
                                   "代价是有时会把歌词也识别进来。")
    simplify_on = st.checkbox("中文一律输出简体（繁体自动转换）", value=True, disabled=running,
                              help="whisper 和机翻后端都常吐繁体字，这里统一转成简体。"
                              + "（当前用：" + han_convert.backend_name() + "；只转中文，"
                              + "日文/韩文同形字一律不动）")
    stitch_on = st.checkbox("自动缝合被切断的半句", value=True, disabled=running,
                            help="停顿早触发有时把一句话切成两条（如 \"They must move.\" +"
                                 " \"As one.\"）。开启后按词法线索把后半句**原位并回**上一条："
                                 "条目号不变、译文跟着重出，首条显示延迟一点不加。"
                                 "若发现本来该分开的两句被误并，取消勾选即可。"
                                 "只对空格分词的语种（英/德/法…）生效，中文不动。")
    language = st.selectbox("语言", ["自动检测", "中文", "English", "日本語"],
                            index=0, disabled=running,
                            help="指定语言最准（听日语就选「日本語」，别用自动）。"
                                 "「自动检测」现在的规则：静音窗里幻觉出的「日本語字幕」"
                                 "一类文本不投票；文字脚本（假名/汉字）优先于模型自评；"
                                 "冷启动要 2 票一致才钉语种；中↔日之间切换要 3 票。")
    quality_label = st.selectbox("识别准确度", list(QUALITY_MAP), index=0, disabled=running,
                                 help="解码时同时考察多少种可能（beam 宽度）。贪心最快但"
                                      "一错到底，典型症状是把近音词听成另一个词（the females"
                                      "→ the funeral）。实测本机 beam=5 与贪心延迟相同，"
                                      "故默认「均衡」（=Whisper 官方默认）；只有换 medium"
                                      "模型或多路并发压满 CPU 时才降到「速度优先」。")
    hotwords = st.text_input("热词 / 提示词（可选）", value="", disabled=running,
                             help="把容易听错的人名、术语、话题词写在这里（逗号或空格分隔），"
                                  "例如：females, gorilla, hierarchy。它作为 Whisper 的 "
                                  "initial_prompt 引导解码，只影响识别用词，不会出现在字幕里。"
                                  "faster-whisper 1.1+ 已移除 hotwords 参数，这是唯一可用的"
                                  "词汇引导手段。上限约 200 字。")
    if hotwords.strip() and LANG_MAP.get(language) == "ja" and is_chinese_text(hotwords):
        # 听日语却喂纯汉字热词，会把模型往中文带（语种和用词双重偏置）
        st.caption(":orange[语言选了日本語，但热词里一个假名都没有 —— 中文热词会把模型往中文带，"
                   "日语术语请写片假名（例：サーバー, 負荷試験）。]")
    keep_wav = st.checkbox("保留本次录音（供整篇精修重识别）", value=True, disabled=running,
                           help="边识别边把送进模型的那份 16kHz 单声道音频写成 "
                                "records/session-*.wav（一小时约 115MB）。停止后可点"
                                "「整篇精修重识别」：拿**整段**录音重跑一遍 —— 不切窗、"
                                "跨句上下文、临时用高精度档位（还能临时换更大的模型），"
                                "比实时字幕准，另存一份导出，不影响实时那份。不需要就取消勾选。")

    device_idx = loopback_device = None
    loopback_devices = {}
    auto_pick = True
    if is_system:
        try:
            import pyaudiowpatch  # noqa: F401
        except Exception:  # noqa: BLE001
            deps_ok = False
            st.error("未安装 PyAudioWPatch，请运行：`pip install PyAudioWPatch`")
        loopback_devices = cached_loopback_devices()
        if not loopback_devices:
            st.warning("未检测到 WASAPI 回环设备。")
        auto_pick = st.checkbox("自动检测有声音的回环设备", value=True,
                                disabled=running or not loopback_devices,
                                help="开始时短暂探测各输出设备音量，自动挑正在出声的那个。")
        if not auto_pick:
            loopback_device = st.selectbox(
                "回环设备（输出设备）", options=list(loopback_devices) or [None],
                format_func=lambda i: loopback_devices.get(i, "系统默认输出的回环"),
                disabled=running or not loopback_devices)
    else:
        devices = list_input_devices()
        if not devices:
            st.warning("未检测到可用的麦克风输入设备。")
        device_idx = st.selectbox("麦克风设备", options=list(devices) or [None],
                                  format_func=lambda i: devices.get(i, "默认设备"),
                                  disabled=running or not devices)

    st.subheader(":material/translate: 实时翻译成中文")
    st.caption("原文本身已是中文的句子会自动跳过翻译 —— 机翻中文只会把标点和语序"
               "弄乱（悬浮窗此时只显示一行）。")
    provider_label = st.selectbox("翻译后端", list(PROVIDER_MAP), index=1, disabled=running)
    provider = PROVIDER_MAP[provider_label]
    appid = secret = base_url = api_key = model_name = ""
    if provider == "openai":
        base_url = st.text_input("base_url", value="https://api.deepseek.com/v1", disabled=running)
        api_key = st.text_input("api_key", type="password", disabled=running)
        model_name = st.text_input("模型名", value="deepseek-chat", disabled=running)
    elif provider == "baidu":
        appid = st.text_input("百度 appid", disabled=running)
        secret = st.text_input("百度 secret", type="password", disabled=running)
    if provider == "none":
        st.caption("只显示原文字幕")

    st.subheader(":material/desktop_windows: 桌面悬浮窗")
    st.caption("与本页共用同一个引擎和音频源，双语绑定显示；网页标题栏地址栏之外也能看。")
    enable_overlay = st.checkbox("启用桌面悬浮字幕窗", value=False)
    ov_entries = st.slider("同屏显示条数", 1, 4, 2)
    ov_font = st.slider("字号", 12, 72, 20)
    ov_color = st.selectbox("原文颜色", list(COLORS), index=0)
    ov_trans_color = st.selectbox("译文颜色", list(COLORS), index=1)
    ov_effect = st.selectbox("字体特效", ["阴影", "描边", "无"], index=0,
                             help="阴影=单向淡影不糊字；描边=同色加粗轮廓")
    ov_font_name = st.text_input("字体", value="黑体")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        start_clicked = st.button(":material/play_arrow: 开始", type="primary",
                                  disabled=not deps_ok or running, width="stretch")
    with col_b:
        stop_clicked = st.button(":material/stop: 停止", disabled=not running, width="stretch")

    if start_clicked:
        chosen_loopback = loopback_device
        if is_system and auto_pick:
            with st.spinner("正在检测各输出设备的音频信号…（请确保视频正在播放）"):
                best, peaks = pick_loud_loopback_device()
            if best is not None and peaks:
                chosen_loopback = best
                st.toast(f"已自动选择：{loopback_devices.get(best, best)}")
            else:
                st.warning("没探测到任何设备有音频：请确认正在播放且未静音。")
        with st.spinner(f"正在加载模型 {model_size}（{backend} / {device}）…"):
            model = load_model(backend, model_size, device)
        err = RT.start(
            model,
            device=device_idx if not is_system else None,
            source="system" if is_system else "mic",
            loopback_device=chosen_loopback,
            chunk_seconds=chunk_seconds,
            language=LANG_MAP.get(language),
            vad_probe=vad_on,
            stitch_fragments=stitch_on,
            decode_quality=QUALITY_MAP[quality_label],
            initial_prompt=hotwords.strip() or None,
            record=keep_wav, record_dir=str(Path(__file__).parent / "records"),
            auto_unfilter=unfilter_on,
            simplify=simplify_on,
            provider=provider, appid=appid, secret=secret,
            base_url=base_url, api_key=api_key, model_name=model_name,
        )
        if err:
            st.error(err)
        st.rerun()

    if stop_clicked:
        RT.stop()
        st.rerun()

    st.metric("状态", "运行中" if running else "已停止")
    st.caption(f"字幕服务：{RT.server.url}")

# ---------------------------------------------------------------- 悬浮窗管理
ov_fp = (ov_entries, ov_font, ov_color, ov_trans_color, ov_effect, ov_font_name)
ov_argv = [
    "--follow", RT.server.url,
    "--entries", str(ov_entries),
    "--font-size", str(ov_font),
    "--color", COLORS[ov_color],
    "--trans-color", COLORS[ov_trans_color],
    "--effect", {"阴影": "shadow", "描边": "outline", "无": "none"}[ov_effect],
    "--font", ov_font_name,
]
try:
    RT.overlay(enable_overlay, ov_fp, ov_argv)
except Exception as exc:  # noqa: BLE001
    st.error(f"悬浮窗启动失败：{exc}")

# ---------------------------------------------------------------------- 顶部
st.title(":material/mic: 音频实时转字幕 · 控制中心")
st.caption("本地 Whisper 识别（唯一引擎）→ 本页与桌面悬浮窗同时显示，双语绑定，可导出 SRT / TXT")

if not RT.running() and RT.store.status.get("error"):
    st.error(RT.store.status["error"])


# ------------------------------------------------------ 实时区（局部自动刷新）
@st.fragment(run_every="1s")
def live_region() -> None:
    if not RT.running():
        st.caption("未运行。配置好后点左侧「开始」，然后说话或播放视频。")
    else:
        eng = RT.engine
        RT.ensure_dispatcher()   # 分发线程掉了就自动续上（不靠人重启）
        level = min(1.0, max(0.0, eng.last_rms * 12))
        backlog = eng.backlog_seconds
        src = "系统声音" if eng.source == "system" else "麦克风"
        lang_now = eng.language or eng.detected_language or "检测中"
        st.progress(level, text=(
            f"{src} · 语种 {lang_now} · 准确度 {getattr(RT, 'decode_quality', '')} · "
            f"设备 {getattr(RT, 'model_dev', '')} · "
            f"已识别 {eng.processed_chunks} 段（缝合 {eng.stitcher.merges} 处） · "
            f"时长 {eng.elapsed():.0f}s · "
            f"待识别 {backlog:.1f}s · 悬浮窗 {'已开启' if RT.proc else '未开启'}"
            + (f" · 异常跳过 {eng.thread_errors + RT.disp_errors} 次"
               if eng.thread_errors or RT.disp_errors else "")
            + (f" · 分发已自愈 {RT.disp_restarts} 次" if RT.disp_restarts else "")
            + ("" if RT.threads_ok() else
               " · ⚠ 有线程已退出：点「停止」再「开始」")))
        with st.expander(":material/search: 卡住时自查（当前状态快照）", expanded=False):
            st.json(RT.diagnostics())
            st.caption("读法：**引擎在跑=False** → 线程死了或已停，点「停止」再「开始」；"
                       "**引擎在跑=True 但已识别段数不涨** → 音频没进来或被静音门挡住"
                       "（看「收到音频块」「电平」）；**待识别秒数一直涨** → 模型追不上；"
                       "**栈尾**非空就是刚才异常的那一行。")
        if eng.thread_error_log:
            # 线程兜住过一次异常 = 这里就是"卡住"的原因，把栈尾直接摆在界面上
            st.code(eng.thread_error_log[-1], language=None)
        if eng.filter_disabled:
            st.info(":material/music_off: 背景音乐太重，语音一度被「非语音过滤」整段吃掉；"
                    "已自动关掉该过滤，字幕恢复正常（代价是有时会把歌词也识别进来）。"
                    "想恢复过滤：停止再开始。")
        stall = eng.stall_seconds()
        if stall > 0:
            a, b = eng.decoding_span
            st.error(f"识别**挂住**了：第 {a:.1f}–{b:.1f} 秒这一窗已经解了 "
                     f"{stall + 8.0:.0f} 秒还没返回。GPU/CT2 调用挂住时**不抛异常**，"
                     "所以以前什么提示都没有 —— 只能看见字幕不动。"
                     "点「停止」再「开始」即可恢复；反复出现就把模型降一档，"
                     "或把设备切到 CPU 验证是不是显卡的问题。")
        elif eng.stall_seconds(2.0) > 0:
            st.caption(":orange[这一窗解得偏慢（>2s）：模型/档位对当前设备偏重，"
                       "积压很快会开始增长。]")
        if backlog > 6:
            st.warning(f"识别追不上音频：已积压 {backlog:.0f}s —— 模型/档位对当前设备太重了。"
                       "有 GPU 就把上面的设备切到 GPU（或「强制 GPU」），或把模型降一档"
                       "（medium→small→base）；否则字幕会越来越晚，看起来就像卡住。")
        # 语种自动切换：只在新一次切换时提示一次
        sw = eng.language_switches
        seen = st.session_state.setdefault("_lang_sw", sw)
        if sw > seen:
            st.session_state["_lang_sw"] = sw
            st.toast(f":material/languages: {eng.notice}", icon="🌐")
        if backlog > eng.chunk_seconds * 2:
            st.warning(f"识别跟不上采集（积压 {backlog:.0f}s），已自动丢弃最旧音频。"
                       "请改用更小的模型或加大「每段时长」。")
        if eng.hint:
            st.warning(eng.hint)
        if getattr(RT, "load_note", ""):
            st.warning(RT.load_note)

    with st.container(height=460, border=True, autoscroll=True):
        with RT.lock:
            caps = list(RT.captions[-RENDER_CAPS:])
            trans = dict(RT.trans)
        if not caps:
            st.info("暂无字幕。", icon=":material/subtitles:")
        else:
            for cap in caps:
                tr = trans.get(cap.index)
                if tr:
                    st.markdown(f"**{cap.label}** {cap.text}\n\n"
                                f"<span style='color:#B8860B'>{tr}</span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"**{cap.label}** {cap.text}")
                st.divider()
        n = len(RT.captions)
        if n > RENDER_CAPS:
            st.caption(f"页面显示最近 {RENDER_CAPS} 条，导出包含全部 {n} 条")
live_region()


# ------------------------------------------------ 整篇精修重识别（离线）
@st.fragment(run_every=1.0)
def polish_region() -> None:
    """用本轮录音把整段音频重跑一遍：实时字幕不动，另外产出一份"精修版"。"""
    if not (RT.record_path or RT.polish_src or RT.polish_busy
            or RT.polish_rows or RT.polish_error):
        return
    wav = RT.polish_src or RT.record_path
    try:
        size = Path(wav).stat().st_size if wav else 0
    except OSError:
        size = 0
    audio_s = size / 32_000.0          # 先按 16k WAV 估个底，探测成功再用真值
    if size:
        try:
            import media_extract

            audio_s = media_extract.probe_media(wav)["duration"] or audio_s
        except Exception:  # noqa: BLE001：探测不了就留着体积反推的估值
            pass
    st.markdown("### :material/auto_fix_high: 整篇精修重识别（离线）")
    if size:
        st.caption(f"精修对象 `${os.path.basename(wav)}`（{audio_s / 60:.1f} 分钟；源文件 "
                   f"{size / 1e6:.0f}MB）。实时字幕是 4 秒一窗抢时效切出来的；精修把整段"
                   "音频一次喂给模型：不切窗、跨句上下文、解码临时用「高精度」，还可以临时"
                   "换更大的模型 —— 用时间换更准的成片字幕，实时那份不受影响。")
    else:
        st.caption("本轮没有录音文件（开始前勾「保留本次录音」；或者在下面「本地视频提取」"
                   "里挑个视频，直接精修出双语字幕）。")
    if RT.polish_busy and RT.running():
        st.info("精修和实时识别共用同一个模型实例（档位不能同时两样）：先点「停止」，"
                "再点「开始精修」。")
    c1, c2 = st.columns([2, 1])
    with c1:
        psize = st.selectbox("精修用模型", POLISH_SIZES, index=0, disabled=RT.polish_busy,
                             help="「与实时相同」不额外下载；换 small/medium 首次会自动下载权重"
                                  "（small 约 480MB、medium 约 1.5GB）。CPU 上 medium 整篇很慢，"
                                  "有 GPU 就放心换。")
    with c2:
        st.write("")
        go = st.button(":material/auto_fix_high: 开始精修", type="primary",
                       disabled=RT.polish_busy or RT.running() or not size,
                       width="stretch")
    if go:
        try:
            if psize == POLISH_SIZES[0]:
                pm = load_model(backend, model_size, device)
            else:
                with st.spinner(f"正在加载精修模型 {psize}（{device}）…"):
                    pm = load_model(backend, psize, device)
        except Exception as exc:  # noqa: BLE001
            pm = None
            st.error(f"模型不可用：{exc}")
        if pm is not None:
            perr = RT.start_polish(pm, language=LANG_MAP.get(language),
                                   initial_prompt=hotwords.strip() or None,
                                   translate=(provider != "none"))
            if perr:
                st.warning(perr)
    if RT.polish_busy:
        el = time.time() - (RT.polish_started_at or time.time())
        if RT.polish_total:
            st.progress(min(1.0, RT.polish_done / RT.polish_total), text=RT.polish_status)
        else:
            st.progress(0.03, text=f"{RT.polish_status} 已用 {el:.0f}s（整段录音 "
                                  f"{audio_s:.0f}s）… 大模型整篇要等一会儿，可以先放着")
    elif RT.polish_error:
        st.error(RT.polish_error)
    if RT.polish_rows:
        rows = list(RT.polish_rows)
        st.success(f"精修完成：{len(rows)} 条，用时 {RT.polish_secs:.0f}s"
                   f"（整段 {audio_s:.0f}s 音频）")
        with st.container(height=280, border=True):
            for cap, tr in rows[-30:]:
                if tr and tr != cap.text:
                    st.markdown(f"**{cap.label}** {cap.text}\n\n"
                                f"<span style='color:#B8860B'>{tr}</span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"**{cap.label}** {cap.text}")
        biling = [replace(c, text=c.text + (("\n" + t) if t and t != c.text else ""))
                  for c, t in rows]
        d1, d2, d3 = st.columns(3)
        d1.download_button(":material/download: 精修版 SRT", data=to_srt(biling),
                           file_name="captions_polished.srt", mime="application/x-subrip")
        txt = "\n".join(f"{c.label}  {c.text}"
                         + (f"\n    {t}" if t and t != c.text else "") for c, t in rows)
        d2.download_button(":material/download: 精修版 TXT", data=txt.encode("utf-8"),
                           file_name="captions_polished.txt", mime="text/plain")
        if d3.button(":material/delete: 删除本轮录音", width="stretch",
                     disabled=not RT.record_path):
            try:
                os.remove(RT.record_path)
            except OSError:
                pass
            RT.record_path = ""
            if RT.polish_src and not RT.polish_busy:
                RT.polish_src = ""
                RT.polish_rows = []
            st.rerun()


# ------------------------------------------- 本地视频 → 16k 单声道 WAV
def pick_file(title: str = "选择文件") -> str:
    """弹一个系统文件选择框。Streamlit 的上传控件只会把文件搬进服务器缓存，
    要"就地读原路径的视频"（几 GB 的文件不该复制一份）只能自己弹对话框。

    设了环境变量 ALC_PICK 就直接返回它（不弹框）：无桌面会话/远程/自动化时用。
    """
    pre = os.environ.get("ALC_PICK", "").strip()
    if pre:
        return pre
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:  # noqa: BLE001：没 tkinter 就在输入框里手打路径
        return ""
    try:
        root = tk.Tk()
    except Exception:  # noqa: BLE001：远程/无桌面会话下开不出窗口
        return ""
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        kinds = " ".join("*" + s for s in MEDIA_SUFFIXES)
        return filedialog.askopenfilename(
            title=title, filetypes=[("视频/音频", kinds), ("所有文件", "*.*")]) or ""
    finally:
        root.destroy()   # Tk 只能在建它的那个线程里用，所以每次现建现销


def _x_browse() -> None:
    """「浏览」的回调：只能在回调里写 x_src。

    写成 button 的 if 分支体就晚了 —— 那时同一轮的 text_input(key="x_src") 已经
    实例化完，Streamlit 会抛 WidgetAlreadyInstantiatedError。回调在重渲染前跑，合法。
    """
    got = pick_file("选择视频 / 音频文件")
    if got:
        st.session_state["x_src"] = got


def reveal_in_folder(path: str) -> None:
    """在资源管理器里选中这个文件（非 Windows 就只打开所在目录）。"""
    p = Path(path)
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent)])
    except OSError:
        pass


def _xst() -> dict:
    """提取任务的状态。放 session_state：它和实时识别链路（Runtime）互不相干。"""
    return st.session_state.setdefault("extract", {
        "busy": False, "done": 0.0, "total": 0.0, "paths": [], "error": "",
        "secs": 0.0, "wall": 0.0, "probed": "", "probe": None})


def _xrun(x: dict, src: str, out_dir: str, start: float, duration: float,
          part_mb) -> None:
    """后台线程里跑提取：一小时的视频要几十秒，别把界面钉住。"""
    import media_extract as me

    t0 = time.time()
    try:
        paths, secs = me.extract_wav(src, out_dir.strip() or None, start=start,
                                     duration=duration or None, part_mb=part_mb,
                                     on_progress=lambda s: x.update(done=s))
        x["paths"] = [str(p) for p in paths]
        x["secs"] = secs
    except Exception as exc:  # noqa: BLE001
        x["error"] = str(exc) or repr(exc)
    finally:
        x["wall"] = time.time() - t0
        x["busy"] = False


@st.fragment(run_every=1.0)
def extract_card() -> None:
    """抽音轨给云端模型：实时链路只能听声卡，而云端 ASR 要的是文件。

    16k 单声道 16bit 就是 Whisper 系内部使用的格式（约 1.8MB/分钟），云端不用再
    转码；按大小切段是因为多数接口有 25MB 上限（OpenAI Whisper API 就是）。
    """
    import media_extract as me

    x = _xst()
    with st.expander(":material/movie_filter: 本地视频 → 16k 单声道 WAV（给云端大模型）",
                     expanded=bool(x["busy"] or x["paths"])):
        st.caption(me.backend_note())
        if not me.media_backend():
            st.warning("没有可用的解码器，这一步做不了（实时识别不受影响）。"
                       "按上面的提示装一个就能用。")
            return
        c1, c2 = st.columns([5, 1])
        src = c1.text_input("视频/音频文件路径", key="x_src",
                            placeholder="把路径粘到这里，或右边浏览选择",
                            label_visibility="collapsed")
        c2.write("")
        c2.button(":material/folder_open: 浏览", key="x_browse", width="stretch",
                  on_click=_x_browse)
        if not src or not os.path.isfile(src):
            st.caption("mp4 / mkv / mov / webm / m4a / mp3 / wav … 都可以。")
            return
        if x["probed"] != src:
            x["probed"], x["error"], x["probe"] = src, "", None
            try:
                x["probe"] = me.probe_media(src)
            except Exception as exc:  # noqa: BLE001
                x["error"] = str(exc) or repr(exc)
        if x["error"]:
            st.error(x["error"])
            return
        info = x["probe"] or {}
        if not info.get("has_audio"):
            st.error("这个文件里没有音轨，抽不出音频。")
            return
        dur = float(info.get("duration") or 0.0)
        kinds = "、".join(sorted({str(s.get("type")) for s in info.get("streams", [])}))
        st.caption(f"时长 {dur / 60:.1f} 分钟 · 轨道 {kinds} · 导出的 16k 单声道 WAV 约 "
                   f"{info.get('wav_mb', 0):.0f}MB（约 1.8MB/分钟）")
        k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
        start = k1.number_input("起始（秒）", min_value=0.0, value=0.0, step=1.0,
                                key="x_start", disabled=x["busy"])
        ddur = k2.number_input("时长（秒，0=到结尾）", min_value=0.0, value=0.0,
                               step=5.0, key="x_dur", disabled=x["busy"])
        pmb = k3.number_input("每段上限（MB）", min_value=0.0,
                              value=float(me.DEFAULT_PART_MB), step=1.0, key="x_part",
                              disabled=x["busy"],
                              help="云端接口常有 25MB 上限，默认 24 留点余量；填 0 = 不切段")
        out_dir = k4.text_input("输出目录", key="x_out", disabled=x["busy"],
                                placeholder="留空 = 视频所在目录")
        want = float(ddur) if ddur else max(0.0, dur - float(start))
        est = want * me.BYTES_PER_SEC
        n_parts = int(-(-est // (float(pmb) * me.MB))) if pmb else 1
        b1, b2 = st.columns([1, 1])
        go = b1.button(":material/graphic_eq: 提取 16k WAV", type="primary",
                       disabled=x["busy"] or want <= 0, width="stretch",
                       help="约 " + f"{est / 1e6:.0f}" + "MB，切成 " + str(max(1, n_parts)) + " 段")
        rec = b2.button(":material/auto_fix_high: 不提取，直接本机识别出双语字幕",
                       disabled=x["busy"], width="stretch",
                       help="整段喂给本机模型（不切窗 + 高精度解码）；需要模型空闲：先停止实时识别")
        if go:
            x.update({"busy": True, "done": 0.0, "total": want, "paths": [],
                      "error": "", "secs": 0.0, "wall": 0.0})
            threading.Thread(target=_xrun, daemon=True,
                             args=(x, src, out_dir, float(start), float(ddur),
                                   float(pmb) or None)).start()
        if rec:
            try:
                pm = load_model(backend, model_size, device)
                perr = RT.start_polish(pm, language=LANG_MAP.get(language),
                                       initial_prompt=hotwords.strip() or None,
                                       translate=(provider != "none"), src=src)
                if perr:
                    st.warning(perr)
                else:
                    st.toast("已开始整篇精修，进度看上面的精修区")
            except Exception as exc:  # noqa: BLE001
                st.error(f"模型不可用：{exc}")
        if x["busy"]:
            frac = min(1.0, x["done"] / x["total"]) if x["total"] else 0.0
            st.progress(frac if frac > 0.02 else 0.02,
                        text=f"已提取 {x['done']:.0f}s / {x['total']:.0f}s…")
        elif x["error"]:
            st.error(x["error"])
        elif x["paths"]:
            st.success(f"提取完成：{len(x['paths'])} 个文件 · {x['secs']:.0f}s 音频 · "
                       f"用时 {x['wall']:.1f}s")
            for i_pth, pth in enumerate(x["paths"]):
                if not os.path.isfile(pth):
                    continue
                sz = os.path.getsize(pth)
                name = Path(pth).name
                if sz <= 25 * me.MB:
                    st.download_button(f":material/download: {name}（{sz / 1e6:.1f}MB）",
                                       data=Path(pth).read_bytes(), file_name=name,
                                       mime="audio/wav",
                                       key="dl_%d_%s_%s" % (i_pth, Path(pth).parent.name, name))
                else:
                    st.markdown(name + f"（{sz / 1e6:.1f}MB）—— 太大不在页面下载，"
                                       "用下面的按钮到文件夹里拿")
            if st.button(":material/folder_open: 打开所在文件夹", key="x_reveal"):
                reveal_in_folder(x["paths"][0])
            st.caption("云端直接用：16kHz / 单声道 / 16bit PCM 就是 Whisper 系的内部格式，"
                       "接口不用再转码；切段后每个文件都卡在 25MB 上限内。")


polish_region()
extract_card()

# -------------------------------------------------------------- 导出 / 清空
with st.container(horizontal=True):
    if st.button(":material/delete: 清空字幕"):
        with RT.lock:
            RT.captions = []
            RT.trans = {}
        st.rerun()
    if RT.captions:
        with RT.lock:
            caps = list(RT.captions)
            trans = dict(RT.trans)
        bilingual = [replace(c, text=c.text + ("\n" + trans[c.index]
                        if trans.get(c.index) else "")) for c in caps]
        st.download_button(":material/download: 下载 SRT（双语）", data=to_srt(bilingual),
                           file_name="captions.srt", mime="application/x-subrip")
        txt = "\n".join(
            f"{c.label}  {c.text}" + (f"\n    {trans[c.index]}" if trans.get(c.index) else "")
            for c in caps)
        st.download_button(":material/download: 下载 TXT（双语）",
                           data=txt.encode("utf-8"),
                           file_name="captions.txt", mime="text/plain")

st.caption(
    "延迟 ≈「句尾停顿时长 + 推理时间」：VAD 语音段判停（约 0.6 秒）+ faster-whisper 推理约 0.1~0.3 秒，"
    "整句说完约 1~2 秒出原文；语言「自动检测」持续探测，中途换语种会自动跟随并提示。"
    "播放视频请选「系统声音」源，且播放软件的输出设备要和所选回环设备一致（自动检测会帮你挑）。")
