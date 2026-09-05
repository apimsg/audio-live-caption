"""本地视频/音频 → 16kHz 单声道 16bit WAV（喂给云端大模型做转写）。

实时链路只能"听声卡"，而云端 ASR / 多模态模型要的是**文件**。16k 单声道 16bit 正是
Whisper 系内部使用的格式 —— 云端拿到它不需要再转码，体积也只有原始音轨的零头
（约 1.8MB/分钟，一小时 115MB）。

解码优先用 PyAV（wheel 自带整套 FFmpeg 库，机器上不用装 ffmpeg.exe）；只有外部
ffmpeg 时走子进程兜底；两个都没有就把该装什么写清楚。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16_000
BYTES_PER_SEC = TARGET_SR * 2          # 16bit 单声道
MB = 1_000_000
# 多数云端转写接口有 25MB 上限（OpenAI Whisper API 就是），默认按 24MB 切，留点余量
DEFAULT_PART_MB = 24
WAV_HEADER_BYTES = 44            # RIFF 头也要算进云端的文件大小账里
# 界面接受的后缀（真正能不能解，最后由解码器说了算）
MEDIA_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts", ".flv", ".m4v",
                  ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus", ".wma",
                  ".m2ts", ".mpg", ".mpeg", ".wmv", ".3gp")


class MediaError(RuntimeError):
    """给界面看的中文错误：消息本身就够用户处理，不必再看堆栈。"""


def media_backend() -> str:
    """"pyav" / "ffmpeg" / ""（都没有）。PyAV 优先：不依赖系统 PATH。"""
    try:
        import av  # noqa: F401

        return "pyav"
    except Exception:  # noqa: BLE001
        return "ffmpeg" if find_ffmpeg() else ""


def find_ffmpeg() -> str:
    """外部 ffmpeg 可执行文件：环境变量 ALC_FFMPEG（全路径）优先，其次 PATH。"""
    env = os.environ.get("ALC_FFMPEG", "").strip()
    if env and os.path.isfile(env):
        return env
    return shutil.which("ffmpeg") or ""


def backend_note() -> str:
    """一句人话：当前靠什么解码，缺了什么该怎么补。"""
    b = media_backend()
    if b == "pyav":
        return "解码器：PyAV（wheel 自带 FFmpeg 库，无需另装）"
    if b == "ffmpeg":
        return f"解码器：外部 ffmpeg（{find_ffmpeg()}）"
    return ("本机既没有 PyAV 也没有 ffmpeg.exe：装一个即可 —— "
            "「pip install av」（推荐，自带解码库），或装 ffmpeg 并把可执行文件"
            "全路径写进环境变量 ALC_FFMPEG。")


def _check_input(src) -> Path:
    p = Path(src)
    if not p.is_file():
        raise MediaError(f"文件不存在：{p}")
    return p


def probe_media(src) -> dict:
    """读容器信息：时长、轨道、字节数、有没有音轨、导出的 WAV 体积预估。"""
    p = _check_input(src)
    out = {"path": str(p), "bytes": p.stat().st_size, "duration": 0.0,
           "streams": [], "has_audio": False, "backend": media_backend()}
    if out["backend"] == "pyav":
        import av

        try:
            c = av.open(str(p))
        except Exception as exc:  # noqa: BLE001
            raise MediaError(f"打不开这个文件（{exc}）") from None
        try:
            if c.duration:
                out["duration"] = float(c.duration) / 1_000_000.0   # 容器时长单位=微秒
            for s in c.streams:
                cc = getattr(s, "codec_context", None)
                info = {"type": s.type, "codec": getattr(cc, "name", "?")}
                if s.type == "audio" and cc is not None:
                    info["rate"] = getattr(cc, "rate", None)
                    info["layout"] = getattr(getattr(cc, "layout", None), "name", None)
                    if s.duration and s.time_base:
                        info["duration"] = float(s.duration) * float(s.time_base)
                    out["has_audio"] = True
                    if not out["duration"]:
                        out["duration"] = info.get("duration") or 0.0
                out["streams"].append(info)
        finally:
            c.close()
    elif out["backend"] == "ffmpeg":
        import re

        # 没装 ffprobe 也能问：ffmpeg 只给 -i 时会把容器信息打到 stderr
        txt = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(p)],
                             capture_output=True, text=True, errors="ignore").stderr or ""
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", txt)
        if m:
            out["duration"] = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                               + float(m.group(3)))
        out["has_audio"] = "Audio:" in txt
        out["streams"] = [t.strip() for t in re.findall(r"\)\s+((?:Video|Audio):.*)", txt)]
    else:
        raise MediaError(backend_note())
    out["wav_bytes"] = int(round(out["duration"] * BYTES_PER_SEC))
    out["wav_mb"] = out["wav_bytes"] / MB
    return out


def _iter_pyav(path: Path, start: float, until):
    """PyAV：解码音轨并边解边重采样成 16k 单声道 s16。"""
    import av

    c = av.open(str(path))
    try:
        if not c.streams.audio:
            raise MediaError("这个文件里没有音轨（只有画面）。")
        a = c.streams.audio[0]
        rs = av.audio.resampler.AudioResampler(format="s16", layout="mono",
                                               rate=TARGET_SR)
        if start > 0:
            # 偏移单位是这条流的 time_base；只能落到关键帧附近，剩下的用跳样本补齐
            try:
                c.seek(int(start / float(a.time_base)), stream=a)
            except Exception:  # noqa: BLE001
                pass
        skipped = start if start > 0 else 0.0
        done = 0.0
        for frame in c.decode(a):
            for out in rs.resample(frame) or []:
                buf = out.to_ndarray().reshape(-1)
                if skipped > 0:
                    drop = min(int(skipped * TARGET_SR), buf.size)
                    buf = buf[drop:]
                    skipped -= drop / TARGET_SR
                if buf.size == 0:
                    continue
                yield buf
                done += buf.size / TARGET_SR
                if until is not None and done >= until:
                    return
    finally:
        c.close()


def _ffmpeg_argv(path: Path, start: float) -> list[str]:
    """外部 ffmpeg 兜底的命令行：裸 PCM(s16le/16k/单声道) 打到标准输出。"""
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(TARGET_SR),
            "-f", "s16le", "-"]
    return cmd


def _iter_ffmpeg(path: Path, start: float, until):
    exe = find_ffmpeg()
    if not exe:
        raise MediaError(backend_note())
    popen = subprocess.Popen(_ffmpeg_argv(path, start), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, bufsize=1 << 20)
    done = 0.0
    try:
        while True:
            raw = popen.stdout.read(BYTES_PER_SEC)        # 一次一秒，进度平滑
            if not raw:
                break
            buf = np.frombuffer(raw, dtype="<i2")
            done += buf.size / TARGET_SR
            yield buf
            if until is not None and done >= until:
                break
    finally:
        try:
            popen.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        if popen.poll() is None:
            popen.kill()
        popen.wait(timeout=5)
    if done == 0:
        raise MediaError("ffmpeg 没能从这个文件里解出音频（编码不支持？）")


def iter_pcm16(src, *, start: float = 0.0, duration=None):
    """产出 16k 单声道 int16 数据块（list 的每一项都是连续数组）。"""
    p = _check_input(src)
    start = max(0.0, float(start or 0.0))
    until = float(duration) if duration and float(duration) > 0 else None
    b = media_backend()
    if b == "pyav":
        it = _iter_pyav(p, start, until)
    elif b == "ffmpeg":
        it = _iter_ffmpeg(p, start, until)
    else:
        raise MediaError(backend_note())
    for buf in it:
        if buf is not None and buf.size:
            yield np.ascontiguousarray(buf, dtype="<i2")


def load_audio_f32(src, *, start: float = 0.0, duration=None) -> np.ndarray:
    """整段读成 16k 单声道 float32（本地识别/精修直接吃这个）。"""
    if Path(src).suffix.lower() == ".wav":
        try:
            from caption_engine import read_wav_mono16

            return read_wav_mono16(src)
        except Exception:  # noqa: BLE001：不是我们那种 16k 单声道 WAV，交给解码器
            pass
    parts = list(iter_pcm16(src, start=start, duration=duration))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32) / 32768.0


def out_path_for(src, out_dir=None, *, start: float = 0.0, duration=None) -> Path:
    """默认输出路径：<out_dir 或同目录>/<原名>-16k[-起-止].wav。"""
    p = Path(src)
    tag = "-16k"
    if start or duration:
        tag += "-%ds-%ds" % (int(start or 0), int(start or 0) + int(duration or 0))
    d = Path(out_dir) if out_dir else p.parent
    return d / f"{p.stem}{tag}.wav"



def extract_wav(src, out_dir=None, *, start: float = 0.0, duration=None,
                part_mb=None, on_progress=None):
    """提取 16k 单声道 WAV；给了 part_mb 就按大小切段（云端接口常用）。

    返回 (文件列表, 已提取秒数)。on_progress(秒) 用来刷界面进度。
    """
    p = _check_input(src)
    base = out_path_for(p, out_dir, start=start, duration=duration)
    base.parent.mkdir(parents=True, exist_ok=True)
    # 切段时先把 RIFF 头的额度扣掉，否则每段都会比上限多出 44 字节
    max_samples = ((int(float(part_mb) * MB) - WAV_HEADER_BYTES) // 2) if part_mb else None

    def _name(prefix: str, no: int) -> str:
        if not max_samples and no == 0:
            return prefix + ".wav"
        return f"{prefix}-part{no + 1:02d}.wav"

    n = 0
    while True:                       # 已有同名文件就换个序号：绝不悄悄覆盖
        prefix = base.stem if n == 0 else f"{base.stem}-{n}"
        probe = [_name(prefix, i) for i in range(2 if max_samples else 1)]
        if not any((base.parent / q).exists() for q in probe):
            break
        n += 1
    paths: list[Path] = []
    written = 0
    wf = None

    def _open(part_no: int):
        path = base.parent / _name(prefix, part_no)
        h = wave.open(str(path), "wb")
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(TARGET_SR)
        paths.append(path)
        return h

    try:
        part_no = in_part = 0
        for buf in iter_pcm16(p, start=start, duration=duration):
            pos = 0
            while pos < buf.size:
                if wf is None:
                    wf, in_part = _open(part_no), 0
                room = buf.size - pos
                if max_samples:
                    room = min(room, max_samples - in_part)
                if room <= 0:                   # 这一段满了：开下一段
                    wf.close()
                    wf, part_no = None, part_no + 1
                    continue
                wf.writeframes(np.ascontiguousarray(buf[pos:pos + room]).tobytes())
                pos += room
                in_part += room
                written += room
                if on_progress is not None:
                    on_progress(written / TARGET_SR)
    finally:
        if wf is not None:
            wf.close()
    if written == 0:
        for q in paths:
            try:
                q.unlink()
            except OSError:
                pass
        raise MediaError("没解出任何音频（文件为空，或起点已超出时长）。")
    return paths, written / TARGET_SR

