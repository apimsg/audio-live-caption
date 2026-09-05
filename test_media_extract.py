"""本地视频 → 16k 单声道 WAV 的回归测试（不联网、不要麦克风、不要 ffmpeg.exe）。

样本由测试自己用 PyAV 现造（真 mp4：h264 画面 + aac 音轨），所以这条同时也验证了
"解码器到底在不在"。断言都按行为写，不绑定 pyav 还是外部 ffmpeg。
"""

import os
import sys
import wave

import numpy as np

sys.path.insert(0, ".")

import media_extract as me
from caption_engine import write_wav_mono16

SR = me.TARGET_SR
TMP = os.path.join(".tmp", "media-test")
os.makedirs(TMP, exist_ok=True)


def _add_video(c, size):
    """给输出容器加一条视频轨；编码器不在就返回 None（样本退化成纯音轨）。"""
    import av

    try:
        v = c.add_stream("libx264", rate=25)
    except Exception:  # noqa: BLE001
        return None
    v.width = v.height = size
    v.pix_fmt = "yuv420p"
    return v                      # 别自己去设 time_base：PyAV 18 下 mux 会崩


def _make_mp4(path, seconds=3.0, hz=440.0, rate=48000, video=True):
    """造一个真视频文件：h264 画面 + aac 正弦音轨。"""
    import av

    c = av.open(path, mode="w")
    a = c.add_stream("aac", rate=rate)
    a.bit_rate = 96000
    v = _add_video(c, 64) if video else None
    tone = (0.3 * np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate)
            * 32767).astype(np.int16)
    for i in range(0, tone.size, 1024):
        ch = tone[i:i + 1024]
        if ch.size < 1024:
            ch = np.concatenate([ch, np.zeros(1024 - ch.size, np.int16)])
        fr = av.AudioFrame.from_ndarray(ch.reshape(1, -1), format="s16", layout="mono")
        fr.sample_rate = rate
        for p in a.encode(fr):
            c.mux(p)
    for p in a.encode():
        c.mux(p)
    if v is not None:
        for n in range(int(25 * seconds)):
            img = np.full((64, 64, 3), (n * 3) % 255, dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf.pts = n
            for p in v.encode(vf):
                c.mux(p)
        for p in v.encode():
            c.mux(p)
    c.close()
    return path, v is not None


def _make_png(path):
    """只有画面没有音轨的"媒体文件"（PNG 在 PyAV 里也是一条视频流）。"""
    import av

    c = av.open(path, mode="w", format="image2")
    s = c.add_stream("png")
    s.width = s.height = 16
    s.pix_fmt = "rgb24"
    fr = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
    for p in s.encode(fr):
        c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return path


assert me.media_backend() in ("pyav", "ffmpeg"), me.backend_note()
print("backend:", me.media_backend(), "|", me.backend_note())

clip, has_video = _make_mp4(os.path.join(TMP, "clip.mp4"))
out = os.path.join(TMP, "out")
import shutil

shutil.rmtree(out, ignore_errors=True)     # 断言里写死了首个输出名，必须从空目录开始
os.makedirs(out, exist_ok=True)

info = me.probe_media(clip)
assert info["has_audio"] and 2.5 <= info["duration"] <= 3.6, info
assert has_video == any(s["type"] == "video" for s in info["streams"]), info["streams"]
assert abs(info["wav_bytes"] - info["duration"] * me.BYTES_PER_SEC) <= 1, info
assert info["wav_mb"] == info["wav_bytes"] / me.MB

# 提取：必须是标准 16k / 单声道 / 16bit 的 PCM WAV（云端接口认的就是这个）
paths, secs = me.extract_wav(clip, out)
assert len(paths) == 1 and paths[0].name == "clip-16k.wav", paths
with wave.open(str(paths[0])) as wf:
    assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, SR)
    n_frames, raw = wf.getnframes(), wf.readframes(wf.getnframes())
assert abs(n_frames / SR - secs) < 1e-6, (n_frames, secs)
assert 2.5 <= secs <= 3.6, secs
assert len(raw) == n_frames * 2

# 重采样得保住音高：48k 的 440Hz 正弦 → 16k 之后还应该在 440Hz 附近
a16 = me.load_audio_f32(clip)
assert a16.dtype == np.float32 and a16.size == n_frames, (a16.dtype, a16.size)
sp = np.abs(np.fft.rfft(a16 * np.hanning(a16.size)))
assert 400.0 <= float(np.argmax(sp)) * SR / a16.size <= 480.0, "音高被重采样搞坏了"
assert 0.05 <= float(np.sqrt(np.mean(a16 ** 2))) <= 0.6, "有电平但不削顶"

# 按大小切段（云端接口有 25MB 上限）：每段都不许超，拼回去时长不能丢
p2, s2 = me.extract_wav(clip, out, part_mb=0.02)
assert len(p2) >= 4, p2
assert all(os.path.getsize(str(p)) <= int(0.02 * me.MB) for p in p2), p2
tot = np.concatenate([np.frombuffer(wave.open(str(p)).readframes(-1), dtype="<i2")
                      for p in p2])
assert tot.size == a16.size, (tot.size, a16.size)
assert abs(s2 - a16.size / SR) < 0.05, s2
assert all("-part" in p.name for p in p2[1:]), [p.name for p in p2]

# 起点 / 时长：只要中间那一秒
p3, s3 = me.extract_wav(clip, out, start=1.0, duration=1.0)
assert 0.9 <= s3 <= 1.2, s3
assert p3[0].name == "clip-16k-1s-2s.wav", p3[0].name

# 不许悄悄覆盖上一次的成果
p4, _ = me.extract_wav(clip, out)
p5, _ = me.extract_wav(clip, out)
assert p4[0] != p5[0] and p4[0].exists() and p5[0].exists(), (p4[0], p5[0])

# 错误路径：中文、可读、不留空文件
try:
    me.probe_media(os.path.join(TMP, "nope.mp4"))
    raise AssertionError("不存在的文件该报错")
except me.MediaError as e:
    assert "不存在" in str(e), e
try:
    me.extract_wav(_make_png(os.path.join(TMP, "no_audio.png")), out)
    raise AssertionError("没音轨的文件该报错")
except me.MediaError as e:
    assert "音轨" in str(e) or "音频" in str(e), e
before = set(os.listdir(out))
try:
    me.extract_wav(clip, out, start=9999.0)
    raise AssertionError("起点超出该报错")
except me.MediaError as e:
    assert "音频" in str(e) or "起点" in str(e), e
assert set(os.listdir(out)) <= before, "失败时不该留下半个文件"

# 本项目录的 16k WAV 走标准库快路径（不依赖解码器，数值必须一模一样）
plain = os.path.join(TMP, "plain.wav")
sig = (0.25 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)).astype(np.float32)
write_wav_mono16(plain, sig)
back = me.load_audio_f32(plain)
assert back.size == SR and np.allclose(back, sig, atol=2.0 / 32768), back[:3]

# 外部 ffmpeg 兜底：至少命令行构造是对的（本机没装 binary 也只能量这个）
argv = me._ffmpeg_argv(me.Path("a b.mp4"), 12.5)
assert argv[-1] == "-" and "-ss" in argv and "12.500" in argv
assert argv[argv.index("-ar") + 1] == str(SR) and argv[argv.index("-ac") + 1] == "1"
assert argv[argv.index("-f") + 1] == "s16le" and "-vn" in argv
assert "a b.mp4" in argv

# 精修能直接吃视频（不用先手动转 WAV）
import polish

SEGS = [(0.0, 3.0, "hello there my friend, and so on keep going for a while today and "
                   "tomorrow and next week and the week after that"),
        (3.0, 3.03, "the end.")]


class _M:
    language = None

    def transcribe(self, audio, **kw):
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
        assert abs(audio.size - SR * 3) < SR, audio.size      # 真从 mp4 里解出了 3 秒
        assert kw.get("condition_on_previous_text") is True
        return {"text": "", "segments": SEGS}

rows = polish.polish(_M(), clip)
assert len(rows) == 3, [r[0].text for r in rows]      # 超长段被按逗号切开
assert rows[0][0].text == "hello there my friend,", rows[0][0].text
assert rows[-1][0].text == "the end.", rows[-1][0].text
assert 0.0 <= rows[0][0].start < rows[1][0].start < 3.03, rows
assert rows[-1][0].end <= 3.04, rows[-1][0].end       # 时间轴落在视频时长内
print("polish 吃视频 ->", [r[0].text for r in rows])

print("MEDIA EXTRACT OK")
