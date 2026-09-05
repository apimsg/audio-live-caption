"""真实验证（① 的核心验收）：通过 whisper_backend 统一适配器，
下载 faster-whisper base(int8) + 真实语音素材，实测识别质量与耗时。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

# --- 素材：校验魔数，坏文件重下 ---
wav = Path("models/jfk.flac")


def good_flac(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 100_000:
        return False
    with open(p, "rb") as fh:
        return fh.read(4) == b"fLaC"


if not good_flac(wav):
    import urllib.request

    if wav.exists():
        wav.unlink()
    wav.parent.mkdir(exist_ok=True)
    for url in ("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
                "https://github.com/openai/whisper/raw/main/tests/jfk.flac"):
        try:
            urllib.request.urlretrieve(url, str(wav))
            if good_flac(wav):
                print("素材 OK", wav.stat().st_size, "bytes", flush=True)
                break
        except Exception as exc:  # noqa: BLE001
            print("素材下载失败", url, exc, flush=True)
    else:
        raise SystemExit("no audio sample available")

from faster_whisper.audio import decode_audio

import whisper_backend
from whisper_backend import decode_options as wb_decode_options

audio = decode_audio(str(wav), sampling_rate=16_000)
dur = len(audio) / 16_000
print(f"音频 {dur:.1f}s", flush=True)
assert dur > 5.0, dur

t0 = time.time()
model = whisper_backend.load_model("faster", "base", "models")
print(f"模型加载 {time.time() - t0:.1f}s (backend={model.backend})", flush=True)
# 默认解码档必须是官方档 balanced（beam_size=5）：写回贪心会重新引入
# "the females → the funeral" 这类近音词误识别，而实测本机它并不省时间。
assert getattr(model, "quality", None) == "balanced", getattr(model, "quality", None)
assert wb_decode_options("balanced")["beam_size"] == 5

t1 = time.time()
res = model.transcribe(audio, language=None)
dt = time.time() - t1
print(f"识别耗时 {dt:.2f}s  RTF={dt / dur:.2f}", flush=True)
print("归一化结果:", {k: v for k, v in res.items() if k != "text"}, flush=True)
print("文本:", res["text"], flush=True)

low = res["text"].lower()
assert "ask not" in low and "country" in low, res["text"]
assert res["language"] == "en" and res["speech_end"] and res["has_segments"]
# 2 倍实时以上才算达标（int8 base CPU 通常 6~10x）
assert dt / dur < 0.5, f"RTF 过高: {dt / dur:.2f}"
print("FASTWHISPER REAL OK (RTF %.2f)" % (dt / dur))
