"""停止后的「整篇精修重识别」：用本轮录音把整段音频再识别一遍。

实时链路为了低延迟必须切窗（4 秒）、早触发 flush、每窗独立解码
（condition_on_previous_text=False），所以偶发听错词、半句被切断。精修拿
**整段**录音重跑：不切窗、开启跨句上下文、还可以临时换更大的模型 —— 用时间换
准确率，而且跑在自己的线程里，不打断正在进行的实时识别。

录音来源可以是"录音留档"（CaptionEngine(record=True) 写 records/*.wav，16k 单声道，
一小时约 115MB），也可以是**任意本地视频文件** —— 读音频统一走 media_extract，
所以"给一段录好的网课视频，直接出双语 SRT"这条路是同一段代码。
"""

from __future__ import annotations

from caption_engine import Caption
from translate_engine import needs_translation

_MIN_SAMPLES = 1_600        # 0.1 秒以下不值得跑


# 整篇解码（跨句上下文 + 大 beam）爱把三五句合成一个长段 —— 拿去做成片字幕
# 一条挂 10 秒没法看。所以按标点重切：优先句末标点，其次句中标点，都没标点才按
# 词数硬切；时间轴按字符数比例分配（Whisper 的整篇时间戳本来就是估算值）。
_HARD_PUNCT = ("。", "？", "！", "?", "!", ".", "…")
_SOFT_PUNCT = ("，", ",", "、", ";", "；", ":", "：")
_MAX_SECONDS = 6.0
_MAX_WORDS = 13


def _cut_points(text):
    """返回两组可切下标（标点字符之后的位置）：硬标点在前。"""
    hard, soft = [], []
    for i, ch in enumerate(text):
        if ch in _HARD_PUNCT:
            hard.append(i + 1)
        elif ch in _SOFT_PUNCT:
            soft.append(i + 1)
    return hard, soft


def _units(text, end):
    """切点处的"长度"：西文按词数，没有空格的语言（中/日）按字符数。"""
    return len(text[:end].split()) if " " in text else len(text[:end])


def split_segment(text, parts):
    """把一段文本尽量均匀切成 parts 块，切点优先落在标点后。"""
    text = " ".join(text.split())
    if not text:
        return []
    if parts <= 1:
        return [text]
    total = _units(text, len(text))
    hard, soft = _cut_points(text)
    n = len(text)
    hard = [p for p in hard if p < n]              # 末尾标点切不出第二块，不算候选
    soft = [p for p in soft if p < n]
    if not hard and not soft:                       # 整段没标点：按长度硬切
        words = text.split()
        if " " in text:
            size = -(-len(words) // parts)
            return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
        size = -(-total // parts)
        return [text[i:i + size] for i in range(0, total, size)]
    # 候选切点：句末标点优先；句中标点按"多绕 3 个单位"的代价参与竞争
    cands = [(p, 0) for p in hard] + [(p, 3) for p in soft]
    cuts = []
    for k in range(1, parts):
        ideal = total * k / parts
        pick = min(cands, key=lambda pc: (abs(_units(text, pc[0]) - ideal) + pc[1], pc[0]))
        best = pick[0]
        if cuts and best <= cuts[-1]:
            continue
        if cuts and _units(text, best) - _units(text, cuts[-1]) < 2:
            continue                                # 切出两单位以下碎片，不如不切
        cuts.append(best)
    out, prev = [], 0
    for c in cuts + [n]:
        piece = text[prev:c].strip()
        if piece:
            out.append(piece)
        prev = c
    return out


def refine_segments(segments, max_seconds=_MAX_SECONDS, max_words=_MAX_WORDS):
    """逐段过一遍：超时长/超词数的段切开并重分时间轴，其余原样保留。"""
    out = []
    for start, end, text in segments or []:
        clean = " ".join((text or "").split())
        if not clean:
            continue
        dur = max(0.0, float(end) - float(start))
        n_words = len(clean.split())
        # "超一点就切"会抖：切完的半条又刚好超线，再 refine 一次又被切。
        # 所以超过 25% 才动手，块数按超倍数的上取整 —— 结果是幂等的。
        over = max(dur / max_seconds, float(n_words) / max_words)
        parts = int(-(-over // 1.0)) if over > 1.25 else 1
        pieces = split_segment(clean, parts)
        if len(pieces) <= 1:
            out.append((float(start), float(end), clean))
            continue
        span = sum(len(p) for p in pieces) or 1
        at = float(start)
        for j, p in enumerate(pieces):
            tail = float(end) if j == len(pieces) - 1 else at + dur * len(p) / span
            out.append((round(at, 3), round(max(tail, at + 0.2), 3), p))
            at = tail
    return out

def polished_captions(segments, *, t0: float = 0.0, index0: int = 0,
                      split_long=True) -> list[Caption]:
    """适配器返回的 [(start, end, text)] → 字幕条目（时间轴来自整段录音）。"""
    caps: list[Caption] = []
    if split_long:
        segments = refine_segments(segments)
    for start, end, text in segments or []:
        clean = " ".join((text or "").split())
        if not clean:
            continue
        caps.append(Caption(index=index0 + len(caps), start=float(start),
                            end=float(end), text=clean,
                            wall_clock=float(t0) + float(end)))
    return caps


def polish(model, wav_path, *, language: str | None = None,
           initial_prompt: str | None = None, translator=None,
           on_step=None, t0: float = 0.0):
    """整篇重识别（可选逐句翻译）。返回 [(Caption, 译文)]，失败抛异常由调用方兜。

    wav_path 既可以是本项目录的 16k WAV，也可以是**任意本地视频/音频**（mp4/mkv/mov/
    mp3…）：读音频统一走 media_extract.load_audio_f32，WAV 有标准库快路径。
    translator 传"翻译器对象"（有 .translate(text, context=...)），不是线程 worker ——
    精修是离线跑批，顺序翻译反而能保留跨句上下文，不需要异步。
    """
    import media_extract

    audio = media_extract.load_audio_f32(wav_path)
    if audio.size < _MIN_SAMPLES:
        return []
    result = model.transcribe(audio, language=language, simplify=True,
                              initial_prompt=initial_prompt,
                              condition_on_previous_text=True)
    caps = polished_captions(result.get("segments") if isinstance(result, dict) else None,
                             t0=t0)
    if not caps:                                  # 后端没给逐段（老 mock）：整篇一条
        text = " ".join(((result.get("text") if isinstance(result, dict) else "")
                         or "").split())
        caps = polished_captions([(0.0, audio.size / 16000.0, text)], t0=t0) if text else []

    rows: list[tuple[Caption, str]] = []
    prev = ""
    for i, cap in enumerate(caps):
        trans = cap.text
        # 原文已经是中文（或语种本来就标成 zh）就别机翻：只会丢标点、乱语序
        if translator is not None and needs_translation(cap.text, language):
            try:
                trans = translator.translate(cap.text, context=prev) or cap.text
            except Exception:  # noqa: BLE001：翻译失败回退原文，绝不废掉整篇
                trans = cap.text
        prev = cap.text
        rows.append((cap, trans))
        if on_step is not None:
            on_step(i + 1, len(caps), cap)
    return rows
