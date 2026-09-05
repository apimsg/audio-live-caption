"""融合架构回归测试（全部无需麦克风/网络/可见窗口）：

1. EntryBoard：原文与译文按条绑定，乱序到达也能对上，超容量整条淘汰
2. CaptionStore/CaptionServer/CaptionClient：浏览器进程↔悬浮窗进程 JSON 往返
3. 引擎语言缓存：自动检测只做一次，后续送检带固定 language
4. 静音早触发：说话停顿后不满一窗也立刻送检（压低延迟的关键）
5. wrap_lines / board_to_lines 纯函数
"""

import os
import shutil
import sys
import threading
import time

sys.path.insert(0, ".")

import numpy as np

import polish
from caption_engine import (TARGET_SR, CaptionEngine, read_wav_mono16, to_srt,
                           write_wav_mono16)


def to_srt_frag(caps):
    return to_srt(caps)
from caption_service import CaptionClient, CaptionServer, CaptionStore
from desktop_caption import EntryBoard, board_to_lines, wrap_lines

# --- 1. 字幕板：双语绑定 ---
b = EntryBoard(2)
b.put_src(1, "A one")
b.put_src(2, "B two")
b.put_trans(2, "译文二")
b.put_trans(1, "译文一")          # 译文晚到/乱序也必须绑定回原条目
snap = b.snapshot()
assert snap == [(1, {"src": "A one", "trans": "译文一"}),
                (2, {"src": "B two", "trans": "译文二"})], snap
b.put_src(3, "C three")           # 超容量：整条淘汰最旧，不留半截双语
snap = b.snapshot()
assert [i for i, _ in snap] == [2, 3], snap
b.put_trans(1, "不该出现")        # 已淘汰条目的迟到译文应被忽略
assert all("不该出现" not in v["trans"] for _, v in b.snapshot())

lines = board_to_lines(b, 10_000, lambda t: len(t))
assert lines == [("src", "B two"), ("tr", "译文二"), ("src", "C three")], lines
print("board pairing OK")

# --- 2. 共享服务往返（rev/epoch 增量协议） ---
store = CaptionStore()
srv = CaptionServer(store).start()
try:
    cl = CaptionClient(srv.url)
    store.put(7, src="hello world", start=0.0, end=4.0, wall=1.0)
    resp = cl.fetch(-1)
    it = resp["items"][0]
    assert it["i"] == 7 and it["src"] == "hello world" and it["trans"] == "", it
    rev1 = it["rev"]
    assert cl.fetch(rev1)["items"] == []          # 无更新时增量为空

    # 关键回归：译文晚于原文到达，也必须被跟随端拉到（旧协议按编号过滤会漏）
    store.put(7, trans="你好世界")
    patch = cl.fetch(rev1)["items"]
    assert len(patch) == 1 and patch[0]["i"] == 7 and patch[0]["trans"] == "你好世界", patch
    rev2 = patch[0]["rev"]
    store.put(7, trans="你好世界")                # 相同值回填不产生新 rev
    assert cl.fetch(rev2)["items"] == []

    # epoch：clear（新一轮识别）后条目清空、epoch+1，跟随端据此重置
    e0 = resp["status"]["epoch"]
    store.put(9, src="next")
    store.clear()
    r = cl.fetch(-1)
    assert r["status"]["epoch"] == e0 + 1 and r["items"] == []
finally:
    srv.stop()
print("caption service roundtrip OK")

# --- 3. 语言缓存 ---
class _LangModel:
    def __init__(self):
        self.langs = []

    def transcribe(self, audio, *a, **k):
        self.langs.append(k.get("language"))
        return {"text": "hello there", "language": "en"}


lm = _LangModel()
eng = CaptionEngine(lm, chunk_seconds=1.0)            # language=None → 自动
loud = np.ones(32_000, dtype=np.float32) * 0.2
eng._transcribe(loud, 0.0, 2.0)
eng._transcribe(loud, 2.0, 4.0)
eng._transcribe(loud, 4.0, 6.0)
assert lm.langs == [None, None, "en"], lm.langs   # 连续 2 票一致才钉，第 3 窗起复用语种
assert eng.language == "en"
print("language cache OK")

# --- 4. 静音早触发（走真实 _run 线程路径） ---
class _EarlyModel:
    def __init__(self):
        self.sizes = []

    def transcribe(self, audio, *a, **k):
        self.sizes.append(int(audio.shape[0]))
        return {"text": "just a short sentence"}


em = _EarlyModel()
eng2 = CaptionEngine(em, chunk_seconds=4.0, flush_after_silence=0.3)
eng2.started_at = time.time()
eng2._thread = threading.Thread(target=eng2._run, daemon=True)
eng2._thread.start()
try:
    voice = (np.ones(8_000, dtype=np.float32) * 0.2)          # 0.5s 语音×3 = 1.5s
    for _ in range(3):
        eng2._audio_q.put((voice.copy(), 16_000))
        time.sleep(0.05)
    quiet = np.zeros(8_000, dtype=np.float32)                 # 0.5s 静音触发早送检
    for _ in range(2):
        eng2._audio_q.put((quiet.copy(), 16_000))
        time.sleep(0.05)
    caps = []
    deadline = time.time() + 4
    while not caps and time.time() < deadline:
        caps = eng2.drain()
        time.sleep(0.1)
    assert caps, "静音早触发没有产出字幕"
    assert em.sizes[0] < 4 * 16_000, em.sizes   # 关键：窗口不足 4s 就被送检
    print("silence early-flush OK:", em.sizes[0], "samples")
finally:
    eng2._stop.set()
    eng2._thread.join(timeout=3)

# --- 6. 翻译上下文链（worker 把上一句传给翻译器）+ 缓存键含上下文 ---
from caption_engine import Caption
from translate_engine import Translator, TranslatorWorker

rec = []


class _CtxTranslator:
    last_error = ""

    def translate(self, text, context=""):
        rec.append((text, context))
        return "T:" + text


wkr = TranslatorWorker(_CtxTranslator())
wkr.start()
wkr.submit(Caption(1, 0.0, 1.0, "first line", 0.0), True)
wkr.submit(Caption(2, 1.0, 2.0, "second line", 0.0), True)
_t1, tt1 = wkr.out.get(timeout=3)
_t2, tt2 = wkr.out.get(timeout=3)
wkr.stop()
assert tt1 == "T:first line" and tt2 == "T:second line"
assert rec == [("first line", ""), ("second line", "first line")], rec

# --- 6b. 原文已是中文就不送翻译（根治"中文到中文错乱"；粤语/日语照旧要翻） ---
from translate_engine import is_chinese_text, needs_translation

assert needs_translation('hello there', 'en') and needs_translation('hello there', None)
assert needs_translation('会議は三時です', 'ja') and needs_translation('会議は三時です', None)
assert not needs_translation('我们明天见', 'zh') and not needs_translation('我们明天见', None)
assert not needs_translation('我们明天见', 'zh-CN')
assert needs_translation('唔該晒', 'yue')          # 粤语写成汉字，但正需要转普通话
assert is_chinese_text('会議は三時です') is False   # 出现假名就不算中文


class _Counting:
    last_error = ""

    def __init__(self):
        self.calls = []

    def translate(self, text, context=""):
        self.calls.append(text)
        return "译:" + text


tr2 = _Counting()
wkr2 = TranslatorWorker(tr2)
wkr2.start()
cap_zh = Caption(1, 0.0, 1.0, "我们明天见", 0.0)
wkr2.submit(cap_zh, True, "zh")                 # 语种中文：翻译器一次都不该被调用
wkr2.submit(Caption(2, 1.0, 2.0, "会議は三時です", 0.0), True, "ja")
wkr2.submit(Caption(3, 2.0, 3.0, "see you there", 0.0), True, "")   # 语种未知：脚本兜底
got6 = {}
while len(got6) < 3:
    c, txt = wkr2.out.get(timeout=3)
    got6[c.index] = txt
wkr2.stop()
assert tr2.calls == ["会議は三時です", "see you there"], tr2.calls
assert got6[1] == cap_zh.text, got6      # 哨兵：译文=原文 → 界面/悬浮窗只出一行
assert got6[2] == "译:会議は三時です" and got6[3] == "译:see you there", got6
print("same-language skip OK")


calls = {"n": 0}


class _CtxCount(Translator):
    def _do_translate(self, text, context=""):
        calls["n"] += 1
        return "T:" + text


ct = _CtxCount()
ct.translate("x", context="a")
ct.translate("x", context="b")      # 上下文不同 → 不能命中缓存
ct.translate("x", context="a")      # 相同 → 命中
assert calls["n"] == 2, calls

# upsert：跟随端落条目（创建/回填/更新）
b2 = EntryBoard(2)
b2.upsert(5, src="hello")
b2.upsert(5, trans="你好")           # 原位补译文
b2.upsert(5, src="hello world")      # 原文修正也要生效
assert b2.snapshot() == [(5, {"src": "hello world", "trans": "你好"})], b2.snapshot()
b2.upsert(6, trans="孤立译文")        # 无原文不创建
assert len(b2.snapshot()) == 1
print("translation context + upsert OK")

# --- 8. 半句缝合：早触发切断的句子按"原位修订"并回去 ---
import caption_engine as ceng


class _FragModel:
    """两次送检吐出被切成两半的同一句。"""

    def __init__(self, texts=("They must move.", "As one.")):
        self.texts = list(texts)
        self.n = 0

    def transcribe(self, audio, *a, **k):
        t = self.texts[min(self.n, len(self.texts) - 1)]
        self.n += 1
        return {"text": t, "language": "en", "language_probability": 0.99}


assert ceng.stitch_texts("They must move.", "As one.") == "They must move as one."
assert ceng.stitch_texts("They are fighting", "for access to the females.") \
    == "They are fighting for access to the females."
assert ceng.stitch_texts("Talk to the", "people now.") == "Talk to the people now."
# 绝不该并的几种：真句尾 / 后条不是接续词开头 / 中文无词边界 / 并完超长
assert ceng.stitch_texts("Really?", "as one.") is None
assert ceng.stitch_texts("It was a good day.", "The end.") is None
assert ceng.stitch_texts("他们必须移动", "如同一体") is None
assert ceng.stitch_texts("They must move.", "as one " * 40) is None

loud9 = np.ones(32_000, dtype=np.float32) * 0.2
eng9 = CaptionEngine(_FragModel(), chunk_seconds=1.0, vad_probe=False)
eng9._transcribe(loud9, 0.0, 2.0)
eng9._transcribe(loud9, 2.0, 4.0)
frag = eng9.drain()
assert [x.index for x in frag] == [1, 1], frag          # 同一条目号 = 修订
assert frag[0].text == "They must move.", frag[0].text
assert frag[-1].text == "They must move as one.", frag[-1].text
assert frag[-1].end == 4.0 and eng9.stitcher.merges == 1
assert eng9.processed_chunks == 2
assert len(to_srt_frag(frag[-1:]).splitlines()) == 3

eng10 = CaptionEngine(_FragModel(), chunk_seconds=1.0, vad_probe=False,
                      stitch_fragments=False)
eng10._transcribe(loud9, 0.0, 2.0)
eng10._transcribe(loud9, 2.0, 4.0)
raw = eng10.drain()
assert [x.index for x in raw] == [1, 2], raw            # 关掉缝合→维持两条
assert eng10.stitcher.merges == 0

# 缝合链：三个碎片逐步并成整句，条目号始终是第一条的
eng11 = CaptionEngine(_FragModel(["Go", "and", "win."]), chunk_seconds=1.0,
                      vad_probe=False)
for i in range(3):
    eng11._transcribe(loud9, i * 2.0, i * 2.0 + 2.0)
chain = eng11.drain()
assert [x.index for x in chain] == [1, 1, 1], chain
assert chain[-1].text == "Go and win.", chain[-1].text

# 共享服务：原文修订必须让旧译文作废（否则英文整句配中文半句）
st2 = CaptionStore()
st2.put(3, src="They must move.", start=0.0, end=2.0, wall=1.0)
st2.put(3, trans="他们必须移动。")
before = st2.snapshot()[0]
assert before["trans"] == "他们必须移动。"
st2.put(3, src="They must move as one.")                # 修订（条目号不变）
after = st2.snapshot()[0]
assert after["src"] == "They must move as one." and after["trans"] == "", after
assert after["rev"] > before["rev"], after              # rev 递增 → 跟随端会拉到
b3 = EntryBoard(2)
b3.upsert(3, src="They must move.", trans="他们必须移动。")
b3.upsert(3, src="They must move as one.", trans="")    # 客户端照收作废指令
assert b3.snapshot() == [(3, {"src": "They must move as one.", "trans": ""})], b3.snapshot()
print("sentence stitching + revision OK")

# --- 9. 录音留档 + 整篇精修重识别 ---
# 用工作区里的目录：受限环境下系统 TEMP 常常只许建目录不许写文件
tmpd = os.path.join(".tmp", "alc-rec-%d" % os.getpid())
os.makedirs(tmpd, exist_ok=True)
try:
    eng12 = CaptionEngine(_FragModel(), chunk_seconds=1.0, vad_probe=False,
                          record=True, record_dir=tmpd)
    eng12._open_recorder()
    assert eng12.record_path and os.path.isfile(eng12.record_path), eng12.record_path
    assert eng12.record_error == "", eng12.record_error
    tone = (0.4 * np.sin(2 * np.pi * 300 * np.arange(4800) / TARGET_SR)).astype(np.float32)
    eng12._write_record(tone)
    eng12._write_record(tone)
    eng12._close_recorder()
    back = read_wav_mono16(eng12.record_path)
    assert eng12.recorded_samples == 9600 and back.shape[0] == 9600, eng12.recorded_samples
    assert float(np.abs(back - np.concatenate([tone, tone])).max()) < 1e-3   # 16bit 量化误差内
    # 不勾选时零开销：不开文件、不占磁盘
    eng13 = CaptionEngine(_FragModel(), chunk_seconds=1.0, vad_probe=False)
    eng13._open_recorder()
    assert eng13.record_path == "" and eng13._rec is None
    eng13._write_record(tone)                      # 不该炸
    eng13._close_recorder()

    class _WholeModel:
        """整篇重识别的模型形状：一次给全部逐段时间轴。"""

        quality = "balanced"

        def __init__(self, segments, text=""):
            self._segs, self._text, self.kw = segments, text, {}

        def transcribe(self, audio, **kw):
            self.kw = kw
            return {"text": self._text or " ".join(t for _a, _b, t in self._segs),
                    "language": "en", "language_probability": 1.0,
                    "speech_end": None, "has_segments": bool(self._segs),
                    "segments": self._segs}

    class _Trans:
        def __init__(self, boom=False):
            self.calls, self.boom = [], boom

        def translate(self, text, context=""):
            self.calls.append((text, context))
            if self.boom:
                raise RuntimeError("网络断了")
            return "中文：" + text

    segs9 = [(0.0, 2.0, "They must move."), (2.0, 3.4, "  as   one."), (3.4, 3.6, "  ")]
    wm = _WholeModel(segs9)
    steps = []
    rows = polish.polish(wm, eng12.record_path, language="en", t0=1000.0,
                         on_step=lambda d, t, c: steps.append((d, t)))
    assert wm.kw.get("condition_on_previous_text") is True, wm.kw      # 精修必须开启跨句上下文
    assert len(rows) == 2, rows                                        # 空段被丢掉
    caps9 = [c for c, _t in rows]
    assert [c.text for c in caps9] == ["They must move.", "as one."], [c.text for c in caps9]
    assert [c.index for c in caps9] == [0, 1] and caps9[1].start == 2.0
    assert caps9[0].wall_clock == 1002.0, caps9[0].wall_clock          # 墙钟 = 会话起点 + 段尾
    assert steps == [(1, 2), (2, 2)], steps
    assert to_srt([c for c, _t in rows]).splitlines()[1].startswith("00:00:00,000")

    tr = _Trans()
    rows2 = polish.polish(_WholeModel(segs9), eng12.record_path, translator=tr)
    assert [t for _c, t in rows2] == ["中文：They must move.", "中文：as one."], rows2
    assert tr.calls[1][1] == "They must move.", tr.calls               # 上一句原文当上下文
    rows3 = polish.polish(_WholeModel(segs9), eng12.record_path, translator=_Trans(boom=True))
    assert [t for _c, t in rows3] == ["They must move.", "as one."], rows3  # 翻译挂了就回退原文

    # 中文原文 + 语种=zh：精修也不许机翻中文（只会丢标点、乱语序）
    zh_segs = [(0.0, 2.0, "我们明天下午三点开会。"), (2.0, 4.0, "记得带合同。")]
    tr_zh = _Trans()
    rows_zh = polish.polish(_WholeModel(zh_segs), eng12.record_path,
                            translator=tr_zh, language="zh")
    assert tr_zh.calls == [], tr_zh.calls
    assert [t for _c, t in rows_zh] == ["我们明天下午三点开会。", "记得带合同。"], rows_zh
    # 同样的汉字文本，语种是粤语 → 必须翻（要转成普通话）
    tr_yue = _Trans()
    polish.polish(_WholeModel(zh_segs), eng12.record_path,
                  translator=tr_yue, language="yue")
    assert len(tr_yue.calls) == 2, tr_yue.calls

    only_text = _WholeModel([], text="hello  world")
    rows4 = polish.polish(only_text, eng12.record_path)
    assert len(rows4) == 1 and rows4[0][0].text == "hello world", rows4  # 后端不给逐段→整篇一条
    assert polish.polish(_WholeModel([]), eng12.record_path) == []
    # 整篇解码爱合成超长段 → 按标点重切，且必须幂等（否则反复 refine 会一直抖）
    long9 = [(0.0, 10.58, "And so my fellow Americans, ask not what your country "
                          "can do for you, ask what you can do for your country.")]
    cut9 = polish.refine_segments(long9)
    assert len(cut9) == 2, cut9
    assert cut9[0][2].endswith("for you,") and cut9[1][2].startswith("ask"), cut9
    assert cut9[0][0] == 0.0 and abs(cut9[-1][1] - 10.58) < 1e-6, cut9
    assert polish.refine_segments(cut9) == cut9, "refine 必须幂等"
    assert polish.refine_segments([(0.0, 2.0, "Hello there.")]) == [(0.0, 2.0, "Hello there.")]
    assert len(polish.refine_segments(
        [(0.0, 30.0, " ".join("w%d," % i for i in range(60)))])
    ) >= 4, "超长段必须切开"
    zh9 = polish.refine_segments([(0.0, 12.0, "第一句话。第二句话。第三句话讲完了。")])
    assert len(zh9) == 2 and zh9[0][2] == "第一句话。第二句话。", zh9
    assert polish.split_segment("one two three four five six", 3) \
        == ["one two", "three four", "five six"]
    assert len(polish.polished_captions(long9, t0=100.0)) == 2
    assert len(polish.polished_captions(long9, split_long=False)) == 1
    assert polish.polished_captions(None) == []
finally:
    shutil.rmtree(tmpd, ignore_errors=True)
print("record + offline polish OK")

# --- 7. 折行纯函数 ---
segs = wrap_lines("十十十十十十十十", 50, lambda t: len(t) * 10)
assert segs == ["十十十十十", "十十十"], segs
assert wrap_lines("a\nb", 999, lambda t: len(t)) == ["a", "b"]
print("FUSION OK")
