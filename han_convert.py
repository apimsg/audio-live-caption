"""繁体 → 简体：字幕里的中文一律固定成简体。

背景：whisper 听中文时常输出繁体，机翻后端也常给繁体；要的是简体字幕，
所以「识别原文」和「译文」两条出口都要归一。

两层实现：
  ① 装了 zhconv（纯 Python、OpenCC 词库）就用它 —— 带词级校正，最准；
  ② 没装就用下面这份内置高频对照表（字幕常见繁体字基本都在里面）。

两者都只是逐字/逐词替换，所以**只允许对中文轨道调用**：日文汉字共用大量同形字，
拿去转日文会把用字改错，那是不可逆的字幕污染（判据见 is_chinese_lang）。
"""

from __future__ import annotations

# 繁→简，一格一对（组内第 1 字=繁，第 2 字=简），以空格分隔。
# 不用两条平行字符串：那种写法少一个字符就整体错位（本文件真的错过一次）。
# 长度不对当场炸在导入期 —— 宁可装不上，也不要给出错的字幕。
_PAIRS = (
    "們们 這这 說说 話话 語语 詞词 記记 訴诉 "
    "讀读 寫写 學学 問问 應应 該该 請请 謝谢 "
    "識识 譯译 護护 讚赞 讓让 給给 經经 結结 "
    "統统 變变 寶宝 樣样 義义 習习 業业 從从 "
    "場场 聲声 萬万 與与 為为 無无 時时 東东 "
    "車车 見见 貝贝 資资 財财 質质 買买 賣卖 "
    "親亲 觀观 將将 導导 師师 類类 數数 據据 "
    "關关 開开 聽听 覺觉 進进 連连 運运 過过 "
    "違违 遠远 適适 選选 遺遗 鐵铁 長长 門门 "
    "間间 雲云 電电 馬马 駐驻 騎骑 驗验 魚鱼 "
    "鳴鸣 鳥鸟 點点 黨党 齊齐 龍龙 實实 對对 "
    "盡尽 個个 來来 兩两 內内 準准 幾几 動动 "
    "務务 勞劳 勢势 匯汇 區区 華华 協协 單单 "
    "嚴严 喪丧 團团 園园 書书 會会 機机 歡欢 "
    "歲岁 歸归 當当 極极 樓楼 標标 樹树 歷历 "
    "氣气 漢汉 滿满 愛爱 難难 順顺 頭头 題题 "
    "顏颜 風风 飛飞 飯饭 體体 發发 髮发 傢家 "
    "頻频 豐丰 醫医 舊旧 製制 後后 於于 裡里 "
    "裏里 麼么 網网 絡络 紅红 純纯 紙纸 級级 "
    "紀纪 約约 編编 縮缩 線线 餘余 號号 圓圆 "
    "夢梦 邊边 鐘钟 週周 國国 "
    "議议 論论 討讨 認认 試测 測测 檢检 設设 "
    "計计 術术 戰战 爭争 驚惊 關关 樂乐 銀银 "
    "線线 舉举 藝艺 審审 務务 寵宠 習习 異异 "
    "眾众 軍军 隊队 員员 產产 價价 賓宾 採采 "
    "訪访 報报 聞闻 廣广 視视 錄录 遊游 戲戏 "
    "劇剧 戶户 碼码 冊册 證证 隱隐 權权 規规 "
    "則则 組组 織织 領领 響响 敗败 繼继 續续 "
    "備备 劃划 決决 懷怀 擔担 緊紧 張张 興兴 "
    "奮奋 鍛锻 煉炼 訓训 練练 復复 績绩 麼么 "
    "種种 誰谁 嗎吗 億亿 麵面 雞鸡 難难 風风 "
    "飛飞 飯饭 飲饮 養养 藥药 營营 運运 遲迟 "
    "選选 遺遗 鄰邻 醫医 鑒鉴 銷销 鏡镜 戀恋 "
    "應应 動动 體体 髮发 驗验 駐驻 騎骑 豐丰 "
    "舊旧 製制 蓋盖 範范 約约 級约 "
    "講讲 畫画 現现 傳传 優优 競竞 訊讯 許许 "
    "詳详 課课 調调 談谈 處处 補补 裝装 轉转 "
    "輪轮 辦办 達达 針针 錢钱 館馆 獨独 溫温 "
    "燈灯 淺浅 貴贵 腦脑 臉脸 慶庆 隨随 擊击 "
    "積积 穩稳 縣县 購购 贏赢 橫横 "
)

_BAD = [t for t in _PAIRS.split() if len(t) != 2]   # 导入期自检
if _BAD:
    raise ValueError("han_convert: bad pair tokens %r" % (_BAD[:6],))
_TABLE = {t[0]: t[1] for t in _PAIRS.split()}

try:                                   # 可选依赖：装不上也不影响运行
    from zhconv import convert as _zc
except Exception:  # noqa: BLE001
    _zc = None

_convert = None          # 真正要用的转换函数（None = 用内置表）
_variant = ""            # zhconv 的简体变体名
zhconv_rejected = False  # 装了 zhconv 但自检不通过（见 _pick_worker）


def _pick_worker() -> None:
    """选词库之前必须自检 —— 这条是被真实事故逼出来的。

    zhconv 的 "cn" 别名**什么都不转、原样返回**（实测；简体变体叫 zh-hans），
    那就是"看起来在工作、其实静默无事"，正是最难查的一类故障。宁可退回内置表，
    也不能让一个不干活的后端占着位置。
    """
    global _convert, _variant, zhconv_rejected
    if _zc is None:
        return
    for name in ("zh-hans", "cn-hans", "cn"):
        try:
            if _zc("應該說的", name) == "应该说的":
                _convert, _variant = _zc, name
                return
        except Exception:  # noqa: BLE001
            continue
    zhconv_rejected = True


_pick_worker()


def backend_name() -> str:
    """当前用的是哪套（界面/快照里说清楚，别让人以为装了词库就在用）。"""
    if _convert is not None:
        return "zhconv 词库（%s）" % _variant
    if zhconv_rejected:
        return "内置对照表（%d 字；装了 zhconv 但自检不通过）" % len(_TABLE)
    return "内置对照表（%d 字）" % len(_TABLE)


def is_chinese_lang(lang):
    """只有中文（含粤语）允许转简体；日文/韩文同形字多，绝对不能动。"""
    code = (lang or "").split("-")[0].strip().lower()
    return code in ("zh", "yue", "cmn")


def has_kana(text: str) -> bool:
    """有没有假名（平/片/半角）—— 判断"这段其实是日文"最可靠的信号。"""
    return any("\u3040" <= ch <= "\u30ff" or "\uff66" <= ch <= "\uff9f" for ch in text)


def to_simplified_if_chinese(text: str) -> str:
    """只在"看起来是中文"时转简体，混了假名就原样返回。

    译文正常都是中文，但后端跑偏/失败回退时可能原样给回一句日文，那时逐字
    替换会把「時→时」这类改到日文里 —— 那是污染，不是修正。
    """
    if not text or has_kana(text):
        return text or ""
    return to_simplified(text)


def to_simplified(text: str) -> str:
    """把繁体中文写成简体。空值原样返回，任何异常都退回原文（不能因它丢字幕）。"""
    if not text:
        return text or ""
    try:
        if _convert is not None:
            return _convert(text, _variant) or text
        return "".join(_TABLE.get(ch, ch) for ch in text)
    except Exception:  # noqa: BLE001
        return text
