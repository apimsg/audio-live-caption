"""可插拔的实时翻译层。

后端可配置：
- "translators"  免费在线（translators 库，无需 key）
- "argos"        纯离线（argostranslate，需 pip install argostranslate 并下载模型）
- "baidu"        百度翻译开放平台（appid + secret）
- "openai"       OpenAI 兼容大模型（base_url + api_key + model，兼容 DeepSeek/通义/智谱/GPT 等）
- "none"         不翻译（原样返回）

对外接口统一为 ``Translator.translate(text) -> str``；失败时回退原文，
绝不阻断字幕显示。线程安全，内置 LRU 缓存避免重复翻译。
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import random
import threading
import time
from collections import OrderedDict
from typing import Callable

from han_convert import to_simplified_if_chinese


class LRUCache:
    def __init__(self, maxsize: int = 512) -> None:
        self._data: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


class Translator:
    """带缓存与超时的翻译器基类。子类实现 `_do_translate`。"""

    def __init__(self, *, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self._cache = LRUCache()
        self._lock = threading.Lock()
        self.last_error = ""

    def translate(self, text: str, context: str = "") -> str:
        """翻译一句；context 是上一句原文（习语/断句的理解靠它）。

        缓存键含上下文：不同上下文分别缓存。
        """
        text = (text or "").strip()
        if not text:
            return text
        key = text if not context else f"‹{context}›\u0000{text}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            result = self._translate_locked(text, context)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            result = text  # 失败回退原文
        self._cache.put(key, result)
        return result

    def _translate_locked(self, text: str, context: str = "") -> str:
        with self._lock:
            return self._do_translate(text, context)

    def _do_translate(self, text: str, context: str = "") -> str:  # pragma: no cover
        raise NotImplementedError


class NoneTranslator(Translator):
    """关闭翻译，原样返回。"""

    def _do_translate(self, text: str, context: str = "") -> str:
        return text


class TranslatorsProvider(Translator):
    """免费在线：translators 库（google/bing/…），无需 key。

    自动在多个服务商之间回退，任一返回中文即用。
    """

    _FALLBACKS = ("bing", "google", "tencent", "baidu", "alibaba", "deepl", "yandex")

    def __init__(self, *, target: str = "zh-CN", engine: str = "bing", **kw) -> None:
        super().__init__(**kw)
        self.target = target
        self.engine = engine

    def _do_translate(self, text: str, context: str = "") -> str:
        import translators as ts

        # 让首选服务排在最前，其余作回退
        order = [self.engine] + [e for e in self._FALLBACKS if e != self.engine]
        pool = getattr(ts, "translators_pool", set())
        if pool:
            order = [e for e in order if e in pool] or list(pool)[:4]

        last_err = "无可用服务商"
        for svc in order:
            try:
                out = ts.translate_text(
                    text,
                    translator=svc,
                    from_language="auto",
                    to_language=self.target,
                    timeout=self.timeout,
                    if_ignore_limit_of_length=True,
                    if_print_warning=False,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{svc}:{type(exc).__name__}:{exc}"
                continue
            out = (out or "").strip()
            if out:
                return out
        self.last_error = last_err
        return text


class ArgosProvider(Translator):
    """纯离线：argostranslate（需安装并下载 zh 模型）。"""

    def __init__(self, *, target: str = "zh", **kw) -> None:
        super().__init__(**kw)
        self.target = target
        self._pair = None

    def _load_pair(self):
        import argostranslate.package  # type: ignore
        import argostranslate.translate  # type: ignore

        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        if self.target == "zh":
            pkgs = [p for p in available if p.from_code == "en" and p.to_code == "zh"]
        else:
            pkgs = [p for p in available if p.to_code == self.target]
        if not pkgs:
            raise RuntimeError(
                "argos 未找到目标语言模型，请先运行安装下载（见 README）"
            )
        argostranslate.package.install_from_path(pkgs[0].download())
        installed = argostranslate.translate.get_installed_languages()
        from_lang = next(l for l in installed if l.code == pkgs[0].from_code)
        to_lang = next(l for l in installed if l.code == pkgs[0].to_code)
        self._pair = from_lang.get_translation(to_lang)

    def _do_translate(self, text: str, context: str = "") -> str:
        if self._pair is None:
            self._load_pair()
        return self._pair.translate(text)


class BaiduProvider(Translator):
    """百度翻译开放平台（https://fanyi-api.baidu.com/api/trans/vip/translate）。"""

    def __init__(self, *, appid: str, secret: str, target: str = "zh", **kw) -> None:
        super().__init__(**kw)
        self.appid = appid
        self.secret = secret
        self.target = target

    def _do_translate(self, text: str, context: str = "") -> str:
        import requests

        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5(f"{self.appid}{text}{salt}{self.secret}".encode()).hexdigest()
        r = requests.post(
            "https://fanyi-api.baidu.com/api/trans/vip/translate",
            data={
                "q": text,
                "from": "auto",
                "to": self.target,
                "appid": self.appid,
                "salt": salt,
                "sign": sign,
            },
            timeout=self.timeout,
        )
        payload = r.json()
        if "error_code" in payload and payload["error_code"] != "52000":
            raise RuntimeError(f"百度返回错误 {payload.get('error_code')}: {payload.get('error_msg')}")
        dst = payload.get("trans_result") or []
        return " ".join(item.get("dst", "") for item in dst).strip() or text


class OpenAICompatProvider(Translator):
    """OpenAI 兼容大模型：兼容 DeepSeek / 通义 / 智谱 / GPT 等。

    base_url 示例：https://api.deepseek.com/v1 、 https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        target: str = "中文（简体）",
        system_prompt: str | None = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.target = target
        self.system_prompt = system_prompt or (
            f"你是专业字幕翻译。请把用户给出的字幕句翻译成{target}；"
            "若消息里有\"上文\"，务必结合语境理解（例如英文习语 stiff upper lip "
            "意为\"强忍悲痛不动声色\"，不能直译成身体部位）。"
            "只输出译文，不要解释，不要加引号，若已是目标语言则原样返回。"
        )

    def _do_translate(self, text: str, context: str = "") -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        user = text if not context else f"上文：{context}\n\n翻译这一句：{text}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        out = (
            (payload.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return out or text


PROVIDERS = ("none", "translators", "argos", "baidu", "openai")


def make_translator(
    provider: str,
    *,
    target: str = "zh-CN",
    timeout: float = 20.0,
    appid: str = "",
    secret: str = "",
    base_url: str = "",
    api_key: str = "",
    model: str = "deepseek-chat",
) -> Translator:
    """根据配置构造翻译器。任何未知/缺失配置都退化为不翻译。"""
    provider = (provider or "none").strip().lower()
    if provider in ("", "none", "off", "false"):
        return NoneTranslator()
    if provider == "translators":
        return TranslatorsProvider(target=target, timeout=timeout)
    if provider == "argos":
        return ArgosProvider(target=target.replace("-", "").replace("CN", ""), timeout=timeout)
    if provider == "baidu":
        return BaiduProvider(appid=appid, secret=secret, target=target, timeout=timeout)
    if provider == "openai":
        if not api_key:
            return NoneTranslator()
        return OpenAICompatProvider(
            base_url=base_url or "https://api.deepseek.com/v1",
            api_key=api_key,
            model=model,
            target=target,
            timeout=timeout,
        )
    return NoneTranslator()


def run_blocking(translator: Translator, text: str) -> str:
    """供子线程/UI 调用：确保翻译结果、不抛异常。"""
    try:
        return translator.translate(text)
    except Exception:  # noqa: BLE001
        return text

# 本项目目标语言永远是中文。原文已经是中文时再"翻一次中文"，机翻只会把标点弄丢、
# 语序弄乱（"中文到中文错乱"就是它），还白等一个网络往返。
_KANA_RE = re.compile("[\u3040-\u30ff\u31f0-\u31ff]")
_HANZI_RE = re.compile("[\u4e00-\u9fff\uf900-\ufaff\u3400-\u4dbf]")


def is_chinese_text(text: str) -> bool:
    """汉字为主且没有假名（假名一出现就说明是日语）。"""
    if not text:
        return False
    if _KANA_RE.search(text):
        return False
    return len(_HANZI_RE.findall(text)) >= 2


def needs_translation(text: str, lang: str | None = None) -> bool:
    """这句该不该送翻译。lang 是识别侧给的语种码（zh/en/ja/yue…），None = 未知。"""
    code = (lang or "").split("-")[0].strip().lower()
    if code == "zh":
        return False                 # 简体中文字幕：再翻只会翻坏
    if code:
        return True                  # 明确别的语种（含粤语：汉字写法但要转普通话）
    return not is_chinese_text(text)  # 语种未知：退回文字脚本判断


class TranslatorWorker:
    """后台翻译线程：提交 Caption → 输出 (caption, translated)，绝不阻塞界面。

    自动把**上一句原文**作为上下文传给翻译器（习语/断句的关键）；
    翻译失败一律回退原文。只用到 caption 的 .text/.index，不做类型硬绑定。
    """

    def __init__(self, translator) -> None:
        self.translator = translator
        self._in: queue.Queue = queue.Queue()
        self.out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._prev = ""
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, caption, enabled: bool, lang: str | None = None) -> None:
        ctx, self._prev = self._prev, caption.text
        self._in.put((caption, enabled, ctx, lang))

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                caption, enabled, ctx, lang = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            # 不翻时的哨兵就是原文本身：界面/悬浮窗看到 trans == src 只出一行
            translated = caption.text
            try:
                # needs_translation 也放进 try：这条线程死了译文就永远不再更新
                if enabled and needs_translation(caption.text, lang):
                    translated = self.translator.translate(caption.text, context=ctx) or caption.text
                    # 译文是给人看的中文，云端/本地模型常吐繁体 → 统一简体。
                    # 只在"真的翻了"之后转：不翻时 translated 就是原文（可能是日文），
                    # 拿去转简体会把日文用字改坏。
                    translated = to_simplified_if_chinese(translated)
            except Exception:  # noqa: BLE001
                translated = caption.text
            self.out.put((caption, translated))