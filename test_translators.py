"""验证 translate_engine 里 TranslatorsProvider 的真实调用是否成功。"""

import sys

sys.path.insert(0, ".")

import socket

socket.setdefaulttimeout(25)

from translate_engine import TranslatorsProvider

t = TranslatorsProvider(target="zh-CN", engine="google", timeout=25)
for text in ("Hello, this is a real-time caption test.", "The weather is nice today."):
    try:
        out = t.translate(text)
        ok = out != text and any("\u4e00" <= c <= "\u9fff" for c in out)
        print(f"{text!r} -> {out!r}  {'OK 已译成中文' if ok else 'WARN 未翻译'}")
    except Exception as exc:  # noqa: BLE001
        print(f"{text!r} FAIL {type(exc).__name__}: {exc}")

print("last_error:", t.last_error or "(空)")