r"""打包分发：只带源码，绝不带环境/模型/录音/本机状态。

    .venv\Scripts\python pack_release.py              # 生成 dist\audio-live-caption-<日期>.zip
    .venv\Scripts\python pack_release.py --list       # 只列出会带哪些文件，不写包
    .venv\Scripts\python pack_release.py --name demo  # 自定义包名

为什么用白名单而不是排除表：排除表会漏 —— 谁往目录里丢个大文件（模型权重、录音、
venv）就会被"顺手打包"。白名单是反过来：不认识的一律不进包，而且单个文件超过
_MAX_BYTES 直接报错停下，不是悄悄收进去。

对方拿到 zip：解压成一个文件夹 → 双击 run.bat（首次自动建 .venv 并装依赖，机器上
要有 Python 3.10~3.12）→ 模型权重在它首次使用时自动下载到它自己的 models/。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
ROOT_NAME = "audio-live-caption"     # 解压出来的顶层文件夹名
_MAX_BYTES = 1_000_000               # 单文件上限：超了就是有东西混进来

# 子目录一律不进包（本项目源码全平铺在根目录；模型/录音/venv 都在子目录里）
_EXCLUDE_DIRS = {".venv", "venv", "env", "models", "records", ".tmp", "tmp",
                 "__pycache__", "dist", ".git", ".idea", ".vscode", "node_modules"}
# 文件名黑名单（本机状态 / 隐私 / 大文件 / 构建垃圾）—— 白名单之外的第二道闸
_EXCLUDE_RE = re.compile(r"(^overlay_pos\.json$|^pip-|^t_[0-9]|^tmp|^\.|"
                         r"\.(wav|mp3|m4a|mp4|mkv|avi|zip|7z|pt|bin|gguf|ct2|"
                         r"safetensors|log|pyc|pyo|lock|db)$|^\.env$)", re.I)
# 少任何一个就不是一个能跑的包
_REQUIRED = ["run.bat", "requirements.txt", "README.md", ".gitignore",
             "launcher.py", "app.py", "caption_engine.py", "whisper_backend.py",
             "caption_service.py", "desktop_caption.py", "translate_engine.py",
             "media_extract.py", "polish.py", "han_convert.py", "pack_release.py"]
# 认识的后缀/文件名才进包
_OK_SUFFIX = {".py", ".md", ".txt", ".bat"}
_OK_NAMES = {".gitignore", "LICENSE", "NOTES.txt"}


def collect(root: Path) -> list:
    out = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir():
            continue                                  # 子目录一律不进
        if entry.name in _OK_NAMES:                   # .gitignore 等显式白名单
            out.append(entry)
            continue
        if entry.suffix.lower() not in _OK_SUFFIX:
            continue                                  # 不认识的类型不进
        if _EXCLUDE_RE.search(entry.name):
            continue                                  # 命中黑名单不进
        out.append(entry)
    return out


def check(files: list) -> list:
    """返回问题清单（空 = 允许打包）。"""
    problems = []
    names = {f.name for f in files}
    for need in _REQUIRED:
        if need not in names:
            problems.append("缺少必需文件：%s" % need)
    for f in files:
        size = f.stat().st_size
        if size > _MAX_BYTES:
            problems.append("%s 有 %.1f MB，超过单文件上限 %.1f MB —— 这种不该进包"
                            % (f.name, size / 1048576.0, _MAX_BYTES / 1048576.0))
        elif size == 0:
            problems.append("%s 是空文件" % f.name)
        if f.suffix.lower() == ".bat":
            raw = f.read_bytes()
            if raw.count(b"\r") == 0:
                problems.append("%s 没有 CRLF 换行，cmd 可能解析不了" % f.name)
            if any(b > 127 for b in raw):
                problems.append("%s 含非 ASCII 字节，cmd 代码页会乱码" % f.name)
    return problems


def build(name: str, dry: bool = False) -> int:
    files = collect(HERE)
    problems = check(files)
    print("进包文件 %d 个（源码平铺在根目录，子目录一律不带）：" % len(files))
    total = 0
    for f in files:
        kb = f.stat().st_size / 1024.0
        total += f.stat().st_size
        print("    %-28s %7.1f KB" % (f.name, kb))
    print("    %-28s %7.1f KB" % ("合计（未压缩）", total / 1024.0))
    # 顺手提醒：这些目录存在但不进包（对方首次运行会自己生成）
    skipped = [d for d in sorted(_EXCLUDE_DIRS) if (HERE / d).is_dir()]
    if skipped:
        print("本地存在但不进包：" + "  ".join(skipped))
    if problems:
        print("")
        for p in problems:
            print("  [x] " + p)
        print("有上面这些问题就别发出去 —— 修掉再打包。")
        return 1
    if dry:
        print("(--list：只看清单，没写包)")
        return 0
    dist = HERE / "dist"
    dist.mkdir(exist_ok=True)
    zp = dist / (name + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(f, arcname="%s/%s" % (ROOT_NAME, f.name))
    raw = zp.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    print("")
    print("已生成 %s  (%.2f MB)" % (zp, len(raw) / 1048576.0))
    print("SHA256 %s" % sha)
    print("发出去时把上面这行 SHA256 一起给对方，让他校验下载没被改/没下坏。")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="打包分发（只带源码，不带环境/模型/录音）")
    p.add_argument("--name", default="", help="包名（默认 audio-live-caption-YYYYMMDD）")
    p.add_argument("--list", action="store_true", help="只打印会进包的文件，不写包")
    args = p.parse_args(argv)
    name = args.name or "%s-%s" % (ROOT_NAME, datetime.now().strftime("%Y%m%d"))
    if not name.startswith(ROOT_NAME):
        name = ROOT_NAME + "-" + name
    return build(name, dry=args.list)


if __name__ == "__main__":
    raise SystemExit(main())
