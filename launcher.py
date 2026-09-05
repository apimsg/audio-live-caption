"""统一启动器：浏览器控制中心是唯一主程序（桌面悬浮窗由它一键唤起/跟随）。

用法：
    python launcher.py        # 交互式菜单
    python launcher.py web    # 直接进浏览器控制中心
    python launcher.py check  # 只自检设备与依赖

桌面悬浮窗独立运行模式已移除：悬浮窗与网页共用同一份引擎/音频/模型，
在控制中心左侧勾选「启用桌面悬浮字幕窗」即可，外观也在那里调。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent


def ask(text: str, default: str = "") -> str:
    tip = f"[{default}] " if default else ""
    try:
        raw = input(f"{text} {tip}").strip()
    except EOFError:
        return default
    return raw or default


def run_web() -> int:
    import os
    import subprocess

    exe = Path(sys.executable)
    stream = exe.parent / ("streamlit.exe" if os.name == "nt" else "streamlit")
    target = str(HERE / "app.py")
    if stream.exists():
        cmd = [str(stream), "run", target]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", target]
    print("启动浏览器控制中心：", " ".join(cmd))
    print("打开后默认 http://localhost:8501（Ctrl+C 退出）；"
          "左侧勾选「启用桌面悬浮字幕窗」可同时出桌面双语字幕。")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def run_check() -> int:
    import desktop_caption

    return desktop_caption.main(["--check"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="音频实时转字幕 统一启动器")
    p.add_argument("what", nargs="?", choices=["web", "check", "desktop"],
                   default=None, help="快捷启动：跳过菜单直接进 控制中心/自检")
    p.add_argument("--check", action="store_true", help="只自检")
    args = p.parse_args(argv)

    if args.what == "check" or args.check:
        return run_check()
    if args.what == "web":
        return run_web()
    if args.what == "desktop":
        print("桌面悬浮窗独立运行模式已移除：悬浮窗现在跟随浏览器控制中心"
              "（共用一份引擎与音频）。正在启动控制中心…\n")
        return run_web()

    print("\n==== 音频实时转字幕 ====")
    print("  1. 浏览器控制中心（推荐：一份引擎同时喂网页 + 桌面悬浮窗）")
    print("  2. 自检设备与依赖")
    print("  0. 退出")
    choice = ask("请选择", "1")
    if choice == "2":
        return run_check()
    if choice == "0":
        return 0
    return run_web()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
