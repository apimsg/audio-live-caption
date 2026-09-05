"""桌面悬浮实时字幕窗（置顶 / 镂空透明 / 可拖动 / 字体特效 / 双语绑定）。

本窗口**只支持跟随模式**：识别与翻译由浏览器控制中心（app.py）里唯一的引擎
完成，本窗口通过本机 HTTP 拉取字幕渲染（rev 增量 + epoch 重置协议）——
两个界面共用同一份音频与模型，双语天然同步，也不会互相抢 CPU。

平时不需要手动启动它：控制中心左侧勾选「启用桌面悬浮字幕窗」会自动带起。
手动连接某个端口（端口显示在页面左侧）：
    python desktop_caption.py --follow http://127.0.0.1:PORT

窗口操作：左键拖动 · 右键菜单（字号/特效/颜色/置顶/条数/退出）· Esc 退出。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict

from caption_engine import list_loopback_devices
from caption_service import CaptionClient

# Windows 上用该颜色做"镂空"透明：窗口背景完全透明，只留下字幕文字。
TRANSPARENT_KEY = "#010203"
HINT_COLOR = "#FFB347"        # 黄色提醒：画在字幕正上方，与字幕同轴居中
POS_FILE_NAME = "overlay_pos.json"   # 记住上次显示位置（拖过就写在这里）
COLORS = {"白": "#FFFFFF", "黄": "#FFE81A", "青": "#38E1FF", "绿": "#7CFC8A", "粉": "#FF9AD1"}


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="桌面悬浮实时字幕窗（跟随浏览器控制中心，一份引擎两个界面）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--follow", default="", metavar="URL",
                   help="跟随控制中心引擎拉取字幕，如 http://127.0.0.1:52634"
                        "（端口显示在控制中心页面左侧）")
    # 外观（控制中心里的设置会自动转成这些参数）
    p.add_argument("--font", default="黑体", help="字体名")
    p.add_argument("--font-size", type=int, default=20, help="字号")
    p.add_argument("--color", default="#FFFFFF", help="原文颜色")
    p.add_argument("--trans-color", default="#FFE81A", help="译文颜色")
    p.add_argument("--effect", choices=["shadow", "outline", "none"], default="shadow",
                   help="字体特效：shadow=单向淡影 outline=同色描边 none=无")
    p.add_argument("--shadow-color", default="#404040", help="淡影颜色")
    p.add_argument("--entries", type=int, default=2, help="同屏显示几条字幕（双语绑定计算）")
    p.add_argument("--width", type=int, default=1100, help="字幕区宽度像素")
    p.add_argument("--y-offset", type=int, default=140, help="距屏幕底部像素")
    p.add_argument("--no-topmost", action="store_true", help="启动时不置顶")
    # 辅助
    p.add_argument("--check", action="store_true", help="只打印采集设备后退出")
    return p


def make_config(args: argparse.Namespace) -> dict:
    """外观配置（跟随模式下只有渲染参数有意义）。"""
    return {
        "font": args.font,
        "font_size": max(10, int(args.font_size)),
        "color": args.color,
        "trans_color": args.trans_color,
        "effect": args.effect,
        "shadow_color": None if args.shadow_color.lower() in ("none", "") else args.shadow_color,
        "entries": max(1, int(args.entries)),
        "width": int(args.width),
        "y_offset": int(args.y_offset),
        "topmost": not args.no_topmost,
    }


# ------------------------------------------------- 字幕板（双语绑定的核心）
class EntryBoard:
    """按"条"管理字幕：一条 = 原文 + 可选译文，绑定渲染，永不拆散。"""

    def __init__(self, max_entries: int = 2) -> None:
        self.max_entries = max_entries
        self._data: "OrderedDict[int, dict]" = OrderedDict()

    def put_src(self, index: int, src: str) -> None:
        self._data[index] = {"src": src, "trans": ""}
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def put_trans(self, index: int, trans: str) -> None:
        item = self._data.get(index)
        if item is not None and trans and trans != item["src"]:
            item["trans"] = trans

    def has(self, index: int) -> bool:
        return index in self._data

    def upsert(self, index: int, src: str | None = None,
               trans: str | None = None) -> None:
        """按 rev 增量协议落条目：新条创建，已存在的原位更新（译文回填）。"""
        item = self._data.get(index)
        if item is None:
            if not src:
                return
            self._data[index] = {"src": src, "trans": ""}
            item = self._data[index]
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
        elif src and item["src"] != src:
            item["src"] = src
        # trans="" 是服务端在说"原文变了、旧译文作废"，必须照收
        if trans is not None:
            item["trans"] = trans

    def clear(self) -> None:
        self._data = OrderedDict()

    def newest_index(self) -> int:
        return max(self._data) if self._data else -1

    def snapshot(self) -> list[tuple[int, dict]]:
        return [(i, dict(v)) for i, v in self._data.items()]


def wrap_lines(text: str, max_px: int, measure) -> list[str]:
    """按像素宽度逐字折行；measure(text)->像素宽。纯函数，可单测。"""
    out: list[str] = []
    line = ""
    for ch in text:
        if ch == "\n":
            out.append(line)
            line = ""
            continue
        probe = line + ch
        if line and measure(probe) > max_px:
            out.append(line)
            line = ch
        else:
            line = probe
    out.append(line)
    return [s for s in out if s]


def board_to_lines(board: EntryBoard, max_px: int, measure) -> list[tuple[str, str]]:
    """把字幕板转成 [(kind, text)]，kind ∈ src/tr；同一条的原文译文相邻。"""
    lines: list[tuple[str, str]] = []
    for _i, item in board.snapshot():
        for s in wrap_lines(item["src"], max_px, measure) or [""]:
            lines.append(("src", s))
        if item["trans"]:
            for s in wrap_lines(item["trans"], max_px, measure) or [""]:
                lines.append(("tr", s))
    return lines


# ------------------------------------------------------------- 窗口外壳 UI
# ---- 布局与位置记忆（纯函数，方便无 GUI 测试）---------------------------
def split_hint_lines(lines: list[tuple[str, str]]) -> tuple[list, list]:
    """把提醒行摘出来：提醒永远排在字幕**正上方**，不参与条数轮换。"""
    hints = [(k, t) for k, t in lines if k == "hint"]
    rows = [(k, t) for k, t in lines if k != "hint"]
    return hints, rows


def load_pos(path: str) -> tuple[int, int] | None:
    """上次显示位置（没记过/文件坏了都当没记过，绝不能因此开不了窗）。"""
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return int(d["x"]), int(d["y"])
    except Exception:  # noqa: BLE001
        return None


def save_pos(path: str, x: int, y: int) -> None:
    try:
        import json
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"x": int(x), "y": int(y)}, fh)
    except Exception:  # noqa: BLE001
        pass


def resolve_origin(stored, width: int, height: int,
                   sw: int, sh: int, y_offset: int) -> tuple[int, int]:
    """有记忆且还在屏内 → 回上次位置；否则水平居中贴底。

    "还在屏内"这条必须判：拔掉一台显示器后记忆坐标可能落在已经不存在的屏幕上，
    窗口会永远看不见 —— 那种情况自动退回默认位置。
    """
    if stored is not None:
        x, y = stored
        if -sw <= x <= sw - 40 and -sh <= y <= sh - 40:
            return int(x), int(y)
    return (sw - width) // 2, sh - height - y_offset


class OverlayUI:
    """置顶镂空透明、无边框可拖动的字幕窗口；渲染 + 右键菜单 + 拖拽。"""

    def __init__(self, cfg: dict) -> None:
        import tkinter as tk
        import tkinter.font as tkfont

        self._tk = tk
        self._tkfont = tkfont
        self.cfg = cfg
        self.font_size = cfg["font_size"]
        self.color = cfg["color"]
        self.trans_color = cfg["trans_color"]
        self.effect = cfg["effect"]
        self.shadow_color = cfg["shadow_color"]
        self.topmost = cfg["topmost"]
        self.lines: list[tuple[str, str]] = []
        self.alive = True
        self._drag = None
        self._measure_cache: dict = {}

        root = self.root = tk.Tk()
        root.title("实时字幕")
        root.overrideredirect(True)
        root.attributes("-topmost", self.topmost)
        root.configure(bg=TRANSPARENT_KEY)
        try:
            root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            pass

        self.width = cfg["width"]
        height = 40 + cfg["entries"] * 2 * (self.font_size + 12)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.height = height
        # 位置：每次启动都回到上次拖到的地方（没有记忆或已不在屏内才用默认）
        from pathlib import Path as _P
        self.pos_file = str(_P(__file__).with_name(POS_FILE_NAME))
        x, y = resolve_origin(load_pos(self.pos_file), self.width, height,
                              sw, sh, cfg["y_offset"])
        root.geometry(f"{self.width}x{height}+{x}+{y}")

        self.canvas = tk.Canvas(root, width=self.width, height=height,
                                bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right)
        root.bind("<Escape>", lambda _e: self.quit())
        self._build_menu()

    # ---- 字体与测量 -----------------------------------------------------
    def _font_tuple(self, size: int):
        return (self.cfg["font"], size, "bold")

    def measure(self, text: str) -> float:
        key = self.font_size
        f = self._measure_cache.get(key)
        if f is None:
            f = self._measure_cache[key] = self._tkfont.Font(font=self._font_tuple(key))
        return f.measure(text)

    # ---- 渲染 -----------------------------------------------------------
    def set_lines(self, lines: list[tuple[str, str]]) -> None:
        self.lines = lines
        self.redraw()

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        size = self.font_size
        font = self._font_tuple(size)
        step = size + 12
        cx = self.width // 2
        y = size // 2 + 8
        hints, rows = split_hint_lines(self.lines)
        # 黄色提醒画在字幕正上方（与字幕同轴居中），并把它的高度让出来：
        # 否则提醒出现/消失时字幕块会上下跳，看着像窗口在抖。
        if hints:
            hsize = max(12, size // 2)
            hstep = hsize + 7
            for _kind, htext in hints:
                c.create_text(cx, y + hsize // 2, text=htext,
                              font=(self.cfg["font"], hsize), fill=HINT_COLOR)
                y += hstep
        for kind, text in rows:
            color = self.color if kind == "src" else self.trans_color
            if self.effect == "shadow" and self.shadow_color:
                c.create_text(cx + 2, y + 2, text=text, font=font, fill=self.shadow_color)
            elif self.effect == "outline":
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.create_text(cx + dx, y + dy, text=text, font=font, fill=color)
            c.create_text(cx, y, text=text, font=font, fill=color)
            y += step
        new_h = max(48, y + 6)
        if new_h != self.height:
            self.height = new_h
            c.configure(height=new_h)
            self.root.geometry(f"{self.width}x{new_h}+{self.root.winfo_x()}+{self.root.winfo_y()}")

    # ---- 拖拽 / 菜单 -----------------------------------------------------
    def _on_press(self, e):
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _on_drag(self, e):
        if self._drag:
            dx, dy = self._drag
            self.root.geometry(f"+{e.x_root - dx}+{e.y_root - dy}")

    def _on_release(self, _e):
        self._drag = None
        # 松手就记一次位置：下次启动回到这里（"记住上次显示位置"）
        save_pos(self.pos_file, self.root.winfo_x(), self.root.winfo_y())

    def _build_menu(self):
        tk = self._tk
        menu = self.menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="字号 +", command=lambda: self._bump(2))
        menu.add_command(label="字号 -", command=lambda: self._bump(-2))
        menu.add_command(label="特效：阴影→描边→无 循环", command=self._cycle_effect)
        for name, hexv in COLORS.items():
            menu.add_command(label=f"{name}字", command=lambda h=hexv: self._set_color(h))
        menu.add_command(label="切换置顶", command=self._toggle_topmost)
        menu.add_command(label="回到默认位置（清除记忆）", command=self._reset_pos)
        menu.add_command(label="显示上一条", command=self._show_more)
        menu.add_command(label="显示条数 -", command=self._show_less)
        menu.add_separator()
        menu.add_command(label="退出 (或按 Esc)", command=lambda: self.quit())

    def _on_right(self, e):
        try:
            self.menu.tk_popup(e.x_root, e.y_root)
        finally:
            self.menu.grab_release()

    def _bump(self, delta: int):
        self.font_size = max(12, min(96, self.font_size + delta))
        self.redraw()

    def _cycle_effect(self):
        order = ["shadow", "outline", "none"]
        self.effect = order[(order.index(self.effect) + 1) % 3]
        self.redraw()

    def _set_color(self, hexv: str):
        self.color = hexv
        self.redraw()

    def _reset_pos(self):
        """清除位置记忆并回到默认（水平居中贴底）。"""
        try:
            import os
            if self.pos_file and os.path.isfile(self.pos_file):
                os.remove(self.pos_file)
        except Exception:  # noqa: BLE001
            pass
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = resolve_origin(None, self.width, self.height, sw, sh,
                              self.cfg["y_offset"])
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def _toggle_topmost(self):
        self.topmost = not self.topmost
        self.root.attributes("-topmost", self.topmost)

    def _show_more(self):
        self.cfg["entries"] = min(6, self.cfg["entries"] + 1)
        if self.on_entries_changed:
            self.on_entries_changed(self.cfg["entries"])
        self.redraw()

    def _show_less(self):
        self.cfg["entries"] = max(1, self.cfg["entries"] - 1)
        if self.on_entries_changed:
            self.on_entries_changed(self.cfg["entries"])
        self.redraw()

    on_entries_changed = None  # 驱动层可注入回调（重建 EntryBoard 容量）

    def quit(self):
        self.alive = False
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass

    # ---- 周期驱动 --------------------------------------------------------
    def schedule(self, step, interval_ms: int = 200) -> None:
        """安全循环：step 内任何异常都不会杀死刷新，错误直接画到窗口上。"""
        def tick():
            if not self.alive:
                return
            try:
                step()
            except Exception as exc:  # noqa: BLE001
                print("界面错误：", repr(exc), flush=True)
                self.lines = [("hint", f"界面错误：{exc!r}")]
                try:
                    self.redraw()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self.root.after(interval_ms, tick)

        self.root.after(50, tick)

    def run(self) -> None:
        self.root.mainloop()


# ------------------------------------------------------------- 跟随模式
def run_follow(cfg: dict, url: str) -> int:
    ui = OverlayUI(cfg)
    board = EntryBoard(cfg["entries"])

    def on_entries(n: int):
        keep = dict(board._data)
        board.max_entries = n
        board._data = OrderedDict((k, v) for k, v in keep.items())
        while len(board._data) > n:
            board._data.popitem(last=False)

    ui.on_entries_changed = on_entries
    client = CaptionClient(url)
    since_rev = -1
    epoch = None
    misses = 0
    ui.set_lines([("hint", f"等待浏览器版字幕…（{url}）")])

    def step():
        nonlocal since_rev, misses, epoch
        try:
            resp = client.fetch(since_rev)
            misses = 0
        except Exception:
            misses += 1
            if misses == 1:
                ui.set_lines([("hint", "连接不上浏览器版，重试中…")])
            if misses >= 15:
                ui.set_lines([("hint", "浏览器版已关闭，3 秒后退出")])
                ui.root.after(3000, ui.quit)
            return
        st = resp.get("status", {})
        ep = st.get("epoch")
        if epoch is None:
            epoch = ep
        elif ep is not None and ep != epoch:
            # 浏览器版开始新一轮识别：清空旧字幕，下次全量重拉
            epoch = ep
            board.clear()
            since_rev = -1
            return
        for it in resp.get("items", []):
            board.upsert(int(it.get("i", -1)),
                         src=it.get("src") or None,
                         trans=it.get("trans"))   # 空串=作废旧译文（原文被修订）
            rev = int(it.get("rev", 0))
            if rev > since_rev:
                since_rev = rev
        lines = board_to_lines(board, ui.width - 20, ui.measure)
        hint = st.get("hint") or st.get("error") or ""
        if hint:
            lines.insert(0, ("hint", hint))
        notice = st.get("notice") or ""
        if notice:
            lines.insert(0, ("hint", notice))
        if not lines:
            lines = [("hint", "已连接：等待识别结果…（先播放视频/说话）")]
        ui.set_lines(lines)

    ui.schedule(step, 400)
    ui.run()
    return 0


def run_check() -> int:
    print("=== 采集设备（WASAPI 回环，用于「系统声音」字幕）===", flush=True)
    lb = list_loopback_devices()
    if not lb:
        print("  未发现 WASAPI 回环设备（检查 PyAudioWPatch / 系统输出设备）", flush=True)
    for idx, name in lb.items():
        print(f"  [{idx}] {name}", flush=True)
    print("=== 麦克风输入设备 ===", flush=True)
    try:
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                print(f"  [{i}] {d['name']}  sr={d['default_samplerate']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  读取失败：{exc}", flush=True)
    print("（识别与翻译请在浏览器控制中心里启动；桌面悬浮窗跟随它，不再独立运行）",
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return run_check()
    cfg = make_config(args)
    if args.follow:
        return run_follow(cfg, args.follow)
    print("桌面悬浮窗现只支持跟随模式：请先在浏览器控制中心启动，\n"
          "  页面左侧「字幕服务」处会显示形如 http://127.0.0.1:PORT 的地址，\n"
          f"  再用 --follow <该地址> 启动本窗口（或直接在网页勾选启用悬浮窗）。\n"
          f"当前收到参数：follow={args.follow!r}。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
