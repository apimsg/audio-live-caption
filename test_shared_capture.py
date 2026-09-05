"""验证：桌面版与浏览器版（两个进程）能否同时打开同一个 WASAPI 回环设备。

父进程再拉起一个子进程，两者同时打开同一个回环端点 2 秒：
两个都打开成功 => 共享模式允许多读取者，两个界面可共存。
"""
import sys
import time


def open_stream(seconds: float, tag: str) -> int:
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        dev = pa.get_default_wasapi_loopback()
        idx, rate = int(dev["index"]), int(dev["defaultSampleRate"] or 48_000)
        ch = int(dev["maxInputChannels"] or 1)
        state = {"blocks": 0}

        def cb(in_data, frame_count, time_info, status):
            state["blocks"] += 1
            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
            input_device_index=idx, frames_per_buffer=int(rate * 0.1),
            stream_callback=cb,
        )
        stream.start_stream()
        time.sleep(seconds)
        active = bool(stream.is_active())
        stream.stop_stream()
        stream.close()
        print(f"[{tag}] 设备 idx={idx} 打开成功 active={active} 回调块={state['blocks']}",
              flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] 打开失败: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        pa.terminate()


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(open_stream(2.5, "child 浏览器版模拟"))

    import subprocess
    from pathlib import Path

    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).name), "--child"],
        stdin=subprocess.DEVNULL,   # inherit 输出，避免沙箱管道限制
    )
    time.sleep(0.6)                 # 确保子进程已经先占住设备
    rc_parent = open_stream(2.0, "parent 桌面版模拟")
    child.wait()
    print("COEXIST OK" if rc_parent == 0 and child.returncode == 0
          else "COEXIST FAIL", flush=True)
    sys.exit(0 if rc_parent == 0 and child.returncode == 0 else 1)