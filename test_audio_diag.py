"""诊断音频设备：列出所有输入设备与主机 API，并逐个尝试打开。"""
import sounddevice as sd

print("=== PortAudio / 默认设备 ===")
try:
    print("default device:", sd.default.device)
except Exception as e:  # noqa: BLE001
    print("default.device 读取失败:", e)

print("\n=== 可输入设备 ===")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"[{i}] name={d['name']!r} hostapi={d['hostapi']} sr={d['default_samplerate']} chan={d['max_input_channels']}")

print("\n=== 主机 API ===")
for i, h in enumerate(sd.query_hostapis()):
    print(i, repr(h["name"]), "devices=", h["devices"])

print("\n=== 逐个尝试打开输入流 ===")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] <= 0:
        continue
    sr = int(d["default_samplerate"]) or 44100
    try:
        with sd.InputStream(device=i, samplerate=sr, channels=1, dtype="float32", blocksize=0, latency="low", extra_settings=sd.WasapiSettings()):
            print(f"[{i}] OK (WASAPI) sr={sr}")
    except Exception as e:  # noqa: BLE001
        print(f"[{i}] WASAPI 失败: {type(e).__name__}: {e}")
    try:
        with sd.InputStream(device=i, samplerate=sr, channels=1, dtype="float32", blocksize=0, latency="low"):
            print(f"[{i}] OK (默认 API) sr={sr}")
    except Exception as e:  # noqa: BLE001
        print(f"[{i}] 默认API 失败: {type(e).__name__}: {e}")

print("\n=== Windows 音频服务 ===")
try:
    import subprocess
    out = subprocess.run(["sc", "query", "Audiosrv"], capture_output=True, text=True, timeout=10)
    print("\n".join(l for l in out.stdout.splitlines() if "STATE" in l or "SERVICE_NAME" in l))
except Exception as e:  # noqa: BLE001
    print("services 查询失败:", e)