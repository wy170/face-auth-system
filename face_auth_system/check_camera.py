# 测试 Windows 相机应用是否能用
# 如果这个脚本也失败，说明是系统级问题

import subprocess
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("尝试启动 Windows 相机应用测试...\n")

# 方法 1: 启动 Windows 相机 App
print("1. 尝试启动 Windows 相机 App...")
try:
    subprocess.run(["start", "microsoft.windows.camera:"], shell=True, timeout=3)
    print("   相机应用已启动 - 请检查是否有画面")
    print("   如果相机 App 也打不开，说明是系统级问题\n")
except Exception as e:
    print(f"   启动失败：{e}\n")

# 方法 2: 检查摄像头隐私设置详情
print("2. 检查摄像头应用权限...")
import winreg
try:
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam")
    value, _ = winreg.QueryValueEx(key, "Value")
    print(f"   摄像头权限：{value}")
    winreg.CloseKey(key)
except Exception as e:
    print(f"   读取失败：{e}")

print("\n3. 建议操作：")
print("   - 按 Win 键，搜索'相机'，手动打开相机应用测试")
print("   - 如果相机 App 能用：说明是 Python/OpenCV 的问题")
print("   - 如果相机 App 也不能用：检查笔记本摄像头物理开关")
print("   - 某些笔记本有 Fn+ 相机图标 的快捷键关闭摄像头")
