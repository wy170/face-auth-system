# -*- coding: utf-8 -*-
import cv2
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("测试摄像头 - 使用不同后端...\n")

# 尝试不同后端
backends = [
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "Media Foundation"),
    (cv2.CAP_ANY, "Auto Detect"),
]

for backend, name in backends:
    print(f"尝试 {name} (后端 {backend})...")
    cap = cv2.VideoCapture(0, backend)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  [成功] {name} 可用！\n")
            cap.release()
            
            # 保存一帧测试
            cv2.imwrite("camera_test.jpg", frame)
            print("  测试图片已保存：camera_test.jpg")
            print("\n>>> 修改 recognize_lite.py: 使用 cv2.VideoCapture(0, cv2.CAP_DSHOW)")
            sys.exit(0)
        else:
            print(f"  [失败] {name} 能打开但无法读取\n")
            cap.release()
    else:
        print(f"  [失败] {name} 无法打开\n")

print("所有后端都失败了...")
