"""
测试摄像头索引
"""

import cv2

print("正在检测摄像头...\n")

# 测试索引 0-3
for i in range(4):
    print(f"测试摄像头索引 {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 尝试用 DirectShow
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [OK] 索引 {i} 可用！分辨率：{width}x{height}")
            cap.release()
            print(f"  >>> 建议修改 WEBCAM_INDEX = {i}\n")
        else:
            print(f"  [FAIL] 索引 {i} 能打开但无法读取画面\n")
            cap.release()
    else:
        print(f"  [FAIL] 索引 {i} 无法打开\n")

print("检测完成！")
print("\n如果所有索引都失败，请检查：")
print("  1. 摄像头是否已连接")
print("  2. 摄像头权限是否开启（设置 → 隐私 → 摄像头）")
print("  3. 是否有其他程序占用摄像头")
