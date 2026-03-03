"""
人脸识别认证系统 - 最终稳定版
简化逻辑，确保稳定运行
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ===== 配置 =====
WEBCAM_INDEX = 0
RECOGNITION_THRESHOLD = 1.5  # 距离阈值（调大，更容易识别已录入人员）
STABLE_COUNT = 2             # 连续 2 帧相同就确认（更灵敏）
LOST_COUNT = 5               # 连续 5 帧丢失才切换为未知
# ================

def get_font(size=28):
    try:
        return ImageFont.truetype("simhei.ttf", size)
    except:
        return ImageFont.load_default()

def draw_text(frame, text, pos, size=28, color=(255,255,255)):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = get_font(size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 加载模型
print("加载模型...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog_net = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
print("✓ 模型就绪\n")

# 加载已知人脸
print("加载已知人员...")
known_encodings = []
known_names = []
known_ids = []

with open("database.json", 'r', encoding='utf-8') as f:
    db = json.load(f)

for p in db.get('persons', []):
    npy_file = f"known_faces/{p['id']}.npy"
    if os.path.exists(npy_file):
        known_encodings.append(np.load(npy_file))
        known_names.append(p['name'])
        known_ids.append(p['id'])
        print(f"  ✓ {p['name']}")

print(f"\n共 {len(known_names)} 人\n")

# 打开摄像头
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

print("✓ 摄像头就绪")
print("按 'q' 退出 | 按 's' 截图")
print("=" * 40)

# 状态跟踪
display_name = "未检测到人脸"
display_id = ""
stable_name = None
stable_count = 0
lost_count = 0
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    
    if len(faces) > 0:
        # 取最大人脸
        face = max(faces, key=lambda f: f[2]*f[3])
        x, y, w, h = face
        
        # 识别
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (96, 96))
        blob = cv2.dnn.blobFromImage(roi, 1/255.0, (96,96), (0,0,0), swapRB=True)
        recog_net.setInput(blob)
        embedding = recog_net.forward().flatten()
        
        # 匹配
        distances = [np.linalg.norm(embedding - ke) for ke in known_encodings]
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        
        if best_dist < RECOGNITION_THRESHOLD:
            # 已录入人员 - 绿色框
            rec_name = known_names[best_idx]
            rec_id = known_ids[best_idx]
            conf = max(0, 1 - best_dist/RECOGNITION_THRESHOLD)
            
            # 稳定性检查
            if rec_name == stable_name:
                stable_count += 1
                lost_count = 0
            else:
                stable_name = rec_name
                stable_count = 1
                lost_count = 0
            
            # 连续 2 帧相同就更新显示
            if stable_count >= STABLE_COUNT:
                display_name = rec_name
                display_id = rec_id
                last_time = time.time()
            
            color = (0, 255, 0)  # 绿色
            label = f"{display_name} | {display_id} ({conf:.0%})"
            print(f"✓ 识别：{label} (距离={best_dist:.3f})")
        else:
            # 检测到人脸但未识别 - 红色框
            lost_count += 1
            if lost_count >= LOST_COUNT:
                display_name = "未知人员"
                display_id = ""
            
            color = (0, 0, 255)  # 红色
            label = "未知人员"
    else:
        # 未检测到人脸
        lost_count += 1
        if lost_count >= LOST_COUNT:
            display_name = "未检测到人脸"
            display_id = ""
        
        color = (0, 0, 255)  # 红色
        label = display_name
    
    # 绘制
    if len(faces) > 0:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
        frame = draw_text(frame, label, (x+5, y+h-30), size=24, color=(255,255,255))
    
    # 状态栏
    frame = draw_text(frame, f"状态：{display_name}", (10, 10), size=20, color=(0,255,255))
    frame = draw_text(frame, datetime.now().strftime("%H:%M:%S"), (10, 35), size=18)
    
    cv2.imshow('Face Auth', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fn = f"shot_{int(time.time())}.jpg"
        cv2.imwrite(fn, frame)
        print(f"✓ 截图：{fn}")

cap.release()
cv2.destroyAllWindows()
print("\n退出")
