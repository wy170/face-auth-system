"""
人脸识别稳定性测试
自动测试连续识别、不同距离/角度的表现
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

# 配置
WEBCAM_INDEX = 0
RECOGNITION_THRESHOLD = 1.2
TEST_DURATION = 60  # 测试时长（秒）

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
print("=" * 50)
print("   人脸识别稳定性测试")
print("=" * 50)

print("\n加载模型...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog_net = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")

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

print(f"\n共 {len(known_names)} 人")

# 打开摄像头
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

# 测试统计
stats = {
    'total_frames': 0,
    'face_detected': 0,
    'recognized': 0,
    'unknown': 0,
    'recognition_results': {},  # {name: count}
    'distances': [],
    'start_time': time.time()
}

print("\n" + "=" * 50)
print(f"开始测试，持续 {TEST_DURATION} 秒...")
print("按 'q' 提前退出 | 按 's' 截图")
print("=" * 50 + "\n")

display_name = "准备中"
last_update = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    stats['total_frames'] += 1
    elapsed = time.time() - stats['start_time']
    
    # 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    
    if len(faces) > 0:
        stats['face_detected'] += 1
        face = max(faces, key=lambda f: f[2]*f[3])
        x, y, w, h = face
        
        # 识别
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (96, 96))
        blob = cv2.dnn.blobFromImage(roi, 1/255.0, (96,96), (0,0,0), swapRB=True)
        recog_net.setInput(blob)
        embedding = recog_net.forward().flatten()
        
        distances = [np.linalg.norm(embedding - ke) for ke in known_encodings]
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        stats['distances'].append(best_dist)
        
        if best_dist < RECOGNITION_THRESHOLD:
            stats['recognized'] += 1
            name = known_names[best_idx]
            stats['recognition_results'][name] = stats['recognition_results'].get(name, 0) + 1
            conf = max(0, 1 - best_dist/RECOGNITION_THRESHOLD)
            
            color = (0, 255, 0)
            label = f"{name} ({conf:.0%})"
            
            # 每秒打印一次日志
            if time.time() - last_update > 1.0:
                print(f"[{elapsed:.0f}s] {name} - 距离:{best_dist:.3f} - 置信度:{conf:.0%}")
                last_update = time.time()
        else:
            stats['unknown'] += 1
            color = (0, 0, 255)
            label = "未知人员"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
        frame = draw_text(frame, label, (x+5, y+h-30), size=24, color=(255,255,255))
    else:
        label = "未检测到人脸"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    # 显示测试信息
    fps = stats['total_frames'] / elapsed if elapsed > 0 else 0
    info = f"测试：{elapsed:.0f}/{TEST_DURATION}s | FPS:{fps:.1f} | 检测:{stats['face_detected']} | 识别:{stats['recognized']}"
    frame = draw_text(frame, info, (10, 10), size=18, color=(0,255,255))
    
    cv2.imshow('Stability Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or elapsed >= TEST_DURATION:
        break
    elif key == ord('s'):
        fn = f"test_shot_{int(time.time())}.jpg"
        cv2.imwrite(fn, frame)
        print(f"✓ 截图：{fn}")

# 生成报告
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("   测试报告")
print("=" * 50)
print(f"测试时长：{elapsed:.1f} 秒")
print(f"总帧数：{stats['total_frames']}")
print(f"人脸检测：{stats['face_detected']} 帧 ({stats['face_detected']/stats['total_frames']*100:.1f}%)")
print(f"识别成功：{stats['recognized']} 帧")
print(f"未知人员：{stats['unknown']} 帧")

if stats['recognized'] > 0:
    print(f"\n识别分布:")
    for name, count in sorted(stats['recognition_results'].items(), key=lambda x: -x[1]):
        pct = count / stats['recognized'] * 100
        print(f"  {name}: {count} 帧 ({pct:.1f}%)")

if stats['distances']:
    avg_dist = np.mean(stats['distances'])
    min_dist = np.min(stats['distances'])
    max_dist = np.max(stats['distances'])
    print(f"\n距离统计:")
    print(f"  平均：{avg_dist:.4f}")
    print(f"  最小：{min_dist:.4f}")
    print(f"  最大：{max_dist:.4f}")
    print(f"  阈值：{RECOGNITION_THRESHOLD}")

print("\n" + "=" * 50)
print("测试完成！")
print("=" * 50)
