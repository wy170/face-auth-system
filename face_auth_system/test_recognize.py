"""
人脸识别测试 - 单帧测试
不显示实时窗口，只测试单次识别是否准确
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import json
import os
import time
from PIL import Image, ImageDraw, ImageFont

# 配置
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "database.json"
WEBCAM_INDEX = 0
RECOGNITION_THRESHOLD = 1.0

def get_chinese_font(size=28):
    try:
        return ImageFont.truetype("simhei.ttf", size)
    except:
        return ImageFont.load_default()

def draw_chinese_text(frame, text, position, font_size=28, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font(font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 加载模型
print("加载模型...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog_net = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
print("模型加载完成")

# 加载已知人脸
print("\n加载已知人员...")
known_encodings = []
known_names = []
known_ids = []

with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
    database = json.load(f)

for person in database.get('persons', []):
    person_id = person.get('id', '')
    npy_file = os.path.join(KNOWN_FACES_DIR, f"{person_id}.npy")
    if os.path.exists(npy_file):
        encoding = np.load(npy_file)
        known_encodings.append(encoding)
        known_names.append(person['name'])
        known_ids.append(person_id)
        print(f"  ✓ {person['name']} ({person_id})")

print(f"\n共加载 {len(known_names)} 人\n")

# 打开摄像头
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

print("摄像头已打开，按任意键拍照测试，按 'q' 退出\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # 显示预览
    cv2.putText(frame, "Press any key to capture (q=quit)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Preview - Press Key', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:  # 任意键拍照
        print("\n📸 拍照中...")
        
        # 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("❌ 未检测到人脸")
            continue
        
        # 取最大的人脸
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        face_roi = frame[y:y+h, x:x+w]
        
        # 提取特征
        face_resized = cv2.resize(face_roi, (96, 96))
        blob = cv2.dnn.blobFromImage(face_resized, 1/255.0, (96, 96), (0,0,0), swapRB=True, crop=False)
        recog_net.setInput(blob)
        embedding = recog_net.forward().flatten()
        
        # 识别
        distances = [np.linalg.norm(embedding - ke) for ke in known_encodings]
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        confidence = max(0, 1.0 - (best_dist / RECOGNITION_THRESHOLD))
        
        name = known_names[best_idx] if best_dist < RECOGNITION_THRESHOLD else "未知人员"
        person_id = known_ids[best_idx] if best_dist < RECOGNITION_THRESHOLD else ""
        
        print(f"\n识别结果:")
        print(f"  姓名：{name}")
        print(f"  ID: {person_id}")
        print(f"  距离：{best_dist:.4f} (阈值：{RECOGNITION_THRESHOLD})")
        print(f"  置信度：{confidence:.2%}")
        
        # 保存结果图
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f"{name} ({confidence:.0%})"
        frame = draw_chinese_text(frame, label, (x, y-10), font_size=24, color=(0, 255, 0))
        
        result_file = f"test_result_{int(time.time())}.jpg"
        cv2.imwrite(result_file, frame)
        print(f"\n结果已保存：{result_file}")
        print("-" * 50)

cap.release()
cv2.destroyAllWindows()
print("\n测试结束")
