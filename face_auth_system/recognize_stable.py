"""
人脸识别认证系统 - 稳定版 v2.0
解决绿色框闪烁问题，增加时间平滑和置信度衰减
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

# ==================== 配置参数 ====================
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "database.json"
WEBCAM_INDEX = 0
DETECTION_INTERVAL = 2           # 每 2 帧检测一次
RECOGNITION_THRESHOLD = 1.0      # 识别距离阈值（调小更严格）
CONFIDENCE_DECAY = 0.1           # 置信度衰减率（每秒）
MIN_CONFIDENCE = 0.3             # 最低显示置信度
STABLE_FRAMES = 5                # 连续稳定帧数才确认身份
TIME_WINDOW = 2.0                # 时间窗口（秒），窗口内识别结果会平滑
# ==================================================

# 中文字体配置
def get_chinese_font(size=28):
    try:
        return ImageFont.truetype("simhei.ttf", size)
    except:
        try:
            return ImageFont.truetype("simsun.ttc", size)
        except:
            return ImageFont.load_default()

def draw_chinese_text(frame, text, position, font_size=28, color=(255, 255, 255)):
    """使用 PIL 在图像上绘制中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font(font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class StableFaceAuth:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.frame_count = 0
        
        # 稳定性跟踪
        self.current_name = "未检测到人脸"
        self.current_id = ""
        self.current_confidence = 0.0
        self.last_recog_time = 0
        
        # 时间窗口平滑
        self.recent_results = []  # [(timestamp, name, id, confidence), ...]
        
        # 连续帧计数
        self.consecutive_name = None
        self.consecutive_count = 0
        
        self.load_models()
        self.load_known_faces()
    
    def load_models(self):
        """加载 AI 模型"""
        print("\n[INFO] 加载 AI 模型...")
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        model_path = "openface.nn4.small2.v1.t7"
        if os.path.exists(model_path):
            self.recog_net = cv2.dnn.readNetFromTorch(model_path)
            print(f"   [OK] OpenFace 模型已加载")
        else:
            self.recog_net = None
            print(f"   [WARN] 模型不存在：{model_path}")
    
    def detect_faces(self, frame):
        """检测人脸"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def get_face_encoding(self, frame, face_rect):
        """提取人脸特征"""
        x1, y1, x2, y2 = face_rect
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        face_roi = cv2.resize(face_roi, (96, 96))
        blob = cv2.dnn.blobFromImage(
            face_roi, 1/255.0, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )
        
        self.recog_net.setInput(blob)
        embedding = self.recog_net.forward()
        return embedding.flatten()
    
    def load_known_faces(self):
        """加载已知人脸"""
        print("\n[INFO] 加载已知人员...")
        
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                database = json.load(f)
            
            for person in database.get('persons', []):
                person_id = person.get('id', '')
                npy_file = os.path.join(KNOWN_FACES_DIR, f"{person_id}.npy")
                
                if os.path.exists(npy_file):
                    encoding = np.load(npy_file)
                    self.known_encodings.append(encoding)
                    self.known_names.append(person['name'])
                    self.known_ids.append(person_id)
                    print(f"   [OK] {person['name']} ({person_id})")
            
            print(f"\n[OK] 共加载 {len(self.known_names)} 人")
        except Exception as e:
            print(f"[ERROR] 加载失败：{e}")
    
    def recognize_face(self, frame):
        """检测并识别人脸"""
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            self.current_name = "未检测到人脸"
            self.current_id = ""
            self.current_confidence = 0.0
            return frame
        
        # 找最大的人脸
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        x1, y1, x2, y2 = x, y, x+w, y+h
        
        name = "未知人员"
        person_id = ""
        confidence = 0.0
        
        if self.recog_net and self.known_encodings:
            encoding = self.get_face_encoding(frame, (x1, y1, x2, y2))
            
            if encoding is not None:
                # 计算距离
                distances = [np.linalg.norm(encoding - ke) for ke in self.known_encodings]
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                
                # 距离转置信度（距离越小置信度越高）
                confidence = max(0, 1.0 - (best_dist / RECOGNITION_THRESHOLD))
                
                if best_dist < RECOGNITION_THRESHOLD:
                    name = self.known_names[best_idx]
                    person_id = self.known_ids[best_idx]
                    
                    # 调试信息
                    print(f"[DEBUG] {name} | 距离={best_dist:.3f} | 置信度={confidence:.2%}")
        
        # ============ 时间平滑逻辑 ============
        now = time.time()
        
        # 添加新结果到窗口
        self.recent_results.append((now, name, person_id, confidence))
        
        # 清理旧结果（超出时间窗口的）
        cutoff = now - TIME_WINDOW
        self.recent_results = [(t, n, i, c) for t, n, i, c in self.recent_results if t > cutoff]
        
        # 统计窗口内最常见的身份
        if self.recent_results:
            name_counts = {}
            for t, n, i, c in self.recent_results:
                if n not in name_counts:
                    name_counts[n] = {'count': 0, 'id': i, 'total_conf': 0.0}
                name_counts[n]['count'] += 1
                name_counts[n]['id'] = i
                name_counts[n]['total_conf'] += c
            
            # 找出现最多的
            best_name = max(name_counts.keys(), key=lambda n: name_counts[n]['count'])
            best_data = name_counts[best_name]
            
            # 只有当出现频率足够高且置信度足够时才采用
            if best_data['count'] >= 2 and best_data['total_conf'] / best_data['count'] >= MIN_CONFIDENCE:
                # 连续帧计数
                if best_name == self.consecutive_name:
                    self.consecutive_count += 1
                else:
                    self.consecutive_name = best_name
                    self.consecutive_count = 1
                
                # 达到稳定帧数才切换
                if self.consecutive_count >= STABLE_FRAMES:
                    self.current_name = best_name
                    self.current_id = best_data['id']
                    self.current_confidence = best_data['total_conf'] / best_data['count']
                    self.last_recog_time = now
            else:
                self.consecutive_count = 0
        
        # 置信度衰减（如果长时间未更新）
        if now - self.last_recog_time > 1.0:
            decay = CONFIDENCE_DECAY * (now - self.last_recog_time)
            self.current_confidence = max(0, self.current_confidence - decay)
        
        # ============ 绘制结果 ============
        if self.current_confidence >= MIN_CONFIDENCE and self.current_name != "未知人员":
            color = (0, 255, 0)  # 绿色
            thickness = 3
            label = f"{self.current_name} | {self.current_id}"
            label += f" ({self.current_confidence:.0%})"
        else:
            color = (0, 0, 255)  # 红色
            thickness = 2
            label = "未知人员"
        
        # 画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 画标签背景
        cv2.rectangle(frame, (x1, y2 - 40), (x2, y2), color, cv2.FILLED)
        
        # 绘制中文标签
        frame = draw_chinese_text(frame, label, (x1 + 6, y2 - 34), font_size=24, color=(255, 255, 255))
        
        return frame
    
    def run(self):
        """启动识别"""
        print("\n[INFO] 启动摄像头...")
        video_capture = cv2.VideoCapture(WEBCAM_INDEX)
        
        if not video_capture.isOpened():
            print("[ERROR] 无法打开摄像头")
            return
        
        print("\n[OK] 系统就绪！")
        print("   按 'q' 退出 | 按 's' 截图")
        print("-" * 50)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[ERROR] 无法读取画面")
                break
            
            self.frame_count += 1
            if self.frame_count % DETECTION_INTERVAL == 0:
                frame = self.recognize_face(frame)
            
            # 显示状态
            status = f"帧：{self.frame_count} | {self.current_name}"
            frame = draw_chinese_text(frame, status, (10, 10), font_size=24, color=(0, 255, 255))
            
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame = draw_chinese_text(frame, time_str, (10, 40), font_size=18, color=(255, 255, 255))
            
            cv2.imshow('Face Auth - Stable', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] 截图：{filename}")
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("\n[INFO] 已退出")


if __name__ == "__main__":
    print("=" * 50)
    print("   人脸识别认证系统 v2.0 (稳定版)")
    print("   解决绿色框闪烁问题")
    print("=" * 50)
    
    system = StableFaceAuth()
    system.run()
