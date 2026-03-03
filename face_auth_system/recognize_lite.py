"""
人脸识别认证系统 - 轻量版 v1.0
无需 dlib，使用 OpenFace 模型
"""

# 设置 UTF-8 编码（Windows 兼容）
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
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "database.json"
WEBCAM_INDEX = 0
DETECTION_INTERVAL = 2  # 每 2 帧检测一次
CONFIDENCE_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 1.2  # 识别距离阈值（调大一点，更容易识别成功）
DEBUG_MODE = True  # 显示距离调试信息
STABLE_FRAME_COUNT = 3  # 连续 3 帧成功才切换状态

# 中文字体配置
try:
    chinese_font = ImageFont.truetype("simhei.ttf", 28)  # 黑体
    small_font = ImageFont.truetype("simhei.ttf", 20)
except:
    try:
        chinese_font = ImageFont.truetype("simsun.ttc", 28)  # 宋体
        small_font = ImageFont.truetype("simsun.ttc", 20)
    except:
        chinese_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

def draw_chinese_text(frame, text, position, font_size=28, color=(255, 255, 255)):
    """使用 PIL 在图像上绘制中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class FaceAuthSystemLite:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.frame_count = 0
        self.current_name = "未检测到人脸"
        self.current_id = ""
        self.last_distance = 999
        
        # 稳定性跟踪
        self.stable_counter = 0
        self.stable_name = None  # 改为 None，方便判断是否有值
        self.stable_id = None
        self.last_success_time = 0
        
        self.load_models()
        self.load_known_faces()
    
    def load_models(self):
        """加载 AI 模型"""
        print("\n[INFO] 加载 AI 模型...")
        
        # Haar 级联人脸检测
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # OpenFace 识别模型
        model_path = "openface.nn4.small2.v1.t7"
        if os.path.exists(model_path):
            self.recog_net = cv2.dnn.readNetFromTorch(model_path)
            self.use_dnn_detector = False
            print("   - 使用 Haar 级联检测器 (轻量)")
            print("   - 使用 OpenFace 识别模型")
            print("   [OK] 模型加载完成")
        else:
            self.recog_net = None
            print(f"   [WARN] 识别模型不存在：{model_path}")
            print("   将使用备用方案")
    
    def detect_faces_haar(self, frame):
        """Haar 级联检测人脸"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def get_face_encoding(self, frame, face_rect):
        """从人脸区域提取特征编码"""
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
        """加载已知人脸数据"""
        print("\n[INFO] 加载已知人员数据...")
        
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
                    print(f"   [OK] 加载：{person['name']} ({person_id})")
                
                elif os.path.exists(os.path.join(KNOWN_FACES_DIR, f"{person_id}_images")):
                    # 从图片目录生成编码
                    print(f"   [INFO] 发现图片目录：{person['name']}，正在生成特征编码...")
                    encoding = self._generate_encoding_from_images(person_id)
                    if encoding is not None:
                        npy_file = os.path.join(KNOWN_FACES_DIR, f"{person_id}.npy")
                        np.save(npy_file, encoding)
                        print(f"   [OK] 编码已保存：{person_id}.npy")
                        self.known_encodings.append(encoding)
                        self.known_names.append(person['name'])
                        self.known_ids.append(person_id)
            
            if self.known_encodings:
                print(f"\n[OK] 已加载 {len(self.known_names)} 个已知人员")
            else:
                print(f"\n[WARN] 未加载到任何有效人脸数据")
                
        except Exception as e:
            print(f"[ERROR] 加载数据库失败：{e}")
    
    def _generate_encoding_from_images(self, person_id):
        """从图片目录生成平均编码"""
        images_dir = os.path.join(KNOWN_FACES_DIR, f"{person_id}_images")
        encodings = []
        
        for img_name in os.listdir(images_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    faces = self.detect_faces_haar(img)
                    if len(faces) > 0:
                        face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = face
                        encoding = self.get_face_encoding(img, (x, y, x+w, y+h))
                        if encoding is not None:
                            encodings.append(encoding)
                            print(f"      处理：{img_name}")
        
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            print(f"      成功从 {len(encodings)} 张图片生成编码")
            return avg_encoding
        return None
    
    def recognize_face(self, frame):
        """检测并识别人脸"""
        faces = self.detect_faces_haar(frame)
        
        if len(faces) == 0:
            self.current_name = "未检测到人脸"
            self.current_id = ""
            return frame
        
        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = x, y, x+w, y+h
            name = "未知人员"
            person_id = ""
            
            if self.recog_net and self.known_encodings:
                encoding = self.get_face_encoding(frame, (x1, y1, x2, y2))
                
                if encoding is not None:
                    distances = [np.linalg.norm(encoding - ke) for ke in self.known_encodings]
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]
                    self.last_distance = best_dist
                    
                    if DEBUG_MODE:
                        print(f"[DEBUG] 距离={best_dist:.4f} | 阈值={RECOGNITION_THRESHOLD} | 目标={self.known_names[best_idx]}")
                    
                    # 识别成功
                    if best_dist < RECOGNITION_THRESHOLD:
                        target_name = self.known_names[best_idx]
                        target_id = self.known_ids[best_idx]
                        
                        # 稳定性检查 - 连续稳定才显示
                        if target_name == self.stable_name:
                            self.stable_counter += 1
                        else:
                            self.stable_name = target_name
                            self.stable_id = target_id
                            self.stable_counter = 1
                        
                        # 达到稳定帧数，更新显示
                        if self.stable_counter >= STABLE_FRAME_COUNT:
                            name = self.stable_name
                            person_id = self.stable_id
                            self.last_success_time = time.time()
                        
                        # 即使未稳定，如果最近成功过也保持显示（避免闪烁）
                        if self.stable_name is not None and time.time() - self.last_success_time < 0.5:
                            name = self.stable_name
                            person_id = self.stable_id
                    else:
                        # 识别失败，衰减计数
                        self.stable_counter = max(0, self.stable_counter - 1)
                        # 如果最近成功过，保持显示 1 秒
                        if self.stable_name is not None and time.time() - self.last_success_time < 1.0:
                            name = self.stable_name
                            person_id = self.stable_id
            
            self.current_name = name
            self.current_id = person_id
            
            # 绘制边框
            if name != "未知人员":
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (0, 0, 255)
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 显示标签
            label = f"{name}"
            if person_id:
                label += f" | {person_id}"
            
            if DEBUG_MODE and self.last_distance < 999:
                label += f" ({self.last_distance:.2f})"
            
            # 使用 PIL 绘制中文标签
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            frame = draw_chinese_text(frame, label, (x1 + 6, y2 - 30), font_size=24, color=(255, 255, 255))
        
        return frame
    
    def run(self):
        """启动实时识别"""
        print("\n[INFO] 启动摄像头...")
        video_capture = cv2.VideoCapture(WEBCAM_INDEX)
        
        if not video_capture.isOpened():
            print("[ERROR] 无法打开摄像头，请检查设备")
            return
        
        print("\n[OK] 系统就绪！")
        print("   按 'q' 键退出")
        print("   按 's' 键截图保存")
        print("-" * 50)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[ERROR] 无法读取摄像头画面")
                break
            
            self.frame_count += 1
            if self.frame_count % DETECTION_INTERVAL == 0:
                frame = self.recognize_face(frame)
            
            # 显示状态（使用 PIL 绘制中文）
            status_text = f"帧：{self.frame_count} | 当前：{self.current_name}"
            frame = draw_chinese_text(frame, status_text, (10, 10), font_size=24, color=(0, 255, 255))
            
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame = draw_chinese_text(frame, time_str, (10, 40), font_size=18, color=(255, 255, 255))
            
            cv2.imshow('Face Auth System Lite', frame)  # 窗口标题用英文避免乱码
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] 截图已保存：{filename}")
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("\n[INFO] 系统已退出")


if __name__ == "__main__":
    print("=" * 50)
    print("   人脸识别认证系统 v1.0 (轻量版)")
    print("   无需 dlib，快速部署")
    print("=" * 50)
    
    system = FaceAuthSystemLite()
    system.run()
