"""
人脸识别认证系统 - 实时检测并显示身份信息
用途：考场/考务人员身份验证
"""

import cv2
import face_recognition
import numpy as np
import json
import os
from datetime import datetime

# 配置
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "database.json"
WEBCAM_INDEX = 0  # 摄像头索引，默认 0
DETECTION_INTERVAL = 3  # 每 N 帧检测一次人脸，提升性能
CONFIDENCE_THRESHOLD = 0.6  # 识别置信度阈值

class FaceAuthSystem:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.frame_count = 0
        self.current_name = "未检测到人脸"
        self.current_id = ""
        self.last_detection_time = None
        
        # 加载已知人脸数据
        self.load_known_faces()
        
    def load_known_faces(self):
        """从数据库加载已知人员的人脸编码"""
        if not os.path.exists(DATABASE_FILE):
            print(f"⚠️  数据库文件不存在：{DATABASE_FILE}")
            print("   请先运行 enroll.py 录入人员信息")
            return
            
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                database = json.load(f)
            
            for person in database.get('persons', []):
                encoding_file = os.path.join(KNOWN_FACES_DIR, person['encoding_file'])
                if os.path.exists(encoding_file):
                    encoding = np.load(encoding_file)
                    self.known_encodings.append(encoding)
                    self.known_names.append(person['name'])
                    self.known_ids.append(person.get('id', ''))
            
            print(f"✅ 已加载 {len(self.known_names)} 个已知人员")
            for name, pid in zip(self.known_names, self.known_ids):
                print(f"   - {name} (ID: {pid})")
                
        except Exception as e:
            print(f"❌ 加载数据库失败：{e}")
    
    def recognize_face(self, frame):
        """检测并识别人脸"""
        # 缩小图像以加快处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 检测人脸位置
        face_locations = face_recognition.face_locations(rgb_small)
        
        if not face_locations:
            self.current_name = "未检测到人脸"
            self.current_id = ""
            return frame
        
        # 检测人脸编码
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 还原到原始尺寸
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # 匹配已知人脸
            name = "未知人员"
            person_id = ""
            
            if self.known_encodings:
                matches = face_recognition.compare_faces(
                    self.known_encodings, 
                    face_encoding,
                    tolerance=0.5  # 更严格的匹配阈值
                )
                
                # 计算距离，找最佳匹配
                face_distances = face_recognition.face_distance(
                    self.known_encodings, 
                    face_encoding
                )
                
                if len(face_distances) > 0:
                    best_match_idx = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_idx]
                    
                    if matches[best_match_idx] and confidence >= CONFIDENCE_THRESHOLD:
                        name = self.known_names[best_match_idx]
                        person_id = self.known_ids[best_match_idx]
                        self.last_detection_time = datetime.now()
            
            self.current_name = name
            self.current_id = person_id
            
            # 绘制边框和标签
            if name != "未知人员":
                color = (0, 255, 0)  # 绿色 - 已识别
            else:
                color = (0, 0, 255)  # 红色 - 未知
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # 显示姓名和 ID
            label = f"{name}"
            if person_id:
                label += f" | {person_id}"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """启动实时识别"""
        print("\n🎥 启动摄像头...")
        video_capture = cv2.VideoCapture(WEBCAM_INDEX)
        
        if not video_capture.isOpened():
            print("❌ 无法打开摄像头，请检查设备")
            return
        
        print("\n✅ 系统就绪！")
        print("   按 'q' 键退出")
        print("   按 's' 键截图保存")
        print("-" * 50)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            # 每隔几帧检测一次，提升性能
            self.frame_count += 1
            if self.frame_count % DETECTION_INTERVAL == 0:
                frame = self.recognize_face(frame)
            else:
                # 非检测帧，只显示上次的识别结果
                if self.current_name != "未检测到人脸":
                    # 可以在这里添加轻量级的追踪逻辑
                    pass
            
            # 显示状态信息
            status_text = f"FPS: {video_capture.get(cv2.CAP_PROP_FPS):.1f} | "
            status_text += f"帧：{self.frame_count} | "
            status_text += f"当前：{self.current_name}"
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示时间
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, time_str, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('人脸识别认证系统', frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 截图保存
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 截图已保存：{filename}")
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("\n👋 系统已退出")


if __name__ == "__main__":
    print("=" * 50)
    print("   人脸识别认证系统 v1.0")
    print("   考场/考务身份验证")
    print("=" * 50)
    
    system = FaceAuthSystem()
    system.run()
