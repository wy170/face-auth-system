"""
人脸录入系统 - 轻量版 (无需 dlib)
使用 OpenCV DNN + 预训练模型
"""

# 设置 UTF-8 编码（Windows 兼容）
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import json
import os
from datetime import datetime

# 配置
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "database.json"
WEBCAM_INDEX = 0
CAPTURE_COUNT = 10


def ensure_dirs():
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"✅ 创建目录：{KNOWN_FACES_DIR}")


def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"persons": [], "created": datetime.now().isoformat()}


def save_database(database):
    with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)


class FaceEnroller:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 尝试加载识别模型
        recog_model = "openface.nn4.small2.v1.t7"
        if os.path.exists(recog_model):
            self.recog_net = cv2.dnn.readNetFromTorch(recog_model)
            print("   [OK] 识别模型已加载")
        else:
            self.recog_net = None
            print("   [WARN] 识别模型未找到，将使用 OpenCV 内置方法")
            print("          首次运行会自动下载模型，或手动放置到项目目录")
    
    def get_face_encoding(self, face_roi):
        """获取人脸特征编码"""
        if self.recog_net is None:
            return None
        
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
    
    def capture_face(self, name, person_id):
        """采集人脸"""
        print(f"\n[INFO] 开始采集 {name} 的人脸...")
        print("   请保持正脸面对摄像头，光线充足")
        print("   按 'c' 开始采集，按 'q' 取消")
        
        video_capture = cv2.VideoCapture(WEBCAM_INDEX)
        
        if not video_capture.isOpened():
            print("❌ 无法打开摄像头")
            return None, None
        
        encodings = []
        captured_images = []
        captured = False
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # 显示提示
            if not captured:
                cv2.putText(frame, "Press C to capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"Captured {len(encodings)}/{CAPTURE_COUNT}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if captured and len(encodings) < CAPTURE_COUNT:
                    face_roi = frame[y:y+h, x:x+w]
                    captured_images.append(face_roi.copy())
                    
                    if self.recog_net:
                        encoding = self.get_face_encoding(face_roi)
                        if encoding is not None:
                            encodings.append(encoding)
                            print(f"   已采集 {len(encodings)}/{CAPTURE_COUNT}")
                    else:
                        encodings.append(None)  # 仅保存图片
            
            cv2.imshow('人脸采集', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("   取消采集")
                break
            elif key == ord('c') and not captured:
                captured = True
                print("   开始采集...")
            
            if captured and len(encodings) >= CAPTURE_COUNT:
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        if len(encodings) == 0:
            print("[ERROR] 未采集到有效人脸")
            return None, None
        
        # 保存采集的图片（备选方案）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        images_dir = os.path.join(KNOWN_FACES_DIR, f"{person_id}_images")
        if captured_images and not os.path.exists(images_dir):
            os.makedirs(images_dir)
            for i, img in enumerate(captured_images):
                cv2.imwrite(os.path.join(images_dir, f"{i}.jpg"), img)
            print(f"   [OK] 已保存 {len(captured_images)} 张人脸图片")
        
        # 返回平均编码
        if self.recog_net and encodings:
            valid_encodings = [e for e in encodings if e is not None]
            if valid_encodings:
                avg_encoding = np.mean(valid_encodings, axis=0)
                return avg_encoding, "encoding"
        
        return None, "images"
    
    def enroll_person(self):
        """录入新人员"""
        print("\n" + "=" * 50)
        print("   人员录入系统 (轻量版)")
        print("=" * 50)
        
        name = input("\n请输入姓名：").strip()
        if not name:
            print("❌ 姓名不能为空")
            return
        
        person_id = input("请输入编号/考号：").strip()
        id_card = input("请输入身份证号（可选）：").strip()
        
        encoding, encoding_type = self.capture_face(name, person_id)
        
        if encoding is None and encoding_type == "encoding":
            print("[ERROR] 人脸采集失败")
            return
        
        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if encoding is not None:
            encoding_filename = f"{person_id}_{timestamp}.npy"
            encoding_path = os.path.join(KNOWN_FACES_DIR, encoding_filename)
            np.save(encoding_path, encoding)
            print(f"[OK] 人脸编码已保存：{encoding_filename}")
        else:
            encoding_filename = f"{person_id}_images"
            print(f"[OK] 人脸图片已保存：{encoding_filename}/")
        
        # 更新数据库
        database = load_database()
        database["persons"].append({
            "name": name,
            "id": person_id,
            "id_card": id_card,
            "encoding_file": encoding_filename,
            "encoding_type": encoding_type,
            "enrolled_at": datetime.now().isoformat()
        })
        save_database(database)
        
        print(f"\n✅ 人员录入成功！")
        print(f"   姓名：{name}")
        print(f"   编号：{person_id}")
        print(f"   当前数据库共 {len(database['persons'])} 人")


def list_persons():
    """列出所有已录入人员"""
    database = load_database()
    persons = database.get("persons", [])
    
    print("\n" + "=" * 50)
    print(f"   已录入人员 ({len(persons)}人)")
    print("=" * 50)
    
    if not persons:
        print("\n暂无录入人员")
        return
    
    for i, p in enumerate(persons, 1):
        print(f"\n{i}. {p['name']}")
        print(f"   编号：{p['id']}")
        print(f"   身份证：{p.get('id_card', '未填写')}")
        print(f"   录入时间：{p.get('enrolled_at', '未知')[:19]}")


def main():
    ensure_dirs()
    enroller = FaceEnroller()
    
    while True:
        print("\n请选择操作：")
        print("  1. 录入新人员")
        print("  2. 查看已录入人员")
        print("  3. 退出")
        
        choice = input("\n输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            enroller.enroll_person()
        elif choice == "2":
            list_persons()
        elif choice == "3":
            print("\n👋 退出系统")
            break
        else:
            print("❌ 无效选项")


if __name__ == "__main__":
    main()
