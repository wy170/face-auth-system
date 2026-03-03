"""
人脸录入系统 - 采集并保存人员信息和人脸编码
用途：事先录入考生/考务人员身份
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
WEBCAM_INDEX = 0
CAPTURE_COUNT = 10  # 采集多少张照片取平均


def ensure_dirs():
    """确保目录存在"""
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"✅ 创建目录：{KNOWN_FACES_DIR}")


def load_database():
    """加载现有数据库"""
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"persons": [], "created": datetime.now().isoformat()}


def save_database(database):
    """保存数据库"""
    with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)


def capture_face_encoding(name, person_id, id_card=""):
    """采集人脸并生成编码"""
    print(f"\n📷 开始采集 {name} 的人脸...")
    print("   请保持正脸面对摄像头，光线充足")
    print("   按 'c' 开始采集，按 'q' 取消")
    
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not video_capture.isOpened():
        print("❌ 无法打开摄像头")
        return None
    
    encodings = []
    captured = False
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # 显示提示
        if not captured:
            cv2.putText(frame, "Press 'C' to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Captured {len(encodings)}/{CAPTURE_COUNT}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 检测人脸
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            for (top, right, bottom, left) in face_locations:
                # 绘制人脸框
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                if captured and len(encodings) < CAPTURE_COUNT:
                    # 提取编码
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
                    encodings.append(face_encoding)
                    print(f"   已采集 {len(encodings)}/{CAPTURE_COUNT}")
        
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
        print("❌ 未采集到有效人脸")
        return None
    
    # 计算平均编码
    avg_encoding = np.mean(encodings, axis=0)
    return avg_encoding


def enroll_person():
    """录入新人员"""
    print("\n" + "=" * 50)
    print("   人员录入系统")
    print("=" * 50)
    
    # 输入信息
    name = input("\n请输入姓名：").strip()
    if not name:
        print("❌ 姓名不能为空")
        return
    
    person_id = input("请输入编号/考号：").strip()
    id_card = input("请输入身份证号（可选）：").strip()
    
    # 采集人脸
    encoding = capture_face_encoding(name, person_id, id_card)
    
    if encoding is None:
        print("❌ 人脸采集失败")
        return
    
    # 保存编码文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoding_filename = f"{person_id}_{timestamp}.npy"
    encoding_path = os.path.join(KNOWN_FACES_DIR, encoding_filename)
    np.save(encoding_path, encoding)
    print(f"✅ 人脸编码已保存：{encoding_path}")
    
    # 更新数据库
    database = load_database()
    database["persons"].append({
        "name": name,
        "id": person_id,
        "id_card": id_card,
        "encoding_file": encoding_filename,
        "enrolled_at": datetime.now().isoformat()
    })
    save_database(database)
    
    print(f"\n✅ 人员录入成功！")
    print(f"   姓名：{name}")
    print(f"   编号：{person_id}")
    print(f"   身份证号：{id_card if id_card else '未填写'}")
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
    
    while True:
        print("\n请选择操作：")
        print("  1. 录入新人员")
        print("  2. 查看已录入人员")
        print("  3. 退出")
        
        choice = input("\n输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            enroll_person()
        elif choice == "2":
            list_persons()
        elif choice == "3":
            print("\n👋 退出系统")
            break
        else:
            print("❌ 无效选项")


if __name__ == "__main__":
    main()
