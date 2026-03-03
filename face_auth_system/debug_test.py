"""
人脸识别调试工具 - 测试编码和距离
"""

import cv2
import numpy as np
import os

# 配置
KNOWN_FACES_DIR = "known_faces"

def load_model():
    """加载 OpenFace 模型"""
    model_path = "openface.nn4.small2.v1.t7"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        return None
    
    print(f"✅ 加载模型：{model_path}")
    return cv2.dnn.readNetFromTorch(model_path)

def get_encoding(net, img_path):
    """从图片获取人脸编码"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # 使用 Haar 级联检测人脸
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"   ⚠️ 未检测到人脸：{img_path}")
        return None
    
    # 取最大的人脸
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = face
    face_roi = img[y:y+h, x:x+w]
    
    # 生成编码
    face_roi = cv2.resize(face_roi, (96, 96))
    blob = cv2.dnn.blobFromImage(
        face_roi, 1/255.0, (96, 96),
        (0, 0, 0), swapRB=True, crop=False
    )
    
    net.setInput(blob)
    embedding = net.forward()
    return embedding.flatten()

def main():
    print("=" * 60)
    print("  人脸识别编码调试工具")
    print("=" * 60)
    
    # 加载模型
    net = load_model()
    if net is None:
        return
    
    # 加载已知编码
    npy_file = os.path.join(KNOWN_FACES_DIR, "9680.npy")
    if not os.path.exists(npy_file):
        print(f"❌ 编码文件不存在：{npy_file}")
        print("   请先运行 recognize_lite.py 生成编码")
        return
    
    known_encoding = np.load(npy_file)
    print(f"\n✅ 已加载已知编码：{npy_file}")
    print(f"   编码维度：{known_encoding.shape}")
    
    # 测试摄像头实时画面
    print("\n[INFO] 打开摄像头进行测试...")
    print("   按 'q' 退出，按 't' 测试当前帧")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 显示提示
        cv2.putText(frame, "Press T to test, Q to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Debug - Press T to test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            # 保存当前帧并测试
            temp_file = "temp_test_frame.jpg"
            cv2.imwrite(temp_file, frame)
            print(f"\n[测试] 分析当前帧...")
            
            current_encoding = get_encoding(net, temp_file)
            if current_encoding is not None:
                # 计算距离
                distance = np.linalg.norm(current_encoding - known_encoding)
                print(f"   当前编码维度：{current_encoding.shape}")
                print(f"   与已知编码距离：{distance:.4f}")
                print(f"   阈值参考：<0.6=相同，0.6-1.0=相似，>1.0=不同")
                
                if distance < 0.6:
                    print(f"   ✅ 识别成功！(距离={distance:.4f} < 0.6)")
                elif distance < 1.0:
                    print(f"   ⚠️ 边界值 (距离={distance:.4f})，建议调整阈值")
                else:
                    print(f"   ❌ 识别失败 (距离={distance:.4f} > 1.0)")
            else:
                print(f"   ❌ 无法从当前帧提取编码")
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 调试结束")

if __name__ == "__main__":
    main()
