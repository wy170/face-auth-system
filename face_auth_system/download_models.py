#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载人脸识别所需模型文件
运行：python download_models.py
"""

import os
import urllib.request
import ssl

# 项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

print("=" * 50)
print("  人脸识别模型下载脚本")
print("=" * 50)

# 模型文件列表
MODELS = [
    {
        "name": "openface.nn4.small2.v1.t7",
        "url": "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7",
        "desc": "OpenFace 人脸识别模型 (12MB)"
    },
    {
        "name": "res10_300x300_ssd_iter_140000.caffemodel",
        "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face/master/res10_300x300_ssd_iter_140000.caffemodel",
        "desc": "SSD 人脸检测模型 (11MB)"
    },
    {
        "name": "deploy.prototxt",
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "desc": "SSD 检测器配置文件"
    }
]

def download_file(url, output_path, desc):
    """下载文件"""
    if os.path.exists(output_path):
        size = os.path.getsize(output_path) / 1024 / 1024
        print(f"[SKIP] {output_path} 已存在 ({size:.2f}MB)")
        return True
    
    print(f"[INFO] 下载 {desc} ...")
    
    try:
        # 禁用 SSL 验证（避免证书问题）
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\r   进度：{percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, report_progress)
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024 / 1024
            print(f"\r[OK] 下载完成：{size:.2f}MB")
            return True
    except Exception as e:
        print(f"\n[ERROR] 下载失败：{e}")
        return False
    
    return False

def main():
    success = 0
    total = len(MODELS)
    
    print(f"\n[INFO] 开始下载 {total} 个模型文件...\n")
    
    for model in MODELS:
        if download_file(model["url"], model["name"], model["desc"]):
            success += 1
        print()
    
    print("=" * 50)
    print(f"  下载完成：{success}/{total}")
    print("=" * 50)
    
    if success == total:
        print("\n[INFO] 所有模型已就绪！现在可以：")
        print("  1. 重新录入人员：python enroll_lite.py")
        print("  2. 启动识别系统：python recognize_lite.py")
    else:
        print("\n[WARN] 部分模型下载失败，但程序仍可运行（使用备选方案）")
        print("  - 如果没有 OpenFace 模型，将无法生成特征编码")
        print("  - 如果没有 SSD 模型，将使用 Haar 级联检测器")
    
    print()

if __name__ == "__main__":
    main()
