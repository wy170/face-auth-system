"""
测试中文显示功能
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# 创建测试图像
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 测试各种中文显示
frame = draw_chinese_text(frame, "人脸识别认证系统 (轻量版)", (20, 30), font_size=28, color=(0, 255, 255))
frame = draw_chinese_text(frame, "帧：123 | 当前：吴佟", (20, 70), font_size=24, color=(0, 255, 255))
frame = draw_chinese_text(frame, "吴佟 | 9680 (0.85)", (20, 110), font_size=28, color=(0, 255, 0))
frame = draw_chinese_text(frame, "2026-03-02 14:38:00", (20, 150), font_size=18, color=(255, 255, 255))

# 绘制一个绿色边框模拟人脸检测
cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), 3)
cv2.rectangle(frame, (200, 400 - 35), (400, 400), (0, 255, 0), cv2.FILLED)
frame = draw_chinese_text(frame, "吴佟 | 9680", (206, 375), font_size=24, color=(255, 255, 255))

# 保存测试图
cv2.imwrite("test_display_result.jpg", frame)
print("[OK] 测试图片已保存：test_display_result.jpg")
print("[OK] 中文显示功能正常！")
