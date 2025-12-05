#!/usr/bin/env python3
"""
创建测试图像：模拟原始图像和AI增强图像
"""

import cv2
import numpy as np


def create_test_images():
    """
    创建一对测试图像：
    - original.jpg: 原始图像（正常亮度，丰富细节）
    - enhanced.jpg: AI增强图像（色彩鲜艳，对比度高，但可能丢失细节）
    """
    
    # 图像尺寸
    height, width = 512, 512
    
    # ==================== 创建原始图像 ====================
    # 创建一个包含多种频率成分的复杂图像
    original = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 背景渐变（低频）
    for i in range(height):
        for j in range(width):
            original[i, j] = [
                int(100 + 50 * np.sin(2 * np.pi * i / height)),
                int(120 + 40 * np.cos(2 * np.pi * j / width)),
                int(140 + 30 * np.sin(2 * np.pi * (i + j) / (height + width)))
            ]
    
    # 添加几何图形（中频）
    cv2.circle(original, (width // 4, height // 4), 60, (80, 150, 200), -1)
    cv2.circle(original, (3 * width // 4, height // 4), 60, (200, 100, 80), -1)
    cv2.rectangle(original, (width // 4 - 40, 3 * height // 4 - 40),
                  (width // 4 + 40, 3 * height // 4 + 40), (100, 200, 100), -1)
    cv2.rectangle(original, (3 * width // 4 - 40, 3 * height // 4 - 40),
                  (3 * width // 4 + 40, 3 * height // 4 + 40), (200, 200, 50), -1)
    
    # 添加文字（高频细节）
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, 'ORIGINAL', (width // 2 - 120, height // 2),
                font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 添加细节纹理（高频）
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    original = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 添加网格线（高频细节）
    for i in range(0, height, 32):
        cv2.line(original, (0, i), (width, i), (255, 255, 255), 1)
    for j in range(0, width, 32):
        cv2.line(original, (j, 0), (j, height), (255, 255, 255), 1)
    
    # ==================== 创建增强图像 ====================
    # 模拟AI增强：提高对比度和饱和度，但可能丢失细节
    enhanced = original.copy()
    
    # 转换到HSV空间以增强饱和度
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # 增强饱和度
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)  # 增强亮度
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 增强对比度
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2Lab).astype(np.float32)
    l_channel = lab[:, :, 0]
    l_mean = np.mean(l_channel)
    lab[:, :, 0] = np.clip((l_channel - l_mean) * 1.3 + l_mean, 0, 255)
    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    
    # 轻微模糊（模拟AI增强可能丢失的高频细节）
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    
    # 保存图像
    cv2.imwrite('/home/ubuntu/original.jpg', original)
    cv2.imwrite('/home/ubuntu/enhanced.jpg', enhanced)
    
    print("✓ 测试图像已创建:")
    print("  - /home/ubuntu/original.jpg (原始图像)")
    print("  - /home/ubuntu/enhanced.jpg (AI增强图像)")
    
    # 创建对比图
    comparison = np.hstack([original, enhanced])
    cv2.imwrite('/home/ubuntu/test_images_comparison.jpg', comparison)
    print("  - /home/ubuntu/test_images_comparison.jpg (对比图)")


if __name__ == "__main__":
    create_test_images()
