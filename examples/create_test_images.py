#!/usr/bin/env python3
"""
创建测试图像：模拟原始图像和AI增强图像
"""

import os
from pathlib import Path

import cv2
import numpy as np


def create_test_images(height: int = 512, width: int = 512, seed: int | None = None, output_dir: str | None = None):
    """
    创建一对测试图像：
    - original.jpg: 原始图像（正常亮度，丰富细节）
    - enhanced.jpg: AI增强图像（色彩鲜艳，对比度高，但可能丢失细节）
    参数：
      height, width: 图像尺寸
      seed: 可选随机种子（用于复现噪声）
      output_dir: 可选输出目录（路径字符串或 None 使用项目默认）
    """
    # 可选的随机种子以保证可复现
    if seed is not None:
        np.random.seed(seed)

    # ==================== 创建原始图像 ====================
    # 生成背景通道：注意 c0/c1 需要显式广播到 (height, width) 再堆叠
    ys = np.arange(height).reshape(height, 1)
    xs = np.arange(width).reshape(1, width)
    # shape (height,1)
    c0 = 100 + 50 * np.sin(2 * np.pi * ys / height)
    # shape (1,width)
    c1 = 120 + 40 * np.cos(2 * np.pi * xs / width)
    c2 = 140 + 30 * np.sin(2 * np.pi * (ys + xs) /
                           (height + width))      # shape (height,width)
    c0 = np.broadcast_to(c0, (height, width))
    c1 = np.broadcast_to(c1, (height, width))
    original = np.stack([c0, c1, c2], axis=-1)
    original = np.clip(original, 0, 255).astype(np.uint8)

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
    original = np.clip(original.astype(np.int16) +
                       noise, 0, 255).astype(np.uint8)

    # 添加网格线（高频细节）
    # 使用数组切片替代循环绘制网格线（更快且更简洁）
    original[::32, :, :] = 255
    original[:, ::32, :] = 255

    # ==================== 创建增强图像 ====================
    # 模拟AI增强：提高对比度和饱和度，但可能丢失细节
    enhanced = original.copy()

    # 转换到HSV空间以增强饱和度
    # 以 float 进行比例调整，最后一次性转换回 uint8
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 增强对比度
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    l_mean = float(np.mean(l))
    lab[:, :, 0] = np.clip((l - l_mean) * 1.3 + l_mean, 0, 255)
    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 轻微模糊（模拟AI增强可能丢失的高频细节）
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

    # 保存图像（默认到项目根的 test_images 文件夹，支持传入自定义 output_dir）
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "test_images"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / "original.jpg"
    enhanced_path = output_dir / "enhanced.jpg"
    comparison_path = output_dir / "test_images_comparison.jpg"

    # 检查写入结果并在失败时抛出提示
    if not cv2.imwrite(str(original_path), original):
        raise IOError(f"Failed to write image: {original_path}")
    if not cv2.imwrite(str(enhanced_path), enhanced):
        raise IOError(f"Failed to write image: {enhanced_path}")

    print("✓ 测试图像已创建:")
    print(f"  - {os.path.relpath(original_path)} (原始图像)")
    print(f"  - {os.path.relpath(enhanced_path)} (AI增强图像)")

    # 创建对比图
    comparison = np.hstack([original, enhanced])
    if not cv2.imwrite(str(comparison_path), comparison):
        raise IOError(f"Failed to write image: {comparison_path}")
    print(f"  - {os.path.relpath(str(comparison_path))} (对比图)")


if __name__ == "__main__":
    create_test_images()
