#!/usr/bin/env python3
"""
创建测试图像：模拟原始图像和AI增强图像
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np


def set_random_seed(seed: int | None) -> np.random.Generator:
    """返回一个 numpy Generator（不再设置全局 RNG），便于可复现性且无全局副作用。"""
    # 使用 default_rng 保持现代 RNG 语义，返回 generator 由调用方传递给需要随机性的函数。
    return np.random.default_rng(seed)


def generate_background(height: int, width: int) -> np.ndarray:
    """生成基础背景（三个通道），返回 uint8 图像。"""
    ys = np.arange(height).reshape(height, 1)
    xs = np.arange(width).reshape(1, width)
    c0 = 100 + 50 * np.sin(2 * np.pi * ys / height)   # shape (height, 1)
    c1 = 120 + 40 * np.cos(2 * np.pi * xs / width)    # shape (1, width)
    # broadcasts to (height, width)
    c2 = 140 + 30 * np.sin(2 * np.pi * (ys + xs) / (height + width))
    # 将 c0/c1 显式广播为 (height, width) 后再 stack，避免 shape 不匹配错误
    c0 = np.broadcast_to(c0, (height, width))
    c1 = np.broadcast_to(c1, (height, width))
    bg = np.stack([c0, c1, c2], axis=-1)
    return np.clip(bg, 0, 255).astype(np.uint8)


def add_geometric_shapes(img: np.ndarray):
    """在图像上绘制中频几何图形（就地修改）。"""
    height, width = img.shape[:2]
    cv2.circle(img, (width // 4, height // 4), 60, (80, 150, 200), -1)
    cv2.circle(img, (3 * width // 4, height // 4), 60, (200, 100, 80), -1)
    cv2.rectangle(img, (width // 4 - 40, 3 * height // 4 - 40),
                  (width // 4 + 40, 3 * height // 4 + 40), (100, 200, 100), -1)
    cv2.rectangle(img, (3 * width // 4 - 40, 3 * height // 4 - 40),
                  (3 * width // 4 + 40, 3 * height // 4 + 40), (200, 200, 50), -1)


def add_text(img: np.ndarray, text: str = "ORIGINAL"):
    """在图像中心添加文字（就地修改）。"""
    height, width = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (width // 2 - 120, height // 2),
                font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)


def add_texture_noise(img: np.ndarray, noise_range: tuple[int, int] = (-20, 20), rng: np.random.Generator | None = None):
    """向图像添加高频随机噪声（就地修改）。"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.integers(
        noise_range[0], noise_range[1], size=img.shape, dtype=np.int16)
    img[:] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_grid_lines(img: np.ndarray, step: int = 32, color: int = 255):
    """添加网格线（就地修改）。"""
    img[::step, :, :] = color
    img[:, ::step, :] = color


def enhance_saturation_value(img: np.ndarray, sat_scale: float = 1.5, val_scale: float = 1.2) -> np.ndarray:
    """在 HSV 空间调整饱和度与明度并返回新图像（不修改原图）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_contrast_lab(img: np.ndarray, contrast_scale: float = 1.3) -> np.ndarray:
    """在 LAB 空间对亮度通道进行线性对比度增强并返回新图像。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    l_mean = float(np.mean(l))
    lab[:, :, 0] = np.clip((l - l_mean) * contrast_scale + l_mean, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_blur(img: np.ndarray, ksize: tuple[int, int] = (3, 3), sigma: float = 0.5) -> np.ndarray:
    """对图像应用高斯模糊并返回新图像。"""
    return cv2.GaussianBlur(img, ksize, sigma)


def simulate_ai_enhancement(img: np.ndarray,
                            sat_scale: float = 1.5,
                            val_scale: float = 1.2,
                            contrast_scale: float = 1.3,
                            blur_ksize: tuple[int, int] = (3, 3),
                            blur_sigma: float = 0.5) -> np.ndarray:
    """模拟 AI 增强流水线：饱和度/明度 -> 对比度 -> 模糊。返回增强后的图像。"""
    enhanced = enhance_saturation_value(img, sat_scale, val_scale)
    enhanced = enhance_contrast_lab(enhanced, contrast_scale)
    enhanced = apply_blur(enhanced, blur_ksize, blur_sigma)
    return enhanced


def ensure_output_dir(path: str | Path | None) -> Path:
    """确保并返回输出目录 Path。"""
    if path is None:
        output_dir = Path(__file__).parent.parent / "test_images"
    else:
        output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_image(path: Path, img: np.ndarray):
    """保存单张图像，保存失败时抛出 IOError。"""
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Failed to write image: {path}")


def create_comparison_image(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """水平拼接原始与增强图以便对比（返回新图像）。"""
    return np.hstack([original, enhanced])


def create_test_images(height: int = 512, width: int = 512, seed: int | None = None, output_dir: str | None = None):
    """
    创建一对测试图像并保存。
    注意：以下函数会就地修改 original（add_geometric_shapes / add_text / add_texture_noise / add_grid_lines）。
    """
    # 设置并获取 RNG（不影响全局 np.random 状态）
    rng = set_random_seed(seed)

    # 生成原始图像并逐步添加细节（会就地修改 original）
    original = generate_background(height, width)
    add_geometric_shapes(original)
    add_text(original)
    add_texture_noise(original, rng=rng)
    add_grid_lines(original)

    # 模拟增强（返回新图）
    enhanced = simulate_ai_enhancement(original)

    # 在拼接前检查形状/通道一致性，避免 np.hstack 报错
    if original.shape != enhanced.shape:
        logging.warning("original 与 enhanced 尺寸或通道不匹配，尝试调整为相同 dtype 和通道数。")
        # 强制转换为相同 dtype/通道（简单策略）
        h = min(original.shape[0], enhanced.shape[0])
        w = min(original.shape[1], enhanced.shape[1])
        original = original[:h, :w]
        enhanced = enhanced[:h, :w]

    # 准备输出目录并保存
    output_dir = ensure_output_dir(output_dir)
    original_path = output_dir / "original.jpg"
    enhanced_path = output_dir / "enhanced.jpg"
    comparison_path = output_dir / "test_images_comparison.jpg"

    save_image(original_path, original)
    save_image(enhanced_path, enhanced)

    logging.info("✓ 测试图像已创建:")
    logging.info(f"  - {os.path.relpath(os.fspath(original_path))} (原始图像)")
    logging.info(f"  - {os.path.relpath(os.fspath(enhanced_path))} (AI增强图像)")

    comparison = create_comparison_image(original, enhanced)
    save_image(comparison_path, comparison)
    logging.info(f"  - {os.path.relpath(os.fspath(comparison_path))} (对比图)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="生成测试图像（original + enhanced + comparison）")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    create_test_images(height=args.height, width=args.width,
                       seed=args.seed, output_dir=args.output_dir)
