#!/usr/bin/env python3
"""
基于频域的自适应图像融合算法
作者：资深计算机视觉工程师
功能：在频域中平衡AI增强图像的色彩与原图的细节
核心思想：增强图主导低频信号（色彩），原图主导高频信号（细节）
"""

import os
from typing import Literal, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_frequency_weight_matrix(
    shape: Tuple[int, int],
    filter_type: Literal['linear', 'gaussian',
                         'cosine', 'quadratic'] = 'linear'
) -> np.ndarray:
    """
    构建频域权重矩阵，从中心（低频）到边缘（高频）的过渡
    - 处理小图像和 max_distance==0 的边界情况
    - 在应用非线性函数前将归一化距离裁剪到 [0,1]
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    # 创建网格坐标
    y, x = np.ogrid[:rows, :cols]

    # 计算每个点到中心的距离
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)

    # 计算最大距离（对角线的一半），保证非零
    max_distance = np.hypot(center_row, center_col)
    if max_distance == 0:
        return np.ones((rows, cols), dtype=np.float32)

    # 归一化距离到 [0, 1] 范围，并裁剪，避免后续函数异常
    normalized_distance = np.clip(distance / max_distance, 0.0, 1.0)

    # 根据不同的过滤器类型计算权重
    if filter_type == 'linear':
        weight = 1.0 - normalized_distance

    elif filter_type == 'gaussian':
        # sigma 在归一化坐标系中，允许调整为稳健值
        sigma = 0.5
        weight = np.exp(-(normalized_distance**2) / (2.0 * sigma**2))

    elif filter_type == 'cosine':
        # 对归一化距离在 [0,1] 上使用余弦平滑
        weight = 0.5 * (1.0 + np.cos(np.pi * normalized_distance))

    elif filter_type == 'quadratic':
        weight = (1.0 - normalized_distance)**2

    else:
        raise ValueError(f"不支持的过滤器类型: {filter_type}")

    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    return weight


def frequency_fusion(
    original_path: str,
    enhanced_path: str,
    output_path: str,
    filter_type: Literal['linear', 'gaussian',
                         'cosine', 'quadratic'] = 'linear',
    color_space: Literal['YUV', 'Lab'] = 'YUV',
    visualize: bool = False
) -> np.ndarray:
    """
    基于频域的图像融合算法

    参数:
        original_path: 原始图像路径
        enhanced_path: AI增强图像路径
        output_path: 输出融合图像路径
        filter_type: 频域权重过渡方式 ('linear', 'gaussian', 'cosine', 'quadratic')
        color_space: 色彩空间 ('YUV' 或 'Lab')
        visualize: 是否可视化中间结果

    返回:
        融合后的BGR图像
    """

    # ==================== 步骤1: 读取并预处理图像 ====================
    print(f"[1/6] 读取图像...")
    original_img = cv2.imread(original_path)
    enhanced_img = cv2.imread(enhanced_path)

    if original_img is None:
        raise FileNotFoundError(f"无法读取原始图像: {original_path}")
    if enhanced_img is None:
        raise FileNotFoundError(f"无法读取增强图像: {enhanced_path}")

    # 确保两张图像尺寸一致
    if original_img.shape != enhanced_img.shape:
        print(
            f"   图像尺寸不一致，将增强图从 {enhanced_img.shape[:2]} 缩放到 {original_img.shape[:2]}")
        enhanced_img = cv2.resize(enhanced_img,
                                  (original_img.shape[1],
                                   original_img.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

    # ==================== 步骤2: 色彩空间转换 ====================
    print(f"[2/6] 转换色彩空间到 {color_space}...")
    if color_space == 'YUV':
        # 转换到YUV色彩空间
        original_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
        enhanced_yuv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2YUV)

        # 提取亮度通道（Y）
        original_luma = original_yuv[:, :, 0].astype(np.float32)
        enhanced_luma = enhanced_yuv[:, :, 0].astype(np.float32)

    elif color_space == 'Lab':
        # 转换到Lab色彩空间
        original_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)
        enhanced_lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2Lab)

        # 提取亮度通道（L）
        original_luma = original_lab[:, :, 0].astype(np.float32)
        enhanced_luma = enhanced_lab[:, :, 0].astype(np.float32)

    else:
        raise ValueError(f"不支持的色彩空间: {color_space}")

    # ==================== 步骤3: 傅里叶变换到频域 ====================
    print(f"[3/6] 执行傅里叶变换...")

    # 确保亮度通道为 float32（cv2.dft 需要）
    original_luma_f = original_luma.astype(np.float32)
    enhanced_luma_f = enhanced_luma.astype(np.float32)

    # 使用 cv2.dft 得到复数两通道表示
    dft_original = cv2.dft(original_luma_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_original_shifted = np.fft.fftshift(dft_original, axes=(0, 1))

    dft_enhanced = cv2.dft(enhanced_luma_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_enhanced_shifted = np.fft.fftshift(dft_enhanced, axes=(0, 1))

    # 使用 cartToPolar 更高效地获取幅值与相位（保持 float32）
    magnitude_original, phase_original = cv2.cartToPolar(
        dft_original_shifted[:, :, 0], dft_original_shifted[:,
                                                            :, 1], angleInDegrees=False
    )
    # 只需要增强图的幅度（相位不使用），因此丢弃相位输出以节省内存/计算
    magnitude_enhanced, _ = cv2.cartToPolar(
        dft_enhanced_shifted[:, :, 0], dft_enhanced_shifted[:,
                                                            :, 1], angleInDegrees=False
    )

    # ==================== 步骤4: 构建频域权重矩阵并融合 ====================
    print(f"[4/6] 构建 {filter_type} 权重矩阵并融合频谱...")

    # 创建权重矩阵（中心为1，边缘为0）
    weight_matrix = create_frequency_weight_matrix(
        shape=original_luma.shape,
        filter_type=filter_type
    )

    fused_magnitude = (weight_matrix * magnitude_enhanced +
                       (1.0 - weight_matrix) * magnitude_original).astype(np.float32)

    fused_phase = phase_original.astype(np.float32)

    # ==================== 步骤5: 逆傅里叶变换回空间域 ====================
    print(f"[5/6] 执行逆傅里叶变换...")

    # 用幅度和相位重建实部与虚部，确保 float32
    fused_real = (fused_magnitude * np.cos(fused_phase)).astype(np.float32)
    fused_imag = (fused_magnitude * np.sin(fused_phase)).astype(np.float32)
    fused_dft_shifted = np.stack(
        [fused_real, fused_imag], axis=-1).astype(np.float32)

    # 将零频分量移回原位（显式 axes）
    fused_dft = np.fft.ifftshift(fused_dft_shifted, axes=(0, 1))

    # 执行逆DFT，得到单通道实数（float32）
    fused_luma = cv2.idft(fused_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 裁剪到有效范围并转换为 uint8
    fused_luma = np.clip(fused_luma, 0, 255).astype(np.uint8)

    # ==================== 步骤6: 合并通道并转回BGR ====================
    print(f"[6/6] 合并通道并转换回BGR色彩空间...")

    # 合并融合后的亮度通道和原始的色度通道，直接在原始数组副本上赋值以保留类型与范围
    if color_space == 'YUV':
        fused_yuv = original_yuv.copy()
        fused_yuv[:, :, 0] = fused_luma
        fused_bgr = cv2.cvtColor(fused_yuv, cv2.COLOR_YUV2BGR)

    elif color_space == 'Lab':
        fused_lab = original_lab.copy()
        fused_lab[:, :, 0] = fused_luma
        fused_bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 保存结果
    cv2.imwrite(output_path, fused_bgr)
    print(f"✓ 融合图像已保存到: {output_path}")

    # ==================== 可视化（可选）====================
    if visualize:
        # 计算可视化保存路径：优先使用 output_path 的目录，否则当前工作目录
        vis_dir = os.path.dirname(output_path) if output_path else os.getcwd()
        if vis_dir == "":
            vis_dir = os.getcwd()
        vis_save_path = os.path.join(vis_dir, 'fusion_visualization.png')
        # 确保目录存在
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        visualize_results(
            original_img, enhanced_img, fused_bgr,
            magnitude_original, magnitude_enhanced, fused_magnitude,
            weight_matrix, filter_type,
            save_path=vis_save_path
        )

    return fused_bgr


def visualize_results(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    fused_img: np.ndarray,
    mag_original: np.ndarray,
    mag_enhanced: np.ndarray,
    mag_fused: np.ndarray,
    weight_matrix: np.ndarray,
    filter_type: str,
    save_path: Optional[str] = None
) -> None:
    """
    可视化融合过程的中间结果
    """
    fig = plt.figure(figsize=(18, 12))

    # 第一行：空间域图像对比
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像 (Original)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.title('增强图像 (Enhanced)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    plt.title('融合图像 (Fused)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(weight_matrix, cmap='hot')
    plt.title(f'权重矩阵 ({filter_type})', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 第二行：频域幅度谱（对数尺度）
    plt.subplot(3, 4, 5)
    plt.imshow(np.log1p(mag_original), cmap='gray')
    plt.title('原图幅度谱 (log)', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(np.log1p(mag_enhanced), cmap='gray')
    plt.title('增强图幅度谱 (log)', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(np.log1p(mag_fused), cmap='gray')
    plt.title('融合幅度谱 (log)', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 8)
    # 显示权重矩阵的径向剖面
    center = weight_matrix.shape[0] // 2
    profile = weight_matrix[center, :]
    plt.plot(profile, linewidth=2)
    plt.title(f'权重剖面 ({filter_type})', fontsize=12)
    plt.xlabel('距离中心的像素')
    plt.ylabel('权重值')
    plt.grid(True, alpha=0.3)

    # 第三行：细节对比（裁剪中心区域）
    h, w = original_img.shape[:2]
    crop_size = min(h, w) // 3
    y1, y2 = h//2 - crop_size//2, h//2 + crop_size//2
    x1, x2 = w//2 - crop_size//2, w//2 + crop_size//2

    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(original_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    plt.title('原图细节', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(cv2.cvtColor(enhanced_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    plt.title('增强图细节', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(cv2.cvtColor(fused_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    plt.title('融合图细节', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 12)
    # 显示差异图
    diff = cv2.absdiff(original_img, fused_img)
    plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    plt.title('原图与融合图差异', fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    # 计算默认保存路径（如果未提供）
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'fusion_visualization.png')
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存到: {save_path}")
    plt.close()


def compare_filter_types(
    original_path: str,
    enhanced_path: str,
    output_dir: str = '/home/ubuntu'
) -> None:
    """
    对比不同权重过渡方式的融合效果
    """
    filter_types = ['linear', 'gaussian', 'cosine', 'quadratic']
    results = []

    print("\n" + "="*70)
    print("对比不同权重过渡方式的融合效果")
    print("="*70)

    # 读取图像并检查
    original_img = cv2.imread(original_path)
    enhanced_img = cv2.imread(enhanced_path)
    if original_img is None or enhanced_img is None:
        raise FileNotFoundError(
            "compare_filter_types: 无法读取 original 或 enhanced 图像，请检查路径。")

    for filter_type in filter_types:
        print(f"\n处理 {filter_type} 过滤器...")
        output_path = f"{output_dir}/fused_{filter_type}.jpg"

        fused_img = frequency_fusion(
            original_path=original_path,
            enhanced_path=enhanced_path,
            output_path=output_path,
            filter_type=filter_type,
            visualize=False
        )
        results.append((filter_type, fused_img))

    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 显示原图和增强图
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('AI增强图像', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 显示权重矩阵对比
    axes[0, 2].set_title('权重剖面对比', fontsize=14, fontweight='bold')
    for filter_type in filter_types:
        weight = create_frequency_weight_matrix(
            original_img.shape[:2], filter_type)
        center = weight.shape[0] // 2
        profile = weight[center, :]
        axes[0, 2].plot(profile, label=filter_type, linewidth=2)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlabel('距离中心的像素')
    axes[0, 2].set_ylabel('权重值')

    # 显示不同过滤器的融合结果
    for idx, (filter_type, fused_img) in enumerate(results):
        col = idx
        axes[1, col].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
        axes[1, col].set_title(
            f'融合结果 ({filter_type})', fontsize=14, fontweight='bold')
        axes[1, col].axis('off')

    plt.tight_layout()
    comparison_path = f"{output_dir}/filter_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比结果已保存到: {comparison_path}")
    plt.close()


# ==================== 示例代码 ====================
if __name__ == "__main__":
    """
    使用示例
    """

    # 示例1: 基本使用（线性权重）
    print("\n" + "="*70)
    print("示例1: 基本频域融合（线性权重）")
    print("="*70)

    # 注意：请替换为实际的图像路径
    original_path = "/home/ubuntu/original.jpg"
    enhanced_path = "/home/ubuntu/enhanced.jpg"
    output_path = "/home/ubuntu/fused_linear.jpg"

    # 检查文件是否存在（无需重复导入 os，已在文件顶部导入）
    if os.path.exists(original_path) and os.path.exists(enhanced_path):
        fused_img = frequency_fusion(
            original_path=original_path,
            enhanced_path=enhanced_path,
            output_path=output_path,
            filter_type='linear',
            color_space='YUV',
            visualize=True
        )

        # 示例2: 对比不同权重过渡方式
        print("\n" + "="*70)
        print("示例2: 对比不同权重过渡方式")
        print("="*70)

        compare_filter_types(
            original_path=original_path,
            enhanced_path=enhanced_path,
            output_dir='/home/ubuntu'
        )
    else:
        print("\n⚠ 测试图像不存在，自动创建演示图像进行测试...")
        # 创建演示原图：渐变 + 圆形高亮
        h, w = 512, 768
        base = np.tile(np.linspace(30, 220, w, dtype=np.uint8), (h, 1))
        original_demo = cv2.merge([base, base, base])
        cv2.circle(original_demo, (w//2, h//2),
                   min(h, w)//6, (200, 180, 60), -1)

        # 创建增强图：对原图做局部增强（直方图均衡化亮度通道）
        enhanced_demo = original_demo.copy()
        yuv = cv2.cvtColor(enhanced_demo, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced_demo = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # 保存示例图像并运行融合
        demo_dir = os.path.dirname(output_path) or os.getcwd()
        os.makedirs(demo_dir, exist_ok=True)
        cv2.imwrite(original_path, original_demo)
        cv2.imwrite(enhanced_path, enhanced_demo)

        fused_img = frequency_fusion(
            original_path=original_path,
            enhanced_path=enhanced_path,
            output_path=output_path,
            filter_type='linear',
            color_space='YUV',
            visualize=True
        )

        compare_filter_types(
            original_path=original_path,
            enhanced_path=enhanced_path,
            output_dir=demo_dir
        )
