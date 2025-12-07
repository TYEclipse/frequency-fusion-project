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
    - 修正最大距离计算：使用到四个角点的最大欧氏距离，保证归一化范围正确
    - 处理小图像和 max_distance==0 的边界情况
    - 在应用非线性函数前将归一化距离裁剪到 [0,1]
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    # 创建网格坐标
    y, x = np.ogrid[:rows, :cols]

    # 计算每个点到中心的距离
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)

    # 计算最大距离（使用到四个角点的最大距离以更精确归一化）
    corner_coords = [(0, 0), (0, cols - 1),
                     (rows - 1, 0), (rows - 1, cols - 1)]
    max_distance = 0.0
    for (r, c) in corner_coords:
        d = np.hypot(r - center_row, c - center_col)
        if d > max_distance:
            max_distance = d

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
    visualize: bool = False,
    fuse_luminance_only: bool = True
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

    新增参数:
        fuse_luminance_only: 若为 True，则仅对色彩空间的通道0（Y 或 L）做频域融合，
                            其余通道直接从 enhanced_img 保留（以保持色彩）。
                            设为 False 则对所有通道都做频域融合（与原行为一致）。

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

    # ==================== 步骤2: 色彩空间转换（保留三通道） ====================
    print(f"[2/6] 转换色彩空间到 {color_space}...")
    if color_space == 'YUV':
        original_space = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
        enhanced_space = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2YUV)
    elif color_space == 'Lab':
        original_space = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)
        enhanced_space = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2Lab)
    else:
        raise ValueError(f"不支持的色彩空间: {color_space}")

    # ==================== 步骤3: 构建频域权重矩阵 ====================
    print(f"[3/6] 构建频域权重矩阵...")
    rows, cols = original_space.shape[:2]
    weight_matrix = create_frequency_weight_matrix((rows, cols), filter_type)

    # 准备融合后三通道容器（保持 uint8）
    fused_space = original_space.copy()

    # 为可视化保存通道0的频谱（原/增强/融合）
    # 只保存通道0 的频谱用于可视化，避免在处理多个通道时占用大量内存
    mag0_orig = None
    mag0_enh = None
    mag0_fused = None

    # 决定需要处理的通道索引（若只融合亮度則仅 c=0）
    channels_to_process = [0] if fuse_luminance_only else [0, 1, 2]

    # ==================== 步骤4: 对每个通道在频域中融合 ====================
    print(f"[4/6] 对通道执行频域融合 ({filter_type})... 处理通道: {channels_to_process}")
    for c in range(3):
        # 如果当前通道不在处理列表，直接从 enhanced 或 original 中选择保留策略
        if c not in channels_to_process:
            # 选择从 enhanced 保留色度更自然（因为增强通常改善色彩）
            fused_space[:, :, c] = enhanced_space[:, :, c]
            continue

        # 提取通道并转为 float32
        orig_chan_f = original_space[:, :, c].astype(np.float32)
        enh_chan_f = enhanced_space[:, :, c].astype(np.float32)

        # DFT 并移位
        dft_orig = cv2.dft(orig_chan_f, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_orig_shift = np.fft.fftshift(dft_orig, axes=(0, 1))

        dft_enh = cv2.dft(enh_chan_f, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_enh_shift = np.fft.fftshift(dft_enh, axes=(0, 1))

        # 幅值与相位
        mag_orig, phase_orig = cv2.cartToPolar(
            dft_orig_shift[:, :, 0], dft_orig_shift[:,
                                                    :, 1], angleInDegrees=False
        )
        mag_enh, _ = cv2.cartToPolar(
            dft_enh_shift[:, :, 0], dft_enh_shift[:,
                                                  :, 1], angleInDegrees=False
        )

        # 融合幅值（增强图低频优先，原图高频保留）
        fused_mag = (weight_matrix * mag_enh +
                     (1.0 - weight_matrix) * mag_orig).astype(np.float32)
        fused_phase = phase_orig.astype(np.float32)

        # 仅在通道0 时保存频谱用于后续可视化，避免无谓内存分配
        if c == 0:
            mag0_orig = mag_orig
            mag0_enh = mag_enh
            mag0_fused = fused_mag

        # 从幅值与相位重建复数频谱（实/虚）
        fused_real = (fused_mag * np.cos(fused_phase)).astype(np.float32)
        fused_imag = (fused_mag * np.sin(fused_phase)).astype(np.float32)
        fused_dft_shifted = np.stack(
            [fused_real, fused_imag], axis=-1).astype(np.float32)

        # 移回原位并逆 DFT
        fused_dft = np.fft.ifftshift(fused_dft_shifted, axes=(0, 1))
        fused_chan = cv2.idft(
            fused_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # 截断并写回 uint8 通道
        fused_space[:, :, c] = np.clip(fused_chan, 0, 255).astype(np.uint8)

    # ==================== 步骤5: 转回 BGR 并保存 ====================
    print(f"[5/6] 合并通道并转换回BGR色彩空间...")
    if color_space == 'YUV':
        fused_bgr = cv2.cvtColor(fused_space, cv2.COLOR_YUV2BGR)
    else:  # Lab
        fused_bgr = cv2.cvtColor(fused_space, cv2.COLOR_Lab2BGR)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 检查写入是否成功
    ok = cv2.imwrite(output_path, fused_bgr)
    if not ok:
        raise IOError(f"无法写入融合图像到: {output_path}")
    print(f"✓ 融合图像已保存到: {output_path}")

    # ==================== 步骤6: 可视化（可选，使用通道0幅值以保持原接口）====================
    if visualize:
        vis_dir = os.path.dirname(output_path) or os.getcwd()
        os.makedirs(vis_dir, exist_ok=True)
        vis_save_path = os.path.join(vis_dir, 'fusion_visualization.png')

        # 兼容原可视化函数：传递通道0 的幅值（若未计算过则从列表获取第一个）
        # 兼容原可视化接口：若未计算到通道0 的频谱，则回退为零矩阵
        if mag0_orig is None:
            mag0_orig = np.zeros((rows, cols), dtype=np.float32)
        if mag0_enh is None:
            mag0_enh = np.zeros((rows, cols), dtype=np.float32)
        if mag0_fused is None:
            mag0_fused = np.zeros((rows, cols), dtype=np.float32)

        visualize_results(
            original_img, enhanced_img, fused_bgr,
            mag0_orig, mag0_enh, mag0_fused,
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
    Visualize intermediate results of fusion process
    """
    fig = plt.figure(figsize=(18, 12))

    # First row: spatial domain image comparison
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    plt.title('Fused Image', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(weight_matrix, cmap='hot')
    plt.title(f'Weight Matrix ({filter_type})', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Second row: frequency magnitude spectra (log scale)
    plt.subplot(3, 4, 5)
    plt.imshow(np.log1p(mag_original), cmap='gray')
    plt.title('Original Magnitude Spectrum (log)', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(np.log1p(mag_enhanced), cmap='gray')
    plt.title('Enhanced Magnitude Spectrum (log)', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(np.log1p(mag_fused), cmap='gray')
    plt.title('Fused Magnitude Spectrum (log)', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 8)
    # show radial profile of the weight matrix
    center = weight_matrix.shape[0] // 2
    profile = weight_matrix[center, :]
    plt.plot(profile, linewidth=2)
    plt.title(f'Weight Profile ({filter_type})', fontsize=12)
    plt.xlabel('Pixels from center')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)

    # Third row: detail crops
    h, w = original_img.shape[:2]
    crop_size = min(h, w) // 3
    y1, y2 = h//2 - crop_size//2, h//2 + crop_size//2
    x1, x2 = w//2 - crop_size//2, w//2 + crop_size//2

    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(original_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    plt.title('Original Details', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(cv2.cvtColor(enhanced_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Details', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(cv2.cvtColor(fused_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    plt.title('Fused Details', fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 12)
    # difference image
    diff = cv2.absdiff(original_img, fused_img)
    plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    plt.title('Original vs Fused Difference', fontsize=12)
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
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()


def compare_filter_types(
    original_path: str,
    enhanced_path: str,
    output_dir: str = '/home/ubuntu'
) -> None:
    """
    Compare fusion results for different weight transition types
    """
    filter_types = ['linear', 'gaussian', 'cosine', 'quadratic']
    results = []

    print("\n" + "="*70)
    print("Compare different weight transition types")
    print("="*70)

    # 读取图像并检查
    original_img = cv2.imread(original_path)
    enhanced_img = cv2.imread(enhanced_path)
    if original_img is None or enhanced_img is None:
        raise FileNotFoundError(
            "compare_filter_types: 无法读取 original 或 enhanced 图像，请检查路径。")

    # 若尺寸不一致，将增强图缩放到与原图相同尺寸（选择合适的插值）
    # 如果尺寸不一致，为了可视化展示我们本地缩放 enhanced_img（不写磁盘）
    if original_img.shape != enhanced_img.shape:
        print(
            f"   图像尺寸不一致，将增强图从 {enhanced_img.shape[:2]} 缩放到 {original_img.shape[:2]}（仅用于展示）")
        # 若增强图更大则用 INTER_AREA 下采样；否则用 INTER_CUBIC 上采样
        if enhanced_img.shape[0] > original_img.shape[0] or enhanced_img.shape[1] > original_img.shape[1]:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        enhanced_img = cv2.resize(
            enhanced_img,
            (original_img.shape[1], original_img.shape[0]),
            interpolation=interp
        )
    # 不再写入临时文件，直接传入原始路径；frequency_fusion 会在读取时做必要的缩放
    enhanced_path_to_use = enhanced_path

    for filter_type in filter_types:
        print(f"\n处理 {filter_type} 过滤器...")
        output_path = f"{output_dir}/fused_{filter_type}.jpg"

        fused_img = frequency_fusion(
            original_path=original_path,
            enhanced_path=enhanced_path_to_use,
            output_path=output_path,
            filter_type=filter_type,
            visualize=False
        )
        results.append((filter_type, fused_img))

    # 新增：RGB 逐像素平均融合（直接在 RGB 空间做平均）
    print("\n处理 rgb_average 平均融合...")
    rgb_avg = ((original_img.astype(np.float32) +
               enhanced_img.astype(np.float32)) / 2.0).astype(np.uint8)
    rgb_avg_path = f"{output_dir}/fused_rgb_average.jpg"
    cv2.imwrite(rgb_avg_path, rgb_avg)
    results.append(('rgb_average', rgb_avg))

    # 创建对比图（扩展列以包含 rgb_average）
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # 显示原图和增强图
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('AI Enhanced Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 权重剖面对比（放在第0行第2列）
    axes[0, 2].set_title('Weight Profile Comparison',
                         fontsize=14, fontweight='bold')
    for filter_type in filter_types:
        weight = create_frequency_weight_matrix(
            original_img.shape[:2], filter_type)
        center = weight.shape[0] // 2
        profile = weight[center, :]
        axes[0, 2].plot(profile, label=filter_type, linewidth=2)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlabel('Pixels from center')
    axes[0, 2].set_ylabel('Weight')

    # 保持第0行第3和第4列为空（或可用于其他可视化扩展）
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')

    # 显示不同过滤器的融合结果（包含新增的 rgb_average）
    for idx, (filter_type, fused_img) in enumerate(results):
        col = idx
        axes[1, col].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
        axes[1, col].set_title(
            f'Fused Result ({filter_type})', fontsize=14, fontweight='bold')
        axes[1, col].axis('off')

    plt.tight_layout()
    comparison_path = f"{output_dir}/filter_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {comparison_path}")
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
