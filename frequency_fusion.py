#!/usr/bin/env python3
"""
基于频域的自适应图像融合算法
作者：资深计算机视觉工程师
功能：在频域中平衡AI增强图像的色彩与原图的细节
核心思想：增强图主导低频信号（色彩），原图主导高频信号（细节）
"""

import os
from typing import List, Literal, Optional, Tuple

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


# 新增：读取并预处理图像（尺寸对齐）
def read_and_preprocess_images(original_path: str, enhanced_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取两张BGR图像，若尺寸不同则把增强图缩放到原图尺寸并返回两个图像。
    """
    orig = cv2.imread(original_path)
    enh = cv2.imread(enhanced_path)
    if orig is None:
        raise FileNotFoundError(f"无法读取原始图像: {original_path}")
    if enh is None:
        raise FileNotFoundError(f"无法读取增强图像: {enhanced_path}")

    if orig.shape != enh.shape:
        # 若增强图更大则用 AREA 下采样；否则用 CUBIC 上采样
        if enh.shape[0] > orig.shape[0] or enh.shape[1] > orig.shape[1]:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        enh = cv2.resize(
            enh, (orig.shape[1], orig.shape[0]), interpolation=interp)
    return orig, enh


# 新增：色彩空间转换（返回两张同色彩空间的图像）
def convert_color_space_pair(original_bgr: np.ndarray, enhanced_bgr: np.ndarray, color_space: Literal['YUV', 'Lab']) -> Tuple[np.ndarray, np.ndarray]:
    if color_space == 'YUV':
        return (cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YUV),
                cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2YUV))
    elif color_space == 'Lab':
        return (cv2.cvtColor(original_bgr, cv2.COLOR_BGR2Lab),
                cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2Lab))
    else:
        raise ValueError(f"不支持的色彩空间: {color_space}")


# 频域工具：计算 DFT 并移位
def compute_dft_shift(channel_f: np.ndarray) -> np.ndarray:
    """
    使用 numpy.fft 计算二维 DFT 并将零频移动到中心处。
    返回 complex64 的频域数组（已 shift）。
    """
    # 确保输入为 float32
    arr = np.asarray(channel_f, dtype=np.float32)
    # 计算 FFT（结果为 complex128，转换为 complex64 以节省内存）
    dft = np.fft.fft2(arr)
    dft_shifted = np.fft.fftshift(dft).astype(np.complex64)
    return dft_shifted


# 频域工具：从移位后的复频谱计算幅值和相位
def dft_to_mag_phase(dft_shifted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从移位后的复频谱返回幅值与相位（均为 float32）。
    """
    # mag: magnitude, phase: angle
    mag = np.abs(dft_shifted).astype(np.float32)
    phase = np.angle(dft_shifted).astype(np.float32)
    return mag, phase


# 频域工具：由幅值与相位重建空域通道
def mag_phase_to_spatial_channel(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    由幅值与相位在频域重建空域通道（使用 numpy.ifft2）。
    返回与原通道相同大小的 float32 实数数组（未裁剪）。
    """
    # 组合为复频谱（complex64）
    comp = (mag.astype(np.float32) * np.exp(1j *
            phase.astype(np.float32))).astype(np.complex64)
    # 将频谱从中心移回原位
    comp_unshift = np.fft.ifftshift(comp)
    # 逆变换（返回 complex，取实部）
    spatial_complex = np.fft.ifft2(comp_unshift)
    spatial = np.real(spatial_complex).astype(np.float32)
    return spatial


# 大块逻辑：对指定通道执行频域融合，返回 fused_space 及用于可视化的通道0频谱
def fuse_channels_in_frequency_domain(original_space: np.ndarray,
                                      enhanced_space: np.ndarray,
                                      weight_matrix: np.ndarray,
                                      channels_to_process: List[int]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    rows, cols = original_space.shape[:2]
    fused_space = original_space.copy()
    mag0_orig = mag0_enh = mag0_fused = None

    for c in range(3):
        if c not in channels_to_process:
            fused_space[:, :, c] = enhanced_space[:, :, c]
            continue

        orig_chan_f = original_space[:, :, c].astype(np.float32)
        enh_chan_f = enhanced_space[:, :, c].astype(np.float32)

        dft_orig_shift = compute_dft_shift(orig_chan_f)
        dft_enh_shift = compute_dft_shift(enh_chan_f)

        mag_orig, phase_orig = dft_to_mag_phase(dft_orig_shift)
        mag_enh, _ = dft_to_mag_phase(dft_enh_shift)

        fused_mag = (weight_matrix * mag_enh + (1.0 - weight_matrix)
                     * mag_orig).astype(np.float32)
        fused_phase = phase_orig.astype(np.float32)

        if c == 0:
            mag0_orig = mag_orig
            mag0_enh = mag_enh
            mag0_fused = fused_mag

        fused_chan = mag_phase_to_spatial_channel(fused_mag, fused_phase)
        fused_space[:, :, c] = np.clip(fused_chan, 0, 255).astype(np.uint8)

    return fused_space, mag0_orig, mag0_enh, mag0_fused


# 保存工具（保证目录）
def save_fused_image(fused_bgr: np.ndarray, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(output_path, fused_bgr)
    if not ok:
        raise IOError(f"无法写入融合图像到: {output_path}")


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
    original_img, enhanced_img = read_and_preprocess_images(
        original_path, enhanced_path)

    # ==================== 步骤2: 色彩空间转换（保留三通道） ====================
    print(f"[2/6] 转换色彩空间到 {color_space}...")
    original_space, enhanced_space = convert_color_space_pair(
        original_img, enhanced_img, color_space)

    # ==================== 步骤3: 构建频域权重矩阵 ====================
    print(f"[3/6] 构建频域权重矩阵...")
    rows, cols = original_space.shape[:2]
    weight_matrix = create_frequency_weight_matrix((rows, cols), filter_type)

    # 决定需要处理的通道索引（若只融合亮度則仅 c=0）
    channels_to_process = [0] if fuse_luminance_only else [0, 1, 2]

    # ==================== 步骤4: 对每个通道在频域中融合 ====================
    print(f"[4/6] 对通道执行频域融合 ({filter_type})... 处理通道: {channels_to_process}")
    fused_space, mag0_orig, mag0_enh, mag0_fused = fuse_channels_in_frequency_domain(
        original_space, enhanced_space, weight_matrix, channels_to_process
    )

    # ==================== 步骤5: 转回 BGR 并保存 ====================
    print(f"[5/6] 合并通道并转换回BGR色彩空间...")
    if color_space == 'YUV':
        fused_bgr = cv2.cvtColor(fused_space, cv2.COLOR_YUV2BGR)
    else:  # Lab
        fused_bgr = cv2.cvtColor(fused_space, cv2.COLOR_Lab2BGR)

    save_fused_image(fused_bgr, output_path)
    print(f"✓ 融合图像已保存到: {output_path}")

    # ==================== 步骤6: 可视化（可选，使用通道0幅值以保持原接口）====================
    if visualize:
        vis_dir = os.path.dirname(output_path) or os.getcwd()
        os.makedirs(vis_dir, exist_ok=True)
        vis_save_path = os.path.join(vis_dir, 'fusion_visualization.png')

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


# 可视化拆分：绘制各行的子函数
def _plot_spatial_row(axs, original_img, enhanced_img, fused_img, weight_matrix, filter_type):
    axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Enhanced Image', fontsize=12, fontweight='bold')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Fused Image', fontsize=12, fontweight='bold')
    axs[2].axis('off')

    # 修复：保留 imshow 的返回值作为 mappable，并作为 colorbar 的第一个参数
    im = axs[3].imshow(weight_matrix, cmap='hot')
    axs[3].set_title(
        f'Weight Matrix ({filter_type})', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
    axs[3].axis('off')


def _plot_frequency_row(axs, mag_original, mag_enhanced, mag_fused, weight_matrix, filter_type):
    axs[0].imshow(np.log1p(mag_original), cmap='gray')
    axs[0].set_title('Original Magnitude Spectrum (log)', fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(np.log1p(mag_enhanced), cmap='gray')
    axs[1].set_title('Enhanced Magnitude Spectrum (log)', fontsize=12)
    axs[1].axis('off')

    axs[2].imshow(np.log1p(mag_fused), cmap='gray')
    axs[2].set_title('Fused Magnitude Spectrum (log)', fontsize=12)
    axs[2].axis('off')

    center = weight_matrix.shape[0] // 2
    profile = weight_matrix[center, :]
    axs[3].plot(profile, linewidth=2)
    axs[3].set_title(f'Weight Profile ({filter_type})', fontsize=12)
    axs[3].set_xlabel('Pixels from center')
    axs[3].set_ylabel('Weight')
    axs[3].grid(True, alpha=0.3)


def _plot_detail_row(axs, original_img, enhanced_img, fused_img):
    h, w = original_img.shape[:2]
    crop_size = min(h, w) // 3
    y1, y2 = h//2 - crop_size//2, h//2 + crop_size//2
    x1, x2 = w//2 - crop_size//2, w//2 + crop_size//2

    axs[0].imshow(cv2.cvtColor(original_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Details', fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(enhanced_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    axs[1].set_title('Enhanced Details', fontsize=12)
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(fused_img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    axs[2].set_title('Fused Details', fontsize=12)
    axs[2].axis('off')

    diff = cv2.absdiff(original_img, fused_img)
    axs[3].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axs[3].set_title('Original vs Fused Difference', fontsize=12)
    axs[3].axis('off')


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

    # 第一行：空间域比较（4 子图）
    axs1 = [plt.subplot2grid((3, 4), (0, i)) for i in range(4)]
    _plot_spatial_row(axs1, original_img, enhanced_img,
                      fused_img, weight_matrix, filter_type)

    # 第二行：频域幅值（4 子图）
    axs2 = [plt.subplot2grid((3, 4), (1, i)) for i in range(4)]
    _plot_frequency_row(axs2, mag_original, mag_enhanced,
                        mag_fused, weight_matrix, filter_type)

    # 第三行：细节裁剪（4 子图）
    axs3 = [plt.subplot2grid((3, 4), (2, i)) for i in range(4)]
    _plot_detail_row(axs3, original_img, enhanced_img, fused_img)

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


def _run_fusions_for_filter_types(original_path: str, enhanced_path: str, output_dir: str, filter_types: List[str]) -> List[Tuple[str, np.ndarray]]:
    """
    对给定的 filter_types 列表逐一调用 frequency_fusion（visualize=False），
    将每个结果 (filter_type, fused_img) 收集并返回。
    """
    results: List[Tuple[str, np.ndarray]] = []
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
    return results


def _compute_and_save_rgb_average(original_img: np.ndarray, enhanced_img: np.ndarray, output_dir: str) -> Tuple[np.ndarray, str]:
    """
    计算 original 与 enhanced 的 RGB 逐像素平均并保存到 output_dir。
    返回 (rgb_avg_image, saved_path)
    """
    rgb_avg = ((original_img.astype(np.float32) +
               enhanced_img.astype(np.float32)) / 2.0).astype(np.uint8)
    rgb_avg_path = f"{output_dir}/fused_rgb_average.jpg"
    cv2.imwrite(rgb_avg_path, rgb_avg)
    return rgb_avg, rgb_avg_path


def _create_and_save_comparison_figure(original_img: np.ndarray,
                                       enhanced_img: np.ndarray,
                                       results: List[Tuple[str, np.ndarray]],
                                       output_dir: str,
                                       filter_types: List[str]) -> str:
    """
    根据 results 绘制对比图并保存。保持原有布局：2 行 x 5 列（第一行：原图、增强图、权重剖面、空、空；第二行：各融合结果）。
    返回保存路径。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

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

    # 保持第0行第3和第4列为空
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')

    # 显示不同过滤器的融合结果（results 顺序应包含 filter_types 后跟 rgb_average）
    for idx, (_filter_type, fused_img) in enumerate(results):
        col = idx
        # 保证不越界（本函数设计与原来一致，默认 results 长度 <=5）
        if col < 5:
            axes[1, col].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
            axes[1, col].set_title(
                f'Fused Result ({_filter_type})', fontsize=14, fontweight='bold')
            axes[1, col].axis('off')

    plt.tight_layout()
    comparison_path = f"{output_dir}/filter_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Comparison saved to: {comparison_path}")
    return comparison_path


def compare_filter_types(
    original_path: str,
    enhanced_path: str,
    output_dir: str = '/home/ubuntu'
) -> None:
    """
    拆分后的 compare_filter_types：负责协调流程，具体子任务由 _run_fusions_for_filter_types 等完成。
    保持原有对外行为与保存路径一致。
    """
    filter_types = ['linear', 'gaussian', 'cosine', 'quadratic']
    print("\n" + "="*70)
    print("Compare different weight transition types")
    print("="*70)

    # 1) 读取并对齐图像用于展示与平均计算（不改变传入 frequency_fusion 的路径）
    original_img, enhanced_img = read_and_preprocess_images(
        original_path, enhanced_path)

    # 2) 运行频域融合（对每种 filter）
    results = _run_fusions_for_filter_types(
        original_path, enhanced_path, output_dir, filter_types)

    # 3) 计算并保存 RGB 平均，加入结果列表
    rgb_avg_img, _ = _compute_and_save_rgb_average(
        original_img, enhanced_img, output_dir)
    results.append(('rgb_average', rgb_avg_img))

    # 4) 绘制并保存对比图
    _create_and_save_comparison_figure(
        original_img, enhanced_img, results, output_dir, filter_types)


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
