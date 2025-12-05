#!/usr/bin/env python3
"""
基于频域的自适应图像融合算法
作者：资深计算机视觉工程师
功能：在频域中平衡AI增强图像的色彩与原图的细节
核心思想：增强图主导低频信号（色彩），原图主导高频信号（细节）
"""

import cv2
import numpy as np
from typing import Tuple, Literal
import matplotlib.pyplot as plt


def create_frequency_weight_matrix(
    shape: Tuple[int, int],
    filter_type: Literal['linear', 'gaussian', 'cosine', 'quadratic'] = 'linear'
) -> np.ndarray:
    """
    构建频域权重矩阵，从中心（低频）到边缘（高频）的过渡
    
    参数:
        shape: 矩阵尺寸 (rows, cols)
        filter_type: 权重过渡方式
            - 'linear': 线性递减（默认）
            - 'gaussian': 高斯分布递减
            - 'cosine': 余弦平滑递减
            - 'quadratic': 二次函数递减
    
    返回:
        权重矩阵，中心为1.0，边缘为0.0
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # 创建网格坐标
    y, x = np.ogrid[:rows, :cols]
    
    # 计算每个点到中心的距离
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    
    # 计算最大距离（对角线的一半）
    max_distance = np.sqrt(center_row**2 + center_col**2)
    
    # 归一化距离到 [0, 1] 范围
    normalized_distance = distance / max_distance
    
    # 根据不同的过滤器类型计算权重
    if filter_type == 'linear':
        # 线性递减：W = 1 - d
        weight = 1.0 - normalized_distance
        
    elif filter_type == 'gaussian':
        # 高斯递减：W = exp(-d^2 / (2*sigma^2))
        sigma = 0.5  # 控制高斯分布的宽度
        weight = np.exp(-(normalized_distance**2) / (2 * sigma**2))
        
    elif filter_type == 'cosine':
        # 余弦平滑递减：W = 0.5 * (1 + cos(pi * d))
        weight = 0.5 * (1.0 + np.cos(np.pi * normalized_distance))
        
    elif filter_type == 'quadratic':
        # 二次函数递减：W = (1 - d)^2
        weight = (1.0 - normalized_distance)**2
        
    else:
        raise ValueError(f"不支持的过滤器类型: {filter_type}")
    
    # 确保权重在 [0, 1] 范围内
    weight = np.clip(weight, 0.0, 1.0)
    
    return weight


def frequency_fusion(
    original_path: str,
    enhanced_path: str,
    output_path: str,
    filter_type: Literal['linear', 'gaussian', 'cosine', 'quadratic'] = 'linear',
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
        print(f"   图像尺寸不一致，将增强图从 {enhanced_img.shape[:2]} 缩放到 {original_img.shape[:2]}")
        enhanced_img = cv2.resize(enhanced_img, 
                                  (original_img.shape[1], original_img.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
    
    # ==================== 步骤2: 色彩空间转换 ====================
    print(f"[2/6] 转换色彩空间到 {color_space}...")
    if color_space == 'YUV':
        # 转换到YUV色彩空间
        original_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
        enhanced_yuv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2YUV)
        
        # 提取亮度通道（Y）和色度通道（U, V）
        original_luma = original_yuv[:, :, 0].astype(np.float32)
        enhanced_luma = enhanced_yuv[:, :, 0].astype(np.float32)
        chroma_channels = original_yuv[:, :, 1:].astype(np.float32)  # 保留原图的色度
        
    elif color_space == 'Lab':
        # 转换到Lab色彩空间
        original_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)
        enhanced_lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2Lab)
        
        # 提取亮度通道（L）和色度通道（a, b）
        original_luma = original_lab[:, :, 0].astype(np.float32)
        enhanced_luma = enhanced_lab[:, :, 0].astype(np.float32)
        chroma_channels = original_lab[:, :, 1:].astype(np.float32)  # 保留原图的色度
    
    else:
        raise ValueError(f"不支持的色彩空间: {color_space}")
    
    # ==================== 步骤3: 傅里叶变换到频域 ====================
    print(f"[3/6] 执行傅里叶变换...")
    
    # 对原始图像的亮度通道进行DFT
    dft_original = cv2.dft(original_luma, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_original_shifted = np.fft.fftshift(dft_original)  # 将零频分量移到中心
    
    # 对增强图像的亮度通道进行DFT
    dft_enhanced = cv2.dft(enhanced_luma, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_enhanced_shifted = np.fft.fftshift(dft_enhanced)
    
    # 分离幅度谱和相位谱
    magnitude_original = cv2.magnitude(dft_original_shifted[:, :, 0], 
                                       dft_original_shifted[:, :, 1])
    phase_original = np.arctan2(dft_original_shifted[:, :, 1], 
                                 dft_original_shifted[:, :, 0])
    
    magnitude_enhanced = cv2.magnitude(dft_enhanced_shifted[:, :, 0], 
                                       dft_enhanced_shifted[:, :, 1])
    phase_enhanced = np.arctan2(dft_enhanced_shifted[:, :, 1], 
                                 dft_enhanced_shifted[:, :, 0])
    
    # ==================== 步骤4: 构建频域权重矩阵并融合 ====================
    print(f"[4/6] 构建 {filter_type} 权重矩阵并融合频谱...")
    
    # 创建权重矩阵（中心为1，边缘为0）
    weight_matrix = create_frequency_weight_matrix(
        shape=original_luma.shape,
        filter_type=filter_type
    )
    
    # 频域融合：增强图主导低频（中心），原图主导高频（边缘）
    fused_magnitude = (weight_matrix * magnitude_enhanced + 
                      (1.0 - weight_matrix) * magnitude_original)
    
    # 使用原图的相位谱（保留原图的结构信息）
    fused_phase = phase_original
    
    # ==================== 步骤5: 逆傅里叶变换回空间域 ====================
    print(f"[5/6] 执行逆傅里叶变换...")
    
    # 从幅度和相位重构复数频谱
    fused_real = fused_magnitude * np.cos(fused_phase)
    fused_imag = fused_magnitude * np.sin(fused_phase)
    fused_dft_shifted = np.stack([fused_real, fused_imag], axis=-1)
    
    # 将零频分量移回原位
    fused_dft = np.fft.ifftshift(fused_dft_shifted)
    
    # 执行逆DFT
    fused_luma = cv2.idft(fused_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # 裁剪到有效范围
    fused_luma = np.clip(fused_luma, 0, 255).astype(np.uint8)
    
    # ==================== 步骤6: 合并通道并转回BGR ====================
    print(f"[6/6] 合并通道并转换回BGR色彩空间...")
    
    # 合并融合后的亮度通道和原始的色度通道
    if color_space == 'YUV':
        fused_yuv = np.zeros_like(original_yuv)
        fused_yuv[:, :, 0] = fused_luma
        fused_yuv[:, :, 1:] = chroma_channels.astype(np.uint8)
        fused_bgr = cv2.cvtColor(fused_yuv, cv2.COLOR_YUV2BGR)
        
    elif color_space == 'Lab':
        fused_lab = np.zeros_like(original_lab)
        fused_lab[:, :, 0] = fused_luma
        fused_lab[:, :, 1:] = chroma_channels.astype(np.uint8)
        fused_bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
    
    # 保存结果
    cv2.imwrite(output_path, fused_bgr)
    print(f"✓ 融合图像已保存到: {output_path}")
    
    # ==================== 可视化（可选）====================
    if visualize:
        visualize_results(
            original_img, enhanced_img, fused_bgr,
            magnitude_original, magnitude_enhanced, fused_magnitude,
            weight_matrix, filter_type
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
    filter_type: str
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
    plt.savefig('/home/ubuntu/fusion_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存到: /home/ubuntu/fusion_visualization.png")
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
    original_img = cv2.imread(original_path)
    enhanced_img = cv2.imread(enhanced_path)
    
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('AI增强图像', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 显示权重矩阵对比
    axes[0, 2].set_title('权重剖面对比', fontsize=14, fontweight='bold')
    for filter_type in filter_types:
        weight = create_frequency_weight_matrix(original_img.shape[:2], filter_type)
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
        axes[1, col].set_title(f'融合结果 ({filter_type})', fontsize=14, fontweight='bold')
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
    
    # 检查文件是否存在
    import os
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
        print("\n⚠ 测试图像不存在，请提供 original.jpg 和 enhanced.jpg")
        print("   将创建演示图像进行测试...")
