#!/usr/bin/env python3
"""
频域图像融合算法使用示例
"""

import sys
import os

# 添加父目录到路径以便导入主模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frequency_fusion import frequency_fusion, compare_filter_types


def example_basic_fusion():
    """
    示例1: 基本的图像融合
    """
    print("\n" + "="*70)
    print("示例1: 基本图像融合（线性权重）")
    print("="*70)
    
    fused_img = frequency_fusion(
        original_path='original.jpg',
        enhanced_path='enhanced.jpg',
        output_path='fused_linear.jpg',
        filter_type='linear',
        color_space='YUV',
        visualize=True
    )
    
    print("✓ 融合完成！查看 fused_linear.jpg 和 fusion_visualization.png")


def example_gaussian_fusion():
    """
    示例2: 使用高斯权重的图像融合
    """
    print("\n" + "="*70)
    print("示例2: 高斯权重融合")
    print("="*70)
    
    fused_img = frequency_fusion(
        original_path='original.jpg',
        enhanced_path='enhanced.jpg',
        output_path='fused_gaussian.jpg',
        filter_type='gaussian',
        color_space='YUV',
        visualize=False
    )
    
    print("✓ 融合完成！查看 fused_gaussian.jpg")


def example_compare_all_filters():
    """
    示例3: 对比所有权重过渡方式
    """
    print("\n" + "="*70)
    print("示例3: 对比所有权重过渡方式")
    print("="*70)
    
    compare_filter_types(
        original_path='original.jpg',
        enhanced_path='enhanced.jpg',
        output_dir='.'
    )
    
    print("✓ 对比完成！查看 filter_comparison.png")


def example_lab_colorspace():
    """
    示例4: 使用Lab色彩空间
    """
    print("\n" + "="*70)
    print("示例4: 使用Lab色彩空间")
    print("="*70)
    
    fused_img = frequency_fusion(
        original_path='original.jpg',
        enhanced_path='enhanced.jpg',
        output_path='fused_lab.jpg',
        filter_type='linear',
        color_space='Lab',
        visualize=False
    )
    
    print("✓ 融合完成！查看 fused_lab.jpg")


if __name__ == "__main__":
    """
    运行所有示例
    
    使用方法:
    1. 将您的原始图像命名为 original.jpg
    2. 将AI增强图像命名为 enhanced.jpg
    3. 将它们放在与此脚本相同的目录下
    4. 运行: python example_usage.py
    """
    
    # 检查测试图像是否存在
    if not os.path.exists('original.jpg') or not os.path.exists('enhanced.jpg'):
        print("\n⚠ 未找到测试图像！")
        print("正在创建演示图像...")
        
        # 导入并运行测试图像生成器
        from create_test_images import create_test_images
        create_test_images()
        print("\n✓ 演示图像已创建！\n")
    
    # 运行示例
    try:
        example_basic_fusion()
        example_gaussian_fusion()
        example_compare_all_filters()
        example_lab_colorspace()
        
        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)
        print("\n生成的文件:")
        print("  - fused_linear.jpg")
        print("  - fused_gaussian.jpg")
        print("  - fused_cosine.jpg")
        print("  - fused_quadratic.jpg")
        print("  - fused_lab.jpg")
        print("  - fusion_visualization.png")
        print("  - filter_comparison.png")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
