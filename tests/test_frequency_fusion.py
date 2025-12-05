#!/usr/bin/env python3
"""
频域图像融合算法单元测试
"""

import unittest
import numpy as np
import cv2
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frequency_fusion import (
    create_frequency_weight_matrix,
    frequency_fusion
)


class TestFrequencyWeightMatrix(unittest.TestCase):
    """测试权重矩阵生成"""
    
    def test_linear_weight_shape(self):
        """测试线性权重矩阵的形状"""
        shape = (256, 256)
        weight = create_frequency_weight_matrix(shape, 'linear')
        self.assertEqual(weight.shape, shape)
    
    def test_linear_weight_range(self):
        """测试线性权重矩阵的值域"""
        weight = create_frequency_weight_matrix((256, 256), 'linear')
        self.assertTrue(np.all(weight >= 0.0))
        self.assertTrue(np.all(weight <= 1.0))
    
    def test_linear_weight_center(self):
        """测试线性权重矩阵中心值"""
        weight = create_frequency_weight_matrix((256, 256), 'linear')
        center_value = weight[128, 128]
        self.assertAlmostEqual(center_value, 1.0, places=2)
    
    def test_gaussian_weight_shape(self):
        """测试高斯权重矩阵的形状"""
        shape = (128, 128)
        weight = create_frequency_weight_matrix(shape, 'gaussian')
        self.assertEqual(weight.shape, shape)
    
    def test_gaussian_weight_range(self):
        """测试高斯权重矩阵的值域"""
        weight = create_frequency_weight_matrix((128, 128), 'gaussian')
        self.assertTrue(np.all(weight >= 0.0))
        self.assertTrue(np.all(weight <= 1.0))
    
    def test_cosine_weight_shape(self):
        """测试余弦权重矩阵的形状"""
        shape = (200, 200)
        weight = create_frequency_weight_matrix(shape, 'cosine')
        self.assertEqual(weight.shape, shape)
    
    def test_quadratic_weight_shape(self):
        """测试二次权重矩阵的形状"""
        shape = (300, 300)
        weight = create_frequency_weight_matrix(shape, 'quadratic')
        self.assertEqual(weight.shape, shape)
    
    def test_invalid_filter_type(self):
        """测试无效的过滤器类型"""
        with self.assertRaises(ValueError):
            create_frequency_weight_matrix((256, 256), 'invalid')


class TestFrequencyFusion(unittest.TestCase):
    """测试图像融合功能"""
    
    @classmethod
    def setUpClass(cls):
        """创建测试图像"""
        cls.test_dir = '/tmp/test_frequency_fusion'
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # 创建简单的测试图像
        cls.original_path = os.path.join(cls.test_dir, 'test_original.jpg')
        cls.enhanced_path = os.path.join(cls.test_dir, 'test_enhanced.jpg')
        cls.output_path = os.path.join(cls.test_dir, 'test_fused.jpg')
        
        # 生成测试图像
        img_original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_enhanced = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        cv2.imwrite(cls.original_path, img_original)
        cv2.imwrite(cls.enhanced_path, img_enhanced)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_basic_fusion(self):
        """测试基本融合功能"""
        result = frequency_fusion(
            self.original_path,
            self.enhanced_path,
            self.output_path,
            filter_type='linear',
            color_space='YUV',
            visualize=False
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertTrue(os.path.exists(self.output_path))
    
    def test_gaussian_fusion(self):
        """测试高斯权重融合"""
        output = os.path.join(self.test_dir, 'test_gaussian.jpg')
        result = frequency_fusion(
            self.original_path,
            self.enhanced_path,
            output,
            filter_type='gaussian',
            color_space='YUV',
            visualize=False
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output))
    
    def test_lab_colorspace(self):
        """测试Lab色彩空间"""
        output = os.path.join(self.test_dir, 'test_lab.jpg')
        result = frequency_fusion(
            self.original_path,
            self.enhanced_path,
            output,
            filter_type='linear',
            color_space='Lab',
            visualize=False
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output))
    
    def test_different_sizes(self):
        """测试不同尺寸的图像"""
        # 创建不同尺寸的增强图
        enhanced_diff = os.path.join(self.test_dir, 'test_enhanced_diff.jpg')
        img_diff = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(enhanced_diff, img_diff)
        
        output = os.path.join(self.test_dir, 'test_diff_size.jpg')
        result = frequency_fusion(
            self.original_path,
            enhanced_diff,
            output,
            filter_type='linear',
            color_space='YUV',
            visualize=False
        )
        
        # 结果应该与原图尺寸一致
        self.assertEqual(result.shape[:2], (256, 256))
    
    def test_invalid_paths(self):
        """测试无效的文件路径"""
        with self.assertRaises(FileNotFoundError):
            frequency_fusion(
                'nonexistent.jpg',
                self.enhanced_path,
                self.output_path,
                visualize=False
            )


class TestWeightMatrixProperties(unittest.TestCase):
    """测试权重矩阵的数学性质"""
    
    def test_symmetry(self):
        """测试权重矩阵的对称性"""
        weight = create_frequency_weight_matrix((256, 256), 'linear')
        
        # 测试上下对称
        self.assertTrue(np.allclose(weight[:128, :], np.flipud(weight[128:, :])))
        
        # 测试左右对称
        self.assertTrue(np.allclose(weight[:, :128], np.fliplr(weight[:, 128:])))
    
    def test_monotonic_decrease(self):
        """测试权重从中心向边缘单调递减"""
        weight = create_frequency_weight_matrix((256, 256), 'linear')
        
        # 检查水平方向
        center_row = weight[128, :]
        for i in range(127):
            self.assertGreaterEqual(center_row[128-i], center_row[128-i-1])
            self.assertGreaterEqual(center_row[128+i], center_row[128+i+1])


if __name__ == '__main__':
    unittest.main()
