# 基于频域的自适应图像融合算法

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

一个基于傅里叶变换的图像融合算法，专门用于平衡AI增强图像的色彩与原图的细节。该算法在频域中让增强图主导低频信号（色彩、整体亮度），让原图主导高频信号（细节、纹理、边缘），实现最佳的视觉效果。

## ✨ 核心特性

**频域精确控制** - 在频域中分别处理低频和高频信息，实现精确的融合控制。通过傅里叶变换将图像分解为不同频率分量，可以独立调整色彩和细节的融合比例。

**色彩保真** - 仅在亮度通道进行融合，完全避免色偏问题。通过在YUV或Lab色彩空间中分离亮度和色度信息，确保色彩的一致性和准确性。

**多种权重方式** - 支持Linear、Gaussian、Cosine、Quadratic四种权重过渡方式，适应不同的应用场景和视觉需求。

**灵活的色彩空间** - 支持YUV和Lab两种色彩空间，可根据具体应用选择最合适的色彩表示方式。

**自动尺寸适配** - 自动处理不同尺寸的输入图像，无需手动调整图像尺寸。

**完整可视化** - 提供详细的频域分析和对比可视化，包括幅度谱、权重矩阵、融合结果等多维度展示。

**易于集成** - 清晰的API设计，完善的文档和示例，易于集成到现有项目中。

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/frequency-fusion-project.git
cd frequency-fusion-project

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```python
from frequency_fusion import frequency_fusion

# 执行图像融合
fused_img = frequency_fusion(
    original_path='original.jpg',
    enhanced_path='enhanced.jpg',
    output_path='fused.jpg',
    filter_type='linear',      # 可选: 'linear', 'gaussian', 'cosine', 'quadratic'
    color_space='YUV',         # 可选: 'YUV', 'Lab'
    visualize=True             # 生成可视化分析图
)
```

### 运行示例

```bash
# 创建测试图像
cd examples
python create_test_images.py

# 运行示例代码
python example_usage.py
```

## 📖 技术原理

### 算法流程

该算法通过以下六个步骤实现高质量的图像融合。

**步骤1：色彩空间转换** - 将图像从BGR转换到YUV或Lab空间，分离亮度和色度信息。这样可以仅在亮度通道进行频域处理，避免直接处理色彩通道导致的色偏问题。YUV空间中Y通道表示亮度，U和V通道表示色度；Lab空间中L通道表示亮度，a和b通道表示色度。

**步骤2：傅里叶变换** - 对亮度通道执行二维离散傅里叶变换（DFT），将空间域图像转换到频域。频域表示中，中心区域代表低频分量（整体色彩和亮度），边缘区域代表高频分量（细节和纹理）。这种转换使得我们可以在频域中独立处理不同频率的信息。

**步骤3：构建权重矩阵** - 创建一个从中心到边缘平滑过渡的权重矩阵W。中心权重为1.0（完全使用增强图的低频），边缘权重为0.0（完全使用原图的高频）。权重矩阵的形状决定了融合的特性，不同的过渡函数产生不同的视觉效果。

**步骤4：频域融合** - 使用权重矩阵融合两张图像的幅度谱。融合公式为 `M_fused = W × M_enhanced + (1 - W) × M_original`，其中M表示幅度谱。同时保留原图的相位谱以保持结构信息，因为相位谱包含了图像的主要结构特征。

**步骤5：逆傅里叶变换** - 将融合后的幅度谱和相位谱结合，执行逆DFT回到空间域，得到融合后的亮度通道。这一步将频域的融合结果转换回可视化的图像数据。

**步骤6：色彩重建** - 将融合后的亮度通道与原始的色度通道合并，转换回BGR色彩空间，得到最终的融合图像。这确保了融合图像保持原图的色彩特征。

### 权重过渡方式

算法支持四种不同的权重过渡方式，每种方式都有其独特的特点和适用场景。

| 权重方式 | 数学公式 | 特点 | 适用场景 |
|---------|---------|------|---------|
| **Linear** | `W(r) = 1 - r/r_max` | 均匀过渡，效果均衡 | 通用场景，首选方式 |
| **Gaussian** | `W(r) = exp(-r²/(2σ²))` | 中心权重保持较高 | AI增强效果较好时 |
| **Cosine** | `W(r) = 0.5(1 + cos(πr))` | 最平滑的视觉过渡 | 追求最佳视觉质量 |
| **Quadratic** | `W(r) = (1 - r/r_max)²` | 边缘权重衰减更快 | AI增强过度时 |

Linear权重提供了最均匀的过渡，适合大多数场景。Gaussian权重在中心区域保持较高的权重，更多地保留增强图的整体效果。Cosine权重提供了最平滑的视觉过渡，适合追求高视觉质量的应用。Quadratic权重在边缘区域衰减更快，更强调原图的细节。

### 核心技术要点

**为什么使用原图的相位谱？** 相位谱包含了图像的结构信息，而幅度谱主要包含能量分布信息。研究表明，人眼对相位信息更敏感。通过保留原图的相位谱，可以确保融合图像保持原图的结构特征，同时获得增强图的色彩优势。

**为什么在亮度通道融合？** 色度信息对人眼感知的影响较小，但直接在频域处理色度通道容易产生色偏。通过仅在亮度通道融合，可以在保持色彩一致性的同时，实现亮度和细节的平衡。

## 🎯 适用场景

该算法在以下场景中表现出色，能够显著提升图像质量。

**AI图像增强后处理** - 恢复AI增强过程中丢失的细节。许多AI增强算法在提升色彩和对比度的同时会丢失高频细节，本算法可以有效恢复这些细节。

**HDR图像融合** - 融合不同曝光的图像，结合多张不同曝光照片的优势，生成高动态范围图像。

**医学图像处理** - 增强医学图像的同时保留诊断细节，确保图像增强不会影响医学诊断的准确性。

**卫星图像处理** - 提升卫星图像的视觉效果，在增强色彩的同时保留地物细节。

**低光照图像增强** - 在提亮图像的同时保留细节，避免噪声放大。

## 📚 API文档

### `frequency_fusion()`

核心融合函数，执行基于频域的图像融合。

**参数：**
- `original_path` (str): 原始图像路径
- `enhanced_path` (str): AI增强图像路径
- `output_path` (str): 输出融合图像路径
- `filter_type` (str): 权重过渡方式，可选 'linear', 'gaussian', 'cosine', 'quadratic'，默认 'linear'
- `color_space` (str): 色彩空间，可选 'YUV', 'Lab'，默认 'YUV'
- `visualize` (bool): 是否生成可视化结果，默认 False

**返回：**
- `np.ndarray`: 融合后的BGR图像

### `create_frequency_weight_matrix()`

创建频域权重矩阵。

**参数：**
- `shape` (Tuple[int, int]): 矩阵尺寸 (rows, cols)
- `filter_type` (str): 权重过渡方式

**返回：**
- `np.ndarray`: 权重矩阵，中心为1.0，边缘为0.0

### `compare_filter_types()`

对比不同权重过渡方式的融合效果。

**参数：**
- `original_path` (str): 原始图像路径
- `enhanced_path` (str): AI增强图像路径
- `output_dir` (str): 输出目录

**输出：**
- 四张不同权重方式的融合图像
- 一张对比分析图

## 🧪 测试

运行单元测试以验证算法的正确性：

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行测试
pytest tests/ -v

# 查看测试覆盖率
pytest tests/ --cov=. --cov-report=html
```

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📮 联系方式

如有任何问题或建议，请通过GitHub Issues联系我们。

## 🔗 相关资源

- [OpenCV官方文档](https://docs.opencv.org/)
- [傅里叶变换教程](https://en.wikipedia.org/wiki/Fourier_transform)
- [数字图像处理](https://en.wikipedia.org/wiki/Digital_image_processing)

---

**开发者**: 资深计算机视觉工程师  
**版本**: 1.0.0  
**最后更新**: 2025-12-05
