# 快速使用指南

## 一分钟上手

### 1. 准备图像

将您的原始图像和AI增强图像放在同一目录下，例如：
- `my_original.jpg`
- `my_enhanced.jpg`

### 2. 运行融合

```python
from frequency_fusion import frequency_fusion

# 执行融合
fused_img = frequency_fusion(
    original_path='my_original.jpg',
    enhanced_path='my_enhanced.jpg',
    output_path='my_fused.jpg',
    filter_type='linear',      # 可选: 'linear', 'gaussian', 'cosine', 'quadratic'
    color_space='YUV',         # 可选: 'YUV', 'Lab'
    visualize=True             # 生成可视化分析图
)
```

### 3. 查看结果

- `my_fused.jpg`: 融合后的图像
- `fusion_visualization.png`: 详细的可视化分析

## 对比不同权重方式

```python
from frequency_fusion import compare_filter_types

# 自动生成四种权重方式的对比
compare_filter_types(
    original_path='my_original.jpg',
    enhanced_path='my_enhanced.jpg',
    output_dir='.'
)
```

这将生成：
- `fused_linear.jpg`
- `fused_gaussian.jpg`
- `fused_cosine.jpg`
- `fused_quadratic.jpg`
- `filter_comparison.png` (对比图)

## 命令行使用

```bash
# 创建一个简单的命令行脚本
python3 -c "
from frequency_fusion import frequency_fusion
frequency_fusion('original.jpg', 'enhanced.jpg', 'fused.jpg', 'linear', 'YUV', True)
"
```

## 权重方式选择建议

| 权重方式 | 特点 | 适用场景 |
|---------|------|---------|
| **linear** | 均匀过渡，效果均衡 | 通用场景，首选 |
| **gaussian** | 更多保留增强图效果 | AI增强效果较好时 |
| **cosine** | 最自然的视觉过渡 | 追求视觉质量时 |
| **quadratic** | 强调原图细节 | AI增强过度时 |

## 常见问题

### Q: 两张图像尺寸不一致怎么办？
A: 算法会自动将增强图缩放到原图尺寸，无需手动处理。

### Q: 选择YUV还是Lab色彩空间？
A: 两者效果相近，YUV计算稍快，Lab在某些场景下色彩更准确。建议先尝试YUV。

### Q: 如何调整融合强度？
A: 可以通过修改权重矩阵的生成参数（如高斯的sigma值）来调整。详见代码中的`create_frequency_weight_matrix`函数。

### Q: 处理速度慢怎么办？
A: 对于大图像，可以先缩小尺寸处理，或者关闭`visualize`选项以加快速度。

## 进阶使用

### 自定义权重矩阵

```python
import numpy as np
from frequency_fusion import create_frequency_weight_matrix

# 创建自定义权重
weight = create_frequency_weight_matrix(
    shape=(512, 512),
    filter_type='gaussian'
)

# 可视化权重分布
import matplotlib.pyplot as plt
plt.imshow(weight, cmap='hot')
plt.colorbar()
plt.savefig('weight_matrix.png')
```

### 批量处理

```python
import os
from frequency_fusion import frequency_fusion

# 批量处理多对图像
image_pairs = [
    ('original1.jpg', 'enhanced1.jpg'),
    ('original2.jpg', 'enhanced2.jpg'),
    ('original3.jpg', 'enhanced3.jpg'),
]

for i, (orig, enh) in enumerate(image_pairs):
    output = f'fused_{i+1}.jpg'
    frequency_fusion(orig, enh, output, 'linear', 'YUV', False)
    print(f'✓ 完成 {i+1}/{len(image_pairs)}')
```

## 性能优化建议

1. **关闭可视化**：`visualize=False` 可显著提升速度
2. **缩小图像**：对于超大图像，先缩放到合适尺寸
3. **批量处理**：使用循环批量处理多张图像
4. **并行处理**：对于大量图像，可使用多进程并行

## 技术支持

如有问题或建议，请查看完整的README.md文档。
