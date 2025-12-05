# 基于频域的自适应图像融合算法

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

简约说明：在频域中以增强图引导低频（色彩/整体亮度），以原图保持高频（细节/纹理），实现色彩与细节的最佳平衡。

---

![cover](docs/cover.png) <!-- 可替换为真实封面图：docs/cover.png -->

目录

- [核心特性](#core-features)
- [快速开始](#quick-start)
- [示例代码](#example-code)
- [算法流程](#algorithm-flow)
- [权重过渡方式](#weight-transition)
- [API 文档](#api-docs)
- [测试](#tests)
- [贡献与许可证](#contributing-license)
- [联系方式](#contact)

---

## ✨ 核心特性 {#core-features}

- 频域精确控制：对低频/高频分别处理，独立调整色彩与细节融合比例。  
- 色彩保真：仅在亮度通道融合（YUV 或 Lab），避免色偏。  
- 多种权重方式：支持 Linear、Gaussian、Cosine、Quadratic。  
- 自动尺寸适配：自动处理不同输入尺寸，无需手动预处理。  
- 完整可视化：幅度谱、权重矩阵与融合对比图一应俱全。  
- 易于集成：清晰 API 与示例，便于嵌入现有流水线。

---

## 🚀 快速开始 {#quick-start}

安装（示例使用 conda，亦可使用 venv）：

```bash
git clone https://github.com/yourusername/frequency-fusion-project.git
cd frequency-fusion-project

conda create -n frequency-fusion-project python=3.12
conda activate frequency-fusion-project
pip install -r requirements.txt
```

示例运行：

```bash
cd examples
python create_test_images.py
python example_usage.py
```

---

## 🧩 示例代码 {#example-code}

```python
from frequency_fusion import frequency_fusion

fused_img = frequency_fusion(
    original_path='original.jpg',
    enhanced_path='enhanced.jpg',
    output_path='fused.jpg',
    filter_type='linear',   # 'linear' | 'gaussian' | 'cosine' | 'quadratic'
    color_space='YUV',      # 'YUV' | 'Lab'
    visualize=True
)
```

---

## 📖 算法流程（六步） {#algorithm-flow}

1. 色彩空间转换：BGR -> YUV / Lab，分离亮度（Y / L）与色度（U/V 或 a/b）。  
2. 傅里叶变换：对亮度通道执行 2D DFT，频域中心为低频。  
3. 构建权重矩阵：从中心到边缘平滑过渡，中心权重为 1（增强图主导），边缘为 0（原图主导）。  
4. 频域融合：M_fused = W *M_enhanced + (1 - W)* M_original；保留原图相位以维持结构。  
5. 逆傅里叶变换：合并幅度与相位，执行逆 DFT，恢复融合后的亮度。  
6. 色彩重建：将融合亮度与原始色度合并，转换回 BGR，输出最终图像。

为什么保留相位？相位包含结构信息；保留原图相位能最大限度保留细节与边缘。

---

## ⚖️ 权重过渡方式（视觉与数学） {#weight-transition}

| 名称 | 公式 | 特点 | 适用场景 |
|---:|:---:|:---|:---|
| Linear | W(r) = 1 - r/r_max | 均匀过渡，平衡 | 通用首选 |
| Gaussian | W(r) = exp(-r²/(2σ²)) | 中心权重高，保持增强效果 | 增强效果优先 |
| Cosine | W(r) = 0.5(1 + cos(π r')) | 最平滑过渡 | 视觉质量优先 |
| Quadratic | W(r) = (1 - r/r_max)² | 中心更突出，边缘快速衰减 | 强调细节恢复 |

说明：r' 表示归一化半径（r / r_max）。

---

## 📚 API 文档（摘要） {#api-docs}

- frequency_fusion(original_path, enhanced_path, output_path, filter_type='linear', color_space='YUV', visualize=False)  
  返回：np.ndarray（融合后的 BGR 图像），会在 output_path 保存结果。

- create_frequency_weight_matrix(shape, filter_type)  
  返回：权重矩阵（中心 1.0，边缘 0.0）。

- compare_filter_types(original_path, enhanced_path, output_dir)  
  生成：四种权重方式下的融合图与对比分析图。

更多信息请查看项目代码中的 docstring。

---

## 🧪 测试 {#tests}

```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

---

## 🤝 贡献与许可证 {#contributing-license}

欢迎提交 Issue、PR 或建议。请参阅 CONTRIBUTING.md。  
许可：MIT — 详见 LICENSE。

---

## 📮 联系方式 {#contact}

通过 GitHub Issues 联系项目维护者。  
开发者：资深计算机视觉工程师  
版本：1.0.0 | 最后更新：2025-12-05

---

附：若需 PPT / 演示用的视觉材料，可将项目中的频谱图、权重可视化与融合前后对比导出为独立图像（docs/ 或 assets/），便于插入幻灯片中。
