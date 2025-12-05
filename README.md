# 基于频域的自适应图像融合算法

简洁概述：在频域中以“增强图主导低频（色彩/整体亮度） + 原图保留高频（细节/纹理）”的策略，实现色彩与细节的最佳平衡，适合图像增强与风格保真场景。

---

<!-- 移除所有 <a id="..."></a> 内联 HTML 锚点，直接保留原始标题文本 -->

## ✨ 核心特性

- 频域精确控制：分别处理低频/高频，独立调整色彩与细节融合比例。  
- 色彩保真：仅在亮度通道融合（YUV 或 Lab），避免色偏。  
- 多种权重策略：支持 Linear / Gaussian / Cosine / Quadratic。  
- 自动尺寸适配：无需手动预处理，不同输入尺寸自动适配。  
- 完整可视化：输出幅度谱、权重矩阵与融合前后对比图。  
- 易于集成：清晰的 API 和示例，便于嵌入现有流水线。

---

## 🚀 快速开始

推荐使用 conda（也可使用 venv）：

```bash
# 从代码仓库获取项目后在项目根目录运行
conda create -n frequency-fusion-project python=3.12 -y
conda activate frequency-fusion-project
pip install -r requirements.txt
pip install .
```

运行示例：

```bash
cd examples
python create_test_images.py
python example_usage.py
```

提示：如需在不同 Python 版本下测试，请调整 conda 环境版本。

---

## 🧩 示例代码

调用示例（可直接复制）：

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

## 📖 算法流程（六步）

1. 色彩空间转换：BGR -> YUV / Lab，分离亮度（Y / L）与色度（U/V 或 a/b）。  
2. 傅里叶变换：对亮度通道执行 2D DFT，频域中心代表低频分量。  
3. 构建权重矩阵：从中心到边缘平滑过渡，中心权重 ≈ 1（增强图主导），边缘 ≈ 0（原图主导）。  
4. 频域融合：M_fused = W *M_enhanced + (1 - W)* M_original；保留原图相位以维持结构。  
5. 逆傅里叶变换：合并幅度与相位，执行逆 DFT，恢复融合后的亮度通道。  
6. 色彩重建：将融合亮度与原始色度合并，转换回 BGR，输出最终图像。

说明：保留相位可最大程度保持结构与边缘细节。

---

## ⚖️ 权重过渡方式

名称与简要公式及特点：

- Linear: W(r) = 1 - r / r_max — 均匀过渡，通用。
- Gaussian: W(r) = exp(-r²/(2σ²)) — 中心权重高，突出增强。
- Cosine: W(r) = 0.5 (1 + cos(π r')) — 最平滑过渡，视觉质量优先。
- Quadratic: W(r) = (1 - r/r_max)² — 中心更突出，局部增强明显。

注：r' 为归一化半径 r / r_max。

---

## 📚 API 文档（摘要）

- frequency_fusion(original_path, enhanced_path, output_path, filter_type='linear', color_space='YUV', visualize=False)  
  返回：np.ndarray（融合后的 BGR 图像），并在 output_path 保存结果。

- create_frequency_weight_matrix(shape, filter_type)  
  返回：权重矩阵（中心为 1.0，边缘为 0.0）。

- compare_filter_types(original_path, enhanced_path, output_dir)  
  生成：四种权重方式下的融合结果与对比分析图。

更多详细说明请参见代码中的 docstring 与模块注释。

---

## 🖼️ 可视化示例

输出项：幅度谱（原始/增强/融合）、权重矩阵热力图、融合前后对比。建议将这些图导出至 docs 或 assets，便于插入演示文稿。

---

## 🧪 测试

推荐使用 pytest：

```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

---

## 🤝 贡献与许可证

欢迎提交 Issue 或 PR。详见项目中的 CONTRIBUTING 文件。许可证：MIT（见项目 LICENSE）。

---

## 📮 联系方式

通过项目的 Issue 系统联系维护者。  
开发者：资深计算机视觉工程师  
版本：1.0.0 | 最后更新：2025-12-05

---

附注：若需 PPT / 演示材料，建议导出频谱图、权重可视化与融合前后对比图到 docs/ 或 assets/，便于直接插入幻灯片。
