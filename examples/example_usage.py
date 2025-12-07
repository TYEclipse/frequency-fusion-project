#!/usr/bin/env python3
"""
频域图像融合算法使用示例
"""

import importlib
import importlib.util
import os
import sys
import traceback
from typing import Optional, Callable

# 确定项目根目录（用于查找本地模块）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_frequency_fusion(name: str = "frequency_fusion"):
    """稳健加载 frequency_fusion 模块（简化注释、保持行为）。"""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pass

    candidates = [
        os.path.join(PROJECT_ROOT, f"{name}.py"),
        os.path.join(PROJECT_ROOT, name, "__init__.py"),
    ]

    last_exc: Optional[Exception] = None
    for path in candidates:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or getattr(spec, "loader", None) is None:
            last_exc = ImportError(f"无法为路径 {path} 获取有效的 ModuleSpec/loader")
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception as e:
            last_exc = e
            continue
        # exec_module 成功后再注册到 sys.modules
        sys.modules[name] = module
        return module

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_exc = e

    msg = f"无法加载模块 '{name}'（既未安装也未在 {PROJECT_ROOT} 中找到有效源码）"
    if last_exc:
        raise ModuleNotFoundError(msg) from last_exc
    raise ModuleNotFoundError(msg)


# 取得模块对象并从中导出所需函数
_frequency_module = _load_frequency_fusion("frequency_fusion")

# 校验模块导出
compare_filter_types = getattr(_frequency_module, "compare_filter_types", None)
frequency_fusion = getattr(_frequency_module, "frequency_fusion", None)
if not callable(frequency_fusion) or not callable(compare_filter_types):
    raise ImportError(
        "加载的模块 'frequency_fusion' 未能导出可调用的 'frequency_fusion' 和 'compare_filter_types'。"
    )

# 直接导入，若导入失败让错误冒泡，便于尽早发现问题

# 优先使用项目内的 test_images 目录（相对于项目根）
TEST_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, 'test_images'))
# 确保目录存在，避免后续写入失败
os.makedirs(TEST_DIR, exist_ok=True)

ORIGINAL_PATH = os.path.join(TEST_DIR, 'original.jpg')
ENHANCED_PATH = os.path.join(TEST_DIR, 'enhanced.jpg')


# 修改示例函数以使用默认的 test_images 路径（可被外部调用时覆盖）
def example_basic_fusion(original_path: str = ORIGINAL_PATH, enhanced_path: str = ENHANCED_PATH):
    """
    示例1: 基本的图像融合
    """
    print("\n" + "="*70)
    print("示例1: 基本图像融合（线性权重）")
    print("="*70)

    frequency_fusion(
        original_path=original_path,
        enhanced_path=enhanced_path,
        output_path=os.path.join(TEST_DIR, 'fused_linear.jpg'),
        filter_type='linear',
        color_space='YUV',
        visualize=True
    )

    print("✓ 融合完成！查看 fused_linear.jpg 和 fusion_visualization.png")


def example_gaussian_fusion(original_path: str = ORIGINAL_PATH, enhanced_path: str = ENHANCED_PATH):
    """
    示例2: 使用高斯权重的图像融合
    """
    print("\n" + "="*70)
    print("示例2: 高斯权重融合")
    print("="*70)

    frequency_fusion(
        original_path=original_path,
        enhanced_path=enhanced_path,
        output_path=os.path.join(TEST_DIR, 'fused_gaussian.jpg'),
        filter_type='gaussian',
        color_space='YUV',
        visualize=False
    )

    print("✓ 融合完成！查看 fused_gaussian.jpg")


def example_compare_all_filters(original_path: str = ORIGINAL_PATH, enhanced_path: str = ENHANCED_PATH):
    """
    示例3: 对比所有权重过渡方式
    """
    print("\n" + "="*70)
    print("示例3: 对比所有权重过渡方式")
    print("="*70)

    compare_filter_types(
        original_path=original_path,
        enhanced_path=enhanced_path,
        output_dir=TEST_DIR
    )

    print("✓ 对比完成！查看 filter_comparison.png")


def example_lab_colorspace(original_path: str = ORIGINAL_PATH, enhanced_path: str = ENHANCED_PATH):
    """
    示例4: 使用Lab色彩空间
    """
    print("\n" + "="*70)
    print("示例4: 使用Lab色彩空间")
    print("="*70)

    frequency_fusion(
        original_path=original_path,
        enhanced_path=enhanced_path,
        output_path=os.path.join(TEST_DIR, 'fused_lab.jpg'),
        filter_type='linear',
        color_space='Lab',
        visualize=False
    )

    print("✓ 融合完成！查看 fused_lab.jpg")


def _load_helper_callable(module_name: str, func_name: str) -> Callable:
    """从 PROJECT_ROOT 加载辅助模块并返回可调用符号（简化实现）。"""
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        mod = None

    if mod is None:
        py_path = os.path.join(PROJECT_ROOT, f"{module_name}.py")
        init_path = os.path.join(PROJECT_ROOT, module_name, "__init__.py")
        for path in (py_path, init_path):
            if not os.path.exists(path):
                continue
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or getattr(spec, "loader", None) is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            except Exception as e:
                raise ImportError(
                    f"从 {path} 导入模块 {module_name} 失败: {e}") from e
            sys.modules[module_name] = mod
            break

    if mod is None:
        raise ModuleNotFoundError(f"无法导入辅助模块: {module_name}")

    fn = getattr(mod, func_name, None)
    if not callable(fn):
        raise ImportError(f"模块 {module_name} 不包含可调用的 {func_name}")
    return fn


if __name__ == "__main__":
    """
    运行所有示例

    使用方法:
    1. 将您的原始图像命名为 original.jpg
    2. 将AI增强图像命名为 enhanced.jpg
    3. 将它们放在项目的 test_images 目录下 (./test_images)
    4. 运行: python example_usage.py
    """

    # 检查测试图像是否存在（优先使用 test_images 目录）
    if not os.path.exists(ORIGINAL_PATH) or not os.path.exists(ENHANCED_PATH):
        print("\n⚠ 未找到测试图像！")
        print(f"正在创建演示图像到: {TEST_DIR} ...")

        # 使用健壮加载器加载 create_test_images.create_test_images
        try:
            create_test_images = _load_helper_callable(
                'create_test_images', 'create_test_images')
        except Exception as e:
            print(f"❌ 无法加载 create_test_images: {e}")
            traceback.print_exc()
            raise

        try:
            # 优先尝试传入目标目录（若 create_test_images 支持该签名）
            create_test_images(TEST_DIR)
        except TypeError:
            # 回退：无参调用
            create_test_images()
        except Exception as e:
            print(f"❌ 生成测试图像时出错: {e}")
            traceback.print_exc()
            raise

        print(
            f"\n✓ 测试图像已创建:\n  - {ORIGINAL_PATH} (原始图像)\n  - {ENHANCED_PATH} (AI增强图像)\n")

    # 运行示例（使用 test_images 中的路径）
    try:
        example_basic_fusion()
        example_gaussian_fusion()
        example_compare_all_filters()
        example_lab_colorspace()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)
        print("\n生成的文件:")
        print(f"  - {os.path.join(TEST_DIR, 'fused_linear.jpg')}")
        print(f"  - {os.path.join(TEST_DIR, 'fused_gaussian.jpg')}")
        print(f"  - {os.path.join(TEST_DIR, 'fused_cosine.jpg')}")
        print(f"  - {os.path.join(TEST_DIR, 'fused_quadratic.jpg')}")
        print(f"  - {os.path.join(TEST_DIR, 'fused_lab.jpg')}")
        print(f"  - {os.path.join(TEST_DIR, 'fused_rgb_average.jpg')}")
        print(f"  - {os.path.join(TEST_DIR, 'fusion_visualization.png')}")
        print(f"  - {os.path.join(TEST_DIR, 'filter_comparison.png')}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
