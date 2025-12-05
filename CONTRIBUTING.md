# 贡献指南

感谢您对频域图像融合项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了bug或有功能建议，请：

1. 检查[Issues](https://github.com/yourusername/frequency-fusion-project/issues)中是否已有相关问题
2. 如果没有，创建一个新的Issue，详细描述问题或建议
3. 对于bug报告，请包含：
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（Python版本、操作系统等）

### 提交代码

1. **Fork仓库**
   ```bash
   # 在GitHub上Fork本仓库
   # 然后克隆您的Fork
   git clone https://github.com/your-username/frequency-fusion-project.git
   cd frequency-fusion-project
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行修改**
   - 遵循现有的代码风格
   - 添加必要的注释
   - 更新相关文档

4. **运行测试**
   ```bash
   # 安装测试依赖
   pip install pytest pytest-cov
   
   # 运行测试
   pytest tests/ -v
   ```

5. **提交更改**
   ```bash
   git add .
   git commit -m "描述您的更改"
   ```

6. **推送到GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建Pull Request**
   - 在GitHub上创建Pull Request
   - 详细描述您的更改
   - 链接相关的Issue

## 代码规范

### Python代码风格

- 遵循[PEP 8](https://www.python.org/dev/peps/pep-0008/)规范
- 使用4个空格缩进
- 行长度限制在127字符以内
- 使用有意义的变量名和函数名

### 文档字符串

使用Google风格的文档字符串：

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    简短描述
    
    详细描述（可选）
    
    参数:
        param1: 参数1的描述
        param2: 参数2的描述
    
    返回:
        返回值的描述
    
    异常:
        ValueError: 异常描述
    """
    pass
```

### 提交信息

提交信息应该清晰明了：

```
类型: 简短描述（不超过50字符）

详细描述（可选，说明为什么做这个更改）

相关Issue: #123
```

类型可以是：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

## 测试

### 编写测试

- 为新功能添加单元测试
- 确保测试覆盖率不降低
- 测试文件放在`tests/`目录下
- 测试文件名以`test_`开头

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_frequency_fusion.py

# 查看测试覆盖率
pytest tests/ --cov=. --cov-report=html
```

## 开发环境设置

1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/frequency-fusion-project.git
   cd frequency-fusion-project
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8 black
   ```

4. **运行示例**
   ```bash
   cd examples
   python create_test_images.py
   python example_usage.py
   ```

## 发布流程

1. 更新版本号（`setup.py`）
2. 更新CHANGELOG.md
3. 创建Git标签
4. 推送到GitHub
5. 创建GitHub Release

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

## 联系方式

如有任何问题，请通过以下方式联系：

- 创建Issue
- 发送邮件到项目维护者

感谢您的贡献！
