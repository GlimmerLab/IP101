# 贡献指南

感谢您对图像处理项目的关注！我们欢迎任何形式的贡献，包括但不限于：

- 报告问题
- 提交修复
- 添加新功能
- 改进文档
- 优化性能

## 开发环境设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/image-processing.git
cd image-processing
```

2. 安装依赖：
```bash
# C++依赖
sudo apt-get install build-essential cmake libopencv-dev

# Python依赖
pip install -r requirements.txt
```

3. 构建项目：
```bash
mkdir build
cd build
cmake ..
make
```

## 代码规范

### C++代码规范

- 使用C++17标准
- 遵循Google C++ Style Guide
- 使用4空格缩进
- 使用驼峰命名法
- 添加适当的注释和文档

### Python代码规范

- 遵循PEP 8规范
- 使用4空格缩进
- 使用snake_case命名法
- 添加适当的文档字符串

## 提交代码

1. 创建新分支：
```bash
git checkout -b feature/your-feature-name
```

2. 提交更改：
```bash
git add .
git commit -m "描述你的更改"
```

3. 推送到远程：
```bash
git push origin feature/your-feature-name
```

4. 创建Pull Request

## 测试要求

- 所有新功能必须包含单元测试
- 所有测试必须通过
- 代码覆盖率应保持在80%以上

## 文档要求

- 所有新功能必须包含文档
- 更新README.md（如果需要）
- 添加示例代码

## 性能优化

- 使用SIMD指令集优化关键代码
- 使用OpenMP进行并行计算
- 优化内存访问模式
- 减少不必要的内存分配

## 问题报告

- 使用GitHub Issues报告问题
- 提供详细的复现步骤
- 包含环境信息
- 提供错误日志

## 代码审查

- 所有Pull Request都需要经过代码审查
- 至少需要一个核心开发者的批准
- 确保代码符合项目规范
- 确保测试覆盖率达标

## 发布流程

1. 更新版本号
2. 更新CHANGELOG.md
3. 创建发布标签
4. 构建发布包
5. 发布到PyPI（Python包）

## 联系方式

- 项目维护者：your-email@example.com
- 项目主页：https://github.com/yourusername/image-processing
- 问题追踪：https://github.com/yourusername/image-processing/issues

感谢您的贡献！