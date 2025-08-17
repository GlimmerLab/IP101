# IP101 GUI 示例

本项目提供了两种GUI界面供用户选择，支持智能依赖管理和自动下载。

## 🚀 快速开始

### Windows用户
```bash
# 双击运行快速启动脚本
gui/quick_start.bat
```

### Linux/macOS用户
```bash
# 运行快速启动脚本
./gui/quick_start.sh
```

### 手动安装
```bash
# 1. 下载依赖
python gui/setup_dependencies.py

# 2. 构建项目
mkdir build && cd build
cmake ..
make

# 3. 运行GUI
./simple_gui  # 简化版
./main_gui    # 高级版
```

## 🎯 简化版GUI (simple_gui.cpp)

### 特点
- **零额外依赖**：只使用OpenCV内置的GUI功能
- **快速启动**：无需安装额外的库
- **实时调节**：支持参数实时调节和预览
- **键盘控制**：简洁的键盘操作界面

### 功能
- 图像加载和保存
- 6种算法实时切换
- 参数实时调节
- 原图和效果图对比显示

### 操作说明
```
键盘控制：
  1-6: 选择算法
   1 - Guided Filter (引导滤波)
   2 - Side Window Filter (侧窗滤波)
   3 - Gamma Correction (伽马校正)
   4 - Dark Channel Defogging (暗通道去雾)
   5 - Cartoon Effect (卡通效果)
   6 - Oil Painting Effect (油画效果)

  l: 加载图像
  s: 保存处理后的图像
  q: 退出程序
```

## 🎨 高级版GUI (main_gui.cpp)

### 特点
- **现代化界面**：基于Dear ImGui的现代GUI
- **丰富交互**：下拉菜单、滑块、按钮等
- **多窗口布局**：可调整的窗口布局
- **专业外观**：类似专业图像处理软件

### 依赖
- Dear ImGui (自动下载)
- GLFW (自动下载)
- OpenGL (系统自带)

## 📦 智能依赖管理

### 自动下载
项目会自动检测并下载所需依赖：
- **ImGui v1.89.9**：现代化GUI库
- **GLFW v3.3.8**：跨平台窗口管理

### 依赖位置
```
third_party/
├── imgui/          # ImGui库
├── glfw/           # GLFW库
└── CMakeLists.txt  # 自动生成的CMake配置
```

### 手动管理
```bash
# 检查依赖状态
python gui/setup_dependencies.py

# 清理依赖
rm -rf third_party/

# 重新下载
python gui/setup_dependencies.py
```

## 🚀 推荐使用

### 初学者/快速体验
推荐使用 **简化版GUI**，因为：
- 无需额外依赖
- 快速上手
- 功能完整

### 专业用户/长期使用
推荐使用 **高级版GUI**，因为：
- 界面更专业
- 操作更便捷
- 功能更丰富

## 📁 输出文件

处理后的图像会保存在 `output/` 目录下：
- `simple_gui_processed_[算法名].jpg` - 简化版输出
- `gui_processed_[算法名].jpg` - 高级版输出

## 🔧 自定义

### 添加新算法
1. 在 `apply_algorithm()` 函数中添加新的算法分支
2. 在参数结构体中添加相应参数
3. 在 `setup_trackbars()` 中添加参数调节控件

### 修改默认参数
编辑 `AlgorithmParams` 结构体中的默认值

### 更改默认图像
修改 `load_image()` 调用中的文件路径

## 🐛 故障排除

### 简化版GUI问题
- **窗口不显示**：检查OpenCV是否正确安装
- **图像加载失败**：确保 `assets/imori.jpg` 文件存在
- **参数调节无效果**：检查算法函数是否正确实现

### 高级版GUI问题
- **编译失败**：运行 `python gui/setup_dependencies.py` 下载依赖
- **运行时错误**：确保OpenGL驱动正常工作
- **图像显示异常**：检查OpenGL版本兼容性

### 依赖下载问题
- **网络连接失败**：检查网络连接，或使用代理
- **Git未安装**：下载并安装Git
- **Python未安装**：下载并安装Python 3.7+

## 📝 开发说明

### 架构设计
- **模块化**：算法选择和参数调节分离
- **可扩展**：易于添加新算法和参数
- **用户友好**：直观的操作界面

### 性能优化
- **实时处理**：参数调节时立即更新效果
- **内存管理**：合理使用OpenCV Mat对象
- **渲染优化**：高效的图像显示机制

### 依赖管理
- **自动检测**：智能检测依赖是否存在
- **自动下载**：一键下载所有必需依赖
- **版本控制**：使用固定版本确保稳定性

## 🎯 项目结构

```
gui/
├── simple_gui.cpp           # 简化版GUI
├── main_gui.cpp            # 高级版GUI
├── CMakeLists.txt          # 构建配置
├── setup_dependencies.py   # 依赖管理脚本
├── quick_start.bat         # Windows快速启动
├── quick_start.sh          # Linux/macOS快速启动
└── README.md               # 说明文档

third_party/                # 自动下载的依赖
├── imgui/                  # ImGui库
├── glfw/                   # GLFW库
└── CMakeLists.txt          # 自动生成的配置
```
