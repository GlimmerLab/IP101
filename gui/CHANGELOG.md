# IP101 GUI 更新日志

## v2.0.0 (2024-12-19)

### 🎨 现代化界面升级
- **Photoshop风格界面**：专业的深色主题设计
- **多窗口布局**：支持窗口停靠和重新排列
- **菜单栏系统**：File、View、Help 标准菜单
- **工具栏**：快速操作按钮（Open、Save、Reset、Apply）
- **状态栏**：实时显示图像信息和当前算法
- **分类标签页**：算法按类别组织（Filters、Correction、Effects、Defogging）

### 🔧 技术改进
- **现代OpenGL**：使用OpenGL 3.3 Core Profile
- **响应式布局**：自适应窗口大小
- **实时参数调整**：滑块控制实时预览效果
- **算法参数修复**：修正了所有算法调用的参数问题
- **OpenGL兼容性**：修复了OpenGL常量定义问题

## v1.0.0 (2024-12-19)

### 🎉 新功能
- **智能依赖管理**：自动检测和下载ImGui、GLFW依赖
- **双版本GUI**：简化版（OpenCV内置）和高级版（ImGui）
- **跨平台支持**：Windows、Linux、macOS一键启动
- **快速启动脚本**：自动环境检查和构建

### 🔧 技术改进
- **模块化设计**：GUI独立为根目录模块
- **自动CMake配置**：智能生成third_party配置
- **版本控制**：固定依赖版本确保稳定性
- **错误处理**：完善的错误提示和故障排除

### 📁 目录结构
```
gui/
├── simple_gui.cpp           # 简化版GUI（零依赖）
├── main_gui.cpp            # 高级版GUI（ImGui）
├── CMakeLists.txt          # 智能构建配置
├── setup_dependencies.py   # 依赖管理脚本
├── quick_start.bat         # Windows快速启动
├── quick_start.sh          # Linux/macOS快速启动
└── README.md               # 详细说明文档

third_party/                # 自动下载的依赖
├── imgui/                  # ImGui v1.89.9
├── glfw/                   # GLFW v3.3.8
└── CMakeLists.txt          # 自动生成的配置
```

### 🚀 使用方式
```bash
# Windows
gui/quick_start.bat

# Linux/macOS
./gui/quick_start.sh

# 手动安装
python gui/setup_dependencies.py
mkdir build && cd build
cmake .. && make
```

### 🎯 特性对比

| 特性 | 简化版GUI | 高级版GUI |
|------|-----------|-----------|
| 依赖 | 仅OpenCV | ImGui + GLFW |
| 界面 | 基础窗口 | 现代化界面 |
| 操作 | 键盘控制 | 鼠标+键盘 |
| 启动速度 | 极快 | 快 |
| 功能丰富度 | 基础 | 丰富 |

### 🔍 算法支持
- ✅ Guided Filter (引导滤波)
- ✅ Side Window Filter (侧窗滤波)
- ✅ Gamma Correction (伽马校正)
- ✅ Dark Channel Defogging (暗通道去雾)
- ✅ Cartoon Effect (卡通效果)
- ✅ Oil Painting Effect (油画效果)

### 🛠️ 开发工具
- **依赖检查**：`python gui/setup_dependencies.py`
- **CMake目标**：`check_gui_deps`, `download_gui_deps`
- **自动构建**：智能检测依赖状态

### 📝 文档
- 详细的使用说明
- 故障排除指南
- 开发扩展指南
- 跨平台兼容性说明
