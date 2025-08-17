#!/bin/bash

echo "========================================"
echo "IP101 GUI 快速启动工具"
echo "========================================"

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到Python3，请先安装Python"
    echo "   下载地址：https://www.python.org/downloads/"
    exit 1
fi

# 检查Git是否可用
if ! command -v git &> /dev/null; then
    echo "❌ 错误：未找到Git，请先安装Git"
    echo "   下载地址：https://git-scm.com/downloads"
    exit 1
fi

echo "✅ 环境检查通过"

# 检查依赖是否存在
if [ ! -d "../third_party/imgui" ]; then
    echo ""
    echo "📦 检测到依赖缺失，正在自动下载..."
    python3 setup_dependencies.py
    if [ $? -ne 0 ]; then
        echo "❌ 依赖下载失败，请检查网络连接"
        exit 1
    fi
else
    echo "✅ 依赖已存在"
fi

# 构建项目
echo ""
echo "🔨 正在构建GUI项目..."
cd ..
if [ -d "build" ]; then
    cd build
else
    mkdir build
    cd build
    cmake ..
    if [ $? -ne 0 ]; then
        echo "❌ CMake配置失败"
        exit 1
    fi
fi

# 编译
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo "✅ 构建完成！"

# 运行GUI
echo ""
echo "🚀 启动GUI程序..."
if [ -f "simple_gui" ]; then
    echo "启动简化版GUI..."
    ./simple_gui &
elif [ -f "main_gui" ]; then
    echo "启动高级版GUI..."
    ./main_gui &
else
    echo "❌ 未找到GUI可执行文件"
    exit 1
fi

echo ""
echo "🎉 GUI启动成功！"
echo ""
echo "📝 使用说明："
echo "   - 简化版GUI：按1-6选择算法，l键加载图像，s键保存，q键退出"
echo "   - 高级版GUI：使用鼠标和界面控件操作"
echo ""
