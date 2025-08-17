#!/bin/bash

echo "🚀 IP101 GUI 依赖快速设置工具"
echo "================================================"

# 检查 Git 是否可用
if ! command -v git &> /dev/null; then
    echo "❌ 错误：未找到 Git，请先安装 Git"
    echo "   下载地址：https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git 已安装"

# 检查是否在 Git 仓库中，如果不是则自动初始化
if [ ! -d ".git" ]; then
    echo "⚠️  当前目录不是 Git 仓库，正在自动初始化..."
    if ! git init; then
        echo "❌ Git 仓库初始化失败"
        exit 1
    fi
    echo "✅ Git 仓库初始化成功"
else
    echo "✅ 当前目录是 Git 仓库"
fi

# 检查 .gitmodules 文件，如果不存在则自动创建
if [ ! -f ".gitmodules" ]; then
    echo "⚠️  未找到 .gitmodules 文件，正在自动创建..."

    # 创建 .gitmodules 文件
    cat > .gitmodules << 'EOF'
[submodule "third_party/imgui"]
	path = third_party/imgui
	url = https://github.com/ocornut/imgui.git
	branch = master

[submodule "third_party/glfw"]
	path = third_party/glfw
	url = https://github.com/glfw/glfw.git
	branch = master
EOF

    echo "✅ .gitmodules 文件创建成功"
else
    echo "✅ 找到 .gitmodules 文件"
fi

# 创建 third_party 目录（如果不存在）
if [ ! -d "third_party" ]; then
    echo "📁 创建 third_party 目录..."
    mkdir -p third_party
fi

# 初始化子模块
echo "🔄 正在初始化 Git 子模块..."
if ! git submodule init; then
    echo "❌ 子模块初始化失败"
    exit 1
fi

# 克隆子模块
echo "🔄 正在克隆子模块..."
if ! git submodule update --init --recursive; then
    echo "❌ 子模块克隆失败"
    exit 1
fi

# 更新到最新版本
echo "🔄 正在更新到 master 分支最新版本..."
if ! git submodule update --remote --merge; then
    echo "❌ 子模块更新失败"
    exit 1
fi

echo ""
echo "🎉 GUI 依赖设置完成！"
echo "📁 依赖位置：third_party/"
echo "🔄 所有依赖已更新到 master 分支最新版本"
echo ""
echo "📝 现在可以运行 CMake 构建项目了"
echo ""
