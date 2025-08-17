@echo off
chcp 65001 >nul
echo 🚀 IP101 GUI 依赖快速设置工具
echo ================================================

REM 检查 Git 是否可用
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Git，请先安装 Git
    echo    下载地址：https://git-scm.com/downloads
    pause
    exit /b 1
)

echo ✅ Git 已安装

REM 检查是否在 Git 仓库中，如果不是则自动初始化
if not exist ".git" (
    echo ⚠️  当前目录不是 Git 仓库，正在自动初始化...
    git init
    if errorlevel 1 (
        echo ❌ Git 仓库初始化失败
        pause
        exit /b 1
    )
    echo ✅ Git 仓库初始化成功
) else (
    echo ✅ 当前目录是 Git 仓库
)

REM 检查 .gitmodules 文件，如果不存在则自动创建
if not exist ".gitmodules" (
    echo ⚠️  未找到 .gitmodules 文件，正在自动创建...

    REM 创建 .gitmodules 文件
    echo [submodule "third_party/imgui"] > .gitmodules
    echo 	path = third_party/imgui >> .gitmodules
    echo 	url = https://github.com/ocornut/imgui.git >> .gitmodules
    echo 	branch = master >> .gitmodules
    echo. >> .gitmodules
    echo [submodule "third_party/glfw"] >> .gitmodules
    echo 	path = third_party/glfw >> .gitmodules
    echo 	url = https://github.com/glfw/glfw.git >> .gitmodules
    echo 	branch = master >> .gitmodules

    echo ✅ .gitmodules 文件创建成功
) else (
    echo ✅ 找到 .gitmodules 文件
)

REM 创建 third_party 目录（如果不存在）
if not exist "third_party" (
    echo 📁 创建 third_party 目录...
    mkdir third_party
)

REM 初始化子模块
echo 🔄 正在初始化 Git 子模块...
git submodule init
if errorlevel 1 (
    echo ❌ 子模块初始化失败
    pause
    exit /b 1
)

REM 克隆子模块
echo 🔄 正在克隆子模块...
git submodule update --init --recursive
if errorlevel 1 (
    echo ❌ 子模块克隆失败
    pause
    exit /b 1
)

REM 更新到最新版本
echo 🔄 正在更新到 master 分支最新版本...
git submodule update --remote --merge
if errorlevel 1 (
    echo ❌ 子模块更新失败
    pause
    exit /b 1
)

echo.
echo 🎉 GUI 依赖设置完成！
echo 📁 依赖位置：third_party/
echo 🔄 所有依赖已更新到 master 分支最新版本
echo.
echo 📝 现在可以运行 CMake 构建项目了
echo.
pause
