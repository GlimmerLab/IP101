@echo off
chcp 65001 >nul
echo ========================================
echo IP101 GUI 快速启动工具
echo ========================================

:: 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python，请先安装Python
    echo    下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 检查Git是否可用
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Git，请先安装Git
    echo    下载地址：https://git-scm.com/downloads
    pause
    exit /b 1
)

echo ✅ 环境检查通过

:: 检查依赖是否存在
if not exist "..\third_party\imgui" (
    echo.
    echo 📦 检测到依赖缺失，正在自动下载...
    python setup_dependencies.py
    if errorlevel 1 (
        echo ❌ 依赖下载失败，请检查网络连接
        pause
        exit /b 1
    )
) else (
    echo ✅ 依赖已存在
)

:: 构建项目
echo.
echo 🔨 正在构建GUI项目...
cd ..
if exist "build" (
    cd build
) else (
    mkdir build
    cd build
    cmake ..
    if errorlevel 1 (
        echo ❌ CMake配置失败
        pause
        exit /b 1
    )
)

:: 编译
cmake --build . --config Release
if errorlevel 1 (
    echo ❌ 编译失败
    pause
    exit /b 1
)

echo ✅ 构建完成！

:: 运行GUI
echo.
echo 🚀 启动GUI程序...
if exist "Release\simple_gui.exe" (
    echo 启动简化版GUI...
    start "" "Release\simple_gui.exe"
) else (
    echo 启动高级版GUI...
    start "" "Release\main_gui.exe"
)

echo.
echo 🎉 GUI启动成功！
echo.
echo 📝 使用说明：
echo    - 简化版GUI：按1-6选择算法，l键加载图像，s键保存，q键退出
echo    - 高级版GUI：使用鼠标和界面控件操作
echo.
pause
