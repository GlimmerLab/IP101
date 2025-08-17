
 @echo off
chcp 65001 >nul
echo 🚀 IP101 GUI Git 子模块管理工具
echo ================================================

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python，请先安装 Python
    echo    下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 运行 Python 脚本
python "%~dp0setup_git_submodules.py"

if errorlevel 1 (
    echo.
    echo ❌ 脚本执行失败
    pause
    exit /b 1
) else (
    echo.
    echo ✅ 脚本执行完成
    pause
)
