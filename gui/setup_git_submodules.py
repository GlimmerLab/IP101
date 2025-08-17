#!/usr/bin/env python3
"""
IP101 GUI Git 子模块管理脚本
使用 Git 子模块管理 ImGui、GLFW 等依赖库
"""

import os
import sys
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def run_command(cmd, cwd=None, check=True):
    """运行命令并处理错误"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_git():
    """检查git是否可用"""
    success, stdout, stderr = run_command("git --version", check=False)
    if not success:
        print("❌ 错误：未找到git，请先安装git")
        print("   下载地址：https://git-scm.com/downloads")
        return False
    print(f"✅ Git版本：{stdout.strip()}")
    return True

def init_submodules():
    """初始化子模块"""
    print("🔄 正在初始化 Git 子模块...")

    # 初始化子模块
    success, stdout, stderr = run_command("git submodule init", cwd=PROJECT_ROOT)
    if not success:
        print(f"❌ 子模块初始化失败：{stderr}")
        return False

    print("✅ 子模块初始化完成")
    return True

def update_submodules():
    """更新子模块到最新版本"""
    print("🔄 正在更新子模块到 master 分支最新版本...")

    # 更新子模块
    success, stdout, stderr = run_command("git submodule update --remote --merge", cwd=PROJECT_ROOT)
    if not success:
        print(f"❌ 子模块更新失败：{stderr}")
        return False

    print("✅ 子模块更新完成")
    return True

def clone_submodules():
    """克隆子模块"""
    print("🔄 正在克隆子模块...")

    # 克隆子模块
    success, stdout, stderr = run_command("git submodule update --init --recursive", cwd=PROJECT_ROOT)
    if not success:
        print(f"❌ 子模块克隆失败：{stderr}")
        return False

    print("✅ 子模块克隆完成")
    return True

def verify_submodules():
    """验证子模块状态"""
    print("🔍 验证子模块状态...")

    # 检查子模块状态
    success, stdout, stderr = run_command("git submodule status", cwd=PROJECT_ROOT)
    if not success:
        print(f"❌ 无法获取子模块状态：{stderr}")
        return False

    print("📋 子模块状态：")
    for line in stdout.strip().split('\n'):
        if line.strip():
            print(f"   {line}")

    return True

def main():
    """主函数"""
    print("🚀 IP101 GUI Git 子模块管理工具")
    print("📦 使用 Git 子模块管理依赖，保持 master 分支最新")
    print("=" * 60)

    # 检查git
    if not check_git():
        return 1

    # 检查是否在 Git 仓库中
    if not (PROJECT_ROOT / ".git").exists():
        print("❌ 错误：当前目录不是 Git 仓库")
        print("   请先初始化 Git 仓库：git init")
        return 1

    # 检查 .gitmodules 文件
    gitmodules_file = PROJECT_ROOT / ".gitmodules"
    if not gitmodules_file.exists():
        print("❌ 错误：未找到 .gitmodules 文件")
        print("   请确保已正确配置子模块")
        return 1

    print(f"✅ 找到 .gitmodules 文件：{gitmodules_file}")

    # 初始化子模块
    if not init_submodules():
        return 1

    # 克隆子模块
    if not clone_submodules():
        return 1

    # 更新子模块到最新版本
    if not update_submodules():
        return 1

    # 验证子模块状态
    if not verify_submodules():
        return 1

    print(f"\n🎉 Git 子模块管理完成！")
    print(f"📁 依赖位置：{PROJECT_ROOT / 'third_party'}")
    print(f"🔄 所有依赖已更新到 master 分支最新版本")
    print(f"\n📝 使用说明：")
    print(f"   1. 在 CMakeLists.txt 中添加：add_subdirectory(third_party)")
    print(f"   2. 链接库：target_link_libraries(your_target imgui)")
    print(f"   3. 包含目录会自动设置")
    print(f"\n🔄 更新依赖命令：")
    print(f"   git submodule update --remote --merge")
    print(f"   或运行此脚本：python gui/setup_git_submodules.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())
