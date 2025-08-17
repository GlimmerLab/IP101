#!/usr/bin/env python3
"""
IP101 GUI 依赖管理脚本
自动下载和管理ImGui、GLFW等依赖库
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

# 依赖配置
DEPENDENCIES = {
    "imgui": {
        "url": "https://github.com/ocornut/imgui.git",
        "branch": "master",
        "dir": "imgui",
        "files": ["imgui.h", "imgui.cpp", "imgui_demo.cpp", "imgui_draw.cpp", "imgui_tables.cpp", "imgui_widgets.cpp"],
        "backends": ["imgui_impl_glfw.h", "imgui_impl_glfw.cpp", "imgui_impl_opengl3.h", "imgui_impl_opengl3.cpp"]
    },
    "glfw": {
        "url": "https://github.com/glfw/glfw.git",
        "branch": "master",
        "dir": "glfw",
        "files": ["include/GLFW/glfw3.h", "src/glfw_config.h", "src/context.c", "src/init.c", "src/input.c", "src/monitor.c", "src/vulkan.c", "src/window.c"]
    }
}

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

def download_dependency(name, config):
    """下载单个依赖"""
    dep_dir = THIRD_PARTY_DIR / config["dir"]

    print(f"\n📦 正在处理 {name}...")

    # 检查是否已存在
    if dep_dir.exists():
        print(f"   📁 {name} 已存在于 {dep_dir}")

        # 更新到最新版本
        if "branch" in config:
            print(f"   🔄 正在更新到 {config['branch']} 分支最新版本...")
            success, stdout, stderr = run_command(
                f'git fetch origin && git reset --hard origin/{config["branch"]}',
                cwd=dep_dir
            )
            if not success:
                print(f"   ❌ 更新失败：{stderr}")
                return False
            print(f"   ✅ {name} 已更新到最新版本")

        return True

    # 创建目录
    dep_dir.mkdir(parents=True, exist_ok=True)

    # 克隆仓库
    print(f"   🔄 正在克隆 {config['url']}...")
    success, stdout, stderr = run_command(
        f'git clone {config["url"]} .',
        cwd=dep_dir
    )

    if not success:
        print(f"   ❌ 克隆失败：{stderr}")
        shutil.rmtree(dep_dir, ignore_errors=True)
        return False

    # 切换到指定分支
    if "branch" in config:
        print(f"   🌿 切换到分支 {config['branch']}...")
        success, stdout, stderr = run_command(
            f'git checkout {config["branch"]}',
            cwd=dep_dir
        )
        if not success:
            print(f"   ❌ 切换分支失败：{stderr}")
            return False

    print(f"   ✅ {name} 下载完成")
    return True

def verify_dependency(name, config):
    """验证依赖文件是否存在"""
    dep_dir = THIRD_PARTY_DIR / config["dir"]

    if not dep_dir.exists():
        return False

    # 检查关键文件
    missing_files = []
    for file in config.get("files", []):
        file_path = dep_dir / file
        if not file_path.exists():
            missing_files.append(file)

    if missing_files:
        print(f"   ⚠️  缺少文件：{', '.join(missing_files)}")
        return False

    return True

def create_cmake_config():
    """创建CMake配置文件"""
    cmake_file = THIRD_PARTY_DIR / "CMakeLists.txt"

    content = """# Third-party dependencies CMakeLists.txt
# 自动生成，请勿手动修改

# ImGui
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
    set(IMGUI_SOURCES
        imgui/imgui.cpp
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp
        imgui/imgui_impl_glfw.cpp
        imgui/imgui_impl_opengl3.cpp
    )

    add_library(imgui STATIC ${IMGUI_SOURCES})
    target_include_directories(imgui PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
    )

    # 编译定义
    target_compile_definitions(imgui PRIVATE
        IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    )

    # 查找并链接GLFW
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW QUIET glfw3)
    endif()

    if(GLFW_FOUND)
        target_link_libraries(imgui ${GLFW_LIBRARIES})
        target_include_directories(imgui PUBLIC ${GLFW_INCLUDE_DIRS})
    else()
        # 使用本地GLFW
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glfw")
            add_subdirectory(glfw)
            target_link_libraries(imgui glfw)
        else()
            # 查找系统GLFW
            find_library(GLFW_LIBRARY NAMES glfw3 glfw)
            if(GLFW_LIBRARY)
                target_link_libraries(imgui ${GLFW_LIBRARY})
            endif()
        endif()
    endif()

    # 查找OpenGL
    find_package(OpenGL REQUIRED)
    target_link_libraries(imgui OpenGL::GL)
endif()

# GLFW (如果ImGui未使用本地GLFW)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glfw" AND NOT TARGET glfw)
    add_subdirectory(glfw)
endif()
"""

    with open(cmake_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 创建CMake配置：{cmake_file}")

def main():
    """主函数"""
    print("🚀 IP101 GUI 依赖管理工具")
    print("📦 使用 master 分支保持依赖最新")
    print("=" * 50)

    # 检查git
    if not check_git():
        return 1

    # 创建third_party目录
    THIRD_PARTY_DIR.mkdir(exist_ok=True)

    # 下载所有依赖
    all_success = True
    for name, config in DEPENDENCIES.items():
        if not download_dependency(name, config):
            all_success = False
            continue

        if not verify_dependency(name, config):
            print(f"   ⚠️  {name} 验证失败，可能需要手动检查")

    # 创建CMake配置
    if all_success:
        create_cmake_config()
        print(f"\n🎉 所有依赖下载/更新完成！")
        print(f"📁 依赖位置：{THIRD_PARTY_DIR}")
        print(f"🔄 所有依赖已更新到 master 分支最新版本")
        print(f"\n📝 使用说明：")
        print(f"   1. 在CMakeLists.txt中添加：add_subdirectory(third_party)")
        print(f"   2. 链接库：target_link_libraries(your_target imgui)")
        print(f"   3. 包含目录会自动设置")
        print(f"   4. 重新运行此脚本可更新依赖到最新版本")
    else:
        print(f"\n❌ 部分依赖下载失败，请检查网络连接或手动下载")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
