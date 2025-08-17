#!/usr/bin/env python3
"""
IP101 图像处理库构建脚本
支持跨平台构建，自动检测环境并配置最优编译选项
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

class IP101Builder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.install_dir = self.project_root / "install"

        # 平台信息
        self.system = platform.system()
        self.machine = platform.machine()
        self.is_windows = self.system == "Windows"
        self.is_linux = self.system == "Linux"
        self.is_macos = self.system == "Darwin"

        # 编译器信息
        self.compiler = self._detect_compiler()

    def _detect_compiler(self):
        """检测可用的编译器"""
        if self.is_windows:
            # 优先使用MSVC
            try:
                result = subprocess.run(["cl"], capture_output=True, text=True)
                if result.returncode == 0:
                    return "MSVC"
            except FileNotFoundError:
                pass

            # 尝试使用MinGW
            try:
                result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return "MinGW"
            except FileNotFoundError:
                pass

            return "Unknown"
        else:
            # Linux/macOS优先使用GCC
            try:
                result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return "GCC"
            except FileNotFoundError:
                pass

            # 尝试使用Clang
            try:
                result = subprocess.run(["clang++", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return "Clang"
            except FileNotFoundError:
                pass

            return "Unknown"

    def _check_opencv(self):
        """检查OpenCV是否可用"""
        try:
            # 尝试导入OpenCV
            import cv2
            print(f"✅ 找到OpenCV版本: {cv2.__version__}")
            return True
        except ImportError:
            print("❌ 未找到OpenCV，请先安装OpenCV")
            return False

    def _create_build_dir(self):
        """创建构建目录"""
        self.build_dir.mkdir(exist_ok=True)
        print(f"📁 构建目录: {self.build_dir}")

    def _configure_cmake(self, args):
        """配置CMake"""
        cmake_args = [
            "cmake",
            "-B", str(self.build_dir),
            "-S", str(self.project_root),
            "-DCMAKE_BUILD_TYPE=" + args.build_type,
            "-DCMAKE_INSTALL_PREFIX=" + str(self.install_dir)
        ]

        # 添加OpenCV路径（如果指定）
        if args.opencv_dir:
            cmake_args.extend(["-DOpenCV_DIR=" + args.opencv_dir])

        # 添加其他选项
        if args.enable_tests:
            cmake_args.append("-DBUILD_TESTS=ON")

        if args.enable_examples:
            cmake_args.append("-DBUILD_EXAMPLES=ON")

        if args.enable_docs:
            cmake_args.append("-DBUILD_DOCS=ON")

        print("🔧 配置CMake...")
        print(f"命令: {' '.join(cmake_args)}")

        result = subprocess.run(cmake_args, cwd=self.project_root)
        if result.returncode != 0:
            print("❌ CMake配置失败")
            return False

        print("✅ CMake配置成功")
        return True

    def _build_project(self, args):
        """构建项目"""
        print("🔨 开始构建...")

        # 确定构建命令
        if self.is_windows:
            build_cmd = ["cmake", "--build", str(self.build_dir), "--config", args.build_type]
            if args.jobs:
                build_cmd.extend(["--parallel", str(args.jobs)])
        else:
            build_cmd = ["cmake", "--build", str(self.build_dir), "--config", args.build_type]
            if args.jobs:
                build_cmd.extend(["-j", str(args.jobs)])

        print(f"命令: {' '.join(build_cmd)}")

        result = subprocess.run(build_cmd, cwd=self.project_root)
        if result.returncode != 0:
            print("❌ 构建失败")
            return False

        print("✅ 构建成功")
        return True

    def _install_project(self):
        """安装项目"""
        print("📦 安装项目...")

        install_cmd = ["cmake", "--install", str(self.build_dir)]
        result = subprocess.run(install_cmd, cwd=self.project_root)

        if result.returncode != 0:
            print("❌ 安装失败")
            return False

        print(f"✅ 安装成功，安装目录: {self.install_dir}")
        return True

    def _run_tests(self):
        """运行测试"""
        print("🧪 运行测试...")

        test_cmd = ["ctest", "--test-dir", str(self.build_dir), "--verbose"]
        result = subprocess.run(test_cmd, cwd=self.project_root)

        if result.returncode != 0:
            print("❌ 测试失败")
            return False

        print("✅ 测试通过")
        return True

    def _show_info(self):
        """显示系统信息"""
        print("=" * 50)
        print("🚀 IP101 图像处理库构建器")
        print("=" * 50)
        print(f"操作系统: {self.system}")
        print(f"架构: {self.machine}")
        print(f"编译器: {self.compiler}")
        print(f"项目根目录: {self.project_root}")
        print("=" * 50)

    def build(self, args):
        """执行完整的构建流程"""
        self._show_info()

        # 检查OpenCV
        if not self._check_opencv():
            return False

        # 创建构建目录
        self._create_build_dir()

        # 配置CMake
        if not self._configure_cmake(args):
            return False

        # 构建项目
        if not self._build_project(args):
            return False

        # 安装项目
        if args.install and not self._install_project():
            return False

        # 运行测试
        if args.test and not self._run_tests():
            return False

        print("\n🎉 构建完成！")
        print(f"📁 构建目录: {self.build_dir}")
        if args.install:
            print(f"📦 安装目录: {self.install_dir}")

        return True

def main():
    parser = argparse.ArgumentParser(description="IP101 图像处理库构建脚本")
    parser.add_argument("--build-type", choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
                       default="Release", help="构建类型")
    parser.add_argument("--opencv-dir", help="OpenCV安装目录")
    parser.add_argument("--jobs", "-j", type=int, help="并行构建任务数")
    parser.add_argument("--install", action="store_true", help="安装库文件")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--enable-tests", action="store_true", help="启用测试构建")
    parser.add_argument("--enable-examples", action="store_true", help="启用示例构建")
    parser.add_argument("--enable-docs", action="store_true", help="启用文档构建")
    parser.add_argument("--clean", action="store_true", help="清理构建目录")

    args = parser.parse_args()

    builder = IP101Builder()

    # 清理构建目录
    if args.clean:
        import shutil
        if builder.build_dir.exists():
            shutil.rmtree(builder.build_dir)
            print("🧹 构建目录已清理")
        return

    # 执行构建
    success = builder.build(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()