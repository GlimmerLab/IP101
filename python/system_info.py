#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP101 System Information Collection Tool

提供跨平台的系统信息收集功能，包括：
- CPU信息（型号、核心数、频率等）
- 内存信息（总内存、可用内存等）
- OpenCV版本信息
- 操作系统信息
"""

import platform
import psutil
import cv2
import subprocess
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CPUInfo:
    """CPU信息结构体"""
    vendor: str
    model: str
    cores: int
    threads: int
    frequency_mhz: float
    architecture: str
    features: List[str]


@dataclass
class MemoryInfo:
    """内存信息结构体"""
    total_physical_kb: int
    available_physical_kb: int
    total_virtual_kb: int
    available_virtual_kb: int


@dataclass
class SystemInfoData:
    """系统信息结构体"""
    os_name: str
    os_version: str
    os_architecture: str
    cpu: CPUInfo
    memory: MemoryInfo
    opencv_version: str
    compiler_info: str


class SystemInfo:
    """系统信息收集类"""

    @staticmethod
    def get_system_info() -> SystemInfoData:
        """获取完整的系统信息"""
        return SystemInfoData(
            os_name=SystemInfo.get_os_name(),
            os_version=SystemInfo.get_os_version(),
            os_architecture=SystemInfo.get_architecture(),
            cpu=SystemInfo.get_cpu_info(),
            memory=SystemInfo.get_memory_info(),
            opencv_version=SystemInfo.get_opencv_version(),
            compiler_info=SystemInfo.get_compiler_info()
        )

    @staticmethod
    def get_cpu_info() -> CPUInfo:
        """获取CPU信息"""
        # 获取CPU型号
        model = SystemInfo._get_cpu_model()

        # 获取核心数
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)

        # 获取CPU频率
        freq = psutil.cpu_freq()
        frequency_mhz = freq.current if freq else 0.0

        # 获取架构
        architecture = SystemInfo.get_architecture()

        # 获取CPU特性
        features = SystemInfo._get_cpu_features()

        return CPUInfo(
            vendor=SystemInfo._get_cpu_vendor(),
            model=model,
            cores=cores,
            threads=threads,
            frequency_mhz=frequency_mhz,
            architecture=architecture,
            features=features
        )

    @staticmethod
    def get_memory_info() -> MemoryInfo:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return MemoryInfo(
            total_physical_kb=int(memory.total / 1024),
            available_physical_kb=int(memory.available / 1024),
            total_virtual_kb=int((memory.total + swap.total) / 1024),
            available_virtual_kb=int((memory.available + swap.free) / 1024)
        )

    @staticmethod
    def get_opencv_version() -> str:
        """获取OpenCV版本信息"""
        return cv2.__version__

    @staticmethod
    def get_os_name() -> str:
        """获取操作系统名称"""
        return platform.system()

    @staticmethod
    def get_os_version() -> str:
        """获取操作系统版本"""
        return platform.release()

    @staticmethod
    def get_architecture() -> str:
        """获取系统架构"""
        return platform.machine()

    @staticmethod
    def get_compiler_info() -> str:
        """获取编译器信息"""
        return f"Python {sys.version}"

    @staticmethod
    def format_system_info(info: SystemInfoData) -> str:
        """格式化系统信息为字符串"""
        lines = []

        lines.append("=== System Information ===")
        lines.append(f"OS: {info.os_name} {info.os_version} ({info.os_architecture})")
        lines.append(f"Compiler: {info.compiler_info}")
        lines.append(f"OpenCV: {info.opencv_version}")
        lines.append("")

        lines.append("=== CPU Information ===")
        lines.append(f"Model: {info.cpu.model}")
        lines.append(f"Vendor: {info.cpu.vendor}")
        lines.append(f"Cores: {info.cpu.cores} physical, {info.cpu.threads} logical")
        lines.append(f"Frequency: {info.cpu.frequency_mhz:.2f} MHz")
        lines.append(f"Architecture: {info.cpu.architecture}")

        if info.cpu.features:
            lines.append(f"Features: {', '.join(info.cpu.features)}")
        lines.append("")

        lines.append("=== Memory Information ===")
        lines.append(f"Physical Memory: {info.memory.total_physical_kb // 1024 // 1024} GB total, "
                    f"{info.memory.available_physical_kb // 1024 // 1024} GB available")
        lines.append(f"Virtual Memory: {info.memory.total_virtual_kb // 1024 // 1024} GB total, "
                    f"{info.memory.available_virtual_kb // 1024 // 1024} GB available")

        return "\n".join(lines)

    @staticmethod
    def has_cpu_feature(feature: str) -> bool:
        """检查CPU是否支持特定特性"""
        features = SystemInfo._get_cpu_features()
        return feature in features

    @staticmethod
    def _get_cpu_model() -> str:
        """获取CPU型号"""
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, check=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    return lines[1].strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        elif platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
            except FileNotFoundError:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, check=True
                )
                return result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        return "Unknown CPU"

    @staticmethod
    def _get_cpu_vendor() -> str:
        """获取CPU厂商"""
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "manufacturer"],
                    capture_output=True, text=True, check=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    return lines[1].strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        elif platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('vendor_id'):
                            return line.split(':')[1].strip()
            except FileNotFoundError:
                pass
        elif platform.system() == "Darwin":  # macOS
            return "Apple"

        return "Unknown"

    @staticmethod
    def _get_cpu_features() -> List[str]:
        """获取CPU特性"""
        features = []

        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    if 'mmx' in content.lower():
                        features.append('MMX')
                    if 'sse' in content.lower():
                        features.append('SSE')
                    if 'sse2' in content.lower():
                        features.append('SSE2')
                    if 'sse3' in content.lower():
                        features.append('SSE3')
                    if 'ssse3' in content.lower():
                        features.append('SSSE3')
                    if 'sse4_1' in content.lower():
                        features.append('SSE4.1')
                    if 'sse4_2' in content.lower():
                        features.append('SSE4.2')
                    if 'avx' in content.lower():
                        features.append('AVX')
                    if 'avx2' in content.lower():
                        features.append('AVX2')
            except FileNotFoundError:
                pass
        elif platform.system() == "Darwin":  # macOS
            # macOS下简化处理
            features.extend(['SSE2', 'SSE3', 'SSSE3', 'SSE4.1', 'SSE4.2'])
        elif platform.system() == "Windows":
            # Windows下简化处理
            features.extend(['SSE2', 'SSE3', 'SSSE3', 'SSE4.1', 'SSE4.2'])

        return features


def get_test_environment_info() -> str:
    """获取测试环境信息的便捷函数"""
    """获取测试环境信息

    Returns:
        格式化的测试环境信息字符串
    """
    info = SystemInfo.get_system_info()
    return SystemInfo.format_system_info(info)


def get_image_info(image_path: str) -> str:
    """获取图像信息

    Args:
        image_path: 图像文件路径

    Returns:
        图像信息字符串
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Cannot read image {image_path}"

        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1

        info_lines = []
        info_lines.append(f"Image: {image_path}")
        info_lines.append(f"Size: {width} x {height} pixels")
        info_lines.append(f"Channels: {channels}")
        info_lines.append(f"Data type: {img.dtype}")
        info_lines.append(f"Memory usage: {img.nbytes / 1024:.2f} KB")

        return "\n".join(info_lines)
    except Exception as e:
        return f"Error getting image info: {str(e)}"


if __name__ == "__main__":
    # 示例用法
    print("=== IP101 System Information ===")
    print(get_test_environment_info())

    # 如果有图像文件，显示图像信息
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print("\n=== Image Information ===")
        print(get_image_info(image_path))
