#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP101 System Information Demo

展示如何使用系统信息收集工具
"""

import cv2
import numpy as np
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from system_info import SystemInfo, get_test_environment_info, get_image_info


def main():
    print("=== IP101 System Information Demo ===")

    # 获取完整的系统信息
    sys_info = SystemInfo.get_system_info()

    # 显示格式化的系统信息
    print(SystemInfo.format_system_info(sys_info))

    # 检查特定CPU特性
    print("=== CPU Feature Check ===")
    features_to_check = ["SSE2", "SSE3", "AVX", "AVX2"]

    for feature in features_to_check:
        has_feature = SystemInfo.has_cpu_feature(feature)
        status = "✓ Supported" if has_feature else "✗ Not Supported"
        print(f"{feature}: {status}")

    # 获取图像信息示例
    print("\n=== Image Information Example ===")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    print(f"Test Image Size: {test_image.shape[1]} x {test_image.shape[0]} pixels")
    print(f"Test Image Channels: {test_image.shape[2]}")
    print(f"Test Image Memory Usage: {test_image.nbytes / 1024:.2f} KB")

    # 获取OpenCV版本
    print("\n=== OpenCV Version ===")
    print(f"OpenCV Version: {SystemInfo.get_opencv_version()}")

    # 使用便捷函数
    print("\n=== Using Convenience Functions ===")
    print("Test Environment Info:")
    print(get_test_environment_info())

    # 如果有图像文件，显示图像信息
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nImage Info for {image_path}:")
        print(get_image_info(image_path))


if __name__ == "__main__":
    main()
