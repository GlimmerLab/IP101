import unittest
import cv2
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from basic.image_processing import ImageProcessor

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # 创建测试图像
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (20, 20), (80, 80), (255, 255, 255), -1)
        self.processor = ImageProcessor()

    def test_gaussian_blur(self):
        # 测试高斯滤波
        result = self.processor.gaussian_blur(self.test_image, 5, 1.0)

        # 检查结果图像大小
        self.assertEqual(result.shape[:2], self.test_image.shape[:2])

        # 检查结果图像范围
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_sobel_edge(self):
        # 测试Sobel边缘检测
        result = self.processor.sobel_edge(self.test_image)

        # 检查结果图像大小
        self.assertEqual(result.shape[:2], self.test_image.shape[:2])
        self.assertEqual(len(result.shape), 2)  # 灰度图像

        # 检查结果图像范围
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_rotate_image(self):
        # 测试图像旋转
        result = self.processor.rotate_image(self.test_image, 45)

        # 检查结果图像大小
        self.assertEqual(result.shape[:2], self.test_image.shape[:2])
        self.assertEqual(result.shape[2], self.test_image.shape[2])

        # 检查结果图像范围
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_rgb_to_gray(self):
        # 测试RGB转灰度
        result = self.processor.rgb_to_gray(self.test_image)

        # 检查结果图像大小
        self.assertEqual(result.shape[:2], self.test_image.shape[:2])
        self.assertEqual(len(result.shape), 2)  # 灰度图像

        # 检查结果图像范围
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_performance(self):
        import time

        # 测试高斯滤波性能
        start = time.time()
        for _ in range(100):
            self.processor.gaussian_blur(self.test_image, 5, 1.0)
        gaussian_time = time.time() - start
        print(f"Gaussian blur average time: {gaussian_time/100:.3f} ms")

        # 测试Sobel边缘检测性能
        start = time.time()
        for _ in range(100):
            self.processor.sobel_edge(self.test_image)
        sobel_time = time.time() - start
        print(f"Sobel edge detection average time: {sobel_time/100:.3f} ms")

        # 测试图像旋转性能
        start = time.time()
        for _ in range(100):
            self.processor.rotate_image(self.test_image, 45)
        rotate_time = time.time() - start
        print(f"Image rotation average time: {rotate_time/100:.3f} ms")

        # 测试RGB转灰度性能
        start = time.time()
        for _ in range(100):
            self.processor.rgb_to_gray(self.test_image)
        gray_time = time.time() - start
        print(f"RGB to gray conversion average time: {gray_time/100:.3f} ms")

if __name__ == '__main__':
    unittest.main()