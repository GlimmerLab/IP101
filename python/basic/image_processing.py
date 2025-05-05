import cv2
import numpy as np
from ctypes import cdll, c_int, c_float, POINTER, c_void_p
import os

class ImageProcessor:
    def __init__(self):
        # 加载动态库
        lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'libimage_processing.so')
        self.lib = cdll.LoadLibrary(lib_path)

        # 定义函数参数类型
        self.lib.gaussianBlur.argtypes = [c_void_p, c_int, c_int, c_int, c_float]
        self.lib.gaussianBlur.restype = c_void_p

        self.lib.sobelEdge.argtypes = [c_void_p, c_int, c_int, c_int]
        self.lib.sobelEdge.restype = c_void_p

        self.lib.rotateImage.argtypes = [c_void_p, c_int, c_int, c_int, c_float, c_float, c_float]
        self.lib.rotateImage.restype = c_void_p

        self.lib.rgbToGray.argtypes = [c_void_p, c_int, c_int, c_int]
        self.lib.rgbToGray.restype = c_void_p

        self.lib.freeImage.argtypes = [c_void_p]
        self.lib.freeImage.restype = None

    def _numpy_to_cpp(self, img):
        """将NumPy数组转换为C++可用的格式"""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img.ctypes.data_as(POINTER(c_void_p))

    def _cpp_to_numpy(self, ptr, rows, cols, channels):
        """将C++返回的指针转换为NumPy数组"""
        if channels == 1:
            shape = (rows, cols)
        else:
            shape = (rows, cols, channels)
        arr = np.ctypeslib.as_array(ptr, shape=shape)
        return arr.copy()

    def gaussian_blur(self, img, ksize=5, sigma=1.0):
        """高斯滤波"""
        rows, cols = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # 调用C++函数
        ptr = self.lib.gaussianBlur(
            self._numpy_to_cpp(img),
            rows, cols, channels,
            c_float(sigma)
        )

        # 转换结果
        result = self._cpp_to_numpy(ptr, rows, cols, channels)
        self.lib.freeImage(ptr)
        return result

    def sobel_edge(self, img):
        """Sobel边缘检测"""
        rows, cols = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # 调用C++函数
        ptr = self.lib.sobelEdge(
            self._numpy_to_cpp(img),
            rows, cols, channels
        )

        # 转换结果
        result = self._cpp_to_numpy(ptr, rows, cols, 1)
        self.lib.freeImage(ptr)
        return result

    def rotate_image(self, img, angle, center=None):
        """旋转图像"""
        rows, cols = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        if center is None:
            center = (cols/2, rows/2)

        # 调用C++函数
        ptr = self.lib.rotateImage(
            self._numpy_to_cpp(img),
            rows, cols, channels,
            c_float(angle),
            c_float(center[0]),
            c_float(center[1])
        )

        # 转换结果
        result = self._cpp_to_numpy(ptr, rows, cols, channels)
        self.lib.freeImage(ptr)
        return result

    def rgb_to_gray(self, img):
        """RGB转灰度"""
        rows, cols = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # 调用C++函数
        ptr = self.lib.rgbToGray(
            self._numpy_to_cpp(img),
            rows, cols, channels
        )

        # 转换结果
        result = self._cpp_to_numpy(ptr, rows, cols, 1)
        self.lib.freeImage(ptr)
        return result

# 使用示例
if __name__ == "__main__":
    # 读取图像
    img = cv2.imread("input.jpg")

    # 创建图像处理器
    processor = ImageProcessor()

    # 应用高斯滤波
    blurred = processor.gaussian_blur(img, 5, 1.0)

    # 应用Sobel边缘检测
    edges = processor.sobel_edge(blurred)

    # 旋转图像
    rotated = processor.rotate_image(img, 45)

    # RGB转灰度
    gray = processor.rgb_to_gray(img)

    # 保存结果
    cv2.imwrite("blurred.jpg", blurred)
    cv2.imwrite("edges.jpg", edges)
    cv2.imwrite("rotated.jpg", rotated)
    cv2.imwrite("gray.jpg", gray)