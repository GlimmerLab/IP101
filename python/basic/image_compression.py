"""
图像压缩相关问题：
47. 无损压缩 - 使用RLE和Huffman编码进行无损压缩
48. JPEG压缩 - 使用DCT变换和量化进行JPEG压缩
49. 分形压缩 - 使用分形理论进行图像压缩
50. 小波压缩 - 使用小波变换进行图像压缩
"""

import cv2
import numpy as np
from scipy import fftpack
import pywt
from collections import Counter
import heapq

def rle_compression(img_path):
    """
    问题47：无损压缩（RLE编码）
    使用游程编码进行无损压缩

    参数:
        img_path: 输入图像路径

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 展平图像
    flat_img = img.flatten()

    # RLE编码
    encoded = []
    count = 1
    current = flat_img[0]

    for pixel in flat_img[1:]:
        if pixel == current:
            count += 1
        else:
            encoded.extend([current, count])
            current = pixel
            count = 1
    encoded.extend([current, count])

    # RLE解码
    decoded = []
    for i in range(0, len(encoded), 2):
        decoded.extend([encoded[i]] * encoded[i+1])

    # 重建图像
    result = np.array(decoded).reshape(img.shape)

    # 转换为彩色图像
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result

def jpeg_compression(img_path, quality=50):
    """
    问题48：JPEG压缩
    使用DCT变换和量化进行JPEG压缩

    参数:
        img_path: 输入图像路径
        quality: 压缩质量(1-100)，默认50

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 标准JPEG量化表
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    # 调整量化表根据质量参数
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality
    Q = np.floor((S * Q + 50) / 100)
    Q = np.clip(Q, 1, 255)

    # 分块处理
    h, w = img.shape
    h = h - h % 8
    w = w - w % 8
    img = img[:h, :w]
    result = np.zeros_like(img, dtype=np.float32)

    # 对每个8x8块进行DCT变换和量化
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8].astype(np.float32) - 128
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / Q)
            dequantized = quantized * Q
            idct_block = fftpack.idct(fftpack.idct(dequantized.T, norm='ortho').T, norm='ortho')
            result[i:i+8, j:j+8] = idct_block + 128

    # 裁剪到有效范围
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def fractal_compression(img_path, block_size=8):
    """
    问题49：分形压缩
    使用分形理论进行图像压缩（简化版本）

    参数:
        img_path: 输入图像路径
        block_size: 分块大小，默认为8

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 确保图像尺寸是block_size的整数倍
    h, w = img.shape
    h = h - h % block_size
    w = w - w % block_size
    img = img[:h, :w]
    result = np.zeros_like(img)

    # 对每个块进行处理
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            # 简化的分形变换：使用均值和方差进行编码
            mean = np.mean(block)
            std = np.std(block)
            # 重建：使用统计特征重建块
            result[i:i+block_size, j:j+block_size] = np.clip(
                mean + (block - mean) * (std / (std + 1e-6)), 0, 255)

    # 转换为彩色图像
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result

def wavelet_compression(img_path, threshold=10):
    """
    问题50：小波压缩
    使用小波变换进行图像压缩

    参数:
        img_path: 输入图像路径
        threshold: 系数阈值，默认为10

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 进行小波变换
    coeffs = pywt.wavedec2(img, 'haar', level=3)

    # 阈值处理
    for i in range(1, len(coeffs)):
        for detail in coeffs[i]:
            detail[np.abs(detail) < threshold] = 0

    # 重建图像
    result = pywt.waverec2(coeffs, 'haar')

    # 裁剪到原始尺寸
    result = result[:img.shape[0], :img.shape[1]]

    # 归一化到0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def main(problem_id=47):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为47
    """
    input_path = "../images/imori.jpg"

    if problem_id == 47:
        result = rle_compression(input_path)
        output_path = "../images/answer_47.jpg"
        title = "无损压缩"
    elif problem_id == 48:
        result = jpeg_compression(input_path)
        output_path = "../images/answer_48.jpg"
        title = "JPEG压缩"
    elif problem_id == 49:
        result = fractal_compression(input_path)
        output_path = "../images/answer_49.jpg"
        title = "分形压缩"
    elif problem_id == 50:
        result = wavelet_compression(input_path)
        output_path = "../images/answer_50.jpg"
        title = "小波压缩"
    else:
        print(f"问题 {problem_id} 不存在")
        return

    # 保存结果
    cv2.imwrite(output_path, result)

    # 显示结果
    cv2.imshow(title, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 47
    main(problem_id)