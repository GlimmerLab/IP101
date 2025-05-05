"""
频域处理相关问题：
43. 傅里叶变换 - 对图像进行傅里叶变换
44. 频域滤波 - 在频域上进行滤波操作
45. DCT变换 - 对图像进行离散余弦变换
46. 小波变换 - 对图像进行小波变换
"""

import cv2
import numpy as np
from scipy import fftpack
import pywt

def fourier_transform(img_path):
    """
    问题43：傅里叶变换
    对图像进行傅里叶变换

    参数:
        img_path: 输入图像路径

    返回:
        傅里叶变换结果图像（幅度谱）
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 进行傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 归一化到0-255
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 转换为彩色图像
    result = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2BGR)

    return result

def frequency_filtering(img_path, filter_type='lowpass', cutoff=30):
    """
    问题44：频域滤波
    在频域上进行滤波操作

    参数:
        img_path: 输入图像路径
        filter_type: 滤波器类型，'lowpass'或'highpass'
        cutoff: 截止频率

    返回:
        滤波后的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 进行傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 获取图像尺寸
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    # 创建滤波器
    mask = np.zeros((rows, cols), np.uint8)
    if filter_type == 'lowpass':
        cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
    elif filter_type == 'highpass':
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 0, -1)
    else:
        raise ValueError(f"不支持的滤波器类型: {filter_type}")

    # 应用滤波器
    fshift = fshift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 归一化到0-255
    result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def dct_transform(img_path, block_size=8):
    """
    问题45：DCT变换
    对图像进行离散余弦变换

    参数:
        img_path: 输入图像路径
        block_size: 分块大小，默认为8

    返回:
        DCT变换结果图像
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

    # 对每个块进行DCT变换
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            result[i:i+block_size, j:j+block_size] = dct_block

    # 取对数并归一化
    result = np.log(np.abs(result) + 1)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def wavelet_transform(img_path, wavelet='db1', level=1):
    """
    问题46：小波变换
    对图像进行小波变换

    参数:
        img_path: 输入图像路径
        wavelet: 小波基函数，默认为'db1'
        level: 分解层数，默认为1

    返回:
        小波变换结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 进行小波变换
    coeffs = pywt.wavedec2(img, wavelet, level=level)

    # 获取近似系数和细节系数
    cA = coeffs[0]
    (cH, cV, cD) = coeffs[1]

    # 将系数归一化到0-255
    cA = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cH = cv2.normalize(cH, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cV = cv2.normalize(cV, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cD = cv2.normalize(cD, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 组合系数
    top = np.concatenate([cA, cH], axis=1)
    bottom = np.concatenate([cV, cD], axis=1)
    result = np.concatenate([top, bottom], axis=0)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def main(problem_id=43):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为43
    """
    input_path = "../images/imori.jpg"

    if problem_id == 43:
        result = fourier_transform(input_path)
        output_path = "../images/answer_43.jpg"
        title = "傅里叶变换"
    elif problem_id == 44:
        result = frequency_filtering(input_path)
        output_path = "../images/answer_44.jpg"
        title = "频域滤波"
    elif problem_id == 45:
        result = dct_transform(input_path)
        output_path = "../images/answer_45.jpg"
        title = "DCT变换"
    elif problem_id == 46:
        result = wavelet_transform(input_path)
        output_path = "../images/answer_46.jpg"
        title = "小波变换"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 43
    main(problem_id)