import numpy as np
import cv2
from typing import Tuple, List
from scipy import signal
from scipy.ndimage import gaussian_filter

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算峰值信噪比(PSNR)

    Args:
        img1: 第一张图像
        img2: 第二张图像

    Returns:
        float: PSNR值(dB)
    """
    # 确保图像大小相同
    assert img1.shape == img2.shape

    # 计算均方误差(MSE)
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    # 避免除以0
    if mse == 0:
        return float('inf')

    # 计算PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)

    return psnr

def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                window_size: int = 11) -> float:
    """计算结构相似性(SSIM)

    Args:
        img1: 第一张图像
        img2: 第二张图像
        window_size: 窗口大小

    Returns:
        float: SSIM值(0-1之间,越大越好)
    """
    # 确保图像大小相同
    assert img1.shape == img2.shape

    # 转换为浮点数
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 创建高斯窗口
    window = gaussian_filter(np.ones((window_size, window_size)),
                           sigma=1.5)
    window = window / np.sum(window)

    # 计算均值
    mu1 = signal.convolve2d(img1, window, mode='valid')
    mu2 = signal.convolve2d(img2, window, mode='valid')

    # 计算方差和协方差
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    # SSIM参数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)

def perceptual_quality_assessment(img1: np.ndarray,
                                img2: np.ndarray) -> float:
    """感知质量评估

    Args:
        img1: 第一张图像
        img2: 第二张图像

    Returns:
        float: 感知质量分数(0-1之间,越大越好)
    """
    # 计算多尺度SSIM
    scales = [1.0, 0.5, 0.25]
    weights = [0.5, 0.3, 0.2]

    score = 0
    for scale, weight in zip(scales, weights):
        # 缩放图像
        h, w = img1.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        img1_scaled = cv2.resize(img1, (new_w, new_h))
        img2_scaled = cv2.resize(img2, (new_w, new_h))

        # 计算SSIM
        ssim = compute_ssim(img1_scaled, img2_scaled)
        score += weight * ssim

    return score

def no_reference_quality_assessment(img: np.ndarray) -> float:
    """无参考质量评估

    Args:
        img: 输入图像

    Returns:
        float: 质量分数(0-1之间,越大越好)
    """
    # 计算图像梯度
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 计算局部对比度
    local_contrast = np.std(gradient_magnitude)

    # 计算图像熵
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # 计算噪声水平
    noise = np.std(img - cv2.GaussianBlur(img, (5, 5), 0))

    # 综合评分
    score = (local_contrast * 0.4 + entropy * 0.3 + (1 - noise/255) * 0.3)

    return np.clip(score, 0, 1)

def multi_scale_quality_assessment(img1: np.ndarray,
                                 img2: np.ndarray) -> Tuple[float, List[float]]:
    """多尺度质量评估

    Args:
        img1: 第一张图像
        img2: 第二张图像

    Returns:
        Tuple[float, List[float]]: 总体质量分数和各尺度分数
    """
    # 定义评估尺度
    scales = [1.0, 0.5, 0.25, 0.125]
    weights = [0.4, 0.3, 0.2, 0.1]

    scores = []
    for scale in scales:
        # 缩放图像
        h, w = img1.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        img1_scaled = cv2.resize(img1, (new_w, new_h))
        img2_scaled = cv2.resize(img2, (new_w, new_h))

        # 计算该尺度的质量分数
        ssim = compute_ssim(img1_scaled, img2_scaled)
        psnr = compute_psnr(img1_scaled, img2_scaled)

        # 归一化PSNR
        psnr_norm = min(psnr / 50.0, 1.0)

        # 综合该尺度的分数
        score = 0.7 * ssim + 0.3 * psnr_norm
        scores.append(score)

    # 计算加权总分
    total_score = sum(s * w for s, w in zip(scores, weights))

    return total_score, scores

def main():
    """主函数,用于测试各种质量评估算法"""
    # 读取测试图像
    img1 = cv2.imread('test_image1.jpg')
    img2 = cv2.imread('test_image2.jpg')

    # 测试各种质量评估算法
    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)
    perceptual_score = perceptual_quality_assessment(img1, img2)
    no_ref_score = no_reference_quality_assessment(img1)
    total_score, scale_scores = multi_scale_quality_assessment(img1, img2)

    # 打印结果
    print(f'PSNR: {psnr:.2f} dB')
    print(f'SSIM: {ssim:.4f}')
    print(f'Perceptual Score: {perceptual_score:.4f}')
    print(f'No-Reference Score: {no_ref_score:.4f}')
    print(f'Multi-Scale Score: {total_score:.4f}')
    print('Scale Scores:', [f'{s:.4f}' for s in scale_scores])

if __name__ == '__main__':
    main()
