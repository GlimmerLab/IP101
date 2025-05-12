import numpy as np
from typing import List, Tuple
import cv2
from scipy.ndimage import gaussian_filter

def compute_patch_similarity(img1: np.ndarray, img2: np.ndarray,
                           pt1: Tuple[int, int], pt2: Tuple[int, int],
                           patch_size: int) -> float:
    """计算两个图像块之间的相似度

    Args:
        img1: 第一张图像
        img2: 第二张图像
        pt1: 第一张图像中的点坐标
        pt2: 第二张图像中的点坐标
        patch_size: 块大小

    Returns:
        float: 两个块之间的相似度(越小越相似)
    """
    half_patch = patch_size // 2
    x1, y1 = pt1
    x2, y2 = pt2

    # 提取图像块
    patch1 = img1[y1-half_patch:y1+half_patch+1,
                  x1-half_patch:x1+half_patch+1]
    patch2 = img2[y2-half_patch:y2+half_patch+1,
                  x2-half_patch:x2+half_patch+1]

    # 计算块之间的欧氏距离
    return np.sum((patch1 - patch2) ** 2)

def diffusion_inpaint(src: np.ndarray, mask: np.ndarray,
                     radius: int = 3, num_iterations: int = 100) -> np.ndarray:
    """基于扩散的图像修复

    Args:
        src: 输入图像
        mask: 修复区域掩码(255表示需要修复的区域)
        radius: 扩散半径
        num_iterations: 迭代次数

    Returns:
        np.ndarray: 修复后的图像
    """
    result = src.copy()
    mask_float = mask.astype(np.float32) / 255.0

    for _ in range(num_iterations):
        next_result = result.copy()

        # 对每个需要修复的像素进行扩散
        for i in range(radius, result.shape[0]-radius):
            for j in range(radius, result.shape[1]-radius):
                if mask[i,j] > 0:
                    sum_pixel = np.zeros(3, dtype=np.float32)
                    weight_sum = 0.0

                    # 在邻域内进行扩散
                    for di in range(-radius, radius+1):
                        for dj in range(-radius, radius+1):
                            if di == 0 and dj == 0:
                                continue

                            ni, nj = i + di, j + dj
                            if mask[ni,nj] == 0:
                                w = 1.0 / (abs(di) + abs(dj))
                                sum_pixel += result[ni,nj] * w
                                weight_sum += w

                    if weight_sum > 1e-6:
                        next_result[i,j] = sum_pixel / weight_sum

        result = next_result

    return result

def patch_match_inpaint(src: np.ndarray, mask: np.ndarray,
                       patch_size: int = 7, search_area: int = 20) -> np.ndarray:
    """基于块匹配的图像修复

    Args:
        src: 输入图像
        mask: 修复区域掩码
        patch_size: 块大小
        search_area: 搜索范围

    Returns:
        np.ndarray: 修复后的图像
    """
    result = src.copy()
    half_patch = patch_size // 2

    # 获取需要修复的点
    inpaint_points = []
    for i in range(half_patch, mask.shape[0]-half_patch):
        for j in range(half_patch, mask.shape[1]-half_patch):
            if mask[i,j] > 0:
                inpaint_points.append((j,i))

    # 对每个需要修复的点找最佳匹配块
    for p in inpaint_points:
        x, y = p
        min_dist = float('inf')
        best_match = None

        # 在搜索区域内寻找最佳匹配
        for i in range(max(half_patch, y-search_area),
                      min(src.shape[0]-half_patch, y+search_area)):
            for j in range(max(half_patch, x-search_area),
                          min(src.shape[1]-half_patch, x+search_area)):
                if mask[i,j] == 0:
                    dist = compute_patch_similarity(src, src, p, (j,i), patch_size)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = (j,i)

        # 复制最佳匹配块
        if best_match is not None:
            bx, by = best_match
            for di in range(-half_patch, half_patch+1):
                for dj in range(-half_patch, half_patch+1):
                    if mask[y+di,x+dj] > 0:
                        result[y+di,x+dj] = src[by+di,bx+dj]

    return result

def patchmatch_inpaint(src: np.ndarray, mask: np.ndarray,
                      patch_size: int = 7, num_iterations: int = 5) -> np.ndarray:
    """基于PatchMatch的图像修复

    Args:
        src: 输入图像
        mask: 修复区域掩码
        patch_size: 块大小
        num_iterations: 迭代次数

    Returns:
        np.ndarray: 修复后的图像
    """
    result = src.copy()
    half_patch = patch_size // 2

    # 初始化随机匹配
    offsets = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.int32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                dx = np.random.randint(0, src.shape[1])
                dy = np.random.randint(0, src.shape[0])
                offsets[i,j] = [dx-j, dy-i]

    # 迭代优化
    for _ in range(num_iterations):
        # 传播
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] > 0:
                    # 检查相邻像素的匹配
                    neighbors = [(j-1,i), (j+1,i), (j,i-1), (j,i+1)]

                    for nx, ny in neighbors:
                        if (0 <= nx < mask.shape[1] and
                            0 <= ny < mask.shape[0]):
                            offset = offsets[ny,nx]
                            match_x = j + offset[0]
                            match_y = i + offset[1]

                            if (0 <= match_x < src.shape[1] and
                                0 <= match_y < src.shape[0]):
                                dist = compute_patch_similarity(
                                    src, src, (j,i), (match_x,match_y), patch_size)
                                if dist < compute_patch_similarity(
                                    src, src, (j,i),
                                    (j+offsets[i,j,0], i+offsets[i,j,1]),
                                    patch_size):
                                    offsets[i,j] = offset

        # 随机搜索
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] > 0:
                    search_radius = src.shape[1]
                    while search_radius > 1:
                        dx = np.random.randint(-search_radius, search_radius)
                        dy = np.random.randint(-search_radius, search_radius)
                        match_x = j + dx
                        match_y = i + dy

                        if (0 <= match_x < src.shape[1] and
                            0 <= match_y < src.shape[0]):
                            dist = compute_patch_similarity(
                                src, src, (j,i), (match_x,match_y), patch_size)
                            if dist < compute_patch_similarity(
                                src, src, (j,i),
                                (j+offsets[i,j,0], i+offsets[i,j,1]),
                                patch_size):
                                offsets[i,j] = [dx, dy]
                        search_radius //= 2

    # 应用最佳匹配
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                offset = offsets[i,j]
                match_x = j + offset[0]
                match_y = i + offset[1]
                if (0 <= match_x < src.shape[1] and
                    0 <= match_y < src.shape[0]):
                    result[i,j] = src[match_y,match_x]

    return result

def video_inpaint(frames: List[np.ndarray], masks: List[np.ndarray],
                 patch_size: int = 7, num_iterations: int = 5) -> List[np.ndarray]:
    """视频修复

    Args:
        frames: 输入视频帧列表
        masks: 每帧的修复掩码列表
        patch_size: 块大小
        num_iterations: 迭代次数

    Returns:
        List[np.ndarray]: 修复后的视频帧列表
    """
    results = [frame.copy() for frame in frames]
    half_patch = patch_size // 2

    # 计算光流场
    flow_forward = []
    flow_backward = []
    for i in range(len(frames)-1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_forward.append(flow)

    for i in range(len(frames)-1, 0, -1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i-1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_backward.append(flow)

    # 迭代修复
    for _ in range(num_iterations):
        for t in range(len(frames)):
            # 获取时空邻域
            temporal_patches = []
            if t > 0:
                warped = cv2.remap(
                    results[t-1],
                    flow_backward[t-1][:,:,0],
                    flow_backward[t-1][:,:,1],
                    cv2.INTER_LINEAR
                )
                temporal_patches.append(warped)
            if t < len(frames)-1:
                warped = cv2.remap(
                    results[t+1],
                    flow_forward[t][:,:,0],
                    flow_forward[t][:,:,1],
                    cv2.INTER_LINEAR
                )
                temporal_patches.append(warped)

            # 修复当前帧
            for i in range(half_patch, frames[t].shape[0]-half_patch):
                for j in range(half_patch, frames[t].shape[1]-half_patch):
                    if masks[t][i,j] > 0:
                        min_dist = float('inf')
                        best_match = None

                        # 空间匹配
                        for di in range(-half_patch, half_patch+1):
                            for dj in range(-half_patch, half_patch+1):
                                if masks[t][i+di,j+dj] == 0:
                                    dist = compute_patch_similarity(
                                        results[t], results[t],
                                        (j,i), (j+dj,i+di), patch_size)
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_match = (j+dj,i+di)

                        # 时间匹配
                        for patch in temporal_patches:
                            for di in range(-half_patch, half_patch+1):
                                for dj in range(-half_patch, half_patch+1):
                                    if (0 <= i+di < patch.shape[0] and
                                        0 <= j+dj < patch.shape[1]):
                                        dist = compute_patch_similarity(
                                            results[t], patch,
                                            (j,i), (j+dj,i+di), patch_size)
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_match = (j+dj,i+di)

                        # 应用最佳匹配
                        if best_match is not None:
                            results[t][i,j] = results[t][best_match[1],best_match[0]]

    return results

def main():
    """主函数,用于测试各种修复算法"""
    # 读取测试图像
    img = cv2.imread('test_image.jpg')
    mask = cv2.imread('test_mask.jpg', cv2.IMREAD_GRAYSCALE)

    # 测试各种修复算法
    result_diffusion = diffusion_inpaint(img, mask)
    result_patch_match = patch_match_inpaint(img, mask)
    result_patchmatch = patchmatch_inpaint(img, mask)

    # 保存结果
    cv2.imwrite('result_diffusion.jpg', result_diffusion)
    cv2.imwrite('result_patch_match.jpg', result_patch_match)
    cv2.imwrite('result_patchmatch.jpg', result_patchmatch)

if __name__ == '__main__':
    main()
