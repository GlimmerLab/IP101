"""
图像匹配相关问题：
56. 模板匹配(SSD) - 使用平方差和进行模板匹配
57. 模板匹配(SAD) - 使用绝对差和进行模板匹配
58. 模板匹配(NCC) - 使用归一化互相关进行模板匹配
59. 模板匹配(ZNCC) - 使用零均值归一化互相关进行模板匹配
60. 特征点匹配 - 使用特征描述子进行图像匹配
"""

import cv2
import numpy as np

def ssd_matching(img_path, template_path):
    """
    问题56：模板匹配(SSD)
    使用平方差和进行模板匹配

    参数:
        img_path: 输入图像路径
        template_path: 模板图像路径

    返回:
        匹配结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算SSD
    for y in range(H-h+1):
        for x in range(W-w+1):
            diff = img[y:y+h, x:x+w] - template
            result[y, x] = np.sum(diff * diff)

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color

def sad_matching(img_path, template_path):
    """
    问题57：模板匹配(SAD)
    使用绝对差和进行模板匹配

    参数:
        img_path: 输入图像路径
        template_path: 模板图像路径

    返回:
        匹配结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算SAD
    for y in range(H-h+1):
        for x in range(W-w+1):
            diff = np.abs(img[y:y+h, x:x+w] - template)
            result[y, x] = np.sum(diff)

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color

def ncc_matching(img_path, template_path):
    """
    问题58：模板匹配(NCC)
    使用归一化互相关进行模板匹配

    参数:
        img_path: 输入图像路径
        template_path: 模板图像路径

    返回:
        匹配结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算模板的范数
    template_norm = np.sqrt(np.sum(template * template))

    # 计算NCC
    for y in range(H-h+1):
        for x in range(W-w+1):
            window = img[y:y+h, x:x+w]
            window_norm = np.sqrt(np.sum(window * window))
            if window_norm > 0 and template_norm > 0:
                result[y, x] = np.sum(window * template) / (window_norm * template_norm)

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc  # NCC使用最大值
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color

def zncc_matching(img_path, template_path):
    """
    问题59：模板匹配(ZNCC)
    使用零均值归一化互相关进行模板匹配

    参数:
        img_path: 输入图像路径
        template_path: 模板图像路径

    返回:
        匹配结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算模板的均值和标准差
    template_mean = np.mean(template)
    template_std = np.std(template)

    # 计算ZNCC
    for y in range(H-h+1):
        for x in range(W-w+1):
            window = img[y:y+h, x:x+w]
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_std > 0 and template_std > 0:
                zncc = np.sum((window - window_mean) * (template - template_mean)) / (window_std * template_std * h * w)
                result[y, x] = zncc

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc  # ZNCC使用最大值
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color

def feature_point_matching(img_path1, img_path2):
    """
    问题60：特征点匹配
    使用特征描述子进行图像匹配

    参数:
        img_path1: 第一张图像路径
        img_path2: 第二张图像路径

    返回:
        匹配结果可视化
    """
    # 读取图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像")

    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 创建BF匹配器
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(descriptors1, descriptors2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制前10个匹配
    result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result

def main(problem_id=56):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为56
    """
    input_path = "../images/imori.jpg"
    template_path = "../images/imori_part.jpg"

    if problem_id == 56:
        result = ssd_matching(input_path, template_path)
        output_path = "../images/answer_56.jpg"
        title = "SSD模板匹配"
    elif problem_id == 57:
        result = sad_matching(input_path, template_path)
        output_path = "../images/answer_57.jpg"
        title = "SAD模板匹配"
    elif problem_id == 58:
        result = ncc_matching(input_path, template_path)
        output_path = "../images/answer_58.jpg"
        title = "NCC模板匹配"
    elif problem_id == 59:
        result = zncc_matching(input_path, template_path)
        output_path = "../images/answer_59.jpg"
        title = "ZNCC模板匹配"
    elif problem_id == 60:
        result = feature_point_matching(input_path, template_path)
        output_path = "../images/answer_60.jpg"
        title = "特征点匹配"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 56
    main(problem_id)