"""
连通域分析相关问题：
61. 4连通域标记 - 使用4连通性进行区域标记
62. 8连通域标记 - 使用8连通性进行区域标记
63. 连通域统计 - 统计连通域的数量和大小
64. 连通域过滤 - 根据面积过滤连通域
65. 连通域属性计算 - 计算连通域的各种属性
"""

import cv2
import numpy as np
from scipy import ndimage

def four_connected_labeling(img_path):
    """
    问题61：4连通域标记
    使用4连通性进行区域标记

    参数:
        img_path: 输入图像路径

    返回:
        标记结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域标记函数
    num_labels, labels = cv2.connectedComponents(binary, connectivity=4)

    # 为标记结果分配不同的颜色
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景为黑色

    # 创建彩色标记图像
    result = colors[labels]

    return result

def eight_connected_labeling(img_path):
    """
    问题62：8连通域标记
    使用8连通性进行区域标记

    参数:
        img_path: 输入图像路径

    返回:
        标记结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域标记函数
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    # 为标记结果分配不同的颜色
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景为黑色

    # 创建彩色标记图像
    result = colors[labels]

    return result

def connected_components_stats(img_path):
    """
    问题63：连通域统计
    统计连通域的数量和大小

    参数:
        img_path: 输入图像路径

    返回:
        统计结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域分析函数
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # 创建彩色结果图像
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 绘制连通域信息
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        center = tuple(map(int, centroids[i]))

        # 绘制边界框
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 绘制中心点
        cv2.circle(result, center, 4, (0, 0, 255), -1)
        # 显示面积
        cv2.putText(result, f"Area: {area}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result

def connected_components_filtering(img_path, min_area=100):
    """
    问题64：连通域过滤
    根据面积过滤连通域

    参数:
        img_path: 输入图像路径
        min_area: 最小面积阈值，默认为100

    返回:
        过滤结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域分析函数
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    # 创建掩码
    mask = np.zeros_like(labels, dtype=np.uint8)

    # 根据面积过滤连通域
    for i in range(1, num_labels):  # 跳过背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    # 转换为彩色图像
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return result

def connected_components_properties(img_path):
    """
    问题65：连通域属性计算
    计算连通域的各种属性

    参数:
        img_path: 输入图像路径

    返回:
        属性可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域分析函数
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # 创建彩色结果图像
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 计算和绘制每个连通域的属性
    for i in range(1, num_labels):  # 跳过背景
        # 获取基本属性
        x, y, w, h, area = stats[i]
        center = tuple(map(int, centroids[i]))

        # 计算轮廓
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 计算周长
            perimeter = cv2.arcLength(contours[0], True)
            # 计算圆形度
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            # 计算矩形度
            extent = area / (w * h) if w * h > 0 else 0

            # 绘制轮廓
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            # 绘制中心点
            cv2.circle(result, center, 4, (0, 0, 255), -1)
            # 显示属性
            cv2.putText(result, f"Area: {area}", (x, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(result, f"Circ: {circularity:.2f}", (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(result, f"Ext: {extent:.2f}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result

def main(problem_id=61):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为61
    """
    input_path = "../images/imori.jpg"

    if problem_id == 61:
        result = four_connected_labeling(input_path)
        output_path = "../images/answer_61.jpg"
        title = "4连通域标记"
    elif problem_id == 62:
        result = eight_connected_labeling(input_path)
        output_path = "../images/answer_62.jpg"
        title = "8连通域标记"
    elif problem_id == 63:
        result = connected_components_stats(input_path)
        output_path = "../images/answer_63.jpg"
        title = "连通域统计"
    elif problem_id == 64:
        result = connected_components_filtering(input_path)
        output_path = "../images/answer_64.jpg"
        title = "连通域过滤"
    elif problem_id == 65:
        result = connected_components_properties(input_path)
        output_path = "../images/answer_65.jpg"
        title = "连通域属性计算"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 61
    main(problem_id)