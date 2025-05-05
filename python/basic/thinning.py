"""
图像细化相关问题：
66. 基本细化算法 - 使用基本的细化算法进行图像细化
67. Hilditch细化 - 使用Hilditch算法进行图像细化
68. Zhang-Suen细化 - 使用Zhang-Suen算法进行图像细化
69. 骨架提取 - 使用形态学操作提取图像骨架
70. 中轴变换 - 计算图像的中轴变换
"""

import cv2
import numpy as np
from scipy import ndimage

def basic_thinning(img_path):
    """
    问题66：基本细化算法
    使用基本的细化算法进行图像细化

    参数:
        img_path: 输入图像路径

    返回:
        细化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0和1的格式
    skeleton = binary.copy() // 255
    changing = True

    # 定义结构元素
    B1 = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [1, 1, 1]], dtype=np.uint8)
    B2 = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [1, 1, 0]], dtype=np.uint8)

    while changing:
        changing = False
        temp = skeleton.copy()

        # 应用细化操作
        for i in range(4):
            # 旋转结构元素
            b1 = np.rot90(B1, i)
            b2 = np.rot90(B2, i)

            # 应用结构元素
            eroded1 = cv2.erode(temp, b1)
            dilated1 = cv2.dilate(eroded1, b1)
            eroded2 = cv2.erode(temp, b2)
            dilated2 = cv2.dilate(eroded2, b2)

            # 更新骨架
            skeleton = skeleton & ~(temp - dilated1)
            skeleton = skeleton & ~(temp - dilated2)

            if np.any(temp != skeleton):
                changing = True

    # 转换回0-255格式
    result = skeleton.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def hilditch_thinning(img_path):
    """
    问题67：Hilditch细化
    使用Hilditch算法进行图像细化

    参数:
        img_path: 输入图像路径

    返回:
        细化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0和1的格式
    skeleton = binary.copy() // 255
    changing = True

    def hilditch_condition(p):
        # 获取8邻域
        p2,p3,p4,p5,p6,p7,p8,p9 = p[0:2], p[1:3], p[2:4], p[3:5], p[4:6], p[5:7], p[6:8], p[7:9]

        # Hilditch条件
        c1 = sum(p[1:]) >= 2 and sum(p[1:]) <= 6  # 连通性条件
        c2 = sum([p2[1], p4[1], p6[1], p8[1]]) >= 1  # 端点条件
        c3 = p2[1] * p4[1] * p6[1] == 0  # 连续性条件1
        c4 = p4[1] * p6[1] * p8[1] == 0  # 连续性条件2

        return c1 and c2 and c3 and c4

    while changing:
        changing = False
        temp = skeleton.copy()

        # 遍历图像
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if temp[i,j] == 1:
                    # 获取3x3邻域
                    neighborhood = []
                    for x in range(i-1, i+2):
                        for y in range(j-1, j+2):
                            neighborhood.append(temp[x,y])

                    # 应用Hilditch条件
                    if hilditch_condition(neighborhood):
                        skeleton[i,j] = 0
                        changing = True

    # 转换回0-255格式
    result = skeleton.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def zhang_suen_thinning(img_path):
    """
    问题68：Zhang-Suen细化
    使用Zhang-Suen算法进行图像细化

    参数:
        img_path: 输入图像路径

    返回:
        细化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0和1的格式
    skeleton = binary.copy() // 255

    def zhang_suen_iteration(img, iter_type):
        changing = False
        rows, cols = img.shape

        # 创建标记数组
        markers = np.zeros_like(img)

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if img[i,j] == 1:
                    # 获取8邻域
                    p2,p3,p4,p5,p6,p7,p8,p9 = (img[i-1,j], img[i-1,j+1], img[i,j+1],
                                              img[i+1,j+1], img[i+1,j], img[i+1,j-1],
                                              img[i,j-1], img[i-1,j-1])

                    # 计算条件
                    A = 0
                    for k in range(len([p2,p3,p4,p5,p6,p7,p8,p9])-1):
                        if [p2,p3,p4,p5,p6,p7,p8,p9][k] == 0 and [p2,p3,p4,p5,p6,p7,p8,p9][k+1] == 1:
                            A += 1
                    B = sum([p2,p3,p4,p5,p6,p7,p8,p9])

                    m1 = p2 * p4 * p6 if iter_type == 0 else p2 * p4 * p8
                    m2 = p4 * p6 * p8 if iter_type == 0 else p2 * p6 * p8

                    if (A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0):
                        markers[i,j] = 1
                        changing = True

        img[markers == 1] = 0
        return img, changing

    # 迭代进行细化
    changing = True
    while changing:
        skeleton, changing1 = zhang_suen_iteration(skeleton, 0)
        skeleton, changing2 = zhang_suen_iteration(skeleton, 1)
        changing = changing1 or changing2

    # 转换回0-255格式
    result = skeleton.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def skeleton_extraction(img_path):
    """
    问题69：骨架提取
    使用形态学操作提取图像骨架

    参数:
        img_path: 输入图像路径

    返回:
        骨架提取结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # 初始化骨架图像
    skeleton = np.zeros_like(binary)

    # 迭代提取骨架
    while True:
        # 形态学开运算
        eroded = cv2.erode(binary, kernel)
        opened = cv2.dilate(eroded, kernel)

        # 提取骨架点
        temp = cv2.subtract(binary, opened)

        # 更新骨架和二值图像
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()

        # 当图像为空时停止迭代
        if cv2.countNonZero(binary) == 0:
            break

    # 转换为彩色图像
    result = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    return result

def medial_axis_transform(img_path):
    """
    问题70：中轴变换
    计算图像的中轴变换

    参数:
        img_path: 输入图像路径

    返回:
        中轴变换结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 计算距离变换
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 归一化距离变换结果
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

    # 提取局部最大值作为中轴点
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated = cv2.dilate(dist_transform, kernel)
    medial_axis = (dist_transform == dilated) & (dist_transform > 20)

    # 转换为uint8类型
    result = medial_axis.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def compute_zhang_suen_manual(image):
    """手动实现Zhang-Suen细化算法

    参数:
        image: 输入的二值图像

    返回:
        skeleton: 细化后的图像
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 确保图像是二值图像
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    skeleton = (binary > 0).astype(np.uint8)

    def get_neighbors(x, y):
        """获取8邻域像素值"""
        return [skeleton[y-1, x], skeleton[y-1, x+1],
                skeleton[y, x+1], skeleton[y+1, x+1],
                skeleton[y+1, x], skeleton[y+1, x-1],
                skeleton[y, x-1], skeleton[y-1, x-1]]

    def count_transitions(neighbors):
        """计算0->1转换次数"""
        transitions = 0
        for i in range(len(neighbors)-1):
            if neighbors[i] == 0 and neighbors[i+1] == 1:
                transitions += 1
        if neighbors[-1] == 0 and neighbors[0] == 1:
            transitions += 1
        return transitions

    def zhang_suen_iteration(phase):
        """执行一次Zhang-Suen迭代"""
        markers = []
        rows, cols = skeleton.shape

        for y in range(1, rows-1):
            for x in range(1, cols-1):
                if skeleton[y, x] != 1:
                    continue

                # 获取8邻域
                P = get_neighbors(x, y)
                # 计算8邻域中1的个数
                P_sum = sum(P)

                if phase == 0:
                    # 第一子迭代条件
                    conditions = [
                        2 <= P_sum <= 6,  # 条件1
                        count_transitions(P) == 1,  # 条件2
                        P[0] * P[2] * P[4] == 0,  # 条件3
                        P[2] * P[4] * P[6] == 0   # 条件4
                    ]
                else:
                    # 第二子迭代条件
                    conditions = [
                        2 <= P_sum <= 6,  # 条件1
                        count_transitions(P) == 1,  # 条件2
                        P[0] * P[2] * P[6] == 0,  # 条件3
                        P[0] * P[4] * P[6] == 0   # 条件4
                    ]

                if all(conditions):
                    markers.append((x, y))

        # 移除标记的点
        for x, y in markers:
            skeleton[y, x] = 0

        return len(markers) > 0

    # 迭代直到没有点可以删除
    while True:
        changed1 = zhang_suen_iteration(0)
        changed2 = zhang_suen_iteration(1)
        if not (changed1 or changed2):
            break

    return skeleton * 255

def main(problem_id=66):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为66
    """
    input_path = "../images/imori.jpg"

    if problem_id == 66:
        result = basic_thinning(input_path)
        output_path = "../images/answer_66.jpg"
        title = "基本细化算法"
    elif problem_id == 67:
        result = hilditch_thinning(input_path)
        output_path = "../images/answer_67.jpg"
        title = "Hilditch细化"
    elif problem_id == 68:
        result = zhang_suen_thinning(input_path)
        output_path = "../images/answer_68.jpg"
        title = "Zhang-Suen细化"
    elif problem_id == 69:
        result = skeleton_extraction(input_path)
        output_path = "../images/answer_69.jpg"
        title = "骨架提取"
    elif problem_id == 70:
        result = medial_axis_transform(input_path)
        output_path = "../images/answer_70.jpg"
        title = "中轴变换"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 66
    main(problem_id)