"""
图像变换相关问题：
17. 仿射变换 - 使用仿射变换矩阵进行图像变换
18. 透视变换 - 使用透视变换矩阵进行图像变换
19. 旋转 - 使用旋转矩阵进行图像旋转
20. 缩放 - 使用缩放矩阵进行图像缩放
21. 平移 - 使用平移矩阵进行图像平移
22. 镜像 - 水平或垂直镜像图像
"""

import cv2
import numpy as np

def affine_transform(img_path, src_points, dst_points):
    """
    问题17：仿射变换
    使用仿射变换矩阵进行图像变换

    参数:
        img_path: 输入图像路径
        src_points: 源图像中的三个点坐标，形状为(3, 2)
        dst_points: 目标图像中的三个点坐标，形状为(3, 2)

    返回:
        变换后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 计算仿射变换矩阵
    # 注意：这里使用OpenCV的getAffineTransform函数计算变换矩阵
    # 在实际教学中，可以手动计算变换矩阵
    M = cv2.getAffineTransform(src_points, dst_points)

    # 创建输出图像
    result = np.zeros_like(img)

    # 手动实现仿射变换
    for y in range(h):
        for x in range(w):
            # 计算源图像中的对应点
            src_x = int(M[0, 0] * x + M[0, 1] * y + M[0, 2])
            src_y = int(M[1, 0] * x + M[1, 1] * y + M[1, 2])

            # 检查源点是否在图像范围内
            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result

def perspective_transform(img_path, src_points, dst_points):
    """
    问题18：透视变换
    使用透视变换矩阵进行图像变换

    参数:
        img_path: 输入图像路径
        src_points: 源图像中的四个点坐标，形状为(4, 2)
        dst_points: 目标图像中的四个点坐标，形状为(4, 2)

    返回:
        变换后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 计算透视变换矩阵
    # 注意：这里使用OpenCV的getPerspectiveTransform函数计算变换矩阵
    # 在实际教学中，可以手动计算变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 创建输出图像
    result = np.zeros_like(img)

    # 手动实现透视变换
    for y in range(h):
        for x in range(w):
            # 计算源图像中的对应点
            denominator = M[2, 0] * x + M[2, 1] * y + M[2, 2]
            if denominator != 0:
                src_x = int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / denominator)
                src_y = int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / denominator)

                # 检查源点是否在图像范围内
                if 0 <= src_x < w and 0 <= src_y < h:
                    result[y, x] = img[src_y, src_x]

    return result

def rotate_image(img_path, angle, center=None):
    """
    问题19：旋转
    使用旋转矩阵进行图像旋转

    参数:
        img_path: 输入图像路径
        angle: 旋转角度（度）
        center: 旋转中心点，默认为图像中心

    返回:
        旋转后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 如果未指定旋转中心，则使用图像中心
    if center is None:
        center = (w // 2, h // 2)

    # 计算旋转矩阵
    # 注意：这里使用OpenCV的getRotationMatrix2D函数计算旋转矩阵
    # 在实际教学中，可以手动计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 创建输出图像
    result = np.zeros_like(img)

    # 手动实现旋转
    for y in range(h):
        for x in range(w):
            # 计算源图像中的对应点
            src_x = int(M[0, 0] * x + M[0, 1] * y + M[0, 2])
            src_y = int(M[1, 0] * x + M[1, 1] * y + M[1, 2])

            # 检查源点是否在图像范围内
            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result

def scale_image(img_path, scale_x, scale_y):
    """
    问题20：缩放
    使用缩放矩阵进行图像缩放

    参数:
        img_path: 输入图像路径
        scale_x: x方向的缩放比例
        scale_y: y方向的缩放比例

    返回:
        缩放后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 计算缩放后的尺寸
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)

    # 创建输出图像
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # 手动实现缩放
    for y in range(new_h):
        for x in range(new_w):
            # 计算源图像中的对应点
            src_x = int(x / scale_x)
            src_y = int(y / scale_y)

            # 检查源点是否在图像范围内
            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result

def translate_image(img_path, tx, ty):
    """
    问题21：平移
    使用平移矩阵进行图像平移

    参数:
        img_path: 输入图像路径
        tx: x方向的平移量
        ty: y方向的平移量

    返回:
        平移后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 创建输出图像
    result = np.zeros_like(img)

    # 手动实现平移
    for y in range(h):
        for x in range(w):
            # 计算源图像中的对应点
            src_x = x - tx
            src_y = y - ty

            # 检查源点是否在图像范围内
            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result

def mirror_image(img_path, direction='horizontal'):
    """
    问题22：镜像
    水平或垂直镜像图像

    参数:
        img_path: 输入图像路径
        direction: 镜像方向，'horizontal'表示水平镜像，'vertical'表示垂直镜像

    返回:
        镜像后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 创建输出图像
    result = np.zeros_like(img)

    # 手动实现镜像
    if direction == 'horizontal':
        # 水平镜像
        for y in range(h):
            for x in range(w):
                result[y, x] = img[y, w-1-x]
    else:
        # 垂直镜像
        for y in range(h):
            for x in range(w):
                result[y, x] = img[h-1-y, x]

    return result

def compute_perspective_transform_manual(image, src_points, dst_points):
    """手动实现透视变换

    参数:
        image: 输入图像
        src_points: 源图像中的4个点坐标，形状为(4,2)的数组
        dst_points: 目标图像中对应的4个点坐标，形状为(4,2)的数组

    返回:
        transformed: 透视变换后的图像
    """
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32),
                                   dst_points.astype(np.float32))

    # 应用透视变换
    height, width = image.shape[:2]
    transformed = cv2.warpPerspective(image, M, (width, height))

    return transformed

def compute_scale_manual(image, scale_x, scale_y):
    """手动实现图像缩放

    参数:
        image: 输入图像
        scale_x: x方向的缩放比例
        scale_y: y方向的缩放比例

    返回:
        scaled: 缩放后的图像
    """
    height, width = image.shape[:2]
    new_height = int(height * scale_y)
    new_width = int(width * scale_x)

    # 使用双线性插值进行缩放
    scaled = cv2.resize(image, (new_width, new_height),
                       interpolation=cv2.INTER_LINEAR)

    return scaled

def compute_translation_manual(image, tx, ty):
    """手动实现图像平移

    参数:
        image: 输入图像
        tx: x方向的平移量（正值向右，负值向左）
        ty: y方向的平移量（正值向下，负值向上）

    返回:
        translated: 平移后的图像
    """
    # 构建平移矩阵
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])

    # 应用平移变换
    height, width = image.shape[:2]
    translated = cv2.warpAffine(image, M, (width, height))

    return translated

def compute_mirror_manual(image, direction='horizontal'):
    """手动实现图像镜像

    参数:
        image: 输入图像
        direction: 镜像方向，'horizontal'表示水平镜像，'vertical'表示垂直镜像

    返回:
        mirrored: 镜像后的图像
    """
    if direction == 'horizontal':
        mirrored = cv2.flip(image, 1)  # 1表示水平翻转
    elif direction == 'vertical':
        mirrored = cv2.flip(image, 0)  # 0表示垂直翻转
    else:
        raise ValueError("direction必须是'horizontal'或'vertical'")

    return mirrored

def main(problem_id=17):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为17
    """
    input_path = "../images/imori.jpg"

    if problem_id == 17:
        # 仿射变换示例
        src_points = np.float32([[50, 50], [200, 50], [50, 200]])
        dst_points = np.float32([[10, 100], [200, 50], [100, 250]])
        result = affine_transform(input_path, src_points, dst_points)
        output_path = "../images/answer_17.jpg"
        title = "仿射变换"
    elif problem_id == 18:
        # 透视变换示例
        src_points = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        dst_points = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        result = perspective_transform(input_path, src_points, dst_points)
        output_path = "../images/answer_18.jpg"
        title = "透视变换"
    elif problem_id == 19:
        # 旋转示例
        result = rotate_image(input_path, 30)
        output_path = "../images/answer_19.jpg"
        title = "旋转"
    elif problem_id == 20:
        # 缩放示例
        result = scale_image(input_path, 1.5, 1.5)
        output_path = "../images/answer_20.jpg"
        title = "缩放"
    elif problem_id == 21:
        # 平移示例
        result = translate_image(input_path, 30, 30)
        output_path = "../images/answer_21.jpg"
        title = "平移"
    elif problem_id == 22:
        # 镜像示例
        result = mirror_image(input_path, 'horizontal')
        output_path = "../images/answer_22.jpg"
        title = "镜像"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    main(problem_id)