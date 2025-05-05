"""
特征提取相关问题：
28. Harris角点检测 - 使用Harris角点检测算法检测图像中的角点
29. SIFT特征 - 使用SIFT算法提取图像特征
30. SURF特征 - 使用SURF算法提取图像特征
31. ORB特征 - 使用ORB算法提取图像特征
32. 特征匹配 - 使用特征描述子进行图像匹配
"""

import cv2
import numpy as np
import scipy.ndimage as ndimage

def harris_corner_detection(img_path, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    问题28：Harris角点检测
    使用Harris角点检测算法检测图像中的角点

    参数:
        img_path: 输入图像路径
        block_size: 邻域大小，默认为2
        ksize: Sobel算子的孔径参数，默认为3
        k: Harris检测参数，默认为0.04
        threshold: 角点检测阈值，默认为0.01

    返回:
        角点检测结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像在x和y方向的梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算梯度的乘积
    Ixx = Ix * Ix
    Ixy = Iy * Ix
    Iyy = Iy * Iy

    # 对梯度乘积进行高斯滤波
    window = np.ones((block_size, block_size)) / (block_size * block_size)
    Sxx = cv2.filter2D(Ixx, -1, window)
    Sxy = cv2.filter2D(Ixy, -1, window)
    Syy = cv2.filter2D(Iyy, -1, window)

    # 计算Harris角点响应
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    harris_response = det - k * (trace * trace)

    # 阈值处理
    corners = np.zeros_like(gray, dtype=np.uint8)
    corners[harris_response > threshold * harris_response.max()] = 255

    # 在原图上标记角点
    result = img.copy()
    result[corners > 0] = [0, 0, 255]  # 红色标记角点

    return result

def sift_features(img_path, nfeatures=0):
    """
    问题29：SIFT特征
    使用SIFT算法提取图像特征

    参数:
        img_path: 输入图像路径
        nfeatures: 提取的特征点数量，默认为0（提取所有特征点）

    返回:
        特征点检测结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象
    sift = cv2.SIFT_create(nfeatures=nfeatures)

    # 检测关键点和描述子
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 在原图上绘制关键点
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return result

def surf_features(img_path, hessian_threshold=100):
    """
    问题30：SURF特征
    使用SURF算法提取图像特征

    参数:
        img_path: 输入图像路径
        hessian_threshold: Hessian矩阵阈值，默认为100

    返回:
        特征点检测结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建SURF对象
    # 注意：OpenCV 3.x中SURF是付费模块，在OpenCV 4.x中已被移除
    # 这里使用替代方法
    try:
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold=hessian_threshold)
        # 检测关键点和描述子
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        # 在原图上绘制关键点
        result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    except:
        # 如果SURF不可用，使用SIFT代替
        print("SURF不可用，使用SIFT代替")
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return result

def orb_features(img_path, nfeatures=500):
    """
    问题31：ORB特征
    使用ORB算法提取图像特征

    参数:
        img_path: 输入图像路径
        nfeatures: 提取的特征点数量，默认为500

    返回:
        特征点检测结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建ORB对象
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # 检测关键点和描述子
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 在原图上绘制关键点
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return result

def feature_matching(img_path1, img_path2, method='sift'):
    """
    问题32：特征匹配
    使用特征描述子进行图像匹配

    参数:
        img_path1: 第一张图像路径
        img_path2: 第二张图像路径
        method: 特征提取方法，可选'sift', 'surf', 'orb'，默认为'sift'

    返回:
        特征匹配结果图像
    """
    # 读取图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像")

    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 根据选择的方法提取特征
    if method == 'sift':
        # 使用SIFT
        feature_extractor = cv2.SIFT_create()
    elif method == 'surf':
        # 使用SURF
        try:
            feature_extractor = cv2.xfeatures2d.SURF_create()
        except:
            print("SURF不可用，使用SIFT代替")
            feature_extractor = cv2.SIFT_create()
    elif method == 'orb':
        # 使用ORB
        feature_extractor = cv2.ORB_create()
    else:
        raise ValueError(f"不支持的方法: {method}")

    # 检测关键点和描述子
    keypoints1, descriptors1 = feature_extractor.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = feature_extractor.detectAndCompute(gray2, None)

    # 创建特征匹配器
    if method == 'orb':
        # ORB使用汉明距离
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        # SIFT和SURF使用欧氏距离
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 进行特征匹配
    matches = matcher.match(descriptors1, descriptors2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制匹配结果
    result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result

def compute_harris_manual(image, k=0.04, window_size=3, threshold=0.01):
    """手动实现Harris角点检测

    参数:
        image: 输入的灰度图像
        k: Harris响应函数参数，默认0.04
        window_size: 局部窗口大小，默认3
        threshold: 角点检测阈值，默认0.01

    返回:
        corners: 角点检测结果图像
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算x和y方向的梯度
    dx = ndimage.sobel(image, axis=1)
    dy = ndimage.sobel(image, axis=0)

    # 计算梯度乘积
    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy

    # 使用高斯窗口进行平滑
    window = np.ones((window_size, window_size)) / (window_size * window_size)
    Sxx = ndimage.convolve(Ixx, window)
    Sxy = ndimage.convolve(Ixy, window)
    Syy = ndimage.convolve(Iyy, window)

    # 计算Harris响应
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    harris_response = det - k * (trace * trace)

    # 阈值处理
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners

def main(problem_id=28):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为28
    """
    input_path = "../images/imori.jpg"
    input_path2 = "../images/imori_part.jpg"  # 用于特征匹配的第二张图像

    if problem_id == 28:
        result = harris_corner_detection(input_path)
        output_path = "../images/answer_28.jpg"
        title = "Harris角点检测"
    elif problem_id == 29:
        result = sift_features(input_path)
        output_path = "../images/answer_29.jpg"
        title = "SIFT特征"
    elif problem_id == 30:
        result = surf_features(input_path)
        output_path = "../images/answer_30.jpg"
        title = "SURF特征"
    elif problem_id == 31:
        result = orb_features(input_path)
        output_path = "../images/answer_31.jpg"
        title = "ORB特征"
    elif problem_id == 32:
        result = feature_matching(input_path, input_path2)
        output_path = "../images/answer_32.jpg"
        title = "特征匹配"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    main(problem_id)