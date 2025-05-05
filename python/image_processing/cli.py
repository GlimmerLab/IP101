import argparse
import cv2
import numpy as np
from .basic.image_processing import ImageProcessor

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="图像处理命令行工具")

    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用的命令")

    # 高斯滤波命令
    blur_parser = subparsers.add_parser("blur", help="应用高斯滤波")
    blur_parser.add_argument("input", help="输入图像路径")
    blur_parser.add_argument("output", help="输出图像路径")
    blur_parser.add_argument("--ksize", type=int, default=5, help="核大小")
    blur_parser.add_argument("--sigma", type=float, default=1.0, help="标准差")

    # Sobel边缘检测命令
    edge_parser = subparsers.add_parser("edge", help="应用Sobel边缘检测")
    edge_parser.add_argument("input", help="输入图像路径")
    edge_parser.add_argument("output", help="输出图像路径")

    # 图像旋转命令
    rotate_parser = subparsers.add_parser("rotate", help="旋转图像")
    rotate_parser.add_argument("input", help="输入图像路径")
    rotate_parser.add_argument("output", help="输出图像路径")
    rotate_parser.add_argument("--angle", type=float, default=45, help="旋转角度")
    rotate_parser.add_argument("--center-x", type=float, help="旋转中心x坐标")
    rotate_parser.add_argument("--center-y", type=float, help="旋转中心y坐标")

    # RGB转灰度命令
    gray_parser = subparsers.add_parser("gray", help="RGB转灰度")
    gray_parser.add_argument("input", help="输入图像路径")
    gray_parser.add_argument("output", help="输出图像路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 创建图像处理器
    processor = ImageProcessor()

    # 读取输入图像
    image = cv2.imread(args.input)
    if image is None:
        print(f"错误：无法读取图像 {args.input}")
        return 1

    # 根据命令处理图像
    if args.command == "blur":
        result = processor.gaussian_blur(image, args.ksize, args.sigma)
    elif args.command == "edge":
        result = processor.sobel_edge(image)
    elif args.command == "rotate":
        center = None
        if args.center_x is not None and args.center_y is not None:
            center = (args.center_x, args.center_y)
        result = processor.rotate_image(image, args.angle, center)
    elif args.command == "gray":
        result = processor.rgb_to_gray(image)
    else:
        parser.print_help()
        return 1

    # 保存结果
    if not cv2.imwrite(args.output, result):
        print(f"错误：无法保存图像 {args.output}")
        return 1

    print(f"处理完成，结果已保存到 {args.output}")
    return 0

if __name__ == "__main__":
    exit(main())