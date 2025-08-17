#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

namespace ip101 {
namespace utils {

/**
 * @brief 统一的图片读取工具类
 *        支持默认图片和手动指定路径
 */
class ImageLoader {
public:
    /**
     * @brief 获取默认测试图片列表
     * @return 默认图片路径列表
     */
    static std::vector<std::string> get_default_images() {
        return {
            "assets/imori.jpg",
            "assets/imori_512x512.jpg",
            "assets/IP101.png"
        };
    }

    /**
     * @brief 加载单张图片
     * @param path 图片路径
     * @return 加载的图片
     * @throws std::runtime_error 如果图片加载失败
     */
    static cv::Mat load_image(const std::string& path) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            throw std::runtime_error("无法加载图像: " + path);
        }
        return img;
    }

    /**
     * @brief 加载所有默认图片
     * @return 成功加载的图片列表
     */
    static std::vector<cv::Mat> load_all_default_images() {
        std::vector<cv::Mat> images;
        for (const auto& path : get_default_images()) {
            try {
                images.push_back(load_image(path));
                std::cout << "✅ 加载默认图片: " << path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "❌ 加载失败: " << e.what() << std::endl;
            }
        }
        return images;
    }

    /**
     * @brief 加载指定路径的图片
     * @param paths 图片路径列表
     * @return 成功加载的图片列表
     */
    static std::vector<cv::Mat> load_images(const std::vector<std::string>& paths) {
        std::vector<cv::Mat> images;
        for (const auto& path : paths) {
            try {
                images.push_back(load_image(path));
                std::cout << "✅ 加载图片: " << path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "❌ 加载失败: " << e.what() << std::endl;
            }
        }
        return images;
    }

    /**
     * @brief 检查图片是否存在
     * @param path 图片路径
     * @return 是否存在
     */
    static bool image_exists(const std::string& path) {
        return std::filesystem::exists(path);
    }

    /**
     * @brief 获取图片信息
     * @param img 图片
     * @return 图片信息字符串
     */
    static std::string get_image_info(const cv::Mat& img) {
        std::string info = "尺寸: " + std::to_string(img.cols) + "x" + std::to_string(img.rows);
        info += ", 通道: " + std::to_string(img.channels());
        info += ", 类型: " + std::to_string(img.type());
        return info;
    }

    /**
     * @brief 验证图片是否有效
     * @param img 图片
     * @return 是否有效
     */
    static bool is_valid_image(const cv::Mat& img) {
        return !img.empty() && img.cols > 0 && img.rows > 0;
    }
};

} // namespace utils
} // namespace ip101

#endif // IMAGE_LOADER_HPP