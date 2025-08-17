#ifndef QUICK_TEST_HPP
#define QUICK_TEST_HPP

#include "image_loader.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <functional>

// 进阶算法头文件
#include <advanced/defogging/dark_channel.hpp>
#include <advanced/enhancement/retinex_msrcr.hpp>
#include <advanced/correction/automatic_white_balance.hpp>
#include <advanced/filtering/guided_filter.hpp>
#include <advanced/detection/rectangle_detection.hpp>
#include <advanced/effects/cartoon_effect.hpp>

namespace ip101 {
namespace utils {

/**
 * @brief 快速测试框架
 *        支持默认图片和手动指定路径的灵活测试
 */
class QuickTestFramework {
private:
    std::string output_dir;
    std::vector<cv::Mat> test_images;
    std::vector<std::string> image_names;

public:
    /**
     * @brief 构造函数
     * @param output 输出目录
     * @param custom_paths 自定义图片路径列表（可选）
     */
    QuickTestFramework(const std::string& output = "quick_test_output",
                      const std::vector<std::string>& custom_paths = {})
        : output_dir(output) {
        std::filesystem::create_directories(output_dir);

        if (custom_paths.empty()) {
            // 使用默认图片
            std::cout << "Using default test images..." << std::endl;
            test_images = ImageLoader::load_all_default_images();
            auto default_paths = ImageLoader::get_default_images();
            for (const auto& path : default_paths) {
                image_names.push_back(std::filesystem::path(path).stem().string());
            }
        } else {
            // 使用自定义图片
            std::cout << "Using custom test images..." << std::endl;
            test_images = ImageLoader::load_images(custom_paths);
            for (const auto& path : custom_paths) {
                image_names.push_back(std::filesystem::path(path).stem().string());
            }
        }

        std::cout << "Successfully loaded " << test_images.size() << " test images" << std::endl;
    }

    /**
     * @brief 测试单个算法
     * @param name 算法名称
     * @param algorithm 算法函数
     */
    template<typename Func>
    void test_algorithm(const std::string& name, Func&& algorithm) {
        std::cout << "\nTesting algorithm: " << name << std::endl;
        std::cout << "==========================================" << std::endl;

        for (size_t i = 0; i < test_images.size(); ++i) {
            if (!ImageLoader::is_valid_image(test_images[i])) {
                std::cout << "  Warning: Skip invalid image: " << image_names[i] << std::endl;
                continue;
            }

            std::cout << "  Processing image: " << image_names[i] << std::endl;
            std::cout << "    " << ImageLoader::get_image_info(test_images[i]) << std::endl;

            cv::Mat result;
            auto start = std::chrono::high_resolution_clock::now();

            try {
                algorithm(test_images[i], result);

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                std::string filename = output_dir + "/" + name + "_" + image_names[i] + ".jpg";
                cv::imwrite(filename, result);

                std::cout << "    Success, time: " << duration.count() << "ms" << std::endl;
                std::cout << "    Saved to: " << filename << std::endl;

            } catch (const std::exception& e) {
                std::cout << "    Failed: " << e.what() << std::endl;
            }

            std::cout << std::endl;
        }
    }

    /**
     * @brief 运行所有进阶算法测试
     */
    void run_all_advanced_tests() {
        std::cout << "Starting IP101 advanced algorithm quick test..." << std::endl;
        std::cout << "Test image count: " << test_images.size() << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << "==========================================" << std::endl;

        // 测试暗通道去雾
        test_algorithm("dark_channel", [](const cv::Mat& src, cv::Mat& dst) {
            ip101::advanced::dark_channel_defogging(src, dst);
        });

        // 测试Retinex增强
        test_algorithm("retinex", [](const cv::Mat& src, cv::Mat& dst) {
            ip101::advanced::retinex_msrcr(src, dst);
        });

        // 测试自动白平衡
        test_algorithm("awb", [](const cv::Mat& src, cv::Mat& dst) {
            ip101::advanced::automatic_white_balance(src, dst);
        });

        // 测试导向滤波
        test_algorithm("guided_filter", [](const cv::Mat& src, cv::Mat& dst) {
            ip101::advanced::guided_filter(src, src, dst, 8, 0.1);
        });

        // 测试矩形检测
        test_algorithm("rectangle_detection", [](const cv::Mat& src, cv::Mat& dst) {
            dst = src.clone();
            std::vector<ip101::advanced::RectangleInfo> rectangles;
            ip101::advanced::detect_rectangles(src, rectangles);
            ip101::advanced::draw_rectangles(dst, rectangles);
        });

        // 测试卡通效果
        test_algorithm("cartoon", [](const cv::Mat& src, cv::Mat& dst) {
            ip101::advanced::cartoon_effect(src, dst);
        });

        std::cout << "\nAll advanced algorithm tests completed!" << std::endl;
        std::cout << "Results saved in: " << output_dir << std::endl;
    }

    /**
     * @brief 运行指定算法测试
     * @param algorithm_name 算法名称
     */
    void run_specific_test(const std::string& algorithm_name) {
        std::cout << "Running specific algorithm test: " << algorithm_name << std::endl;

        if (algorithm_name == "dark_channel") {
            test_algorithm("dark_channel", [](const cv::Mat& src, cv::Mat& dst) {
                ip101::advanced::dark_channel_defogging(src, dst);
            });
        } else if (algorithm_name == "retinex") {
            test_algorithm("retinex", [](const cv::Mat& src, cv::Mat& dst) {
                ip101::advanced::retinex_msrcr(src, dst);
            });
        } else if (algorithm_name == "awb") {
            test_algorithm("awb", [](const cv::Mat& src, cv::Mat& dst) {
                ip101::advanced::automatic_white_balance(src, dst);
            });
        } else if (algorithm_name == "guided_filter") {
            test_algorithm("guided_filter", [](const cv::Mat& src, cv::Mat& dst) {
                ip101::advanced::guided_filter(src, src, dst, 8, 0.1);
            });
        } else if (algorithm_name == "rectangle_detection") {
            test_algorithm("rectangle_detection", [](const cv::Mat& src, cv::Mat& dst) {
                dst = src.clone();
                std::vector<ip101::advanced::RectangleInfo> rectangles;
                ip101::advanced::detect_rectangles(src, rectangles);
                ip101::advanced::draw_rectangles(dst, rectangles);
            });
        } else if (algorithm_name == "cartoon") {
            test_algorithm("cartoon", [](const cv::Mat& src, cv::Mat& dst) {
                ip101::advanced::cartoon_effect(src, dst);
            });
        } else {
            std::cout << "Unknown algorithm: " << algorithm_name << std::endl;
            std::cout << "Supported algorithms: dark_channel, retinex, awb, guided_filter, rectangle_detection, cartoon" << std::endl;
        }
    }

    /**
     * @brief 获取测试图片信息
     */
    void print_test_info() {
        std::cout << "Test information:" << std::endl;
        std::cout << "  Image count: " << test_images.size() << std::endl;
        std::cout << "  Output directory: " << output_dir << std::endl;

        for (size_t i = 0; i < test_images.size(); ++i) {
            std::cout << "  " << (i+1) << ". " << image_names[i]
                      << " - " << ImageLoader::get_image_info(test_images[i]) << std::endl;
        }
    }
};

} // namespace utils
} // namespace ip101

#endif // QUICK_TEST_HPP