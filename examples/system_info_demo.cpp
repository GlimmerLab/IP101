#include <iostream>
#include <opencv2/opencv.hpp>

// 包含系统信息收集工具
#include "utils/system_info.hpp"

using namespace std;
using namespace ip101::utils;

int main() {
    cout << "=== IP101 System Information Demo ===" << endl;

    // 获取完整的系统信息
    auto sys_info = SystemInfo::getSystemInfo();

    // 显示格式化的系统信息
    cout << SystemInfo::formatSystemInfo(sys_info) << endl;

    // 检查特定CPU特性
    cout << "=== CPU Feature Check ===" << endl;
    vector<string> features_to_check = {"SSE2", "SSE3", "AVX", "AVX2"};

    for (const auto& feature : features_to_check) {
        bool has_feature = SystemInfo::hasCPUFeature(feature);
        cout << feature << ": " << (has_feature ? "✓ Supported" : "✗ Not Supported") << endl;
    }

    // 获取图像信息示例
    cout << "\n=== Image Information Example ===" << endl;
    Mat test_image = Mat::zeros(100, 100, CV_8UC3);
    cout << "Test Image Size: " << test_image.cols << " x " << test_image.rows << " pixels" << endl;
    cout << "Test Image Channels: " << test_image.channels() << endl;
    cout << "Test Image Memory Usage: " << test_image.total() * test_image.elemSize() / 1024.0 << " KB" << endl;

    cout << "\n=== OpenCV Version ===" << endl;
    cout << "OpenCV Version: " << SystemInfo::getOpenCVVersion() << endl;

    return 0;
}
