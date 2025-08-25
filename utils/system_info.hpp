#ifndef IP101_SYSTEM_INFO_HPP
#define IP101_SYSTEM_INFO_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#elif defined(__linux__)
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <cpuid.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach/mach.h>
#endif

#include <opencv2/core.hpp>

namespace ip101 {
namespace utils {

/**
 * @brief 系统信息收集类
 *
 * 提供跨平台的系统信息收集功能，包括：
 * - CPU信息（型号、核心数、频率等）
 * - 内存信息（总内存、可用内存等）
 * - OpenCV版本信息
 * - 操作系统信息
 */
class SystemInfo {
public:
    /**
     * @brief CPU信息结构体
     */
    struct CPUInfo {
        std::string vendor;           // CPU厂商
        std::string model;            // CPU型号
        int cores;                    // 物理核心数
        int threads;                  // 逻辑核心数
        double frequency_mhz;         // CPU频率(MHz)
        std::string architecture;     // 架构(x86_64, ARM64等)
        std::vector<std::string> features; // CPU特性
    };

    /**
     * @brief 内存信息结构体
     */
    struct MemoryInfo {
        uint64_t total_physical_kb;   // 总物理内存(KB)
        uint64_t available_physical_kb; // 可用物理内存(KB)
        uint64_t total_virtual_kb;    // 总虚拟内存(KB)
        uint64_t available_virtual_kb; // 可用虚拟内存(KB)
    };

    /**
     * @brief 系统信息结构体
     */
    struct SystemInfoData {
        std::string os_name;          // 操作系统名称
        std::string os_version;       // 操作系统版本
        std::string os_architecture;  // 系统架构
        CPUInfo cpu;                  // CPU信息
        MemoryInfo memory;            // 内存信息
        std::string opencv_version;   // OpenCV版本
        std::string compiler_info;    // 编译器信息
    };

    /**
     * @brief 获取完整的系统信息
     * @return 系统信息结构体
     */
    static SystemInfoData getSystemInfo();

    /**
     * @brief 获取CPU信息
     * @return CPU信息结构体
     */
    static CPUInfo getCPUInfo();

    /**
     * @brief 获取内存信息
     * @return 内存信息结构体
     */
    static MemoryInfo getMemoryInfo();

    /**
     * @brief 获取OpenCV版本信息
     * @return OpenCV版本字符串
     */
    static std::string getOpenCVVersion();

    /**
     * @brief 获取操作系统信息
     * @return 操作系统信息字符串
     */
    static std::string getOSInfo();

    /**
     * @brief 获取编译器信息
     * @return 编译器信息字符串
     */
    static std::string getCompilerInfo();

    /**
     * @brief 格式化系统信息为字符串
     * @param info 系统信息结构体
     * @return 格式化的字符串
     */
    static std::string formatSystemInfo(const SystemInfoData& info);

    /**
     * @brief 检查CPU是否支持特定特性
     * @param feature 特性名称
     * @return 是否支持
     */
    static bool hasCPUFeature(const std::string& feature);

private:
#ifdef _WIN32
    static std::string getWindowsCPUModel();
    static std::string getWindowsOSVersion();
#elif defined(__linux__)
    static std::string getLinuxCPUModel();
    static std::string getLinuxOSVersion();
    static std::string readFileContent(const std::string& filename);
#elif defined(__APPLE__)
    static std::string getMacOSCPUModel();
    static std::string getMacOSVersion();
#endif

    static std::string getCPUArchitecture();
    static std::vector<std::string> getCPUFeatures();
};

} // namespace utils
} // namespace ip101

#endif // IP101_SYSTEM_INFO_HPP
