#include "system_info.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifdef _WIN32
#include <intrin.h>
#elif defined(__linux__)
#include <cpuid.h>
#endif

namespace ip101 {
namespace utils {

SystemInfo::SystemInfoData SystemInfo::getSystemInfo() {
    SystemInfoData info;

    // 获取操作系统信息
    std::string os_info = getOSInfo();
    size_t pos = os_info.find(" ");
    if (pos != std::string::npos) {
        info.os_name = os_info.substr(0, pos);
        info.os_version = os_info.substr(pos + 1);
    } else {
        info.os_name = os_info;
        info.os_version = "Unknown";
    }

    info.os_architecture = getCPUArchitecture();
    info.cpu = getCPUInfo();
    info.memory = getMemoryInfo();
    info.opencv_version = getOpenCVVersion();
    info.compiler_info = getCompilerInfo();

    return info;
}

SystemInfo::CPUInfo SystemInfo::getCPUInfo() {
    CPUInfo cpu;

#ifdef _WIN32
    cpu.model = getWindowsCPUModel();
    cpu.vendor = "Intel"; // 简化处理

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    cpu.cores = sysInfo.dwNumberOfProcessors;
    cpu.threads = sysInfo.dwNumberOfProcessors;

    // 获取CPU频率
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD mhz;
        DWORD size = sizeof(DWORD);
        if (RegQueryValueExA(hKey, "~MHz", NULL, NULL, (LPBYTE)&mhz, &size) == ERROR_SUCCESS) {
            cpu.frequency_mhz = static_cast<double>(mhz);
        }
        RegCloseKey(hKey);
    }

#elif defined(__linux__)
    cpu.model = getLinuxCPUModel();
    cpu.vendor = "Unknown";

    // 获取核心数
    cpu.cores = sysconf(_SC_NPROCESSORS_ONLN);
    cpu.threads = sysconf(_SC_NPROCESSORS_ONLN);

    // 获取CPU频率
    std::string freq_file = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
    std::string freq_str = readFileContent(freq_file);
    if (!freq_str.empty()) {
        cpu.frequency_mhz = std::stod(freq_str) / 1000.0;
    }

#elif defined(__APPLE__)
    cpu.model = getMacOSCPUModel();
    cpu.vendor = "Apple";

    // 获取核心数
    int cores;
    size_t size = sizeof(cores);
    if (sysctlbyname("hw.ncpu", &cores, &size, NULL, 0) == 0) {
        cpu.cores = cores;
        cpu.threads = cores;
    }

    // 获取CPU频率
    uint64_t freq;
    size = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) == 0) {
        cpu.frequency_mhz = static_cast<double>(freq) / 1000000.0;
    }
#endif

    cpu.architecture = getCPUArchitecture();
    cpu.features = getCPUFeatures();

    return cpu;
}

SystemInfo::MemoryInfo SystemInfo::getMemoryInfo() {
    MemoryInfo memory;

#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        memory.total_physical_kb = memInfo.ullTotalPhys / 1024;
        memory.available_physical_kb = memInfo.ullAvailPhys / 1024;
        memory.total_virtual_kb = memInfo.ullTotalVirtual / 1024;
        memory.available_virtual_kb = memInfo.ullAvailVirtual / 1024;
    }

#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        memory.total_physical_kb = si.totalram * si.mem_unit / 1024;
        memory.available_physical_kb = si.freeram * si.mem_unit / 1024;
        memory.total_virtual_kb = (si.totalram + si.totalswap) * si.mem_unit / 1024;
        memory.available_virtual_kb = (si.freeram + si.freeswap) * si.mem_unit / 1024;
    }

#elif defined(__APPLE__)
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    host_t host = mach_host_self();

    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        uint64_t page_size = vm_page_size;
        memory.total_physical_kb = (vm_stats.active_count + vm_stats.inactive_count +
                                   vm_stats.wire_count + vm_stats.free_count) * page_size / 1024;
        memory.available_physical_kb = vm_stats.free_count * page_size / 1024;
        memory.total_virtual_kb = memory.total_physical_kb; // 简化处理
        memory.available_virtual_kb = memory.available_physical_kb; // 简化处理
    }
#endif

    return memory;
}

std::string SystemInfo::getOpenCVVersion() {
    return cv::getVersionString();
}

std::string SystemInfo::getOSInfo() {
#ifdef _WIN32
    return getWindowsOSVersion();
#elif defined(__linux__)
    return getLinuxOSVersion();
#elif defined(__APPLE__)
    return getMacOSVersion();
#else
    return "Unknown OS";
#endif
}

std::string SystemInfo::getCompilerInfo() {
    std::stringstream ss;

#ifdef _MSC_VER
    ss << "MSVC " << _MSC_VER;
#elif defined(__GNUC__)
    ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(__clang__)
    ss << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#else
    ss << "Unknown Compiler";
#endif

    return ss.str();
}

std::string SystemInfo::formatSystemInfo(const SystemInfoData& info) {
    std::stringstream ss;

    ss << "=== System Information ===" << std::endl;
    ss << "OS: " << info.os_name << " " << info.os_version << " (" << info.os_architecture << ")" << std::endl;
    ss << "Compiler: " << info.compiler_info << std::endl;
    ss << "OpenCV: " << info.opencv_version << std::endl;
    ss << std::endl;

    ss << "=== CPU Information ===" << std::endl;
    ss << "Model: " << info.cpu.model << std::endl;
    ss << "Vendor: " << info.cpu.vendor << std::endl;
    ss << "Cores: " << info.cpu.cores << " physical, " << info.cpu.threads << " logical" << std::endl;
    ss << "Frequency: " << std::fixed << std::setprecision(2) << info.cpu.frequency_mhz << " MHz" << std::endl;
    ss << "Architecture: " << info.cpu.architecture << std::endl;

    if (!info.cpu.features.empty()) {
        ss << "Features: ";
        for (size_t i = 0; i < info.cpu.features.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << info.cpu.features[i];
        }
        ss << std::endl;
    }
    ss << std::endl;

    ss << "=== Memory Information ===" << std::endl;
    ss << "Physical Memory: " << (info.memory.total_physical_kb / 1024 / 1024) << " GB total, "
       << (info.memory.available_physical_kb / 1024 / 1024) << " GB available" << std::endl;
    ss << "Virtual Memory: " << (info.memory.total_virtual_kb / 1024 / 1024) << " GB total, "
       << (info.memory.available_virtual_kb / 1024 / 1024) << " GB available" << std::endl;

    return ss.str();
}

bool SystemInfo::hasCPUFeature(const std::string& feature) {
    auto features = getCPUFeatures();
    return std::find(features.begin(), features.end(), feature) != features.end();
}

std::string SystemInfo::getCPUArchitecture() {
#ifdef _WIN32
#ifdef _WIN64
    return "x86_64";
#else
    return "x86";
#endif
#elif defined(__linux__)
#ifdef __x86_64__
    return "x86_64";
#elif defined(__aarch64__)
    return "ARM64";
#elif defined(__arm__)
    return "ARM32";
#else
    return "Unknown";
#endif
#elif defined(__APPLE__)
#ifdef __x86_64__
    return "x86_64";
#elif defined(__aarch64__)
    return "ARM64";
#else
    return "Unknown";
#endif
#else
    return "Unknown";
#endif
}

std::vector<std::string> SystemInfo::getCPUFeatures() {
    std::vector<std::string> features;

#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);

    if (cpuInfo[3] & (1 << 23)) features.push_back("MMX");
    if (cpuInfo[3] & (1 << 25)) features.push_back("SSE");
    if (cpuInfo[3] & (1 << 26)) features.push_back("SSE2");
    if (cpuInfo[2] & (1 << 0))  features.push_back("SSE3");
    if (cpuInfo[2] & (1 << 9))  features.push_back("SSSE3");
    if (cpuInfo[2] & (1 << 19)) features.push_back("SSE4.1");
    if (cpuInfo[2] & (1 << 20)) features.push_back("SSE4.2");
    if (cpuInfo[2] & (1 << 28)) features.push_back("AVX");

    // 检查AVX2
    __cpuid(cpuInfo, 7);
    if (cpuInfo[1] & (1 << 5)) features.push_back("AVX2");

#elif defined(__linux__)
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (edx & (1 << 23)) features.push_back("MMX");
        if (edx & (1 << 25)) features.push_back("SSE");
        if (edx & (1 << 26)) features.push_back("SSE2");
        if (ecx & (1 << 0))  features.push_back("SSE3");
        if (ecx & (1 << 9))  features.push_back("SSSE3");
        if (ecx & (1 << 19)) features.push_back("SSE4.1");
        if (ecx & (1 << 20)) features.push_back("SSE4.2");
        if (ecx & (1 << 28)) features.push_back("AVX");
    }

    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 5)) features.push_back("AVX2");
    }

#elif defined(__APPLE__)
    // macOS下简化处理
    features.push_back("SSE2");
    features.push_back("SSE3");
    features.push_back("SSSE3");
    features.push_back("SSE4.1");
    features.push_back("SSE4.2");
#endif

    return features;
}

#ifdef _WIN32

std::string SystemInfo::getWindowsCPUModel() {
    HKEY hKey;
    char buffer[256];
    DWORD size = sizeof(buffer);

    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
            (LPBYTE)buffer, &size) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return std::string(buffer);
        }
        RegCloseKey(hKey);
    }

    return "Unknown CPU";
}

std::string SystemInfo::getWindowsOSVersion() {
    OSVERSIONINFOEXA osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEXA));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXA);

    if (GetVersionExA((OSVERSIONINFOA*)&osvi)) {
        std::stringstream ss;
        ss << "Windows " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion;
        if (osvi.dwBuildNumber > 0) {
            ss << " (Build " << osvi.dwBuildNumber << ")";
        }
        return ss.str();
    }

    return "Windows Unknown";
}

#elif defined(__linux__)

std::string SystemInfo::getLinuxCPUModel() {
    std::string model = readFileContent("/proc/cpuinfo");
    std::istringstream iss(model);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.substr(0, 10) == "model name") {
            size_t pos = line.find(": ");
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }

    return "Unknown CPU";
}

std::string SystemInfo::getLinuxOSVersion() {
    std::string os_info = readFileContent("/etc/os-release");
    std::istringstream iss(os_info);
    std::string line;
    std::string name, version;

    while (std::getline(iss, line)) {
        if (line.substr(0, 4) == "NAME") {
            size_t pos = line.find("=");
            if (pos != std::string::npos) {
                name = line.substr(pos + 2);
                name = name.substr(0, name.length() - 1); // 移除引号
            }
        } else if (line.substr(0, 7) == "VERSION") {
            size_t pos = line.find("=");
            if (pos != std::string::npos) {
                version = line.substr(pos + 2);
                version = version.substr(0, version.length() - 1); // 移除引号
            }
        }
    }

    if (!name.empty() && !version.empty()) {
        return name + " " + version;
    }

    return "Linux Unknown";
}

std::string SystemInfo::readFileContent(const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        return content;
    }
    return "";
}

#elif defined(__APPLE__)

std::string SystemInfo::getMacOSCPUModel() {
    char buffer[256];
    size_t size = sizeof(buffer);

    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, NULL, 0) == 0) {
        return std::string(buffer);
    }

    return "Unknown CPU";
}

std::string SystemInfo::getMacOSVersion() {
    char buffer[256];
    size_t size = sizeof(buffer);

    if (sysctlbyname("kern.osrelease", buffer, &size, NULL, 0) == 0) {
        return std::string("macOS ") + buffer;
    }

    return "macOS Unknown";
}

#endif

} // namespace utils
} // namespace ip101
