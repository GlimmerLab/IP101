cmake_minimum_required(VERSION 3.10)
project(IP101 VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 平台检测和架构支持
if(WIN32)
    set(PLATFORM_NAME "Windows")
    set(PLATFORM_WINDOWS TRUE)
elseif(UNIX AND NOT APPLE)
    set(PLATFORM_NAME "Linux")
    set(PLATFORM_LINUX TRUE)
elseif(APPLE)
    set(PLATFORM_NAME "macOS")
    set(PLATFORM_MACOS TRUE)
endif()

# 架构检测
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_NAME "x86_64")
    set(ARCH_X86_64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(ARCH_NAME "ARM64")
    set(ARCH_ARM64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(ARCH_NAME "ARM")
    set(ARCH_ARM TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(ARCH_NAME "RISC-V")
    set(ARCH_RISCV TRUE)
endif()

# 手动设置OpenCV路径（可选）
# 方法1：直接在此处设置路径（取消注释并修改为您的OpenCV路径）
set(OpenCV_DIR "E:/Opencv4.11.0/opencv/build")    # Windows示例路径
# set(OpenCV_DIR "/usr/local/opencv4") # Linux示例路径

# 方法2：使用命令行参数设置
# 使用方式：cmake -DOPENCV_DIR=/path/to/opencv/build ..
if(DEFINED OPENCV_DIR)
    set(OpenCV_DIR ${OPENCV_DIR})
    message(STATUS "Using custom OpenCV path: ${OpenCV_DIR}")
endif()

# 查找包
find_package(OpenCV REQUIRED)
find_package(OpenMP)

# 启用OpenMP（如果可用）
if(OpenMP_CXX_FOUND)
    # 对于MSVC，直接使用实验性OpenMP支持，避免标志冲突
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found, continuing without it")
endif()

# 架构特定的优化
if(ARCH_X86_64)
    # x86_64架构优化
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        message(STATUS "Enabled AVX2 for MSVC")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        message(STATUS "Enabled AVX2 and FMA for GCC/Clang")
    endif()
elseif(ARCH_ARM64)
    # ARM64架构优化
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        message(STATUS "Enabled native ARM64 optimizations")
    endif()
elseif(ARCH_ARM)
    # ARM架构优化
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        message(STATUS "Enabled native ARM optimizations")
    endif()
endif()

# 通用优化选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
    # 添加警告控制
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    endif()
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    # 添加MSVC警告控制
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4996 /wd4819")  # 禁用常见警告和编码警告
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /Zi")
    endif()
endif()

# 设置全局包含目录和库链接
set(IP101_COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenCV_INCLUDE_DIRS}
)

set(IP101_COMMON_LIBRARIES ${OpenCV_LIBS})

# 添加子目录
add_subdirectory(utils)
add_subdirectory(cpp)
add_subdirectory(examples)

# 检查GUI依赖并条件性添加GUI
# 使用统一的第三方依赖管理模块
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ThirdPartyDependencies.cmake)

# 条件性添加GUI
if(THIRD_PARTY_DEPS_AVAILABLE)
    message(STATUS "GUI依赖检测通过，编译GUI组件")
    add_subdirectory(gui)
else()
    message(STATUS "GUI依赖缺失，跳过GUI编译")
    message(STATUS "要启用GUI，请先下载依赖：")
    message(STATUS "  python gui/setup_git_submodules.py")
    message(STATUS "  或")
    message(STATUS "  git submodule update --init --recursive")
endif()

# 条件性添加测试
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装配置
include(GNUInstallDirs)

# 安装头文件
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ip101
    FILES_MATCHING PATTERN "*.hpp"
)

# 安装工具类头文件
install(DIRECTORY utils/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ip101/utils
    FILES_MATCHING PATTERN "*.hpp"
)

# 安装库文件
install(TARGETS ip101_basic ip101_advanced
    EXPORT IP101Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 安装所有子模块库
install(TARGETS ip101_defogging ip101_enhancement ip101_detection ip101_effects ip101_advanced_filtering ip101_correction
    EXPORT IP101Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 安装CMake配置文件
install(EXPORT IP101Targets
    FILE IP101Targets.cmake
    NAMESPACE IP101::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IP101
)

# 创建配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/IP101Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/IP101Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IP101
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/IP101ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/IP101Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/IP101ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IP101
)

# 显示构建信息
message(STATUS "==========================================")
message(STATUS "IP101 图像处理库构建信息")
message(STATUS "==========================================")
message(STATUS "架构: ${ARCH_NAME}")
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "使用的编译器: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "编译器版本: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")

if(THIRD_PARTY_DEPS_AVAILABLE)
    message(STATUS "GUI组件: 已启用")
else()
    message(STATUS "GUI组件: 已禁用 (依赖缺失)")
endif()
message(STATUS "==========================================")