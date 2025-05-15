cmake_minimum_required(VERSION 3.10)
project(IP101)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 手动设置OpenCV路径（可选）
# 方法1：直接在此处设置路径（取消注释并修改为您的OpenCV路径）
set(OpenCV_DIR "E:/Opencv4.11.0/opencv/build")    # Windows示例路径
# set(OpenCV_DIR "/usr/local/opencv4") # Linux示例路径

# 方法2：使用命令行参数设置
# 使用方式：cmake -DOPENCV_DIR=/path/to/opencv/build ..
# if(DEFINED OPENCV_DIR)
#     set(OpenCV_DIR ${OPENCV_DIR})
#     message(STATUS "Using custom OpenCV path: ${OpenCV_DIR}")
# endif()

# 查找包
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# 启用OpenMP
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 启用AVX2
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# 设置公共编译选项和包含目录（用于所有子项目）
set(IP101_COMMON_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}")
set(IP101_COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
set(IP101_COMMON_LIBRARIES
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

# 设置包含目录
include_directories(${IP101_COMMON_INCLUDES})

# 添加子目录
add_subdirectory(cpp/basic)
add_subdirectory(examples/basic)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include
)

# 输出信息
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "使用的编译器: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "编译器版本: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")