cmake_minimum_required(VERSION 3.10)
project(edge_detection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 启用OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 启用AVX2
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# 添加可执行文件
add_executable(edge_detection_test
    edge_detection.cpp
    edge_detection_test.cpp
)

# 链接OpenCV库
target_link_libraries(edge_detection_test
    PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

# 包含头文件目录
target_include_directories(edge_detection_test
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)