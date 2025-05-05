cmake_minimum_required(VERSION 3.10)
project(ImageProcessing VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
option(USE_AVX2 "Enable AVX2 optimization" ON)
option(USE_SSE4 "Enable SSE4 optimization" ON)
option(USE_OPENMP "Enable OpenMP parallel optimization" ON)
option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 添加OpenMP支持
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# 添加SIMD优化选项
if(USE_AVX2)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2)
    endif()
endif()

if(USE_SSE4)
    if(MSVC)
        add_compile_options(/arch:SSE4)
    else()
        add_compile_options(-msse4)
    endif()
endif()

# 添加源文件
set(SOURCES
    cpp/basic/color_operations.cpp
    cpp/basic/edge_detection.cpp
    cpp/basic/filtering.cpp
    cpp/basic/image_transform.cpp
)

# 添加头文件
set(HEADERS
    cpp/basic/color_operations.h
    cpp/basic/edge_detection.h
    cpp/basic/filtering.h
    cpp/basic/image_transform.h
)

# 创建库
add_library(image_processing STATIC ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(image_processing PRIVATE ${OpenCV_LIBS})

# 添加包含目录
target_include_directories(image_processing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${OpenCV_INCLUDE_DIRS}
)

# 安装规则
install(TARGETS image_processing
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES ${HEADERS}
    DESTINATION include/image_processing
)

# 构建测试程序
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 构建示例程序
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 添加文档目标
add_custom_target(docs
    COMMAND mkdir -p docs/html
    COMMAND doxygen Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation"
)