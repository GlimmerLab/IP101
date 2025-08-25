# 进阶算法库的CMakeLists.txt

# 添加所有子目录
add_subdirectory(image_defogging)
add_subdirectory(image_enhancement)
add_subdirectory(special_detection)
add_subdirectory(image_effects)
add_subdirectory(advanced_filtering)
add_subdirectory(image_correction)

# 创建统一的进阶算法库（链接所有子模块）
add_library(ip101_advanced INTERFACE)

# 链接所有子模块库
target_link_libraries(ip101_advanced
    INTERFACE
    ip101_defogging
    ip101_enhancement
    ip101_detection
    ip101_effects
    ip101_advanced_filtering
    ip101_correction
    ip101_basic  # 依赖基础库
    ip101_utils  # 依赖工具库
    ${IP101_COMMON_LIBRARIES}
)

# 包含头文件目录
target_include_directories(ip101_advanced
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

# 安装选项
install(TARGETS ip101_advanced
    EXPORT IP101AdvancedTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# 安装头文件
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/advanced/
    DESTINATION include/advanced
    FILES_MATCHING PATTERN "*.hpp"
)