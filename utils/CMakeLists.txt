# Utils 目录的CMakeLists.txt

# 创建工具库
add_library(ip101_utils STATIC
    system_info.cpp
)

# 设置包含目录
target_include_directories(ip101_utils
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/ip101/utils>
    ${OpenCV_INCLUDE_DIRS}
)

# 链接依赖库
target_link_libraries(ip101_utils
    PUBLIC
    ${OpenCV_LIBS}
)

# 设置编译选项
if(MSVC)
    target_compile_options(ip101_utils PRIVATE /wd4244 /wd4267 /wd4996)
else()
    target_compile_options(ip101_utils PRIVATE -Wall -Wextra)
endif()

# 安装配置
install(TARGETS ip101_utils
    EXPORT IP101Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 安装头文件
install(FILES system_info.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ip101/utils
)
