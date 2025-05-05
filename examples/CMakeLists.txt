# 添加示例程序
add_executable(basic_example basic_example.cpp)

# 链接库
target_link_libraries(basic_example
    PRIVATE
        image_processing
        ${OpenCV_LIBS}
)

# 添加包含目录
target_include_directories(basic_example
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp
        ${OpenCV_INCLUDE_DIRS}
)

# 安装示例程序
install(TARGETS basic_example
    RUNTIME DESTINATION bin
)