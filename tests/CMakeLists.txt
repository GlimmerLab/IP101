# 查找GTest（可选）
find_package(GTest QUIET)

if(GTest_FOUND)
    # 添加测试程序
    add_executable(test_image_processing test_image_processing.cpp)

    # 链接库
    target_link_libraries(test_image_processing
        PRIVATE
            image_processing
            ${OpenCV_LIBS}
            GTest::GTest
            GTest::Main
    )

    # 添加包含目录
    target_include_directories(test_image_processing
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/utils
            ${OpenCV_INCLUDE_DIRS}
    )

    # 添加测试
    add_test(NAME test_image_processing COMMAND test_image_processing)

    message(STATUS "GTest found, tests enabled")
else()
    message(STATUS "GTest not found, tests disabled")
endif()