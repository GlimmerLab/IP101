# 基础算法示例程序

# 测试示例列表
set(TEST_EXAMPLES
    color_operations_test
    edge_detection_test
    filtering_test
    feature_extraction_test
    frequency_domain_test
    image_compression_test
    image_enhancement_test
    image_inpainting_test
    image_matching_test
    image_features_test
    image_pyramid_test
    image_quality_test
    image_segmentation_test
    image_transform_test
    morphology_test
    object_detection_test
    super_resolution_test
    texture_analysis_test
    thinning_test
    connected_components_test
)

# 为每个测试示例创建可执行文件
foreach(example ${TEST_EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example}
        PRIVATE
        ip101_basic
        ${IP101_COMMON_LIBRARIES}
    )

    # 设置包含目录
    target_include_directories(${example}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils
        ${OpenCV_INCLUDE_DIRS}
    )
endforeach()