# 进阶算法示例程序
set(ADVANCED_EXAMPLES

    # 高级滤波算法示例
    guided_filter_test.cpp
    side_window_filter_test.cpp
    homomorphic_filter_test.cpp

    # 图像矫正算法示例
    automatic_white_balance_test.cpp
    gamma_correction_test.cpp
    auto_level_test.cpp
    backlight_test.cpp
    illumination_correction_test.cpp

    # 图像去雾算法示例
    dark_channel_test.cpp
    realtime_dehazing_test.cpp
    median_filter_test.cpp
    fast_defogging_test.cpp
    guided_filter_dehazing_test.cpp

    # 图像增强算法示例
    retinex_msrcr_test.cpp
    adaptive_logarithmic_mapping_test.cpp
    automatic_color_equalization_test.cpp
    hdr_test.cpp
    multi_scale_detail_enhancement_test.cpp
    real_time_adaptive_contrast_test.cpp

    # 图像特效算法示例
    cartoon_effect_test.cpp
    motion_blur_test.cpp
    oil_painting_effect_test.cpp
    skin_beauty_test.cpp
    spherize_test.cpp
    unsharp_masking_test.cpp
    vintage_effect_test.cpp

    # 特殊检测算法示例
    rectangle_detection_test.cpp
    color_cast_detection_test.cpp
    license_plate_detection_test.cpp
)

# 为每个示例创建可执行文件
foreach(example ${ADVANCED_EXAMPLES})
    get_filename_component(example_name ${example} NAME_WE)

    add_executable(${example_name} ${example})

    target_link_libraries(${example_name}
        ip101_advanced
        ip101_basic
        ${IP101_COMMON_LIBRARIES}
    )

    # 安装可执行文件
    install(TARGETS ${example_name}
        RUNTIME DESTINATION bin
    )
endforeach()