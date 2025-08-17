# GUI示例程序
# 依赖检测已在主CMakeLists.txt中完成，这里直接编译完整版本
message(STATUS "编译GUI组件...")

# 第三方依赖已通过 ThirdPartyDependencies.cmake 管理

# GUI示例程序列表
set(GUI_EXAMPLES
    # simple_gui.cpp
    main_gui.cpp
)

# 为每个GUI示例创建可执行文件
foreach(example ${GUI_EXAMPLES})
    get_filename_component(example_name ${example} NAME_WE)

    add_executable(${example_name} ${example})

    # 基础链接库
    target_link_libraries(${example_name}
        ip101_advanced
        ip101_basic
        ${IP101_COMMON_LIBRARIES}
    )

    # 高级GUI需要额外依赖
    if(example_name STREQUAL "main_gui")
        if(TARGET imgui)
            target_link_libraries(${example_name} imgui)

            # 直接链接 GLFW 目标
            if(TARGET glfw)
                target_link_libraries(${example_name} glfw)
                message(STATUS "GLFW目标已链接到 ${example_name}")
            else()
                message(ERROR "GLFW目标未找到，无法链接到 ${example_name}")
            endif()

            # 链接 OpenGL - 跨平台支持
            find_package(OpenGL REQUIRED)
            if(WIN32)
                # Windows 平台
                target_link_libraries(${example_name} OpenGL::GL)
            elseif(APPLE)
                # macOS 平台
                find_library(COCOA_LIBRARY Cocoa)
                find_library(IOKIT_LIBRARY IOKit)
                find_library(COREVIDEO_LIBRARY CoreVideo)
                target_link_libraries(${example_name} OpenGL::GL ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY})
            else()
                # Linux 平台
                find_package(X11 REQUIRED)
                target_link_libraries(${example_name} OpenGL::GL ${X11_LIBRARIES})
            endif()
        else()
            message(WARNING "ImGui目标未找到，main_gui可能无法编译")
        endif()
    endif()

    # 安装可执行文件
    install(TARGETS ${example_name}
        RUNTIME DESTINATION bin
    )
endforeach()

# 创建依赖检查目标
add_custom_target(check_gui_deps
    COMMAND ${CMAKE_COMMAND} -E echo "检查GUI依赖..."
    COMMAND ${CMAKE_COMMAND} -E echo "third_party目录: ${CMAKE_SOURCE_DIR}/third_party"
    COMMAND ${CMAKE_COMMAND} -E echo "ImGui: ${CMAKE_SOURCE_DIR}/third_party/imgui"
    COMMAND ${CMAKE_COMMAND} -E echo "GLFW: ${CMAKE_SOURCE_DIR}/third_party/glfw"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "如果依赖缺失，请运行:"
    COMMAND ${CMAKE_COMMAND} -E echo "  python gui/setup_git_submodules.py"
    COMMAND ${CMAKE_COMMAND} -E echo "  或"
    COMMAND ${CMAKE_COMMAND} -E echo "  git submodule update --init --recursive"
    COMMENT "检查GUI依赖状态"
)

# 创建依赖下载目标
add_custom_target(download_gui_deps
    COMMAND ${CMAKE_COMMAND} -E echo "正在下载GUI依赖..."
    COMMAND python ${CMAKE_SOURCE_DIR}/gui/setup_git_submodules.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "下载GUI依赖"
)
