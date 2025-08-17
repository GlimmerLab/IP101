# FindImGui.cmake
# 查找 ImGui 库并设置相关变量

# 查找 ImGui 源文件
set(IMGUI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")

# 检查目录是否存在
if(EXISTS "${IMGUI_INCLUDE_DIR}/imgui.h")
    set(IMGUI_FOUND TRUE)
    message(STATUS "Found ImGui: ${IMGUI_INCLUDE_DIR}")
else()
    set(IMGUI_FOUND FALSE)
    message(STATUS "ImGui not found at: ${IMGUI_INCLUDE_DIR}")
endif()

# 设置 ImGui 源文件
if(IMGUI_INCLUDE_DIR)
    set(IMGUI_SOURCES
        ${IMGUI_INCLUDE_DIR}/imgui.cpp
        ${IMGUI_INCLUDE_DIR}/imgui_demo.cpp
        ${IMGUI_INCLUDE_DIR}/imgui_draw.cpp
        ${IMGUI_INCLUDE_DIR}/imgui_tables.cpp
        ${IMGUI_INCLUDE_DIR}/imgui_widgets.cpp
        ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_opengl3.cpp
    )

    # 检查源文件是否存在
    foreach(source ${IMGUI_SOURCES})
        if(NOT EXISTS ${source})
            message(WARNING "ImGui source file not found: ${source}")
        endif()
    endforeach()

    set(IMGUI_FOUND TRUE)
    message(STATUS "Found ImGui: ${IMGUI_INCLUDE_DIR}")
else()
    set(IMGUI_FOUND FALSE)
    message(STATUS "ImGui not found")
endif()

# 设置包含目录
if(IMGUI_FOUND)
    set(IMGUI_INCLUDE_DIRS
        ${IMGUI_INCLUDE_DIR}
        ${IMGUI_INCLUDE_DIR}/backends
    )
endif()
