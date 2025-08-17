# ThirdPartyDependencies.cmake
# 统一管理第三方依赖的 CMake 模块

# 包含查找模块
include(${CMAKE_CURRENT_LIST_DIR}/FindImGui.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FindGLFW.cmake)

# 检查所有依赖是否可用
set(THIRD_PARTY_DEPS_AVAILABLE TRUE)

if(NOT IMGUI_FOUND)
    set(THIRD_PARTY_DEPS_AVAILABLE FALSE)
    message(WARNING "ImGui not found - GUI components will be disabled")
endif()

if(NOT GLFW_FOUND)
    set(THIRD_PARTY_DEPS_AVAILABLE FALSE)
    message(WARNING "GLFW not found - GUI components will be disabled")
endif()

# 如果所有依赖都可用，设置 GLFW
if(THIRD_PARTY_DEPS_AVAILABLE)
    # 使用本地 GLFW - 跨平台配置
    message(STATUS "Setting up local GLFW for ${CMAKE_SYSTEM_NAME}...")

    # 设置 GLFW 的跨平台选项
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target" FORCE)

    # 添加GLFW子目录
    add_subdirectory(${GLFW_ROOT_DIR} ${CMAKE_BINARY_DIR}/third_party/glfw)

    # 确保 GLFW 目标被创建
    if(NOT TARGET glfw)
        message(ERROR "GLFW target not created after add_subdirectory")
        set(THIRD_PARTY_DEPS_AVAILABLE FALSE)
    else()
        message(STATUS "GLFW target created successfully for ${CMAKE_SYSTEM_NAME}")
        set(GLFW_AVAILABLE TRUE)
    endif()
endif()

# 如果所有依赖都可用，创建 ImGui 库
if(THIRD_PARTY_DEPS_AVAILABLE)
    # 创建 ImGui 静态库
    add_library(imgui STATIC ${IMGUI_SOURCES})

    # 设置包含目录 - 确保在编译时能找到所有头文件
    target_include_directories(imgui PUBLIC
        ${IMGUI_INCLUDE_DIR}
        ${IMGUI_INCLUDE_DIR}/backends
        ${GLFW_ROOT_DIR}/include
    )

    # 编译定义
    target_compile_definitions(imgui PRIVATE
        IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    )

    # 链接 GLFW - 直接链接到GLFW目标
    if(TARGET glfw)
        target_link_libraries(imgui glfw)
        message(STATUS "ImGui linked to GLFW target")
    else()
        message(ERROR "GLFW target not available for ImGui linking")
    endif()

    # 查找并链接 OpenGL - 跨平台支持
    find_package(OpenGL REQUIRED)
    if(WIN32)
        # Windows 平台
        target_link_libraries(imgui OpenGL::GL)
    elseif(APPLE)
        # macOS 平台
        find_library(COCOA_LIBRARY Cocoa)
        find_library(IOKIT_LIBRARY IOKit)
        find_library(COREVIDEO_LIBRARY CoreVideo)
        target_link_libraries(imgui OpenGL::GL ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY})
    else()
        # Linux 平台
        find_package(X11 REQUIRED)
        target_link_libraries(imgui OpenGL::GL ${X11_LIBRARIES})
    endif()

    message(STATUS "Third-party dependencies configured successfully")
    message(STATUS "ImGui include dir: ${IMGUI_INCLUDE_DIR}")
    message(STATUS "GLFW root dir: ${GLFW_ROOT_DIR}")
    message(STATUS "GLFW target available: $<TARGET_EXISTS:glfw>")
else()
    message(STATUS "Third-party dependencies not available - GUI components disabled")
endif()
