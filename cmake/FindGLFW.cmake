# FindGLFW.cmake
# 查找 GLFW 库并设置相关变量

# 查找 GLFW 目录
set(GLFW_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/glfw")

# 检查目录是否存在
if(EXISTS "${GLFW_ROOT_DIR}/CMakeLists.txt")
    set(GLFW_FOUND TRUE)
    message(STATUS "Found GLFW: ${GLFW_ROOT_DIR}")
else()
    set(GLFW_FOUND FALSE)
    message(STATUS "GLFW not found at: ${GLFW_ROOT_DIR}")
endif()
