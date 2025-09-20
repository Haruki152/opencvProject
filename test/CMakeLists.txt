cmake_minimum_required(VERSION 3.10.0)
project(opencv_project VERSION 0.1.0 LANGUAGES CXX)

add_executable(opencv_project src/main.cpp)
# 查找并链接OpenCV库
find_package(OpenCV REQUIRED)
target_link_libraries(opencv_project PRIVATE ${OpenCV_LIBS})
target_include_directories(opencv_project PRIVATE ${OpenCV_INCLUDE_DIRS})
# 构建时自动复制 resources 文件夹到 build 目录
add_custom_command(TARGET opencv_project POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:opencv_project>/resources)

# 添加 resources 文件夹到安装/打包范围
install(DIRECTORY resources/ DESTINATION resources)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
