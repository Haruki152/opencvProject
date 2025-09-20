
cmake_minimum_required(VERSION 3.10.0)
project(opencv_project VERSION 0.1.0 LANGUAGES C CXX)

add_executable(opencv_project src/main.cpp)

# 查找并链接OpenCV库
find_package(OpenCV REQUIRED)
target_link_libraries(opencv_project PRIVATE ${OpenCV_LIBS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
