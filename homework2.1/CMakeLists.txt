

cmake_minimum_required(VERSION 3.10.0)
project(homework2.1 VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
add_executable(homework2.1 src/main.cpp)
target_link_libraries(homework2.1 PRIVATE ${OpenCV_LIBS})
target_include_directories(homework2.1 PRIVATE ${OpenCV_INCLUDE_DIRS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
