cmake_minimum_required(VERSION 3.21)
project(AD)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/external/autodiff/include)
link_directories(${CMAKE_SOURCE_DIR}/external/mylib/lib)


add_executable(AD forward.cpp
        function.hpp
        reverse.cpp
        main.cpp)
target_include_directories(AD PRIVATE ${CMAKE_SOURCE_DIR}/external/autodiff)
target_link_libraries(AD Eigen3::Eigen)