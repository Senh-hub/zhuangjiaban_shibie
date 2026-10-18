cmake_minimum_required(VERSION 3.16)

project(zhuangjiaban_shibie LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(zhuangjiaban_core
    src/xiangji_canshu.cpp
    src/chuantong_shibie.cpp
    src/shuzi_fenlei.cpp
    src/pnp_jiesuan.cpp
    src/nn_shibie.cpp
    src/huizhi.cpp
)

target_link_libraries(zhuangjiaban_core
    PUBLIC
    ${OpenCV_LIBS}
    openvino::runtime
)

add_executable(huanjing_ceshi
    app/huanjing_ceshi.cpp
)
target_link_libraries(huanjing_ceshi zhuangjiaban_core)

add_executable(chuantong_video
    app/chuantong_video.cpp
)
target_link_libraries(chuantong_video zhuangjiaban_core)

add_executable(nn_video
    app/nn_video.cpp
)
target_link_libraries(nn_video zhuangjiaban_core)

add_executable(tupian_pingce
    app/tupian_pingce.cpp
)
target_link_libraries(tupian_pingce zhuangjiaban_core)
