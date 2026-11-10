cmake_minimum_required(VERSION 3.22.1)

project(smolt)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)

add_executable(
    smolt
    src/main.cpp
    src/vm.cpp
)

target_compile_options(
    smolt
    PRIVATE
    -std=c++20
    -Wall
    -O3
    -fwrapv # signed overflow wraps, like the web host
)

target_link_libraries(
    smolt
    PRIVATE
    SDL2::SDL2
    SDL2::SDL2main
    SDL2_image::SDL2_image
)

