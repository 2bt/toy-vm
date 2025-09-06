cmake_minimum_required(VERSION 3.22.1)

project(vm)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)

add_executable(
    vm
    src/main.cpp
    src/vm.cpp
)

target_compile_options(
    vm
    PRIVATE
    -std=c++20
    -Wall
    -O2
)

target_link_libraries(
    vm
    PRIVATE
    SDL2::SDL2
    SDL2::SDL2main
    SDL2_image::SDL2_image
)

