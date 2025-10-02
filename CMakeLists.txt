cmake_minimum_required(VERSION 3.22.1)

project(toy)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)

add_executable(
    toy
    src/main.cpp
    src/vm.cpp
)

target_compile_options(
    toy
    PRIVATE
    -std=c++20
    -Wall
    -O3
    -mbmi2 # for _pext_u32
)

target_link_libraries(
    toy
    PRIVATE
    SDL2::SDL2
    SDL2::SDL2main
    SDL2_image::SDL2_image
)

