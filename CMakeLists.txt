cmake_minimum_required(VERSION 3.16)
project(ChePP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-O3 -g -march=native -Wall)

add_subdirectory(engine)
add_subdirectory(nnue_training)


enable_testing()
