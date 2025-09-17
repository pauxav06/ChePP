cmake_minimum_required(VERSION 3.25)
project(nnue_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

file(GLOB_RECURSE DATAGEN_SOURCES
        "data/*.cpp"
)

include_directories(
        "data"
        "data/stream"
        "data/converter"
        "data/binpack"
        "data/utils"
        "datasets"
)

add_executable(datagen ${DATAGEN_SOURCES})

if(UNIX)
    target_link_libraries(datagen PRIVATE pthread)
endif()
