cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(
            /O2
            /W4
            /permissive-
            /Zi
            /arch:AVX2
    )
else()
    add_compile_options(
            -O3
            -g
            -march=native
            -Wall
            -Wextra
            -pedantic
    )
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

