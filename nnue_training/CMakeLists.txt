cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

if(MSVC)
    add_compile_options(
            /W4
            /permissive-
            /Zi
            /arch:AVX2
    )

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/O2)
    endif()
else()
    add_compile_options(
            -g
            -march=native
            -Wall
            -Wextra
            -pedantic
    )

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O3)
    endif()
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

target_link_libraries(datagen PRIVATE nlohmann_json::nlohmann_json)


