cmake_minimum_required(VERSION 3.5)
project(ChePP LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.1
)

FetchContent_MakeAvailable(argparse)

FetchContent_Declare(
        tl_expected
        GIT_REPOSITORY https://github.com/TartanLlama/expected.git
        GIT_TAG v1.1.0
)

FetchContent_MakeAvailable(tl_expected)

FetchContent_Declare(
        hedley
        GIT_REPOSITORY https://github.com/nemequ/hedley.git
        GIT_TAG master
)

FetchContent_MakeAvailable(hedley)

add_subdirectory(engine)
#add_subdirectory(nnue_training)
