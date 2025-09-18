cmake_minimum_required(VERSION 3.16)
project(ChePP LANGUAGES C CXX)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)


add_subdirectory(engine)
add_subdirectory(nnue_training)

enable_testing()
