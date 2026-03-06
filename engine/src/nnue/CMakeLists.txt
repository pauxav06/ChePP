cmake_policy(SET CMP0079 NEW)

set(HWY_ENABLE_CONTRIB OFF)
set(HWY_ENABLE_LTO ON)
set(HWY_ENABLE_TESTS OFF)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG master)
FetchContent_MakeAvailable(highway)

FetchContent_Declare(mdspan GIT_REPOSITORY https://github.com/kokkos/mdspan.git GIT_TAG stable)
FetchContent_MakeAvailable(mdspan)

add_library(ChePP_nnue STATIC nnue.cpp)

embed_bin(TARGET ChePP_nnue NAME weights BIN ${CMAKE_CURRENT_SOURCE_DIR}/weights/weights.net DIR ${CMAKE_CURRENT_SOURCE_DIR}/weights QUALIFIER "")

target_link_libraries(ChePP_nnue PRIVATE warnings optimization target ChePP_core mdspan)
target_link_libraries(ChePP_nnue PUBLIC hwy)
target_compile_definitions(ChePP_nnue PUBLIC HWY_COMPILE_ONLY_STATIC HWY_AUTOTUNE_STDSORT)
target_include_directories(ChePP_nnue PUBLIC include)
