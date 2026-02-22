cmake_policy(SET CMP0079 NEW)

set(HWY_ENABLE_CONTRIB OFF)
set(HWY_ENABLE_LTO ON)
set(HWY_ENABLE_TESTS OFF)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG master)
FetchContent_MakeAvailable(highway)

FetchContent_Declare(mdspan GIT_REPOSITORY https://github.com/kokkos/mdspan.git GIT_TAG stable)
FetchContent_MakeAvailable(mdspan)

add_library(ChePP_nnue STATIC nnue.cpp)

target_link_libraries(ChePP_nnue PRIVATE mdspan ChePP_core)
target_link_libraries(ChePP_nnue PUBLIC hwy)
target_include_directories(ChePP_nnue PUBLIC .)
target_compile_definitions(ChePP_nnue PUBLIC HWY_COMPILE_ONLY_STATIC HWY_AUTOTUNE_STDSORT)
