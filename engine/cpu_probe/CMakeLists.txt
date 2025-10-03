cmake_minimum_required(VERSION 3.14)
project(cpu_probe LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        cpuinfo
        GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
        GIT_TAG main
)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cpuinfo)

add_executable(cpu_feature_flags src/cpu_probe.cpp)
target_link_libraries(cpu_feature_flags PRIVATE cpuinfo)
