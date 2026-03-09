cmake_minimum_required(VERSION 3.5)

cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0069 NEW)

project(ChePP LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(CheckSourceCompiles)
include(CheckSourceRuns)

include(cmake/options.cmake)
include(cmake/warnings.cmake)
include(cmake/optimizations.cmake)
include(cmake/sanitizers.cmake)
include(cmake/targets.cmake)
include(cmake/compat.cmake)

# Dependencies

FetchContent_Declare(argparse GIT_REPOSITORY https://github.com/p-ranav/argparse.git GIT_TAG v3.1)
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(hedley GIT_REPOSITORY https://github.com/nemequ/hedley.git GIT_TAG master)
FetchContent_MakeAvailable(hedley)

set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG master)
FetchContent_MakeAvailable(highway)

FetchContent_Declare(fathom GIT_REPOSITORY https://github.com/jdart1/Fathom.git GIT_TAG master)
FetchContent_MakeAvailable(fathom)

include(CTest)
enable_testing()

add_subdirectory(engine)
#add_subdirectory(nnue_training)
