cmake_minimum_required(VERSION 3.5)

cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0069 NEW)

project(ChePP LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
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


# Compatibility layer for partial cpp standard support

add_library(compat INTERFACE)
target_include_directories(compat INTERFACE ${CMAKE_SOURCE_DIR}/compat)

set(PRV_FLAGS ${CMAKE_REQUIRED_FLAGS})

if (MSVC)
    set(CMAKE_REQUIRED_FLAGS "/std:c++23")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    set(CMAKE_REQUIRED_FLAGS "-std=c++23 -fexperimental-library")
else()
    set(CMAKE_REQUIRED_FLAGS "-std=c++23")
endif()

unset(HAVE_EXPECTED CACHE)
check_source_compiles(CXX [[
    #include <version>
    #ifndef __cpp_lib_expected
    #error
    #endif
    int main(void) { return 0; }
]] HAVE_EXPECTED)

if(NOT HAVE_EXPECTED)
    message(STATUS "std::expected not found, fetching tl::expected")

    FetchContent_Declare(tl_expected GIT_REPOSITORY https://github.com/TartanLlama/expected.git GIT_TAG v1.1.0)
    FetchContent_MakeAvailable(tl_expected)

    target_link_libraries(compat INTERFACE tl::expected)
else()
    target_compile_definitions(compat INTERFACE USE_STD_EXPEXTED)
    message(STATUS "using std::expected")
endif()

unset(HAVE_MDSPAN CACHE)
check_source_compiles(CXX [[
    #include <version>
    #ifndef __cpp_lib_mdspan
    #error
    #endif
    int main(void) { return 0; }
]] HAVE_MDSPAN)

if(NOT HAVE_MDSPAN)
    message(STATUS "std::mdspan not found, fetching kokkos::mdspan")

    FetchContent_Declare(mdspan GIT_REPOSITORY https://github.com/kokkos/mdspan.git GIT_TAG stable)
    FetchContent_MakeAvailable(mdspan)

    target_link_libraries(compat INTERFACE mdspan)
else()
    target_compile_definitions(compat INTERFACE USE_STD_MDSPAN)
    message(STATUS "using std::mdspan")
endif()

unset(HAVE_FMT CACHE)
check_source_compiles(CXX [[
    #include <version>
    #ifndef __cpp_lib_print
    #error
    #endif
    int main(void) { return 0; }
]] HAVE_FMT)

if(NOT HAVE_FMT)
    message(STATUS "std::format not found, fetching fmtlib")

    set(FMT_MODULE OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG master)
    FetchContent_MakeAvailable(fmt)

    target_link_libraries(compat INTERFACE fmt::fmt)
else()
    target_compile_definitions(compat INTERFACE USE_STD_FMT)
    message(STATUS "using std::format")
endif()

set(CMAKE_REQUIRED_FLAGS ${PRV_FLAGS})

include(CTest)
enable_testing()

add_subdirectory(engine)
#add_subdirectory(nnue_training)
