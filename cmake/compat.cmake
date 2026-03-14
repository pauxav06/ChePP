add_library(compat INTERFACE)
target_include_directories(compat INTERFACE ${CMAKE_SOURCE_DIR}/compat)

set(PRV_FLAGS ${CMAKE_REQUIRED_FLAGS})

if (MSVC)
    set(CMAKE_REQUIRED_FLAGS "/std:c++20")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    set(CMAKE_REQUIRED_FLAGS "-std=c++20 -fexperimental-library")
else()
    set(CMAKE_REQUIRED_FLAGS "-std=c++20")
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
    set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(EXPECTED_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
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

unset(HAVE_RANGES CACHE)
check_source_compiles(CXX [[
    #include <version>
    #ifndef __cpp_lib_ranges
    #error
    #endif
    int main(void) { return 0; }
]] HAVE_RANGES)

if(NOT HAVE_RANGES)
    message(STATUS "std::ranges not found, fetching ranges-v3")

    FetchContent_Declare(range-v3 GIT_REPOSITORY https://github.com/ericniebler/range-v3.git GIT_TAG master)
    FetchContent_MakeAvailable(range-v3)

    target_link_libraries(compat INTERFACE range-v3)
else()
    target_compile_definitions(compat INTERFACE USE_STD_RANGES)
    message(STATUS "using std::format")
endif()

unset(HAVE_SPAN CACHE)
check_source_compiles(CXX [[
    #include <version>
    #ifndef __cpp_lib_span
    #error
    #endif
    int main(void) { return 0; }
]] HAVE_SPAN)

if(NOT HAVE_SPAN)
    message(STATUS "std::span not found, fetching span")

    FetchContent_Declare(
            tcbrindle_span
            GIT_REPOSITORY https://github.com/tcbrindle/span.git
            GIT_TAG        master
    )
    FetchContent_MakeAvailable(tcbrindle_span)

    target_include_directories(compat INTERFACE ${tcbrindle_span_SOURCE_DIR}/include)
else()
    target_compile_definitions(compat INTERFACE USE_STD_SPAN)
    message(STATUS "using std::format")
endif()

set(CMAKE_REQUIRED_FLAGS ${PRV_FLAGS})