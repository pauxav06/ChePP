include(FetchContent)

find_package(GTest QUIET)
if(NOT GTest_FOUND AND NOT TARGET gtest)
    message(STATUS "GoogleTest not found. Fetching from GitHub...")

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

set (CORE_SOURCES
    perft_tests.cpp
    repetitions_test.cpp
    zobrist_tests.cpp
    types_test.cpp
)
add_executable(ChePP_core_tests ${CORE_SOURCES})

target_link_libraries(ChePP_core_tests PRIVATE GTest::gtest GTest::gtest_main ChePP_core
        options compat target warnings optimization)

gtest_discover_tests(ChePP_core_tests)

set (NNUE_SOURCES accum_test.cpp affine_test.cpp relu_test.cpp)

if(HWY_TEST_STANDALONE)
    set(NNUE_GTEST_LIBS "")
else()
    set(NNUE_GTEST_LIBS GTest::gtest GTest::gtest_main)
endif()

foreach (TESTFILE IN LISTS NNUE_SOURCES)
    get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
    add_executable(${TESTNAME} ${TESTFILE})
    target_link_libraries(${TESTNAME} PRIVATE
            ChePP_core ChePP_nnue hwy hwy_test ${NNUE_GTEST_LIBS}
            options compat warnings optimization)
    target_include_directories(${TESTNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${TESTNAME} PRIVATE -DHWY_IS_TEST=1)

    gtest_discover_tests(${TESTNAME})
endforeach ()