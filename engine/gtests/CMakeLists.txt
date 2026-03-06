include(FetchContent)

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found. Fetching from GitHub...")

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

set (CORE_SOURCES
    perft_tests.cpp
    repetitions_test.cpp
    zobrist_tests.cpp
    types_test.cpp
)

set(HWY_TEST_STANDALONE OFF)
if(HWY_TEST_STANDALONE)
    set(HWY_GTEST_LIBS "")
else()
    if(HWY_SYSTEM_GTEST)
        if (CMAKE_VERSION VERSION_LESS 3.20)
            set(HWY_GTEST_LIBS GTest::GTest GTest::Main)
        else()
            set(HWY_GTEST_LIBS GTest::gtest GTest::gtest_main)
        endif()
    else()
        set(HWY_GTEST_LIBS gtest gtest_main)
    endif()
endif()

set (NNUE_SOURCES accum_test.cpp affine_test.cpp)

add_executable(ChePP_core_tests ${CORE_SOURCES})
add_executable(ChePP_nnue_tests ${NNUE_SOURCES})

target_compile_options(ChePP_nnue_tests PRIVATE -DHWY_IS_TEST=1)
if(HWY_TEST_STANDALONE)
    target_compile_options(ChePP_nnue_tests PRIVATE -DHWY_TEST_STANDALONE=1)
endif()

target_link_libraries(ChePP_core_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        ChePP_core
        warnings
        optimization
)


target_link_libraries(ChePP_nnue_tests PRIVATE
        hwy
        hwy_test
        ${HWY_TEST_LIBS}
        GTest::gtest
        GTest::gtest_main
        ChePP_core
        mdspan
        ChePP_nnue
        warnings
        optimization
)

target_include_directories(ChePP_nnue_tests PRIVATE . ${ChePP_nnue_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(ChePP_core_tests)
gtest_discover_tests(ChePP_nnue_tests)
