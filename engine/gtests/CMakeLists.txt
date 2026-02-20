include(FetchContent)

set (TEST_SOURCES
        perft_tests.cpp
        repetitions_test.cpp
        zobrist_tests.cpp
        nnue_test.cpp
        types_test.cpp
        affine_test.cpp
        accum_test.cpp
        relu_test.cpp
        #network_test.cpp
)

add_executable(ChePP_tests ${TEST_SOURCES})

target_include_directories(ChePP_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/include
        ${CMAKE_SOURCE_DIR}/engine/gtests
)

target_link_libraries(ChePP_tests PRIVATE
        hwy_test
        GTest::gtest_main
        ChePP_engine
)

include(GoogleTest)
gtest_discover_tests(ChePP_tests PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
#gtest_discover_tests(ChePP_tests)
