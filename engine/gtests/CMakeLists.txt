include(FetchContent)


set (TEST_SOURCES
        perft_tests.cpp
        repetitions_test.cpp
        zobrist_tests.cpp
)

add_executable(ChePP_tests ${TEST_SOURCES})

target_include_directories(ChePP_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/include
)

target_link_libraries(ChePP_tests PRIVATE
        GTest::gtest_main
        ChePP_engine
)


include(GoogleTest)
gtest_discover_tests(ChePP_tests)
