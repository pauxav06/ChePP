cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(
            /O2
            /W4
            /permissive-
            /Zi
            /arch:AVX2
    )
else()
    add_compile_options(
            -O3
            -g
            -march=native
            -Wall
            -Wextra
            -pedantic
    )
endif()


include(FetchContent)

# Fathom is a used to probe tablebases
FetchContent_Declare(
        fathom
        GIT_REPOSITORY https://github.com/basil00/Fathom.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fathom)


set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
# Highway is used for cross platform SIMD code
set(HIGHWAY_BUILD_TESTS OFF CACHE BOOL "Disable Highway tests" FORCE)

FetchContent_Declare(
        highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG master
)
set(BUILD_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

if(NOT TARGET gtest)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG main
    )

    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()


# NNUE weights, embed directly into he binary, avoids any relative path issue at runtime
set(NETWORK_BIN ${CMAKE_CURRENT_SOURCE_DIR}/resources/latest.net)
set(NETWORK_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/ChePP/engine/network_net.h)
set(NETWORK_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/network_net.cpp)
set(NETWORK_CFG ${CMAKE_CURRENT_SOURCE_DIR}/resources/layers.json)

add_executable(bin2h ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2h.cpp)
target_link_libraries(bin2h PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(bin2h PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../external)

add_custom_command(
        OUTPUT ${NETWORK_HEADER} ${NETWORK_SOURCE}
        COMMAND $<TARGET_FILE:bin2h> --raw ${NETWORK_BIN} --config ${NETWORK_CFG} --header ${NETWORK_HEADER} --cpp ${NETWORK_SOURCE}
        DEPENDS ${NETWORK_BIN} ${NETWORK_CFG} scripts/bin2h.cpp bin2h
        COMMENT "Generating embedded network_net resource"
)

add_custom_target(embed_network ALL
        DEPENDS ${NETWORK_HEADER} ${NETWORK_SOURCE}
)

# Sources
set(ENGINE_SOURCES
        ${fathom_SOURCE_DIR}/src/tbprobe.c
        ${NETWORK_SOURCE}
)

add_library(ChePP_engine STATIC ${ENGINE_SOURCES})

target_include_directories(ChePP_engine PUBLIC
        ${CMAKE_SOURCE_DIR}/engine/include
        ${CMAKE_SOURCE_DIR}/external
        ${fathom_SOURCE_DIR}
        ${highway_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_definitions(ChePP_engine PRIVATE TB_NO_THREADS=1)


# Store different versions of the exeutable
# Override the name with -DENGINE_VERSION=name
# Useful for benchmarking elo gains
execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_HASH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT GIT_HASH)
    set(GIT_HASH "nogit")
endif()

set(ENGINE_VERSION "${GIT_HASH}" CACHE STRING "Engine version string")

# The UCI version of the engine
# We could also store the static lib but we don-t really need it
# (Do we even need the static lib to be compiled separately)
add_executable(ChePP_${ENGINE_VERSION} src/main.cpp)
target_link_libraries(ChePP_${ENGINE_VERSION} PRIVATE ChePP_engine)

set_target_properties(ChePP_${ENGINE_VERSION} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/versions
)

#add_executable(ChePP_benchmark src/benchmark.cpp)
#target_link_libraries(ChePP_benchmark PRIVATE ChePP_engine)
#add_dependencies(ChePP_benchmark ChePP_engine)

#set_target_properties(ChePP_benchmark PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin
#)



add_subdirectory(gtests)
