cmake_minimum_required(VERSION 3.14)
project(ChePP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(CPU_PROBE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpu_probe)
set(CPU_PROBE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu_probe_build)
set(CPU_PROBE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu_probe_install)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${CPU_PROBE_BUILD_DIR})
file(MAKE_DIRECTORY ${CPU_PROBE_INSTALL_DIR})
file(MAKE_DIRECTORY ${GENERATED_DIR})

execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CPU_PROBE_SRC_DIR} -B ${CPU_PROBE_BUILD_DIR}
        -DCMAKE_INSTALL_PREFIX=${CPU_PROBE_INSTALL_DIR}
        RESULT_VARIABLE cpures
        OUTPUT_VARIABLE cpu_out
        ERROR_VARIABLE cpu_err
)
message(STATUS "CPU probe configure output:\n${cpu_out}")
if(NOT cpures EQUAL 0)
    message(FATAL_ERROR "CPU probe configure failed:\n${cpu_err}")
endif()

execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${CPU_PROBE_BUILD_DIR} --target cpu_feature_flags
        RESULT_VARIABLE cpures
        OUTPUT_VARIABLE cpu_out
        ERROR_VARIABLE cpu_err
)
message(STATUS "CPU probe build output:\n${cpu_out}")
if(NOT cpures EQUAL 0)
    message(FATAL_ERROR "CPU probe build failed:\n${cpu_err}")
endif()

execute_process(
        COMMAND ${CPU_PROBE_BUILD_DIR}/cpu_feature_flags
        WORKING_DIRECTORY ${GENERATED_DIR}
        RESULT_VARIABLE cpures
        OUTPUT_VARIABLE cpu_out
        ERROR_VARIABLE cpu_err
)
message(STATUS "CPU probe execution output:\n${cpu_out}")
if(NOT cpures EQUAL 0)
    message(FATAL_ERROR "CPU probe execution failed:\n${cpu_err}")
endif()

include(${GENERATED_DIR}/cpu_flags.cmake)
add_library(cpu_flags INTERFACE)
target_compile_options(cpu_flags INTERFACE ${CPU_COMPILE_FLAGS})
message(" cpu compile flags " ${CPU_COMPILE_FLAGS})


if(MSVC)
    add_compile_options(
            /W4
            /permissive-
            /Zi
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/O2)
    else()
        add_compile_options(/O)
    endif()
else()
    add_compile_options(
            -g
            -Wall
            -Wextra
            -pedantic
            -mpopcnt
            -march=native
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O3)
    else()
        add_compile_options(-O)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            message(STATUS "Enabling AddressSanitizer")
            add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
            add_link_options(-fsanitize=address)
        endif()
    endif()
endif()


FetchContent_Declare(
        fathom
        GIT_REPOSITORY https://github.com/basil00/Fathom.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fathom)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
)
FetchContent_MakeAvailable(googletest)
enable_testing()

set(NETWORK_BIN ${CMAKE_CURRENT_SOURCE_DIR}/resources/latest.net)
set(NETWORK_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/ChePP/engine/layers.h)
set(NETWORK_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/layers.cpp)
set(NETWORK_CFG ${CMAKE_CURRENT_SOURCE_DIR}/resources/layers.json)

add_executable(embed_layers src/embed_layers.cpp)
target_link_libraries(embed_layers PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(embed_layers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../external)
target_include_directories(embed_layers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(
        OUTPUT ${NETWORK_HEADER} ${NETWORK_SOURCE}
        COMMAND $<TARGET_FILE:embed_layers>
        --input ${NETWORK_BIN}
        --header ${NETWORK_HEADER}
        --cpp ${NETWORK_SOURCE}
        --config ${NETWORK_CFG}
        DEPENDS ${NETWORK_BIN} ${NETWORK_CFG} src/embed_layers.cpp embed_layers
        COMMENT "Generating embedded NNUE network"
)

add_custom_target(embed_network ALL
        DEPENDS ${NETWORK_HEADER} ${NETWORK_SOURCE}
)

set(ENGINE_SOURCES
        ${fathom_SOURCE_DIR}/src/tbprobe.c
        ${NETWORK_SOURCE}
)

add_library(ChePP_engine STATIC ${ENGINE_SOURCES})
add_dependencies(ChePP_engine embed_network)

target_link_libraries(ChePP_engine cpu_flags)
target_include_directories(ChePP_engine PUBLIC
        include/ChePP/engine
        include/ChePP/nnue
        ${CMAKE_CURRENT_SOURCE_DIR}/../external
        ${fathom_SOURCE_DIR}
        ${highway_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${GENERATED_DIR}
)

target_compile_definitions(ChePP_engine PRIVATE TB_NO_THREADS=1)


execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_HASH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT GIT_HASH)
    set(GIT_HASH "nogit")
endif()
set(ENGINE_VERSION "${GIT_HASH}" CACHE STRING "Engine version string")


add_executable(ChePP_${ENGINE_VERSION} src/main.cpp)
target_link_libraries(ChePP_${ENGINE_VERSION} PRIVATE ChePP_engine)

set_target_properties(ChePP_${ENGINE_VERSION} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/versions
)

# add_executable(ChePP_benchmark src/benchmark.cpp)
# target_link_libraries(ChePP_benchmark PRIVATE ChePP_engine)
# add_dependencies(ChePP_benchmark ChePP_engine)
# set_target_properties(ChePP_benchmark PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin
# )

add_subdirectory(gtests)
