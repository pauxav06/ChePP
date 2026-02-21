cmake_policy(SET CMP0079 NEW)

FetchContent_Declare(
        highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG master
)
set(HWY_ENABLE_CONTRIB OFF)
set(HWY_ENABLE_LTO ON)
set(HWY_ENABLE_TESTS OFF)

FetchContent_MakeAvailable(highway)

add_compile_definitions(HWY_AUTOTUNE_STDSORT)

FetchContent_Declare(
        fathom
        GIT_REPOSITORY https://github.com/basil00/Fathom.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fathom)

configure_file(
        include/ChePP/engine/tbconfig.h
        ${fathom_SOURCE_DIR}/src/tbconfig.h
        COPYONLY
)

if(MSVC)
    add_compile_options(
            /W4
            /permissive-
            /Zi
            /RTC-
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/O)
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
        add_compile_options(-O3 -ffast-math -flto -Ofast -fstrict-aliasing -fomit-frame-pointer)
        add_link_options(-flto)
    else()
        add_compile_options(-O)
        message(STATUS "Enabling AddressSanitizer")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

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
        src/tb.cpp
        ${fathom_SOURCE_DIR}/src/tbprobe.c
        ${NETWORK_SOURCE}
        src/nnue.cpp
)

add_library(ChePP_engine STATIC ${ENGINE_SOURCES})
add_dependencies(ChePP_engine embed_network)

target_link_libraries(ChePP_engine hwy)
target_include_directories(ChePP_engine PUBLIC
        include/ChePP/engine
        include/ChePP/nnue
        src
        ${CMAKE_CURRENT_SOURCE_DIR}/../external
        ${fathom_SOURCE_DIR}
        ${highway_SOURCE_DIR}
)

target_compile_definitions(ChePP_engine PRIVATE CHEPP_USE_TB=1)
target_compile_definitions(ChePP_engine PRIVATE TB_NO_THREADS=1)

add_executable(ChePP src/main.cpp)

set_target_properties(ChePP PROPERTIES
        BUILD_RPATH "/opt/gcc-cpp26/lib64"
)
target_link_libraries(ChePP PRIVATE ChePP_engine)
set_target_properties(ChePP PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/versions
)

add_subdirectory(gtests)
