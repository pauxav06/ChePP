FetchContent_Declare(
        fathom
        GIT_REPOSITORY https://github.com/basil00/Fathom.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fathom)

configure_file(
        tbconfig.h
        ${fathom_SOURCE_DIR}/src/tbconfig.h
        COPYONLY
)

FetchContent_Declare(
        tl_expected
        GIT_REPOSITORY https://github.com/TartanLlama/expected.git
        GIT_TAG v1.1.0
)

FetchContent_MakeAvailable(tl_expected)

FetchContent_Declare(
        hedley
        GIT_REPOSITORY https://github.com/nemequ/hedley.git
        GIT_TAG master
)

FetchContent_MakeAvailable(hedley)

set(TEMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tmp)
set(JSON_NAME ${TEMP_DIR}/manifest.json)

add_executable(embed_bytes embed_magics.cpp)
target_link_libraries(embed_bytes PRIVATE argparse nlohmann_json::nlohmann_json)
target_include_directories(embed_bytes PRIVATE ${hedley_SOURCE_DIR})

add_custom_command(
        OUTPUT ${JSON_NAME}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEMP_DIR}
        COMMAND $<TARGET_FILE:embed_bytes> --bin-dir ${TEMP_DIR} --json ${JSON_NAME}
        DEPENDS embed_bytes
        COMMENT "Embedding bitboards..."
)

set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
set(EMBED_DONE ${GENERATED_DIR}/.done)

add_custom_command(
        OUTPUT ${EMBED_DONE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
        COMMAND $<TARGET_FILE:bin2cpp> ${JSON_NAME} ${GENERATED_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${EMBED_DONE}
        DEPENDS bin2cpp ${JSON_NAME}
        COMMENT "Generating bitboard headers..."
)

add_custom_target(embed ALL DEPENDS ${EMBED_DONE})

add_library(ChePP_core STATIC core.cpp ${fathom_SOURCE_DIR}/src/tbprobe.c)
target_include_directories(ChePP_core PRIVATE ${GENERATED_DIR})
add_dependencies(ChePP_core embed)

target_compile_definitions(ChePP_core PUBLIC TB_NO_THREADS=1)

target_link_libraries(ChePP_core PUBLIC tl::expected argparse)
target_include_directories(ChePP_core PRIVATE ${fathom_SOURCE_DIR}/src)
target_include_directories(ChePP_core PUBLIC ${hedley_SOURCE_DIR}  include)