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

set(MAGICS_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/magics.h)
set(MAGICS_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/magics.cpp)

add_executable(magic_embedder embed_magics.cpp)
target_link_libraries(magic_embedder PRIVATE nlohmann_json::nlohmann_json argparse)
target_include_directories(magic_embedder PRIVATE ${hedley_SOURCE_DIR})

add_custom_command(
        OUTPUT ${MAGICS_HEADER} ${MAGICS_SOURCE}
        COMMAND $<TARGET_FILE:magic_embedder>
        --header ${MAGICS_HEADER}
        --cpp ${MAGICS_SOURCE}
        DEPENDS embed_magics.cpp magic_embedder
        COMMENT "Generating embedded magics"
)

add_custom_target(embed_magics ALL DEPENDS ${MAGICS_HEADER} ${MAGICS_SOURCE})

add_library(ChePP_core STATIC core.cpp ${fathom_SOURCE_DIR}/src/tbprobe.c ${MAGICS_SOURCE})
add_dependencies(ChePP_core embed_magics)

target_compile_definitions(ChePP_core PUBLIC TB_NO_THREADS=1)

target_link_libraries(ChePP_core PUBLIC tl::expected argparse)
target_include_directories(ChePP_core PRIVATE ${fathom_SOURCE_DIR}/src)
target_include_directories(ChePP_core PUBLIC ${hedley_SOURCE_DIR}  include)