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

add_executable(dump_bin dump_bin.cpp)
target_link_libraries(dump_bin PRIVATE warnings optimization target argparse tl::expected)
target_include_directories(dump_bin PRIVATE ${hedley_SOURCE_DIR})

function(generate_bin)
    cmake_parse_arguments(DUMP "" "NAME;OUT_DIR;BIN" "" ${ARGN})

    set(BIN ${DUMP_OUT_DIR}/${DUMP_NAME}.bin)

    add_custom_command(
            OUTPUT ${BIN}

            COMMAND ${CMAKE_COMMAND} -E make_directory ${DUMP_OUT_DIR}
            COMMAND $<TARGET_FILE:dump_bin> --target ${DUMP_NAME} --output ${BIN}

            DEPENDS dump_bin
            VERBATIM
    )

    set(${DUMP_BIN} ${BIN} PARENT_SCOPE)
endfunction()

add_library(ChePP_core STATIC core.cpp ${fathom_SOURCE_DIR}/src/tbprobe.c)

set(GEN_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/generated)
set(ASSETS magic_rooks magic_bishops lines from_to)

foreach(name IN LISTS ASSETS)
    set(GEN_BIN "")
    generate_bin(NAME ${name} OUT_DIR ${GEN_DIR} BIN GEN_BIN)
    embed_bin(TARGET ChePP_core NAME ${name} BIN ${GEN_BIN} DIR ${GEN_DIR} QUALIFIER constexpr)
endforeach ()

target_include_directories(ChePP_core PUBLIC ${GEN_DIR})

target_compile_definitions(ChePP_core PUBLIC TB_NO_THREADS=1 CHEPP_USE_TB=1)
target_link_libraries(ChePP_core PUBLIC tl::expected)
target_include_directories(ChePP_core PRIVATE warnings optimization target ${fathom_SOURCE_DIR}/src)
target_include_directories(ChePP_core PUBLIC include ${hedley_SOURCE_DIR})
