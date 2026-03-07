add_executable(bin2cpp bin2cpp.cpp)
target_link_libraries(bin2cpp PRIVATE options warnings optimization compat target argparse)

function(embed_bin)
    cmake_parse_arguments(EMBED "" "TARGET;NAME;BIN;DIR;QUALIFIER" "" ${ARGN})

    add_custom_command(
            OUTPUT ${EMBED_DIR}/${EMBED_NAME}.h
            BYPRODUCTS ${EMBED_DIR}/${EMBED_NAME}.inc

            COMMAND $<TARGET_FILE:bin2cpp> --input ${EMBED_BIN} --name ${EMBED_NAME} --output_dir ${EMBED_DIR} --q ${EMBED_QUALIFIER}

            DEPENDS ${EMBED_BIN} bin2cpp
            VERBATIM
    )

    add_custom_target(${EMBED_NAME}_GENERATED ALL DEPENDS  ${EMBED_DIR}/${EMBED_NAME}.h)
    add_dependencies(${EMBED_TARGET} ${EMBED_NAME}_GENERATED)
endfunction()

add_subdirectory(core)
add_subdirectory(nnue)
add_subdirectory(engine)

add_executable(ChePP main.cpp)
target_link_libraries(ChePP options compat warnings optimization target ChePP_core ChePP_nnue ChePP_engine)
