if(NOT CMAKE_CROSSCOMPILING)
    add_executable(bin2cpp bin2cpp.cpp)
    target_link_libraries(bin2cpp PRIVATE options warnings optimization compat target argparse)
    export(TARGETS bin2cpp FILE ${CMAKE_BINARY_DIR}/Bin2CppConfig.cmake)
else()
    find_package(Bin2Cpp REQUIRED)
endif()

function(embed_bin)
    cmake_parse_arguments(EMBED "" "TARGET;NAME;BIN;DIR;QUALIFIER" "" ${ARGN})
    file(MAKE_DIRECTORY ${EMBED_DIR}/generated)
    set(OUTPUT ${EMBED_DIR}/generated/${EMBED_NAME}.h)
    add_custom_command(
            OUTPUT ${OUTPUT}
            BYPRODUCTS ${EMBED_DIR}/generated/${EMBED_NAME}.inc

            COMMAND $<TARGET_FILE:bin2cpp> --input ${EMBED_BIN} --name ${EMBED_NAME} --output-dir ${EMBED_DIR} --q ${EMBED_QUALIFIER}

            DEPENDS ${EMBED_BIN} bin2cpp
            VERBATIM
    )

    add_custom_target(${EMBED_NAME}_GENERATED ALL DEPENDS ${OUTPUT})
    add_dependencies(${EMBED_TARGET} ${EMBED_NAME}_GENERATED)
    target_include_directories(${EMBED_TARGET} PUBLIC ${EMBED_DIR})
endfunction()

add_subdirectory(core)
add_subdirectory(nnue)
add_subdirectory(engine)

add_executable(ChePP main.cpp)
target_link_libraries(ChePP options compat warnings optimization target ChePP_core ChePP_nnue ChePP_engine)
