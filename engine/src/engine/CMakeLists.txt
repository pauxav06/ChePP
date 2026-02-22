set(NETWORK_BIN ${CMAKE_CURRENT_SOURCE_DIR}/weights.net)
set(NETWORK_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/weights.h)
set(NETWORK_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/weights.cpp)
set(NETWORK_CFG ${CMAKE_CURRENT_SOURCE_DIR}/weights.json)

add_executable(embed_layers embed_layers.cpp)
target_link_libraries(embed_layers PRIVATE nlohmann_json::nlohmann_json argparse)
target_include_directories(embed_layers PRIVATE ${CMAKE_SOURCE_DIR}/../external)

add_custom_command(
        OUTPUT ${NETWORK_HEADER} ${NETWORK_SOURCE}
        COMMAND $<TARGET_FILE:embed_layers>
        --input ${NETWORK_BIN}
        --header ${NETWORK_HEADER}
        --cpp ${NETWORK_SOURCE}
        --config ${NETWORK_CFG}
        DEPENDS ${NETWORK_BIN} ${NETWORK_CFG} embed_layers.cpp embed_layers
        COMMENT "Generating embedded NNUE network"
)

add_custom_target(embed_network ALL DEPENDS ${NETWORK_HEADER} ${NETWORK_SOURCE})

set(ENGINE_SOURCES ${NETWORK_SOURCE})

add_subdirectory(core)
add_subdirectory(nnue)

add_library(ChePP_engine STATIC ${ENGINE_SOURCES})
add_dependencies(ChePP_engine embed_network)

target_link_libraries(ChePP_engine ChePP_core ChePP_nnue)
target_include_directories(ChePP_engine PUBLIC .)

target_compile_definitions(ChePP_engine PRIVATE CHEPP_USE_TB=1)