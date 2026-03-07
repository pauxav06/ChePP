add_library(ChePP_nnue STATIC nnue.cpp)

embed_bin(TARGET ChePP_nnue NAME weights BIN ${CMAKE_CURRENT_SOURCE_DIR}/weights/weights.net DIR ${CMAKE_CURRENT_SOURCE_DIR}/weights QUALIFIER "")

target_link_libraries(ChePP_nnue PRIVATE warnings optimization target ChePP_core)
target_link_libraries(ChePP_nnue PUBLIC hwy compat)
target_compile_definitions(ChePP_nnue PRIVATE HWY_COMPILE_ONLY_STATIC)
target_compile_definitions(ChePP_nnue PUBLIC HWY_AUTOTUNE_STDSORT)
target_include_directories(ChePP_nnue PUBLIC include)
target_include_directories(ChePP_nnue PRIVATE .)
