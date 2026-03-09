add_library(ChePP_nnue STATIC nnue.cpp)

set(NNUE_URL "https://github.com/pauxav06/chepp_NNUE/releases/download/v0.2.0/weights.net")
set(NNUE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/weights.net")

if(NOT EXISTS "${NNUE_FILE}")
    message(STATUS "Downloading NNUE network...")
    file(DOWNLOAD
            "${NNUE_URL}"
            "${NNUE_FILE}"
            SHOW_PROGRESS
            STATUS download_status
            LOG log_download
            TLS_VERIFY ON
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download NNUE: ${log_download}")
    endif()
endif()

embed_bin(TARGET ChePP_nnue NAME weights BIN ${NNUE_FILE} DIR ${CMAKE_CURRENT_BINARY_DIR} QUALIFIER "")

target_link_libraries(ChePP_nnue PRIVATE options warnings optimization ChePP_core)
target_link_libraries(ChePP_nnue PUBLIC hwy compat)
#target_compile_definitions(ChePP_nnue PRIVATE HWY_COMPILE_ONLY_STATIC)
target_compile_definitions(ChePP_nnue PUBLIC HWY_AUTOTUNE_STDSORT)
target_include_directories(ChePP_nnue PUBLIC include)
target_include_directories(ChePP_nnue PRIVATE .)
