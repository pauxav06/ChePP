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

add_library(ChePP_nnue_core INTERFACE)

embed_bin(TARGET ChePP_nnue_core NAME weights BIN ${NNUE_FILE} DIR ${CMAKE_CURRENT_BINARY_DIR} QUALIFIER "")
target_include_directories(ChePP_nnue_core INTERFACE include)
target_include_directories(ChePP_nnue_core INTERFACE .)

add_library(ChePP_nnue STATIC nnue.cpp)
target_link_libraries(ChePP_nnue PRIVATE compat options warnings optimization ChePP_core noexcept)
target_link_libraries(ChePP_nnue PUBLIC hwy ChePP_nnue_core)
