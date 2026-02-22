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

add_library(ChePP_core STATIC core.cpp ${fathom_SOURCE_DIR}/src/tbprobe.c)

target_compile_definitions(ChePP_core PUBLIC TB_NO_THREADS=1)

target_link_libraries(ChePP_core PUBLIC tl::expected argparse)
target_include_directories(ChePP_core PRIVATE ${fathom_SOURCE_DIR}/src)
target_include_directories(ChePP_core PUBLIC ${hedley_SOURCE_DIR}  include)