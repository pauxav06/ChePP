add_library(options INTERFACE)

option(STATIC "Build static binaries" OFF)

target_compile_options(options INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
        $<$<CXX_COMPILER_ID:AppleClang>:-fexperimental-library>
        $<$<CXX_COMPILER_ID:GNU>:-fconcepts -Wfatal-errors>
        $<$<BOOL:${STATIC}>:-static-libgcc -static-libstdc++ -static>
)

target_link_options(options INTERFACE
        $<$<BOOL:${STATIC}>:-static-libgcc -static-libstdc++ -static>
)

