add_library(options INTERFACE)

target_compile_options(options INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
        $<$<CXX_COMPILER_ID:AppleClang>:-fexperimental-library>
        $<$<CXX_COMPILER_ID:GNU>:-fconcepts -Wfatal-errors>
)

