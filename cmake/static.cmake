add_library(static INTERFACE)

option(STATIC "Build static binaries" OFF)

target_compile_options(static INTERFACE
        $<$<BOOL:${STATIC}>:-static-libgcc -static-libstdc++ -static>
)

target_link_options(static INTERFACE
        $<$<BOOL:${STATIC}>:-static-libgcc -static-libstdc++ -static>
)

