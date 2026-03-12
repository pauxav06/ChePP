target_compile_options(options INTERFACE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>,$<BOOL:${SANITIZE}>>:
            -fsanitize=address
            -fsanitize=undefined
            -fno-omit-frame-pointer
        >
)

target_link_options(options INTERFACE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>,$<BOOL:${SANITIZE}>>:
            -fsanitize=address
            -fsanitize=undefined
        >
)