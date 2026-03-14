target_compile_options(options INTERFACE
        $<$<AND:$<BOOL:${SANITIZE}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -fsanitize=address
            -fsanitize=undefined
            -fno-omit-frame-pointer
        >
)

target_link_options(options INTERFACE
        $<$<AND:$<BOOL:${SANITIZE}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -fsanitize=address
            -fsanitize=undefined
        >
)