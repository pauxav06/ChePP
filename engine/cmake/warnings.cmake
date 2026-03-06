add_library(warnings INTERFACE)

if(MSVC)
    target_compile_options(warnings INTERFACE
            /W4
            /permissive-
            /Zc:__cplusplus
    )
else()
    target_compile_options(warnings INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Wuninitialized
            -Wstrict-aliasing
            -Wstrict-overflow
    )
endif()