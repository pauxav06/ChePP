add_library(warnings INTERFACE)

target_compile_options(warnings INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wuninitialized -Wstrict-aliasing -Wstrict-overflow>
)