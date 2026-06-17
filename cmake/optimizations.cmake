add_library(optimization INTERFACE)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

target_compile_options(optimization INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Zi>
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:/O2 /Oi /Gy>

        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O1 -g>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<CONFIG:Debug>>>:-O3 -fstrict-aliasing -fomit-frame-pointer>
)

target_compile_definitions(optimization INTERFACE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<CONFIG:Debug>>>:NDEBUG>
)