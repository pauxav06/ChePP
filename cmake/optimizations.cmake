add_library(optimization INTERFACE)

set(NO_LTO FALSE CACHE BOOL "Disable Link Time Optimization")
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported AND NOT NO_LTO AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_compile_options(optimization INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Zi>
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:/O2 /Oi /Gy>

        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O1 -g>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<CONFIG:Debug>>>:-O3 -fstrict-aliasing -fomit-frame-pointer>
)

target_compile_definitions(optimization INTERFACE
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<CONFIG:Debug>>>:NDEBUG>
)