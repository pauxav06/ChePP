add_library(target INTERFACE)

set(TARGET_ARCH "" CACHE STRING "Target architecture (SSE2, AVX, AVX2, AVX512)")
if(MSVC)
    if (TARGET_ARCH)
        message(STATUS "MSVC detected. Using target arch: ${TARGET_ARCH}")
        if(TARGET_ARCH STREQUAL "SSE2")
            target_compile_options(target INTERFACE /arch:SSE2)
        elseif(TARGET_ARCH STREQUAL "AVX")
            target_compile_options(target INTERFACE /arch:AVX)
        elseif(TARGET_ARCH STREQUAL "AVX2")
            target_compile_options(target INTERFACE /arch:AVX2)
        elseif(TARGET_ARCH STREQUAL "AVX512")
            target_compile_options(target INTERFACE /arch:AVX512)
        else()
            message(FATAL_ERROR "Unsupported TARGET_ARCH: ${TARGET_ARCH}")
        endif()
    endif()
else()
        target_compile_options(target INTERFACE -march=native)
endif()

get_target_property(COMP_OPTS target INTERFACE_COMPILE_OPTIONS)

set(CMAKE_REQUIRED_FLAGS ${COMP_OPTS})
unset(HAVE_PEXT CACHE)
check_source_runs(CXX [[
    #include <immintrin.h>
    int main () { return _pext_u64(0, 9); }
]] HAVE_PEXT)

if (HAVE_PEXT)
    add_compile_definitions(CHEPP_PEXT=1)
else()
    add_compile_definitions(CHEPP_PEXT=0)
endif()
