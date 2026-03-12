add_library(target INTERFACE)

set(ARCH "" CACHE STRING "Target architecture")

if(NOT ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        if(MSVC)
            set(ARCH "")
        else()
            set(ARCH "armv8-a")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|amd64")
        if(MSVC)
            set(ARCH "SSE2")
        else()
            set(ARCH "x86-64")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64")
        set(ARCH "power8")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
        set(ARCH "rv64gc")
    else()
        set(ARCH "native")
    endif()
endif()

message(STATUS "Using target arch: ${ARCH}")

target_compile_options(target INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<BOOL:${ARCH}>>:/arch:${ARCH}>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<BOOL:${ARCH}>>:-march=${ARCH}>
)

set(CMAKE_REQUIRED_FLAGS ${COMP_OPTS})
unset(HAVE_PEXT CACHE)
check_source_compiles(CXX [[
    #if !defined(__BMI2__)
    #error
    #else
    #include <immintrin.h>
    #endif
    int main () { return _pext_u64(0, 9); }
]] HAVE_PEXT)

if (HAVE_PEXT)
    add_compile_definitions(USE_PEXT=1)
else()
    add_compile_definitions(USE_PEXT=0)
endif()

set(CMAKE_REQUIRED_FLAGS ${PRV_FLAGS})