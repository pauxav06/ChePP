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
