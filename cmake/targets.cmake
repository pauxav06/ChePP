add_library(target INTERFACE)

set(ARCH "" CACHE STRING "Target architecture")

if(NOT ARCH)
    if(MSVC)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(ARCH "")
        else()
            set(ARCH "SSE2")
        endif()
    else ()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(ARCH "armv8-a")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
            set(ARCH "x86-64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64")
            set(ARCH "power8")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
            set(ARCH "rv64gc")
        else()
            set(ARCH "native")
        endif()
    endif()
endif()

if(MSVC)
    if (ARCH)
        message(STATUS "Using target arch: ${ARCH}")
        target_compile_options(target INTERFACE /arch:${ARCH})
    endif()
else()
    message(STATUS "Using target arch: ${ARCH}")
    target_compile_options(target INTERFACE -march=${ARCH})
endif()

get_target_property(COMP_OPTS target INTERFACE_COMPILE_OPTIONS)

set(PRV_FLAGS ${CMAKE_REQUIRED_FLAGS})

set(CMAKE_REQUIRED_FLAGS ${COMP_OPTS})
unset(HAVE_PEXT CACHE)
check_source_compiles(CXX [[
    #include <immintrin.h>
    int main () { return _pext_u64(0, 9); }
]] HAVE_PEXT)

if (HAVE_PEXT)
    add_compile_definitions(CHEPP_PEXT=1)
else()
    add_compile_definitions(CHEPP_PEXT=0)
endif()

set(CMAKE_REQUIRED_FLAGS ${PRV_FLAGS})