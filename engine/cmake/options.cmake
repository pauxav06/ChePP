add_library(options INTERFACE)

if(MSVC)
    target_compile_options(options INTERFACE /bigobj)
else()
    if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        target_compile_options(target INTERFACE -fexperimental-library)
    endif()
endif()

