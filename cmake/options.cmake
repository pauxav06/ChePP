add_library(options INTERFACE)

if(MSVC)
    target_compile_options(options INTERFACE /bigobj)
else()
    if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        target_compile_options(options INTERFACE -fexperimental-library)
    endif()
    #add_compile_options(-static-libgcc -static-libstdc++)
    #add_link_options(-static-libgcc -static-libstdc++)
endif()

