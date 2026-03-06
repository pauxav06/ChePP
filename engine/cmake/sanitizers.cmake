if(NOT MSVC)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        message(STATUS "Enabling Address and Undefined Behavior Sanitizers")

        add_compile_options(
                -fsanitize=address
                -fsanitize=undefined
                -fno-omit-frame-pointer
        )

        add_link_options(
                -fsanitize=address
                -fsanitize=undefined
        )
    endif()
endif()