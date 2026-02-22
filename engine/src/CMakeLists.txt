if(MSVC)
    add_compile_options(
            /W4
            /permissive-
            /Zi
            /RTC-
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/O)
    else()
        add_compile_options(/O)
    endif()
else()
    add_compile_options(
            -g
            -Wall
            -Wextra
            -pedantic
            -mpopcnt
            -march=native
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O3 -ffast-math -flto -Ofast -fstrict-aliasing -fomit-frame-pointer)
        add_compile_definitions(NDEBUG)
        add_link_options(-flto)
    else()
        add_compile_options(-O)
        message(STATUS "Enabling AddressSanitizer")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

add_subdirectory(engine)

add_executable(ChePP main.cpp)
target_link_libraries(ChePP ChePP_engine)