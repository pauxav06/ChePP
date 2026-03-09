add_subdirectory(src)

if (NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(gtests)
endif()
