
include(cmake/warnings.cmake)
include(cmake/optimizations.cmake)
include(cmake/sanitizers.cmake)
include(cmake/targets.cmake)

add_subdirectory(src)
add_subdirectory(gtests)
enable_testing()