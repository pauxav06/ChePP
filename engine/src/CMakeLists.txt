add_subdirectory(engine)

add_executable(ChePP main.cpp)
target_link_libraries(ChePP ChePP_engine)