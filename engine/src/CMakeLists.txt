add_subdirectory(engine)

add_executable(bin2cpp bin2cpp.cpp)
target_link_libraries(bin2cpp PRIVATE nlohmann_json::nlohmann_json)

add_executable(ChePP main.cpp)
target_link_libraries(ChePP ChePP_engine)