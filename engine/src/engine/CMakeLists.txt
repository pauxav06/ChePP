add_library(ChePP_engine STATIC engine.cpp)

target_link_libraries(ChePP_engine PUBLIC options compat warnings optimization target ChePP_core ChePP_nnue noexcept)
target_include_directories(ChePP_engine PUBLIC include)
enable_lto(ChePP_engine)