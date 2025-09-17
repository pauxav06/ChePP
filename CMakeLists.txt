cmake_minimum_required(VERSION 3.16)
project(ChePP LANGUAGES C CXX)

add_subdirectory(engine)
add_subdirectory(nnue_training)

enable_testing()
