cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(pokerbots_cpp VERSION 0.1.0 LANGUAGES CXX)
add_compile_options(-std=c++11)

add_library(${PROJECT_NAME} SHARED
  permutation_filter.cpp
  permutation_filter.hpp)

target_link_libraries(${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/libpbots_calc.so
  ${Boost_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

add_executable(main
  main.cpp)
target_link_libraries(main
  ${PROJECT_NAME})
set_target_properties(main PROPERTIES LINKER_LANGUAGE CXX)
