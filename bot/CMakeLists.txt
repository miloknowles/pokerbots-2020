cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(pokerbots_cpp VERSION 0.1.0 LANGUAGES CXX)
add_compile_options(-std=c++11)

find_package(Boost REQUIRED COMPONENTS system program_options filesystem)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Build our library first.
add_subdirectory(./pokerbots_cpp)

# Build the pure C++ bot.
message(STATUS "Building the pure C++ bot")
find_package(Boost REQUIRED COMPONENTS system program_options)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp_skeleton)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pokerbots_cpp/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pokerbots_cpp/OMPEval)
add_executable(cpp_player
  cfr_player.cpp
  # cfr_player.hpp
  main.cpp)
target_link_libraries(cpp_player
  cpp_skeleton
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_player
  ${PROJECT_NAME}_cfr)

# Copy necessary files to final executable directory.
add_custom_command(
  TARGET cpp_player POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_SOURCE_DIR}/pokerbots_cpp/preflop_equity.txt
          ${CMAKE_CURRENT_BINARY_DIR}/preflop_equity.txt)

add_custom_command(
  TARGET cpp_player POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_SOURCE_DIR}/avg_strategy.txt
          ${CMAKE_CURRENT_BINARY_DIR}/avg_strategy.txt)
