cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

project(pokerbots_cpp VERSION 0.1.0 LANGUAGES CXX)
add_compile_options(-std=c++11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# find_package(BOOST)
add_subdirectory(./pokerbots_cpp_python)

find_package(Boost REQUIRED COMPONENTS system program_options)
add_executable(cpp_player
  cpp_skeleton/actions.cpp
  cpp_skeleton/bot.hpp
  cpp_skeleton/runner.cpp
  cpp_skeleton/states.cpp
  player.hpp
  player.cpp
  main.cpp)
target_link_libraries(cpp_player
  ${Boost_LIBRARIES}
  pokerbots_cpp) 
