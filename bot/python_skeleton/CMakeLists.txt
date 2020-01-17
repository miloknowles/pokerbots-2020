cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

project(pokerbots_cpp VERSION 0.1.0 LANGUAGES CXX)
add_compile_options(-std=c++11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(BUILD_TESTS OFF)
# set(BUILD_PYTHON_WRAPPER OFF)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Build our library first.
add_subdirectory(./pokerbots_cpp_python)

# Build the pure C++ bot.
if (NOT BUILD_PYTHON_WRAPPER)
  message(STATUS "BUILD_PYTHON_WRAPPER is OFF, building pure C++ bot")
  find_package(Boost REQUIRED COMPONENTS system program_options)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pokerbots_cpp_python/)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pokerbots_cpp_python/OMPEval)
  add_executable(cpp_player
    cpp_skeleton/actions.cpp
    cpp_skeleton/bot.hpp
    cpp_skeleton/runner.cpp
    cpp_skeleton/states.cpp
    history_tracker.cpp
    history_tracker.hpp
    player.hpp
    player.cpp
    main.cpp)
  target_link_libraries(cpp_player
    ${Boost_LIBRARIES}
    pokerbots_cpp)

  # Copy necessary files to final executable directory.
  add_custom_command(
    TARGET cpp_player POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/pokerbots_cpp_python/preflop_equity.txt
            ${CMAKE_CURRENT_BINARY_DIR}/preflop_equity.txt)

  enable_testing()  
  set(TEST_SOURCES
    history_tracker.cpp
    history_tracker.hpp
    test_history_tracker.cpp)

  # Make a single executable for all tests.
  # Can use --gtest_filter option to run specific tests.
  add_executable(run_tests ${TEST_SOURCES} ./pokerbots_cpp_python/gtest/gtest-all.cc)
  target_link_libraries(run_tests
    ${PROJECT_NAME}
    pthread)
  set_target_properties(run_tests PROPERTIES COMPILE_FLAGS "-std=c++11")
  add_test(NAME run_tests COMMAND run_tests --gtest_color=yes)
endif()
