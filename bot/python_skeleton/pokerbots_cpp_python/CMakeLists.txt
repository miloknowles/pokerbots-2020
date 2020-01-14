cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(pokerbots_cpp VERSION 0.1.0 LANGUAGES CXX)
add_compile_options(-std=c++11)

# Needed to find pyconfig.h
find_package(Boost REQUIRED COMPONENTS python)
# include_directories(/usr/include/python3.5m/)
include_directories(/usr/include/python2.7/)
include_directories()

add_library(${PROJECT_NAME} SHARED
  permutation_filter.cpp
  permutation_filter.hpp)
target_link_libraries(${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/libpbots_calc.so)
  # ${Boost_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

add_library(${PROJECT_NAME}_python SHARED
  bindings.cpp)
target_link_libraries(${PROJECT_NAME}_python
  ${Boost_LIBRARIES}
  ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_python PROPERTIES LINKER_LANGUAGE CXX)

add_executable(main
  main.cpp)
target_link_libraries(main
  ${PROJECT_NAME})
set_target_properties(main PROPERTIES LINKER_LANGUAGE CXX)

# configure_file(./__init__.py ${CMAKE_CURRENT_BINARY_DIR}/__init__.py COPYONLY)
# configure_file(alphafish_python/test.py ${CMAKE_CURRENT_BINARY_DIR}/test.py COPYONLY)
# configure_file(${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}.so ./lib${PROJECT_NAME}.so COPYONLY)

# Suppress prefix "lib" because Python does not allow this prefix
set_target_properties(${PROJECT_NAME}_python PROPERTIES PREFIX "")

# Copy libraries to directory above.
add_custom_command(
  TARGET ${PROJECT_NAME}_python POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_python.so
          ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}_python.so)

add_custom_command(
  TARGET ${PROJECT_NAME}_python POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}.so
          ${CMAKE_SOURCE_DIR}/lib${PROJECT_NAME}.so)

enable_testing()  

# Uncomment this to collect all source files in this directory.
# Currently not used because certain tests need to be excluded.
# file(GLOB TEST_SOURCES "*.cpp")
set(TEST_SOURCES
  test_permutation_filter.cpp)

# Make a single executable for all tests.
# Can use --gtest_filter option to run specific tests.
add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} ./gtest/gtest-all.cc)
target_link_libraries(${PROJECT_NAME}_test
  ${PROJECT_NAME}
  pthread)
set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++11")
add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)
