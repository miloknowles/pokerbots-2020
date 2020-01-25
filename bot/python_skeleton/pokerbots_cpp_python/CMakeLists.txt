# Needed to find pyconfig.h
find_package(Boost REQUIRED COMPONENTS python system program_options)
# include_directories(/usr/include/python2.7/)
include_directories(/usr/include/python3.5/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OMPEval/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp_skeleton)
link_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(/usr/local/lib/)

add_library(omp SHARED
  OMPEval/omp/CardRange.cpp
  OMPEval/omp/CardRange.h
  # OMPEval/omp/CombinedRange.cpp
  # OMPEval/omp/CombinedRange.h
  OMPEval/omp/Constants.h
  # OMPEval/omp/EquityCalculator.cpp
  # OMPEval/omp/EquityCalculator.h
  OMPEval/omp/Hand.h
  OMPEval/omp/HandEvaluator.cpp
  # OMPEval/omp/HandEvaluator.h
  # OMPEval/omp/Random.h
  # OMPEval/omp/Util.h
  OMPEval/libdivide/libdivide.h
)
target_link_libraries(omp
  pthread)
set_target_properties(omp PROPERTIES LINKER_LANGUAGE CXX)

add_library(${PROJECT_NAME}_cfr SHARED
  engine_modified.cpp
  engine_modified.hpp)
target_link_libraries(${PROJECT_NAME}_cfr
  omp)
set_target_properties(${PROJECT_NAME}_cfr PROPERTIES LINKER_LANGUAGE CXX)

# Build the main c++ library.
add_library(${PROJECT_NAME} SHARED
  permutation_filter.cpp
  permutation_filter.hpp
  history_tracker.cpp
  history_tracker.hpp)
target_link_libraries(${PROJECT_NAME}
  omp
  ${CMAKE_CURRENT_SOURCE_DIR}/libpbots_calc.so)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Optionally build the python wrappers.
if (BUILD_PYTHON_WRAPPER)
  message(STATUS "BUILD_PYTHON_WRAPPER is ON, building the wrappers")
  add_library(${PROJECT_NAME}_python SHARED
    bindings.cpp)
  target_link_libraries(${PROJECT_NAME}_python
    ${PROJECT_NAME}
    ${Boost_LIBRARIES})
  set_target_properties(${PROJECT_NAME}_python PROPERTIES LINKER_LANGUAGE CXX)

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
endif()

if(BUILD_TESTS)
  enable_testing()  
  set(TEST_SOURCES
    test_history_tracker.cpp
    test_permutation_filter.cpp)

  # Make a single executable for all tests.
  # Can use --gtest_filter option to run specific tests.
  add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} ./gtest/gtest-all.cc)
  target_link_libraries(${PROJECT_NAME}_test
    ${PROJECT_NAME}
    pthread)
  set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++11")
  add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)

  add_custom_command(
    TARGET ${PROJECT_NAME}_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/preflop_equity.txt
            ${CMAKE_CURRENT_BINARY_DIR}/preflop_equity.txt)
endif()
