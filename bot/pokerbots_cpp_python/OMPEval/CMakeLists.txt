cmake_minimum_required(VERSION 3.0)
# project(OMPEval)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(SOURCE_FILES omp/CardRange.cpp omp/CardRange.h
  omp/CombinedRange.cpp omp/CombinedRange.h omp/Constants.h omp/EquityCalculator.cpp
  omp/EquityCalculator.h omp/Hand.h omp/HandEvaluator.cpp omp/HandEvaluator.h
  omp/Random.h omp/Util.h libdivide/libdivide.h)
add_library(ompeval SHARED
  ${SOURCE_FILES})

add_executable (ompeval_test
  ompeval.cpp benchmark.cpp)
target_link_libraries(ompeval_test
  ompeval)
set_target_properties(ompeval_test PROPERTIES LINKER_LANGUAGE CXX)
