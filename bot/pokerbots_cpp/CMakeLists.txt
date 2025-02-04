# Needed to find pyconfig.h
add_compile_options(-std=c++11)

# find_package(Boost REQUIRED COMPONENTS system program_options filesystem)
include_directories(/usr/include/python3.5/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OMPEval/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp_skeleton)
link_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
link_directories(/usr/local/lib/)

#=============== OMPEVAL LIBRARY =================
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
  OMPEval/omp/HandEvaluator.h
  # OMPEval/omp/Random.h
  # OMPEval/omp/Util.h
  OMPEval/libdivide/libdivide.h
)
target_link_libraries(omp pthread)
set_target_properties(omp PROPERTIES LINKER_LANGUAGE CXX)

#================ CPP SKELETON LIBRARY ============
add_library(cpp_skeleton
  ../cpp_skeleton/actions.cpp
  ../cpp_skeleton/bot.hpp
  ../cpp_skeleton/runner.cpp
  ../cpp_skeleton/states.cpp)
target_link_libraries(cpp_skeleton boost_filesystem boost_system boost_program_options) # ${Boost_LIBRARIES})
set_target_properties(cpp_skeleton PROPERTIES LINKER_LANGUAGE CXX)

#================ CFR LIBRARY =====================
add_library(${PROJECT_NAME}_cfr SHARED
  infoset.cpp
  infoset.hpp
  regret_matched_strategy.cpp
  regret_matched_strategy.hpp
  engine_modified.cpp
  engine_modified.hpp
  cfr.cpp)
target_link_libraries(${PROJECT_NAME}_cfr
  ${CMAKE_CURRENT_SOURCE_DIR}/libpbots_calc.so
  cpp_skeleton
  boost_filesystem
  boost_program_options
  omp)
set_target_properties(${PROJECT_NAME}_cfr PROPERTIES LINKER_LANGUAGE CXX)

#================= PLAYER LIBRARY ==================
add_library(${PROJECT_NAME}_player SHARED
  permutation_filter.cpp
  permutation_filter.hpp
  history_tracker.cpp
  history_tracker.hpp)
target_link_libraries(${PROJECT_NAME}_player
  omp
  boost_filesystem
  boost_program_options
  ${CMAKE_CURRENT_SOURCE_DIR}/libpbots_calc.so)
set_target_properties(${PROJECT_NAME}_player PROPERTIES LINKER_LANGUAGE CXX)

#================= CFR EXECUTABLE =================
if(BUILD_CFR_EXECUTABLES)
  add_executable(cfr_small
    cfr_trainer.cpp
    cfr_trainer.hpp
    cfr_main_small.cpp)
  target_link_libraries(cfr_small
    boost_filesystem
    boost_program_options
    ${PROJECT_NAME}_cfr)

  add_executable(cfr_medium
    cfr_trainer.cpp
    cfr_trainer.hpp
    cfr_main_medium.cpp)
  target_link_libraries(cfr_medium
    boost_filesystem
    boost_program_options
    ${PROJECT_NAME}_cfr)

  add_executable(cfr_large
    cfr_trainer.cpp
    cfr_trainer.hpp
    cfr_main_large.cpp)
  target_link_libraries(cfr_large
    boost_filesystem
    boost_program_options
    ${PROJECT_NAME}_cfr)

  add_executable(cfr_10_16
    cfr_trainer.cpp
    cfr_trainer.hpp
    cfr_main_10_16.cpp)
  target_link_libraries(cfr_10_16
    boost_filesystem
    boost_program_options
    ${PROJECT_NAME}_cfr)
endif()


if(BUILD_CLUSTERING)
  add_executable(hand_clustering
    hand_clustering.cpp
    hand_clustering.hpp
    hand_clustering_main.cpp)
  target_link_libraries(hand_clustering
    boost_filesystem
    boost_program_options
    ${PROJECT_NAME}_cfr)
  
  add_custom_command(
    TARGET hand_clustering POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/opponent_clusters.txt
            ${CMAKE_CURRENT_BINARY_DIR}/opponent_clusters.txt)
endif()

# Optionally build the python wrappers.
# if (BUILD_PYTHON_WRAPPER)
#   message(STATUS "BUILD_PYTHON_WRAPPER is ON, building the wrappers")
#   add_library(${PROJECT_NAME}_python SHARED
#     bindings.cpp)
#   target_link_libraries(${PROJECT_NAME}_python
#     ${PROJECT_NAME}
#     ${Boost_LIBRARIES})
#   set_target_properties(${PROJECT_NAME}_python PROPERTIES LINKER_LANGUAGE CXX)

#   # Suppress prefix "lib" because Python does not allow this prefix
#   set_target_properties(${PROJECT_NAME}_python PROPERTIES PREFIX "")

#   # Copy libraries to directory above.
#   add_custom_command(
#     TARGET ${PROJECT_NAME}_python POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy
#             ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_python.so
#             ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}_python.so)

#   add_custom_command(
#     TARGET ${PROJECT_NAME}_python POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy
#             ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}.so
#             ${CMAKE_SOURCE_DIR}/lib${PROJECT_NAME}.so)
# endif()

if(BUILD_TESTS)
  enable_testing()  
  set(TEST_SOURCES
    test_cfr.cpp
    test_history_tracker.cpp
    test_permutation_filter.cpp)

  # Make a single executable for all tests.
  # Can use --gtest_filter option to run specific tests.
  add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} ./gtest/gtest-all.cc)
  target_link_libraries(${PROJECT_NAME}_test
    ${PROJECT_NAME}_player
    ${PROJECT_NAME}_cfr
    pthread)
  set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++11")
  add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)

  add_custom_command(
    TARGET ${PROJECT_NAME}_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/preflop_equity.txt
            ${CMAKE_CURRENT_BINARY_DIR}/preflop_equity.txt)
endif()
