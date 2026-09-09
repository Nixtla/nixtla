cmake_minimum_required(VERSION 3.15...3.25)
project(coreforecast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_NEWPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

include_directories(include/coreforecast)
# SYSTEM keeps the third-party headers out of the warning gate below
include_directories(SYSTEM external_libs/skiplist/src/cpp
                    external_libs/stl-cpp/include)
set(SKIPLIST_SOURCE external_libs/skiplist/src/cpp/SkipList.cpp)
# vendored source, built into our targets, so its warnings aren't ours
set_source_files_properties(
  ${SKIPLIST_SOURCE}
  PROPERTIES COMPILE_OPTIONS $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-w>)

# MSVC is left out until /W4 on the pybind11 headers gets its own pass
set(COREFORECAST_WARNINGS $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)
# Off by default so a new compiler warning about existing code can't break a
# source install; the warnings job in CI turns it on.
option(COREFORECAST_WERROR "Treat compiler warnings as errors" OFF)
if(COREFORECAST_WERROR)
  list(APPEND COREFORECAST_WARNINGS
       $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>)
endif()

file(GLOB SOURCES src/*.cpp)
pybind11_add_module(_lib ${SOURCES} ${SKIPLIST_SOURCE})
target_compile_options(_lib PRIVATE ${COREFORECAST_WARNINGS})

# Off by default so wheel builds are unchanged. ON builds a doctest runner for
# the kernels in tests/cpp, which need no Python; CI runs it under the
# sanitizers with _GLIBCXX_ASSERTIONS so an out-of-range index fails there
# before a wheel exists.
option(COREFORECAST_BUILD_TESTS "Build the C++ kernel tests" OFF)
if(COREFORECAST_BUILD_TESTS)
  enable_testing()
  file(GLOB TEST_SOURCES tests/cpp/*.cpp)
  add_executable(kernel_tests ${TEST_SOURCES} ${SKIPLIST_SOURCE})
  target_include_directories(kernel_tests SYSTEM
                             PRIVATE external_libs/doctest/doctest)
  target_compile_options(kernel_tests PRIVATE ${COREFORECAST_WARNINGS})
  add_test(NAME kernel_tests COMMAND kernel_tests)
endif()

install(TARGETS _lib DESTINATION .)
