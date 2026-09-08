cmake_minimum_required(VERSION 3.15...3.25)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_NEWPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

include_directories(include/coreforecast)
# SYSTEM keeps the third-party headers out of the warning gate below
include_directories(
  SYSTEM
  external_libs/skiplist/src/cpp
  external_libs/stl-cpp/include
  external_libs/eigen)
file(GLOB SOURCES src/*.cpp external_libs/skiplist/src/cpp/SkipList.cpp)
pybind11_add_module(_lib ${SOURCES})

# MSVC is left out until /W4 on the Eigen and pybind11 headers gets its own pass
target_compile_options(
  _lib PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)
# Off by default so a new compiler warning about existing code can't break a
# source install; the warnings job in CI turns it on.
option(COREFORECAST_WERROR "Treat compiler warnings as errors" OFF)
if(COREFORECAST_WERROR)
  target_compile_options(
    _lib PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>)
endif()
# vendored source, built into the same target, so its warnings aren't ours
set_source_files_properties(
  external_libs/skiplist/src/cpp/SkipList.cpp
  PROPERTIES COMPILE_OPTIONS $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-w>)

install(TARGETS _lib DESTINATION .)
