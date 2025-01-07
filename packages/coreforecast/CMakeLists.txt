cmake_minimum_required(VERSION 3.15...3.25)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_NEWPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

include_directories(
  include/coreforecast
  external_libs/skiplist/src/cpp
  external_libs/stl-cpp/include
  external_libs/eigen)
file(GLOB SOURCES src/*.cpp external_libs/skiplist/src/cpp/SkipList.cpp)
pybind11_add_module(_lib ${SOURCES})

install(TARGETS _lib DESTINATION .)
