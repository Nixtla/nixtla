cmake_minimum_required(VERSION 3.25)
project(coreforecast)

include(FetchContent)
option(USE_OPENMP "Enable OpenMP" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

if(USE_OPENMP)
    if(APPLE)
        find_package(OpenMP)

        if(NOT OpenMP_FOUND)
            # libomp 15.0+ from brew is keg-only, so have to search in other locations.
            # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
            execute_process(COMMAND brew --prefix libomp
                OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES omp)
            set(OpenMP_CXX_LIB_NAMES omp)
            set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        endif()
    endif()

    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(APPLE)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

if(UNIX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -O0 -g -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -O3 -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Ot /Oy /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
endif()

if(SKBUILD)
    set(LIBRARY_OUTPUT_PATH ${SKBUILD_PLATLIB_DIR}/coreforecast/lib)
else()
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/python/coreforecast/lib)
endif()

include_directories(include)
FetchContent_Declare(
    stl-cpp
    GIT_REPOSITORY https://github.com/jmoralez/stl-cpp.git
    GIT_TAG 13d26c0d0653ddcdbf853de3f92f56faa831a330
)
FetchContent_GetProperties(stl-cpp)

if(NOT stl-cpp_POPULATED)
    FetchContent_Populate(stl-cpp)
    include_directories(${stl-cpp_SOURCE_DIR}/include)
endif()

FetchContent_Declare(
    skiplist
    GIT_REPOSITORY https://github.com/paulross/skiplist.git
    GIT_TAG aace571a8564067820ff7a5e34ba68d0a9782153
)
FetchContent_GetProperties(skiplist)

if(NOT skiplist_POPULATED)
    FetchContent_Populate(skiplist)
    include_directories(${skiplist_SOURCE_DIR}/src/cpp)
endif()

file(GLOB SOURCES src/*.cpp ${skiplist_SOURCE_DIR}/src/cpp/SkipList.cpp)
add_library(coreforecast SHARED ${SOURCES})

if(MSVC)
    set_target_properties(coreforecast PROPERTIES OUTPUT_NAME "libcoreforecast")
endif()

if(USE_OPENMP AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_link_libraries(coreforecast PUBLIC OpenMP::OpenMP_CXX)
endif()
