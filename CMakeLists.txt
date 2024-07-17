cmake_minimum_required(VERSION 3.10)
project(Benchmarks)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

############################## Custom build modes ###############################

set(CMAKE_CXX_FLAGS_SANITIZE "-fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
  CACHE STRING
  "Flags used by the C++ compiler during Sanitize builds."
  FORCE)
set(CMAKE_C_FLAGS_SANITIZE "-fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
  CACHE STRING
  "Flags used by the C compiler during Sanitize builds."
  FORCE)
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
  ${CMAKE_EXE_LINKER_FLAGS_DEBUG} CACHE STRING
  "Flags used for linking binaries during Sanitize builds."
  FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} CACHE STRING
  "Flags used by the shared libraries linker during Sanitize builds."
  FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_SANITIZE		  CMAKE_EXE_LINKER_FLAGS_SANITIZE
  CMAKE_C_FLAGS_SANITIZE		  CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  )
  
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Sanitize."
  FORCE)

if(MSVC)
  # accurate profiling (closer to Release) in RelWithDebInfo: enable automatic inlining
  string(REPLACE "/Ob1" "/Ob2" CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "${CMAKE_C_FLAGS_RELWITHDEBINFO_INIT}")
  string(REPLACE "/Ob1" "/Ob2" CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT}")
  # better performances in Release
  string(REPLACE "/Ob2" "/Ob3" CMAKE_C_FLAGS_RELEASE_INIT "${CMAKE_C_FLAGS_RELEASE_INIT}")
  string(REPLACE "/Ob2" "/Ob3" CMAKE_CXX_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT}")
endif(MSVC)

############################### External Projects ###############################

if(WIN32)
  set(EXTERNAL_PROJECTS_LIB_SUFFIX ${CMAKE_LINK_LIBRARY_SUFFIX})
else()
  set(EXTERNAL_PROJECTS_LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif(WIN32)

include(ExternalProject)

ExternalProject_Add(googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
  DOWNLOAD_DIR $ENV{HOME}/.cmake-downloads/${CMAKE_PROJECT_NAME}
		CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${Benchmarks_BINARY_DIR}/deps $<IF:$<CONFIG:>,,-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}>
                   -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DBUILD_SHARED_LIBS=ON
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                   -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE} -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
  BUILD_BYPRODUCTS ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX} ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}benchmark_main${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
  
# MonetDB
ExternalProject_Add(monetdb
  GIT_REPOSITORY https://github.com/MonetDB/MonetDBLite-C
  GIT_TAG master
  GIT_SHALLOW ON
  DOWNLOAD_DIR $ENV{HOME}/.cmake-downloads/${CMAKE_PROJECT_NAME}
  CMAKE_ARGS $<IF:$<CONFIG:>,,-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}>
    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE} -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS} -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS} -DCMAKE_LINKER=${CMAKE_LINKER}
  BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target libmonetdb5
  INSTALL_COMMAND ""
)
ExternalProject_Add_Step(monetdb copyIncludes
  DEPENDEES install
  COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/src/embedded
                  ${Benchmarks_BINARY_DIR}/deps/include/embedded)
ExternalProject_Add_Step(monetdb copyLibs
  DEPENDEES install
  COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/${CMAKE_SHARED_LIBRARY_PREFIX}libmonetdb5${EXTERNAL_PROJECTS_LIB_SUFFIX}
          ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}libmonetdb5${EXTERNAL_PROJECTS_LIB_SUFFIX}
      BYPRODUCTS ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}libmonetdb5${EXTERNAL_PROJECTS_LIB_SUFFIX})
if(WIN32)
  ExternalProject_Add_Step(monetdb copyDLL
      DEPENDEES install
      COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/${CMAKE_SHARED_LIBRARY_PREFIX}libmonetdb5${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${Benchmarks_BINARY_DIR}/
      BYPRODUCTS ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}libmonetdb5${CMAKE_SHARED_LIBRARY_SUFFIX})
endif(WIN32)

# DuckDB
if(WIN32)
  set(DuckDB_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
else()
  set(DuckDB_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif(WIN32)
ExternalProject_Add(duckdblib
  GIT_REPOSITORY https://github.com/duckdb/duckdb.git
  GIT_TAG v0.5.1
  GIT_SHALLOW ON
  DOWNLOAD_DIR $ENV{HOME}/.cmake-downloads/${CMAKE_PROJECT_NAME}
  CMAKE_ARGS -DBUILD_SHELL=FALSE -DBUILD_UNITTESTS=FALSE -DBUILD_TPCH_EXTENSION=FALSE -DENABLE_SANITIZER=FALSE -DENABLE_UBSAN=FALSE
    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE} -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    -DCMAKE_INSTALL_PREFIX:PATH=${Benchmarks_BINARY_DIR}/deps $<IF:$<CONFIG:>,,-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}>
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${DuckDB_CXX_FLAGS}
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS} -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS} -DCMAKE_LINKER=${CMAKE_LINKER}
  BUILD_BYPRODUCTS ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}duckdb${EXTERNAL_PROJECTS_LIB_SUFFIX}
)
if(WIN32)
  ExternalProject_Add_Step(duckdblib copyDLL
      DEPENDEES install
      COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/src/${CMAKE_SHARED_LIBRARY_PREFIX}duckdb${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${Benchmarks_BINARY_DIR}/
      BYPRODUCTS ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}duckdb${CMAKE_SHARED_LIBRARY_SUFFIX})
endif(WIN32)

################################# Core API and engine plugins #################################

option(BUILD_WOLFRAM_ENGINE "Build the Wolfram Mathematica engine plugin." OFF)

add_subdirectory(Core)
add_subdirectory(BulkEngine)
if(BUILD_WOLFRAM_ENGINE)
  add_subdirectory(WolframEngine)
endif(BUILD_WOLFRAM_ENGINE)

if(WIN32)
  add_custom_command(DEPENDS Core
        COMMAND ${CMAKE_COMMAND} -E copy ${Core_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}Core${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${Benchmarks_BINARY_DIR}/
      OUTPUT ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}Core${CMAKE_SHARED_LIBRARY_SUFFIX})
  list(APPEND PluginList ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}Core${CMAKE_SHARED_LIBRARY_SUFFIX})
  add_custom_command(DEPENDS BulkEngine
        COMMAND ${CMAKE_COMMAND} -E copy ${BulkEngine_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}BulkEngine${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${Benchmarks_BINARY_DIR}/
      OUTPUT ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}BulkEngine${CMAKE_SHARED_LIBRARY_SUFFIX})
  list(APPEND PluginList ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}BulkEngine${CMAKE_SHARED_LIBRARY_SUFFIX})
if(BUILD_WOLFRAM_ENGINE)
  add_custom_command(DEPENDS WolframEngine
        COMMAND ${CMAKE_COMMAND} -E copy ${WolframEngine_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}WolframEngine${CMAKE_SHARED_LIBRARY_SUFFIX}
          ${Benchmarks_BINARY_DIR}/
      OUTPUT ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}WolframEngine${CMAKE_SHARED_LIBRARY_SUFFIX})
  list(APPEND PluginList ${Benchmarks_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}WolframEngine${CMAKE_SHARED_LIBRARY_SUFFIX})
endif(BUILD_WOLFRAM_ENGINE)
   add_custom_target(copyCoreDLLs ALL DEPENDS ${PluginList})
endif(WIN32)

####################### Other Options and Library Dependencies #######################

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# required to support GCC < 9 and libstdc++ < 9 (see https://stackoverflow.com/a/33159746)
# still would compile for GCC >= 9 and libstdc++ >= 9 but with unnecessary experimental fs symbols
# we can get rid of it once we do not want to support anymore OS with old libstd (e.g. ubuntu bionic)
if(NOT WIN32)
  set(CXX_FILESYSTEM_LIBRARIES "stdc++fs")
endif()

include_directories(${Core_SOURCE_DIR}/Source)
link_directories(${Core_BINARY_DIR})
include_directories(${BulkEngine_SOURCE_DIR}/Source)
link_directories(${BulkEngine_BINARY_DIR})
if(BUILD_WOLFRAM_ENGINE)
  include_directories(${WolframEngine_SOURCE_DIR}/Source)
  link_directories(${WolframEngine_BINARY_DIR})
endif(BUILD_WOLFRAM_ENGINE)

include_directories(${Benchmarks_BINARY_DIR}/deps/include)
link_directories(${Benchmarks_BINARY_DIR}/deps/lib)

################################ ITT module interface ################################

set(VTune_DIR "" CACHE PATH "Where to look for VTune installation")
find_path(ITT_NOTIFY_INCLUDE_DIR "ittnotify.h" HINTS ${VTune_DIR}/include)

add_library(ITTNotifySupport INTERFACE)
target_sources(ITTNotifySupport INTERFACE ${PROJECT_SOURCE_DIR}/Benchmarks/ITTNotifySupport.hpp)
target_include_directories(ITTNotifySupport INTERFACE ${PROJECT_SOURCE_DIR}/Benchmarks/)
if(ITT_NOTIFY_INCLUDE_DIR)
  message(VERBOSE "found itt notify header in ${ITT_NOTIFY_INCLUDE_DIR}")
  target_include_directories(ITTNotifySupport SYSTEM INTERFACE ${ITT_NOTIFY_INCLUDE_DIR})
  target_link_libraries(ITTNotifySupport INTERFACE ${ITT_NOTIFY_INCLUDE_DIR}/../lib64/libittnotify${CMAKE_STATIC_LIBRARY_SUFFIX})
  if(NOT WIN32)
    target_link_libraries(ITTNotifySupport dl)
  endif(NOT WIN32)
  if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(ITTNotifySupport INTERFACE -DWITH_ITT_NOTIFY)
  else()
    target_compile_options(ITTNotifySupport INTERFACE -DINTEL_NO_ITTNOTIFY_API)
  endif()
endif(ITT_NOTIFY_INCLUDE_DIR)

################################ Benchmark executable ################################

set(BenchmarkFiles Benchmarks/Benchmarks.cpp)
set(MicroBenchmarkFiles Benchmarks/MicroBenchmarks.cpp)

add_executable(Benchmarks ${ImplementationFiles} ${BenchmarkFiles})
target_compile_options(Benchmarks PUBLIC -DARROW_STATIC)
target_link_libraries(Benchmarks Threads::Threads)
target_link_libraries(Benchmarks ${CXX_FILESYSTEM_LIBRARIES})
if(WIN32)
  target_link_libraries(Benchmarks shlwapi.lib)  
endif(WIN32)
target_link_libraries(Benchmarks ITTNotifySupport)
add_dependencies(Benchmarks googlebenchmark)
add_dependencies(Benchmarks monetdb duckdblib)
target_link_libraries(Benchmarks libmonetdb5 duckdb)
target_link_libraries(Benchmarks benchmark)
target_link_libraries(Benchmarks benchmark_main)
target_link_libraries(Benchmarks Core)
add_dependencies(Benchmarks Core)
add_dependencies(Benchmarks BulkEngine) # dynamically loaded
if(BUILD_WOLFRAM_ENGINE)
  add_dependencies(Benchmarks WolframEngine) # dynamically loaded
endif(BUILD_WOLFRAM_ENGINE)
if(WIN32)
  add_dependencies(Benchmarks copyCoreDLLs)
endif(WIN32)

target_compile_options(Benchmarks PUBLIC -DBENCHMARK_STATIC_DEFINE)
target_link_libraries(Benchmarks Threads::Threads)
set_target_properties(Benchmarks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(MSVC)
  target_compile_options(Benchmarks PUBLIC "/Zc:__cplusplus")
  target_compile_options(Benchmarks PUBLIC "/EHsc")
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(Benchmarks rt)
  target_link_libraries(Benchmarks dl)
endif() 

add_executable(MicroBenchmarks ${MicroBenchmarkFiles})
target_link_libraries(MicroBenchmarks Threads::Threads)
if(WIN32)
  target_link_libraries(MicroBenchmarks shlwapi.lib)
endif(WIN32)
target_link_libraries(MicroBenchmarks ITTNotifySupport)
add_dependencies(MicroBenchmarks googlebenchmark)
target_link_libraries(MicroBenchmarks benchmark)
target_link_libraries(MicroBenchmarks benchmark_main)
target_link_libraries(MicroBenchmarks Threads::Threads)
set_target_properties(MicroBenchmarks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(MSVC)
  target_compile_options(MicroBenchmarks PUBLIC "/Zc:__cplusplus")
  target_compile_options(MicroBenchmarks PUBLIC "/EHsc")
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(MicroBenchmarks rt)
  target_link_libraries(MicroBenchmarks dl)
endif()

#################################### Install ###################################

set_target_properties(Benchmarks PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS Benchmarks RUNTIME DESTINATION bin)
if(WIN32)
  FILE(GLOB ALL_DEPENDENT_DLLS "${Benchmarks_BINARY_DIR}/*.dll")
  install(FILES ${ALL_DEPENDENT_DLLS} DESTINATION bin)
else()
  install(FILES ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}duckdb${EXTERNAL_PROJECTS_LIB_SUFFIX} DESTINATION lib)
  install(FILES ${Benchmarks_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}libmonetdb5${EXTERNAL_PROJECTS_LIB_SUFFIX} DESTINATION lib)  
endif(WIN32)

set_target_properties(MicroBenchmarks PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS MicroBenchmarks RUNTIME DESTINATION bin)
